import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from shapely.geometry import Point

from step05_quantum_classical_benchmarking import (
    QTS_READOUT_WIRES,
    build_edge_geometry_from_graph,
    edge_length_weight,
    transition_features,
)

from pipeline_config import (
    EXPORTS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    candidates_path,
    ensure_project_dirs,
    make_qml_device,
    qts_weights_path,
    unified_graph_path,
)


CASE_NAMES = [
    "Rozelle_Interchange_NSW",
    "West_Gate_Tunnel_VIC",
    "NorthConnex_NSW",
    "Light_Horse_Interchange_NSW",
    "Domain_Tunnel_VIC",
    "M80_Princes_Freeway_VIC",
]


def _parse_transition_task(value):
    task = str(value).strip().lower()
    if task != "transition":
        raise argparse.ArgumentTypeError(
            "Step09 standalone training is retired; run Step06 calibration, then "
            "invoke Step09 with '--task transition' to use its canonical QTS weights"
        )
    return task


def write_step09_consolidated_outputs(case_names):
    rows = []
    records = []
    for case_name in case_names:
        source = REPORTS_DIR / f"step09_{case_name}_transitions.json"
        if not source.exists():
            continue
        with open(source, "r", encoding="utf-8") as f:
            payload = json.load(f)

        records.append(
            {
                "case_study": case_name,
                "source_file": str(source),
                "source_filename": source.name,
                "payload": payload,
            }
        )

        if not isinstance(payload, list) or not payload:
            rows.append(
                {
                    "case_study": case_name,
                    "transition_index": None,
                    "from_ping_index": None,
                    "to_ping_index": None,
                    "from": None,
                    "to": None,
                    "quantum_score": None,
                    "transition_probability": None,
                    "route_distance_m": None,
                    "observed_displacement_m": None,
                    "transition_count_in_file": 0,
                    "source_file": str(source),
                    "source_filename": source.name,
                }
            )
            continue

        total = len(payload)
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                item = {}
            rows.append(
                {
                    "case_study": case_name,
                    "transition_index": idx,
                    "from_ping_index": item.get("from_ping_index"),
                    "to_ping_index": item.get("to_ping_index"),
                    "from": item.get("from"),
                    "to": item.get("to"),
                    "quantum_score": item.get("quantum_score"),
                    "transition_probability": item.get("transition_probability"),
                    "route_distance_m": item.get("route_distance_m"),
                    "observed_displacement_m": item.get("observed_displacement_m"),
                    "transition_count_in_file": total,
                    "source_file": str(source),
                    "source_filename": source.name,
                }
            )

    out_csv = REPORTS_DIR / "step09_transitions_all_cases.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json = REPORTS_DIR / "step09_transitions_all_cases.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved consolidated Step09 CSV: {out_csv}")
    print(f"Saved consolidated Step09 JSON: {out_json}")


def _to_bool_flexible(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    if isinstance(value, (int, np.integer)):
        return int(value) != 0

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return False
        return float(value) != 0.0

    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "off", "", "none", "nan"}:
        return False

    raise ValueError(f"Invalid boolean literal: {value!r}")


def _load_graphml_tolerant(graphml_path):
    return ox.load_graphml(
        graphml_path,
        # osmnx coerces node/edge ids via dtypes["osmid"] (default int), which
        # crashes on the string "geoscape_tunnel_x_y" / "geoscape_i_j" ids
        # injected for West Gate by inject_missing_geoscape_tunnels (step01).
        node_dtypes={"osmid": str},
        edge_dtypes={
            "osmid": str,
            "oneway": _to_bool_flexible,
            "reversed": _to_bool_flexible,
            "bridge": _to_bool_flexible,
            "tunnel": _to_bool_flexible,
        },
    )


def norm_edge(edge_like):
    if isinstance(edge_like, (list, tuple)) and len(edge_like) >= 3:
        return [str(edge_like[0]), str(edge_like[1]), int(edge_like[2])]
    return edge_like


def _coerce_graph_id(value):
    try:
        text = str(value).strip()
        if text == "":
            return value
        if text.lower() in {"nan", "none"}:
            return value
        num = float(text)
        if num.is_integer():
            return int(num)
        return value
    except Exception:
        return value


def validate_transition_records(inference_results):
    for match in inference_results:
        if "to" not in match or "quantum_score" not in match:
            raise ValueError(f"Invalid transition record: {match}")


def build_route_geodataframe(inference_results, graphml_path):
    G = _load_graphml_tolerant(graphml_path)
    _, edges = ox.graph_to_gdfs(G)

    route_segments = []
    skipped_edges = 0
    for match in inference_results:
        u, v, key = match["to"]
        lookup_keys = [
            (_coerce_graph_id(u), _coerce_graph_id(v), int(_coerce_graph_id(key)) if str(key).strip() != "" else 0),
            (str(u), str(v), int(_coerce_graph_id(key)) if str(key).strip() != "" else 0),
            (_coerce_graph_id(v), _coerce_graph_id(u), int(_coerce_graph_id(key)) if str(key).strip() != "" else 0),
            (str(v), str(u), int(_coerce_graph_id(key)) if str(key).strip() != "" else 0),
        ]

        geom = None
        for lk in lookup_keys:
            try:
                geom = edges.loc[lk, "geometry"]
                break
            except KeyError:
                continue

        if geom is None:
            skipped_edges += 1
            continue
        route_segments.append(geom)

    if not route_segments:
        if skipped_edges:
            print(f"Skipped {skipped_edges} missing edge geometries while building route.")
        return None

    if skipped_edges:
        print(f"Skipped {skipped_edges} missing edge geometries while building route.")

    return gpd.GeoDataFrame(geometry=route_segments, crs=edges.crs).to_crs("EPSG:4326")


def _safe_parse_datetime(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception as exc:
        raise ValueError(f"Invalid timestamp format: {value}") from exc


def print_vertical_profile(inference_results, candidates_data, csv_output_path=None):
    print("\n" + "=" * 40)
    print("INFERRED PATH VERTICAL PROFILE")
    print("=" * 40)
    print(f"{'Timestamp':<30} | {'Layer':<6} | {'Confidence'}")
    print("-" * 55)

    rows = []
    for i, match in enumerate(inference_results):
        target_edge = match["to"]
        ping_index = int(match.get("to_ping_index", i + 1))
        candidates = candidates_data[ping_index]["candidates"]
        timestamp = candidates_data[ping_index]["timestamp"]

        selected_layer = "N/A"
        for cand in candidates:
            if norm_edge(cand["edge_id"]) == norm_edge(target_edge):
                selected_layer = cand["layer"]
                break

        status = "TUNNEL" if selected_layer == -1 else "SURFACE"

        print(f"{timestamp:<30} | {selected_layer:<6} ({status}) | {match['quantum_score']:.4f}")
        rows.append(
            {
                "timestamp": timestamp,
                "ping_index": ping_index,
                "layer": selected_layer,
                "status": status,
                "quantum_score": float(match["quantum_score"]),
            }
        )
    print("=" * 40 + "\n")

    if csv_output_path:
        pd.DataFrame(rows).to_csv(csv_output_path, index=False)
        print(f"Saved vertical profile CSV: {csv_output_path}")


def export_quantum_route(inference_results, graphml_path, output_name="step09_quantum_route.geojson"):
    print(f"--- Exporting GeoJSON for {output_name} ---")
    gdf_route = build_route_geodataframe(inference_results, graphml_path)
    if gdf_route is not None:
        gdf_route.to_file(output_name, driver="GeoJSON")
        print(f"Successfully saved route to {output_name}")


def export_route_geometry_csv(inference_results, graphml_path, csv_output_path, case_name):
    gdf_route = build_route_geodataframe(inference_results, graphml_path)
    if gdf_route is None or gdf_route.empty:
        print(f"No route geometry available for CSV export: {csv_output_path}")
        return

    rows = []
    segment_idx = 0
    for geom in gdf_route.geometry:
        if geom is None:
            continue

        if geom.geom_type == "LineString":
            coords = list(geom.coords)
            for point_idx, (x, y) in enumerate(coords):
                rows.append(
                    {
                        "case_study": case_name,
                        "segment_index": segment_idx,
                        "point_index": point_idx,
                        "lon": float(x),
                        "lat": float(y),
                    }
                )
            segment_idx += 1
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                coords = list(part.coords)
                for point_idx, (x, y) in enumerate(coords):
                    rows.append(
                        {
                            "case_study": case_name,
                            "segment_index": segment_idx,
                            "point_index": point_idx,
                            "lon": float(x),
                            "lat": float(y),
                        }
                    )
                segment_idx += 1

    pd.DataFrame(rows).to_csv(csv_output_path, index=False)
    print(f"Saved route geometry CSV: {csv_output_path}")


def plot_quantum_route_map(inference_results, graphml_path, output_png="step09_quantum_route_map.png", show_plot=False):
    gdf_route = build_route_geodataframe(inference_results, graphml_path)
    if gdf_route is None:
        print("No route geometry found. Route map PNG not created.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    gdf_route.plot(ax=ax, linewidth=2.5, color="#00aaff")
    ax.set_title("Quantum-Inferred Route Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"Route map plot saved as {output_png}")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_quantum_confidence(inference_results, output_png="step09_quantum_confidence_plot.png", show_plot=False):
    scores = [m["quantum_score"] for m in inference_results]
    timestamps = range(len(scores))

    fig = plt.figure(figsize=(10, 4))
    plt.plot(timestamps, scores, marker="o", linestyle="-", color="#00aaff", label="QTS compatibility probability")
    plt.fill_between(timestamps, scores, alpha=0.2, color="#00aaff")

    plt.axvspan(3, 7, color="red", alpha=0.1, label="High Ambiguity (Portal Transition)")

    plt.title("Quantum Transition Scorer: Inference Confidence")
    plt.xlabel("GPS Ping Sequence")
    plt.ylabel("QTS compatibility probability")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"Confidence plot saved as {output_png}")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_3d_quantum_route(inference_results, candidates_data, output_png="step09_quantum_3d_route.png", show_plot=False, case_name=None):
    lats, lons, layers, confidences = [], [], [], []

    for i, match in enumerate(inference_results):
        target_edge = match["to"]
        ping_index = int(match.get("to_ping_index", i + 1))
        candidates = candidates_data[ping_index]["candidates"]
        orig_coords = candidates_data[ping_index]["original_coords"]

        selected_layer = 0
        for cand in candidates:
            if norm_edge(cand["edge_id"]) == norm_edge(target_edge):
                selected_layer = cand["layer"]
                break

        lats.append(orig_coords[0])
        lons.append(orig_coords[1])
        layers.append(selected_layer)
        confidences.append(match["quantum_score"])

    if not lats:
        print("No points available for 3D plot. 3D PNG not created.")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(lons, lats, layers, color="gray", alpha=0.5, linestyle="--", label="Inferred Trajectory")
    sc = ax.scatter(
        lons,
        lats,
        layers,
        c=confidences,
        cmap="RdYlGn",
        s=100,
        edgecolors="black",
        depthshade=False,
    )

    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label("QTS compatibility probability", rotation=270, labelpad=15)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Vertical Layer")
    title_suffix = f" ({case_name.replace('_', ' ')})" if case_name else ""
    ax.set_title(f"3D Quantum Ambiguity Resolution{title_suffix}")
    ax.view_init(elev=20.0, azim=-35)

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"3D route plot saved as {output_png}")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_space_time_quantum_cube(inference_results, candidates_data, output_png="step09_quantum_space_time_cube.png", show_plot=False, case_name=None):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    lats, lons, times, layers, confidences = [], [], [], [], []

    start_time = _safe_parse_datetime(candidates_data[0]["timestamp"])

    for i, match in enumerate(inference_results):
        target_edge = match["to"]
        ping_index = int(match.get("to_ping_index", i + 1))
        node_data = candidates_data[ping_index]

        current_time = _safe_parse_datetime(node_data["timestamp"])
        seconds_elapsed = (current_time - start_time).total_seconds()

        selected_layer = next(
            (c["layer"] for c in node_data["candidates"] if norm_edge(c["edge_id"]) == norm_edge(target_edge)),
            0,
        )

        lats.append(node_data["original_coords"][0])
        lons.append(node_data["original_coords"][1])
        times.append(seconds_elapsed)
        layers.append(selected_layer)
        confidences.append(match["quantum_score"])

    if not times:
        print("No points available for space-time cube. PNG not created.")
        return

    for j in range(len(times) - 1):
        color = "#8B4513" if layers[j + 1] == -1 else "#1E90FF"
        ax.plot(lons[j : j + 2], lats[j : j + 2], times[j : j + 2], color=color, linewidth=3, alpha=0.8)

    sc = ax.scatter(lons, lats, times, c=confidences, cmap="RdYlGn", s=100, edgecolors="black")
    ax.plot(lons, lats, [0] * len(times), color="black", alpha=0.2, linestyle=":")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Time (Seconds from start)")
    title_suffix = f" ({case_name.replace('_', ' ')})" if case_name else ""
    ax.set_title(f"Space-Time Transition Scorer{title_suffix}\n(Blue=Surface, Brown=Tunnel)")

    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label("QTS compatibility probability")

    portal_idx = next((i for i, l in enumerate(layers) if l == -1), None)
    if portal_idx is not None:
        ax.text(
            lons[portal_idx],
            lats[portal_idx],
            times[portal_idx],
            " Portal Entry",
            color="red",
            fontweight="bold",
        )

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Space-time cube plot saved as {output_png}")
    if show_plot:
        plt.show()
    plt.close(fig)


def write_step09_visual_consolidated_outputs(case_names):
    frames = []
    plot_rows = []
    route_frames = []
    for case_name in case_names:
        source = REPORTS_DIR / f"step09_{case_name}_vertical_profile.csv"
        if source.exists():
            df = pd.read_csv(source)
            if not df.empty:
                df = df.copy()
                df["case_study"] = case_name
                df["source_file"] = str(source)
                df["source_filename"] = source.name
                frames.append(df)

        for plot_name in [
            f"step09_{case_name}_quantum_route_map.png",
            f"step09_{case_name}_quantum_confidence_plot.png",
            f"step09_{case_name}_quantum_3d_route.png",
            f"step09_{case_name}_quantum_space_time_cube.png",
        ]:
            plot_path = PLOTS_DIR / plot_name
            plot_rows.append(
                {
                    "case_study": case_name,
                    "plot_name": plot_name,
                    "plot_path": str(plot_path),
                    "exists": bool(plot_path.exists()),
                }
            )

        route_source = REPORTS_DIR / f"step09_{case_name}_route_geometry.csv"
        if route_source.exists():
            route_df = pd.read_csv(route_source)
            if not route_df.empty:
                route_df = route_df.copy()
                route_df["source_file"] = str(route_source)
                route_df["source_filename"] = route_source.name
                route_frames.append(route_df)

    out_csv = REPORTS_DIR / "step09_vertical_profile_all_cases.csv"
    if frames:
        pd.concat(frames, axis=0, ignore_index=True).to_csv(out_csv, index=False)
    else:
        pd.DataFrame(
            columns=["timestamp", "layer", "status", "quantum_score", "case_study", "source_file", "source_filename"]
        ).to_csv(out_csv, index=False)

    plot_manifest = REPORTS_DIR / "step09_plot_manifest_all_cases.csv"
    pd.DataFrame(plot_rows).to_csv(plot_manifest, index=False)

    route_csv = REPORTS_DIR / "step09_route_geometry_all_cases.csv"
    if route_frames:
        pd.concat(route_frames, axis=0, ignore_index=True).to_csv(route_csv, index=False)
    else:
        pd.DataFrame(
            columns=["case_study", "segment_index", "point_index", "lon", "lat", "source_file", "source_filename"]
        ).to_csv(route_csv, index=False)

    print(f"Saved consolidated Step09 vertical profiles: {out_csv}")
    print(f"Saved consolidated Step09 plot manifest: {plot_manifest}")
    print(f"Saved consolidated Step09 route geometry: {route_csv}")

# 1. Initialize Quantum Device
num_qubits = 4
dev = make_qml_device(qml, wires=num_qubits, prefer_gpu=True, gpu_fraction=0.8)
_QTS_WEIGHTS_SHAPE = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=num_qubits)


def load_case_qts_weights(case_name):
    """Load the case's Step06-trained QTS weights.

    Transition outputs are scientific results, so silently substituting an
    untrained circuit would make them incomparable with Steps 05--07.
    """
    path = qts_weights_path(case_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing trained QTS weights {path}; run Step06 calibration first.")
    weights = np.asarray(np.load(path), dtype=float)
    if weights.shape != _QTS_WEIGHTS_SHAPE:
        raise ValueError(
            f"Invalid QTS weight shape in {path}: expected {_QTS_WEIGHTS_SHAPE}, got {weights.shape}."
        )
    return weights


_QTS_READOUT = qml.PauliZ(QTS_READOUT_WIRES[0])
for _wire in QTS_READOUT_WIRES[1:]:
    _QTS_READOUT = _QTS_READOUT + qml.PauliZ(_wire)
_QTS_READOUT = _QTS_READOUT / float(len(QTS_READOUT_WIRES))

@qml.qnode(dev)
def transition_circuit(features, weights):
    """Exact Step05/Step06 two-upload QTS circuit."""
    qml.AngleEmbedding(pnp.pi * features, wires=range(num_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights[0:1], wires=range(num_qubits))
    qml.AngleEmbedding(pnp.pi * features, wires=range(num_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights[1:2], wires=range(num_qubits))
    return qml.expval(_QTS_READOUT)


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _candidate_edge_attrs(candidate):
    """Adapt a Step08 record to canonical ``transition_features`` fields."""
    return {
        "layer": _to_float(candidate.get("layer", 0.0), 0.0),
        "maxspeed": _to_float(candidate.get("speed_limit", candidate.get("maxspeed", 60.0)), 60.0),
        "bearing": _to_float(candidate.get("bearing", 0.0), 0.0),
    }


def calculate_transition_score(cand_prev, cand_curr, weights, transition_dist=None):
    """Score one pair with the exact Step05/Step06 feature semantics.

    The global decoder supplies directed route distance.  The optional
    projected-point separation is retained only for standalone visual probes
    that do not have a graph object.
    """
    prev_x = _to_float(cand_prev.get("edge_x"), float("nan"))
    prev_y = _to_float(cand_prev.get("edge_y"), float("nan"))
    curr_x = _to_float(cand_curr.get("edge_x"), float("nan"))
    curr_y = _to_float(cand_curr.get("edge_y"), float("nan"))
    if transition_dist is None:
        transition_dist = (
            float(np.hypot(curr_x - prev_x, curr_y - prev_y))
            if np.all(np.isfinite([prev_x, prev_y, curr_x, curr_y]))
            else 150.0
        )
    features = np.asarray(
        transition_features(
            _candidate_edge_attrs(cand_prev),
            _candidate_edge_attrs(cand_curr),
            float(transition_dist),
        ),
        dtype=float,
    )
    z = float(transition_circuit(features, weights))
    return float(np.clip((z + 1.0) / 2.0, 1e-9, 1.0))


def _normalize_edge_id(edge_id):
    if isinstance(edge_id, (list, tuple)) and len(edge_id) >= 3:
        return [str(edge_id[0]), str(edge_id[1]), int(edge_id[2])]
    if isinstance(edge_id, str):
        if edge_id.startswith("[") and edge_id.endswith("]"):
            try:
                val = json.loads(edge_id)
                return _normalize_edge_id(val)
            except Exception:
                pass
        parts = edge_id.split(",") if "," in edge_id else edge_id.split("_")
        if len(parts) >= 3:
            return [str(parts[0]), str(parts[1]), int(float(parts[2]))]
    return [str(edge_id), "", 0]

def _resolve_candidate_edge(graph, candidate):
    normalized = _normalize_edge_id(candidate.get("edge_id"))
    raw_u, raw_v, raw_key = normalized
    u_values = [raw_u, _coerce_graph_id(raw_u)]
    v_values = [raw_v, _coerce_graph_id(raw_v)]
    for u in dict.fromkeys(u_values):
        for v in dict.fromkeys(v_values):
            data = graph.get_edge_data(u, v)
            if data is None:
                continue
            if graph.is_multigraph():
                key_values = [raw_key, str(raw_key), _coerce_graph_id(raw_key)]
                for key in dict.fromkeys(key_values):
                    if key in data:
                        return (u, v, key)
                continue
            return (u, v, 0)
    return None


def _edge_attrs(graph, edge):
    data = graph.get_edge_data(edge[0], edge[1])
    if data is None:
        return {}
    return data.get(edge[2], {}) if graph.is_multigraph() else data


def directed_candidate_route_distance(
    graph,
    previous_candidate,
    current_candidate,
    geometry_cache=None,
    hop_cache=None,
):
    """Directed along-network distance using Step05/06 semantics.

    Candidate offsets are measured on the complete edge geometry.  A
    same-edge move uses signed forward progress (clamped at zero); it must
    not use ``abs`` because that makes reverse travel indistinguishable from
    forward travel.  ``None`` denotes a graph-unreachable cross-edge pair.
    """
    previous_edge = _resolve_candidate_edge(graph, previous_candidate)
    current_edge = _resolve_candidate_edge(graph, current_candidate)
    if previous_edge is None or current_edge is None:
        return None

    geometry_cache = {} if geometry_cache is None else geometry_cache
    hop_cache = {} if hop_cache is None else hop_cache

    def geometry(edge):
        if edge not in geometry_cache:
            geometry_cache[edge] = build_edge_geometry_from_graph(
                graph, edge, _edge_attrs(graph, edge)
            )
        return geometry_cache[edge]

    def offset(candidate, edge):
        edge_geometry = geometry(edge)
        x = _to_float(candidate.get("edge_x"), np.nan)
        y = _to_float(candidate.get("edge_y"), np.nan)
        if np.all(np.isfinite([x, y])):
            return float(edge_geometry.project(Point(float(x), float(y))))
        return float(np.clip(
            _to_float(candidate.get("edge_offset_m"), 0.0),
            0.0,
            edge_geometry.length,
        ))

    previous_offset = offset(previous_candidate, previous_edge)
    current_offset = offset(current_candidate, current_edge)
    if previous_edge == current_edge:
        return max(current_offset - previous_offset, 0.0)

    pair = (previous_edge, current_edge)
    if pair not in hop_cache:
        try:
            hop_cache[pair] = float(nx.shortest_path_length(
                graph,
                previous_edge[1],
                current_edge[0],
                weight=edge_length_weight,
            ))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            hop_cache[pair] = None
    hop = hop_cache[pair]
    if hop is None:
        return None
    return (
        max(float(geometry(previous_edge).length) - previous_offset, 0.0)
        + hop
        + current_offset
    )


def _candidate_observation_xy(ping):
    values = ping.get("projected_coords", [None, None])
    if isinstance(values, (list, tuple)) and len(values) >= 2:
        xy = (_to_float(values[0], np.nan), _to_float(values[1], np.nan))
        if np.all(np.isfinite(xy)):
            return xy

    # Backward-compatible proxy for candidate files generated before Step08
    # stored metric observation coordinates.  The closest candidate's
    # projected point is preferable to treating latitude/longitude degrees as
    # metres.
    candidates = ping.get("candidates", [])
    if candidates:
        best = min(candidates, key=lambda item: _to_float(item.get("dist_m"), np.inf))
        xy = (_to_float(best.get("edge_x"), np.nan), _to_float(best.get("edge_y"), np.nan))
        if np.all(np.isfinite(xy)):
            return xy
    return None


def _decode_candidate_sequence(data, weights, graph, sigma_d=50.0, sigma_xy=45.0):
    """Globally decode each contiguous candidate segment in log space.

    Unlike the old pairwise greedy loop, the selected destination at time
    ``t`` is necessarily the selected source at ``t+1``.  Directed graph
    reachability is a hard constraint; empty candidate timestamps and total
    connectivity loss explicitly start a new independent segment.
    """
    geometry_cache = {}
    hop_cache = {}
    qts_cache = {}

    def route_distance(prev_candidate, prev_edge, curr_candidate, curr_edge):
        del prev_edge, curr_edge  # Resolved internally by the shared helper.
        return directed_candidate_route_distance(
            graph,
            prev_candidate,
            curr_candidate,
            geometry_cache=geometry_cache,
            hop_cache=hop_cache,
        )

    def qts_score(prev_edge, curr_edge, distance):
        key = (prev_edge, curr_edge, round(float(distance)))
        if key not in qts_cache:
            features = np.asarray(
                transition_features(
                    _edge_attrs(graph, prev_edge),
                    _edge_attrs(graph, curr_edge),
                    distance,
                ),
                dtype=float,
            )
            z_value = float(transition_circuit(features, weights))
            qts_cache[key] = float(np.clip((z_value + 1.0) / 2.0, 1e-9, 1.0))
        return qts_cache[key]

    def emission_log(candidate):
        distance = max(_to_float(candidate.get("dist_m"), 0.0), 0.0)
        return -0.5 * (distance / max(float(sigma_xy), 1e-6)) ** 2

    results = []
    segment_id = 0
    segment_times = []
    state_scores = {}
    backpointers = {}
    selected_transition = {}

    def finish_segment():
        nonlocal segment_id
        if len(segment_times) < 2 or not state_scores:
            return
        selected = {segment_times[-1]: max(state_scores, key=state_scores.get)}
        for time_index in reversed(segment_times[1:]):
            selected[time_index - 1] = backpointers[time_index][selected[time_index]]

        for prev_time, curr_time in zip(segment_times[:-1], segment_times[1:]):
            curr_index = selected[curr_time]
            meta = selected_transition[curr_time][curr_index]
            results.append(
                {
                    "segment_id": int(segment_id),
                    "from_ping_index": int(prev_time),
                    "to_ping_index": int(curr_time),
                    "timestamp": str(data[curr_time].get("timestamp", "")),
                    "from": _normalize_edge_id(data[prev_time]["candidates"][selected[prev_time]].get("edge_id")),
                    "to": _normalize_edge_id(data[curr_time]["candidates"][curr_index].get("edge_id")),
                    "quantum_score": float(meta["qts_score"]),
                    "transition_probability": float(meta["transition_probability"]),
                    "route_distance_m": float(meta["route_distance_m"]),
                    "observed_displacement_m": float(meta["observed_displacement_m"]),
                }
            )
        segment_id += 1

    for time_index, ping in enumerate(data):
        candidates = ping.get("candidates", [])
        resolved = [_resolve_candidate_edge(graph, candidate) for candidate in candidates]
        usable_indices = [idx for idx, edge in enumerate(resolved) if edge is not None]

        if not usable_indices:
            finish_segment()
            segment_times = []
            state_scores = {}
            backpointers = {}
            selected_transition = {}
            continue

        if not segment_times:
            segment_times = [time_index]
            state_scores = {idx: emission_log(candidates[idx]) for idx in usable_indices}
            continue

        # Candidate gaps are explicit sequence breaks, even if both endpoints
        # happen to be graph-connected.
        if time_index != segment_times[-1] + 1:
            finish_segment()
            segment_times = [time_index]
            state_scores = {idx: emission_log(candidates[idx]) for idx in usable_indices}
            backpointers = {}
            selected_transition = {}
            continue

        prev_time = segment_times[-1]
        prev_ping = data[prev_time]
        prev_candidates = prev_ping.get("candidates", [])
        prev_edges = [_resolve_candidate_edge(graph, candidate) for candidate in prev_candidates]
        prev_xy = _candidate_observation_xy(prev_ping)
        curr_xy = _candidate_observation_xy(ping)
        obs_disp = (
            float(np.hypot(curr_xy[0] - prev_xy[0], curr_xy[1] - prev_xy[1]))
            if prev_xy is not None and curr_xy is not None
            else 0.0
        )

        new_scores = {}
        new_backpointers = {}
        new_transition_meta = {}
        for curr_index in usable_indices:
            curr_edge = resolved[curr_index]
            best_score = -np.inf
            best_prev = None
            best_meta = None
            for prev_index, previous_score in state_scores.items():
                prev_edge = prev_edges[prev_index]
                if prev_edge is None:
                    continue
                distance = route_distance(
                    prev_candidates[prev_index], prev_edge, candidates[curr_index], curr_edge
                )
                if distance is None:
                    continue
                q_score = qts_score(prev_edge, curr_edge, distance)
                distance_factor = float(
                    np.exp(-abs(float(distance) - obs_disp) / max(float(sigma_d), 1e-6))
                )
                probability = float(np.clip(distance_factor * q_score, 1e-12, 1.0))
                score = previous_score + np.log(probability) + emission_log(candidates[curr_index])
                if score > best_score:
                    best_score = score
                    best_prev = prev_index
                    best_meta = {
                        "qts_score": q_score,
                        "transition_probability": probability,
                        "route_distance_m": distance,
                        "observed_displacement_m": obs_disp,
                    }
            if best_prev is not None:
                new_scores[curr_index] = best_score
                new_backpointers[curr_index] = best_prev
                new_transition_meta[curr_index] = best_meta

        if not new_scores:
            finish_segment()
            segment_times = [time_index]
            state_scores = {idx: emission_log(candidates[idx]) for idx in usable_indices}
            backpointers = {}
            selected_transition = {}
            continue

        segment_times.append(time_index)
        state_scores = new_scores
        backpointers[time_index] = new_backpointers
        selected_transition[time_index] = new_transition_meta

    finish_segment()
    return sorted(results, key=lambda item: item["to_ping_index"])


def run_step_4_inference(candidates_json, weights, graphml_path=None, sigma_d=50.0, sigma_xy=45.0):
    with open(candidates_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if graphml_path is None:
        filename = Path(candidates_json).name
        case_name = filename.removeprefix("step08_").removesuffix("_candidates.json")
        graphml_path = unified_graph_path(case_name)
    if not Path(graphml_path).exists():
        raise FileNotFoundError(f"GraphML required for directed global decoding: {graphml_path}")
    graph = _load_graphml_tolerant(graphml_path)

    candidate_counts = [len(ping.get("candidates", [])) for ping in data]
    print(
        f"--- Running global graph-constrained QTS inference on {len(data)} pings "
        f"({sum(count == 0 for count in candidate_counts)} empty) ---"
    )
    results = _decode_candidate_sequence(
        data,
        np.asarray(weights, dtype=float),
        graph,
        sigma_d=sigma_d,
        sigma_xy=sigma_xy,
    )
    print(f"Quantum scorer complete: decoded {len(results)} coherent transitions.")
    return results


def run_transition_workflow(skip_visualization):
    for case_name in CASE_NAMES:
        json_path = candidates_path(case_name)
        if not json_path.exists():
            print(f"Skipping {case_name}: missing {json_path}")
            continue

        print(f"\n=== Transition scoring for {case_name} ===")
        case_weights = load_case_qts_weights(case_name)
        graphml_path = unified_graph_path(case_name)
        best_route = run_step_4_inference(
            str(json_path),
            case_weights,
            graphml_path=str(graphml_path),
        )
        out_path = REPORTS_DIR / f"step09_{case_name}_transitions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(best_route, f, indent=2)
        print(f"Inferred {len(best_route)} transitions via PQC.")
        print(f"Saved transitions: {out_path}")

        if skip_visualization:
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            candidates_data = json.load(f)

        if not best_route:
            print(f"No inferred route for {case_name}.")
            continue

        validate_transition_records(best_route)
        print_vertical_profile(best_route, candidates_data, csv_output_path=REPORTS_DIR / f"step09_{case_name}_vertical_profile.csv")

        if graphml_path.exists():
            export_quantum_route(best_route, str(graphml_path), output_name=str(EXPORTS_DIR / f"step09_{case_name}_quantum_route.geojson"))
            export_route_geometry_csv(
                best_route,
                str(graphml_path),
                csv_output_path=REPORTS_DIR / f"step09_{case_name}_route_geometry.csv",
                case_name=case_name,
            )
            plot_quantum_route_map(
                best_route,
                str(graphml_path),
                output_png=str(PLOTS_DIR / f"step09_{case_name}_quantum_route_map.png"),
                show_plot=False,
            )

        plot_quantum_confidence(best_route, output_png=str(PLOTS_DIR / f"step09_{case_name}_quantum_confidence_plot.png"), show_plot=False)
        plot_3d_quantum_route(best_route, candidates_data, output_png=str(PLOTS_DIR / f"step09_{case_name}_quantum_3d_route.png"), show_plot=False, case_name=case_name)
        plot_space_time_quantum_cube(
            best_route,
            candidates_data,
            output_png=str(PLOTS_DIR / f"step09_{case_name}_quantum_space_time_cube.png"),
            show_plot=False,
            case_name=case_name,
        )

    write_step09_consolidated_outputs(CASE_NAMES)
    if not skip_visualization:
        write_step09_visual_consolidated_outputs(CASE_NAMES)


def run_parameter_shift_training_workflow(seed, max_workers):
    """Reject the obsolete standalone trainer.

    QTS weights must come from Step06's truck-disjoint calibration.  Keeping
    this explicit failure protects programmatic callers that used the former
    function even though the CLI no longer advertises its tasks.
    """
    del seed, max_workers
    raise RuntimeError(
        "Step09 standalone parameter-shift training is retired because it used "
        "a scientifically divergent data/label path. Run Step06 calibration, "
        "then use Step09 with '--task transition' to load canonical Step06 QTS weights."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step09: transition scoring and visual analytics using Step06-trained QTS weights."
    )
    parser.add_argument(
        "--task",
        type=_parse_transition_task,
        metavar="transition",
        default="transition",
        help="Only the canonical Step06-trained transition workflow is supported.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Compatibility option; transition inference is deterministic and ignores this value.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Compatibility option; the transition workflow currently runs cases sequentially.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Run transition scoring only (skip route exports and plotting).",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    run_transition_workflow(skip_visualization=bool(args.skip_visualization))
