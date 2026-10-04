import osmnx as ox
import networkx as nx
import geopandas as gpd
from pathlib import Path
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree
import json
import re
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pipeline_config import (
    DEFAULT_SEED,
    ensure_project_dirs,
    REPORTS_DIR,
    STEP08_CANDIDATE_SEARCH_RADIUS_M,
    metric_crs_for_state,
    metric_crs_for_case,
    unified_graph_path,
    candidates_path,
    TRAJECTORIES_DIR,
    default_max_workers,
    geoscape_standard_dir,
    split_calibration_and_holdout_trucks,
)

def _json_default(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, pd.Timestamp): return value.isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def to_flag(value):
    return _to_bool_flexible(value)


def _strtree_indices(tree, geometries, query_geometry):
    """Return positional indices for both Shapely 1.x and 2.x STRtree APIs."""
    hits = tree.query(query_geometry)
    if len(hits) == 0:
        return []
    first = hits[0]
    if isinstance(first, (int, np.integer)):
        return [int(value) for value in hits]
    position_by_identity = {id(geometry): idx for idx, geometry in enumerate(geometries)}
    return [position_by_identity[id(geometry)] for geometry in hits if id(geometry) in position_by_identity]


_PING_PROVENANCE_FIELDS = (
    "device_id",
    "source_case_name",
    "source_truck_id",
    "source_row_index",
    "source_split",
    "source_split_seed",
    "source_coordinate_fields",
    "coordinate_source",
)


def _ping_provenance(row):
    provenance = {}
    for field in _PING_PROVENANCE_FIELDS:
        value = row.get(field)
        provenance[field] = None if pd.isna(value) else value
    return provenance


def _causally_fill_observed_track(track):
    """Fill missing observed XY values using only earlier effective fixes.

    This mirrors the deployed constant-velocity missing-fix policy: the last
    two observed-or-imputed states determine velocity, and a single prior
    state gives a zero-velocity hold.  Leading fixes that cannot be filled
    without looking ahead are retained as unavailable and excluded later.
    """
    effective_rows = []
    history = []
    for _, source in track.iterrows():
        row = source.copy()
        x = pd.to_numeric(pd.Series([row.get("obs_x")]), errors="coerce").iloc[0]
        y = pd.to_numeric(pd.Series([row.get("obs_y")]), errors="coerce").iloc[0]
        imputed = not (np.isfinite(x) and np.isfinite(y))

        if imputed and history:
            previous = history[-1]
            previous_x = float(previous["_effective_x"])
            previous_y = float(previous["_effective_y"])
            velocity_x = 0.0
            velocity_y = 0.0
            if len(history) >= 2:
                previous_previous = history[-2]
                dt_previous = float(
                    np.clip(
                        float(previous["_effective_time"])
                        - float(previous_previous["_effective_time"]),
                        0.25,
                        30.0,
                    )
                )
                velocity_x = (
                    previous_x - float(previous_previous["_effective_x"])
                ) / dt_previous
                velocity_y = (
                    previous_y - float(previous_previous["_effective_y"])
                ) / dt_previous

            current_time = float(row["_effective_time"])
            dt_current = float(
                np.clip(current_time - float(previous["_effective_time"]), 0.25, 30.0)
            )
            if not np.isfinite(x):
                x = previous_x + velocity_x * dt_current
            if not np.isfinite(y):
                y = previous_y + velocity_y * dt_current

        row["_effective_x"] = float(x) if np.isfinite(x) else np.nan
        row["_effective_y"] = float(y) if np.isfinite(y) else np.nan
        row["_position_imputed"] = bool(imputed and np.isfinite(x) and np.isfinite(y))
        effective_rows.append(row)
        if np.isfinite(x) and np.isfinite(y):
            history.append(row)

    return pd.DataFrame(effective_rows)


def _select_heldout_observed_track(trajectory, sample_count=14):
    """Select one canonical held-out truck and return ordered usable fixes."""
    required = {"truck_id", "obs_x", "obs_y"}
    missing = sorted(required - set(trajectory.columns))
    if missing:
        raise ValueError(f"Step03 trajectory is missing required columns: {missing}")

    trajectory = trajectory.copy()
    trajectory["_source_row_index"] = np.arange(len(trajectory), dtype=int)
    trajectory["truck_id"] = trajectory["truck_id"].astype(str)
    _calibration_trucks, heldout_trucks = split_calibration_and_holdout_trucks(
        trajectory["truck_id"].unique(), seed=DEFAULT_SEED
    )

    for truck_id in heldout_trucks:
        track = trajectory.loc[trajectory["truck_id"] == str(truck_id)].copy()
        if track.empty:
            continue
        if "timestamp" in track.columns:
            numeric_time = pd.to_numeric(track["timestamp"], errors="coerce")
        else:
            numeric_time = pd.Series(np.nan, index=track.index, dtype=float)
        track["_effective_time"] = numeric_time.where(
            np.isfinite(numeric_time), track["_source_row_index"].astype(float)
        )
        track = track.sort_values(
            ["_effective_time", "_source_row_index"], kind="mergesort"
        ).reset_index(drop=True)
        effective = _causally_fill_observed_track(track)
        usable = effective.loc[
            np.isfinite(effective["_effective_x"])
            & np.isfinite(effective["_effective_y"])
        ].reset_index(drop=True)
        if len(usable) >= int(sample_count):
            return str(truck_id), usable

    raise ValueError(
        "No canonical held-out truck has enough causally available observed fixes "
        f"for a {int(sample_count)}-ping Step08 sample."
    )


def generate_case_aligned_truck_data(case_name, output_path):
    """Export an ordered observed-coordinate sample from one held-out truck."""
    graph_path = unified_graph_path(case_name)
    traj_path = TRAJECTORIES_DIR / f"step03_{case_name}_trajectories_v5.csv"
    if not graph_path.exists() or not traj_path.exists():
        raise FileNotFoundError(
            f"Cannot generate aligned Step08 data for {case_name}: "
            f"missing {graph_path} or {traj_path}."
        )

    trajectory = pd.read_csv(traj_path)
    truck_id, track = _select_heldout_observed_track(trajectory, sample_count=14)
    sample_indices = np.linspace(0, len(track) - 1, 14, dtype=int)
    samples = track.iloc[sample_indices].reset_index(drop=True)

    aligned = gpd.GeoDataFrame(
        samples.copy(),
        geometry=gpd.points_from_xy(samples["_effective_x"], samples["_effective_y"]),
        crs=metric_crs_for_case(case_name),
    ).to_crs("EPSG:4326")

    rows = []
    for idx, row in aligned.iterrows():
        timestamp = row.get("actual_time", row.get("timestamp", idx))
        rows.append(
            {
                "device_id": truck_id,
                "timestamp": str(timestamp),
                "lat": float(row.geometry.y),
                "lon": float(row.geometry.x),
                "hdop": float(row.get("hdop", 1.5)),
                "source_case_name": case_name,
                "source_truck_id": truck_id,
                "source_row_index": int(row["_source_row_index"]),
                "source_split": "canonical_holdout",
                "source_split_seed": int(DEFAULT_SEED),
                "source_coordinate_fields": "obs_x,obs_y",
                "coordinate_source": (
                    "causal_imputation" if bool(row["_position_imputed"]) else "observed_gnss"
                ),
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)
    imputed_count = sum(row["coordinate_source"] == "causal_imputation" for row in rows)
    print(
        f"Generated held-out observed truck data: {output_path} "
        f"(truck={truck_id}, pings={len(rows)}, causally_imputed={imputed_count})"
    )


def _to_bool_flexible(value):
    """Parse booleans robustly for GraphML attributes like oneway/reversed/tunnel/bridge."""
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
    """Load GraphML while tolerating bool-like strings such as '0.0'/'1.0'."""
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

hard_case_locations = {
    "NorthConnex_NSW": {"coords": (-33.7589, 151.0464), "state": "nsw"},
    "Rozelle_Interchange_NSW": {"coords": (-33.8702, 151.1722), "state": "nsw"},
    "West_Gate_Tunnel_VIC": {"coords": (-37.8246, 144.8621), "state": "vic"},
    "Light_Horse_Interchange_NSW": {"coords": (-33.7980, 150.8540), "state": "nsw"},
    "Domain_Tunnel_VIC": {"coords": (-37.8362, 144.9700), "state": "vic"},
    "M80_Princes_Freeway_VIC": {"coords": (-37.8276, 144.8169), "state": "vic"}
}

GEOSCAPE_BASE_PATH = geoscape_standard_dir()

def get_geoscape_path(state_abbr):
    return str(GEOSCAPE_BASE_PATH / f"{state_abbr.lower()}_roads.shp")

def pull_unified_maps(locations_dict, radius_meters=2500):
    ensure_project_dirs()
    for name, info in locations_dict.items():
        lat, lon = info["coords"]
        state = info["state"]
        metric_crs = metric_crs_for_state(state)
        print(f"--- Processing Unified Map: {name} ---")
        try:
            G_osm = ox.graph_from_point((lat, lon), dist=radius_meters, network_type='drive', simplify=True)
            G_osm = ox.project_graph(G_osm, to_crs=metric_crs)
            out_path = unified_graph_path(name)
            ox.save_graphml(G_osm, filepath=str(out_path))
            print(f"Successfully saved {out_path}\n")
        except Exception as e:
            print(f"Failed {name}: {e}\n")

def generate_candidates(gps_csv_path, graphml_path, k=32, search_radius=STEP08_CANDIDATE_SEARCH_RADIUS_M):
    print(f"--- Generating Candidates for {Path(gps_csv_path).name} ---")
    G = _load_graphml_tolerant(graphml_path)
    df_gps = pd.read_csv(gps_csv_path)
    required_cols = {"lat", "lon", "timestamp"}
    missing_cols = sorted(required_cols - set(df_gps.columns))
    if missing_cols:
        raise ValueError(f"GPS CSV missing required columns: {missing_cols}")

    # Coerce malformed coordinates to NaN so they are safely excluded from spatial ops.
    df_gps["lat"] = pd.to_numeric(df_gps["lat"], errors="coerce")
    df_gps["lon"] = pd.to_numeric(df_gps["lon"], errors="coerce")

    df_gps = df_gps.reset_index(drop=True)
    df_gps["_row_id"] = np.arange(len(df_gps), dtype=int)

    finite_mask = np.isfinite(df_gps["lat"]) & np.isfinite(df_gps["lon"])
    bounds_mask = (
        df_gps["lat"].between(-90.0, 90.0, inclusive="both")
        & df_gps["lon"].between(-180.0, 180.0, inclusive="both")
    )
    valid_mask = finite_mask & bounds_mask

    invalid_gps = df_gps.loc[~valid_mask].copy()
    valid_gps = df_gps.loc[valid_mask].copy()

    if not invalid_gps.empty:
        print(
            f"Filtered {len(invalid_gps)} invalid GPS rows "
            f"(NaN/inf/out-of-range) from {Path(gps_csv_path).name}."
        )

    nodes, edges = ox.graph_to_gdfs(G)

    # Remove unusable road geometries before building the full-line spatial
    # index.  Candidate membership and distance are measured against the WKT
    # centreline, not its centroid (which can be far from a long/curved edge).
    valid_edge_mask = edges.geometry.notna() & (~edges.geometry.is_empty) & edges.geometry.is_valid
    edges_valid = edges.loc[valid_edge_mask].copy()

    if edges_valid.empty:
        raise ValueError(f"No valid edge geometries found in {graphml_path}")

    edge_geometries = list(edges_valid.geometry)
    tree = STRtree(edge_geometries)

    gdf_gps = gpd.GeoDataFrame(
        valid_gps, geometry=gpd.points_from_xy(valid_gps.lon, valid_gps.lat), crs="EPSG:4326"
    ).to_crs(edges.crs)

    indexed_results = []
    nearest_edge_dists_m = []
    for idx, row in gdf_gps.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            nearest_edge_dists_m.append(np.nan)
            indexed_results.append({
                "_row_id": int(row["_row_id"]),
                "timestamp": str(row.timestamp),
                "original_coords": [row.lat, row.lon],
                "projected_coords": [None, None],
                "candidates": [],
                **_ping_provenance(row),
            })
            continue

        point = Point(float(row.geometry.x), float(row.geometry.y))
        nearby_indices = _strtree_indices(tree, edge_geometries, point.buffer(float(search_radius)))
        ranked = sorted(
            (
                (float(edge_geometries[i].distance(point)), int(i))
                for i in nearby_indices
                if edge_geometries[i].distance(point) <= float(search_radius)
            ),
            key=lambda item: (item[0], item[1]),
        )[: min(int(k), len(edges_valid))]

        # Diagnostics should report distance to the nearest *line*, even when
        # no edge falls inside the configured search radius.
        nearest_dist = min(float(geometry.distance(point)) for geometry in edge_geometries)
        nearest_edge_dists_m.append(nearest_dist)

        candidates = []
        for dist, i in ranked:
            edge_data = edges_valid.iloc[i]
            edge_geometry = edge_geometries[i]
            
            # TUNNEL LOGIC: Force layer -1 if tunnel is detected to enable quantum scoring
            tunnel_tag = edge_data.get('tunnel', 'no')
            is_tunnel = "yes" if to_flag(tunnel_tag) else "no"
            
            try:
                layer = int(edge_data.get('layer', 0))
            except (TypeError, ValueError):
                layer = 0
                
            if is_tunnel == "yes" and layer >= 0:
                layer = -1 # Force vertical ambiguity

            # SPEED LOGIC: Ensure clean integer
            raw_speed = edge_data.get('maxspeed', 60)
            try:
                m = re.search(r"\d+", str(raw_speed))
                clean_speed = int(m.group(0)) if m else 60
            except (TypeError, ValueError):
                clean_speed = 60

            raw_bearing = edge_data.get("bearing", 0.0)
            try:
                clean_bearing = float(raw_bearing) % 360.0
            except (TypeError, ValueError):
                clean_bearing = 0.0

            offset_m = float(edge_geometry.project(point))
            projected = edge_geometry.interpolate(offset_m)

            candidates.append({
                "edge_id": list(edges_valid.index[i]),
                "dist_m": float(dist),
                "speed_limit": clean_speed,
                "bearing": clean_bearing,
                "is_tunnel": is_tunnel,
                "layer": layer,
                # Field names retained for downstream compatibility;
                # they hold the closest point on the full edge geometry.
                "edge_x": float(projected.x),
                "edge_y": float(projected.y),
                "edge_offset_m": offset_m,
                "edge_length_m": float(edge_geometry.length),
            })

        indexed_results.append({
            "_row_id": int(row["_row_id"]),
            "timestamp": str(row.timestamp),
            "original_coords": [row.lat, row.lon],
            "projected_coords": [float(row.geometry.x), float(row.geometry.y)],
            "candidates": candidates,
            **_ping_provenance(row),
        })

    for _, row in invalid_gps.iterrows():
        indexed_results.append(
            {
                "_row_id": int(row["_row_id"]),
                "timestamp": str(row.get("timestamp", "")),
                "original_coords": [row.get("lat"), row.get("lon")],
                "projected_coords": [None, None],
                "candidates": [],
                **_ping_provenance(row),
            }
        )

    indexed_results.sort(key=lambda x: x["_row_id"])
    results = [
        {
            "timestamp": item["timestamp"],
            "original_coords": item["original_coords"],
            "projected_coords": item.get("projected_coords", [None, None]),
            "candidates": item["candidates"],
            **{field: item.get(field) for field in _PING_PROVENANCE_FIELDS},
        }
        for item in indexed_results
    ]

    graph_name = Path(graphml_path).name.replace("_unified.graphml", "").removeprefix("step01_")

    candidate_counts = [len(item["candidates"]) for item in results]
    empty_pings = sum(1 for c in candidate_counts if c == 0)
    valid_pairs = sum(
        1
        for i in range(1, len(candidate_counts))
        if candidate_counts[i - 1] > 0 and candidate_counts[i] > 0
    )

    nearest_arr = np.array(nearest_edge_dists_m, dtype=float)
    finite_nearest = nearest_arr[np.isfinite(nearest_arr)]

    print(
        f"[{graph_name}] Candidate diagnostics: "
        f"pings={len(results)}, empty_pings={empty_pings}, valid_pairs={valid_pairs}, "
        f"search_radius_m={search_radius}"
    )
    if finite_nearest.size > 0:
        print(
            f"[{graph_name}] Nearest-edge distance stats (m): "
            f"min={finite_nearest.min():.2f}, median={np.median(finite_nearest):.2f}, max={finite_nearest.max():.2f}"
        )

    output_path = candidates_path(graph_name)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4, default=_json_default)
    print(f"Generated {output_path} with vertical layers fixed.")
    return results


def _generate_candidates_for_case(case_name, gps_csv_path):
    path = unified_graph_path(case_name)
    if not path.exists():
        return f"Skipping {case_name}: missing graph file {path}"
    generate_candidates(str(gps_csv_path), str(path))
    return f"Completed candidates for {case_name}"


def write_step08_consolidated_outputs(case_names):
    records = []
    rows = []

    for case_name in case_names:
        source = candidates_path(case_name)
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

        if not isinstance(payload, list):
            continue

        for ping_idx, ping in enumerate(payload):
            timestamp = ping.get("timestamp") if isinstance(ping, dict) else None
            provenance = {
                field: ping.get(field) if isinstance(ping, dict) else None
                for field in _PING_PROVENANCE_FIELDS
            }
            coords = ping.get("original_coords", [None, None]) if isinstance(ping, dict) else [None, None]
            candidates = ping.get("candidates", []) if isinstance(ping, dict) else []
            if not isinstance(candidates, list):
                candidates = []

            if not candidates:
                rows.append(
                    {
                        "case_study": case_name,
                        "ping_index": ping_idx,
                        "timestamp": timestamp,
                        **provenance,
                        "original_lat": coords[0] if len(coords) > 0 else None,
                        "original_lon": coords[1] if len(coords) > 1 else None,
                        "candidate_index": None,
                        "edge_id": None,
                        "dist_m": None,
                        "speed_limit": None,
                        "is_tunnel": None,
                        "layer": None,
                        "candidate_count_in_ping": 0,
                        "source_file": str(source),
                        "source_filename": source.name,
                    }
                )
                continue

            total = len(candidates)
            for cand_idx, cand in enumerate(candidates):
                if not isinstance(cand, dict):
                    cand = {}
                rows.append(
                    {
                        "case_study": case_name,
                        "ping_index": ping_idx,
                        "timestamp": timestamp,
                        **provenance,
                        "original_lat": coords[0] if len(coords) > 0 else None,
                        "original_lon": coords[1] if len(coords) > 1 else None,
                        "candidate_index": cand_idx,
                        "edge_id": cand.get("edge_id"),
                        "dist_m": cand.get("dist_m"),
                        "speed_limit": cand.get("speed_limit"),
                        "is_tunnel": cand.get("is_tunnel"),
                        "layer": cand.get("layer"),
                        "candidate_count_in_ping": total,
                        "source_file": str(source),
                        "source_filename": source.name,
                    }
                )

    out_json = REPORTS_DIR / "step08_candidates_all_cases.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=_json_default)

    out_csv = REPORTS_DIR / "step08_candidates_all_cases.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Saved consolidated Step08 candidates JSON: {out_json}")
    print(f"Saved consolidated Step08 candidates CSV: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate edge candidates for all hard-case maps.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_max_workers(),
        help="Maximum parallel workers to use (default: all CPU cores).",
    )
    parser.add_argument(
        "--generate-synthetic-if-missing",
        action="store_true",
        help=(
            "Legacy option name: export one canonical held-out truck's observed/"
            "causally-imputed fixes when the Step08 input is missing."
        ),
    )
    parser.add_argument(
        "--force-generate-synthetic",
        action="store_true",
        help=(
            "Legacy option name: regenerate Step08 input from one canonical held-out "
            "truck before candidate generation."
        ),
    )
    args = parser.parse_args()

    ensure_project_dirs()
    case_names = list(hard_case_locations.keys())
    case_to_gps = {}
    for case_name in case_names:
        gps_csv = TRAJECTORIES_DIR / f"step08_{case_name}_filtered_truck_data.csv"
        if args.force_generate_synthetic:
            generate_case_aligned_truck_data(case_name, gps_csv)
        elif not gps_csv.exists() and args.generate_synthetic_if_missing:
            generate_case_aligned_truck_data(case_name, gps_csv)

        if not gps_csv.exists():
            raise FileNotFoundError(
                f"Missing input trajectory file: {gps_csv}. "
                "Run with --generate-synthetic-if-missing or --force-generate-synthetic."
            )

        case_to_gps[case_name] = gps_csv

    max_workers = max(1, min(int(args.max_workers), len(case_names)))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_generate_candidates_for_case, case_name, case_to_gps[case_name])
            for case_name in case_names
        ]
        for fut in as_completed(futures):
            print(fut.result())

    write_step08_consolidated_outputs(case_names)
