"""Validate generated truck trajectories against graph topology and GNSS recovery rules."""
from pathlib import Path
import re

import networkx as nx
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, Point

from pipeline_config import REPORTS_DIR, ensure_project_dirs, trajectory_path, unified_graph_path


CASE_NAMES = [
    "Rozelle_Interchange_NSW", "West_Gate_Tunnel_VIC", "NorthConnex_NSW",
    "Light_Horse_Interchange_NSW", "Domain_Tunnel_VIC", "M80_Princes_Freeway_VIC",
]

MIN_TUNNEL_EDGE_COVERAGE = 0.50
MIN_BRIDGE_EDGE_COVERAGE = 0.50
MAX_TRUCK_SPEED_MPS = 25.0
MAX_SPEED_ERROR_MPS = 0.05
MAX_ACCELERATION_MPS2 = 0.10


def to_flag(value):
    """Boolean flag parser matching GraphML round-tripping (e.g. "1.0"/"0.0")."""
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    try:
        return float(s) != 0
    except (TypeError, ValueError):
        pass
    return s in {"1", "true", "yes", "y"}


def node_id(value):
    text = str(value)
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else text
    except ValueError:
        return text


def _graph_edges_with_keys(graph):
    if graph.is_multigraph():
        yield from (
            (str(u), str(v), key, data)
            for u, v, key, data in graph.edges(keys=True, data=True)
        )
    else:
        yield from (
            (str(u), str(v), 0, data)
            for u, v, data in graph.edges(data=True)
        )


def _resolve_edge_identity(graph, u, v, raw_key):
    """Resolve an exact GraphML (u, v, key), preserving parallel edges."""
    u, v = node_id(u), node_id(v)
    if not graph.has_edge(u, v):
        return None
    if not graph.is_multigraph():
        return (u, v, 0)
    available = graph.get_edge_data(u, v) or {}
    candidates = [raw_key, str(raw_key)]
    try:
        numeric = float(str(raw_key).strip())
        if numeric.is_integer():
            candidates.extend([int(numeric), str(int(numeric))])
    except (TypeError, ValueError):
        pass
    for candidate in candidates:
        if candidate in available:
            return (u, v, candidate)
    return None


def _edge_data(graph, identity):
    u, v, key = identity
    if graph.is_multigraph():
        return graph.get_edge_data(u, v, key) or {}
    return graph.get_edge_data(u, v) or {}


def _edge_geometry(graph, identity):
    u, v, _key = identity
    data = _edge_data(graph, identity)
    text = data.get("geometry")
    if text:
        try:
            return shapely_wkt.loads(str(text))
        except Exception:
            pass
    return LineString([
        (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
        (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])),
    ])


def _edge_speed_limit_mps(data):
    text = str(data.get("maxspeed", 60))
    match = re.search(r"[-+]?\d*\.?\d+", text)
    speed_kmh = float(match.group(0)) if match else 60.0
    if "mph" in text.lower():
        speed_kmh *= 1.609344
    return float(np.clip(speed_kmh, 5.0, 130.0) / 3.6)


def _length_weight(_u, _v, data):
    """NetworkX weight callback robust to GraphML strings/MultiGraphs."""
    if isinstance(data, dict) and data and all(
        isinstance(value, dict) for value in data.values()
    ):
        values = [float(attrs.get("length", 1.0)) for attrs in data.values()]
        return min(values) if values else 1.0
    return float(data.get("length", 1.0))


def validate_case(case):
    graph_path = unified_graph_path(case)
    csv_path = trajectory_path(case)
    graph = nx.read_graphml(graph_path)
    frame = pd.read_csv(
        csv_path,
        dtype={
            "node_id": str,
            "next_node_id": str,
            "edge_u": str,
            "edge_v": str,
            "edge_key": str,
        },
    )
    invalid_edges = 0
    invalid_fix_links = 0
    invalid_sequence_links = 0
    invalid_true_geometry = 0
    invalid_path_distance = 0
    speed_limit_violations = 0
    acceleration_violations = 0
    cadence_violations = 0
    multilevel_trucks = 0
    tunnel_trucks = 0
    tunnel_exits = 0
    recovered_exits = 0
    open_after_exit = 0
    used_edges = set()
    observed_level_transitions = set()

    for _truck, track in frame.groupby("truck_id", sort=False):
        track = track.reset_index(drop=True)
        layers = set(pd.to_numeric(track["true_layer"], errors="coerce").dropna().astype(int))
        multilevel_trucks += int(len(layers) > 1)
        tunnel_trucks += int((track["true_tunnel"] == 1).any())
        previous_speed = None
        for i, row in track.iterrows():
            u, v = node_id(row["edge_u"]), node_id(row["edge_v"])
            identity = _resolve_edge_identity(graph, u, v, row.get("edge_key", 0))
            if identity is None:
                invalid_edges += 1
            else:
                used_edges.add(identity)
                geometry = _edge_geometry(graph, identity)
                point = Point(float(row["true_x"]), float(row["true_y"]))
                invalid_true_geometry += int(float(geometry.distance(point)) > 1e-5)
                recorded_speed = float(row.get("true_speed_mps", np.nan))
                speed_limit = min(
                    MAX_TRUCK_SPEED_MPS,
                    _edge_speed_limit_mps(_edge_data(graph, identity)),
                )
                speed_limit_violations += int(
                    not np.isfinite(recorded_speed)
                    or recorded_speed < 0.0
                    or recorded_speed > speed_limit + MAX_SPEED_ERROR_MPS
                )
                if previous_speed is not None and np.isfinite(recorded_speed):
                    acceleration_violations += int(
                        abs(recorded_speed - previous_speed) > MAX_ACCELERATION_MPS2
                    )
                previous_speed = recorded_speed
            if node_id(row["node_id"]) != u or node_id(row["next_node_id"]) != v:
                invalid_fix_links += 1
            if i < len(track) - 1:
                next_row = track.iloc[i + 1]
                next_u, next_v = node_id(next_row["edge_u"]), node_id(next_row["edge_v"])
                next_identity = _resolve_edge_identity(
                    graph, next_u, next_v, next_row.get("edge_key", 0)
                )
                current_path_distance = float(row.get("true_path_distance_m", np.nan))
                next_path_distance = float(next_row.get("true_path_distance_m", np.nan))
                delta_path = next_path_distance - current_path_distance
                dt = float(next_row["timestamp"]) - float(row["timestamp"])
                cadence_violations += int(not np.isfinite(dt) or abs(dt - 1.0) > 1e-9)
                expected_distance = float(row.get("true_speed_mps", np.nan)) * dt
                invalid_path_distance += int(
                    not np.isfinite(delta_path)
                    or delta_path < -1e-9
                    or not np.isfinite(expected_distance)
                    or abs(delta_path - expected_distance) > 1e-4
                )
                if identity is not None and next_identity is not None and identity != next_identity and v != next_u:
                    try:
                        skipped_distance = float(
                            nx.shortest_path_length(graph, v, next_u, weight=_length_weight)
                        )
                    except (nx.NetworkXError, nx.NetworkXNoPath):
                        skipped_distance = np.inf
                    if not np.isfinite(skipped_distance) or skipped_distance > delta_path + 1e-3:
                        invalid_sequence_links += 1
                observed_level_transitions.add((int(row["true_layer"]), int(next_row["true_layer"])))
                if int(row["true_tunnel"]) == 1 and int(next_row["true_tunnel"]) == 0:
                    tunnel_exits += 1
                    window = track.iloc[i + 1:min(i + 11, len(track))]
                    states = window["quality_state"].astype(str).str.lower()
                    recovered_exits += int(states.str.contains("recovery|portal", regex=True).any())
                    open_after_exit += int((window["true_tunnel"] == 0).any())

    tunnel_rows = frame[frame["true_tunnel"] == 1]
    open_rows = frame[frame["true_tunnel"] == 0]
    tunnel_hdop = float(pd.to_numeric(tunnel_rows["hdop"], errors="coerce").mean()) if not tunnel_rows.empty else np.nan
    open_hdop = float(pd.to_numeric(open_rows["hdop"], errors="coerce").mean()) if not open_rows.empty else np.nan
    tunnel_sat = float(pd.to_numeric(tunnel_rows["sat_count"], errors="coerce").mean()) if not tunnel_rows.empty else np.nan
    open_sat = float(pd.to_numeric(open_rows["sat_count"], errors="coerce").mean()) if not open_rows.empty else np.nan
    graph_has_tunnel = any(
        to_flag(data.get("tunnel", 0)) or float(data.get("layer", 0) or 0) < 0
        for *_edge, data in graph.edges(data=True)
    )
    keyed_graph_edges = list(_graph_edges_with_keys(graph))
    graph_edges = {(u, v, key) for u, v, key, _data in keyed_graph_edges}
    graph_layers = {int(float(data.get("layer", 0) or 0)) for _u, _v, _key, data in keyed_graph_edges}
    observed_layers = set(pd.to_numeric(frame["true_layer"], errors="coerce").dropna().astype(int))
    feasible_level_transitions = set()
    incoming_layers = {}
    outgoing_layers = {}
    for u, v, _key, data in keyed_graph_edges:
        layer = int(float(data.get("layer", 0) or 0))
        incoming_layers.setdefault(str(v), set()).add(layer)
        outgoing_layers.setdefault(str(u), set()).add(layer)
    for node in graph.nodes:
        for left in incoming_layers.get(str(node), set()):
            for right in outgoing_layers.get(str(node), set()):
                feasible_level_transitions.add((left, right))
    tunnel_graph_edges = {
        (u, v, key) for u, v, key, data in keyed_graph_edges
        if to_flag(data.get("tunnel", 0)) or float(data.get("layer", 0) or 0) < 0
    }
    bridge_graph_edges = {
        (u, v, key) for u, v, key, data in keyed_graph_edges
        if to_flag(data.get("bridge", 0)) or float(data.get("layer", 0) or 0) > 0
    }
    road_classes = {str(data.get("highway", "unknown")) for _u, _v, _key, data in keyed_graph_edges}
    used_road_classes = set()
    for identity in used_edges:
        data = _edge_data(graph, identity)
        used_road_classes.add(str(data.get("highway", "unknown")))
    quality_contrast_passed = (
        not graph_has_tunnel or
        (np.isfinite(tunnel_hdop) and tunnel_hdop > open_hdop and tunnel_sat < open_sat)
    )
    tunnel_coverage = (
        len(used_edges & tunnel_graph_edges) / max(1, len(tunnel_graph_edges))
        if tunnel_graph_edges else np.nan
    )
    bridge_coverage = (
        len(used_edges & bridge_graph_edges) / max(1, len(bridge_graph_edges))
        if bridge_graph_edges else np.nan
    )
    passed = (
        invalid_edges == 0 and invalid_fix_links == 0 and invalid_sequence_links == 0
        and invalid_true_geometry == 0 and invalid_path_distance == 0
        and speed_limit_violations == 0 and acceleration_violations == 0
        and cadence_violations == 0
        and (not graph_has_tunnel or (tunnel_trucks > 0 and tunnel_exits > 0))
        and recovered_exits == tunnel_exits and open_after_exit == tunnel_exits
        and quality_contrast_passed and graph_layers.issubset(observed_layers)
        and (not tunnel_graph_edges or tunnel_coverage >= MIN_TUNNEL_EDGE_COVERAGE)
        and (not bridge_graph_edges or bridge_coverage >= MIN_BRIDGE_EDGE_COVERAGE)
    )
    return {
        "case": case, "trucks": int(frame["truck_id"].nunique()), "pings": int(len(frame)),
        "invalid_edges": invalid_edges, "invalid_fix_links": invalid_fix_links,
        "invalid_sequence_links": invalid_sequence_links,
        "invalid_true_geometry": invalid_true_geometry,
        "invalid_path_distance": invalid_path_distance,
        "speed_limit_violations": speed_limit_violations,
        "acceleration_violations": acceleration_violations,
        "cadence_violations": cadence_violations,
        "multilevel_trucks": multilevel_trucks, "tunnel_trucks": tunnel_trucks,
        "tunnel_exits": tunnel_exits, "recovered_exits": recovered_exits,
        "open_after_exit": open_after_exit, "tunnel_hdop_mean": tunnel_hdop,
        "open_hdop_mean": open_hdop, "tunnel_sat_mean": tunnel_sat,
        "open_sat_mean": open_sat, "quality_contrast_passed": quality_contrast_passed,
        "edge_coverage_pct": 100.0 * len(used_edges) / max(1, len(graph_edges)),
        "tunnel_edge_coverage_pct": 100.0 * tunnel_coverage if tunnel_graph_edges else np.nan,
        "bridge_edge_coverage_pct": 100.0 * bridge_coverage if bridge_graph_edges else np.nan,
        "graph_levels": str(sorted(graph_layers)), "observed_levels": str(sorted(observed_layers)),
        "level_coverage_passed": graph_layers.issubset(observed_layers),
        "feasible_level_transitions": str(sorted(feasible_level_transitions)),
        "observed_level_transitions": str(sorted(observed_level_transitions)),
        "level_transition_coverage_pct": 100.0 * len(feasible_level_transitions & observed_level_transitions) / max(1, len(feasible_level_transitions)),
        "road_class_coverage_pct": 100.0 * len(road_classes & used_road_classes) / max(1, len(road_classes)),
        "validation_passed": passed,
    }


def _consecutive_runs(mask):
    """Yield (start_idx, length) for each run of consecutive True values."""
    run_start = None
    for i, value in enumerate(list(mask) + [False]):
        if value and run_start is None:
            run_start = i
        elif not value and run_start is not None:
            yield run_start, i - run_start
            run_start = None


def trajectory_duration_stats(case):
    """Summarize duration, path distance, cadence, and blackout runs."""
    csv_path = trajectory_path(case)
    frame = pd.read_csv(csv_path)
    frame["actual_time"] = pd.to_datetime(frame["actual_time"], errors="coerce")

    per_truck = []
    for truck_id, track in frame.groupby("truck_id", sort=False):
        track = track.sort_values("timestamp").reset_index(drop=True)
        n_fixes = len(track)
        times = track["actual_time"]
        duration_s = float((times.iloc[-1] - times.iloc[0]).total_seconds()) if n_fixes > 1 else 0.0
        if "true_path_distance_m" in track:
            path_distance = pd.to_numeric(track["true_path_distance_m"], errors="coerce")
            distance_m = float(path_distance.iloc[-1] - path_distance.iloc[0]) if n_fixes > 1 else 0.0
        else:
            dx = track["true_x"].diff().to_numpy()[1:]
            dy = track["true_y"].diff().to_numpy()[1:]
            distance_m = float(np.hypot(dx, dy).sum()) if n_fixes > 1 else 0.0

        quality = track["quality_state"].astype(str).str.lower()
        blackout_mask = quality.str.contains("blackout")
        portal_mask = quality.str.contains("portal")
        blackout_runs = list(_consecutive_runs(blackout_mask.to_numpy()))
        fix_interval_s = duration_s / max(1, n_fixes - 1)

        per_truck.append({
            "case": case, "truck_id": truck_id, "n_fixes": n_fixes,
            "duration_s": duration_s, "distance_m": distance_m,
            "fix_interval_s": fix_interval_s,
            "tunnel_fix_fraction": float((track["true_tunnel"] == 1).mean()),
            "blackout_fix_fraction": float(blackout_mask.mean()),
            "n_blackout_intervals": len(blackout_runs),
            "max_blackout_interval_s": max((length * fix_interval_s for _s, length in blackout_runs), default=0.0),
            "mean_blackout_interval_s": (
                float(np.mean([length * fix_interval_s for _s, length in blackout_runs])) if blackout_runs else 0.0
            ),
            "n_portal_transitions": int((portal_mask.astype(int).diff() == 1).sum()),
        })

    per_truck_df = pd.DataFrame(per_truck)
    return {
        "case": case,
        "n_trucks": int(per_truck_df["truck_id"].nunique()),
        "fixes_per_truck_mean": float(per_truck_df["n_fixes"].mean()),
        "duration_s_mean": float(per_truck_df["duration_s"].mean()),
        "duration_s_median": float(per_truck_df["duration_s"].median()),
        "distance_m_mean": float(per_truck_df["distance_m"].mean()),
        "fix_interval_s_mean": float(per_truck_df["fix_interval_s"].mean()),
        "tunnel_fix_fraction_mean": float(per_truck_df["tunnel_fix_fraction"].mean()),
        "blackout_fix_fraction_mean": float(per_truck_df["blackout_fix_fraction"].mean()),
        "n_blackout_intervals_mean": float(per_truck_df["n_blackout_intervals"].mean()),
        "max_blackout_interval_s_mean": float(per_truck_df["max_blackout_interval_s"].mean()),
        "max_blackout_interval_s_p95": float(per_truck_df["max_blackout_interval_s"].quantile(0.95)),
        "mean_blackout_interval_s_mean": float(per_truck_df["mean_blackout_interval_s"].mean()),
        "n_portal_transitions_mean": float(per_truck_df["n_portal_transitions"].mean()),
        "sustained_multiminute_blackout_pct_trucks": float(
            (per_truck_df["max_blackout_interval_s"] >= 120.0).mean() * 100.0
        ),
    }


def main():
    ensure_project_dirs()
    rows = [validate_case(case) for case in CASE_NAMES]
    result = pd.DataFrame(rows)
    output = REPORTS_DIR / "step03_trajectory_topology_and_recovery_validation.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Saved trajectory validation: {output}")

    duration_rows = [trajectory_duration_stats(case) for case in CASE_NAMES]
    duration_result = pd.DataFrame(duration_rows)
    duration_output = REPORTS_DIR / "step03_trajectory_duration_and_blackout_stats.csv"
    duration_result.to_csv(duration_output, index=False)
    print(duration_result.to_string(index=False))
    print(f"Saved trajectory duration/blackout statistics: {duration_output}")

    if not result["validation_passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
