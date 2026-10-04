import pandas as pd
import numpy as np
import networkx as nx
import argparse
import os
import json
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import KDTree
import pennylane as qml
from pennylane import numpy as pnp
from pathlib import Path
from pipeline_config import DEFAULT_SEED, INFERENCE_CANDIDATE_RADIUS_M, PLOTS_DIR, REPORTS_DIR, default_max_workers, deterministic_ansatz_init, ensure_project_dirs, load_confidence_threshold, make_qml_device, qts_weights_path, unified_graph_path, trajectory_path, weights_path
try:
    from shapely.geometry import Point
    from shapely.strtree import STRtree
except ImportError:
    Point = None
    STRtree = None


CASE_NAMES = [
    "Rozelle_Interchange_NSW",
    "West_Gate_Tunnel_VIC",
    "NorthConnex_NSW",
    "Light_Horse_Interchange_NSW",
    "Domain_Tunnel_VIC",
    "M80_Princes_Freeway_VIC",
]


def write_step04_consolidated_outputs(case_names):
    records = []
    rows = []
    for case_name in case_names:
        source = REPORTS_DIR / f"step04_{case_name}_quantum_viterbi_path.json"
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

        path = payload.get("quantum_path", []) if isinstance(payload, dict) else []
        for idx, edge in enumerate(path):
            u = edge[0] if isinstance(edge, list) and len(edge) > 0 else None
            v = edge[1] if isinstance(edge, list) and len(edge) > 1 else None
            k = edge[2] if isinstance(edge, list) and len(edge) > 2 else None
            rows.append(
                {
                    "case_study": case_name,
                    "path_index": idx,
                    "u": u,
                    "v": v,
                    "k": k,
                    "seed": payload.get("seed") if isinstance(payload, dict) else None,
                    "source_file": str(source),
                    "source_filename": source.name,
                }
            )

    out_json = REPORTS_DIR / "step04_quantum_viterbi_path_all_cases.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    out_csv = REPORTS_DIR / "step04_quantum_viterbi_path_all_cases.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved consolidated Step04 JSON: {out_json}")
    print(f"Saved consolidated Step04 CSV: {out_csv}")


def write_step04_visual_consolidated_outputs(case_names):
    frames = []
    for case_name in case_names:
        source = REPORTS_DIR / f"step04_{case_name}_plot_data.csv"
        if not source.exists():
            continue
        df = pd.read_csv(source)
        if not df.empty:
            frames.append(df)

    out_csv = REPORTS_DIR / "step04_plot_data_all_cases.csv"
    if frames:
        pd.concat(frames, axis=0, ignore_index=True).to_csv(out_csv, index=False)
    else:
        pd.DataFrame(
            columns=[
                "case_study",
                "point_index",
                "obs_x",
                "obs_y",
                "obs_z",
                "snapped_x",
                "snapped_y",
                "snapped_z",
                "snapped_u",
                "snapped_v",
                "snapped_k",
            ]
        ).to_csv(out_csv, index=False)
    print(f"Saved consolidated Step04 plot data: {out_csv}")


def to_float_safely(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_prob_from_expectation(z_val):
    """Map Pauli-Z expectation [-1, 1] to probability [0, 1]."""
    z = to_float_safely(z_val, 0.0)
    return float(np.clip((z + 1.0) / 2.0, 1e-9, 1.0))


def to_flag(value):
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    try:
        return float(s) != 0
    except (TypeError, ValueError):
        pass
    return s in {"1", "true", "yes", "y"}


def _pick_xy_columns(df):
    x_candidates = ["obs_x", "x_noisy", "x"]
    y_candidates = ["obs_y", "y_noisy", "y"]

    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)

    if x_col is None or y_col is None:
        raise ValueError(
            "Could not find trajectory XY columns. Expected one of "
            "obs_x/x_noisy/x and obs_y/y_noisy/y."
        )

    return x_col, y_col


def _graph_edge_attrs(graph, edge):
    u, v, k = edge
    data = graph.get_edge_data(u, v)
    if data is None:
        return {}
    if graph.is_multigraph():
        return data.get(k, {})
    return data


def build_step04_plot_data(graph, traj_df, snapped_edges, case_name):
    from step05_quantum_classical_benchmarking import build_edge_geometry_from_graph

    x_col, y_col = _pick_xy_columns(traj_df)
    z_series = traj_df["z_noisy"] if "z_noisy" in traj_df.columns else pd.Series([0.0] * len(traj_df))

    point_count = min(len(traj_df), len(snapped_edges))
    rows = []
    for idx in range(point_count):
        edge = snapped_edges[idx]
        u_node = graph.nodes[edge[0]]
        attrs = _graph_edge_attrs(graph, edge)
        snapped_z = float(attrs.get("layer", 0)) * 5.0
        obs_x = to_float_safely(traj_df.iloc[idx][x_col], np.nan)
        obs_y = to_float_safely(traj_df.iloc[idx][y_col], np.nan)
        snapped_x, snapped_y = float(u_node["x"]), float(u_node["y"])
        if Point is not None and np.all(np.isfinite([obs_x, obs_y])):
            edge_geometry = build_edge_geometry_from_graph(graph, edge, attrs)
            projection = edge_geometry.interpolate(edge_geometry.project(Point(obs_x, obs_y)))
            snapped_x, snapped_y = float(projection.x), float(projection.y)
        rows.append(
            {
                "case_study": case_name,
                "point_index": idx,
                "obs_x": float(traj_df.iloc[idx][x_col]),
                "obs_y": float(traj_df.iloc[idx][y_col]),
                "obs_z": float(z_series.iloc[idx]),
                "snapped_x": snapped_x,
                "snapped_y": snapped_y,
                "snapped_z": snapped_z,
                "snapped_u": str(edge[0]),
                "snapped_v": str(edge[1]),
                "snapped_k": int(edge[2]),
            }
        )

    return pd.DataFrame(rows)


def visualize_3d_match(graph_path, trajectory_csv, snapped_edges, output_png=None):
    from step05_quantum_classical_benchmarking import build_edge_geometry_from_graph

    graph = nx.read_graphml(graph_path)
    traj_df = pd.read_csv(trajectory_csv)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    if graph.is_multigraph():
        edge_iter = graph.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, 0, data) for u, v, data in graph.edges(data=True))

    for u, v, _k, data in edge_iter:
        u_node = graph.nodes[u]
        v_node = graph.nodes[v]
        z_level = float(data.get("layer", 0)) * 5.0
        ax.plot(
            [float(u_node["x"]), float(v_node["x"])],
            [float(u_node["y"]), float(v_node["y"])],
            [z_level, z_level],
            color="lightgrey",
            alpha=0.3,
            linewidth=1,
        )

    x_col, y_col = _pick_xy_columns(traj_df)
    z_source = traj_df["z_noisy"] if "z_noisy" in traj_df.columns else 0
    ax.scatter(traj_df[x_col], traj_df[y_col], z_source, c="red", s=2, alpha=0.5, label="Noisy GPS (SDM)")

    snapped_coords = []
    for idx, edge in enumerate(snapped_edges):
        u_node = graph.nodes[edge[0]]
        attrs = _graph_edge_attrs(graph, edge)
        z_level = float(attrs.get("layer", 0)) * 5.0
        snapped_x, snapped_y = float(u_node["x"]), float(u_node["y"])
        if idx < len(traj_df) and Point is not None:
            obs_x = to_float_safely(traj_df.iloc[idx][x_col], np.nan)
            obs_y = to_float_safely(traj_df.iloc[idx][y_col], np.nan)
            if np.all(np.isfinite([obs_x, obs_y])):
                edge_geometry = build_edge_geometry_from_graph(graph, edge, attrs)
                projection = edge_geometry.interpolate(edge_geometry.project(Point(obs_x, obs_y)))
                snapped_x, snapped_y = float(projection.x), float(projection.y)
        snapped_coords.append([snapped_x, snapped_y, z_level])

    if snapped_coords:
        snapped_df = pd.DataFrame(snapped_coords, columns=["x", "y", "z"])
        ax.plot(snapped_df["x"], snapped_df["y"], snapped_df["z"], color="blue", linewidth=2, label="Quantum Snapped Path")

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Relative Layer (Z)")
    ax.set_title("Quantum Map-Matching: 3D Resolution of Vertical Ambiguity")
    ax.legend()

    if output_png:
        plt.savefig(output_png, dpi=250, bbox_inches="tight")
        print(f"Saved 3D visualization: {output_png}")
    plt.close(fig)


# Sentinel distinct from None so a cached "no path exists" result (None)
# is distinguishable from "not yet computed" in the transition hop cache.
_UNSET_HOP = object()


def edge_length_weight(_u, _v, data):
    """NetworkX weight helper that works for Graph and MultiGraph edge payloads."""
    if isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
        lengths = [to_float_safely(attrs.get("length"), 1.0) for attrs in data.values()]
        return min(lengths) if lengths else 1.0
    return to_float_safely(data.get("length"), 1.0)


class QuantumMapMatcher:
    def __init__(self, n_qubits=9, seed=42, weights_file=None):
        self.n_qubits = n_qubits
        self.dev = make_qml_device(qml, wires=self.n_qubits, prefer_gpu=True, gpu_fraction=0.8)
        loaded_weights = None
        if weights_file and Path(weights_file).exists():
            try:
                loaded_weights = pnp.array(np.load(weights_file), requires_grad=False)
            except Exception:
                loaded_weights = None
        if loaded_weights is None:
            if weights_file:
                print(f"  WARNING: calibrated weights not found at {weights_file}; using deterministic fallback init.")
            loaded_weights = pnp.array(
                deterministic_ansatz_init(
                    qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=self.n_qubits), restart=int(seed)
                ),
                requires_grad=False,
            )
        self.weights = loaded_weights

    def normalize_features(self, data, traffic_val):
        # Shared with calibration and the benchmark implementation. Must be
        # wrapped as a pnp array (not a plain list) so `features * pnp.pi`
        # in the circuit below works, matching QuantumEmissionModel.score
        # in step05_quantum_classical_benchmarking.py.
        from step05_quantum_classical_benchmarking import extract_9_features
        return pnp.array(extract_9_features(data, traffic_regime=traffic_val), requires_grad=False)

    def get_qnode(self):
        # Must match QuantumEmissionModel in step05 exactly: the weight
        # tensor loaded from weights_path(case_name) was trained against
        # that circuit's re-uploading + local-readout structure.
        from step05_quantum_classical_benchmarking import QMM_READOUT_WIRES

        readout = qml.PauliZ(QMM_READOUT_WIRES[0])
        for w in QMM_READOUT_WIRES[1:]:
            readout = readout + qml.PauliZ(w)
        readout = readout / float(len(QMM_READOUT_WIRES))

        @qml.qnode(self.dev)
        def circuit(features, weights):
            qml.AngleEmbedding(features * pnp.pi, wires=range(self.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights[0:1], wires=range(self.n_qubits))
            qml.AngleEmbedding(features * pnp.pi, wires=range(self.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights[1:2], wires=range(self.n_qubits))
            return qml.expval(readout)

        return circuit

class QuantumViterbiMatcher:
    def __init__(
        self,
        graph_path,
        trajectory_csv,
        seed=42,
        candidate_radius=INFERENCE_CANDIDATE_RADIUS_M,
        fallback_k=10,
        weights_file=None,
        qts_weights_file=None,
        sigma_xy=45.0,
        sigma_d=50.0,
        confidence_dip_threshold=None,
        kappa_min=20,
        kappa_max=120,
    ):
        self.G = nx.read_graphml(graph_path)
        self.traj_df = pd.read_csv(trajectory_csv)
        self.matcher = QuantumMapMatcher(seed=seed, weights_file=weights_file)
        self.qnode = self.matcher.get_qnode()
        # TeX-aligned edge-pair QTS: C_QTS(e_i,e_j) multiplies the classical
        # distance-mismatch term in the transition weight (see transition_prob).
        from step05_quantum_classical_benchmarking import QuantumTransitionScorer, build_edge_geometry_from_graph
        self.transition_scorer = QuantumTransitionScorer(
            seed=seed + 17,
            weights_file=qts_weights_file,
        )
        self.qts_weights_file = str(qts_weights_file) if qts_weights_file else None
        self._build_edge_geometry_from_graph = build_edge_geometry_from_graph
        self.candidate_radius = float(candidate_radius)
        self.fallback_k = max(1, int(fallback_k))
        self.sigma_xy = float(sigma_xy)
        self.sigma_d = float(sigma_d)
        # Network-hop distance between prev_edge's end node and curr_edge's
        # start node, cached per pair (not used for same-edge transitions;
        # see transition_prob).
        self._transition_hop_cache = {}
        self._edge_length_cache = {}
        self._trellis_breaks = []
        if confidence_dip_threshold is None:
            raise ValueError(
                "confidence_dip_threshold must be supplied from Step06 validation calibration"
            )
        self.confidence_dip_threshold = float(confidence_dip_threshold)
        self.kappa_min = max(1, int(kappa_min))
        self.kappa_max = max(self.kappa_min, int(kappa_max))
        
        # Build a spatial index for fast candidate lookup. Geometry-aware
        #: search tests true
        # distance to each edge's full line geometry (WKT centerline when
        # present, else the straight endpoint segment), not a single
        # per-edge midpoint, so a long or curved edge is not excluded from
        # candidacy just because its midpoint sits outside the radius.
        self.edge_coords = []
        self.edge_ids = []
        self.edge_geometries = []
        if self.G.is_multigraph():
            edge_iter = self.G.edges(keys=True, data=True)
        else:
            edge_iter = ((u, v, 0, data) for u, v, data in self.G.edges(data=True))

        for u, v, k, data in edge_iter:
            edge = (u, v, k)
            u_data = self.G.nodes[u]
            v_data = self.G.nodes[v]
            mid_x = (to_float_safely(u_data.get('x')) + to_float_safely(v_data.get('x'))) / 2
            mid_y = (to_float_safely(u_data.get('y')) + to_float_safely(v_data.get('y'))) / 2
            self.edge_coords.append([mid_x, mid_y])
            self.edge_ids.append(edge)
            self.edge_geometries.append(self._build_edge_geometry_from_graph(self.G, edge, data))

        self.edge_strtree = STRtree(self.edge_geometries) if (STRtree is not None and self.edge_geometries) else None
        self.tree = KDTree(self.edge_coords)
        self._edge_geometry_by_id = dict(zip(self.edge_ids, self.edge_geometries))

    def get_candidates(self, x, y, radius=None, previous_candidates=None):
        """Finds edges within search radius of a GPS ping."""
        radius_val = self.candidate_radius if radius is None else float(radius)

        if np.isnan(x) or np.isnan(y):
            if previous_candidates:
                return list(previous_candidates)
            return self.edge_ids[: self.fallback_k]

        if self.edge_strtree is not None:
            point = Point(float(x), float(y))
            bbox_idxs = self.edge_strtree.query(point.buffer(radius_val))
            matches = [int(i) for i in bbox_idxs if self.edge_geometries[int(i)].distance(point) <= radius_val]
            if matches:
                return [self.edge_ids[i] for i in matches]

            expand_radius = max(radius_val, 1.0)
            bbox_idxs = np.asarray([], dtype=int)
            for _ in range(10):
                expand_radius *= 2.0
                bbox_idxs = self.edge_strtree.query(point.buffer(expand_radius))
                if len(bbox_idxs) >= self.fallback_k or len(bbox_idxs) >= len(self.edge_ids):
                    break
            if len(bbox_idxs) == 0:
                return self.edge_ids[: self.fallback_k]
            distances = sorted((self.edge_geometries[int(i)].distance(point), int(i)) for i in bbox_idxs)
            return [self.edge_ids[i] for _, i in distances[: self.fallback_k]]

        # Legacy fallback (only reachable if Shapely is unavailable).
        idxs = self.tree.query_ball_point([x, y], radius_val)
        candidates = [self.edge_ids[i] for i in idxs]
        if not candidates:
            nearest = self.tree.query([x, y], k=min(self.fallback_k, len(self.edge_ids)))[1]
            nearest = np.atleast_1d(nearest)
            return [self.edge_ids[int(i)] for i in nearest]
        return candidates

    def get_edge_attrs(self, edge):
        u, v, k = edge
        data = self.G.get_edge_data(u, v)
        if data is None:
            return {}
        if self.G.is_multigraph():
            return data.get(k, {})
        return data

    def project_to_edge(self, edge, x, y):
        """Project ``(x, y)`` onto the edge's full WKT centreline.

        Candidate lookup, the spatial emission, and route-distance scoring
        must use the same geometry.  Falling back to the endpoint chord is
        handled once by ``build_edge_geometry_from_graph`` when no valid WKT
        is available.
        """
        if Point is None or not np.all(np.isfinite([x, y])):
            return np.nan, np.nan
        geometry = self._edge_geometry_by_id.get(edge)
        if geometry is None or geometry.is_empty:
            return np.nan, np.nan
        projected = geometry.interpolate(geometry.project(Point(float(x), float(y))))
        return float(projected.x), float(projected.y)

    def _edge_length(self, edge):
        length = self._edge_length_cache.get(edge)
        if length is None:
            geometry = self._edge_geometry_by_id.get(edge)
            length = float(geometry.length) if geometry is not None else 0.0
            self._edge_length_cache[edge] = length
        return length

    def _edge_arclength_of_point(self, edge, x, y):
        """Arc-length offset of the point on `edge` nearest (x, y), from the
        edge geometry's own start (same computation as step05)."""
        if Point is None:
            return 0.0
        geometry = self._edge_geometry_by_id.get(edge)
        if geometry is None:
            return 0.0
        return float(geometry.project(Point(float(x), float(y))))

    def spatial_log_prob(self, edge, x, y, sigma=None):
        sigma_val = self.sigma_xy if sigma is None else float(sigma)
        if np.isnan(x) or np.isnan(y):
            return np.log(1e-6)
        ex, ey = self.project_to_edge(edge, x, y)
        if np.isnan(ex) or np.isnan(ey):
            return np.log(1e-6)
        dist = float(np.hypot(x - ex, y - ey))
        return -0.5 * (dist / max(sigma_val, 1e-6)) ** 2

    def observation_displacement(self, prev_ping, curr_ping):
        px = to_float_safely(prev_ping.get("obs_x"), np.nan)
        py = to_float_safely(prev_ping.get("obs_y"), np.nan)
        cx = to_float_safely(curr_ping.get("obs_x"), np.nan)
        cy = to_float_safely(curr_ping.get("obs_y"), np.nan)
        if np.isnan(px) or np.isnan(py) or np.isnan(cx) or np.isnan(cy):
            return 0.0
        return float(np.hypot(cx - px, cy - py))

    def predict_observation(self, prev_prev_ping, prev_ping, curr_ping):
        """Fill a missing observation from the *processed* state history.

        ``run_matching`` feeds this function its previously observed or
        predicted rows, rather than re-reading the raw CSV.  A multi-fix GPS
        blackout therefore continues the finite-difference trajectory instead
        of predicting only its first missing row and then reverting to NaNs.
        State is reset at every truck boundary.
        """
        if prev_prev_ping is None:
            return curr_ping

        p2x = to_float_safely(prev_prev_ping.get("obs_x"), np.nan)
        p2y = to_float_safely(prev_prev_ping.get("obs_y"), np.nan)
        p1x = to_float_safely(prev_ping.get("obs_x"), np.nan)
        p1y = to_float_safely(prev_ping.get("obs_y"), np.nan)
        cx = to_float_safely(curr_ping.get("obs_x"), np.nan)
        cy = to_float_safely(curr_ping.get("obs_y"), np.nan)

        if np.isnan(p2x) or np.isnan(p2y) or np.isnan(p1x) or np.isnan(p1y):
            return curr_ping

        pred = curr_ping.copy()
        vx = p1x - p2x
        vy = p1y - p2y
        if np.isnan(cx):
            pred["obs_x"] = p1x + vx
        if np.isnan(cy):
            pred["obs_y"] = p1y + vy
        return pred

    def max_qmm_expectation(self, candidates, ping):
        if not candidates:
            return -1.0
        vals = []
        for edge in candidates:
            q_raw = self.qnode(
                self.matcher.normalize_features(self.get_edge_attrs(edge), ping.get("time_qubit_val", 0.5)),
                self.matcher.weights,
            )
            vals.append(float(np.clip(q_raw, -1.0, 1.0)))
        return max(vals) if vals else -1.0

    def emission_log_prob(self, edge, ping):
        q_raw = self.qnode(
            self.matcher.normalize_features(self.get_edge_attrs(edge), ping.get("time_qubit_val", 0.5)),
            self.matcher.weights,
        )
        quantum_prob = to_prob_from_expectation(q_raw)
        spatial_lp = self.spatial_log_prob(
            edge,
            to_float_safely(ping.get("obs_x"), np.nan),
            to_float_safely(ping.get("obs_y"), np.nan),
        )
        gauss_prob = float((1.0 / (2.0 * np.pi * max(self.sigma_xy, 1e-6) ** 2)) * np.exp(spatial_lp))
        return float(np.log(np.clip(gauss_prob * quantum_prob, 1e-12, 1.0)))

    def transition_prob(self, prev_edge, curr_edge, obs_disp, prev_xy=None, curr_xy=None):
        """Route-distance transition score (same model as
        step05._transition_prob_with_delta). Same-edge transitions, the
        dominant type at ~93-95% of consecutive fixes, are scored as the along-edge arc-length between the two
        projected positions; cross-edge transitions add the remaining
        distance to prev_edge's end and from curr_edge's start to the
        cached network hop, all measured against each edge's own geometry
        rather than its endpoint nodes."""
        same_edge = prev_edge == curr_edge

        if same_edge:
            hop = 0.0
        else:
            pair_key = (prev_edge, curr_edge)
            hop = self._transition_hop_cache.get(pair_key, _UNSET_HOP)
            if hop is _UNSET_HOP:
                try:
                    hop = float(
                        nx.shortest_path_length(
                            self.G,
                            prev_edge[1],
                            curr_edge[0],
                            weight=edge_length_weight,
                        )
                    )
                except nx.NetworkXNoPath:
                    # Hard graph-reachability constraint (ported from step05):
                    # no path at all from the previous edge's
                    # end node means this pair is excluded, not down-weighted.
                    # Only applies to genuine cross-edge transitions.
                    hop = None
                self._transition_hop_cache[pair_key] = hop
            if hop is None:
                return 0.0

        if prev_xy is not None and curr_xy is not None:
            prev_arc = self._edge_arclength_of_point(prev_edge, *prev_xy)
            curr_arc = self._edge_arclength_of_point(curr_edge, *curr_xy)
            if same_edge:
                # Directed progress: backward motion on a directed edge must
                # not be converted into an equally plausible forward move.
                dist = max(curr_arc - prev_arc, 0.0)
            else:
                remaining_prev = max(self._edge_length(prev_edge) - prev_arc, 0.0)
                dist = remaining_prev + hop + curr_arc
        else:
            dist = float(max(obs_disp, 0.0)) if same_edge else hop

        try:
            mismatch = abs(dist - float(max(obs_disp, 0.0)))
            base_prob = float(np.exp(-(mismatch / max(self.sigma_d, 1e-6))))
            qts_prob = float(
                self.transition_scorer.score(
                    self.get_edge_attrs(prev_edge),
                    self.get_edge_attrs(curr_edge),
                    dist,
                )
            )
            # w_tau(e_i,e_j) = exp(-|deltaD - deltax|/sigma_d) * C_QTS(e_i,e_j)
            fused = base_prob * qts_prob
            return float(np.clip(fused, 1e-9, 1.0))
        except (TypeError, ValueError):
            return 1e-6

    def run_matching(self, max_steps=None, progress_every=100):
        # Initialize Viterbi Trellis
        # path_scores[time_step][edge_id] = max_log_prob
        path_scores = []
        backpointers = []
        total_steps = len(self.traj_df) if max_steps is None else min(len(self.traj_df), int(max_steps))
        if total_steps < 2:
            raise RuntimeError("Need at least 2 trajectory points for Viterbi matching.")
        self._trellis_breaks = []

        print(f"Running Viterbi over {total_steps} points...")
        
        # 1. Process Initial Observation. ``processed_pings`` is deliberately
        # stateful: it contains the observations actually used by the decoder
        # (measured when available, extrapolated during a blackout).
        p0 = self.traj_df.iloc[0].copy()
        processed_pings = [p0]
        candidates = self.get_candidates(
            to_float_safely(p0.get("obs_x"), np.nan),
            to_float_safely(p0.get("obs_y"), np.nan),
            previous_candidates=None,
        )
        
        initial_probs = {}
        for edge in candidates:
            initial_probs[edge] = self.emission_log_prob(edge, p0)

        if not initial_probs:
            raise RuntimeError("No initial candidates found for the first observation.")
            
        path_scores.append(initial_probs)
        previous_active_states = list(initial_probs.keys())

        ambiguity_mode = False
        freeze_anchor = 0
        adaptive_window = self.kappa_min

        # 2. Viterbi Recursion
        for t in range(1, total_steps):
            raw_curr_ping = self.traj_df.iloc[t].copy()
            raw_prev_ping = self.traj_df.iloc[t - 1]

            # Separate vehicles are independent sequences.  Never create a
            # graph transition, kinematic velocity, or Viterbi backpointer
            # across a truck boundary.
            truck_boundary = False
            if "truck_id" in self.traj_df.columns:
                truck_boundary = str(raw_curr_ping.get("truck_id")) != str(raw_prev_ping.get("truck_id"))

            if truck_boundary:
                curr_ping = raw_curr_ping
                curr_candidates = self.get_candidates(
                    to_float_safely(curr_ping.get("obs_x"), np.nan),
                    to_float_safely(curr_ping.get("obs_y"), np.nan),
                    previous_candidates=None,
                )
                new_probs = {
                    edge: self.emission_log_prob(edge, curr_ping)
                    for edge in curr_candidates
                }
                if not new_probs:
                    raise RuntimeError(f"No candidates found at truck boundary t={t}.")
                path_scores.append(new_probs)
                backpointers.append({edge: edge for edge in new_probs})
                self._trellis_breaks.append(t)
                previous_active_states = list(new_probs)
                processed_pings = [curr_ping]
                ambiguity_mode = False
                freeze_anchor = t
                adaptive_window = self.kappa_min
                continue

            prev_ping = processed_pings[-1]
            prev_prev_ping = processed_pings[-2] if len(processed_pings) > 1 else None
            curr_ping = raw_curr_ping

            probe_candidates = self.get_candidates(
                to_float_safely(curr_ping.get("obs_x"), np.nan),
                to_float_safely(curr_ping.get("obs_y"), np.nan),
                previous_candidates=previous_active_states,
            )
            qmm_z = self.max_qmm_expectation(probe_candidates, curr_ping)
            raw_x = to_float_safely(curr_ping.get("obs_x"), np.nan)
            raw_y = to_float_safely(curr_ping.get("obs_y"), np.nan)
            observation_missing = not (np.isfinite(raw_x) and np.isfinite(raw_y))

            if qmm_z <= self.confidence_dip_threshold:
                if not ambiguity_mode:
                    freeze_anchor = max(0, t - adaptive_window)
                ambiguity_mode = True
                adaptive_window = min(self.kappa_max, adaptive_window + 1)
            elif ambiguity_mode:
                ambiguity_mode = False
                adaptive_window = self.kappa_min

            if observation_missing or ambiguity_mode:
                curr_ping = self.predict_observation(prev_prev_ping, prev_ping, curr_ping)

            # Persist the filled state before processing the next timestamp.
            processed_pings.append(curr_ping)

            curr_candidates = self.get_candidates(
                to_float_safely(curr_ping.get("obs_x"), np.nan),
                to_float_safely(curr_ping.get("obs_y"), np.nan),
                previous_candidates=previous_active_states,
            )
            obs_disp = self.observation_displacement(prev_ping, curr_ping)

            # Projected observation positions used to place the vehicle at
            # its true along-edge location, not an edge endpoint node.
            prev_obs_x = to_float_safely(prev_ping.get("obs_x"), np.nan)
            prev_obs_y = to_float_safely(prev_ping.get("obs_y"), np.nan)
            curr_obs_x = to_float_safely(curr_ping.get("obs_x"), np.nan)
            curr_obs_y = to_float_safely(curr_ping.get("obs_y"), np.nan)
            have_xy = np.isfinite(prev_obs_x) and np.isfinite(prev_obs_y) and np.isfinite(curr_obs_x) and np.isfinite(curr_obs_y)
            prev_xy = (prev_obs_x, prev_obs_y) if have_xy else None
            curr_xy = (curr_obs_x, curr_obs_y) if have_xy else None

            new_probs = {}
            new_backpointers = {}

            prev_items = list(path_scores[t - 1].items())
            for curr_edge in curr_candidates:
                best_prev_score = -float('inf')
                best_prev_edge = None

                emission_lp = self.emission_log_prob(curr_edge, curr_ping)

                for prev_edge, prev_score in prev_items:
                    # Transition Probability: Topological Distance
                    # Graphs use the local GDA2020 / MGA zone, so this is metric (meters).
                    # Use the same unnormalized pairwise potential as the
                    # canonical Step05 decoder.  Candidate-set normalization
                    # made one edge-pair score depend on unrelated candidates.
                    p_trans = self.transition_prob(
                        prev_edge,
                        curr_edge,
                        obs_disp=obs_disp,
                        prev_xy=prev_xy,
                        curr_xy=curr_xy,
                    )
                    if p_trans <= 0.0:
                        # Hard graph-reachability constraint (ported from
                        # step05): never selectable while any
                        # reachable predecessor exists for curr_edge,
                        # regardless of accumulated cumulative score.
                        continue

                    # Log-space scoring to prevent underflow
                    total_score = prev_score + np.log(p_trans) + emission_lp

                    if total_score > best_prev_score:
                        best_prev_score = total_score
                        best_prev_edge = prev_edge

                if best_prev_edge is not None:
                    new_probs[curr_edge] = best_prev_score
                    new_backpointers[curr_edge] = best_prev_edge

            if not new_probs:
                # Total connectivity loss (fallback rule, ported
                # from step05): every current candidate is graph-unreachable
                # from every surviving predecessor. Re-anchor the trellis at
                # the CURRENT candidates using only their emission score
                # (same rule as t=0 initialization) instead of silently
                # copying t-1's stale score distribution forward, which
                # would freeze the decoded path at the OLD edge set and
                # discard this timestep's emission evidence. Self-pointing
                # backpointers mark this as a documented break.
                for curr_edge in curr_candidates:
                    new_probs[curr_edge] = self.emission_log_prob(curr_edge, curr_ping)
                new_backpointers = {edge: edge for edge in new_probs.keys()}
                self._trellis_breaks.append(t)
            
            path_scores.append(new_probs)
            backpointers.append(new_backpointers)
            previous_active_states = list(new_probs.keys())

            if progress_every and (t % progress_every == 0 or t == total_steps - 1):
                print(
                    f"  > Step {t}/{total_steps - 1}: active states={len(new_probs)}, "
                    f"z={qmm_z:.3f}, ambiguity={'on' if ambiguity_mode else 'off'}, "
                    f"W={adaptive_window}, anchor={freeze_anchor}"
                )

        if self._trellis_breaks:
            print(
                f"trellis re-anchored at {len(self._trellis_breaks)}/{total_steps - 1} "
                f"step(s) at a sequence boundary or after total connectivity loss"
            )

        # 3. Backtrack to find the optimal path
        return self.backtrack(path_scores, backpointers, total_steps)

    def backtrack(self, scores, pointers, total_steps):
        # Segmented backtrack (same as step05): a single backpointer walk
        # would treat a re-anchor's self-pointing sentinel as a real
        # same-edge transition and smear that edge backward over the true
        # pre-break path. Each segment between re-anchor points is
        # backtracked independently instead.
        from step05_quantum_classical_benchmarking import _segmented_backtrack
        return _segmented_backtrack(scores, pointers, self._trellis_breaks, total_steps)

    @staticmethod
    def serialize_path(path):
        return [[str(u), str(v), int(k)] for u, v, k in path]


def save_viterbi_output(case_name, seed, snapped_path, params):
    payload = {
        "schema": "quantum_viterbi_path_v1",
        "case": case_name,
        "seed": int(seed),
        "params": params,
        "quantum_path": QuantumViterbiMatcher.serialize_path(snapped_path),
    }

    out_primary = REPORTS_DIR / f"step04_{case_name}_quantum_viterbi_path.json"
    with open(out_primary, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_primary


def _run_single_case(graph_path, traj_path, seed, radius, fallback_k):
    print(f"\n=== Running {Path(graph_path).name} ===")
    case_name = Path(graph_path).name.replace("_unified.graphml", "").removeprefix("step01_")
    case_weights_file = weights_path(case_name)
    case_qts_weights_file = qts_weights_path(case_name)
    if not case_weights_file.exists():
        raise FileNotFoundError(
            f"Missing trained QMM weights {case_weights_file}; run Step06 calibration first."
        )
    if not case_qts_weights_file.exists():
        raise FileNotFoundError(
            f"Missing trained QTS weights {case_qts_weights_file}; run Step06 calibration first."
        )
    matcher_system = QuantumViterbiMatcher(
        graph_path,
        traj_path,
        seed=seed,
        candidate_radius=radius,
        fallback_k=fallback_k,
        weights_file=str(case_weights_file),
        qts_weights_file=str(case_qts_weights_file),
        confidence_dip_threshold=load_confidence_threshold(case_name),
    )
    snapped_path = matcher_system.run_matching(max_steps=None, progress_every=200)
    output_path = save_viterbi_output(
        case_name,
        seed,
        snapped_path,
        {
            "candidate_radius": float(radius),
            "fallback_k": int(fallback_k),
            "weights_file": str(case_weights_file),
            "qts_weights_file": str(case_qts_weights_file),
        },
    )

    case_plot_png = PLOTS_DIR / f"step04_{case_name}_3d_noisy_vs_quantum.png"
    visualize_3d_match(graph_path, traj_path, snapped_path, output_png=case_plot_png)

    graph = nx.read_graphml(graph_path)
    traj_df = pd.read_csv(traj_path)
    step04_df = build_step04_plot_data(graph, traj_df, snapped_path, case_name)
    case_csv = REPORTS_DIR / f"step04_{case_name}_plot_data.csv"
    step04_df.to_csv(case_csv, index=False)
    print(f"Saved Step04 plot data CSV: {case_csv}")

    return f"Successfully snapped {len(snapped_path)} points across the 3D network for {Path(graph_path).name}. Output: {output_path}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deterministic quantum-Viterbi map matching.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for quantum weight initialization.")
    parser.add_argument("--graph", type=str, default=None, help="GraphML filename for single-case run.")
    parser.add_argument("--trajectory", type=str, default=None, help="Trajectory CSV filename for single-case run.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_max_workers(),
        help="Maximum parallel workers to use (default: all CPU cores).",
    )
    parser.add_argument("--candidate-radius", type=float, default=INFERENCE_CANDIDATE_RADIUS_M, help="Candidate edge search radius in meters.")
    parser.add_argument("--fallback-k", type=int, default=10, help="Fallback nearest-candidate count when radius search misses.")
    args = parser.parse_args()

    ensure_project_dirs()

    if (args.graph is None) ^ (args.trajectory is None):
        raise ValueError("Provide both --graph and --trajectory for a single-case run, or neither to run all hard cases.")

    if args.graph and args.trajectory:
        run_targets = [(args.graph, args.trajectory)]
    else:
        run_targets = [
            (f"step01_{case_name}_unified.graphml", f"step03_{case_name}_trajectories_v5.csv")
            for case_name in CASE_NAMES
        ]

    resolved_targets = []
    for idx, (graph_name, traj_name) in enumerate(run_targets):
        if args.graph and args.trajectory:
            graph_arg = Path(graph_name)
            traj_arg = Path(traj_name)
            graph_path = graph_arg if graph_arg.is_absolute() else Path(__file__).resolve().parent / graph_arg
            traj_path = traj_arg if traj_arg.is_absolute() else Path(__file__).resolve().parent / traj_arg
        else:
            case_name = graph_name.replace("_unified.graphml", "").removeprefix("step01_")
            graph_path = unified_graph_path(case_name)
            traj_path = trajectory_path(case_name)
        if not graph_path.exists() or not traj_path.exists():
            print(f"Skipping {graph_name}: missing graph or trajectory file.")
            continue

        resolved_targets.append((graph_path, traj_path, args.seed + idx))

    if resolved_targets:
        max_workers = max(1, min(int(args.max_workers), len(resolved_targets)))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    _run_single_case,
                    graph_path,
                    traj_path,
                    seed,
                    args.candidate_radius,
                    args.fallback_k,
                )
                for (graph_path, traj_path, seed) in resolved_targets
            ]
            for fut in as_completed(futures):
                print(fut.result())

    write_step04_consolidated_outputs(CASE_NAMES)
    write_step04_visual_consolidated_outputs(CASE_NAMES)
