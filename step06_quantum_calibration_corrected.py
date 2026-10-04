import pennylane as qml
from pennylane import numpy as pnp
import pandas as pd
import numpy as np
import networkx as nx
import os
import argparse
import json
import pickle
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import KDTree
from pipeline_config import DEFAULT_SEED, INFERENCE_CANDIDATE_RADIUS_M, QMM_NEGATIVE_SAMPLE_RADIUS_M, QTS_NEGATIVE_SAMPLE_RADIUS_M, REPORTS_DIR, classical_matched_qts_weights_path, classical_matched_weights_path, confidence_threshold_path, default_max_workers, deterministic_ansatz_init, ensure_project_dirs, make_qml_device, mlp_ablation_model_path, qts_weights_path, svm_ablation_model_path, split_calibration_and_holdout_trucks, trajectory_path, unified_graph_path, weights_path
from step05_quantum_classical_benchmarking import (
    MapMatchingBenchmarker,
    QMM_READOUT_WIRES,
    QTS_READOUT_WIRES,
    CLASSICAL_MATCHED_EMISSION_HIDDEN,
    CLASSICAL_MATCHED_TRANSITION_HIDDEN,
    bounded_classical_matched_score,
    build_edge_geometry_from_graph,
    classical_matched_weight_count,
    edge_length_weight,
    extract_9_features,
    transition_features,
)
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

base_dir = Path(__file__).resolve().parent
ensure_project_dirs()


def write_step06_calibration_consolidated_outputs(case_names):
    frames = []
    for case_name in case_names:
        source = REPORTS_DIR / f"step06_{case_name}_training_loss.csv"
        if not source.exists():
            continue
        df = pd.read_csv(source)
        if df.empty:
            continue
        df = df.copy()
        df["case_study"] = case_name
        df["source_file"] = str(source)
        df["source_filename"] = source.name
        frames.append(df)

    out_csv = REPORTS_DIR / "step06_training_loss_all_cases.csv"
    if frames:
        pd.concat(frames, axis=0, ignore_index=True).to_csv(out_csv, index=False)
    else:
        pd.DataFrame(columns=["epoch", "loss", "case_study", "source_file", "source_filename"]).to_csv(out_csv, index=False)
    print(f"Saved consolidated Step06 training loss: {out_csv}")


def to_float_safely(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def node_id_string(value):
    """Normalize CSV node IDs to the string representation used by GraphML."""
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def edge_key_int(value, default=0):
    """Normalize a trajectory/GraphML multiedge key without changing its ID."""
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return int(number)
    except (TypeError, ValueError):
        pass
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None if default is None else int(default)


def trajectory_edge_id(sample, road_graph):
    """Return the exact directed ``(u, v, key)`` recorded by a trajectory.

    Parallel MultiDiGraph records are distinct states.  In particular, a
    missing key must not silently resolve to the first parallel edge, because
    that changes both the positive's features and which edge is considered a
    negative.  A DiGraph has one edge per ordered node pair and is represented
    canonically with key 0.
    """
    u_raw = sample.get("edge_u", sample.get("true_node_u", None))
    v_raw = sample.get("edge_v", sample.get("true_node_v", None))
    if u_raw is None or v_raw is None or pd.isna(u_raw) or pd.isna(v_raw):
        return None
    u, v = node_id_string(u_raw), node_id_string(v_raw)
    if road_graph.is_multigraph():
        raw_key = sample.get("edge_key", None)
        if raw_key is None or pd.isna(raw_key):
            return None
        key = edge_key_int(raw_key, None)
        if key is None:
            return None
        if road_graph.has_edge(u, v, key):
            return (u, v, key)
        # Some NetworkX loaders retain a numeric-looking key as a string.
        for stored_key in (road_graph.get_edge_data(u, v) or {}):
            if edge_key_int(stored_key, -1) == key:
                return (u, v, edge_key_int(stored_key, key))
        return None
    if road_graph.has_edge(u, v):
        return (u, v, 0)
    return None


def catalog_edge_id(item):
    """Canonical edge identity for one row returned by :func:`edge_catalog`."""
    return (str(item[0]), str(item[1]), edge_key_int(item[2], 0))


def _empty_candidate_audit(unit_name="fixes"):
    return {
        f"input_{unit_name}": 0,
        "valid_inference_observations": 0,
        "resolved_true_edges": 0,
        "candidate_oracle_hits": 0,
        "candidate_oracle_denominator": 0,
        "candidate_oracle_recall": 0.0,
        "retained_examples": 0,
        "ranking_pairs": 0,
        "skipped_missing_inference_observation": 0,
        "skipped_unresolved_true_edge": 0,
        "skipped_true_not_in_live_candidates": 0,
        "skipped_no_live_negative": 0,
    }


def _finish_candidate_audit(audit):
    denominator = int(audit.get("candidate_oracle_denominator", 0))
    hits = int(audit.get("candidate_oracle_hits", 0))
    audit["candidate_oracle_recall"] = float(hits / denominator) if denominator else 0.0
    return audit


def pick_xy_columns(df_like):
    x_col = next((c for c in ["obs_x", "x_noisy", "x"] if c in df_like.columns), None)
    y_col = next((c for c in ["obs_y", "y_noisy", "y"] if c in df_like.columns), None)
    if x_col is None or y_col is None:
        raise ValueError("Could not find XY columns. Expected obs_x/obs_y, x_noisy/y_noisy, or x/y.")
    return x_col, y_col


def bool_flag(v):
    if v is None:
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(v) and float(v) != 0.0)
    text = str(v).strip().lower()
    if text in {"true", "yes", "y", "on"}:
        return True
    if text in {"", "false", "no", "n", "off", "none", "nan", "null"}:
        return False
    try:
        number = float(text)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number != 0.0)


def add_causal_effective_observations(rows):
    """Mirror Step05's causal missing-fix semantics for calibration inputs."""
    result = rows.copy()
    x_col, y_col = pick_xy_columns(result)
    result["_inference_x"] = np.nan
    result["_inference_y"] = np.nan
    group_columns = ["truck_id"] if "truck_id" in result.columns else []
    grouped = result.groupby(group_columns, sort=False) if group_columns else [(None, result)]
    for _group, track in grouped:
        if "timestamp" in track.columns:
            track = track.sort_values("timestamp")
        history = []
        for index, row in track.iterrows():
            x = to_float_safely(row.get(x_col), np.nan)
            y = to_float_safely(row.get(y_col), np.nan)
            if not (np.isfinite(x) and np.isfinite(y)) and history:
                prev = history[-1]
                vx = vy = 0.0
                if len(history) >= 2:
                    older = history[-2]
                    dt_prev = float(np.clip(prev[2] - older[2], 0.25, 30.0))
                    vx = (prev[0] - older[0]) / dt_prev
                    vy = (prev[1] - older[1]) / dt_prev
                curr_t = to_float_safely(row.get("timestamp"), prev[2] + 1.0)
                dt_curr = float(np.clip(curr_t - prev[2], 0.25, 30.0))
                x = prev[0] + vx * dt_curr if not np.isfinite(x) else x
                y = prev[1] + vy * dt_curr if not np.isfinite(y) else y
            curr_t = to_float_safely(row.get("timestamp"), float(len(history)))
            result.at[index, "_inference_x"] = x
            result.at[index, "_inference_y"] = y
            if np.isfinite(x) and np.isfinite(y):
                history.append((float(x), float(y), float(curr_t)))
    return result


def edge_catalog(road_graph):
    """Return directed edges, attributes, metric midpoint coordinates, and
    full line geometry (WKT centerline when present, else the straight
    endpoint segment) for every edge. The geometry entry (index 6) keeps
    negative sampling geometrically consistent with inference-time
    candidate generation."""
    iterator = (
        road_graph.edges(keys=True, data=True)
        if road_graph.is_multigraph()
        else ((u, v, 0, data) for u, v, data in road_graph.edges(data=True))
    )
    rows = []
    for u, v, key, data in iterator:
        ux = to_float_safely(road_graph.nodes[u].get("x"))
        uy = to_float_safely(road_graph.nodes[u].get("y"))
        vx = to_float_safely(road_graph.nodes[v].get("x"))
        vy = to_float_safely(road_graph.nodes[v].get("y"))
        edge = (str(u), str(v), edge_key_int(key, 0))
        geometry = build_edge_geometry_from_graph(road_graph, edge, data)
        rows.append((str(u), str(v), edge[2], data, (ux + vx) / 2.0, (uy + vy) / 2.0, geometry))
    if not rows:
        raise RuntimeError("Calibration graph has no edges.")
    return rows


def build_catalog_strtree(catalog):
    """STRtree over each catalog row's true edge geometry (index 6), for
    geometry-aware negative sampling. Returns
    None if Shapely/geometries are unavailable, so callers fall back to a
    legacy midpoint KD-tree."""
    if STRtree is None:
        return None
    geometries = [row[6] for row in catalog]
    if any(g is None for g in geometries):
        return None
    return STRtree(geometries), geometries


def catalog_items_within_radius(
    catalog, strtree_bundle, x, y, radius, max_items=None, fallback_items=None
):
    """True-geometry radius search over a calibration edge catalog, with a
    nearest-by-true-distance fallback (not arbitrary catalog order) when the
    radius search is empty. Mirrors
    MapMatchingBenchmarker._get_candidates_by_geometry so calibration-time
    negative sampling and inference-time candidate generation stay
    geometrically consistent."""
    if strtree_bundle is not None and Point is not None:
        strtree, geometries = strtree_bundle
        point = Point(float(x), float(y))
        bbox_idxs = strtree.query(point.buffer(radius))
        matches = [int(i) for i in bbox_idxs if geometries[int(i)].distance(point) <= radius]
        if matches:
            if max_items is not None and len(matches) > max_items:
                matches.sort(key=lambda i: geometries[i].distance(point))
                matches = matches[:max_items]
            return [catalog[i] for i in matches]

        k = fallback_items or max_items or 1
        expand_radius = max(radius, 1.0)
        bbox_idxs = np.asarray([], dtype=int)
        for _ in range(10):
            expand_radius *= 2.0
            bbox_idxs = strtree.query(point.buffer(expand_radius))
            if len(bbox_idxs) >= k or len(bbox_idxs) >= len(catalog):
                break
        if len(bbox_idxs) == 0:
            return []
        distances = sorted((geometries[int(i)].distance(point), int(i)) for i in bbox_idxs)
        return [catalog[i] for _, i in distances[:k]]

    # Legacy midpoint fallback (only reachable if Shapely is unavailable).
    coords = np.asarray([[item[4], item[5]] for item in catalog], dtype=float)
    tree = KDTree(coords)
    idxs = tree.query_ball_point([x, y], radius)
    if idxs:
        return [catalog[i] for i in idxs]
    k = fallback_items or max_items or 1
    _, nearest = tree.query([x, y], k=min(k, len(catalog)))
    nearest = np.atleast_1d(nearest)
    return [catalog[int(i)] for i in nearest]


def true_edge_data(sample, road_graph):
    """Resolve the trajectory's actual directed edge; never infer it from noisy XY."""
    edge = trajectory_edge_id(sample, road_graph)
    if edge is None:
        raw = (
            sample.get("edge_u", sample.get("true_node_u", "")),
            sample.get("edge_v", sample.get("true_node_v", "")),
            sample.get("edge_key", 0),
        )
        raise KeyError(f"Exact true edge {raw!r} is absent from the calibration graph.")
    u, v, key = edge
    return road_graph.edges[u, v, key] if road_graph.is_multigraph() else road_graph.edges[u, v]


def get_normalized_features_for_true_edge(sample, road_graph):
    traffic_val = to_float_safely(sample.get("time_qubit_val", 0.5), 0.5)
    return extract_9_features(true_edge_data(sample, road_graph), traffic_regime=traffic_val)


def build_edge_examples(rows, road_graph, catalog):
    """Create one true-edge and one nearby incorrect-edge example per fix."""
    examples = []
    for _, sample in rows.iterrows():
        traffic = to_float_safely(sample.get("time_qubit_val", 0.5), 0.5)
        true_edge = trajectory_edge_id(sample, road_graph)
        if true_edge is None:
            continue
        tx = to_float_safely(sample.get("true_x"), np.nan)
        ty = to_float_safely(sample.get("true_y"), np.nan)
        true_features = extract_9_features(true_edge_data(sample, road_graph), traffic_regime=traffic)
        examples.append((true_features, 1.0))
        alternatives = [
            item for item in catalog
            if catalog_edge_id(item) != true_edge
        ]
        if not alternatives:
            continue
        negative = min(alternatives, key=lambda item: (item[4] - tx) ** 2 + (item[5] - ty) ** 2)
        examples.append((extract_9_features(negative[3], traffic_regime=traffic), 0.0))
    return examples


def build_ranking_pairs(
    rows,
    road_graph,
    catalog,
    radius=QMM_NEGATIVE_SAMPLE_RADIUS_M,
    max_negatives=4,
    return_audit=False,
):
    """Pair the true edge's features against up to ``max_negatives`` nearby
    candidate edges' features per fix. Negatives are drawn by geometric
    distance from the noisy/causally-imputed fix to each edge's full geometry
    (WKT centerline when present, else its endpoint segment), via the same search used at inference time
    (MapMatchingBenchmarker._get_candidates_by_geometry), not a coarse
    per-edge midpoint, falling back to nearest-K
    by geometric distance if the radius search is empty. Used for a pairwise
    ranking loss instead of binary classification: the actual downstream
    task (Viterbi candidate selection) is "rank the true edge above every
    plausible alternative in its candidate set", which a true/negative
    classification pair only approximates.
    """
    strtree_bundle = build_catalog_strtree(catalog)
    true_rows, negative_rows = [], []
    audit = _empty_candidate_audit("fixes")
    for _, sample in rows.iterrows():
        audit["input_fixes"] += 1
        traffic = to_float_safely(sample.get("time_qubit_val", 0.5), 0.5)
        # Candidate sets at deployment are constructed from the noisy fix or
        # its causal blackout prediction, never from the simulator's latent
        # true position.  Draw ranking negatives from that same live set.
        candidate_x = to_float_safely(sample.get("_inference_x"), np.nan)
        candidate_y = to_float_safely(sample.get("_inference_y"), np.nan)
        if not (np.isfinite(candidate_x) and np.isfinite(candidate_y)):
            audit["skipped_missing_inference_observation"] += 1
            continue
        audit["valid_inference_observations"] += 1
        true_edge = trajectory_edge_id(sample, road_graph)
        if true_edge is None:
            audit["skipped_unresolved_true_edge"] += 1
            continue
        audit["resolved_true_edges"] += 1
        audit["candidate_oracle_denominator"] += 1

        nearby = catalog_items_within_radius(
            catalog,
            strtree_bundle,
            candidate_x,
            candidate_y,
            radius,
            fallback_items=10,
        )

        nearby_by_id = {catalog_edge_id(item): item for item in nearby}
        if true_edge not in nearby_by_id:
            audit["skipped_true_not_in_live_candidates"] += 1
            continue
        audit["candidate_oracle_hits"] += 1
        true_item = nearby_by_id[true_edge]
        true_features = extract_9_features(true_item[3], traffic_regime=traffic)

        if Point is not None and true_item[6] is not None:
            point = Point(float(candidate_x), float(candidate_y))
            nearby = sorted(
                nearby,
                key=lambda item: (float(item[6].distance(point)), catalog_edge_id(item)),
            )
        else:
            nearby = sorted(
                nearby,
                key=lambda item: (
                    (item[4] - candidate_x) ** 2 + (item[5] - candidate_y) ** 2,
                    catalog_edge_id(item),
                ),
            )
        negatives = [
            extract_9_features(item[3], traffic_regime=traffic)
            for item in nearby
            if catalog_edge_id(item) != true_edge
        ][:max_negatives]
        if not negatives:
            audit["skipped_no_live_negative"] += 1
            continue

        audit["retained_examples"] += 1
        for neg_features in negatives:
            true_rows.append(true_features)
            negative_rows.append(neg_features)
            audit["ranking_pairs"] += 1

    audit = _finish_candidate_audit(audit)
    if return_audit:
        return true_rows, negative_rows, audit
    return true_rows, negative_rows

def get_actual_road_coords(row, G_graph):
    """
    Retrieves the actual metric coordinates from the GraphML.
    Prioritizes 'true_node' column from trajectory generation.
    """
    # If trajectory includes explicit true node ids, use those first.
    for node_col in ["true_node_u", "true_node"]:
        if node_col in row.index and pd.notna(row[node_col]):
            u = str(row[node_col])
            if u in G_graph.nodes:
                return float(G_graph.nodes[u]["x"]), float(G_graph.nodes[u]["y"])
    
    # Fallback: Use the 'true_x/true_y' columns if they exist
    if "true_x" in row.index and "true_y" in row.index and pd.notna(row["true_x"]) and pd.notna(row["true_y"]):
        return float(row["true_x"]), float(row["true_y"])

    # v5 fallback: project observation to nearest graph node for a stable reference.
    x_col, y_col = pick_xy_columns(row.to_frame().T)
    ox = to_float_safely(row[x_col])
    oy = to_float_safely(row[y_col])
    best_node = None
    best_d2 = float("inf")
    for n, data in G_graph.nodes(data=True):
        nx_ = to_float_safely(data.get("x"))
        ny_ = to_float_safely(data.get("y"))
        d2 = (nx_ - ox) ** 2 + (ny_ - oy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_node = n

    if best_node is None:
        return None, None
    return float(G_graph.nodes[best_node]["x"]), float(G_graph.nodes[best_node]["y"])

# --- 2. QUANTUM ENGINE (9-QUBIT REGISTER) ---
# Circuit structure (data re-uploading + local multi-qubit readout) must stay
# identical to QuantumEmissionModel in step05, since the weight tensor
# trained here is loaded directly into that class at inference time.
n_qubits = 9
dev = make_qml_device(qml, wires=n_qubits, prefer_gpu=True, gpu_fraction=0.8)


def _local_readout(wires):
    obs = qml.PauliZ(wires[0])
    for w in wires[1:]:
        obs = obs + qml.PauliZ(w)
    return obs / float(len(wires))


_qmm_readout = _local_readout(QMM_READOUT_WIRES)


@qml.qnode(dev, diff_method="adjoint")
def quantum_scoring_circuit(features, weights):
    qml.AngleEmbedding(pnp.pi * features, wires=range(n_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights[0:1], wires=range(n_qubits))
    qml.AngleEmbedding(pnp.pi * features, wires=range(n_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights[1:2], wires=range(n_qubits))
    return qml.expval(_qmm_readout)


def bounded_quantum_score(features, weights):
    """Use the identical Pauli-Z-to-[0,1] mapping in training and inference."""
    return (1.0 + quantum_scoring_circuit(features, weights)) / 2.0


# --- QTS ENGINE (4-QUBIT REGISTER) ---
qts_n_qubits = 4
qts_dev = make_qml_device(qml, wires=qts_n_qubits, prefer_gpu=True, gpu_fraction=0.8)
_qts_readout = _local_readout(QTS_READOUT_WIRES)


@qml.qnode(qts_dev, diff_method="adjoint")
def qts_scoring_circuit(features, weights):
    qml.AngleEmbedding(pnp.pi * features, wires=range(qts_n_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights[0:1], wires=range(qts_n_qubits))
    qml.AngleEmbedding(pnp.pi * features, wires=range(qts_n_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights[1:2], wires=range(qts_n_qubits))
    return qml.expval(_qts_readout)


def bounded_qts_score(features, weights):
    return (1.0 + qts_scoring_circuit(features, weights)) / 2.0

# --- 3. TRAINING (CALIBRATED TARGETS) ---
def target_from_sample(sample):
    sat = to_float_safely(sample.get("sat_count"), 0.0)
    hdop = to_float_safely(sample.get("hdop"), 99.0)
    is_tunnel = bool_flag(sample.get("true_tunnel", 0))
    quality_state = str(sample.get("quality_state", "")).lower()

    if sat >= 8 and hdop <= 1.5 and not is_tunnel:
        return 1.0
    if is_tunnel or "blackout" in quality_state:
        return 0.0
    if "degraded" in quality_state:
        return 0.35
    if "moderate" in quality_state or "recovery" in quality_state:
        return 0.55
    return 0.75


def ambiguity_label(sample):
    """Reference label used only to select the operating threshold."""
    quality_state = str(sample.get("quality_state", "")).lower()
    return int(bool_flag(sample.get("true_tunnel", 0)) or "blackout" in quality_state)


def select_balanced_accuracy_threshold(scores, labels):
    """Select score <= threshold as ambiguous using held-out balanced accuracy."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.size == 0 or np.unique(labels).size < 2:
        raise RuntimeError("Threshold calibration requires both ambiguous and nominal validation samples.")
    unique_scores = np.unique(scores)
    candidates = np.concatenate((
        [np.nextafter(unique_scores[0], -np.inf)],
        (unique_scores[:-1] + unique_scores[1:]) / 2.0,
        [unique_scores[-1]],
    ))
    rows = []
    for threshold in candidates:
        pred = (scores <= threshold).astype(int)
        tp = int(np.sum((pred == 1) & (labels == 1)))
        tn = int(np.sum((pred == 0) & (labels == 0)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        fn = int(np.sum((pred == 0) & (labels == 1)))
        sensitivity = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        balanced_accuracy = 0.5 * (sensitivity + specificity)
        rows.append({
            "threshold": float(threshold),
            "balanced_accuracy": float(balanced_accuracy),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })
    # Prefer the more conservative (lower) threshold when scores tie.
    selected = max(rows, key=lambda row: (row["balanced_accuracy"], -row["threshold"]))
    return selected, rows


def train_calibrated_model(
    case_name,
    trajectory_df,
    road_graph,
    seed=DEFAULT_SEED,
    restarts=5,
    validation_fraction=0.2,
):
    training_data = add_causal_effective_observations(trajectory_df.copy())
    if training_data.empty:
        raise RuntimeError(f"No suitable calibration samples found for {case_name}.")

    if "truck_id" not in training_data.columns:
        raise ValueError("Trajectory-level calibration splitting requires truck_id.")
    split_rng = np.random.default_rng(int(seed) + 100003)
    trucks = np.asarray(sorted(training_data["truck_id"].astype(str).unique()))
    split_rng.shuffle(trucks)
    validation_truck_count = max(1, int(round(len(trucks) * float(validation_fraction))))
    validation_truck_count = min(validation_truck_count, max(1, len(trucks) - 1))
    validation_trucks = set(trucks[:validation_truck_count])
    validation_data = training_data[training_data["truck_id"].astype(str).isin(validation_trucks)].copy()
    fit_data = training_data[~training_data["truck_id"].astype(str).isin(validation_trucks)].copy()
    # Bound circuit-evaluation cost while preserving strict truck separation.
    # Sampling is deterministic and occurs only after the group split.
    fit_data = fit_data.sample(n=min(320, len(fit_data)), random_state=int(seed) + 200003)
    validation_data = validation_data.sample(
        n=min(160, len(validation_data)), random_state=int(seed) + 300007
    )

    catalog = edge_catalog(road_graph)
    RANK_MARGIN = 0.2
    fit_true, fit_neg, fit_candidate_audit = build_ranking_pairs(
        fit_data, road_graph, catalog, return_audit=True
    )
    validation_true, validation_neg, validation_candidate_audit = build_ranking_pairs(
        validation_data, road_graph, catalog, return_audit=True
    )
    if not fit_true or not validation_true:
        raise RuntimeError(f"Could not construct ranking pairs for {case_name}.")
    fit_true_arr = np.asarray(fit_true, dtype=float)
    fit_neg_arr = np.asarray(fit_neg, dtype=float)
    validation_true_arr = np.asarray(validation_true, dtype=float)
    validation_neg_arr = np.asarray(validation_neg, dtype=float)
    # A constant-output scorer gives every candidate the same score, so
    # true_score - neg_score = 0 for every pair and the hinge loss reduces
    # exactly to the margin: this is the ranking-loss analogue of the old
    # constant-0.5-baseline MSE used as a training-usefulness gate.
    baseline_validation_loss = float(RANK_MARGIN)
    print(
        f"Calibrating {case_name} with {len(fit_data)} fit points/{len(fit_true_arr)} ranking pairs and "
        f"{len(validation_data)} validation points/{len(validation_true_arr)} ranking pairs "
        f"across {int(restarts)} restarts; constant baseline hinge loss={baseline_validation_loss:.6f}..."
    )
    print(
        f"  candidate oracle: fit {fit_candidate_audit['candidate_oracle_hits']}/"
        f"{fit_candidate_audit['candidate_oracle_denominator']} "
        f"({fit_candidate_audit['candidate_oracle_recall']:.1%}); validation "
        f"{validation_candidate_audit['candidate_oracle_hits']}/"
        f"{validation_candidate_audit['candidate_oracle_denominator']} "
        f"({validation_candidate_audit['candidate_oracle_recall']:.1%})"
    )

    best_weights = None
    best_validation_loss = float("inf")
    loss_history = []
    restart_summary = []
    for restart in range(max(1, int(restarts))):
        restart_seed = int(seed) + 1009 * restart
        w = pnp.array(
            deterministic_ansatz_init(
                qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits), restart=restart
            ),
            requires_grad=True,
        )
        opt = qml.AdamOptimizer(stepsize=0.01)
        batch_size = 32
        max_epochs = 12
        patience = 3
        min_delta = 1e-4
        restart_best_loss = float("inf")
        restart_best_epoch = -1
        restart_best_weights = None
        stale_epochs = 0
        restart_losses = []
        for epoch in range(max_epochs):
            epoch_rng = np.random.default_rng(restart_seed + 7919 * epoch)
            order = epoch_rng.permutation(len(fit_true_arr))
            batch_losses = []
            for start in range(0, len(order), batch_size):
                batch_idx = order[start:start + batch_size]
                batch_true = fit_true_arr[batch_idx]
                batch_neg = fit_neg_arr[batch_idx]

                def cost_fn(weights):
                    true_scores = bounded_quantum_score(pnp.array(batch_true), weights)
                    neg_scores = bounded_quantum_score(pnp.array(batch_neg), weights)
                    return pnp.mean(pnp.clip(RANK_MARGIN - (true_scores - neg_scores), 0, None))

                w, cost = opt.step_and_cost(cost_fn, w)
                batch_losses.append(float(cost))

            train_true_scores = np.asarray(bounded_quantum_score(pnp.array(fit_true_arr), w), dtype=float)
            train_neg_scores = np.asarray(bounded_quantum_score(pnp.array(fit_neg_arr), w), dtype=float)
            validation_true_scores = np.asarray(
                bounded_quantum_score(pnp.array(validation_true_arr), w), dtype=float
            )
            validation_neg_scores = np.asarray(
                bounded_quantum_score(pnp.array(validation_neg_arr), w), dtype=float
            )
            train_loss = float(np.mean(np.clip(RANK_MARGIN - (train_true_scores - train_neg_scores), 0, None)))
            validation_loss = float(
                np.mean(np.clip(RANK_MARGIN - (validation_true_scores - validation_neg_scores), 0, None))
            )
            restart_losses.append(train_loss)
            loss_history.append({
                "restart": restart, "restart_seed": restart_seed, "epoch": epoch,
                "loss": train_loss, "validation_loss": validation_loss,
            })
            if validation_loss < restart_best_loss - min_delta:
                restart_best_loss = validation_loss
                restart_best_epoch = epoch
                restart_best_weights = pnp.array(np.asarray(w, dtype=float), requires_grad=False)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        validation_loss = restart_best_loss
        restart_summary.append(
            {
                "restart": restart,
                "restart_seed": restart_seed,
                "final_train_loss": restart_losses[-1],
                "validation_loss": validation_loss,
                "selected_epoch": restart_best_epoch,
                "epochs_completed": len(restart_losses),
                "constant_baseline_hinge_loss": baseline_validation_loss,
                "beats_constant_baseline": bool(validation_loss < baseline_validation_loss),
            }
        )
        print(
            f"[{case_name}] restart {restart + 1}/{int(restarts)}: "
            f"validation ranking loss={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_weights = restart_best_weights

    loss_csv = REPORTS_DIR / f"step06_{case_name}_training_loss.csv"
    pd.DataFrame(loss_history).to_csv(loss_csv, index=False)
    restart_csv = REPORTS_DIR / f"step06_{case_name}_restart_selection.csv"
    pd.DataFrame(restart_summary).to_csv(restart_csv, index=False)
    selected = min(restart_summary, key=lambda row: row["validation_loss"])
    if not selected["beats_constant_baseline"]:
        raise RuntimeError(
            f"{case_name} calibration rejected: best validation ranking loss "
            f"{selected['validation_loss']:.6f} does not beat constant baseline "
            f"{baseline_validation_loss:.6f}."
        )
    # Inference thresholds max_e QMM(e) over the live candidate set.  Calibrate
    # that same random variable, using noisy/causally-imputed coordinates,
    # rather than the score of the known true edge (which inference does not
    # know and which has a systematically different distribution).
    catalog_strtree = build_catalog_strtree(catalog)
    validation_scores = []
    validation_labels = []
    for _, row in validation_data.iterrows():
        x = to_float_safely(row.get("_inference_x"), np.nan)
        y = to_float_safely(row.get("_inference_y"), np.nan)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        candidates = catalog_items_within_radius(
            catalog,
            catalog_strtree,
            x,
            y,
            INFERENCE_CANDIDATE_RADIUS_M,
            fallback_items=10,
        )
        traffic = to_float_safely(row.get("time_qubit_val", 0.5), 0.5)
        candidate_scores = [
            float(quantum_scoring_circuit(
                pnp.array(extract_9_features(item[3], traffic_regime=traffic)),
                best_weights,
            ))
            for item in candidates
        ]
        if candidate_scores:
            validation_scores.append(max(candidate_scores))
            validation_labels.append(ambiguity_label(row))
    selected_threshold, threshold_rows = select_balanced_accuracy_threshold(
        validation_scores, validation_labels
    )
    threshold_curve_path = REPORTS_DIR / f"step06_{case_name}_threshold_selection.csv"
    pd.DataFrame(threshold_rows).to_csv(threshold_curve_path, index=False)
    # Balanced accuracy == 0.5 is the degenerate "always predict one class"
    # operating point: no candidate threshold separated ambiguous from
    # nominal validation samples better than chance. This is not caught by
    # the constant-baseline ranking-loss gate above (a training-usefulness
    # check), so flag it explicitly rather than silently shipping a
    # non-functional detector.
    is_chance_level = bool(selected_threshold["balanced_accuracy"] <= 0.5 + 1e-9)
    if is_chance_level:
        print(
            f"  WARNING: {case_name} confidence-dip threshold is chance-level "
            f"(balanced_accuracy={selected_threshold['balanced_accuracy']:.3f}); "
            "the calibrated QMM score does not separate ambiguous from nominal "
            "validation samples for this case."
        )
    operating_threshold = (
        -1.000001 if is_chance_level else float(selected_threshold["threshold"])
    )
    threshold_out = confidence_threshold_path(case_name)
    with open(threshold_out, "w", encoding="utf-8") as handle:
        json.dump({
            "case": case_name,
            "threshold": operating_threshold,
            "validation_selected_threshold": float(selected_threshold["threshold"]),
            "detector_enabled": not is_chance_level,
            "selection_split": "held-out Step06 calibration validation split",
            "selection_metric": "balanced accuracy",
            "ambiguity_rule": "true_tunnel == 1 or quality_state contains blackout",
            "score_semantics": "maximum QMM expectation over inference-style candidates at noisy/causally-imputed XY",
            "chance_level_detector": is_chance_level,
            **{k: v for k, v in selected_threshold.items() if k != "threshold"},
        }, handle, indent=2)

    metadata_path = REPORTS_DIR / f"step06_{case_name}_selection_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "case": case_name,
                "selection_rule": "minimum held-out validation pairwise ranking (hinge) loss",
                "loss_type": "margin ranking loss over local candidate sets",
                "ranking_margin": float(RANK_MARGIN),
                "base_seed": int(seed),
                "restarts": int(restarts),
                "validation_fraction": float(validation_fraction),
                "fit_points": int(len(fit_data)),
                "validation_points": int(len(validation_data)),
                "fit_trucks": int(fit_data["truck_id"].nunique()),
                "validation_trucks": int(validation_data["truck_id"].nunique()),
                "fit_ranking_pairs": int(len(fit_true_arr)),
                "validation_ranking_pairs": int(len(validation_true_arr)),
                "fit_candidate_audit": fit_candidate_audit,
                "validation_candidate_audit": validation_candidate_audit,
                "constant_baseline_validation_loss": baseline_validation_loss,
                "selected_restart": int(selected["restart"]),
                "selected_restart_seed": int(selected["restart_seed"]),
                "selected_epoch": int(selected["selected_epoch"]),
                "selected_validation_loss": float(selected["validation_loss"]),
                "beats_constant_baseline": bool(selected["beats_constant_baseline"]),
                "selected_confidence_threshold": float(operating_threshold),
                "validation_selected_confidence_threshold": float(selected_threshold["threshold"]),
                "threshold_validation_balanced_accuracy": float(selected_threshold["balanced_accuracy"]),
            },
            handle,
            indent=2,
        )
    print(f"Saved training diagnostics: {loss_csv}")
    print(f"Saved restart selection diagnostics: {restart_csv}")
    print(
        f"[{case_name}] selected confidence threshold="
        f"{operating_threshold:.6f} "
        f"(balanced accuracy={selected_threshold['balanced_accuracy']:.4f})"
    )
    return best_weights, operating_threshold


def train_calibrated_classical_model(
    case_name,
    trajectory_df,
    road_graph,
    seed=DEFAULT_SEED,
    restarts=5,
    validation_fraction=0.2,
):
    """Matched classical control for the QMM:
    identical truck-disjoint fit/validation split, identical 320/160-point
    budget, identical pairwise margin-ranking loss and Adam training loop
    (same restarts/batch size/epochs/patience) as train_calibrated_model —
    the only difference is a classical MLP forward pass in place of the
    quantum circuit, so a resulting RMSE difference can be attributed to
    "quantum vs. classical", not to unequal data, loss, or training budget.
    """
    # Match the QMM's inference-time candidate-set semantics.  Without this
    # transformation the matched classical control would draw negatives around
    # different coordinates from both the QMM and the deployed decoder.
    training_data = add_causal_effective_observations(trajectory_df.copy())
    if training_data.empty:
        raise RuntimeError(f"No suitable calibration samples found for {case_name}.")
    if "truck_id" not in training_data.columns:
        raise ValueError("Trajectory-level calibration splitting requires truck_id.")

    # Same split RNG stream as the QMM (same seed offsets), so the matched
    # classical control sees the exact same fit/validation trucks and points.
    split_rng = np.random.default_rng(int(seed) + 100003)
    trucks = np.asarray(sorted(training_data["truck_id"].astype(str).unique()))
    split_rng.shuffle(trucks)
    validation_truck_count = max(1, int(round(len(trucks) * float(validation_fraction))))
    validation_truck_count = min(validation_truck_count, max(1, len(trucks) - 1))
    validation_trucks = set(trucks[:validation_truck_count])
    validation_data = training_data[training_data["truck_id"].astype(str).isin(validation_trucks)].copy()
    fit_data = training_data[~training_data["truck_id"].astype(str).isin(validation_trucks)].copy()
    fit_data = fit_data.sample(n=min(320, len(fit_data)), random_state=int(seed) + 200003)
    validation_data = validation_data.sample(
        n=min(160, len(validation_data)), random_state=int(seed) + 300007
    )

    catalog = edge_catalog(road_graph)
    RANK_MARGIN = 0.2
    fit_true, fit_neg, fit_candidate_audit = build_ranking_pairs(
        fit_data, road_graph, catalog, return_audit=True
    )
    validation_true, validation_neg, validation_candidate_audit = build_ranking_pairs(
        validation_data, road_graph, catalog, return_audit=True
    )
    if not fit_true or not validation_true:
        raise RuntimeError(f"Could not construct ranking pairs for {case_name}.")
    fit_true_arr = np.asarray(fit_true, dtype=float)
    fit_neg_arr = np.asarray(fit_neg, dtype=float)
    validation_true_arr = np.asarray(validation_true, dtype=float)
    validation_neg_arr = np.asarray(validation_neg, dtype=float)
    baseline_validation_loss = float(RANK_MARGIN)
    n_inputs = 9
    hidden_size = CLASSICAL_MATCHED_EMISSION_HIDDEN
    n_params = classical_matched_weight_count(n_inputs, hidden_size)
    print(
        f"Calibrating {case_name} classical-matched emission control with {len(fit_data)} fit points/"
        f"{len(fit_true_arr)} ranking pairs and {len(validation_data)} validation points/"
        f"{len(validation_true_arr)} ranking pairs across {int(restarts)} restarts "
        f"({n_params} parameters vs. QMM's 54); constant baseline hinge loss={baseline_validation_loss:.6f}..."
    )
    print(
        f"  candidate oracle: fit {fit_candidate_audit['candidate_oracle_hits']}/"
        f"{fit_candidate_audit['candidate_oracle_denominator']} "
        f"({fit_candidate_audit['candidate_oracle_recall']:.1%}); validation "
        f"{validation_candidate_audit['candidate_oracle_hits']}/"
        f"{validation_candidate_audit['candidate_oracle_denominator']} "
        f"({validation_candidate_audit['candidate_oracle_recall']:.1%})"
    )

    def scorer(features, weights):
        return bounded_classical_matched_score(features, weights, n_inputs, hidden_size)

    best_weights = None
    best_validation_loss = float("inf")
    loss_history = []
    restart_summary = []
    for restart in range(max(1, int(restarts))):
        restart_seed = int(seed) + 1009 * restart
        w = pnp.array(
            deterministic_ansatz_init((n_params,), restart=restart), requires_grad=True
        )
        opt = qml.AdamOptimizer(stepsize=0.01)
        batch_size = 32
        max_epochs = 12
        patience = 3
        min_delta = 1e-4
        restart_best_loss = float("inf")
        restart_best_epoch = -1
        restart_best_weights = None
        stale_epochs = 0
        restart_losses = []
        for epoch in range(max_epochs):
            epoch_rng = np.random.default_rng(restart_seed + 7919 * epoch)
            order = epoch_rng.permutation(len(fit_true_arr))
            for start in range(0, len(order), batch_size):
                batch_idx = order[start:start + batch_size]
                batch_true = fit_true_arr[batch_idx]
                batch_neg = fit_neg_arr[batch_idx]

                def cost_fn(weights):
                    true_scores = scorer(pnp.array(batch_true), weights)
                    neg_scores = scorer(pnp.array(batch_neg), weights)
                    return pnp.mean(pnp.clip(RANK_MARGIN - (true_scores - neg_scores), 0, None))

                w, cost = opt.step_and_cost(cost_fn, w)

            train_true_scores = np.asarray(scorer(pnp.array(fit_true_arr), w), dtype=float)
            train_neg_scores = np.asarray(scorer(pnp.array(fit_neg_arr), w), dtype=float)
            validation_true_scores = np.asarray(scorer(pnp.array(validation_true_arr), w), dtype=float)
            validation_neg_scores = np.asarray(scorer(pnp.array(validation_neg_arr), w), dtype=float)
            train_loss = float(np.mean(np.clip(RANK_MARGIN - (train_true_scores - train_neg_scores), 0, None)))
            validation_loss = float(
                np.mean(np.clip(RANK_MARGIN - (validation_true_scores - validation_neg_scores), 0, None))
            )
            restart_losses.append(train_loss)
            loss_history.append({
                "restart": restart, "restart_seed": restart_seed, "epoch": epoch,
                "loss": train_loss, "validation_loss": validation_loss,
            })
            if validation_loss < restart_best_loss - min_delta:
                restart_best_loss = validation_loss
                restart_best_epoch = epoch
                restart_best_weights = pnp.array(np.asarray(w, dtype=float), requires_grad=False)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        validation_loss = restart_best_loss
        restart_summary.append({
            "restart": restart, "restart_seed": restart_seed,
            "final_train_loss": restart_losses[-1], "validation_loss": validation_loss,
            "selected_epoch": restart_best_epoch, "epochs_completed": len(restart_losses),
            "constant_baseline_hinge_loss": baseline_validation_loss,
            "beats_constant_baseline": bool(validation_loss < baseline_validation_loss),
        })
        print(
            f"[{case_name}] classical-matched restart {restart + 1}/{int(restarts)}: "
            f"validation ranking loss={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_weights = restart_best_weights

    loss_csv = REPORTS_DIR / f"step06_{case_name}_classical_matched_training_loss.csv"
    pd.DataFrame(loss_history).to_csv(loss_csv, index=False)
    restart_csv = REPORTS_DIR / f"step06_{case_name}_classical_matched_restart_selection.csv"
    pd.DataFrame(restart_summary).to_csv(restart_csv, index=False)
    selected = min(restart_summary, key=lambda row: row["validation_loss"])
    if not selected["beats_constant_baseline"]:
        # Unlike the QMM/QTS gates above, a failing matched-classical control
        # is not treated as fatal: at the hardest, most nearly-degenerate
        # ranking site (M80), the QMM itself only narrowly beats this same
        # baseline (2/5 restarts), so the matched MLP failing here is a
        # finding about the site and the control. The best-found weights are
        # still saved, and beats_constant_baseline=False in the metadata
        # below records this for downstream reporting.
        print(
            f"WARNING: {case_name} classical-matched calibration did not beat constant "
            f"baseline (best validation ranking loss {selected['validation_loss']:.6f} vs. "
            f"{baseline_validation_loss:.6f}); saving best-found weights anyway with "
            "beats_constant_baseline=False."
        )
    metadata_path = REPORTS_DIR / f"step06_{case_name}_classical_matched_selection_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({
            "case": case_name,
            "control_type": "matched classical MLP",
            "n_parameters": int(n_params),
            "reference_quantum_parameters": 54,
            "selection_rule": "minimum held-out validation pairwise ranking (hinge) loss",
            "ranking_margin": float(RANK_MARGIN),
            "base_seed": int(seed),
            "restarts": int(restarts),
            "validation_fraction": float(validation_fraction),
            "fit_points": int(len(fit_data)),
            "validation_points": int(len(validation_data)),
            "fit_ranking_pairs": int(len(fit_true_arr)),
            "validation_ranking_pairs": int(len(validation_true_arr)),
            "fit_candidate_audit": fit_candidate_audit,
            "validation_candidate_audit": validation_candidate_audit,
            "selected_restart": int(selected["restart"]),
            "selected_epoch": int(selected["selected_epoch"]),
            "selected_validation_loss": float(selected["validation_loss"]),
            "constant_baseline_validation_loss": baseline_validation_loss,
            "beats_constant_baseline": bool(selected["beats_constant_baseline"]),
        }, handle, indent=2)
    print(f"Saved classical-matched training diagnostics: {loss_csv}")
    return best_weights


def train_calibrated_classical_qts_model(
    case_name,
    trajectory_df,
    road_graph,
    seed=DEFAULT_SEED,
    restarts=5,
    validation_fraction=0.2,
):
    """Matched classical control for the QTS, counterpart to
    train_calibrated_classical_model."""
    if "truck_id" not in trajectory_df.columns:
        raise ValueError("QTS calibration requires truck_id for train/validation truck separation.")
    split_rng = np.random.default_rng(int(seed) + 400009)
    trucks = np.asarray(sorted(trajectory_df["truck_id"].astype(str).unique()))
    split_rng.shuffle(trucks)
    validation_truck_count = max(1, int(round(len(trucks) * float(validation_fraction))))
    validation_truck_count = min(validation_truck_count, max(1, len(trucks) - 1))
    validation_trucks = trucks[:validation_truck_count]
    fit_trucks = trucks[validation_truck_count:]

    catalog = edge_catalog(road_graph)
    RANK_MARGIN = 0.2
    fit_true, fit_neg, fit_candidate_audit = build_transition_ranking_pairs(
        trajectory_df,
        road_graph,
        catalog,
        fit_trucks,
        cap=320,
        rng_seed=int(seed) + 500009,
        return_audit=True,
    )
    validation_true, validation_neg, validation_candidate_audit = build_transition_ranking_pairs(
        trajectory_df,
        road_graph,
        catalog,
        validation_trucks,
        cap=160,
        rng_seed=int(seed) + 600011,
        return_audit=True,
    )
    if not fit_true or not validation_true:
        raise RuntimeError(f"Could not construct QTS transition pairs for {case_name}.")
    fit_true_arr = np.asarray(fit_true, dtype=float)
    fit_neg_arr = np.asarray(fit_neg, dtype=float)
    validation_true_arr = np.asarray(validation_true, dtype=float)
    validation_neg_arr = np.asarray(validation_neg, dtype=float)
    baseline_validation_loss = float(RANK_MARGIN)
    n_inputs = 4
    hidden_size = CLASSICAL_MATCHED_TRANSITION_HIDDEN
    n_params = classical_matched_weight_count(n_inputs, hidden_size)
    print(
        f"Calibrating {case_name} classical-matched QTS control with {len(fit_true_arr)} fit / "
        f"{len(validation_true_arr)} validation transition pairs across {int(restarts)} restarts "
        f"({n_params} parameters vs. QTS's 24); constant baseline hinge loss={baseline_validation_loss:.6f}..."
    )
    print(
        f"  next-fix candidate oracle: fit "
        f"{fit_candidate_audit['next_candidate_oracle_hits']}/"
        f"{fit_candidate_audit['next_candidate_oracle_denominator']} "
        f"({fit_candidate_audit['next_candidate_oracle_recall']:.1%}); validation "
        f"{validation_candidate_audit['next_candidate_oracle_hits']}/"
        f"{validation_candidate_audit['next_candidate_oracle_denominator']} "
        f"({validation_candidate_audit['next_candidate_oracle_recall']:.1%})"
    )

    def scorer(features, weights):
        return bounded_classical_matched_score(features, weights, n_inputs, hidden_size)

    best_weights = None
    best_validation_loss = float("inf")
    restart_summary = []
    for restart in range(max(1, int(restarts))):
        restart_seed = int(seed) + 1013 * restart
        w = pnp.array(deterministic_ansatz_init((n_params,), restart=restart), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=0.01)
        batch_size = 32
        max_epochs = 12
        patience = 3
        min_delta = 1e-4
        restart_best_loss = float("inf")
        restart_best_epoch = -1
        restart_best_weights = None
        stale_epochs = 0
        restart_losses = []
        for epoch in range(max_epochs):
            epoch_rng = np.random.default_rng(restart_seed + 7919 * epoch)
            order = epoch_rng.permutation(len(fit_true_arr))
            for start in range(0, len(order), batch_size):
                batch_idx = order[start:start + batch_size]
                batch_true = fit_true_arr[batch_idx]
                batch_neg = fit_neg_arr[batch_idx]

                def cost_fn(weights):
                    true_scores = scorer(pnp.array(batch_true), weights)
                    neg_scores = scorer(pnp.array(batch_neg), weights)
                    return pnp.mean(pnp.clip(RANK_MARGIN - (true_scores - neg_scores), 0, None))

                w, cost = opt.step_and_cost(cost_fn, w)

            validation_true_scores = np.asarray(scorer(pnp.array(validation_true_arr), w), dtype=float)
            validation_neg_scores = np.asarray(scorer(pnp.array(validation_neg_arr), w), dtype=float)
            train_true_scores = np.asarray(scorer(pnp.array(fit_true_arr), w), dtype=float)
            train_neg_scores = np.asarray(scorer(pnp.array(fit_neg_arr), w), dtype=float)
            train_loss = float(np.mean(np.clip(RANK_MARGIN - (train_true_scores - train_neg_scores), 0, None)))
            validation_loss = float(
                np.mean(np.clip(RANK_MARGIN - (validation_true_scores - validation_neg_scores), 0, None))
            )
            restart_losses.append(train_loss)
            if validation_loss < restart_best_loss - min_delta:
                restart_best_loss = validation_loss
                restart_best_epoch = epoch
                restart_best_weights = pnp.array(np.asarray(w, dtype=float), requires_grad=False)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        validation_loss = restart_best_loss
        restart_summary.append({
            "restart": restart, "restart_seed": restart_seed,
            "validation_loss": validation_loss, "selected_epoch": restart_best_epoch,
            "beats_constant_baseline": bool(validation_loss < baseline_validation_loss),
        })
        print(
            f"[{case_name}] classical-matched QTS restart {restart + 1}/{int(restarts)}: "
            f"validation ranking loss={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_weights = restart_best_weights

    restart_csv = REPORTS_DIR / f"step06_{case_name}_classical_matched_qts_restart_selection.csv"
    pd.DataFrame(restart_summary).to_csv(restart_csv, index=False)
    selected = min(restart_summary, key=lambda row: row["validation_loss"])
    if not selected["beats_constant_baseline"]:
        raise RuntimeError(
            f"{case_name} classical-matched QTS calibration rejected: best validation ranking loss "
            f"{selected['validation_loss']:.6f} does not beat constant baseline "
            f"{baseline_validation_loss:.6f}."
        )
    metadata_path = REPORTS_DIR / f"step06_{case_name}_classical_matched_qts_selection_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "case": case_name,
                "control_type": "matched classical transition MLP",
                "n_parameters": int(n_params),
                "reference_quantum_parameters": 24,
                "selection_rule": "minimum held-out validation pairwise ranking (hinge) loss",
                "ranking_margin": float(RANK_MARGIN),
                "base_seed": int(seed),
                "restarts": int(restarts),
                "validation_fraction": float(validation_fraction),
                "fit_transition_pairs": int(len(fit_true_arr)),
                "validation_transition_pairs": int(len(validation_true_arr)),
                "fit_candidate_audit": fit_candidate_audit,
                "validation_candidate_audit": validation_candidate_audit,
                "selected_restart": int(selected["restart"]),
                "selected_epoch": int(selected["selected_epoch"]),
                "selected_validation_loss": float(selected["validation_loss"]),
                "constant_baseline_validation_loss": baseline_validation_loss,
                "beats_constant_baseline": bool(selected["beats_constant_baseline"]),
            },
            handle,
            indent=2,
        )
    print(f"Saved classical-matched QTS restart diagnostics: {restart_csv}")
    return best_weights


def build_transition_ranking_pairs(
    trajectory_df,
    road_graph,
    catalog,
    truck_ids,
    radius=QTS_NEGATIVE_SAMPLE_RADIUS_M,
    max_negatives=3,
    cap=320,
    rng_seed=0,
    return_audit=False,
):
    """Pair each true consecutive (e_i, e_j) trajectory transition against up
    to ``max_negatives`` false transitions (e_i, e_k) where e_k is a nearby
    candidate around the noisy/causally-imputed next-fix location that is not the actual next
    edge. "Nearby" is true distance to each edge's full geometry, matching
    inference-time candidate generation, not a coarse per-edge midpoint.
    Feature computation is identical to inference
    (transition_features + route distance measured from inference-available
    noisy/causally-imputed positions against each edge's
    own geometry via edge_length_weight for network hops), so the trained
    QTS sees the same feature semantics it will be scored on.

    Same-edge handling: e_i == e_j (same-edge continuation) is
    ~93-95% of all consecutive fixes, and its network hop from e_i's end
    node back to e_i's own start node is essentially always unreachable in
    a directed road graph, so that hop cannot be used as the training
    distance without discarding the dominant transition type. Same-edge
    transitions (for both the true label and any same-edge negative
    candidate) are scored as the along-edge arc-length between the two
    projected positions, mirroring the inference-time fix in
    step05._transition_prob_with_delta.
    """
    strtree_bundle = build_catalog_strtree(catalog)
    truck_id_set = set(str(t) for t in truck_ids)
    subset = trajectory_df[trajectory_df["truck_id"].astype(str).isin(truck_id_set)].copy()
    subset = add_causal_effective_observations(subset)
    time_col = "timestamp" if "timestamp" in subset.columns else None

    edge_geom_cache = {}

    def edge_geometry(edge, data):
        geometry = edge_geom_cache.get(edge)
        if geometry is None:
            geometry = build_edge_geometry_from_graph(road_graph, edge, data)
            edge_geom_cache[edge] = geometry
        return geometry

    def route_distance(edge_i, data_i, geom_i, target_edge, target_data, target_geom, prev_xy, curr_xy):
        """Route distance from prev_xy on ``edge_i`` to curr_xy on
        ``target_edge``, via each edge's own geometry rather than
        endpoint nodes. Returns None if the pair is graph-unreachable
        (cross-edge only; same-edge is always reachable by definition)."""
        same_edge = edge_i == target_edge
        if same_edge:
            if Point is None or geom_i is None or None in prev_xy or None in curr_xy:
                return 0.0
            prev_arc = float(geom_i.project(Point(float(prev_xy[0]), float(prev_xy[1]))))
            curr_arc = float(geom_i.project(Point(float(curr_xy[0]), float(curr_xy[1]))))
            return max(curr_arc - prev_arc, 0.0)

        try:
            hop = float(
                nx.shortest_path_length(
                    road_graph, edge_i[1], target_edge[0], weight=edge_length_weight
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        if Point is None or geom_i is None or target_geom is None or None in prev_xy or None in curr_xy:
            return hop
        prev_arc = float(geom_i.project(Point(float(prev_xy[0]), float(prev_xy[1]))))
        remaining_prev = max(float(geom_i.length) - prev_arc, 0.0)
        curr_arc = float(target_geom.project(Point(float(curr_xy[0]), float(curr_xy[1]))))
        return remaining_prev + hop + curr_arc

    true_rows, negative_rows = [], []
    audit = {
        "input_transitions": 0,
        "valid_inference_transitions": 0,
        "resolved_true_transitions": 0,
        "previous_candidate_oracle_hits": 0,
        "previous_candidate_oracle_denominator": 0,
        "previous_candidate_oracle_recall": 0.0,
        "next_candidate_oracle_hits": 0,
        "next_candidate_oracle_denominator": 0,
        "next_candidate_oracle_recall": 0.0,
        "retained_transitions": 0,
        "ranking_pairs_before_cap": 0,
        "ranking_pairs": 0,
        "skipped_missing_inference_observation": 0,
        "skipped_unresolved_true_edge": 0,
        "skipped_previous_true_not_in_live_candidates": 0,
        "skipped_next_true_not_in_live_candidates": 0,
        "skipped_unreachable_true_transition": 0,
        "skipped_no_reachable_live_negative": 0,
    }
    for _, track in subset.groupby("truck_id", sort=False):
        track = track.sort_values(time_col) if time_col else track
        track = track.reset_index(drop=True)
        for i in range(len(track) - 1):
            audit["input_transitions"] += 1
            row_i, row_j = track.iloc[i], track.iloc[i + 1]
            edge_i = trajectory_edge_id(row_i, road_graph)
            edge_j = trajectory_edge_id(row_j, road_graph)
            if edge_i is None or edge_j is None:
                audit["skipped_unresolved_true_edge"] += 1
                continue
            if road_graph.is_multigraph():
                data_i = road_graph.edges[edge_i]
                data_j = road_graph.edges[edge_j]
            else:
                data_i = road_graph.edges[edge_i[0], edge_i[1]]
                data_j = road_graph.edges[edge_j[0], edge_j[1]]
            audit["resolved_true_transitions"] += 1

            # Training must use the same noisy/causally-imputed coordinates
            # available at inference, never the simulator's latent
            # true_x/true_y path.
            px = to_float_safely(row_i.get("_inference_x"), np.nan)
            py = to_float_safely(row_i.get("_inference_y"), np.nan)
            tx = to_float_safely(row_j.get("_inference_x"), np.nan)
            ty = to_float_safely(row_j.get("_inference_y"), np.nan)
            if not (np.isfinite(px) and np.isfinite(py) and np.isfinite(tx) and np.isfinite(ty)):
                audit["skipped_missing_inference_observation"] += 1
                continue
            audit["valid_inference_transitions"] += 1
            prev_xy = (px, py)
            curr_xy = (tx, ty)
            prev_candidates = catalog_items_within_radius(
                catalog,
                strtree_bundle,
                px,
                py,
                radius,
                fallback_items=10,
            )
            next_candidates = catalog_items_within_radius(
                catalog,
                strtree_bundle,
                tx,
                ty,
                radius,
                fallback_items=10,
            )
            prev_candidate_ids = {catalog_edge_id(item) for item in prev_candidates}
            next_candidate_by_id = {
                catalog_edge_id(item): item for item in next_candidates
            }
            audit["previous_candidate_oracle_denominator"] += 1
            audit["next_candidate_oracle_denominator"] += 1
            previous_hit = edge_i in prev_candidate_ids
            next_hit = edge_j in next_candidate_by_id
            audit["previous_candidate_oracle_hits"] += int(previous_hit)
            audit["next_candidate_oracle_hits"] += int(next_hit)
            if not previous_hit:
                audit["skipped_previous_true_not_in_live_candidates"] += 1
            if not next_hit:
                audit["skipped_next_true_not_in_live_candidates"] += 1
            if not previous_hit or not next_hit:
                continue

            geom_i = edge_geometry(edge_i, data_i)
            geom_j = edge_geometry(edge_j, data_j)

            true_dist = route_distance(
                edge_i,
                data_i,
                geom_i,
                edge_j,
                data_j,
                geom_j,
                prev_xy,
                curr_xy,
            )
            if true_dist is None:
                audit["skipped_unreachable_true_transition"] += 1
                continue

            if Point is not None:
                point = Point(float(tx), float(ty))
                next_candidates = sorted(
                    next_candidates,
                    key=lambda item: (float(item[6].distance(point)), catalog_edge_id(item)),
                )
            else:
                next_candidates = sorted(
                    next_candidates,
                    key=lambda item: (
                        (item[4] - tx) ** 2 + (item[5] - ty) ** 2,
                        catalog_edge_id(item),
                    ),
                )
            negatives = []
            for item in next_candidates:
                target_edge = catalog_edge_id(item)
                if target_edge == edge_j:
                    continue
                neg_dist = route_distance(
                    edge_i,
                    data_i,
                    geom_i,
                    target_edge,
                    item[3],
                    item[6],
                    prev_xy,
                    curr_xy,
                )
                if neg_dist is None:
                    continue
                negatives.append(transition_features(data_i, item[3], neg_dist))
                if len(negatives) >= max_negatives:
                    break
            if not negatives:
                audit["skipped_no_reachable_live_negative"] += 1
                continue

            true_feat = transition_features(data_i, data_j, true_dist)
            audit["retained_transitions"] += 1
            for neg_feat in negatives:
                true_rows.append(true_feat)
                negative_rows.append(neg_feat)
                audit["ranking_pairs_before_cap"] += 1

    if cap is not None and len(true_rows) > cap:
        rng = np.random.default_rng(rng_seed)
        keep = rng.choice(len(true_rows), size=cap, replace=False)
        true_rows = [true_rows[i] for i in keep]
        negative_rows = [negative_rows[i] for i in keep]
    audit["ranking_pairs"] = int(len(true_rows))
    for prefix in ("previous", "next"):
        denominator = int(audit[f"{prefix}_candidate_oracle_denominator"])
        hits = int(audit[f"{prefix}_candidate_oracle_hits"])
        audit[f"{prefix}_candidate_oracle_recall"] = (
            float(hits / denominator) if denominator else 0.0
        )
    if return_audit:
        return true_rows, negative_rows, audit
    return true_rows, negative_rows


def train_qts_model(case_name, trajectory_df, road_graph, seed=DEFAULT_SEED, restarts=5, validation_fraction=0.2):
    """Train the 4-qubit QTS on real vs. false edge-pair transitions with the
    same margin ranking loss used for the QMM.
    """
    if "truck_id" not in trajectory_df.columns:
        raise ValueError("QTS calibration requires truck_id for train/validation truck separation.")
    split_rng = np.random.default_rng(int(seed) + 400009)
    trucks = np.asarray(sorted(trajectory_df["truck_id"].astype(str).unique()))
    split_rng.shuffle(trucks)
    validation_truck_count = max(1, int(round(len(trucks) * float(validation_fraction))))
    validation_truck_count = min(validation_truck_count, max(1, len(trucks) - 1))
    validation_trucks = trucks[:validation_truck_count]
    fit_trucks = trucks[validation_truck_count:]

    catalog = edge_catalog(road_graph)
    RANK_MARGIN = 0.2
    fit_true, fit_neg, fit_candidate_audit = build_transition_ranking_pairs(
        trajectory_df,
        road_graph,
        catalog,
        fit_trucks,
        cap=320,
        rng_seed=int(seed) + 500009,
        return_audit=True,
    )
    validation_true, validation_neg, validation_candidate_audit = build_transition_ranking_pairs(
        trajectory_df,
        road_graph,
        catalog,
        validation_trucks,
        cap=160,
        rng_seed=int(seed) + 600011,
        return_audit=True,
    )
    if not fit_true or not validation_true:
        raise RuntimeError(f"Could not construct QTS transition pairs for {case_name}.")
    fit_true_arr = np.asarray(fit_true, dtype=float)
    fit_neg_arr = np.asarray(fit_neg, dtype=float)
    validation_true_arr = np.asarray(validation_true, dtype=float)
    validation_neg_arr = np.asarray(validation_neg, dtype=float)
    baseline_validation_loss = float(RANK_MARGIN)
    print(
        f"Calibrating {case_name} QTS with {len(fit_true_arr)} fit / {len(validation_true_arr)} validation "
        f"transition pairs across {int(restarts)} restarts; constant baseline hinge loss={baseline_validation_loss:.6f}..."
    )
    print(
        f"  next-fix candidate oracle: fit "
        f"{fit_candidate_audit['next_candidate_oracle_hits']}/"
        f"{fit_candidate_audit['next_candidate_oracle_denominator']} "
        f"({fit_candidate_audit['next_candidate_oracle_recall']:.1%}); validation "
        f"{validation_candidate_audit['next_candidate_oracle_hits']}/"
        f"{validation_candidate_audit['next_candidate_oracle_denominator']} "
        f"({validation_candidate_audit['next_candidate_oracle_recall']:.1%})"
    )

    best_weights = None
    best_validation_loss = float("inf")
    restart_summary = []
    for restart in range(max(1, int(restarts))):
        restart_seed = int(seed) + 1013 * restart
        w = pnp.array(
            deterministic_ansatz_init(
                qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=qts_n_qubits), restart=restart
            ),
            requires_grad=True,
        )
        opt = qml.AdamOptimizer(stepsize=0.01)
        batch_size = 32
        max_epochs = 12
        patience = 3
        min_delta = 1e-4
        restart_best_loss = float("inf")
        restart_best_weights = None
        stale_epochs = 0
        for epoch in range(max_epochs):
            epoch_rng = np.random.default_rng(restart_seed + 7927 * epoch)
            order = epoch_rng.permutation(len(fit_true_arr))
            for start in range(0, len(order), batch_size):
                batch_idx = order[start:start + batch_size]
                batch_true = fit_true_arr[batch_idx]
                batch_neg = fit_neg_arr[batch_idx]

                def cost_fn(weights):
                    true_scores = bounded_qts_score(pnp.array(batch_true), weights)
                    neg_scores = bounded_qts_score(pnp.array(batch_neg), weights)
                    return pnp.mean(pnp.clip(RANK_MARGIN - (true_scores - neg_scores), 0, None))

                w, _cost = opt.step_and_cost(cost_fn, w)

            validation_true_scores = np.asarray(
                bounded_qts_score(pnp.array(validation_true_arr), w), dtype=float
            )
            validation_neg_scores = np.asarray(
                bounded_qts_score(pnp.array(validation_neg_arr), w), dtype=float
            )
            validation_loss = float(
                np.mean(np.clip(RANK_MARGIN - (validation_true_scores - validation_neg_scores), 0, None))
            )
            if validation_loss < restart_best_loss - min_delta:
                restart_best_loss = validation_loss
                restart_best_weights = pnp.array(np.asarray(w, dtype=float), requires_grad=False)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        restart_summary.append({
            "restart": restart,
            "restart_seed": restart_seed,
            "validation_loss": restart_best_loss,
            "beats_constant_baseline": bool(restart_best_loss < baseline_validation_loss),
        })
        print(f"[{case_name}] QTS restart {restart + 1}/{int(restarts)}: validation ranking loss={restart_best_loss:.6f}")
        if restart_best_loss < best_validation_loss:
            best_validation_loss = restart_best_loss
            best_weights = restart_best_weights

    selected = min(restart_summary, key=lambda row: row["validation_loss"])
    qts_summary_path = REPORTS_DIR / f"step06_{case_name}_qts_restart_selection.csv"
    pd.DataFrame(restart_summary).to_csv(qts_summary_path, index=False)
    if not selected["beats_constant_baseline"]:
        raise RuntimeError(
            f"{case_name} QTS calibration rejected: best validation ranking loss "
            f"{selected['validation_loss']:.6f} does not beat constant baseline "
            f"{baseline_validation_loss:.6f}."
        )

    weights_out = qts_weights_path(case_name)
    np.save(weights_out, np.array(best_weights, dtype=float))
    print(f"[{case_name}] QTS trained; validation ranking loss={selected['validation_loss']:.6f}, saved to {weights_out}")

    metadata_path = REPORTS_DIR / f"step06_{case_name}_qts_selection_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "case": case_name,
                "selection_rule": "minimum held-out validation pairwise ranking (hinge) loss",
                "ranking_margin": float(RANK_MARGIN),
                "base_seed": int(seed),
                "restarts": int(restarts),
                "fit_transition_pairs": int(len(fit_true_arr)),
                "validation_transition_pairs": int(len(validation_true_arr)),
                "fit_candidate_audit": fit_candidate_audit,
                "validation_candidate_audit": validation_candidate_audit,
                "constant_baseline_validation_loss": baseline_validation_loss,
                "selected_restart": int(selected["restart"]),
                "selected_validation_loss": float(selected["validation_loss"]),
                "beats_constant_baseline": bool(selected["beats_constant_baseline"]),
                "weights_file": str(weights_out),
            },
            handle,
            indent=2,
        )
    return best_weights

def run_meaningful_benchmark(case_name, graph_path, trajectory_csv, seed, confidence_threshold):
    print("\n--- RUNNING STEP05 BENCHMARK WITH STEP06 CALIBRATED WEIGHTS ---")
    bench = MapMatchingBenchmarker(
        graph_path,
        trajectory_csv,
        seed=seed,
        candidate_radius=INFERENCE_CANDIDATE_RADIUS_M,
        candidate_fallback_k=10,
        weights_file=str(weights_path(case_name)),
        qts_weights_file=str(qts_weights_path(case_name)) if qts_weights_path(case_name).exists() else None,
        confidence_dip_threshold=confidence_threshold,
    )
    q_path = bench.run_viterbi(mode="quantum", progress_every=0)
    c_path = bench.run_viterbi(mode="classical", progress_every=0)
    _, summary = bench.analyze_results(q_path, c_path)
    print(f"Calibrated benchmark summary for {case_name}: {summary}")


def train_and_save_ablation_classifiers(
    case_name,
    graph_path,
    calibration_df,
    seed,
    weights_file,
    qts_weights_file,
    confidence_threshold,
    radius=INFERENCE_CANDIDATE_RADIUS_M,
):
    """Fit the ablation-grid MLP and RBF-SVM once on the calibration-pool
    trucks only, and persist them, matching the fixed QMM/QTS calibration
    split so that no held-out evaluation truck is seen in training. Training here reuses MapMatchingBenchmarker's own
    _train_ablation_mlp/_train_budget_matched_svm on a benchmarker
    restricted to the calibration pool for both decoding and the ablation
    training data. The classifiers use fix-level true/false labels within
    inference-style local candidate sets and truck-group validation.
    """
    fd, tmp_name = tempfile.mkstemp(prefix="step06_calibration_pool_", suffix=".csv")
    os.close(fd)
    calibration_csv = Path(tmp_name)
    try:
        calibration_df.to_csv(calibration_csv, index=False)
        bench = MapMatchingBenchmarker(
            graph_path,
            calibration_csv,
            seed=seed,
            candidate_radius=radius,
            weights_file=str(weights_file),
            qts_weights_file=str(qts_weights_file) if qts_weights_file else None,
            confidence_dip_threshold=float(confidence_threshold),
            classical_training_csv=str(calibration_csv),
        )
    finally:
        try:
            calibration_csv.unlink()
        except OSError:
            pass

    with open(mlp_ablation_model_path(case_name), "wb") as handle:
        pickle.dump({"model": bench.mlp_model, "selection": bench.mlp_selection}, handle)
    with open(svm_ablation_model_path(case_name), "wb") as handle:
        pickle.dump({"model": bench.svm_model, "selection": bench.svm_selection}, handle)
    print(
        f"[{case_name}] Saved ablation-grid MLP/SVM (calibration-pool-only, "
        f"n={len(calibration_df)} rows) to {mlp_ablation_model_path(case_name).name} "
        f"and {svm_ablation_model_path(case_name).name}"
    )


def _run_calibration_case(case_name, seed, run_benchmark=False, restarts=5, validation_fraction=0.2):
    graph_path = unified_graph_path(case_name)
    traj_path = trajectory_path(case_name)
    if not graph_path.exists() or not traj_path.exists():
        return f"Skipping {case_name}: missing graph or trajectory file."

    print(f"\n=== Calibrating {case_name} ===")
    G = nx.read_graphml(graph_path)
    df = pd.read_csv(traj_path)

    # Restrict calibration to the case's calibration-pool trucks only, so no
    # truck used here can also be drawn as a step07 held-out evaluation
    # truck. The calibration/holdout boundary is fixed by
    # pipeline_config.split_calibration_and_holdout_trucks and is the same
    # partition step07 uses to build its held-out draw pool.
    calibration_trucks, _holdout_trucks = split_calibration_and_holdout_trucks(
        df["truck_id"].astype(str).unique()
    )
    calibration_df = df[df["truck_id"].astype(str).isin(set(calibration_trucks))].copy()

    opt_weights, confidence_threshold = train_calibrated_model(
        case_name,
        calibration_df,
        G,
        seed=seed,
        restarts=restarts,
        validation_fraction=validation_fraction,
    )
    weights_out = weights_path(case_name)
    np.save(weights_out, np.array(opt_weights, dtype=float))

    train_qts_model(
        case_name,
        calibration_df,
        G,
        seed=seed + 700003,
        restarts=restarts,
        validation_fraction=validation_fraction,
    )

    # Matched classical controls — same data/loss/budget as
    # the quantum calibration above, just a classical scoring function.
    classical_weights = train_calibrated_classical_model(
        case_name, calibration_df, G, seed=seed, restarts=restarts, validation_fraction=validation_fraction,
    )
    np.save(classical_matched_weights_path(case_name), np.array(classical_weights, dtype=float))

    classical_qts_weights = train_calibrated_classical_qts_model(
        case_name, calibration_df, G, seed=seed + 700003, restarts=restarts, validation_fraction=validation_fraction,
    )
    np.save(classical_matched_qts_weights_path(case_name), np.array(classical_qts_weights, dtype=float))

    train_and_save_ablation_classifiers(
        case_name,
        graph_path,
        calibration_df,
        seed=seed,
        weights_file=weights_out,
        qts_weights_file=qts_weights_path(case_name),
        confidence_threshold=confidence_threshold,
    )

    if run_benchmark:
        run_meaningful_benchmark(case_name, graph_path, traj_path, seed, confidence_threshold)
    return f"Saved optimized weights to: {weights_out}; threshold={confidence_threshold:.6f}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate quantum scoring across all hard-case networks.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_max_workers(),
        help="Maximum parallel workers to use (default: all CPU cores).",
    )
    parser.add_argument(
        "--force-parallel",
        action="store_true",
        help="Force multi-process calibration across cases (can be unstable on some macOS/Python setups).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base seed for reproducible weight initialization.")
    parser.add_argument("--restarts", type=int, default=5, help="Independent calibration initializations per case.")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Held-out fraction used only for restart selection.")
    parser.add_argument(
        "--with-benchmark",
        action="store_true",
        help="After each case calibration, run the Step05 benchmark using calibrated Step06 weights (slower).",
    )
    args = parser.parse_args()

    requested_workers = max(1, min(int(args.max_workers), len(CASE_NAMES)))
    max_workers = requested_workers if args.force_parallel else 1
    if requested_workers > 1 and max_workers == 1:
        print(
            "Stability mode: running Step06 in serial (1 worker). "
            "Use --force-parallel to override."
        )

    if max_workers == 1:
        for idx, case_name in enumerate(CASE_NAMES):
            print(
                _run_calibration_case(
                    case_name,
                    int(args.seed) + idx,
                    run_benchmark=bool(args.with_benchmark),
                    restarts=int(args.restarts),
                    validation_fraction=float(args.validation_fraction),
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    _run_calibration_case,
                    case_name,
                    int(args.seed) + idx,
                    bool(args.with_benchmark),
                    int(args.restarts),
                    float(args.validation_fraction),
                )
                for idx, case_name in enumerate(CASE_NAMES)
            ]
            for fut in as_completed(futures):
                print(fut.result())

    write_step06_calibration_consolidated_outputs(CASE_NAMES)
