import argparse
import traceback
import json
import pickle
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from scipy.spatial import KDTree
try:
    from shapely import wkt as shapely_wkt
    from shapely.geometry import Point, LineString
    from shapely.strtree import STRtree
except ImportError:  # Endpoint projection remains available without Shapely.
    shapely_wkt = None
    Point = None
    LineString = None
    STRtree = None
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from pipeline_config import (
    DEFAULT_SEED,
    INFERENCE_CANDIDATE_RADIUS_M,
    PLOTS_DIR,
    QMM_NEGATIVE_SAMPLE_RADIUS_M,
    QTS_NEGATIVE_SAMPLE_RADIUS_M,
    REPORTS_DIR,
    default_max_workers,
    deterministic_ansatz_init,
    ensure_project_dirs,
    make_qml_device,
    load_confidence_threshold,
    qts_weights_path,
    trajectory_path,
    unified_graph_path,
    weights_path,
)

CASE_NAMES = [
    "Rozelle_Interchange_NSW",
    "West_Gate_Tunnel_VIC",
    "NorthConnex_NSW",
    "Light_Horse_Interchange_NSW",
    "Domain_Tunnel_VIC",
    "M80_Princes_Freeway_VIC",
]

# Assumed vertical clearance between adjacent GraphML layer indices, used
# only to compute a supplementary layer-aware proxy error.
# This is a fixed engineering-typical assumption, not a per-edge surveyed
# or design elevation value — the graph carries no measured z-coordinate —
# so the resulting metric is explicitly a synthetic proxy, not true 3D
# RMSE, and is reported alongside (never in place of) horizontal RMSE.
ASSUMED_LEVEL_SPACING_M = 4.5

# Sentinel distinct from None so a cached "no path exists" result (None)
# is distinguishable from "not yet computed" in the transition hop cache.
_UNSET_HOP = object()

# run_viterbi(mode=...) values whose transition score uses a learned
# (QTS or matched-classical) factor rather than the plain classical
# distance-mismatch term. Modes not listed here get the classical
# transition regardless of their emission scorer (this is
# what makes "qmm_entanglement" a true QMM-only ablation cell rather than
# silently also picking up QTS).
TRANSITION_SCORED_VITERBI_MODES = {"quantum", "qts_only", "classical_matched", "level_ekf_hmm_quantum"}


def _pick_xy_columns(df, x_candidates, y_candidates, label):
    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)
    if x_col is None or y_col is None:
        raise ValueError(
            f"Could not find {label} XY columns. Expected one of "
            f"{x_candidates} and {y_candidates}."
        )
    return x_col, y_col


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_prob_from_expectation(z_val):
    z = float(np.clip(z_val, -1.0, 1.0))
    return float(np.clip((z + 1.0) / 2.0, 1e-9, 1.0))


def _to_flag(value):
    """Parse GraphML/CSV boolean values without depending on their dtype.

    NetworkX commonly round-trips numeric GraphML attributes as strings such
    as ``"1.0"``.  Treat every finite non-zero numeric representation as true
    and retain explicit textual true/false spellings, so ``"1.0"`` is read
    as true rather than false.
    """
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value) and float(value) != 0.0)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "on"}:
        return True
    if text in {"", "false", "no", "n", "off", "none", "nan", "null"}:
        return False
    try:
        number = float(text)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number != 0.0)


def _edge_key_to_string(edge):
    u, v, k = edge
    return f"{u}_{v}_{int(k)}"


def _node_id_string(value):
    """Normalize numeric CSV node identifiers to GraphML string identifiers."""
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def _parse_edge_key(value):
    if isinstance(value, tuple) and len(value) >= 2:
        if len(value) >= 3:
            return str(value[0]), str(value[1]), int(_to_float(value[2], 0.0))
        return str(value[0]), str(value[1]), 0

    if isinstance(value, list) and len(value) >= 2:
        if len(value) >= 3:
            return str(value[0]), str(value[1]), int(_to_float(value[2], 0.0))
        return str(value[0]), str(value[1]), 0

    if isinstance(value, str):
        parts = value.split("_")
        if len(parts) >= 3:
            return parts[0], parts[1], int(_to_float(parts[2], 0.0))
        if len(parts) == 2:
            return parts[0], parts[1], 0

    return None


def build_edge_geometry_from_graph(graph, edge, attrs):
    """Full-geometry LineString for an edge: the WKT road centerline when
    present, otherwise the straight segment between its endpoint nodes.
    Standalone (graph-only) counterpart to
    MapMatchingBenchmarker._build_edge_geometry, shared with step06's
    calibration-time negative sampling so training and inference use the
    same geometric representation of every edge."""
    if LineString is None:
        return None
    geometry_text = (attrs or {}).get("geometry")
    if shapely_wkt is not None and geometry_text:
        try:
            geometry = shapely_wkt.loads(str(geometry_text))
            if geometry is not None and not geometry.is_empty:
                return geometry
        except Exception:
            pass
    u, v, _k = edge
    ax = _to_float(graph.nodes[u].get("x"))
    ay = _to_float(graph.nodes[u].get("y"))
    bx = _to_float(graph.nodes[v].get("x"))
    by = _to_float(graph.nodes[v].get("y"))
    return LineString([(ax, ay), (bx, by)])


def edge_length_weight(_u, _v, data):
    if isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
        lengths = [_to_float(attrs.get("length"), 1.0) for attrs in data.values()]
        return min(lengths) if lengths else 1.0
    return _to_float(data.get("length"), 1.0)


def _segmented_backtrack(path_scores, backpointers, trellis_breaks, total_steps):
    """Backtrack the Viterbi trellis in independent segments split at each
    documented re-anchor point. A single backpointer walk would treat a
    re-anchor's self-pointing sentinel as a real same-edge transition and
    smear the re-anchored edge backward over the true pre-break path, also
    hiding the break from the impossible-jump diagnostic.

    A break at time t means backpointers[t-1] is the self-pointing sentinel
    built during re-anchoring and must not be dereferenced. Every timestep
    strictly before a break is instead the independently optimal sub-path
    already recorded in path_scores/backpointers from before the trellis
    was reset, so each segment backtracks on its own starting from its own
    argmax."""
    break_set = set(trellis_breaks)
    path = [None] * total_steps
    end = total_steps - 1
    while end >= 0:
        current_edge = max(path_scores[end], key=path_scores[end].get)
        path[end] = current_edge
        t = end
        while t not in break_set and t > 0:
            current_edge = backpointers[t - 1].get(current_edge, current_edge)
            t -= 1
            path[t] = current_edge
        end = t - 1
    return path


def extract_9_features(edge_data, traffic_regime=0.0):
    """
    Standardized 9-feature extraction for both PQC and classical baseline models.
    """
    length = _to_float(edge_data.get("length"), 0.0)
    bearing = _to_float(edge_data.get("bearing"), 0.0)

    maxspeed = edge_data.get("maxspeed", 60.0)
    if isinstance(maxspeed, list) and maxspeed:
        maxspeed = _to_float(maxspeed[0], 60.0)
    else:
        maxspeed = _to_float(maxspeed, 60.0)

    tunnel = 1.0 if _to_flag(edge_data.get("tunnel", 0)) else 0.0
    bridge = 1.0 if _to_flag(edge_data.get("bridge", 0)) else 0.0

    layer = _to_float(edge_data.get("layer", 0), 0.0)

    lanes = edge_data.get("lanes", 2)
    if isinstance(lanes, list) and lanes:
        lanes = _to_float(lanes[0], 2.0)
    else:
        lanes = _to_float(lanes, 2.0)

    oneway = 1.0 if _to_flag(edge_data.get("oneway", 0)) else 0.0

    f0 = min(length / 500.0, 1.0)
    f1 = (bearing % 360.0) / 360.0
    f2 = float(np.clip((maxspeed - 10.0) / 110.0, 0.0, 1.0))
    f3 = float(np.clip((layer + 3.0) / 8.0, 0.0, 1.0))
    f4 = float(np.clip(lanes / 8.0, 0.0, 1.0))
    f5 = oneway
    f6 = tunnel
    f7 = bridge
    f8 = float(np.clip(_to_float(traffic_regime, 0.0), 0.0, 1.0))

    return [f0, f1, f2, f3, f4, f5, f6, f7, f8]



# Local, few-qubit readout wires for the QMM/QTS circuits (see
# QMM_READOUT_WIRES / QTS_READOUT_WIRES below). Kept deliberately small (3 of
# 9 qubits; 2 of 4 qubits) rather than summing over the full register: a
# gradient-variance scan (diagnostic_barren_plateau_scan.py) shows variance
# collapsing sharply with qubit count at fixed depth, consistent with the
# barren-plateau literature already cited in the paper (McClean et al.,
# Cerezo et al.). A wide/global observable would make this worse, not
# better, so the readout stays local while the circuit gains expressivity
# instead via data re-uploading (two encoding rounds).
QMM_READOUT_WIRES = (0, 3, 6)
QTS_READOUT_WIRES = (0, 2)


def _local_readout_observable(wires):
    obs = qml.PauliZ(wires[0])
    for w in wires[1:]:
        obs = obs + qml.PauliZ(w)
    return obs / float(len(wires))


class QuantumEmissionModel:
    def __init__(self, n_qubits=9, seed=42, weights_file=None, entangled=True):
        self.n_qubits = int(n_qubits)
        self.entangled = bool(entangled)
        self.dev = make_qml_device(qml, wires=self.n_qubits, prefer_gpu=True, gpu_fraction=0.8)
        readout = _local_readout_observable(QMM_READOUT_WIRES)

        if weights_file and Path(weights_file).exists():
            try:
                loaded = np.load(weights_file)
                self.weights = pnp.array(loaded, requires_grad=False)
            except Exception:
                self.weights = pnp.array(
                    deterministic_ansatz_init(
                        qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=self.n_qubits), restart=int(seed)
                    ),
                    requires_grad=False,
                )
        else:
            self.weights = pnp.array(
                deterministic_ansatz_init(
                    qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=self.n_qubits), restart=int(seed)
                ),
                requires_grad=False,
            )

        @qml.qnode(self.dev)
        def _circuit(features, weights):
            # Data re-uploading: the feature encoding is applied twice,
            # interleaved with the two trainable entangling blocks, instead
            # of once followed by a deeper stack of random layers. This
            # increases expressivity per parameter without adding the extra
            # entangling depth that drives barren plateaus.
            qml.AngleEmbedding(pnp.pi * features, wires=range(self.n_qubits), rotation="Y")
            if self.entangled:
                qml.StronglyEntanglingLayers(weights[0:1], wires=range(self.n_qubits))
                qml.AngleEmbedding(pnp.pi * features, wires=range(self.n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights[1:2], wires=range(self.n_qubits))
            else:
                # Keep the same trainable parameter count while removing all
                # multi-qubit gates for the entanglement ablation.
                for wire in range(self.n_qubits):
                    qml.Rot(*weights[0, wire], wires=wire)
                qml.AngleEmbedding(pnp.pi * features, wires=range(self.n_qubits), rotation="Y")
                for wire in range(self.n_qubits):
                    qml.Rot(*weights[1, wire], wires=wire)
            return qml.expval(readout)

        self.circuit = _circuit

    def score(self, edge_data, traffic_val):
        features = pnp.array(
            extract_9_features(edge_data, traffic_regime=traffic_val),
            requires_grad=False,
        )
        return float(self.circuit(features, self.weights))


class LSTMEdgeClassifier:
    """Small dependency-free LSTM feature encoder with a fitted linear head."""

    def __init__(self, input_size=9, hidden_size=12, seed=42):
        rng = np.random.default_rng(int(seed))
        self.hidden_size = int(hidden_size)
        scale = 1.0 / np.sqrt(input_size + hidden_size)
        self.kernel = rng.normal(0.0, scale, size=(input_size + hidden_size, 4 * hidden_size))
        self.bias = np.zeros(4 * hidden_size, dtype=float)
        self.output_weights = None

    @staticmethod
    def _sigmoid(values):
        values = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-values))

    def encode(self, sequence):
        h = np.zeros(self.hidden_size, dtype=float)
        c = np.zeros(self.hidden_size, dtype=float)
        for features in np.asarray(sequence, dtype=float):
            gates = np.concatenate([features, h]) @ self.kernel + self.bias
            i, f, g, o = np.split(gates, 4)
            i, f, o = self._sigmoid(i), self._sigmoid(f), self._sigmoid(o)
            c = f * c + i * np.tanh(g)
            h = o * np.tanh(c)
        return h

    def fit(self, sequences, labels, ridge=1e-3):
        encoded = np.vstack([self.encode(sequence) for sequence in sequences])
        design = np.column_stack([encoded, np.ones(len(encoded))])
        identity = np.eye(design.shape[1])
        identity[-1, -1] = 0.0
        self.output_weights = np.linalg.solve(
            design.T @ design + ridge * identity,
            design.T @ np.asarray(labels, dtype=float),
        )
        return self

    def predict_proba(self, sequence):
        if self.output_weights is None:
            return 0.5
        encoded = np.append(self.encode(sequence), 1.0)
        return float(np.clip(self._sigmoid(encoded @ self.output_weights), 1e-9, 1.0))


def transition_features(prev_edge_data, curr_edge_data, transition_dist):
    """Shared (e_i, e_j) edge-pair feature vector for the QTS: used identically
    by calibration (step06) and inference (here) so the trained weights see
    the same feature semantics in both places."""
    prev_layer = _to_float(prev_edge_data.get("layer", 0.0), 0.0)
    curr_layer = _to_float(curr_edge_data.get("layer", 0.0), 0.0)
    layer_delta = min(abs(curr_layer - prev_layer) / 3.0, 1.0)

    prev_speed = _to_float(prev_edge_data.get("maxspeed", 60.0), 60.0)
    curr_speed = _to_float(curr_edge_data.get("maxspeed", 60.0), 60.0)
    speed_feat = min(abs(curr_speed - prev_speed) / 100.0, 1.0)

    dist_feat = min(_to_float(transition_dist, 150.0) / 150.0, 1.0)

    prev_bearing = _to_float(prev_edge_data.get("bearing", 0.0), 0.0)
    curr_bearing = _to_float(curr_edge_data.get("bearing", 0.0), 0.0)
    raw_delta = abs(curr_bearing - prev_bearing) % 360.0
    heading_delta = min(raw_delta, 360.0 - raw_delta) / 180.0

    return [layer_delta, speed_feat, dist_feat, heading_delta]


class QuantumTransitionScorer:
    def __init__(self, n_qubits=4, seed=42, weights_file=None):
        self.n_qubits = int(n_qubits)
        self.dev = make_qml_device(qml, wires=self.n_qubits, prefer_gpu=True, gpu_fraction=0.8)
        readout = _local_readout_observable(QTS_READOUT_WIRES)

        loaded_weights = None
        if weights_file and Path(weights_file).exists():
            try:
                loaded_weights = pnp.array(np.load(weights_file), requires_grad=False)
            except Exception:
                loaded_weights = None
        if loaded_weights is None:
            loaded_weights = pnp.array(
                deterministic_ansatz_init(
                    qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=self.n_qubits), restart=int(seed)
                ),
                requires_grad=False,
            )
        self.weights = loaded_weights

        @qml.qnode(self.dev)
        def _circuit(features, weights):
            # Data re-uploading, matching the QMM circuit's design.
            qml.AngleEmbedding(pnp.pi * features, wires=range(self.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights[0:1], wires=range(self.n_qubits))
            qml.AngleEmbedding(pnp.pi * features, wires=range(self.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights[1:2], wires=range(self.n_qubits))
            return qml.expval(readout)

        self.circuit = _circuit

    def score(self, prev_edge_data, curr_edge_data, transition_dist):
        features = np.array(
            transition_features(prev_edge_data, curr_edge_data, transition_dist),
            dtype=float,
        )
        z = float(self.circuit(features, self.weights))
        return float(np.clip((z + 1.0) / 2.0, 1e-9, 1.0))


# Matched classical controls: a small classical MLP
# forward pass, differentiable with the same pennylane.numpy/autograd stack
# used for the quantum circuits, so it can be trained by the identical
# qml.AdamOptimizer margin-ranking loop in step06 — same 9/4 input features,
# same 320/160-point truck-disjoint calibration split, same loss, same
# restarts/epochs/batch size. Hidden sizes are chosen so the parameter count
# is close to (not identical to, since the two architecture families are not
# directly comparable parameter-for-parameter) the quantum circuits' 54
# (QMM: 2 StronglyEntanglingLayers x 9 wires x 3 params) and 24 (QTS: 2 x 4 x
# 3) trainable parameters.
CLASSICAL_MATCHED_EMISSION_HIDDEN = 5   # 11*5+1 = 56 params, vs QMM's 54
CLASSICAL_MATCHED_TRANSITION_HIDDEN = 4  # 6*4+1 = 25 params, vs QTS's 24


def classical_matched_weight_count(n_inputs, hidden_size):
    return n_inputs * hidden_size + hidden_size + hidden_size + 1


def classical_matched_forward(features, weights, n_inputs, hidden_size):
    """Small MLP forward pass (n_inputs -> hidden_size, tanh -> 1 logit),
    written with plain array ops so it is differentiable by the same
    autograd-based optimizer used for the quantum circuits. `features` may
    be a single (n_inputs,) vector or a (batch, n_inputs) array."""
    idx = 0
    w1 = weights[idx: idx + n_inputs * hidden_size].reshape(n_inputs, hidden_size)
    idx += n_inputs * hidden_size
    b1 = weights[idx: idx + hidden_size]
    idx += hidden_size
    w2 = weights[idx: idx + hidden_size]
    idx += hidden_size
    b2 = weights[idx]
    hidden = pnp.tanh(pnp.dot(features, w1) + b1)
    return pnp.dot(hidden, w2) + b2


def bounded_classical_matched_score(features, weights, n_inputs, hidden_size):
    """Same [0, 1] Pauli-Z-style mapping convention as bounded_quantum_score,
    via a sigmoid on the classical logit instead of a Z-expectation."""
    logit = classical_matched_forward(features, weights, n_inputs, hidden_size)
    return 1.0 / (1.0 + pnp.exp(-logit))


class ClassicalMatchedEmissionModel:
    """Classical counterpart to QuantumEmissionModel, trained by
    step06.train_calibrated_classical_model with the identical procedure
    used for the QMM."""

    def __init__(self, seed=42, weights_file=None, hidden_size=CLASSICAL_MATCHED_EMISSION_HIDDEN):
        self.n_inputs = 9
        self.hidden_size = int(hidden_size)
        n_params = classical_matched_weight_count(self.n_inputs, self.hidden_size)
        if weights_file and Path(weights_file).exists():
            try:
                self.weights = pnp.array(np.load(weights_file), requires_grad=False)
            except Exception:
                self.weights = pnp.array(
                    deterministic_ansatz_init((n_params,), restart=int(seed)), requires_grad=False
                )
        else:
            self.weights = pnp.array(
                deterministic_ansatz_init((n_params,), restart=int(seed)), requires_grad=False
            )

    def score(self, edge_data, traffic_val):
        features = pnp.array(
            extract_9_features(edge_data, traffic_regime=traffic_val),
            requires_grad=False,
        )
        return float(bounded_classical_matched_score(features, self.weights, self.n_inputs, self.hidden_size))


class ClassicalMatchedTransitionScorer:
    """Classical counterpart to QuantumTransitionScorer."""

    def __init__(self, seed=42, weights_file=None, hidden_size=CLASSICAL_MATCHED_TRANSITION_HIDDEN):
        self.n_inputs = 4
        self.hidden_size = int(hidden_size)
        n_params = classical_matched_weight_count(self.n_inputs, self.hidden_size)
        if weights_file and Path(weights_file).exists():
            try:
                self.weights = pnp.array(np.load(weights_file), requires_grad=False)
            except Exception:
                self.weights = pnp.array(
                    deterministic_ansatz_init((n_params,), restart=int(seed)), requires_grad=False
                )
        else:
            self.weights = pnp.array(
                deterministic_ansatz_init((n_params,), restart=int(seed)), requires_grad=False
            )

    def score(self, prev_edge_data, curr_edge_data, transition_dist):
        features = pnp.array(
            transition_features(prev_edge_data, curr_edge_data, transition_dist),
            requires_grad=False,
        )
        return float(bounded_classical_matched_score(features, self.weights, self.n_inputs, self.hidden_size))


class MapMatchingBenchmarker:
    def __init__(
        self,
        graph_path,
        traj_csv,
        seed=42,
        candidate_radius=INFERENCE_CANDIDATE_RADIUS_M,
        candidate_fallback_k=10,
        weights_file=None,
        qts_weights_file=None,
        classical_matched_weights_file=None,
        classical_matched_qts_weights_file=None,
        beam_width=None,
        sigma_xy=45.0,
        sigma_d=50.0,
        confidence_dip_threshold=None,
        kappa_min=20,
        kappa_max=120,
        checkpoint_id=None,
        checkpoint_every=100,
        classical_training_csv=None,
        mlp_model_file=None,
        svm_model_file=None,
    ):
        self.G = nx.read_graphml(graph_path)
        self.df = pd.read_csv(traj_csv)
        self.classical_training_df = (
            pd.read_csv(classical_training_csv) if classical_training_csv else self.df
        )
        self.seed = int(seed)
        self.candidate_radius = float(candidate_radius)
        self.candidate_fallback_k = int(candidate_fallback_k)
        self.quantum = QuantumEmissionModel(seed=seed, weights_file=weights_file)
        self.quantum_no_entanglement = QuantumEmissionModel(
            seed=seed,
            weights_file=weights_file,
            entangled=False,
        )
        # Frozen/random-weight control: identical circuit
        # architecture to self.quantum, but no weights_file is ever passed —
        # always the untrained deterministic-init draw, so it measures
        # whether calibration is contributing anything beyond the circuit's
        # fixed inductive bias.
        self.quantum_frozen = QuantumEmissionModel(seed=seed, weights_file=None)
        self.transition_scorer = QuantumTransitionScorer(seed=seed + 17, weights_file=qts_weights_file)
        # Matched classical controls — same features, same
        # training procedure, only the scoring function differs.
        self.classical_matched = ClassicalMatchedEmissionModel(seed=seed, weights_file=classical_matched_weights_file)
        self.classical_matched_transition_scorer = ClassicalMatchedTransitionScorer(
            seed=seed + 17, weights_file=classical_matched_qts_weights_file
        )
        self.sigma_xy = float(sigma_xy)
        self.sigma_d = float(sigma_d)
        if confidence_dip_threshold is None:
            raise ValueError(
                "confidence_dip_threshold must be supplied from Step06 validation calibration"
            )
        self.confidence_dip_threshold = float(confidence_dip_threshold)
        self.kappa_min = max(1, int(kappa_min))
        self.kappa_max = max(self.kappa_min, int(kappa_max))
        self.beam_width = int(beam_width) if beam_width is not None else None
        self.checkpoint_id = str(checkpoint_id) if checkpoint_id else None
        self.checkpoint_every = max(1, int(checkpoint_every))
        self._quantum_score_cache = {}
        # Network-hop distance between prev_edge's end node and curr_edge's
        # start node, cached per (prev_edge, curr_edge) pair since it does
        # not depend on the vehicle's position along either edge. Not used for same-edge transitions.
        self._transition_hop_cache = {}
        self._edge_length_cache = {}
        self._trellis_breaks = []
        self._connectivity_reanchors = []
        self._run_diagnostics = {}
        if self.beam_width is not None and self.beam_width < 1:
            self.beam_width = None

        self.obs_x_col, self.obs_y_col = _pick_xy_columns(
            self.df,
            ["obs_x", "x_noisy", "x", "lon"],
            ["obs_y", "y_noisy", "y", "lat"],
            "observation",
        )

        self.true_x_col = None
        self.true_y_col = None
        if "true_x" in self.df.columns and "true_y" in self.df.columns:
            self.true_x_col, self.true_y_col = "true_x", "true_y"
        elif "x_true" in self.df.columns and "y_true" in self.df.columns:
            self.true_x_col, self.true_y_col = "x_true", "y_true"

        # ``edge_key`` alone is not a serialized edge ID: it identifies a
        # parallel record only together with edge_u and edge_v.  Treating the
        # scalar key as a generic edge column made key-aware truth resolution
        # fall through to the first parallel edge.
        self.true_edge_col = next(
            (c for c in ["true_edge_id", "edge_id", "true_edge"] if c in self.df.columns),
            None,
        )

        self.edge_ids = []
        self.edge_midpoints = []
        self._edge_geometry_cache = {}
        if self.G.is_multigraph():
            edge_iter = self.G.edges(keys=True, data=True)
        else:
            edge_iter = ((u, v, 0, data) for u, v, data in self.G.edges(data=True))

        edge_geometries = []
        for u, v, k, data in edge_iter:
            edge = (str(u), str(v), int(k))
            self.edge_ids.append(edge)
            ux = _to_float(self.G.nodes[u].get("x"))
            uy = _to_float(self.G.nodes[u].get("y"))
            vx = _to_float(self.G.nodes[v].get("x"))
            vy = _to_float(self.G.nodes[v].get("y"))
            self.edge_midpoints.append(((ux + vx) / 2.0, (uy + vy) / 2.0))
            geometry = self._build_edge_geometry(edge, data)
            self._edge_geometry_cache[edge] = geometry
            edge_geometries.append(geometry)

        if not self.edge_ids:
            raise RuntimeError("Graph contains no edges to benchmark.")

        self.edges_by_uv = {}
        for edge in self.edge_ids:
            uv = (str(edge[0]), str(edge[1]))
            self.edges_by_uv.setdefault(uv, []).append(edge)

        # Geometry-aware candidate index: search
        # tests true distance to each edge's full line geometry (WKT
        # centerline when present, else the straight endpoint segment),
        # not a single per-edge midpoint, so a long or curved edge is not
        # excluded from candidacy just because its midpoint happens
        # to sit outside the search radius. This reuses the same geometry
        # objects that _project_to_edge scores against, so candidate
        # generation and candidate scoring are geometrically consistent.
        self.edge_geometries = edge_geometries
        if STRtree is not None and edge_geometries:
            self.edge_strtree = STRtree(edge_geometries)
        else:
            self.edge_strtree = None
        self.edge_tree = KDTree(self.edge_midpoints)
        self.levels = sorted({self._edge_layer(edge) for edge in self.edge_ids})
        level_started = time.perf_counter()
        self.level_transition_matrix = self._build_level_transition_matrix()
        self.level_ekf_states, self.level_probabilities = self._run_level_probability_ekf()
        self.level_precompute_seconds = time.perf_counter() - level_started
        # Ablation-grid MLP/SVM are fit once on the calibration-pool trucks
        # only, identically to QMM/QTS: a
        # mlp_model_file/svm_model_file supplied here is loaded rather than
        # retrained, so 20 evaluation seeds reuse the same fixed classifier
        # and never see a held-out evaluation truck. Training (the fallback below)
        # remains available for the one-time calibration step that produces
        # these files.
        self.mlp_model, self.mlp_selection = self._load_or_train_sklearn_model(
            mlp_model_file, self._train_ablation_mlp
        )
        self.svm_model, self.svm_selection = self._load_or_train_sklearn_model(
            svm_model_file, self._train_budget_matched_svm
        )
        self.lstm_model = self._train_ablation_lstm()

    @staticmethod
    def _load_or_train_sklearn_model(model_file, train_fn):
        if model_file and Path(model_file).exists():
            with open(model_file, "rb") as handle:
                payload = pickle.load(handle)
            selection = payload.get("selection", {})
            expected_supervision = (
                "fix-level correctness within inference-style local candidate sets"
            )
            if selection.get("supervision") != expected_supervision:
                raise RuntimeError(
                    f"Stale ablation model {model_file}: expected {expected_supervision!r}. "
                    "Rerun step06_quantum_calibration_corrected.py before Step07."
                )
            return payload["model"], selection
        return train_fn()

    def _edge_attrs(self, edge):
        u, v, k = edge
        data = self.G.get_edge_data(u, v)
        if data is None:
            return {}
        if self.G.is_multigraph():
            return data.get(int(k), {})
        return data

    def _edge_layer(self, edge):
        attrs = self._edge_attrs(edge)
        return int(_to_float(attrs.get("layer", 0.0), 0.0))

    def _build_level_transition_matrix(self):
        """Estimate graph-constrained level transitions without trajectory labels."""
        level_to_idx = {level: idx for idx, level in enumerate(self.levels)}
        counts = np.ones((len(self.levels), len(self.levels)), dtype=float) * 0.05
        for level, idx in level_to_idx.items():
            counts[idx, idx] += 2.0
        incoming = {}
        outgoing = {}
        for edge in self.edge_ids:
            incoming.setdefault(edge[1], []).append(edge)
            outgoing.setdefault(edge[0], []).append(edge)
        for node in self.G.nodes:
            for prev_edge in incoming.get(str(node), []):
                for curr_edge in outgoing.get(str(node), []):
                    i = level_to_idx[self._edge_layer(prev_edge)]
                    j = level_to_idx[self._edge_layer(curr_edge)]
                    counts[i, j] += 1.0
        return counts / counts.sum(axis=1, keepdims=True)

    def _build_edge_geometry(self, edge, attrs):
        """Full-geometry LineString for an edge: the WKT road centerline
        when present, otherwise the straight segment between its endpoint
        nodes. Shared by candidate generation (_get_candidates) and
        candidate scoring (_project_to_edge) so both use the same
        geometric representation of the edge."""
        return build_edge_geometry_from_graph(self.G, edge, attrs)

    def _project_to_edge(self, edge, x, y):
        """Return the closest point on an edge geometry (endpoint fallback)."""
        if Point is not None:
            geometry = self._edge_geometry_cache.get(edge)
            if geometry is None:
                geometry = self._build_edge_geometry(edge, self._edge_attrs(edge))
                self._edge_geometry_cache[edge] = geometry
            point = geometry.interpolate(geometry.project(Point(float(x), float(y))))
            if point is not None and not point.is_empty:
                return float(point.x), float(point.y)
        u, v, _ = edge
        ax = _to_float(self.G.nodes[u].get("x"))
        ay = _to_float(self.G.nodes[u].get("y"))
        bx = _to_float(self.G.nodes[v].get("x"))
        by = _to_float(self.G.nodes[v].get("y"))
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        frac = 0.0 if denom <= 0 else np.clip(((x - ax) * dx + (y - ay) * dy) / denom, 0.0, 1.0)
        return float(ax + frac * dx), float(ay + frac * dy)

    def _edge_length(self, edge):
        """Total geometric length of an edge, cached (used to find the
        remaining distance from a projected point to the edge's end)."""
        length = self._edge_length_cache.get(edge)
        if length is None:
            geometry = self._edge_geometry_cache.get(edge)
            if geometry is None:
                geometry = self._build_edge_geometry(edge, self._edge_attrs(edge))
                self._edge_geometry_cache[edge] = geometry
            length = float(geometry.length)
            self._edge_length_cache[edge] = length
        return length

    def _edge_arclength_of_point(self, edge, x, y):
        """Arc-length offset (from the edge geometry's start) of the point
        on the edge nearest (x, y). Used to place a projected observation
        at its true position along an edge's own geometry, rather than at
        one of the edge's endpoint nodes."""
        if Point is None:
            return 0.0
        geometry = self._edge_geometry_cache.get(edge)
        if geometry is None:
            geometry = self._build_edge_geometry(edge, self._edge_attrs(edge))
            self._edge_geometry_cache[edge] = geometry
        return float(geometry.project(Point(float(x), float(y))))

    def _signal_degradation_probability(self, ping):
        """Infer degradation from observable GNSS quality fields only (HDOP,
        satellite count) — never the simulator's own quality_state label or
        true level, since a real receiver has no access to either."""
        hdop = _to_float(ping.get("hdop"), 1.5)
        satellites = _to_float(ping.get("sat_count"), 7.0)
        logit = 0.8 * (hdop - 2.0) + 0.45 * (6.0 - satellites)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0))))

    def _run_level_probability_ekf(self):
        """Run a causal graph-level filter, then use its soft output in a 2-D CV EKF."""
        n = len(self.df)
        states = np.zeros((n, 4), dtype=float)
        beliefs = np.zeros((n, len(self.levels)), dtype=float)
        if n == 0:
            return states, beliefs
        first = self.df.iloc[0]
        x0 = _to_float(first.get(self.obs_x_col), np.nan)
        y0 = _to_float(first.get(self.obs_y_col), np.nan)
        if not np.isfinite(x0) or not np.isfinite(y0):
            midpoint_array = np.asarray(self.edge_midpoints, dtype=float)
            x0, y0 = np.nanmean(midpoint_array, axis=0)
        state = np.array([x0, y0, 0.0, 0.0], dtype=float)
        covariance = np.diag([2500.0, 2500.0, 400.0, 400.0])
        belief = np.ones(len(self.levels), dtype=float) / max(1, len(self.levels))
        level_to_idx = {level: idx for idx, level in enumerate(self.levels)}
        previous_time = None

        for t, (_, ping) in enumerate(self.df.iterrows()):
            if t > 0 and self._trajectory_boundary(t):
                # A concatenated CSV may contain independent trucks.  Reset
                # every latent filter quantity rather than propagating the
                # preceding vehicle's position, velocity, covariance, and
                # layer belief into the next trajectory.
                reset_x = _to_float(ping.get(self.obs_x_col), np.nan)
                reset_y = _to_float(ping.get(self.obs_y_col), np.nan)
                if not np.isfinite(reset_x) or not np.isfinite(reset_y):
                    midpoint_array = np.asarray(self.edge_midpoints, dtype=float)
                    reset_x, reset_y = np.nanmean(midpoint_array, axis=0)
                state = np.array([reset_x, reset_y, 0.0, 0.0], dtype=float)
                covariance = np.diag([2500.0, 2500.0, 400.0, 400.0])
                belief = np.ones(len(self.levels), dtype=float) / max(1, len(self.levels))
                previous_time = None
            raw_time = _to_float(ping.get("timestamp"), float(t))
            dt = 1.0 if previous_time is None else float(np.clip(raw_time - previous_time, 0.25, 30.0))
            previous_time = raw_time
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
            q = 4.0
            Gq = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]], dtype=float)
            state = F @ state
            covariance = F @ covariance @ F.T + Gq @ (np.eye(2) * q) @ Gq.T

            ox = _to_float(ping.get(self.obs_x_col), np.nan)
            oy = _to_float(ping.get(self.obs_y_col), np.nan)
            hdop = max(0.5, _to_float(ping.get("hdop"), 1.5))
            satellites = _to_float(ping.get("sat_count"), 7.0)
            if np.isfinite(ox) and np.isfinite(oy):
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
                gps_sigma = max(6.0, 7.5 * hdop) * (1.5 if satellites < 5 else 1.0)
                R = np.eye(2) * gps_sigma * gps_sigma
                innovation = np.array([ox, oy]) - H @ state
                S = H @ covariance @ H.T + R
                K = covariance @ H.T @ np.linalg.inv(S)
                state = state + K @ innovation
                covariance = (np.eye(4) - K @ H) @ covariance

            candidates = self._get_candidates(state[0], state[1])
            predicted_belief = belief @ self.level_transition_matrix
            degradation = self._signal_degradation_probability(ping)
            evidence = np.full(len(self.levels), 1e-6, dtype=float)
            nearest_by_level = {}
            sigma = max(20.0, float(np.sqrt(max(covariance[0, 0], covariance[1, 1]))))
            for edge in candidates:
                level = self._edge_layer(edge)
                idx = level_to_idx[level]
                px, py = self._project_to_edge(edge, state[0], state[1])
                distance = float(np.hypot(state[0] - px, state[1] - py))
                attrs = self._edge_attrs(edge)
                is_subterranean = self._edge_tunnel_flag(edge) or level < 0
                signal_likelihood = (0.20 + 0.80 * degradation) if is_subterranean else (1.0 - 0.65 * degradation)
                evidence[idx] += np.exp(-0.5 * (distance / sigma) ** 2) * max(0.05, signal_likelihood)
                if level not in nearest_by_level or distance < nearest_by_level[level][0]:
                    nearest_by_level[level] = (distance, px, py)
            belief = predicted_belief * evidence
            total = float(belief.sum())
            belief = belief / total if total > 0 else np.ones_like(belief) / len(belief)

            # Feed the soft level estimate back to the EKF as a road-constrained pseudo-measurement.
            projections = [(belief[level_to_idx[level]], values) for level, values in nearest_by_level.items()]
            if projections:
                weight_sum = sum(weight for weight, _ in projections)
                road_x = sum(weight * values[1] for weight, values in projections) / max(weight_sum, 1e-9)
                road_y = sum(weight * values[2] for weight, values in projections) / max(weight_sum, 1e-9)
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
                road_sigma = 18.0 + 35.0 * (1.0 - float(np.max(belief)))
                R = np.eye(2) * road_sigma * road_sigma
                S = H @ covariance @ H.T + R
                K = covariance @ H.T @ np.linalg.inv(S)
                state = state + K @ (np.array([road_x, road_y]) - H @ state)
                covariance = (np.eye(4) - K @ H) @ covariance

            states[t] = state
            beliefs[t] = belief
        return states, beliefs

    def _level_ekf_emission_log_prob(self, edge, point_index):
        state = self.level_ekf_states[int(point_index)]
        belief = self.level_probabilities[int(point_index)]
        level_idx = self.levels.index(self._edge_layer(edge))
        px, py = self._project_to_edge(edge, state[0], state[1])
        distance = float(np.hypot(state[0] - px, state[1] - py))
        spatial_lp = -0.5 * (distance / max(self.sigma_xy, 1e-6)) ** 2
        return float(spatial_lp + np.log(np.clip(belief[level_idx], 1e-12, 1.0)))

    def _edge_tunnel_flag(self, edge):
        attrs = self._edge_attrs(edge)
        return _to_flag(attrs.get("tunnel", 0))

    def _resolve_true_edge(self, row):
        """Resolve exact directed truth; never substitute a parallel/reverse edge."""
        edge_u = row.get("edge_u")
        edge_v = row.get("edge_v")
        edge_key = row.get("edge_key", None)

        if pd.notna(edge_u) and pd.notna(edge_v):
            u = _node_id_string(edge_u)
            v = _node_id_string(edge_v)
            # A DiGraph has no parallel-key ambiguity and is represented by
            # canonical key 0 throughout the decoder.
            if not self.G.is_multigraph() and (u, v, 0) in self.edge_ids:
                return (u, v, 0)
            if edge_key is None or pd.isna(edge_key):
                return None
            key_number = _to_float(edge_key, np.nan)
            if not np.isfinite(key_number) or not float(key_number).is_integer():
                return None
            exact = (u, v, int(key_number))
            if exact in self.edges_by_uv.get((u, v), []):
                return exact
            return None

        if self.true_edge_col is not None:
            parsed = _parse_edge_key(row.get(self.true_edge_col))
            if parsed is not None and parsed in self.edges_by_uv.get((parsed[0], parsed[1]), []):
                return parsed

        return None

    def _get_candidates(self, x, y, radius=None):
        if np.isnan(x) or np.isnan(y):
            # With no causal position estimate there is no defensible spatial
            # ordering.  Returning the first K GraphML records makes the
            # result depend on serialization order and guarantees arbitrary
            # errors during a blackout.  Keep all graph states until a finite
            # observation/prediction is available instead.
            return list(self.edge_ids)

        if self.edge_strtree is not None:
            return self._get_candidates_by_geometry(x, y, radius=radius)

        # Legacy fallback (only reachable if Shapely is unavailable): a
        # single per-edge midpoint is a coarser proxy than true edge
        # geometry and can miss long or curved edges within the search
        # radius.
        search_radius = self.candidate_radius if radius is None else float(radius)
        idxs = self.edge_tree.query_ball_point([x, y], r=search_radius)
        if idxs:
            return [self.edge_ids[int(i)] for i in idxs]

        nearest = self.edge_tree.query([x, y], k=min(self.candidate_fallback_k, len(self.edge_ids)))[1]
        nearest = np.atleast_1d(nearest)
        return [self.edge_ids[int(i)] for i in nearest]

    def _get_candidates_by_geometry(self, x, y, radius=None):
        """Radius search against true edge geometry via an STRtree, with a
        true-nearest-distance fallback (not arbitrary graph order) when the
        radius search finds nothing."""
        point = Point(float(x), float(y))
        search_radius = self.candidate_radius if radius is None else float(radius)
        bbox_idxs = self.edge_strtree.query(point.buffer(search_radius))
        matches = [
            int(i) for i in bbox_idxs
            if self.edge_geometries[int(i)].distance(point) <= search_radius
        ]
        if matches:
            return [self.edge_ids[i] for i in matches]

        expand_radius = max(search_radius, 1.0)
        bbox_idxs = np.asarray([], dtype=int)
        for _ in range(10):
            expand_radius *= 2.0
            bbox_idxs = self.edge_strtree.query(point.buffer(expand_radius))
            if len(bbox_idxs) >= self.candidate_fallback_k or len(bbox_idxs) >= len(self.edge_ids):
                break

        if len(bbox_idxs) == 0:
            # Extremely defensive fallback for a malformed/empty STRtree.
            # Rank the complete catalog by actual geometry distance rather
            # than falling back to graph insertion order.
            distances = sorted(
                (geometry.distance(point), i)
                for i, geometry in enumerate(self.edge_geometries)
                if geometry is not None
            )
            return [self.edge_ids[i] for _, i in distances[: self.candidate_fallback_k]]

        distances = sorted((self.edge_geometries[int(i)].distance(point), int(i)) for i in bbox_idxs)
        return [self.edge_ids[i] for _, i in distances[: self.candidate_fallback_k]]

    def _candidate_support_diagnostics(self, x, y, candidates, radius, true_edge):
        """Separate strict-radius support from nearest-edge fallback support."""
        if not np.isfinite(x) or not np.isfinite(y):
            return {
                "strict_radius_contains_true": False,
                "candidate_fallback_used": False,
                "candidate_unlocalized": True,
                "true_edge_observation_distance_m": np.nan,
            }
        point = Point(float(x), float(y))
        search_radius = float(radius)
        candidate_distances = [
            float(self._edge_geometry_cache[edge].distance(point))
            for edge in candidates
            if edge in self._edge_geometry_cache
        ]
        true_distance = np.nan
        if true_edge is not None and true_edge in self._edge_geometry_cache:
            true_distance = float(self._edge_geometry_cache[true_edge].distance(point))
        return {
            "strict_radius_contains_true": bool(
                np.isfinite(true_distance) and true_distance <= search_radius
            ),
            "candidate_fallback_used": bool(
                candidate_distances and min(candidate_distances) > search_radius
            ),
            "candidate_unlocalized": False,
            "true_edge_observation_distance_m": true_distance,
        }

    def _transition_prob(self, prev_edge, curr_edge, mode="classical"):
        return self._transition_prob_with_delta(prev_edge, curr_edge, obs_disp=0.0, mode=mode)

    def _transition_prob_with_delta(self, prev_edge, curr_edge, obs_disp, mode="classical",
                                     prev_xy=None, curr_xy=None, direction_reliability=1.0):
        """Route-distance transition score between two projected positions.

        Same-edge handling: when prev_edge == curr_edge (88-95% of all
        consecutive fixes in this dataset), the end-to-start network
        distance is not on the vehicle's path of travel and is usually
        unreachable in a directed road graph, so it is not used.

        Fixed distance model:
          - same_edge:     signed forward arclength change; backward progress
                           is explicitly penalized rather than hidden by abs()
          - cross-edge:    (remaining distance from prev_xy to prev_edge's end)
                            + shortest_path_length(prev_edge.end, curr_edge.start)
                            + (distance from curr_edge's start to curr_xy)
        both measured against each edge's own road geometry via
        _edge_arclength_of_point, not edge endpoint nodes.
        """
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
                    # Hard graph-reachability constraint: a
                    # candidate with no path at all from the previous edge's
                    # end node is excluded from the transition (probability
                    # 0), not merely down-weighted. This only applies to
                    # genuine cross-edge transitions — same-edge
                    # continuation is never subject to this check.
                    hop = None
                self._transition_hop_cache[pair_key] = hop
            if hop is None:
                return 0.0

        if prev_xy is not None and curr_xy is not None:
            prev_arc = self._edge_arclength_of_point(prev_edge, *prev_xy)
            curr_arc = self._edge_arclength_of_point(curr_edge, *curr_xy)
            if same_edge:
                signed_progress = curr_arc - prev_arc
                dist = max(signed_progress, 0.0)
                reverse_potential = float(np.exp(
                    -direction_reliability * max(-signed_progress, 0.0)
                    / max(self.sigma_d, 1e-6)
                ))
            else:
                remaining_prev = max(self._edge_length(prev_edge) - prev_arc, 0.0)
                dist = remaining_prev + hop + curr_arc
                reverse_potential = 1.0
        else:
            # No projected positions supplied (e.g. legacy/direct callers):
            # fall back to observed displacement for same-edge and to the
            # raw network hop for cross-edge, rather than fabricating a
            # position.
            dist = float(max(obs_disp, 0.0)) if same_edge else hop
            reverse_potential = 1.0

        try:
            # TeX-aligned mismatch term: exp(-|ΔD-Δx|/sigma_d)
            mismatch = abs(dist - float(max(obs_disp, 0.0)))
            course_potential = self._course_potential(
                curr_edge, prev_xy, curr_xy, float(direction_reliability)
            )
            base_prob = float(
                np.exp(-(mismatch / max(self.sigma_d, 1e-6)))
                * reverse_potential
                * course_potential
            )

            # QTS_TRANSITION_MODES: QMM (emission) and QTS
            # (transition) can be toggled independently, so the full
            # ablation grid (HMM / HMM+QMM / HMM+QTS / HMM+QMM+QTS) is
            # reachable via run_viterbi(mode=...). dist depends on the
            # projected position along each edge (not just the edge pair),
            # so scores are not cached: a rounded-distance cache key would
            # quantize dist into bins and make the scored value diverge from
            # the exact continuous distance used during training.
            if mode in ("quantum", "qts_only", "level_ekf_hmm_quantum"):
                qts_prob = self.transition_scorer.score(
                    self._edge_attrs(prev_edge),
                    self._edge_attrs(curr_edge),
                    dist,
                )
                fused = base_prob * qts_prob
                return float(np.clip(fused, 1e-9, 1.0))

            if mode == "classical_matched":
                matched_prob = self.classical_matched_transition_scorer.score(
                    self._edge_attrs(prev_edge),
                    self._edge_attrs(curr_edge),
                    dist,
                )
                fused = base_prob * matched_prob
                return float(np.clip(fused, 1e-9, 1.0))

            return float(np.clip(base_prob, 1e-9, 1.0))
        except (TypeError, ValueError):
            return 1e-6

    def _spatial_log_prob(self, edge, ping, sigma=None):
        sigma_val = self.sigma_xy if sigma is None else float(sigma)
        # Weight the measurement term by receiver-observable uncertainty.
        # Imputed blackout states and confidence-dip periods must not be
        # scored as ordinary fixed-variance GNSS fixes.
        hdop = max(0.5, _to_float(ping.get("hdop"), 1.5))
        satellites = _to_float(ping.get("sat_count"), 7.0)
        quality_scale = float(np.clip(hdop / 1.5, 0.75, 3.0))
        if satellites < 5:
            quality_scale *= 1.35
        if _to_flag(ping.get("_position_imputed", False)):
            quality_scale = max(quality_scale, 2.5)
        quality_scale *= max(1.0, _to_float(ping.get("_ambiguity_scale"), 1.0))
        sigma_val *= quality_scale
        ox = _to_float(ping[self.obs_x_col], np.nan)
        oy = _to_float(ping[self.obs_y_col], np.nan)
        if np.isnan(ox) or np.isnan(oy):
            return float(np.log(1e-6))
        ex, ey = self._project_to_edge(edge, ox, oy)
        dist = float(np.hypot(ox - ex, oy - ey))
        return float(-0.5 * (dist / max(sigma_val, 1e-6)) ** 2)

    def _observation_displacement(self, prev_ping, curr_ping):
        px = _to_float(prev_ping.get(self.obs_x_col), np.nan)
        py = _to_float(prev_ping.get(self.obs_y_col), np.nan)
        cx = _to_float(curr_ping.get(self.obs_x_col), np.nan)
        cy = _to_float(curr_ping.get(self.obs_y_col), np.nan)
        if np.isnan(px) or np.isnan(py) or np.isnan(cx) or np.isnan(cy):
            return 0.0
        return float(np.hypot(cx - px, cy - py))

    def _predict_observation(self, prev_prev_ping, prev_ping, curr_ping):
        """Causally fill a missing fix from the *effective* prior states.

        Callers pass previously imputed rows, not raw dataframe rows, so
        constant-velocity propagation continues through arbitrary blackout
        lengths. Timestamp deltas are respected; with only one valid prior
        state the causal estimate is a zero-velocity hold.
        """
        pred = curr_ping.copy()
        pred["_position_imputed"] = False
        cx = _to_float(curr_ping.get(self.obs_x_col), np.nan)
        cy = _to_float(curr_ping.get(self.obs_y_col), np.nan)
        if np.isfinite(cx) and np.isfinite(cy):
            return pred
        if prev_ping is None:
            return pred

        p1x = _to_float(prev_ping.get(self.obs_x_col), np.nan)
        p1y = _to_float(prev_ping.get(self.obs_y_col), np.nan)
        if not (np.isfinite(p1x) and np.isfinite(p1y)):
            return pred

        vx = 0.0
        vy = 0.0
        if prev_prev_ping is not None:
            p2x = _to_float(prev_prev_ping.get(self.obs_x_col), np.nan)
            p2y = _to_float(prev_prev_ping.get(self.obs_y_col), np.nan)
            if np.isfinite(p2x) and np.isfinite(p2y):
                t2 = _to_float(prev_prev_ping.get("timestamp"), np.nan)
                t1 = _to_float(prev_ping.get("timestamp"), np.nan)
                dt_prev = t1 - t2 if np.isfinite(t1) and np.isfinite(t2) else 1.0
                dt_prev = float(np.clip(dt_prev, 0.25, 30.0))
                vx = (p1x - p2x) / dt_prev
                vy = (p1y - p2y) / dt_prev

        t1 = _to_float(prev_ping.get("timestamp"), np.nan)
        tc = _to_float(curr_ping.get("timestamp"), np.nan)
        dt_curr = tc - t1 if np.isfinite(tc) and np.isfinite(t1) else 1.0
        dt_curr = float(np.clip(dt_curr, 0.25, 30.0))
        if not np.isfinite(cx):
            pred[self.obs_x_col] = p1x + vx * dt_curr
        if not np.isfinite(cy):
            pred[self.obs_y_col] = p1y + vy * dt_curr
        pred["_position_imputed"] = True
        return pred

    def _trajectory_boundary(self, index):
        """Return true when ``index`` starts an independent trajectory."""
        if index <= 0:
            return True
        row = self.df.iloc[index]
        prev = self.df.iloc[index - 1]
        if "truck_id" in self.df.columns and str(row.get("truck_id")) != str(prev.get("truck_id")):
            return True
        if "timestamp" in self.df.columns:
            curr_t = _to_float(row.get("timestamp"), np.nan)
            prev_t = _to_float(prev.get("timestamp"), np.nan)
            if np.isfinite(curr_t) and np.isfinite(prev_t) and curr_t <= prev_t:
                return True
        return False

    def _build_effective_observations(self):
        """Build one causal observed/imputed row per fix, reset per truck."""
        effective = []
        history = []
        for index in range(len(self.df)):
            if self._trajectory_boundary(index):
                history = []
            raw = self.df.iloc[index].copy()
            raw["_position_imputed"] = False
            prev_ping = history[-1] if history else None
            prev_prev_ping = history[-2] if len(history) >= 2 else None
            filled = self._predict_observation(prev_prev_ping, prev_ping, raw)
            effective.append(filled)
            history.append(filled)
        return effective

    def _observation_direction_reliability(self, prev_ping, curr_ping, obs_disp):
        """Weight direction evidence using observable GNSS quality and motion."""
        hdop = max(0.5, _to_float(curr_ping.get("hdop"), 1.5))
        satellites = _to_float(curr_ping.get("sat_count"), 7.0)
        hdop_term = float(np.exp(-max(hdop - 1.0, 0.0) / 3.0))
        satellite_term = float(np.clip(satellites / 8.0, 0.0, 1.0))
        motion_term = float(np.clip(obs_disp / 8.0, 0.0, 1.0))
        imputed_term = 0.25 if (
            _to_flag(prev_ping.get("_position_imputed", False))
            or _to_flag(curr_ping.get("_position_imputed", False))
        ) else 1.0
        return float(np.clip(hdop_term * satellite_term * motion_term * imputed_term, 0.0, 1.0))

    def _edge_forward_tangent(self, edge, x, y):
        """Unit tangent in the directed edge's forward WKT orientation."""
        geometry = self._edge_geometry_cache.get(edge)
        if geometry is None or Point is None or geometry.is_empty:
            return None
        length = float(geometry.length)
        if length <= 1e-9:
            return None
        arc = float(geometry.project(Point(float(x), float(y))))
        delta = min(5.0, max(0.5, length * 0.02))
        lo = max(0.0, arc - delta)
        hi = min(length, arc + delta)
        if hi - lo <= 1e-9:
            lo, hi = 0.0, length
        p0 = geometry.interpolate(lo)
        p1 = geometry.interpolate(hi)
        dx = float(p1.x - p0.x)
        dy = float(p1.y - p0.y)
        norm = float(np.hypot(dx, dy))
        if norm <= 1e-9:
            return None
        return dx / norm, dy / norm

    def _course_potential(self, edge, prev_xy, curr_xy, reliability):
        """Unnormalized likelihood of observed course under edge direction."""
        if prev_xy is None or curr_xy is None or reliability <= 0.0:
            return 1.0
        dx = float(curr_xy[0] - prev_xy[0])
        dy = float(curr_xy[1] - prev_xy[1])
        norm = float(np.hypot(dx, dy))
        tangent = self._edge_forward_tangent(edge, *curr_xy)
        if norm <= 1e-9 or tangent is None:
            return 1.0
        cosine = float(np.clip((dx * tangent[0] + dy * tangent[1]) / norm, -1.0, 1.0))
        angle = float(np.arccos(cosine))
        sigma_angle = np.deg2rad(45.0)
        return float(np.exp(-0.5 * reliability * (angle / sigma_angle) ** 2))

    def _max_qmm_expectation(self, candidates, ping):
        if not candidates:
            return -1.0
        vals = []
        for edge in candidates:
            q_raw = self._cached_quantum_score(
                self.quantum,
                "entangled",
                edge,
                ping.get("time_qubit_val", 0.5),
            )
            vals.append(float(np.clip(q_raw, -1.0, 1.0)))
        return max(vals) if vals else -1.0

    def _cached_quantum_score(self, model, model_name, edge, traffic_val):
        traffic_key = float(_to_float(traffic_val, 0.5))
        cache_key = (str(model_name), edge, traffic_key)
        if cache_key not in self._quantum_score_cache:
            self._quantum_score_cache[cache_key] = model.score(
                self._edge_attrs(edge),
                traffic_key,
            )
        return self._quantum_score_cache[cache_key]

    def _edge_supervised_data(self):
        """Build fix-level labels over inference-style local candidate sets.

        Labels follow the downstream task of selecting the true edge among
        candidates at a particular fix, not a route-coverage prior. Each
        retained fix contributes its true candidate and up to four local
        false candidates, with truck IDs carried into grouped validation.
        """
        training = self.classical_training_df.copy()
        if training.empty:
            return None, None, None
        x_col, y_col = _pick_xy_columns(
            training,
            ["obs_x", "x_noisy", "x", "lon"],
            ["obs_y", "y_noisy", "y", "lat"],
            "classical-training observation",
        )
        features = []
        labels = []
        groups = []
        audit = {
            "input_fixes": 0,
            "valid_inference_observations": 0,
            "resolved_true_edges": 0,
            "candidate_oracle_hits": 0,
            "candidate_oracle_denominator": 0,
            "candidate_oracle_recall": 0.0,
            "retained_fixes": 0,
            "training_examples": 0,
            "skipped_missing_inference_observation": 0,
            "skipped_unresolved_true_edge": 0,
            "skipped_true_not_in_live_candidates": 0,
            "skipped_no_live_negative": 0,
        }
        if "truck_id" in training.columns:
            grouped = training.groupby("truck_id", sort=False)
        else:
            grouped = [("single_trajectory", training)]
        for truck_id, track in grouped:
            if "timestamp" in track.columns:
                track = track.sort_values("timestamp")
            history = []
            for _, raw in track.iterrows():
                audit["input_fixes"] += 1
                row = raw.copy()
                # The standard trajectory schema uses the same observation
                # columns as the benchmark. Normalize aliases if necessary.
                if x_col != self.obs_x_col:
                    row[self.obs_x_col] = row.get(x_col)
                if y_col != self.obs_y_col:
                    row[self.obs_y_col] = row.get(y_col)
                previous = history[-1] if history else None
                previous_previous = history[-2] if len(history) >= 2 else None
                row = self._predict_observation(previous_previous, previous, row)
                history.append(row)
                x = _to_float(row.get(self.obs_x_col), np.nan)
                y = _to_float(row.get(self.obs_y_col), np.nan)
                if not (np.isfinite(x) and np.isfinite(y)):
                    audit["skipped_missing_inference_observation"] += 1
                    continue
                audit["valid_inference_observations"] += 1
                true_edge = self._resolve_true_edge(raw)
                if true_edge is None:
                    audit["skipped_unresolved_true_edge"] += 1
                    continue
                audit["resolved_true_edges"] += 1
                audit["candidate_oracle_denominator"] += 1
                candidates = self._get_candidates(x, y)
                if true_edge not in candidates:
                    # No classifier can recover a state absent from its live
                    # candidate set; candidate-oracle recall reports this
                    # separately in Step07.
                    audit["skipped_true_not_in_live_candidates"] += 1
                    continue
                audit["candidate_oracle_hits"] += 1
                negatives = [edge for edge in candidates if edge != true_edge]
                if not negatives:
                    audit["skipped_no_live_negative"] += 1
                    continue
                audit["retained_fixes"] += 1
                traffic = row.get("time_qubit_val", 0.5)
                features.append(extract_9_features(
                    self._edge_attrs(true_edge), traffic_regime=traffic
                ))
                labels.append(1)
                groups.append(str(truck_id))
                negatives.sort(key=lambda edge: np.hypot(
                    x - self._project_to_edge(edge, x, y)[0],
                    y - self._project_to_edge(edge, x, y)[1],
                ))
                for edge in negatives[:4]:
                    features.append(extract_9_features(
                        self._edge_attrs(edge), traffic_regime=traffic
                    ))
                    labels.append(0)
                    groups.append(str(truck_id))
        audit["training_examples"] = int(len(features))
        denominator = int(audit["candidate_oracle_denominator"])
        audit["candidate_oracle_recall"] = (
            float(audit["candidate_oracle_hits"] / denominator) if denominator else 0.0
        )
        self._ablation_candidate_audit = audit
        X = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int)
        groups = np.asarray(groups, dtype=str)
        if len(X) < 8 or np.unique(y).size < 2:
            return None, None, None
        return X, y, groups

    def _validation_split(self):
        X, y, groups = self._edge_supervised_data()
        if X is None:
            return None
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            rng = np.random.default_rng(self.seed + 17011)
            shuffled = rng.permutation(unique_groups)
            n_validation = min(
                max(1, int(round(0.2 * len(shuffled)))), len(shuffled) - 1
            )
            validation_groups = set(shuffled[:n_validation])
            validation_mask = np.asarray([group in validation_groups for group in groups])
            fit_idx = np.flatnonzero(~validation_mask)
            val_idx = np.flatnonzero(validation_mask)
            if np.unique(y[fit_idx]).size >= 2 and np.unique(y[val_idx]).size >= 2:
                # Bound tuning cost after the truck-disjoint split.
                if len(fit_idx) > 2560:
                    fit_idx = np.sort(rng.choice(fit_idx, size=2560, replace=False))
                if len(val_idx) > 1280:
                    val_idx = np.sort(rng.choice(val_idx, size=1280, replace=False))
                self._ablation_split_metadata = {
                    "supervision": "fix-level correctness within inference-style local candidate sets",
                    "split": "truck-disjoint",
                    "fit_examples": int(len(fit_idx)),
                    "validation_examples": int(len(val_idx)),
                    "fit_trucks": int(len(set(groups[fit_idx]))),
                    "validation_trucks": int(len(set(groups[val_idx]))),
                    "candidate_audit": getattr(self, "_ablation_candidate_audit", {}),
                }
                return X[fit_idx], X[val_idx], y[fit_idx], y[val_idx]
        # Small custom datasets without two usable groups retain a stratified
        # fallback; production Step06/Step07 trajectories use truck grouping.
        self._ablation_split_metadata = {
            "supervision": "fix-level correctness within inference-style local candidate sets",
            "split": "sample-stratified fallback (insufficient truck groups)",
            "candidate_audit": getattr(self, "_ablation_candidate_audit", {}),
        }
        return train_test_split(X, y, test_size=0.2, random_state=self.seed, stratify=y)

    def _train_ablation_mlp(self):
        split = self._validation_split()
        if split is None:
            return None, {}
        X_fit, X_val, y_fit, y_val = split
        configs = [
            ((16,), 1e-4), ((32,), 1e-4), ((32, 16), 1e-4),
            ((64, 32), 1e-3), ((64, 32, 16), 1e-3),
        ]
        trials = []
        best = None
        for trial, (layers, alpha) in enumerate(configs, start=1):
            model = MLPClassifier(
                hidden_layer_sizes=layers, alpha=alpha, max_iter=500,
                early_stopping=True, random_state=self.seed + trial,
            )
            model.fit(X_fit, y_fit)
            score = balanced_accuracy_score(y_val, model.predict(X_val))
            trials.append({"trial": trial, "layers": str(layers), "alpha": alpha,
                           "validation_balanced_accuracy": float(score)})
            if best is None or score > best[0]:
                best = (score, model, trials[-1])
        return best[1], {
            "selected": best[2], "trials": trials, "budget": len(configs),
            **getattr(self, "_ablation_split_metadata", {}),
        }

    def _train_budget_matched_svm(self):
        split = self._validation_split()
        if split is None:
            return None, {}
        X_fit, X_val, y_fit, y_val = split
        configs = [(0.5, "scale"), (1.0, "scale"), (5.0, "scale"),
                   (1.0, 0.5), (5.0, 0.5)]
        trials = []
        best = None
        for trial, (c_value, gamma) in enumerate(configs, start=1):
            model = SVC(C=c_value, gamma=gamma, kernel="rbf", probability=True,
                        class_weight="balanced", random_state=self.seed + trial)
            model.fit(X_fit, y_fit)
            score = balanced_accuracy_score(y_val, model.predict(X_val))
            trials.append({"trial": trial, "C": c_value, "gamma": str(gamma),
                           "validation_balanced_accuracy": float(score)})
            if best is None or score > best[0]:
                best = (score, model, trials[-1])
        return best[1], {
            "selected": best[2], "trials": trials, "budget": len(configs),
            **getattr(self, "_ablation_split_metadata", {}),
        }

    def _ablation_training_data(self):
        true_edges = {
            edge for _, row in self.classical_training_df.iterrows()
            for edge in [self._resolve_true_edge(row)] if edge is not None
        }

        sequences = []
        labels = []
        for edge in self.edge_ids:
            attrs = self._edge_attrs(edge)
            sequences.append([
                extract_9_features(attrs, traffic_regime=0.0),
                extract_9_features(attrs, traffic_regime=0.5),
                extract_9_features(attrs, traffic_regime=1.0),
            ])
            labels.append(1.0 if edge in true_edges else 0.0)

        if labels and not any(labels):
            rng = np.random.default_rng(self.seed)
            labels = [1.0 if rng.random() > 0.8 else 0.0 for _ in labels]
        return sequences, labels

    def _train_ablation_lstm(self):
        sequences, labels = self._ablation_training_data()
        if not sequences:
            return None
        return LSTMEdgeClassifier(seed=self.seed + 31).fit(sequences, labels)

    def _emission_log_prob(self, mode, edge, ping, point_index=None):
        spatial_lp = self._spatial_log_prob(edge, ping)

        if mode in ("level_ekf_hmm", "level_ekf_hmm_quantum"):
            if point_index is None:
                raise ValueError("level_ekf_hmm emission requires a point index")
            base_lp = self._level_ekf_emission_log_prob(edge, point_index)
            if mode == "level_ekf_hmm_quantum":
                # C4: the only baseline with a motion model
                # AND a layer posterior is otherwise never given the QMM/QTS
                # factors, so its dominance over ST-QMM is uninformative
                # about whether quantum scoring helps. Scale its emission by
                # the same trained QMM factor ST-QMM uses.
                q_raw = self._cached_quantum_score(
                    self.quantum, "entangled", edge, ping.get("time_qubit_val", 0.5)
                )
                quantum_prob = _to_prob_from_expectation(q_raw)
                return float(base_lp + np.log(np.clip(quantum_prob, 1e-12, 1.0)))
            return base_lp

        # "qts_only" deliberately falls through to the bare spatial_lp return
        # at the bottom of this function — it uses the QTS transition factor
        # (see _transition_prob_with_delta) but no learned emission factor,
        # giving the HMM+QTS-only ablation cell.
        if mode in {"quantum", "qmm_entanglement", "qmm_no_entanglement", "frozen"}:
            model = {
                "qmm_no_entanglement": self.quantum_no_entanglement,
                "frozen": self.quantum_frozen,
            }.get(mode, self.quantum)
            model_name = {"qmm_no_entanglement": "no_entanglement", "frozen": "frozen"}.get(mode, "entangled")
            q_raw = self._cached_quantum_score(
                model,
                model_name,
                edge,
                ping.get("time_qubit_val", 0.5),
            )
            quantum_prob = _to_prob_from_expectation(q_raw)
            # TeX-aligned emission: Gaussian spatial term scaled by QMM confidence.
            gauss_prob = float((1.0 / (2.0 * np.pi * max(self.sigma_xy, 1e-6) ** 2)) * np.exp(spatial_lp))
            return float(np.log(np.clip(gauss_prob * quantum_prob, 1e-12, 1.0)))

        if mode == "classical_matched":
            # Matched classical control: identical fusion
            # shape to the quantum emission branch above, scored by the
            # classical MLP trained with the same procedure instead of a
            # circuit.
            matched_prob = self._cached_quantum_score(
                self.classical_matched,
                "classical_matched",
                edge,
                ping.get("time_qubit_val", 0.5),
            )
            matched_prob = float(np.clip(matched_prob, 1e-9, 1.0))
            gauss_prob = float((1.0 / (2.0 * np.pi * max(self.sigma_xy, 1e-6) ** 2)) * np.exp(spatial_lp))
            return float(np.log(np.clip(gauss_prob * matched_prob, 1e-12, 1.0)))

        if mode == "mlp" and self.mlp_model is not None:
            feats = extract_9_features(self._edge_attrs(edge), traffic_regime=ping.get("time_qubit_val", 0.5))
            if hasattr(self.mlp_model, "predict_proba") and len(getattr(self.mlp_model, "classes_", [])) == 2:
                pred = float(self.mlp_model.predict_proba([feats])[0][1])
            else:
                pred = float(self.mlp_model.predict([feats])[0])
            pred = float(np.clip(pred, 1e-9, 1.0))
            mlp_lp = np.log(pred)
            gauss_prob = float((1.0 / (2.0 * np.pi * max(self.sigma_xy, 1e-6) ** 2)) * np.exp(spatial_lp))
            return float(np.log(np.clip(gauss_prob * np.exp(mlp_lp), 1e-12, 1.0)))

        if mode == "svm" and self.svm_model is not None:
            feats = extract_9_features(self._edge_attrs(edge), traffic_regime=ping.get("time_qubit_val", 0.5))
            pred = float(self.svm_model.predict_proba([feats])[0][1])
            gauss_prob = float((1.0 / (2.0 * np.pi * max(self.sigma_xy, 1e-6) ** 2)) * np.exp(spatial_lp))
            return float(np.log(np.clip(gauss_prob * np.clip(pred, 1e-9, 1.0), 1e-12, 1.0)))

        if mode == "lstm" and self.lstm_model is not None:
            traffic = float(np.clip(ping.get("time_qubit_val", 0.5), 0.0, 1.0))
            attrs = self._edge_attrs(edge)
            sequence = [
                extract_9_features(attrs, traffic_regime=max(0.0, traffic - 0.1)),
                extract_9_features(attrs, traffic_regime=traffic),
            ]
            recurrent_prob = self.lstm_model.predict_proba(sequence)
            gauss_prob = float((1.0 / (2.0 * np.pi * max(self.sigma_xy, 1e-6) ** 2)) * np.exp(spatial_lp))
            return float(np.log(np.clip(gauss_prob * recurrent_prob, 1e-12, 1.0)))

        if mode == "hmm_extra":
            attrs = self._edge_attrs(edge)
            features = extract_9_features(attrs, traffic_regime=ping.get("time_qubit_val", 0.5))
            # Engineered road-context prior: speed, tunnel/bridge, layer,
            # lanes, traffic regime, and junction features augment the HMM.
            context = np.asarray(features[2:], dtype=float)
            context_prob = float(np.clip(0.25 + 0.75 * np.mean(context), 1e-9, 1.0))
            return float(spatial_lp + np.log(context_prob))

        return spatial_lp

    def _checkpoint_path(self, mode):
        if not self.checkpoint_id:
            return None
        return REPORTS_DIR / f"step05_{self.checkpoint_id}_{mode}_checkpoint.pkl"

    def _checkpoint_signature(self):
        digest = hashlib.sha256()
        digest.update(np.asarray(self.quantum.weights, dtype=float).tobytes())
        digest.update(np.asarray(self.quantum_no_entanglement.weights, dtype=float).tobytes())
        digest.update(np.asarray(self.transition_scorer.weights, dtype=float).tobytes())
        digest.update(
            repr(
                (
                    len(self.df), self.seed, self.candidate_radius,
                    self.candidate_fallback_k,
                    self.sigma_xy, self.sigma_d, self.confidence_dip_threshold,
                    self.kappa_min, self.kappa_max, self.beam_width,
                )
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def _save_viterbi_checkpoint(self, mode, total_steps, next_t, path_scores, backpointers,
                                 ambiguity_mode, freeze_anchor, adaptive_window,
                                 effective_observations, step_diagnostics):
        checkpoint_path = self._checkpoint_path(mode)
        if checkpoint_path is None:
            return
        payload = {
            # v5 also records strict-radius/fallback/post-beam candidate
            # diagnostics and the fair diagnostic-only confidence behavior.
            # v6: transition scores use exact continuous distances, so
            # v5 checkpoints are incompatible and must not be resumed from.
            "version": 6,
            "signature": self._checkpoint_signature(),
            "mode": mode,
            "total_steps": total_steps,
            "next_t": next_t,
            "path_scores": path_scores,
            "backpointers": backpointers,
            "ambiguity_mode": ambiguity_mode,
            "freeze_anchor": freeze_anchor,
            "adaptive_window": adaptive_window,
            "trellis_breaks": list(self._trellis_breaks),
            "connectivity_reanchors": list(self._connectivity_reanchors),
            "effective_observations": effective_observations,
            "step_diagnostics": step_diagnostics,
            "quantum_score_cache": self._quantum_score_cache,
            "transition_hop_cache": self._transition_hop_cache,
        }
        temp_path = checkpoint_path.with_suffix(".pkl.tmp")
        with open(temp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.replace(checkpoint_path)

    def run_viterbi(self, mode="quantum", progress_every=100):
        valid_modes = {
            "classical", "level_ekf_hmm", "hmm_extra", "mlp", "svm", "lstm",
            "qmm_no_entanglement", "qmm_entanglement", "quantum",
            "qts_only", "frozen", "classical_matched", "level_ekf_hmm_quantum",
        }
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(valid_modes)}.")

        total_steps = len(self.df)
        if total_steps < 2:
            raise RuntimeError("Need at least 2 points for Viterbi evaluation.")

        # Timesteps where the trellis was re-anchored due to total
        # connectivity loss (see the recovery strategy below) — exposed for
        # diagnostics rather than left as a silent event.
        self._trellis_breaks = []
        self._connectivity_reanchors = []

        checkpoint_path = self._checkpoint_path(mode)
        state = None
        if checkpoint_path is not None and checkpoint_path.exists():
            try:
                with open(checkpoint_path, "rb") as handle:
                    candidate_state = pickle.load(handle)
                if (
                    candidate_state.get("version") == 6
                    and candidate_state.get("signature") == self._checkpoint_signature()
                    and candidate_state.get("mode") == mode
                    and candidate_state.get("total_steps") == total_steps
                ):
                    state = candidate_state
            except Exception as exc:
                print(f"Ignoring unreadable checkpoint {checkpoint_path}: {exc}", flush=True)

        if state is not None:
            path_scores = state["path_scores"]
            backpointers = state["backpointers"]
            ambiguity_mode = bool(state["ambiguity_mode"])
            freeze_anchor = int(state["freeze_anchor"])
            adaptive_window = int(state["adaptive_window"])
            self._trellis_breaks = list(state.get("trellis_breaks", []))
            self._connectivity_reanchors = list(state.get("connectivity_reanchors", []))
            effective_observations = state.get("effective_observations")
            if not effective_observations or len(effective_observations) != total_steps:
                effective_observations = self._build_effective_observations()
            step_diagnostics = list(state.get("step_diagnostics", []))
            self._quantum_score_cache.update(state.get("quantum_score_cache", {}))
            self._transition_hop_cache.update(state.get("transition_hop_cache", {}))
            start_t = int(state["next_t"])
            print(f"[{mode}] resumed from checkpoint at step {start_t}/{total_steps - 1}", flush=True)
        else:
            effective_observations = self._build_effective_observations()
            step_diagnostics = []
            path_scores = []
            backpointers = []
            p0 = effective_observations[0]
            init_candidates = self._get_candidates(_to_float(p0[self.obs_x_col], np.nan), _to_float(p0[self.obs_y_col], np.nan))
            init_probs = {}
            for edge in init_candidates:
                init_probs[edge] = self._emission_log_prob(mode, edge, p0, point_index=0)
            path_scores.append(init_probs)
            ambiguity_mode = False
            freeze_anchor = 0
            adaptive_window = self.kappa_min
            start_t = 1
            true0 = self._resolve_true_edge(self.df.iloc[0])
            support0 = self._candidate_support_diagnostics(
                _to_float(p0[self.obs_x_col], np.nan),
                _to_float(p0[self.obs_y_col], np.nan),
                init_candidates,
                self.candidate_radius,
                true0,
            )
            step_diagnostics.append({
                "point_index": 0,
                "candidate_count": len(init_candidates),
                "candidate_contains_true": bool(true0 in init_candidates) if true0 is not None else False,
                "position_imputed": _to_flag(p0.get("_position_imputed", False)),
                "ambiguity_mode": False,
                "trellis_break": False,
                "connectivity_reanchor": False,
                "trajectory_boundary": True,
                "beam_contains_true": bool(true0 in init_probs) if true0 is not None else False,
                "effective_x": _to_float(p0.get(self.obs_x_col), np.nan),
                "effective_y": _to_float(p0.get(self.obs_y_col), np.nan),
                **support0,
            })

        for t in range(start_t, total_steps):
            ping = effective_observations[t].copy()
            prev_ping = effective_observations[t - 1]
            trajectory_boundary = self._trajectory_boundary(t)
            if trajectory_boundary:
                ambiguity_mode = False
                freeze_anchor = t
                adaptive_window = self.kappa_min

            # QMM confidence is retained as a diagnostic only.  Changing the
            # candidate radius or emission variance for the full quantum mode
            # alone confounds the QMM/QTS 2x2 ablation and the comparison with
            # classical controls.  Every headline mode therefore receives the
            # same candidate support and GNSS-quality-based emission scaling.
            qmm_z = -1.0
            if mode == "quantum":
                probe_candidates = self._get_candidates(_to_float(ping[self.obs_x_col], np.nan), _to_float(ping[self.obs_y_col], np.nan))
                qmm_z = self._max_qmm_expectation(probe_candidates, ping)
                if qmm_z <= self.confidence_dip_threshold:
                    if not ambiguity_mode:
                        freeze_anchor = max(0, t - 1)
                    ambiguity_mode = True
                    adaptive_window = min(self.kappa_max, adaptive_window + 1)
                elif ambiguity_mode:
                    ambiguity_mode = False
                    adaptive_window = self.kappa_min
                    freeze_anchor = t

            candidate_scale = 1.0
            candidates = self._get_candidates(
                _to_float(ping[self.obs_x_col], np.nan),
                _to_float(ping[self.obs_y_col], np.nan),
                radius=self.candidate_radius * candidate_scale,
            )
            obs_disp = self._observation_displacement(prev_ping, ping)
            direction_reliability = self._observation_direction_reliability(
                prev_ping, ping, obs_disp
            )

            # Projected observation positions used to place the vehicle at
            # its true along-edge location (not an edge endpoint node) for
            # transition-distance scoring.
            prev_obs_x = _to_float(prev_ping.get(self.obs_x_col), np.nan)
            prev_obs_y = _to_float(prev_ping.get(self.obs_y_col), np.nan)
            curr_obs_x = _to_float(ping.get(self.obs_x_col), np.nan)
            curr_obs_y = _to_float(ping.get(self.obs_y_col), np.nan)
            have_xy = np.isfinite(prev_obs_x) and np.isfinite(prev_obs_y) and np.isfinite(curr_obs_x) and np.isfinite(curr_obs_y)

            new_probs = {}
            new_ptrs = {}
            prev_items = list(path_scores[t - 1].items())
            if trajectory_boundary:
                # Independent tracks must never share a Viterbi transition or
                # a dead-reckoning state merely because their rows are
                # consecutive in one CSV.
                prev_items = []
            if self.beam_width is not None and len(prev_items) > self.beam_width:
                prev_items = sorted(prev_items, key=lambda kv: kv[1], reverse=True)[: self.beam_width]

            # Unnormalized pairwise transition potentials.  Normalizing over
            # the *current noisy candidate set* made an edge-pair score change
            # whenever an unrelated candidate entered or left the radius and
            # rewarded predecessors with fewer reachable alternatives.
            trans_potentials = {}
            for prev_edge, _prev_score in prev_items:
                raw_vals = {}
                prev_xy = (prev_obs_x, prev_obs_y) if have_xy else None
                for curr_edge in candidates:
                    # QTS/matched-transition factor is applied only for modes
                    # that request it; QMM-only ablation modes (e.g.
                    # "qmm_entanglement", "frozen") intentionally fall back
                    # to the plain classical transition.
                    transition_mode = mode if mode in TRANSITION_SCORED_VITERBI_MODES else "classical"
                    curr_xy = (curr_obs_x, curr_obs_y) if have_xy else None
                    raw = self._transition_prob_with_delta(
                        prev_edge, curr_edge, obs_disp=obs_disp, mode=transition_mode,
                        prev_xy=prev_xy, curr_xy=curr_xy,
                        direction_reliability=direction_reliability,
                    )
                    # Preserve an exact 0.0 (graph-unreachable)
                    # instead of flooring it to a nonzero value — only
                    # clip the upper bound, so an unreachable candidate
                    # remains excluded from the unnormalized potential.
                    raw = float(np.clip(raw, 0.0, 1.0))
                    raw_vals[curr_edge] = raw
                trans_potentials[prev_edge] = raw_vals

            for curr_edge in candidates:
                emit_lp = self._emission_log_prob(mode, curr_edge, ping, point_index=t)
                best_prev = None
                best_score = -float("inf")

                for prev_edge, prev_score in prev_items:
                    trans = trans_potentials.get(prev_edge, {}).get(curr_edge, 0.0)
                    if trans <= 0.0:
                        # Graph-unreachable: a true hard
                        # constraint, not merely a steep penalty. This
                        # (prev_edge, curr_edge) pair is never selectable
                        # while any graph-reachable predecessor exists for
                        # curr_edge, regardless of how much cumulative score
                        # a disconnected path has otherwise accumulated —
                        # a finite penalty here is not enough, since a
                        # sufficiently strong emission match can and does
                        # outweigh any fixed log-probability gap over a long
                        # enough trajectory.
                        continue
                    score = prev_score + np.log(trans) + emit_lp
                    if score > best_score:
                        best_score = score
                        best_prev = prev_edge

                if best_prev is not None:
                    new_probs[curr_edge] = best_score
                    new_ptrs[curr_edge] = best_prev

            if trajectory_boundary:
                # Intentional track reset: initialize the new trajectory from
                # its own emissions.  It is a backtracking segment boundary,
                # not a connectivity-loss reanchor.
                for curr_edge in candidates:
                    new_probs[curr_edge] = self._emission_log_prob(
                        mode, curr_edge, ping, point_index=t
                    )
                new_ptrs = {edge: edge for edge in new_probs.keys()}
                self._trellis_breaks.append(t)
            elif not new_probs:
                # Total connectivity loss: every
                # current candidate is graph-unreachable from every
                # surviving predecessor in the beam (e.g. a large GPS gap
                # or a candidate-radius miss). Copying t-1's score
                # distribution forward would discard this timestep's emission
                # evidence and freeze the path at a stale edge set, so re-anchor the
                # trellis at the CURRENT candidates using only their
                # emission score (no incoming transition term) — the same
                # rule used to initialize the trellis at t=0 — so decoding
                # tracks the live observation instead of a stale position.
                # Self-pointing backpointers mark this as a documented
                # break in the decoded path rather than a real transition.
                for curr_edge in candidates:
                    new_probs[curr_edge] = self._emission_log_prob(mode, curr_edge, ping, point_index=t)
                new_ptrs = {edge: edge for edge in new_probs.keys()}
                self._trellis_breaks.append(t)
                self._connectivity_reanchors.append(t)

            if self.beam_width is not None and len(new_probs) > self.beam_width:
                top_items = sorted(new_probs.items(), key=lambda kv: kv[1], reverse=True)[: self.beam_width]
                keep_edges = {edge for edge, _ in top_items}
                new_probs = {edge: score for edge, score in top_items}
                new_ptrs = {edge: ptr for edge, ptr in new_ptrs.items() if edge in keep_edges}

            path_scores.append(new_probs)
            backpointers.append(new_ptrs)
            true_edge = self._resolve_true_edge(self.df.iloc[t])
            support = self._candidate_support_diagnostics(
                _to_float(ping.get(self.obs_x_col), np.nan),
                _to_float(ping.get(self.obs_y_col), np.nan),
                candidates,
                self.candidate_radius * candidate_scale,
                true_edge,
            )
            step_diagnostics.append({
                "point_index": int(t),
                "candidate_count": len(candidates),
                "candidate_contains_true": bool(true_edge in candidates) if true_edge is not None else False,
                "position_imputed": _to_flag(ping.get("_position_imputed", False)),
                "ambiguity_mode": bool(ambiguity_mode),
                "trellis_break": bool(t in self._trellis_breaks),
                "connectivity_reanchor": bool(t in self._connectivity_reanchors),
                "trajectory_boundary": bool(trajectory_boundary),
                "beam_contains_true": bool(true_edge in new_probs) if true_edge is not None else False,
                "effective_x": _to_float(ping.get(self.obs_x_col), np.nan),
                "effective_y": _to_float(ping.get(self.obs_y_col), np.nan),
                "candidate_radius_m": float(self.candidate_radius * candidate_scale),
                "direction_reliability": float(direction_reliability),
                **support,
            })

            if t % self.checkpoint_every == 0 or t == total_steps - 1:
                self._save_viterbi_checkpoint(
                    mode,
                    total_steps,
                    t + 1,
                    path_scores,
                    backpointers,
                    ambiguity_mode,
                    freeze_anchor,
                    adaptive_window,
                    effective_observations,
                    step_diagnostics,
                )

            if progress_every and (t % progress_every == 0 or t == total_steps - 1):
                msg = f"[{mode}] step {t}/{total_steps - 1}: active={len(new_probs)}"
                if mode == "quantum":
                    msg += f", z={qmm_z:.3f}, ambiguity={'on' if ambiguity_mode else 'off'}, W={adaptive_window}, anchor={freeze_anchor}"
                print(msg, flush=True)

        if self._connectivity_reanchors:
            print(
                f"[{mode}] trellis re-anchored at {len(self._connectivity_reanchors)}/{total_steps - 1} "
                f"step(s) due to total connectivity loss (documented recovery, see run_viterbi)",
                flush=True,
            )

        path = _segmented_backtrack(path_scores, backpointers, self._trellis_breaks, total_steps)
        self._run_diagnostics[mode] = {
            "steps": step_diagnostics,
            "trellis_breaks": list(self._trellis_breaks),
            "connectivity_reanchors": list(self._connectivity_reanchors),
            "effective_observations": effective_observations,
        }
        return path

    def _point_error(self, edge, idx):
        if self.true_x_col is not None and self.true_y_col is not None:
            tx = _to_float(self.df.iloc[idx][self.true_x_col])
            ty = _to_float(self.df.iloc[idx][self.true_y_col])
        else:
            tx = _to_float(self.df.iloc[idx][self.obs_x_col])
            ty = _to_float(self.df.iloc[idx][self.obs_y_col])
        # Perpendicular (point-to-segment) projection onto the matched edge's
        # geometry, not the edge midpoint — see midpoint error
        # scales with edge length and can report up to half the edge length
        # as "error" on a perfectly correct match.
        ex, ey = self._project_to_edge(edge, tx, ty)
        return float(np.hypot(tx - ex, ty - ey))

    def analyze_results(self, q_path, c_path, mlp_path=None):
        if mlp_path is None:
            mlp_path = []

        n = min(len(q_path), len(c_path))
        if mlp_path:
            n = min(n, len(mlp_path))

        rows = []
        for i in range(n):
            q_edge = q_path[i]
            c_edge = c_path[i]
            m_edge = mlp_path[i] if mlp_path else None

            if self.true_x_col is not None and self.true_y_col is not None:
                tx = _to_float(self.df.iloc[i][self.true_x_col])
                ty = _to_float(self.df.iloc[i][self.true_y_col])
            else:
                tx = _to_float(self.df.iloc[i][self.obs_x_col])
                ty = _to_float(self.df.iloc[i][self.obs_y_col])

            # Perpendicular projection of the true position onto each matched
            # edge — see the point-error definition. Also plotted as the marker location,
            # so the overlay figure and the reported error use the same point.
            qx, qy = self._project_to_edge(q_edge, tx, ty)
            cx, cy = self._project_to_edge(c_edge, tx, ty)

            q_err = self._point_error(q_edge, i)
            c_err = self._point_error(c_edge, i)

            row = {
                "point_index": i,
                "q_u": str(q_edge[0]),
                "q_v": str(q_edge[1]),
                "q_k": int(q_edge[2]),
                "c_u": str(c_edge[0]),
                "c_v": str(c_edge[1]),
                "c_k": int(c_edge[2]),
                "q_proj_x": qx,
                "q_proj_y": qy,
                "c_proj_x": cx,
                "c_proj_y": cy,
                "q_err": q_err,
                "c_err": c_err,
            }

            if m_edge is not None:
                mx, my = self._project_to_edge(m_edge, tx, ty)
                m_err = self._point_error(m_edge, i)
                row.update(
                    {
                        "mlp_u": str(m_edge[0]),
                        "mlp_v": str(m_edge[1]),
                        "mlp_k": int(m_edge[2]),
                        "mlp_proj_x": mx,
                        "mlp_proj_y": my,
                        "mlp_err": m_err,
                    }
                )

            if self.true_x_col is not None and self.true_y_col is not None:
                row["true_x"] = _to_float(self.df.iloc[i][self.true_x_col])
                row["true_y"] = _to_float(self.df.iloc[i][self.true_y_col])

            rows.append(row)

        df_res = pd.DataFrame(rows)

        q_rmse = float(np.sqrt(np.mean(np.square(df_res["q_err"])))) if not df_res.empty else 999.0
        c_rmse = float(np.sqrt(np.mean(np.square(df_res["c_err"])))) if not df_res.empty else 999.0

        summary = {
            "quantum_rmse": q_rmse,
            "classical_rmse": c_rmse,
            "rmse_gain": c_rmse - q_rmse,
            "quantum_mae": float(df_res["q_err"].mean()) if not df_res.empty else 999.0,
            "classical_mae": float(df_res["c_err"].mean()) if not df_res.empty else 999.0,
            "quantum_median": float(df_res["q_err"].median()) if not df_res.empty else 999.0,
            "classical_median": float(df_res["c_err"].median()) if not df_res.empty else 999.0,
            "quantum_p90": float(df_res["q_err"].quantile(0.90)) if not df_res.empty else 999.0,
            "classical_p90": float(df_res["c_err"].quantile(0.90)) if not df_res.empty else 999.0,
            "quantum_p95": float(df_res["q_err"].quantile(0.95)) if not df_res.empty else 999.0,
            "classical_p95": float(df_res["c_err"].quantile(0.95)) if not df_res.empty else 999.0,
            "gain_positive_rate": float((df_res["q_err"] < df_res["c_err"]).mean()) if not df_res.empty else 0.0,
        }

        if "mlp_err" in df_res.columns:
            m_rmse = float(np.sqrt(np.mean(np.square(df_res["mlp_err"]))))
            summary.update(
                {
                    "mlp_rmse": m_rmse,
                    "mlp_mae": float(df_res["mlp_err"].mean()),
                    "mlp_median": float(df_res["mlp_err"].median()),
                    "mlp_p90": float(df_res["mlp_err"].quantile(0.90)),
                    "mlp_p95": float(df_res["mlp_err"].quantile(0.95)),
                    "mlp_gain_vs_classical": c_rmse - m_rmse,
                }
            )
        else:
            summary.update(
                {
                    "mlp_rmse": 999.0,
                    "mlp_mae": 999.0,
                    "mlp_median": 999.0,
                    "mlp_p90": 999.0,
                    "mlp_p95": 999.0,
                    "mlp_gain_vs_classical": 0.0,
                }
            )

        print("--- BENCHMARK SUMMARY ---")
        print(f"Classical RMSE: {c_rmse:.2f}m")
        print(f"Quantum RMSE:   {q_rmse:.2f}m")
        print(f"MLP RMSE:       {summary['mlp_rmse']:.2f}m")

        return df_res, summary

    def _is_same_carriageway(self, m_edge, t_edge):
        """True if m_edge and t_edge are the same physical road segment —
        identical (u, v) node pair, differing only in a multigraph parallel
        key (e.g. adjacent lane record), not a genuinely different road."""
        return (str(m_edge[0]), str(m_edge[1])) == (str(t_edge[0]), str(t_edge[1]))

    @staticmethod
    def _is_reverse_directed_match(m_edge, t_edge):
        return (
            str(m_edge[0]) == str(t_edge[1])
            and str(m_edge[1]) == str(t_edge[0])
        )

    def _is_physical_segment_match(self, m_edge, t_edge):
        """Direction-invariant node-pair match, reported beside exact edge."""
        return self._is_same_carriageway(m_edge, t_edge) or self._is_reverse_directed_match(m_edge, t_edge)

    def _is_relaxed_adjacent_match(self, m_edge, t_edge):
        """True for an exact match, a same-carriageway parallel edge, or a
        directly graph-adjacent edge (m_edge starts where t_edge ends, or
        ends where t_edge starts) — a one-hop tolerance around exact
        directed-edge identity, since a snap to the immediately adjacent
        edge is a materially smaller error than an arbitrary wrong edge
       ."""
        if tuple(m_edge) == tuple(t_edge):
            return True
        if self._is_same_carriageway(m_edge, t_edge):
            return True
        mu, mv = str(m_edge[0]), str(m_edge[1])
        tu, tv = str(t_edge[0]), str(t_edge[1])
        return mu == tv or mv == tu

    def per_fix_diagnostics(self, mode, matched_path):
        """Auditable decoder path and error decomposition for Step07."""
        run_diag = self._run_diagnostics.get(mode, {})
        diag_by_index = {
            int(row["point_index"]): row for row in run_diag.get("steps", [])
        }
        rows = []
        for index, matched in enumerate(matched_path[: len(self.df)]):
            true_edge = self._resolve_true_edge(self.df.iloc[index])
            if true_edge is None or matched is None:
                continue
            matched = tuple(matched)
            true_edge = tuple(true_edge)
            exact = matched == true_edge
            reverse = self._is_reverse_directed_match(matched, true_edge)
            physical = self._is_physical_segment_match(matched, true_edge)
            relaxed = self._is_relaxed_adjacent_match(matched, true_edge)
            step = diag_by_index.get(index, {})
            source = self.df.iloc[index]
            rows.append({
                "point_index": int(index),
                "truck_id": str(source.get("truck_id", "")),
                "timestamp": source.get("timestamp", index),
                "mode": str(mode),
                "true_edge": _edge_key_to_string(true_edge),
                "matched_edge": _edge_key_to_string(matched),
                "directed_exact": bool(exact),
                "physical_segment_match": bool(physical),
                "reverse_directed_match": bool(reverse),
                "adjacent_nonexact_match": bool(relaxed and not physical),
                "layer_match": bool(self._edge_layer(matched) == self._edge_layer(true_edge)),
                "candidate_contains_true": bool(step.get("candidate_contains_true", False)),
                "strict_radius_contains_true": bool(step.get("strict_radius_contains_true", False)),
                "beam_contains_true": bool(step.get("beam_contains_true", False)),
                "candidate_fallback_used": bool(step.get("candidate_fallback_used", False)),
                "candidate_unlocalized": bool(step.get("candidate_unlocalized", False)),
                "true_edge_observation_distance_m": step.get(
                    "true_edge_observation_distance_m", np.nan
                ),
                "candidate_count": int(step.get("candidate_count", 0)),
                "position_imputed": bool(step.get("position_imputed", False)),
                "ambiguity_mode": bool(step.get("ambiguity_mode", False)),
                "trellis_break": bool(step.get("trellis_break", False)),
                "connectivity_reanchor": bool(step.get("connectivity_reanchor", False)),
                "trajectory_boundary": bool(step.get("trajectory_boundary", False)),
                "effective_x": step.get("effective_x", np.nan),
                "effective_y": step.get("effective_y", np.nan),
                "candidate_radius_m": step.get("candidate_radius_m", self.candidate_radius),
            })
        return pd.DataFrame(rows)

    def compute_advanced_metrics(self, matched_path):
        aligned_true = []
        aligned_match = []
        aligned_idx = []

        point_count = min(len(self.df), len(matched_path))
        for i in range(point_count):
            t_edge = self._resolve_true_edge(self.df.iloc[i])
            m_edge = matched_path[i]
            if not isinstance(m_edge, tuple):
                m_edge = _parse_edge_key(m_edge)
            if t_edge is None or m_edge is None:
                continue
            aligned_true.append(t_edge)
            aligned_match.append(m_edge)
            aligned_idx.append(i)

        min_len = min(len(aligned_true), len(aligned_match))
        if min_len < 1:
            return {
                "edge_accuracy": 0.0,
                "z_layer_accuracy": 0.0,
                "tunnel_misclass": 0.0,
                "impossible_jumps": 0,
                "relaxed_adjacent_accuracy": 0.0,
                "same_carriageway_accuracy": 0.0,
                "physical_segment_accuracy": 0.0,
                "reverse_directed_rate": 0.0,
                "layer_aware_rmse": 0.0,
            }

        parsed_true = aligned_true[:min_len]
        parsed_match = aligned_match[:min_len]
        parsed_idx = aligned_idx[:min_len]

        edge_matches = sum(1 for m, t in zip(parsed_match, parsed_true) if tuple(m) == tuple(t))
        edge_accuracy = (edge_matches / min_len) * 100.0

        z_matches = 0
        tunnel_errors = 0
        relaxed_matches = 0
        carriageway_matches = 0
        physical_matches = 0
        reverse_matches = 0
        layer_aware_sq_errors = []

        for m, t, idx in zip(parsed_match, parsed_true, parsed_idx):
            m_layer = self._edge_layer(m)
            t_layer = self._edge_layer(t)
            if m_layer == t_layer:
                z_matches += 1

            if self._edge_tunnel_flag(m) != self._edge_tunnel_flag(t):
                tunnel_errors += 1

            if self._is_relaxed_adjacent_match(m, t):
                relaxed_matches += 1
            if self._is_same_carriageway(m, t):
                carriageway_matches += 1
            if self._is_physical_segment_match(m, t):
                physical_matches += 1
            if self._is_reverse_directed_match(m, t):
                reverse_matches += 1

            # Supplementary layer-aware proxy error: combines
            # the horizontal point-to-geometry error with an assumed fixed
            # vertical spacing per GraphML layer-index difference. This is
            # not a measured or surveyed 3D error — the graph carries no
            # z-coordinate — so it is reported only as a labeled synthetic
            # proxy alongside horizontal RMSE, never in place of it.
            horizontal_error = self._point_error(m, idx)
            vertical_proxy = abs(m_layer - t_layer) * ASSUMED_LEVEL_SPACING_M
            layer_aware_sq_errors.append(horizontal_error ** 2 + vertical_proxy ** 2)

        z_layer_accuracy = (z_matches / min_len) * 100.0
        tunnel_misclass = (tunnel_errors / min_len) * 100.0
        relaxed_adjacent_accuracy = (relaxed_matches / min_len) * 100.0
        same_carriageway_accuracy = (carriageway_matches / min_len) * 100.0
        physical_segment_accuracy = (physical_matches / min_len) * 100.0
        reverse_directed_rate = (reverse_matches / min_len) * 100.0
        layer_aware_rmse = float(np.sqrt(np.mean(layer_aware_sq_errors))) if layer_aware_sq_errors else 0.0

        impossible_jumps = 0
        for i in range(len(parsed_match) - 1):
            e1 = parsed_match[i]
            e2 = parsed_match[i + 1]
            if tuple(e1) == tuple(e2):
                # Consecutive fixes on the same edge are not a transition at
                # all (the common case at 1 Hz sampling), not an
                # "impossible jump" — checking reachability from the edge's
                # end node back to its own start node would wrongly flag the
                # vehicle simply remaining on its current edge as impossible
                # whenever that edge is not part of a cycle.
                continue
            try:
                jump_possible = nx.has_path(self.G, e1[1], e2[0])
            except Exception:
                jump_possible = (e1[1] == e2[0])
            if not jump_possible:
                impossible_jumps += 1

        return {
            "edge_accuracy": edge_accuracy,
            "z_layer_accuracy": z_layer_accuracy,
            "tunnel_misclass": tunnel_misclass,
            "impossible_jumps": impossible_jumps,
            "relaxed_adjacent_accuracy": relaxed_adjacent_accuracy,
            "same_carriageway_accuracy": same_carriageway_accuracy,
            "physical_segment_accuracy": physical_segment_accuracy,
            "reverse_directed_rate": reverse_directed_rate,
            "layer_aware_rmse": layer_aware_rmse,
        }


def plot_benchmark_dashboard(res_df, output_png=None, show_plot=True, title_suffix=""):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(res_df["c_err"], color="red", alpha=0.4, label="Classical-Viterbi Error")
    ax1.plot(res_df["q_err"], color="blue", label="Quantum-Viterbi Error")
    if "mlp_err" in res_df.columns:
        ax1.plot(res_df["mlp_err"], color="purple", alpha=0.8, label="MLP-Viterbi Error")

    ax1.axhline(y=45, color="black", linestyle="--", label="Tunnel Noise Threshold (SDM)")
    title = "Temporal Error Distribution: Classical vs. Quantum"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax1.set_title(title)
    ax1.set_ylabel("Error (Meters)")
    ax1.legend()

    improvement = res_df["c_err"] - res_df["q_err"]
    ax2.hist(improvement, bins=50, color="green", alpha=0.7)
    ax2.set_title("Quantum Advantage (Error Delta)")
    ax2.set_xlabel("Meters Gained (Classical Error - Quantum Error)")

    plt.tight_layout()
    if output_png:
        plt.savefig(output_png, dpi=250, bbox_inches="tight")
        print(f"Saved benchmark dashboard: {output_png}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _serialize_path(path):
    return [[str(u), str(v), int(k)] for u, v, k in path]


def _run_single_case_benchmark(
    label,
    graph_path,
    traj_path,
    seed,
    radius,
    fallback_k,
    beam_width,
    weight_file,
    qts_weight_file,
    confidence_threshold,
    show_plot,
):
    try:
        bench = MapMatchingBenchmarker(
            graph_path,
            traj_path,
            seed=seed,
            candidate_radius=radius,
            candidate_fallback_k=fallback_k,
            weights_file=weight_file,
            qts_weights_file=qts_weight_file,
            confidence_dip_threshold=confidence_threshold,
            beam_width=beam_width,
            checkpoint_id=label,
        )

        q_path = bench.run_viterbi(mode="quantum")
        c_path = bench.run_viterbi(mode="classical")
        hmm_extra_path = bench.run_viterbi(mode="hmm_extra")
        mlp_path = bench.run_viterbi(mode="mlp")
        svm_path = bench.run_viterbi(mode="svm")
        lstm_path = bench.run_viterbi(mode="lstm")
        qmm_no_entanglement_path = bench.run_viterbi(mode="qmm_no_entanglement")
        qmm_entanglement_path = bench.run_viterbi(mode="qmm_entanglement")

        metrics_df, summary = bench.analyze_results(q_path, c_path, mlp_path)
        ablation_paths = {
            "hmm_extra": hmm_extra_path,
            "svm": svm_path,
            "lstm": lstm_path,
            "qmm_no_entanglement": qmm_no_entanglement_path,
            "qmm_entanglement": qmm_entanglement_path,
        }
        for method, path in ablation_paths.items():
            errors = [bench._point_error(edge, idx) for idx, edge in enumerate(path)]
            metrics_df[f"{method}_err"] = pd.Series(errors[: len(metrics_df)])
            summary[f"{method}_rmse"] = float(np.sqrt(np.mean(np.square(errors)))) if errors else 999.0

        summary["hmm_rmse"] = summary["classical_rmse"]
        summary["full_st_qmm_rmse"] = summary["quantum_rmse"]
        summary["mlp_tuning_budget"] = bench.mlp_selection.get("budget", 0)
        summary["svm_tuning_budget"] = bench.svm_selection.get("budget", 0)
        summary["mlp_validation_balanced_accuracy"] = bench.mlp_selection.get("selected", {}).get("validation_balanced_accuracy", np.nan)
        summary["svm_validation_balanced_accuracy"] = bench.svm_selection.get("selected", {}).get("validation_balanced_accuracy", np.nan)

        q_metrics = bench.compute_advanced_metrics(q_path)
        c_metrics = bench.compute_advanced_metrics(c_path)
        mlp_metrics = bench.compute_advanced_metrics(mlp_path)

        summary.update(
            {
                "quantum_edge_accuracy": q_metrics["edge_accuracy"],
                "classical_edge_accuracy": c_metrics["edge_accuracy"],
                "mlp_edge_accuracy": mlp_metrics["edge_accuracy"],
                "quantum_z_layer_accuracy": q_metrics["z_layer_accuracy"],
                "classical_z_layer_accuracy": c_metrics["z_layer_accuracy"],
                "mlp_z_layer_accuracy": mlp_metrics["z_layer_accuracy"],
                "quantum_impossible_jumps": q_metrics["impossible_jumps"],
                "classical_impossible_jumps": c_metrics["impossible_jumps"],
                "mlp_impossible_jumps": mlp_metrics["impossible_jumps"],
                "quantum_tunnel_misclass": q_metrics["tunnel_misclass"],
                "classical_tunnel_misclass": c_metrics["tunnel_misclass"],
                "mlp_tunnel_misclass": mlp_metrics["tunnel_misclass"],
                "qmm_qubits": 9,
                "qts_qubits": 4,
                "edge_accuracy_gain_vs_classical": q_metrics["edge_accuracy"] - c_metrics["edge_accuracy"],
                "z_layer_accuracy_gain_vs_classical": q_metrics["z_layer_accuracy"] - c_metrics["z_layer_accuracy"],
                "impossible_jump_reduction_vs_classical": c_metrics["impossible_jumps"] - q_metrics["impossible_jumps"],
            }
        )

        point_csv = REPORTS_DIR / f"step05_{label}_point_errors.csv"
        metrics_df.to_csv(point_csv, index=False)

        metrics_csv = REPORTS_DIR / f"step05_{label}_benchmark_metrics.csv"
        pd.DataFrame([summary]).to_csv(metrics_csv, index=False)

        tuning_json = REPORTS_DIR / f"step05_{label}_classical_tuning.json"
        with tuning_json.open("w", encoding="utf-8") as handle:
            json.dump({"mlp": bench.mlp_selection, "rbf_svm": bench.svm_selection}, handle, indent=2)

        path_json = REPORTS_DIR / f"step05_{label}_viterbi_paths.json"
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "case": label,
                    "seed": int(seed),
                    "params": {
                        "candidate_radius": float(radius),
                        "fallback_k": int(fallback_k),
                        "beam_width": int(beam_width) if beam_width is not None else None,
                        "weights_file": str(weight_file) if weight_file else None,
                        "qts_weights_file": str(qts_weight_file) if qts_weight_file else None,
                        "confidence_dip_threshold": float(confidence_threshold),
                    },
                    "quantum_path": _serialize_path(q_path),
                    "classical_path": _serialize_path(c_path),
                    "hmm_extra_path": _serialize_path(hmm_extra_path),
                    "mlp_path": _serialize_path(mlp_path),
                    "svm_path": _serialize_path(svm_path),
                    "lstm_path": _serialize_path(lstm_path),
                    "qmm_no_entanglement_path": _serialize_path(qmm_no_entanglement_path),
                    "qmm_entanglement_path": _serialize_path(qmm_entanglement_path),
                },
                f,
                indent=2,
            )

        plot_benchmark_dashboard(
            metrics_df,
            output_png=PLOTS_DIR / f"step05_{label}_benchmark_dashboard.png",
            show_plot=show_plot,
            title_suffix=label,
        )

        return {
            "case": label,
            **summary,
        }
    except Exception as exc:
        print(f"Error handling benchmarking execution case {label}: {exc}")
        traceback.print_exc()
        return {
            "case": label,
            "quantum_rmse": 999,
            "classical_rmse": 999,
            "hmm_rmse": 999,
            "hmm_extra_rmse": 999,
            "mlp_rmse": 999,
            "svm_rmse": 999,
            "lstm_rmse": 999,
            "qmm_no_entanglement_rmse": 999,
            "qmm_entanglement_rmse": 999,
            "full_st_qmm_rmse": 999,
            "rmse_gain": 0,
        }


def write_step05_consolidated_outputs(case_names):
    metrics_rows = []
    point_frames = []
    path_records = []
    path_rows = []

    for case_name in case_names:
        metrics_source = REPORTS_DIR / f"step05_{case_name}_benchmark_metrics.csv"
        if metrics_source.exists():
            df = pd.read_csv(metrics_source)
            if not df.empty:
                row = df.iloc[0].to_dict()
                row["case_study"] = case_name
                row["source_file"] = str(metrics_source)
                row["source_filename"] = metrics_source.name
                metrics_rows.append(row)

        point_source = REPORTS_DIR / f"step05_{case_name}_point_errors.csv"
        if point_source.exists():
            df = pd.read_csv(point_source)
            if not df.empty:
                df = df.copy()
                df["case_study"] = case_name
                df["source_file"] = str(point_source)
                df["source_filename"] = point_source.name
                point_frames.append(df)

        path_source = REPORTS_DIR / f"step05_{case_name}_viterbi_paths.json"
        if path_source.exists():
            with open(path_source, "r", encoding="utf-8") as f:
                payload = json.load(f)
            path_records.append(
                {
                    "case_study": case_name,
                    "source_file": str(path_source),
                    "source_filename": path_source.name,
                    "payload": payload,
                }
            )
            if isinstance(payload, dict):
                for path_type in [
                    "quantum_path",
                    "classical_path",
                    "hmm_extra_path",
                    "mlp_path",
                    "svm_path",
                    "lstm_path",
                    "qmm_no_entanglement_path",
                    "qmm_entanglement_path",
                ]:
                    path = payload.get(path_type, [])
                    for idx, edge in enumerate(path):
                        u = edge[0] if isinstance(edge, list) and len(edge) > 0 else None
                        v = edge[1] if isinstance(edge, list) and len(edge) > 1 else None
                        k = edge[2] if isinstance(edge, list) and len(edge) > 2 else None
                        path_rows.append(
                            {
                                "case_study": case_name,
                                "path_type": path_type.replace("_path", ""),
                                "path_index": idx,
                                "u": u,
                                "v": v,
                                "k": k,
                                "seed": payload.get("seed"),
                                "source_file": str(path_source),
                                "source_filename": path_source.name,
                            }
                        )

    pd.DataFrame(metrics_rows).to_csv(REPORTS_DIR / "step05_benchmark_metrics_all_cases.csv", index=False)
    if point_frames:
        pd.concat(point_frames, axis=0, ignore_index=True).to_csv(
            REPORTS_DIR / "step05_point_errors_all_cases.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(REPORTS_DIR / "step05_point_errors_all_cases.csv", index=False)

    pd.DataFrame(path_rows).to_csv(REPORTS_DIR / "step05_viterbi_paths_all_cases.csv", index=False)
    with open(REPORTS_DIR / "step05_viterbi_paths_all_cases.json", "w", encoding="utf-8") as f:
        json.dump(path_records, f, indent=2)

    print(f"Saved consolidated Step05 metrics: {REPORTS_DIR / 'step05_benchmark_metrics_all_cases.csv'}")
    print(f"Saved consolidated Step05 point errors: {REPORTS_DIR / 'step05_point_errors_all_cases.csv'}")
    print(f"Saved consolidated Step05 paths CSV: {REPORTS_DIR / 'step05_viterbi_paths_all_cases.csv'}")
    print(f"Saved consolidated Step05 paths JSON: {REPORTS_DIR / 'step05_viterbi_paths_all_cases.json'}")


def _step05_to_edge_tuple(edge_like):
    if isinstance(edge_like, (list, tuple)) and len(edge_like) >= 3:
        return (str(edge_like[0]), str(edge_like[1]), int(edge_like[2]))
    raise ValueError(f"Invalid edge format: {edge_like}")


def _step05_project_to_edge(graph, edge, x, y):
    """Standalone counterpart to MapMatchingBenchmarker._project_to_edge for
    plotting code that only has a bare graph, not a benchmarker instance.
    Perpendicular projection onto edge geometry (WKT if present, else the
    straight-line segment between endpoints) — see the point-error definition."""
    u, v, _k = edge
    attrs = graph.get_edge_data(u, v, _k, default=None) if graph.is_multigraph() else graph.get_edge_data(u, v, default=None)
    geometry_text = (attrs or {}).get("geometry")
    if shapely_wkt is not None and Point is not None and geometry_text:
        try:
            geometry = shapely_wkt.loads(str(geometry_text))
            point = geometry.interpolate(geometry.project(Point(float(x), float(y))))
            if point is not None and not point.is_empty:
                return float(point.x), float(point.y)
        except Exception:
            pass
    ax = float(graph.nodes[u]["x"])
    ay = float(graph.nodes[u]["y"])
    bx = float(graph.nodes[v]["x"])
    by = float(graph.nodes[v]["y"])
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    frac = 0.0 if denom <= 0 else np.clip(((x - ax) * dx + (y - ay) * dy) / denom, 0.0, 1.0)
    return float(ax + frac * dx), float(ay + frac * dy)


def _step05_get_actual_xy(row, tree, node_coords):
    if "true_x" in row.index and "true_y" in row.index and pd.notna(row["true_x"]) and pd.notna(row["true_y"]):
        return float(row["true_x"]), float(row["true_y"])

    ox = _to_float(row.get("obs_x", np.nan), np.nan)
    oy = _to_float(row.get("obs_y", np.nan), np.nan)
    if np.isnan(ox) or np.isnan(oy):
        return None, None
    nearest_idx = int(tree.query([ox, oy], k=1)[1])
    tx, ty = node_coords[nearest_idx]
    return float(tx), float(ty)


def generate_step05_final_comparison_plot_from_paths(graph, df, q_path, c_path, output_png=None, show_plot=False, title_suffix=""):
    nodes = list(graph.nodes(data=True))
    node_coords = np.array([[float(d["x"]), float(d["y"])] for _n, d in nodes])
    tree = KDTree(node_coords)

    q_errors = []
    c_errors = []
    rows = []
    point_count = min(len(df), len(q_path), len(c_path))
    print(f"Processing {point_count} points from persisted Viterbi paths...")

    for idx in range(point_count):
        row = df.iloc[idx]
        tx, ty = _step05_get_actual_xy(row, tree, node_coords)
        if tx is None:
            continue

        qx, qy = _step05_project_to_edge(graph, q_path[idx], tx, ty)
        cx, cy = _step05_project_to_edge(graph, c_path[idx], tx, ty)

        q_err = float(np.hypot(tx - qx, ty - qy))
        c_err = float(np.hypot(tx - cx, ty - cy))
        q_errors.append(q_err)
        c_errors.append(c_err)
        rows.append(
            {
                "point_index": idx,
                "true_x": tx,
                "true_y": ty,
                "quantum_proj_x": qx,
                "quantum_proj_y": qy,
                "classical_proj_x": cx,
                "classical_proj_y": cy,
                "quantum_error": q_err,
                "classical_error": c_err,
                "error_reduction": c_err - q_err,
            }
        )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    if not q_errors or not c_errors:
        print("No comparable points available for step05 final-comparison plot.")
        plt.close(fig)
        return pd.DataFrame(rows)

    ax1.plot(c_errors, color="red", alpha=0.3, label=f"Classical-Viterbi (RMSE: {np.sqrt(np.mean(np.square(c_errors))):.2f}m)")
    ax1.plot(q_errors, color="blue", alpha=0.7, label=f"Quantum-Viterbi (RMSE: {np.sqrt(np.mean(np.square(q_errors))):.2f}m)")
    title = "Actual Temporal Error: 3D Resolution"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax1.set_title(title)
    ax1.set_ylabel("Distance from Centerline (m)")
    ax1.legend()

    ax2.hist(np.array(c_errors) - np.array(q_errors), bins=50, color="green")
    ax2.set_title("Viterbi Quantum Advantage (Meters Gained per Point)")
    ax2.set_xlabel("Error Reduction (m)")

    plt.tight_layout()
    if output_png:
        plt.savefig(output_png, dpi=250, bbox_inches="tight")
        print(f"Saved final comparison plot: {output_png}")
    if show_plot:
        plt.show()
    plt.close(fig)
    return pd.DataFrame(rows)


def write_step05_final_comparison_consolidated_outputs(case_names):
    frames = []
    for case_name in case_names:
        source = REPORTS_DIR / f"step05_{case_name}_final_comparison_plot_data.csv"
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

    out_csv = REPORTS_DIR / "step05_final_comparison_plot_data_all_cases.csv"
    if frames:
        pd.concat(frames, axis=0, ignore_index=True).to_csv(out_csv, index=False)
    else:
        pd.DataFrame(
            columns=[
                "point_index",
                "true_x",
                "true_y",
                "quantum_proj_x",
                "quantum_proj_y",
                "classical_proj_x",
                "classical_proj_y",
                "quantum_error",
                "classical_error",
                "error_reduction",
                "case_study",
                "source_file",
                "source_filename",
            ]
        ).to_csv(out_csv, index=False)
    print(f"Saved consolidated Step05 final-comparison plot data: {out_csv}")


def run_step05_final_comparison_for_case(case_name):
    graph_path = unified_graph_path(case_name)
    traj_path = trajectory_path(case_name)
    viterbi_path_file = REPORTS_DIR / f"step05_{case_name}_viterbi_paths.json"

    if not graph_path.exists() or not traj_path.exists() or not viterbi_path_file.exists():
        return f"Skipping {case_name}: missing graph, trajectory, or viterbi path outputs file."

    print(f"\n=== Final comparison for {case_name} ===")
    graph = nx.read_graphml(graph_path)
    df = pd.read_csv(traj_path)

    with open(viterbi_path_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    q_path = [_step05_to_edge_tuple(e) for e in payload.get("quantum_path", [])]
    c_path = [_step05_to_edge_tuple(e) for e in payload.get("classical_path", [])]
    if not q_path or not c_path:
        return f"Skipping {case_name}: viterbi paths JSON missing quantum_path or classical_path."

    step05_fc_df = generate_step05_final_comparison_plot_from_paths(
        graph,
        df,
        q_path,
        c_path,
        output_png=PLOTS_DIR / f"step05_{case_name}_final_viterbi_comparison.png",
        show_plot=False,
        title_suffix=case_name,
    )
    case_csv = REPORTS_DIR / f"step05_{case_name}_final_comparison_plot_data.csv"
    step05_fc_df.to_csv(case_csv, index=False)
    print(f"Saved Step05 final-comparison plot data CSV: {case_csv}")
    return f"Completed final comparison for {case_name}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive quantum vs classical and MLP Viterbi benchmarking.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic initialization.")
    parser.add_argument("--graph", type=str, default=None, help="GraphML filename for a single-case run.")
    parser.add_argument("--trajectory", type=str, default=None, help="Trajectory CSV filename for a single-case run.")
    parser.add_argument(
        "--case-name",
        type=str,
        default=None,
        help="Case label for single-case outputs and resumable checkpoints.",
    )
    parser.add_argument("--radius", type=float, default=INFERENCE_CANDIDATE_RADIUS_M, help="Candidate edge search radius (meters).")
    parser.add_argument("--fallback-k", type=int, default=3, help="Fallback candidate count if radius search returns no edges.")
    parser.add_argument("--weights-file", type=str, default=None, help="Optional .npy calibrated quantum weights file for single-case mode.")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Step06 validation-selected threshold for single-case mode.")
    parser.add_argument("--plot", action="store_true", help="Show benchmark dashboard plots.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_max_workers(),
        help="Maximum parallel workers to use (default: all CPU cores).",
    )
    parser.add_argument(
        "--force-parallel",
        action="store_true",
        help="Force multi-process execution across cases (can be unstable on some macOS/Python setups).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=30,
        help="Maximum active Viterbi states retained per step (set <=0 to disable pruning).",
    )
    args = parser.parse_args()

    ensure_project_dirs()

    if (args.graph is None) ^ (args.trajectory is None):
        raise ValueError("Provide both --graph and --trajectory for single-case mode, or neither to run all hard cases.")

    if args.graph and args.trajectory:
        single_label = args.case_name or Path(args.graph).stem.removeprefix("step01_").removesuffix("_unified")
        targets = [(single_label, args.graph, args.trajectory)]
    else:
        targets = [
            (case_name, f"step01_{case_name}_unified.graphml", f"step03_{case_name}_trajectories_v5.csv")
            for case_name in CASE_NAMES
        ]

    resolved_targets = []
    for idx, (label, graph_name, traj_name) in enumerate(targets):
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
            print(f"Skipping {label}: missing graph or trajectory file.")
            continue

        case_weights = None
        case_qts_weights = None
        if args.graph and args.trajectory and args.weights_file:
            case_weights = args.weights_file
        elif not (args.graph and args.trajectory):
            case_weights = str(weights_path(label))
            qts_wpath = qts_weights_path(label)
            case_qts_weights = str(qts_wpath) if qts_wpath.exists() else None

        confidence_threshold = (
            float(args.confidence_threshold)
            if args.graph and args.trajectory and args.confidence_threshold is not None
            else load_confidence_threshold(label)
        )
        resolved_targets.append((label, graph_path, traj_path, args.seed + idx, case_weights, case_qts_weights, confidence_threshold))

    summary_rows = []
    if resolved_targets:
        requested_workers = max(1, min(int(args.max_workers), len(resolved_targets)))
        # PennyLane + multiprocessing can intermittently hang on macOS when many workers are spawned.
        max_workers = requested_workers if args.force_parallel else 1
        if requested_workers > 1 and max_workers == 1:
            print(
                "Stability mode: running Step05 in serial (1 worker). "
                "Use --force-parallel to override."
            )
        if max_workers == 1:
            print("Running cases sequentially in current process...", flush=True)
            for (label, graph_path, traj_path, seed, weight_file, qts_weight_file, confidence_threshold) in resolved_targets:
                res = _run_single_case_benchmark(
                    label,
                    graph_path,
                    traj_path,
                    seed,
                    args.radius,
                    args.fallback_k,
                    args.beam_width if args.beam_width > 0 else None,
                    weight_file,
                    qts_weight_file,
                    confidence_threshold,
                    args.plot,
                )
                summary_rows.append(res)
                print(f"Completed case: {res['case']}", flush=True)
        else:
            print(f"Initializing parallel pipeline across {max_workers} workers...", flush=True)
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(
                        _run_single_case_benchmark,
                        label,
                        graph_path,
                        traj_path,
                        seed,
                        args.radius,
                        args.fallback_k,
                        args.beam_width if args.beam_width > 0 else None,
                        weight_file,
                        qts_weight_file,
                        confidence_threshold,
                        args.plot,
                    )
                    for (label, graph_path, traj_path, seed, weight_file, qts_weight_file, confidence_threshold) in resolved_targets
                ]
                for fut in as_completed(futures):
                    res = fut.result()
                    summary_rows.append(res)
                    print(f"Completed case: {res['case']}", flush=True)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        out_csv = REPORTS_DIR / "step05_benchmark_summary_all_cases.csv"
        summary_df.to_csv(out_csv, index=False)
        print("\n================ BENCHMARK SUMMARY ================")
        print(summary_df.to_string(index=False))
        print(f"\nSaved benchmark summary: {out_csv}")

    write_step05_consolidated_outputs(CASE_NAMES)
    consolidated_metrics_path = REPORTS_DIR / "step05_benchmark_metrics_all_cases.csv"
    if consolidated_metrics_path.exists():
        consolidated_summary = pd.read_csv(consolidated_metrics_path)
        consolidated_summary = consolidated_summary.rename(columns={"case_study": "case"})
        consolidated_summary = consolidated_summary.drop(
            columns=["source_file", "source_filename"],
            errors="ignore",
        )
        if "case" in consolidated_summary.columns:
            ordered_columns = ["case"] + [
                column for column in consolidated_summary.columns if column != "case"
            ]
            consolidated_summary = consolidated_summary[ordered_columns]
        consolidated_summary.to_csv(
            REPORTS_DIR / "step05_benchmark_summary_all_cases.csv",
            index=False,
        )
        print("Rebuilt six-case Step05 benchmark summary from fresh per-case metrics.")
    for case_name in CASE_NAMES:
        print(run_step05_final_comparison_for_case(case_name))
    write_step05_final_comparison_consolidated_outputs(CASE_NAMES)
