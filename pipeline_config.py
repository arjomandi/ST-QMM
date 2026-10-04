from pathlib import Path
import logging
import os
import json

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
RAW_MAPS_DIR = DATA_DIR / "raw_maps"
GRAPHS_DIR = DATA_DIR / "graphs"
TRAJECTORIES_DIR = DATA_DIR / "trajectories"
CANDIDATES_DIR = DATA_DIR / "candidates"

RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"
EXPORTS_DIR = RESULTS_DIR / "exports"

GEOSCAPE_STANDARD_DIR_CANDIDATES = [
    PROJECT_DIR / "Roads" / "Roads APRIL 2026" / "Standard",
    PROJECT_DIR / "Roads" / "Roads MAY 2026" / "Standard",
]

# Canonical seeds for reproducibility across the pipeline.
DEFAULT_SEED = 42
CALIBRATION_SEED = 2026
REPRO_BASE_SEED = 42

# Named search radii (metres). These serve genuinely different purposes and
# are intentionally not equal — but must be single, shared constants so a
# change to one doesn't silently drift from the others (the step05 and
# step07 CLI defaults must agree with the value actually used).
#
# Radius used to build the candidate edge set at inference time (step05,
# step07, and step06's internal benchmark check).
INFERENCE_CANDIDATE_RADIUS_M = 60.0
# Radius used only to draw "nearby wrong edge" negatives for QMM pairwise-
# ranking calibration (step06.build_ranking_pairs). Matches the inference
# radius by design, so QMM negatives are drawn from the same candidate
# neighbourhood it will see at inference.
QMM_NEGATIVE_SAMPLE_RADIUS_M = 60.0
# QTS pairwise-ranking uses the same next-fix candidate set that the decoder
# exposes at inference.  Keeping a separate name documents the role while the
# alias prevents calibration from silently training on candidates excluded by
# the deployed 60 m search.
QTS_NEGATIVE_SAMPLE_RADIUS_M = INFERENCE_CANDIDATE_RADIUS_M
# Radius used only by step08's standalone candidate-file generator, which
# feeds step09's separate inference/visualization workflow (not the
# step05/06/07 RMSE evaluation chain).
STEP08_CANDIDATE_SEARCH_RADIUS_M = 150.0
# Radii swept for the candidate-radius sensitivity study.
CANDIDATE_RADIUS_SENSITIVITY_SWEEP_M = (60.0, 100.0, 150.0, 200.0)

# GDA2020 / MGA zones used by the Australian case studies. Melbourne is in
# zone 55; Sydney is in zone 56.
METRIC_CRS_BY_STATE = {
    "vic": "EPSG:7855",
    "nsw": "EPSG:7856",
}


def metric_crs_for_state(state_abbr):
    """Return the correct GDA2020 / MGA projected CRS for an Australian state."""
    state = str(state_abbr).strip().lower()
    try:
        return METRIC_CRS_BY_STATE[state]
    except KeyError as exc:
        raise ValueError(f"No metric CRS configured for state: {state_abbr!r}") from exc


def metric_crs_for_case(case_name):
    """Resolve a case name ending in a state abbreviation to its metric CRS."""
    state = str(case_name).rsplit("_", 1)[-1]
    return metric_crs_for_state(state)


def default_max_workers(cpu_fraction=0.8):
    """Return a conservative worker count, capped to a CPU utilization fraction."""
    cpu_count = os.cpu_count() or 1
    workers = int(cpu_count * float(cpu_fraction))
    return max(1, min(cpu_count, workers))


def configure_resource_caps(gpu_fraction=0.8):
    """Set soft GPU memory caps for common backends (best-effort, no-op if unsupported)."""
    frac = max(0.1, min(float(gpu_fraction), 0.95))
    os.environ.setdefault("CUPY_GPU_MEMORY_LIMIT", f"{int(frac * 100)}%")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", f"{frac:.2f}")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", f"{frac:.2f}")


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def make_qml_device(qml_module, wires, prefer_gpu=True, gpu_fraction=0.8, strict_gpu=None):
    """Create a PennyLane device with optional strict GPU requirement and explicit logging."""
    configure_resource_caps(gpu_fraction=gpu_fraction)
    logger = logging.getLogger(__name__)

    if strict_gpu is None:
        strict_gpu = _env_flag("QML_STRICT_GPU", default=False)

    if prefer_gpu:
        try:
            device = qml_module.device("lightning.gpu", wires=wires)
            logger.info("Quantum backend selected: lightning.gpu (wires=%s)", wires)
            return device
        except Exception as exc:
            if strict_gpu:
                raise RuntimeError(
                    "GPU requested but unavailable. "
                    "Set QML_STRICT_GPU=0 to allow CPU fallback or install/configure lightning.gpu."
                ) from exc
            logger.warning(
                "Quantum backend fallback: lightning.gpu unavailable (%s). "
                "Trying lightning.qubit on CPU.",
                str(exc),
            )

    try:
        device = qml_module.device("lightning.qubit", wires=wires)
        logger.info("Quantum backend selected: lightning.qubit (CPU, wires=%s)", wires)
        return device
    except Exception as exc:
        logger.warning(
            "Quantum backend fallback: lightning.qubit unavailable (%s). "
            "Using default.qubit on CPU.",
            str(exc),
        )

    device = qml_module.device("default.qubit", wires=wires)
    logger.info("Quantum backend selected: default.qubit (CPU, wires=%s)", wires)
    return device


def deterministic_ansatz_init(shape, restart=0, scale=0.05):
    """Deterministic, non-random near-identity initial weights for a
    StronglyEntanglingLayers ansatz.

    A Haar-random-like start (uniform draw over the full rotation range)
    is the standard driver of barren plateaus in deep/wide entangling
    circuits (see diagnostic_barren_plateau_scan.py, McClean et al. 2018):
    gradients concentrate around zero as the circuit approaches a 2-design.
    Starting instead from a small perturbation around the identity keeps
    the circuit in a well-conditioned gradient regime at the outset, while
    a golden-ratio low-discrepancy phase (rather than a pseudo-random draw)
    still gives each restart a distinct, reproducible starting point.
    """
    size = int(np.prod(shape))
    index = np.arange(1, size + 1, dtype=float)
    phase = (index * (int(restart) + 1) * 0.6180339887498949) % 1.0
    values = float(scale) * np.sin(2.0 * np.pi * phase)
    return values.reshape(shape)


def ensure_project_dirs():
    """Create standard data/results folders for the PQC road-matching pipeline."""
    for directory in [
        DATA_DIR,
        RAW_MAPS_DIR,
        GRAPHS_DIR,
        TRAJECTORIES_DIR,
        CANDIDATES_DIR,
        RESULTS_DIR,
        MODELS_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        EXPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def geoscape_standard_dir():
    """Resolve Geoscape Standard directory from env var or known in-project locations."""
    env_override = os.environ.get("GEOSCAPE_STANDARD_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()

    for candidate in GEOSCAPE_STANDARD_DIR_CANDIDATES:
        if candidate.exists():
            return candidate

    # Default to first known location so callers can still report a helpful path.
    return GEOSCAPE_STANDARD_DIR_CANDIDATES[0]


def geoscape_state_path(state_abbr):
    """Return a state-level Geoscape file (GeoJSON preferred, SHP fallback), if present."""
    base_path = geoscape_standard_dir()
    state = state_abbr.lower().strip()

    candidates = [
        base_path / f"{state}_roads.geojson",
        base_path / f"{state}.geojson",
        base_path / f"{state.upper()}_roads.geojson",
        base_path / f"{state.upper()}.geojson",
        base_path / f"{state}_roads.shp",
        base_path / f"{state}.shp",
        base_path / f"{state.upper()}_roads.shp",
        base_path / f"{state.upper()}.shp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Flexible fallback in case dataset uses custom naming conventions.
    pattern_matches = sorted(base_path.glob(f"*{state}*roads*.geojson"))
    if pattern_matches:
        return pattern_matches[0]

    pattern_matches = sorted(base_path.glob(f"*{state}*roads*.shp"))
    if pattern_matches:
        return pattern_matches[0]

    state_matches = sorted(base_path.glob(f"*{state}*.geojson"))
    if state_matches:
        return state_matches[0]

    state_matches = sorted(base_path.glob(f"*{state}*.shp"))
    if state_matches:
        return state_matches[0]

    return None


def raw_graph_path(case_name):
    return RAW_MAPS_DIR / f"step00_{case_name}.graphml"


def unified_graph_path(case_name):
    return GRAPHS_DIR / f"step01_{case_name}_unified.graphml"


def trajectory_path(case_name):
    return TRAJECTORIES_DIR / f"step03_{case_name}_trajectories_v5.csv"


def candidates_path(case_name):
    return CANDIDATES_DIR / f"step08_{case_name}_candidates.json"


def weights_path(case_name):
    return MODELS_DIR / f"step06_{case_name}_optimized_weights_v5.npy"


def qts_weights_path(case_name):
    """Trained 4-qubit QTS edge-pair compatibility weights for one case study."""
    return MODELS_DIR / f"step06_{case_name}_qts_optimized_weights_v1.npy"


def classical_matched_weights_path(case_name):
    """Matched classical emission-scorer weights:
    same 9 features, same 320/160-point truck-disjoint split, same margin
    ranking loss and Adam training loop as the QMM, just a small classical
    MLP forward function instead of a quantum circuit."""
    return MODELS_DIR / f"step06_{case_name}_classical_matched_weights_v1.npy"


def classical_matched_qts_weights_path(case_name):
    """Matched classical transition-scorer weights, QTS's counterpart to
    classical_matched_weights_path."""
    return MODELS_DIR / f"step06_{case_name}_classical_matched_qts_weights_v1.npy"


def confidence_threshold_path(case_name):
    """Validation-selected QMM ambiguity threshold for one case study."""
    return MODELS_DIR / f"step06_{case_name}_confidence_threshold_v5.json"


def mlp_ablation_model_path(case_name):
    """Ablation-grid MLP classifier, fit once on the calibration-pool trucks
    only, identically to QMM/QTS, and reused across all 20 evaluation seeds
    so that no held-out evaluation truck is ever seen in training."""
    return MODELS_DIR / f"step06_{case_name}_mlp_ablation_model_v1.joblib"


def svm_ablation_model_path(case_name):
    """Budget-matched RBF-SVM classifier, fit once on the calibration-pool
    trucks only; see mlp_ablation_model_path for why this must not be
    retrained per evaluation seed."""
    return MODELS_DIR / f"step06_{case_name}_svm_ablation_model_v1.joblib"


def split_calibration_and_holdout_trucks(truck_ids, holdout_fraction=0.2, seed=DEFAULT_SEED):
    """Deterministically partition a case's truck IDs into a calibration
    pool (step06: QMM/QTS fit+validation truck splits, and the classical
    baseline retraining pool) and a disjoint held-out evaluation pool
    (step07: the per-seed evaluation truck draw).

    This is the single source of truth for the calibration/evaluation
    boundary so no truck can appear on both sides of it.
    """
    trucks = np.asarray(sorted(set(str(t) for t in truck_ids)))
    rng = np.random.default_rng(int(seed) + 900007)
    shuffled = trucks[rng.permutation(len(trucks))]
    holdout_count = max(1, int(round(len(shuffled) * float(holdout_fraction))))
    holdout_count = min(holdout_count, max(1, len(shuffled) - 1))
    holdout_trucks = shuffled[:holdout_count]
    calibration_trucks = shuffled[holdout_count:]
    return calibration_trucks, holdout_trucks


def load_confidence_threshold(case_name):
    """Load a required validation-selected QMM ambiguity threshold."""
    path = confidence_threshold_path(case_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing calibrated confidence threshold for {case_name}: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        return float(json.load(handle)["threshold"])
