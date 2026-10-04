import networkx as nx
import pandas as pd
import numpy as np
import random
import argparse
import json
import re
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from pipeline_config import CALIBRATION_SEED, DEFAULT_SEED, REPORTS_DIR, default_max_workers, ensure_project_dirs, unified_graph_path, trajectory_path


CASE_NAMES = [
    "Rozelle_Interchange_NSW",
    "West_Gate_Tunnel_VIC",
    "NorthConnex_NSW",
    "Light_Horse_Interchange_NSW",
    "Domain_Tunnel_VIC",
    "M80_Princes_Freeway_VIC",
]

OUTPUT_JSON = REPORTS_DIR / "step03_quality_profiles_overrides.json"
DOMAIN_AUDIT_CSV = REPORTS_DIR / "step03_domain_tunnel_synthetic_quality_metrics.csv"
TRANSFER_AUDIT_CSV = REPORTS_DIR / "step03_transferred_quality_profiles_by_case.csv"


def set_global_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))

def get_traffic_regime(dt_obj):
    """Categorizes time into normalized values for the Temporal Qubits."""
    hour = dt_obj.hour
    
    if 6 <= hour < 10:
        return "AM_PEAK", 1.0
    elif 15 <= hour < 19:
        return "PM_PEAK", 0.9
    elif 10 <= hour < 15:
        return "MIDDAY_NORMAL", 0.4
    elif 19 <= hour < 22:
        return "EVENING_NORMAL", 0.2
    else:
        return "OFF_PEAK_NIGHT", 0.0

def get_temporal_features(dt_obj):
    """Encodes cyclical time features into normalized [0, 1] range for Quantum Gates."""
    # Day of Week: 0 (Mon) to 6 (Sun)
    day_val = dt_obj.weekday() / 6.0 
    
    # Month: 1 (Jan) to 12 (Dec)
    month_val = (dt_obj.month - 1) / 11.0
    
    return day_val, month_val

def to_int_safely(value, default=0):
    """Safe conversion for GraphML attributes."""
    if value is None: return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def to_flag(value):
    """Convert various value types to boolean flag.
    
    Handles:
    - Numeric: 0/0.0 -> False, non-zero -> True
    - String: "1", "1.0", "true", "yes", "y" (case-insensitive) -> True
    """
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value) and float(value) != 0.0)
    
    # Handle strings (including numeric strings from GraphML)
    s = str(value).strip().lower()
    
    # Check for numeric strings first (from GraphML storage)
    try:
        num = float(s)
        return bool(np.isfinite(num) and num != 0.0)
    except (ValueError, TypeError):
        pass
    
    # Check for boolean strings
    return s in {"1", "true", "yes", "y", "on"}


def _attribute_text(value):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return " ".join(_attribute_text(item) for item in value)
    if value is None:
        return ""
    return str(value).strip().lower()


def is_truck_eligible_edge(data):
    """Reject non-operational or access-restricted edges from ground-truth walks."""
    highway = _attribute_text(data.get("highway"))
    geoscape_used = to_flag(data.get("source_geoscape", 0))
    status = " ".join(
        part for part in [
            _attribute_text(data.get("status")),
            _attribute_text(data.get("geoscape_status")) if geoscape_used else "",
        ] if part
    )
    construction = _attribute_text(data.get("construction"))
    access_keys = ["access", "vehicle", "motor_vehicle", "hgv"]
    access_values = " ".join(_attribute_text(data.get(key)) for key in access_keys)
    if geoscape_used:
        access_values += " " + _attribute_text(data.get("geoscape_access_type"))
    trafficable = _attribute_text(data.get("geoscape_trafficable")) if geoscape_used else ""

    if "construction" in highway or construction not in {"", "no", "none", "nan"}:
        return False
    if any(token in status for token in ("closed", "proposed", "under construction")):
        return False
    access_tokens = set(re.sub(r"[^a-z0-9]+", " ", access_values).split())
    if access_tokens.intersection({"no", "private"}) or "management only" in access_values:
        return False
    if trafficable and trafficable not in {"2wd", "nan", "none"}:
        return False
    return True


def _sample_from_weighted_ranges(ranges, probs, fallback_range):
    if not isinstance(ranges, list) or not ranges:
        lo, hi = fallback_range
        return float(np.random.uniform(lo, hi))

    cleaned = []
    for item in ranges:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                lo = float(item[0])
                hi = float(item[1])
                if hi < lo:
                    lo, hi = hi, lo
                cleaned.append((lo, hi))
            except (TypeError, ValueError):
                continue

    if not cleaned:
        lo, hi = fallback_range
        return float(np.random.uniform(lo, hi))

    if not isinstance(probs, list) or len(probs) != len(cleaned):
        probs_arr = np.ones(len(cleaned), dtype=float) / len(cleaned)
    else:
        probs_arr = np.array([max(float(p), 0.0) for p in probs], dtype=float)
        if probs_arr.sum() <= 0:
            probs_arr = np.ones(len(cleaned), dtype=float) / len(cleaned)
        else:
            probs_arr = probs_arr / probs_arr.sum()

    idx = int(np.random.choice(np.arange(len(cleaned)), p=probs_arr))
    lo, hi = cleaned[idx]
    return float(np.random.uniform(lo, hi))


def sample_blackout_steps(ping_interval_seconds, quality_profile):
    """Empirical blackout duration sampled from profile ranges (seconds) then converted to steps."""
    sec = _sample_from_weighted_ranges(
        quality_profile.get("blackout_sec_bins"),
        quality_profile.get("blackout_sec_probs"),
        fallback_range=(1.0, 9.0),
    )
    return max(1, int(round(sec / max(float(ping_interval_seconds), 1e-6))))


def environment_noise_sigma(
    is_tunnel,
    regime_multiplier=1.0,
    open_sky_sigma_m=10.0,
    subterranean_sigma_m=45.0,
):
    """Return the base horizontal-noise scale for the current environment.

    Every subterranean sample remains saturated at ``subterranean_sigma_m``.
    In particular, uncertainty never decays merely because the vehicle is far
    from a portal.  Later quality-state branches apply additional degraded and
    blackout multipliers to this base scale.
    """
    base_sigma_m = subterranean_sigma_m if bool(is_tunnel) else open_sky_sigma_m
    return float(base_sigma_m) * float(regime_multiplier)


def sample_recovery_targets(quality_profile):
    """Empirical post-tunnel recovery based on profile-driven seconds/meters distributions."""
    sec = _sample_from_weighted_ranges(
        quality_profile.get("recovery_sec_bins"),
        quality_profile.get("recovery_sec_probs"),
        fallback_range=(1.0, 30.0),
    )
    dist = _sample_from_weighted_ranges(
        quality_profile.get("recovery_dist_bins"),
        quality_profile.get("recovery_dist_probs"),
        fallback_range=(8.0, 400.0),
    )
    return sec, dist


PROFILE_OVERRIDES_FILE = REPORTS_DIR / "step03_quality_profiles_overrides.json"
QUALITY_METRICS_BY_CASE_CSV = REPORTS_DIR / "step03_generated_quality_metrics_by_case.csv"


def load_quality_profile_overrides(path=PROFILE_OVERRIDES_FILE):
    if not Path(path).exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload.get("profiles", payload)
    except Exception as exc:
        print(f"Warning: failed to load quality profile overrides from {path}: {exc}")
    return {}


def get_case_quality_profile(case_name, overrides=None):
    """Case-level quality profile: built-in defaults plus optional overrides."""
    profile = {
        "portal_impair_prob": 0.006,
        "open_sat_range": (6, 8),
        "open_hdop_range": (1.1, 1.7),
        "tunnel_blackout_trigger": 0.18,
        "tunnel_degraded_prob": 0.62,
        "blackout_sec_bins": [[1.0, 1.5], [1.5, 3.0], [3.0, 4.0], [4.0, 9.0]],
        "blackout_sec_probs": [0.50, 0.40, 0.05, 0.05],
        "recovery_sec_bins": [[1.0, 2.0], [2.0, 6.0], [6.0, 10.0], [10.0, 30.0]],
        "recovery_sec_probs": [0.50, 0.40, 0.08, 0.02],
        "recovery_dist_bins": [[8.0, 30.0], [30.0, 90.0], [90.0, 170.0], [170.0, 400.0]],
        "recovery_dist_probs": [0.50, 0.40, 0.08, 0.02],
        "portal_exit_impair_sec_range": [3.0, 20.0],
        "portal_random_impair_sec_range": [1.0, 9.0],
        "regime_multipliers": {
            "AM_PEAK": 1.5,
            "PM_PEAK": 1.5,
            "MIDDAY_NORMAL": 1.0,
            "EVENING_NORMAL": 1.0,
            "OFF_PEAK_NIGHT": 1.0,
        },
    }

    # Domain Tunnel profile, transferred to the other cases by
    # build_quality_profiles.
    if "Domain_Tunnel_VIC" in case_name:
        profile.update({
            "portal_impair_prob": 0.01,
            "open_sat_range": (5, 8),
            "open_sat_values": [5, 6, 7, 8],
            "open_sat_probs": [0.15, 0.35, 0.40, 0.10],
            "open_hdop_range": (1.1, 1.6),
            "tunnel_blackout_trigger": 0.30,
            "tunnel_degraded_prob": 0.70,
            "blackout_sec_bins": [[1.0, 1.5], [1.5, 3.0], [3.0, 4.0], [4.0, 9.0]],
            "blackout_sec_probs": [0.50, 0.40, 0.05, 0.05],
            "recovery_sec_bins": [[1.0, 2.0], [2.0, 6.0], [6.0, 10.0], [10.0, 30.0]],
            "recovery_sec_probs": [0.50, 0.40, 0.08, 0.02],
            "recovery_dist_bins": [[8.0, 30.0], [30.0, 90.0], [90.0, 170.0], [170.0, 400.0]],
            "recovery_dist_probs": [0.50, 0.40, 0.08, 0.02],
            "portal_exit_impair_sec_range": [3.0, 20.0],
            "portal_random_impair_sec_range": [1.0, 9.0],
            "regime_multipliers": {
                "AM_PEAK": 1.5,
                "PM_PEAK": 1.5,
                "MIDDAY_NORMAL": 1.0,
                "EVENING_NORMAL": 1.0,
                "OFF_PEAK_NIGHT": 1.0,
            },
        })

    # Other hard tunnel corridors keep mild uplift from baseline.
    elif any(token in case_name for token in ["Tunnel", "Connex"]):
        profile.update({
            "portal_impair_prob": 0.008,
            "open_hdop_range": (1.1, 1.8),
            "tunnel_blackout_trigger": 0.22,
            "tunnel_degraded_prob": 0.65,
        })

    # Optional runtime overrides, typically created by step03 calibration.
    if overrides is None:
        overrides = load_quality_profile_overrides()
    if isinstance(overrides, dict):
        case_override = overrides.get(case_name)
        if isinstance(case_override, dict):
            profile.update(case_override)

    # Normalize list/tuple fields after merge.
    if "open_sat_range" in profile and isinstance(profile["open_sat_range"], list):
        profile["open_sat_range"] = tuple(profile["open_sat_range"])
    if "open_hdop_range" in profile and isinstance(profile["open_hdop_range"], list):
        profile["open_hdop_range"] = tuple(profile["open_hdop_range"])

    return profile


def get_quality_profile_provenance(explicit_override=None, path=PROFILE_OVERRIDES_FILE):
    if explicit_override is not None:
        return {
            "calibration_status": "explicit_simulation_override",
            "position_error_calibrated": False,
        }
    if not Path(path).exists():
        return {
            "calibration_status": "built_in_fallback_no_profile_file",
            "position_error_calibrated": False,
        }
    try:
        with open(path, "r", encoding="utf-8") as handle:
            meta = json.load(handle).get("meta", {})
        return {
            "calibration_status": str(meta.get("calibration_status", "legacy_unspecified")),
            "position_error_calibrated": bool(meta.get("position_error_calibrated", False)),
        }
    except Exception:
        return {
            "calibration_status": "profile_metadata_unreadable",
            "position_error_calibrated": False,
        }


def class_metrics(sat, hdop):
    healthy = ((sat >= 8) & (hdop <= 1.5)).mean() * 100.0
    moderate = ((sat.between(5, 7, inclusive="both")) & (hdop > 1.5) & (hdop <= 3.0)).mean() * 100.0
    degraded = ((sat.between(2, 4, inclusive="both")) & (hdop > 3.0) & (hdop <= 8.0)).mean() * 100.0
    blackout = ((sat <= 1) | (hdop > 8.0) | sat.isna() | hdop.isna()).mean() * 100.0

    return {
        "healthy_pct": float(healthy),
        "moderate_pct": float(moderate),
        "degraded_pct": float(degraded),
        "blackout_pct": float(blackout),
        "hdop_p50": float(hdop.quantile(0.5)),
        "hdop_p90": float(hdop.quantile(0.9)),
        "hdop_p95": float(hdop.quantile(0.95)),
        "sat_p10": float(sat.quantile(0.1)),
        "sat_p50": float(sat.quantile(0.5)),
        "sat_p90": float(sat.quantile(0.9)),
    }


def synth_metrics_from_df(df):
    sat = pd.to_numeric(df["sat_count"], errors="coerce")
    hdop = pd.to_numeric(df["hdop"], errors="coerce")
    return class_metrics(sat, hdop)


def estimate_graph_complexity(case_name):
    gp = unified_graph_path(case_name)
    if not gp.exists():
        return {
            "tunnel_ratio": 0.02,
            "bridge_ratio": 0.05,
            "junction_ratio": 0.2,
            "complexity": 0.2,
        }

    graph = nx.read_graphml(gp)
    total_edges = max(1, graph.number_of_edges())

    tunnel_edges = 0
    bridge_edges = 0
    if graph.is_multigraph():
        edge_iter = graph.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, 0, data) for u, v, data in graph.edges(data=True))

    for _u, _v, _k, data in edge_iter:
        tunnel_flag = to_flag(data.get("tunnel", 0))
        bridge_flag = to_flag(data.get("bridge", 0))
        try:
            layer_num = float(data.get("layer", 0))
        except (TypeError, ValueError):
            layer_num = 0.0

        if tunnel_flag or layer_num < 0:
            tunnel_edges += 1
        if bridge_flag or layer_num > 0:
            bridge_edges += 1

    deg = dict(graph.degree())
    junction_ratio = sum(1 for d in deg.values() if d >= 3) / max(1, len(deg))

    tunnel_ratio = tunnel_edges / total_edges
    bridge_ratio = bridge_edges / total_edges
    complexity = 0.6 * tunnel_ratio + 0.25 * bridge_ratio + 0.15 * junction_ratio

    return {
        "tunnel_ratio": float(tunnel_ratio),
        "bridge_ratio": float(bridge_ratio),
        "junction_ratio": float(junction_ratio),
        "complexity": float(complexity),
    }


def transfer_profile_to_all_cases(domain_profile):
    profiles = {"Domain_Tunnel_VIC": domain_profile}

    domain_stats = estimate_graph_complexity("Domain_Tunnel_VIC")
    domain_complexity = max(domain_stats["complexity"], 1e-6)

    for case in CASE_NAMES:
        if case == "Domain_Tunnel_VIC":
            continue

        stats = estimate_graph_complexity(case)
        scale = np.clip(stats["complexity"] / domain_complexity, 0.6, 1.6)

        prof = deepcopy(domain_profile)
        prof["portal_impair_prob"] = float(np.clip(domain_profile["portal_impair_prob"] * (0.85 + 0.45 * scale), 0.004, 0.05))
        prof["tunnel_blackout_trigger"] = float(np.clip(domain_profile["tunnel_blackout_trigger"] * (0.85 + 0.35 * scale), 0.12, 0.6))
        prof["tunnel_degraded_prob"] = float(np.clip(domain_profile["tunnel_degraded_prob"] + 0.08 * (scale - 1.0), 0.35, 0.85))

        if "blackout_sec_bins" in domain_profile:
            prof["blackout_sec_bins"] = deepcopy(domain_profile["blackout_sec_bins"])
        if "blackout_sec_probs" in domain_profile:
            prof["blackout_sec_probs"] = deepcopy(domain_profile["blackout_sec_probs"])
        if "recovery_sec_bins" in domain_profile:
            prof["recovery_sec_bins"] = deepcopy(domain_profile["recovery_sec_bins"])
        if "recovery_sec_probs" in domain_profile:
            prof["recovery_sec_probs"] = deepcopy(domain_profile["recovery_sec_probs"])
        if "recovery_dist_bins" in domain_profile:
            prof["recovery_dist_bins"] = deepcopy(domain_profile["recovery_dist_bins"])
        if "recovery_dist_probs" in domain_profile:
            prof["recovery_dist_probs"] = deepcopy(domain_profile["recovery_dist_probs"])

        if "portal_exit_impair_sec_range" in domain_profile and isinstance(domain_profile["portal_exit_impair_sec_range"], list):
            lo, hi = domain_profile["portal_exit_impair_sec_range"]
            stretch = float(np.clip(0.9 + 0.25 * scale, 0.8, 1.25))
            prof["portal_exit_impair_sec_range"] = [float(lo) * stretch, float(hi) * stretch]

        if "portal_random_impair_sec_range" in domain_profile and isinstance(domain_profile["portal_random_impair_sec_range"], list):
            lo, hi = domain_profile["portal_random_impair_sec_range"]
            stretch = float(np.clip(0.9 + 0.20 * scale, 0.8, 1.2))
            prof["portal_random_impair_sec_range"] = [float(lo) * stretch, float(hi) * stretch]

        if "regime_multipliers" in domain_profile and isinstance(domain_profile["regime_multipliers"], dict):
            regime = deepcopy(domain_profile["regime_multipliers"])
            peak_mult = float(np.clip((regime.get("AM_PEAK", 1.5) + regime.get("PM_PEAK", 1.5)) / 2.0, 1.2, 1.9))
            normal_mult = float(np.clip(1.0 + 0.08 * (scale - 1.0), 0.9, 1.2))
            regime.update(
                {
                    "AM_PEAK": peak_mult,
                    "PM_PEAK": peak_mult,
                    "MIDDAY_NORMAL": normal_mult,
                    "EVENING_NORMAL": normal_mult,
                    "OFF_PEAK_NIGHT": normal_mult,
                }
            )
            prof["regime_multipliers"] = regime

        low_hdop, high_hdop = prof.get("open_hdop_range", [1.1, 1.8])
        high_hdop = float(np.clip(high_hdop + 0.2 * (scale - 1.0), 1.5, 2.8))
        prof["open_hdop_range"] = [float(low_hdop), high_hdop]

        vals = prof.get("open_sat_values", [5, 6, 7, 8])
        probs = np.array(prof.get("open_sat_probs", [0.2, 0.35, 0.35, 0.1]), dtype=float)
        if len(vals) == 4 and len(probs) == 4:
            shift = float(np.clip((scale - 1.0) * 0.08, -0.12, 0.12))
            probs[0] += max(0.0, shift)
            probs[1] += max(0.0, shift) * 0.7
            probs[2] -= max(0.0, shift) * 0.8
            probs[3] -= max(0.0, shift) * 0.9
            probs[0] -= max(0.0, -shift) * 0.8
            probs[1] -= max(0.0, -shift) * 0.6
            probs[2] += max(0.0, -shift) * 0.8
            probs[3] += max(0.0, -shift) * 0.9
            probs = np.clip(probs, 0.01, None)
            probs = probs / probs.sum()
            prof["open_sat_probs"] = [float(v) for v in probs]

        profiles[case] = prof

    return profiles


def build_quality_profiles(
    num_trucks,
    pings_per_truck,
    ping_interval_seconds,
    seed,
    max_workers,
    output,
):
    """Write the built-in Domain Tunnel quality profile, transferred to every
    case by graph complexity, to the profile file read during generation.
    These profiles are simulation assumptions, not empirically calibrated."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    calibration_status = "built_in_profile"
    best_profile = get_case_quality_profile("Domain_Tunnel_VIC", overrides={})
    domain_df = generate_ieee_quantum_trajectories_v5(
        unified_graph_path("Domain_Tunnel_VIC"),
        num_trucks=int(num_trucks),
        pings_per_truck=int(pings_per_truck),
        seed=int(seed),
        ping_interval_seconds=float(ping_interval_seconds),
        quality_profile_override={"Domain_Tunnel_VIC": best_profile},
        save_output=False,
    )
    domain_synth_metrics = synth_metrics_from_df(domain_df)
    profiles = transfer_profile_to_all_cases(best_profile)

    payload = {
        "meta": {
            "calibration_status": calibration_status,
            "note": (
                "Built-in simulation assumptions for satellite count, HDOP, position-error sigma, "
                "blackout duration, and recovery distance; no empirical calibration."
            ),
            "calibrated_fields": [],
            "position_error_calibrated": False,
            "blackout_duration_calibrated": False,
            "recovery_distance_calibrated": False,
            "num_trucks": int(num_trucks),
            "pings_per_truck": int(pings_per_truck),
            "ping_interval_seconds": float(ping_interval_seconds),
            "seed": int(seed),
            "max_workers": int(max_workers),
        },
        "domain_synth_metrics": domain_synth_metrics,
        "profiles": profiles,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    domain_rows = [
        {
            "metric": metric_name,
            "calibration_status": calibration_status,
            "domain_synthetic_value": float(domain_synth_metrics.get(metric_name, np.nan)),
        }
        for metric_name in sorted(domain_synth_metrics.keys())
    ]
    pd.DataFrame(domain_rows).to_csv(DOMAIN_AUDIT_CSV, index=False)

    transfer_rows = []
    for case_name in CASE_NAMES:
        stats = estimate_graph_complexity(case_name)
        prof = profiles.get(case_name, {})
        transfer_rows.append(
            {
                "case": case_name,
                "calibration_status": calibration_status,
                "tunnel_ratio": float(stats.get("tunnel_ratio", np.nan)),
                "bridge_ratio": float(stats.get("bridge_ratio", np.nan)),
                "junction_ratio": float(stats.get("junction_ratio", np.nan)),
                "complexity": float(stats.get("complexity", np.nan)),
                "portal_impair_prob": float(prof.get("portal_impair_prob", np.nan)),
                "tunnel_blackout_trigger": float(prof.get("tunnel_blackout_trigger", np.nan)),
                "tunnel_degraded_prob": float(prof.get("tunnel_degraded_prob", np.nan)),
            }
        )
    pd.DataFrame(transfer_rows).to_csv(TRANSFER_AUDIT_CSV, index=False)

    print(f"Saved quality profiles to: {out_path}")
    print(f"Saved Domain profile audit: {DOMAIN_AUDIT_CSV}")
    print(f"Saved transfer audit by case: {TRANSFER_AUDIT_CSV}")
    print("Domain synthetic metrics:", domain_synth_metrics)


def _edge_geometry_length(G, u, v, data):
    """Real length of an edge in metres: WKT geometry if present, else the
    straight-line distance between its endpoint nodes."""
    geometry_text = data.get("geometry")
    if geometry_text:
        try:
            return float(shapely_wkt.loads(str(geometry_text)).length)
        except Exception:
            pass
    try:
        ux, uy = float(G.nodes[u]["x"]), float(G.nodes[u]["y"])
        vx, vy = float(G.nodes[v]["x"]), float(G.nodes[v]["y"])
        return float(np.hypot(vx - ux, vy - uy))
    except Exception:
        return 0.0


def case_pings_per_truck(graph_path, base_pings=50, ping_interval_seconds=1.0, min_tunnel_duration_seconds=200.0,
                          tunnel_ratio_threshold=0.05,
                          min_sparse_tunnel_duration_seconds=90.0):
    """Scale pings-per-truck (and hence route length and simulated duration,
    per generate_ieee_quantum_trajectories_v5's build_valid_scenario_walk /
    resample_route / current_dt logic below) so tunnel-heavy cases actually
    produce a sustained, multi-minute blackout window instead of a fixed
    50-ping / 50-second trajectory regardless of tunnel length.

    5,000 observations / 100 trucks = 50 fixes/truck at 1 Hz
    = 50 seconds of driving, which cannot contain the "sustained,
    multi-minute GNSS-denied windows" the paper describes for a 9 km tunnel
    (e.g. NorthConnex). Cases with negligible tunnel presence are left at
    base_pings — this only lengthens trajectories where the network
    actually motivates it.
    """
    G = nx.read_graphml(graph_path)
    if G.is_multigraph():
        edges = [(str(u), str(v), data) for u, v, _k, data in G.edges(keys=True, data=True)]
    else:
        edges = [(str(u), str(v), data) for u, v, data in G.edges(data=True)]

    total_len = 0.0
    tunnel_len = 0.0
    for u, v, data in edges:
        length = _edge_geometry_length(G, u, v, data)
        total_len += length
        if to_flag(data.get("tunnel", 0)) or to_int_safely(data.get("layer", 0)) < 0:
            tunnel_len += length

    tunnel_ratio = (tunnel_len / total_len) if total_len > 0 else 0.0
    if tunnel_len <= 0.0:
        return base_pings

    if tunnel_ratio < tunnel_ratio_threshold:
        # Sparse tunnel ramps (notably Light Horse) still need enough time to
        # record a directed entrance and exit.  Fifty one-second fixes can
        # reach the target but end before the downstream surface edge.
        min_sparse_pings = int(
            np.ceil(min_sparse_tunnel_duration_seconds / max(ping_interval_seconds, 1e-6))
        )
        return max(int(base_pings), min_sparse_pings)

    min_pings_for_duration = int(np.ceil(min_tunnel_duration_seconds / max(ping_interval_seconds, 1e-6)))
    return max(int(base_pings), min_pings_for_duration)


def generate_ieee_quantum_trajectories_v5(
    graph_path,
    num_trucks=100,
    pings_per_truck=50,
    seed=42,
    ping_interval_seconds=1.0,
    quality_profile_override=None,
    save_output=True,
):
    """
    Stage 3: Generates synthetic trajectories with 3D context and multi-temporal qubits.
    """
    print(f"\n--- Generating Advanced Quantum Trajectories for {graph_path} ---")
    ensure_project_dirs()
    
    if not Path(graph_path).exists():
        print(f"File not found: {graph_path}")
        return

    set_global_seed(seed)
    G = nx.read_graphml(graph_path)
    nodes = list(G.nodes())
    all_pings = []
    case_name = Path(graph_path).name.replace("_unified.graphml", "").removeprefix("step01_")
    quality_profile = get_case_quality_profile(case_name, overrides=quality_profile_override)
    quality_provenance = get_quality_profile_provenance(quality_profile_override)

    # Simulation start windows to cover various regimes/days
    start_dates = [
        datetime(2026, 5, 26, 8, 30),  # Tuesday (Weekday) AM Peak
        datetime(2026, 12, 12, 22, 0), # Saturday (Weekend/Holiday) Night
        datetime(2026, 8, 15, 14, 0),  # Saturday Midday
        datetime(2026, 3, 10, 17, 45)  # Monday PM Peak
    ]

    if G.is_multigraph():
        all_graph_edges = [(str(u), str(v), int(k), data) for u, v, k, data in G.edges(keys=True, data=True)]
    else:
        all_graph_edges = [(str(u), str(v), 0, data) for u, v, data in G.edges(data=True)]
    graph_edges = [edge for edge in all_graph_edges if is_truck_eligible_edge(edge[3])]
    excluded_edges = len(all_graph_edges) - len(graph_edges)
    print(f"Truck-route eligibility: {len(graph_edges)} usable edges, {excluded_edges} excluded edges")
    if not graph_edges:
        raise RuntimeError(f"No operational, truck-eligible edges in {graph_path}")
    tunnel_edges = [edge for edge in graph_edges if to_flag(edge[3].get("tunnel", 0)) or to_int_safely(edge[3].get("layer", 0)) < 0]
    bridge_edges = [edge for edge in graph_edges if to_flag(edge[3].get("bridge", 0)) or to_int_safely(edge[3].get("layer", 0)) > 0]
    surface_edges = [edge for edge in graph_edges if to_int_safely(edge[3].get("layer", 0)) == 0]

    def outgoing_edges(node):
        if G.is_multigraph():
            edges = [(str(u), str(v), int(k), data) for u, v, k, data in G.out_edges(node, keys=True, data=True)]
        else:
            edges = [(str(u), str(v), 0, data) for u, v, data in G.out_edges(node, data=True)]
        return [edge for edge in edges if is_truck_eligible_edge(edge[3])]

    def is_immediate_reverse(previous, candidate):
        return previous[0] == candidate[1] and previous[1] == candidate[0]

    def edge_geometry(edge):
        u, v, _k, data = edge
        text = data.get("geometry")
        if text:
            try:
                return shapely_wkt.loads(str(text))
            except Exception:
                pass
        return LineString([
            (float(G.nodes[u]["x"]), float(G.nodes[u]["y"])),
            (float(G.nodes[v]["x"]), float(G.nodes[v]["y"])),
        ])

    def route_length(route):
        return sum(float(edge_geometry(edge).length) for edge in route)

    def edge_speed_mps(edge):
        """Return a conservative operational speed for time-based sampling."""
        raw_speed = edge[3].get("maxspeed", 60)
        match = re.search(r"[-+]?\d*\.?\d+", str(raw_speed))
        speed_kmh = float(match.group(0)) if match else 60.0
        if "mph" in str(raw_speed).lower():
            speed_kmh *= 1.609344
        return float(np.clip(speed_kmh, 5.0, 130.0) / 3.6)

    def route_duration_seconds(route):
        return sum(
            float(edge_geometry(edge).length) / max(edge_speed_mps(edge), 1e-6)
            for edge in route
        )

    def edge_is_tunnel(edge):
        return to_flag(edge[3].get("tunnel", 0)) or to_int_safely(edge[3].get("layer", 0)) < 0

    def edge_is_bridge(edge):
        return to_flag(edge[3].get("bridge", 0)) or to_int_safely(edge[3].get("layer", 0)) > 0

    def bounded_surface_exit_path(target_edge, max_edges=20):
        """Return a short directed path from a tunnel target to open road.

        Some study windows truncate tunnel geometry before its portal, so a
        surface exit is not always present in the local graph.  In that case
        ``None`` is returned and the tunnel-only segment remains usable.
        """
        queue = [(target_edge[1], target_edge, [])]
        seen = {target_edge[1]}
        while queue:
            node, previous, path = queue.pop(0)
            if len(path) >= max_edges:
                continue
            options = [
                edge for edge in outgoing_edges(node)
                if not is_immediate_reverse(previous, edge)
            ]
            for edge in options:
                next_path = path + [edge]
                if not edge_is_tunnel(edge):
                    return next_path
                if edge[1] not in seen:
                    seen.add(edge[1])
                    queue.append((edge[1], edge, next_path))
        return None

    def resample_route(route):
        """Sample a route at the declared cadence with feasible truck speed.

        A trajectory uses a constant conservative cruise speed no greater
        than 90 km/h or any traversed edge's limit.  This produces fixed-time
        samples without instantaneous acceleration or speed-limit violations.
        """
        geometries = [edge_geometry(edge) for edge in route]
        lengths = np.asarray([max(float(geometry.length), 1e-6) for geometry in geometries])
        speeds = np.asarray([edge_speed_mps(edge) for edge in route], dtype=float)
        cruise_speed = float(min(25.0, np.min(speeds)))
        cumulative_distance = np.cumsum(lengths)
        targets = (
            np.arange(pings_per_truck, dtype=float)
            * float(ping_interval_seconds)
            * cruise_speed
        )
        if targets[-1] > cumulative_distance[-1] + 1e-9:
            raise RuntimeError("Route is shorter than its requested fixed-time sampling horizon")

        samples = []
        for target_distance in targets:
            edge_index = int(np.searchsorted(cumulative_distance, target_distance, side="right"))
            edge_index = min(edge_index, len(route) - 1)
            before = 0.0 if edge_index == 0 else cumulative_distance[edge_index - 1]
            local_distance = float(np.clip(target_distance - before, 0.0, lengths[edge_index]))
            point = geometries[edge_index].interpolate(local_distance)
            samples.append((
                float(point.x),
                float(point.y),
                route[edge_index],
                float(target_distance),
                cruise_speed,
            ))
        return samples

    def build_valid_scenario_walk(truck_index, target_pool=None):
        """Create and interpolate a directed route with deliberate level coverage."""
        categories = [tunnel_edges, bridge_edges, surface_edges]
        preferred = target_pool or categories[truck_index % len(categories)] or surface_edges or graph_edges
        for _attempt in range(200):
            target_u, target_v, target_k, target_data = random.choice(preferred)
            target_edge = (target_u, target_v, target_k, target_data)
            target_is_tunnel = edge_is_tunnel(target_edge)
            target_is_bridge = edge_is_bridge(target_edge)
            prefix = []
            # When possible, approach a non-surface target from a real surface edge.
            incoming = []
            if G.is_multigraph():
                incoming = [(str(u), str(v), int(k), data) for u, v, k, data in G.in_edges(target_u, keys=True, data=True)]
            else:
                incoming = [(str(u), str(v), 0, data) for u, v, data in G.in_edges(target_u, data=True)]
            surface_incoming = [
                edge for edge in incoming
                if is_truck_eligible_edge(edge[3])
                and to_int_safely(edge[3].get("layer", 0)) == 0
                and not is_immediate_reverse(edge, (target_u, target_v, target_k, target_data))
            ]
            if surface_incoming and to_int_safely(target_data.get("layer", 0)) != 0:
                candidate_prefix = random.choice(surface_incoming)
                sampling_horizon = max(
                    0.0, (pings_per_truck - 1) * float(ping_interval_seconds)
                )
                # Keep an approach only when at least one regular-cadence fix
                # can still land on the deliberately selected target edge.
                approach_cruise = min(
                    25.0,
                    edge_speed_mps(candidate_prefix),
                    edge_speed_mps(target_edge),
                )
                approach_time = route_length([candidate_prefix]) / max(approach_cruise, 1e-6)
                if approach_time <= max(0.0, sampling_horizon - 1.0):
                    prefix = [candidate_prefix]
            selected = prefix + [target_edge]
            target_position = len(prefix)

            # Preserve a real post-tunnel transition whenever the local graph
            # contains one, even when the prefix plus target already exceeds
            # desired_distance (as at Light Horse).
            exit_path = bounded_surface_exit_path(target_edge) if target_is_tunnel else None
            if exit_path:
                selected.extend(exit_path)

            desired_duration = max(
                float(ping_interval_seconds),
                (pings_per_truck - 1) * float(ping_interval_seconds),
            )
            while route_duration_seconds(selected) < desired_duration and len(selected) < 100:
                current = selected[-1][1]
                options = [
                    edge for edge in outgoing_edges(current)
                    if not is_immediate_reverse(selected[-1], edge)
                ]
                viable = [edge for edge in options if outgoing_edges(edge[1])]
                options = viable or options
                if not options:
                    break
                current_layer = to_int_safely(selected[-1][3].get("layer", 0))
                weights = []
                for edge in options:
                    next_layer = to_int_safely(edge[3].get("layer", 0))
                    # Prefer plausible continuation, with occasional portal/ramp changes.
                    weights.append(3.0 if next_layer == current_layer else 1.5)
                chosen = random.choices(options, weights=weights, k=1)[0]
                selected.append(chosen)
            route_has_target = any(
                (target_is_tunnel and edge_is_tunnel(edge))
                or (target_is_bridge and edge_is_bridge(edge))
                or (not target_is_tunnel and not target_is_bridge)
                for edge in selected
            )
            exited_tunnel = not target_is_tunnel or exit_path is None or any(
                not edge_is_tunnel(edge) for edge in selected[target_position + 1:]
            )
            if route_length(selected) >= 150.0 and route_has_target and exited_tunnel:
                if route_duration_seconds(selected) + 1e-9 < desired_duration:
                    continue
                samples = resample_route(selected)
                sampled_identities = {
                    (edge[0], edge[1], edge[2])
                    for _x, _y, edge, _distance, _speed in samples
                }
                if (target_u, target_v, target_k) not in sampled_identities:
                    continue
                return samples
        if target_pool is not graph_edges:
            # Preferred category (tunnel/bridge/surface) was too sparsely
            # connected to complete a route in 200 tries; broaden the
            # candidate pool once before giving up on this truck entirely.
            return build_valid_scenario_walk(truck_index, target_pool=graph_edges)
        raise RuntimeError(f"Unable to generate a valid interpolated route for truck {truck_index}")

    for truck_idx in range(num_trucks):
        truck_id = f"TRUCK_{truck_idx:03d}"
        truck_start_time = random.choice(start_dates)

        # Track tunnel state so we can model realistic post-tunnel quality recovery.
        prev_is_tunnel = 0
        blackout_steps_remaining = 0
        recovery_seconds_remaining = 0.0
        recovery_dist_remaining_m = 0.0
        recovery_seconds_target = 0.0
        recovery_dist_target_m = 0.0
        portal_impair_steps_remaining = 0
        
        # 1. Directed graph walk with deliberate tunnel/bridge/surface coverage.
        route_samples = build_valid_scenario_walk(truck_idx)

        # 2. Sequential State and Observation Generation
        for step, (true_x, true_y, sampled_edge, true_path_distance_m, true_speed_mps) in enumerate(route_samples):
            
            # Temporal Logic for Qubits
            current_dt = truck_start_time + timedelta(seconds=step * ping_interval_seconds)
            regime_label, time_qubit = get_traffic_regime(current_dt)
            day_qubit, month_qubit = get_temporal_features(current_dt)

            # Extract Road Geometry from the ACTUAL selected edge (not just first outbound)
            edge_attrs = {"layer": 0, "tunnel": 0, "bridge": 0, "bearing": 0, "maxspeed": 60, "osmid": ""}
            edge_u, edge_v, graph_edge_key, edge_data = sampled_edge
            edge_attrs.update(edge_data)
            edge_osmid = edge_data.get('osmid', '')

            is_tunnel = 1 if edge_is_tunnel(sampled_edge) else 0
            is_bridge = 1 if edge_is_bridge(sampled_edge) else 0
            is_oneway = 1 if to_flag(edge_attrs.get("oneway", 0)) else 0

            # Approximate per-ping travel distance from edge speed (5s cadence).
            edge_speed_kmh = max(to_int_safely(edge_attrs.get('maxspeed', 60), default=60), 5)
            step_dist_m = (edge_speed_kmh / 3.6) * ping_interval_seconds

            # Trigger recovery when transitioning from tunnel to open sky.
            if prev_is_tunnel == 1 and is_tunnel == 0:
                rec_sec, rec_dist = sample_recovery_targets(quality_profile)
                recovery_seconds_remaining = rec_sec
                recovery_dist_remaining_m = rec_dist
                recovery_seconds_target = rec_sec
                recovery_dist_target_m = rec_dist

                # Portal exit turbulence window to emulate short reacquisition instability.
                portal_exit_min, portal_exit_max = quality_profile.get("portal_exit_impair_sec_range", [3.0, 20.0])
                portal_impair_steps_remaining = max(
                    1,
                    int(round(np.random.uniform(float(portal_exit_min), float(portal_exit_max)) / max(ping_interval_seconds, 1e-6))),
                )

            # 3. Environment-Aware Noise Model
            regime_multiplier = float(
                quality_profile.get("regime_multipliers", {}).get(
                    regime_label,
                    (1.5 if "PEAK" in regime_label else 1.0),
                )
            )
            noise_sigma = environment_noise_sigma(is_tunnel, regime_multiplier)

            quality_state = "open_sky"

            if is_tunnel == 0 and portal_impair_steps_remaining <= 0 and np.random.rand() < quality_profile["portal_impair_prob"]:
                portal_random_min, portal_random_max = quality_profile.get("portal_random_impair_sec_range", [1.0, 9.0])
                portal_impair_steps_remaining = max(
                    1,
                    int(round(np.random.uniform(float(portal_random_min), float(portal_random_max)) / max(ping_interval_seconds, 1e-6))),
                )

            if is_tunnel == 1:
                quality_state = "tunnel"

                if blackout_steps_remaining <= 0 and np.random.rand() < quality_profile["tunnel_blackout_trigger"]:
                    blackout_steps_remaining = sample_blackout_steps(ping_interval_seconds, quality_profile)

                if blackout_steps_remaining > 0:
                    quality_state = "tunnel_blackout"
                    blackout_steps_remaining -= 1
                    if np.random.rand() < 0.80:
                        obs_x, obs_y = np.nan, np.nan
                    else:
                        obs_x = true_x + np.random.normal(0, noise_sigma * 2.5)
                        obs_y = true_y + np.random.normal(0, noise_sigma * 2.5)
                    hdop = np.random.uniform(8.5, 10.5)
                    sats = np.random.randint(0, 2)
                else:
                    # Tunnel but still receiving partial signal (degraded/moderate states).
                    if np.random.rand() < quality_profile["tunnel_degraded_prob"]:
                        quality_state = "tunnel_degraded"
                        obs_x = true_x + np.random.normal(0, noise_sigma * 1.25)
                        obs_y = true_y + np.random.normal(0, noise_sigma * 1.25)
                        hdop = np.random.uniform(3.2, 8.0)
                        sats = np.random.randint(2, 6)
                    else:
                        quality_state = "tunnel_moderate"
                        obs_x = true_x + np.random.normal(0, noise_sigma)
                        obs_y = true_y + np.random.normal(0, noise_sigma)
                        hdop = np.random.uniform(1.6, 3.0)
                        sats = np.random.randint(5, 7)
            else:
                # Post-tunnel recovery: quality improves over time and traveled distance.
                recovery_active = recovery_seconds_remaining > 0.0 and recovery_dist_remaining_m > 0.0
                if recovery_active:
                    quality_state = "post_tunnel_recovery"
                    if recovery_active:
                        progress_t = 1.0 - (recovery_seconds_remaining / max(recovery_seconds_target, 1e-6))
                        progress_d = 1.0 - (recovery_dist_remaining_m / max(recovery_dist_target_m, 1e-6))
                    progress = float(np.clip((progress_t + progress_d) / 2.0, 0.0, 1.0))

                    # Early recovery resembles degraded GNSS; late recovery converges to open-sky.
                    rec_sigma = (22.0 * (1.0 - progress) + 10.0 * progress) * regime_multiplier
                    obs_x = true_x + np.random.normal(0, rec_sigma)
                    obs_y = true_y + np.random.normal(0, rec_sigma)
                    hdop_low = 1.4 - (0.3 * progress)
                    hdop_high = 3.0 - (1.5 * progress)
                    hdop = np.random.uniform(max(hdop_low, 1.2), max(hdop_high, 1.4))
                    sat_low = int(np.floor(5 + 1 * progress))
                    sat_high = int(np.ceil(7 + 1 * progress))
                    sats = np.random.randint(max(sat_low, 4), max(sat_high, sat_low + 1) + 1)

                    recovery_seconds_remaining = max(0.0, recovery_seconds_remaining - ping_interval_seconds)
                    recovery_dist_remaining_m = max(0.0, recovery_dist_remaining_m - step_dist_m)
                elif portal_impair_steps_remaining > 0:
                    portal_impair_steps_remaining -= 1
                    r = np.random.rand()
                    if r < 0.40:
                        quality_state = "portal_degraded"
                        obs_x = true_x + np.random.normal(0, noise_sigma * 1.4)
                        obs_y = true_y + np.random.normal(0, noise_sigma * 1.4)
                        hdop = np.random.uniform(3.2, 8.0)
                        sats = np.random.randint(2, 5)
                    elif r < 0.85:
                        quality_state = "portal_blackout"
                        if np.random.rand() < 0.7:
                            obs_x, obs_y = np.nan, np.nan
                        else:
                            obs_x = true_x + np.random.normal(0, noise_sigma * 2.5)
                            obs_y = true_y + np.random.normal(0, noise_sigma * 2.5)
                        hdop = np.random.uniform(8.5, 10.5)
                        sats = np.random.randint(0, 2)
                    else:
                        quality_state = "portal_moderate"
                        obs_x = true_x + np.random.normal(0, noise_sigma)
                        obs_y = true_y + np.random.normal(0, noise_sigma)
                        hdop = np.random.uniform(1.6, 3.0)
                        sats = np.random.randint(5, 7)
                else:
                    obs_x = true_x + np.random.normal(0, noise_sigma)
                    obs_y = true_y + np.random.normal(0, noise_sigma)
                    low_sat, high_sat = quality_profile["open_sat_range"]
                    low_hdop, high_hdop = quality_profile["open_hdop_range"]
                    hdop = np.random.uniform(low_hdop, high_hdop)
                    if "open_sat_values" in quality_profile and "open_sat_probs" in quality_profile:
                        sats = int(np.random.choice(quality_profile["open_sat_values"], p=quality_profile["open_sat_probs"]))
                    else:
                        sats = np.random.randint(low_sat, high_sat + 1)

            prev_is_tunnel = is_tunnel

            # 4. Final Feature Vector Compilation
            all_pings.append({
                "truck_id": truck_id,
                "case_name": case_name,
                "timestamp": step,
                "actual_time": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "day_of_week": day_qubit,      # Normalized Day
                "month_of_year": month_qubit,  # Normalized Month
                "regime": regime_label,
                "time_qubit_val": time_qubit,  # Normalized Time
                "node_id": str(edge_u),
                "next_node_id": str(edge_v),
                "edge_u": str(edge_u),
                "edge_v": str(edge_v),
                # Graph identity and OSM/Geoscape provenance are distinct;
                # keeping osmid separate from edge_key keeps exact
                # directed-edge truth stable after GraphML round-tripping.
                "edge_key": str(graph_edge_key),
                "edge_osmid": str(edge_osmid),
                "obs_x": obs_x,
                "obs_y": obs_y,
                "true_x": true_x,
                "true_y": true_y,
                "true_path_distance_m": true_path_distance_m,
                "true_speed_mps": true_speed_mps,
                "hdop": hdop,
                "sat_count": sats,
                "quality_state": quality_state,
                "quality_profile_calibration_status": quality_provenance["calibration_status"],
                "position_error_calibrated": int(quality_provenance["position_error_calibrated"]),
                "true_layer": to_int_safely(edge_attrs.get('layer', 0)),
                "true_tunnel": is_tunnel,
                "true_bridge": is_bridge,
                "true_oneway": is_oneway,
                "true_truck_eligible": 1,
                "true_bearing": edge_attrs.get('bearing', 0)
            })

    # Save Results
    df = pd.DataFrame(all_pings)
    if save_output:
        output_path = trajectory_path(case_name)
        df.to_csv(output_path, index=False)
        print(f"Successfully generated {len(df)} pings.")
        print(f"Data saved to: {output_path}")

    if not df.empty:
        tunnel_pct = float((df["true_tunnel"] == 1).mean() * 100.0)
        nan_obs_pct = float((df[["obs_x", "obs_y"]].isna().any(axis=1)).mean() * 100.0)
        state_dist = (df["quality_state"].value_counts(normalize=True) * 100.0).round(2).to_dict()
        hdop_q = df["hdop"].quantile([0.1, 0.5, 0.9]).to_dict()
        sat_q = df["sat_count"].quantile([0.1, 0.5, 0.9]).to_dict()
        print(
            "Quality summary | "
            f"tunnel={tunnel_pct:.2f}% | obs_nan={nan_obs_pct:.2f}% | "
            f"state_mix={state_dist} | hdop_q={hdop_q} | sat_q={sat_q}"
        )

    return df


def _generate_case_trajectories(case_name, num_trucks, pings_per_truck, seed, ping_interval_seconds):
    graph_path = unified_graph_path(case_name)
    # scale pings-per-truck up for tunnel-heavy cases so
    # their trajectories actually span a sustained, multi-minute blackout
    # window instead of a fixed-length trajectory regardless of tunnel
    # length. Cases without meaningful tunnel presence keep the base value.
    effective_pings_per_truck = case_pings_per_truck(
        graph_path, base_pings=pings_per_truck, ping_interval_seconds=ping_interval_seconds,
    )
    df = generate_ieee_quantum_trajectories_v5(
        graph_path,
        num_trucks=num_trucks,
        pings_per_truck=effective_pings_per_truck,
        seed=seed,
        ping_interval_seconds=ping_interval_seconds,
    )
    sat = pd.to_numeric(df.get("sat_count"), errors="coerce")
    hdop = pd.to_numeric(df.get("hdop"), errors="coerce")

    healthy = ((sat >= 8) & (hdop <= 1.5)).mean() * 100.0
    moderate = ((sat.between(5, 7, inclusive="both")) & (hdop > 1.5) & (hdop <= 3.0)).mean() * 100.0
    degraded = ((sat.between(2, 4, inclusive="both")) & (hdop > 3.0) & (hdop <= 8.0)).mean() * 100.0
    blackout = ((sat <= 1) | (hdop > 8.0) | sat.isna() | hdop.isna()).mean() * 100.0

    summary = {
        "case": case_name,
        "healthy_pct": float(healthy),
        "moderate_pct": float(moderate),
        "degraded_pct": float(degraded),
        "blackout_pct": float(blackout),
        "hdop_p50": float(hdop.quantile(0.5)),
        "hdop_p90": float(hdop.quantile(0.9)),
        "hdop_p95": float(hdop.quantile(0.95)),
        "sat_p10": float(sat.quantile(0.1)),
        "sat_p50": float(sat.quantile(0.5)),
        "sat_p90": float(sat.quantile(0.9)),
    }
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic trajectories for quantum map-matching.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed for reproducible trajectory generation.")
    parser.add_argument(
        "--profile-seed",
        type=int,
        default=CALIBRATION_SEED,
        help="Seed for the quality-profile build stage.",
    )
    parser.add_argument("--num-trucks", type=int, default=100, help="Number of synthetic trucks per network.")
    parser.add_argument("--pings-per-truck", type=int, default=50, help="Number of pings per synthetic truck.")
    parser.add_argument(
        "--ping-interval-seconds",
        type=float,
        default=1.0,
        help="Ping cadence in seconds (default: 1.0, i.e. 1 Hz).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_max_workers(),
        help="Maximum parallel workers to use (default: all CPU cores).",
    )
    parser.add_argument("--skip-profile-build", action="store_true", help="Reuse an existing quality-profile file.")
    parser.add_argument("--profile-num-trucks", type=int, default=80, help="Synthetic trucks for the Domain profile audit.")
    parser.add_argument("--profile-pings-per-truck", type=int, default=140, help="Pings per truck for the Domain profile audit.")
    parser.add_argument(
        "--profile-ping-interval-seconds",
        type=float,
        default=1.0,
        help="Ping interval for the Domain profile audit.",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default=str(OUTPUT_JSON),
        help="Output JSON path for quality profiles.",
    )
    args = parser.parse_args()

    ensure_project_dirs()

    if not args.skip_profile_build:
        build_quality_profiles(
            num_trucks=int(args.profile_num_trucks),
            pings_per_truck=int(args.profile_pings_per_truck),
            ping_interval_seconds=float(args.profile_ping_interval_seconds),
            seed=int(args.profile_seed),
            max_workers=int(args.max_workers),
            output=str(args.profile_output),
        )

    case_list = list(CASE_NAMES)

    max_workers = max(1, min(int(args.max_workers), len(case_list)))
    case_summaries = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                _generate_case_trajectories,
                case_name,
                args.num_trucks,
                args.pings_per_truck,
                args.seed + idx,
                args.ping_interval_seconds,
            )
            for idx, case_name in enumerate(case_list)
        ]
        for fut in as_completed(futures):
            summary = fut.result()
            case_summaries.append(summary)
            print(f"Completed trajectory generation for: {summary['case']}")

    if case_summaries:
        metrics_df = pd.DataFrame(case_summaries).sort_values("case")
        metrics_df.to_csv(QUALITY_METRICS_BY_CASE_CSV, index=False)
        print(f"Saved per-case generated quality metrics: {QUALITY_METRICS_BY_CASE_CSV}")
        print(metrics_df.to_string(index=False))
