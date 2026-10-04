import argparse
import importlib.util
import os
import json
import pickle
import tempfile
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pipeline_config import INFERENCE_CANDIDATE_RADIUS_M, REPRO_BASE_SEED, REPORTS_DIR, classical_matched_qts_weights_path, classical_matched_weights_path, confidence_threshold_path, default_max_workers, ensure_project_dirs, mlp_ablation_model_path, qts_weights_path, svm_ablation_model_path, split_calibration_and_holdout_trucks, trajectory_path, unified_graph_path, weights_path


CASE_NAMES = [
    "Rozelle_Interchange_NSW",
    "West_Gate_Tunnel_VIC",
    "NorthConnex_NSW",
    "Light_Horse_Interchange_NSW",
    "Domain_Tunnel_VIC",
    "M80_Princes_Freeway_VIC",
]


def load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rmse(values):
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(arr))))


def ci95(values):
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(arr.size))


def _require_fresh_step06_artifact(path, case_name, description, source_paths):
    """Reject missing or pre-regeneration Step06 artifacts before spawning jobs.

    The benchmark model classes intentionally support deterministic untrained
    fallbacks for diagnostic use.  That behavior is not valid for Step07: all
    reported modes must use the fixed models fitted by the current Step06 run.
    Comparing modification times here also prevents a partially failed Step06
    run from mixing newly regenerated graphs/trajectories with older weights.
    """
    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(
            f"Missing Step06 {description} for {case_name}: {artifact}. "
            "Run step06_quantum_calibration_corrected.py successfully for all cases first."
        )
    newest_source_mtime_ns = max(Path(source).stat().st_mtime_ns for source in source_paths)
    if artifact.stat().st_mtime_ns < newest_source_mtime_ns:
        raise RuntimeError(
            f"Stale Step06 {description} for {case_name}: {artifact} predates "
            "a current graph, trajectory, decoder, calibration, evaluation, or configuration source. "
            "Rerun step06_quantum_calibration_corrected.py "
            "before Step07."
        )
    return artifact


def _validate_weight_shape(path, expected_shape, case_name, description):
    try:
        actual_shape = tuple(np.load(path, mmap_mode="r").shape)
    except Exception as exc:
        raise RuntimeError(
            f"Unreadable Step06 {description} for {case_name}: {path}"
        ) from exc
    if actual_shape != tuple(expected_shape):
        raise RuntimeError(
            f"Invalid Step06 {description} shape for {case_name}: expected "
            f"{tuple(expected_shape)}, found {actual_shape} in {path}. Rerun Step06."
        )


def _validate_ablation_model(path, case_name, description):
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(
            f"Unreadable Step06 {description} for {case_name}: {path}"
        ) from exc
    expected_supervision = "fix-level correctness within inference-style local candidate sets"
    selection = payload.get("selection", {}) if isinstance(payload, dict) else {}
    if (
        not isinstance(payload, dict)
        or payload.get("model") is None
        or selection.get("supervision") != expected_supervision
    ):
        raise RuntimeError(
            f"Stale or invalid Step06 {description} for {case_name}: {path}. "
            "Rerun step06_quantum_calibration_corrected.py before Step07."
        )


EVAL_MODES = (
    "quantum", "classical", "level_ekf_hmm", "mlp", "svm",
    # Ablation grid and controls:
    # qmm_entanglement = HMM+QMM only, qts_only = HMM+QTS only, quantum =
    # HMM+QMM+QTS (already present above), classical = HMM alone (already
    # present above) — together the full 2x2 grid. frozen and
    # classical_matched are controls; level_ekf_hmm_quantum plugs the QMM/QTS
    # factors into the one baseline that otherwise never sees them.
    "qmm_entanglement", "qts_only", "frozen", "classical_matched", "level_ekf_hmm_quantum",
)


def _run_single_eval(graph_path, traj_path, weights_file, qts_weights_file, confidence_threshold, seed, radius, fallback_k, max_points, run_index=None, classical_matched_weights_file=None, classical_matched_qts_weights_file=None, mlp_model_file=None, svm_model_file=None):
    base_dir = Path(__file__).resolve().parent
    bench_module = load_module(base_dir / "step05_quantum_classical_benchmarking.py", "benchmark_module_worker")
    benchmarker_cls = bench_module.MapMatchingBenchmarker

    bench_traj_path = Path(traj_path)
    temp_path = None
    if max_points is not None:
        try:
            src_df = pd.read_csv(traj_path)
            if "truck_id" in src_df.columns and src_df["truck_id"].nunique() > 1:
                # A streaming matcher must be evaluated on intact trajectories,
                # not shuffled fixes from several vehicles. Hold out one
                # complete truck per seed for decoding; the ablation-grid
                # MLP/SVM are fit once on the calibration pool by step06 and
                # loaded via mlp_model_file/svm_model_file below, exactly
                # like QMM/QTS, so no training set is built here.
                #
                # The held-out truck is drawn only from this case's holdout pool
                # (disjoint from step06's calibration/validation trucks) and selected by run_index into a fixed permutation
                # of that pool, so distinct runs get distinct trucks rather than
                # independent draws that can repeat across the 20 seeds.
                _calibration_trucks, holdout_trucks = split_calibration_and_holdout_trucks(
                    src_df["truck_id"].dropna().astype(str).unique()
                )
                if run_index is None:
                    sample_rng = np.random.default_rng(int(seed) + 7919)
                    held_out_truck = sample_rng.choice(holdout_trucks)
                else:
                    perm_rng = np.random.default_rng(900011)
                    order = perm_rng.permutation(len(holdout_trucks))
                    held_out_truck = holdout_trucks[order[int(run_index) % len(holdout_trucks)]]
                truck_id_str = src_df["truck_id"].astype(str)
                sampled_df = src_df[truck_id_str == held_out_truck].copy()
                fd, tmp_name = tempfile.mkstemp(prefix="step07_heldout_truck_", suffix=".csv")
                os.close(fd)
                temp_path = Path(tmp_name)
                sampled_df.to_csv(temp_path, index=False)
                bench_traj_path = temp_path
            elif len(src_df) > int(max_points):
                sample_rng = np.random.default_rng(int(seed) + 7919)
                keep_idx = np.sort(sample_rng.choice(len(src_df), size=int(max_points), replace=False))
                sampled_df = src_df.iloc[keep_idx].copy()
                fd, tmp_name = tempfile.mkstemp(prefix="step07_sampled_", suffix=".csv")
                os.close(fd)
                temp_path = Path(tmp_name)
                sampled_df.to_csv(temp_path, index=False)
                bench_traj_path = temp_path
        except Exception as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise RuntimeError(
                f"Could not construct the truck-disjoint Step07 evaluation subset from {traj_path}. "
                "Refusing to fall back silently to the full trajectory because that would mix "
                "calibration and held-out trucks."
            ) from exc

    bench = benchmarker_cls(
        graph_path,
        bench_traj_path,
        seed=seed,
        candidate_radius=radius,
        candidate_fallback_k=fallback_k,
        beam_width=30,
        weights_file=str(weights_file),
        qts_weights_file=str(qts_weights_file) if qts_weights_file else None,
        classical_matched_weights_file=str(classical_matched_weights_file) if classical_matched_weights_file else None,
        classical_matched_qts_weights_file=str(classical_matched_qts_weights_file) if classical_matched_qts_weights_file else None,
        confidence_dip_threshold=float(confidence_threshold),
        mlp_model_file=str(mlp_model_file) if mlp_model_file else None,
        svm_model_file=str(svm_model_file) if svm_model_file else None,
    )

    paths = {}
    latency_ms = {}
    for mode in EVAL_MODES:
        # Cold, mode-isolated timing.  In particular, full ST-QMM must not
        # populate entangled-QMM/QTS caches that make later ablation cells
        # appear artificially faster.  Geometry construction is common
        # preprocessing and remains precomputed for every mode.
        for cache_name in (
            "_quantum_score_cache",
            "_transition_hop_cache",
            "_edge_length_cache",
        ):
            getattr(bench, cache_name).clear()
        started = time.perf_counter()
        paths[mode] = bench.run_viterbi(mode=mode, progress_every=0)
        elapsed = time.perf_counter() - started
        if mode in ("level_ekf_hmm", "level_ekf_hmm_quantum"):
            elapsed += float(bench.level_precompute_seconds)
        latency_ms[mode] = 1000.0 * elapsed / max(1, len(bench.df))
    q_path = paths["quantum"]
    c_path = paths["classical"]
    _metrics_df, summary = bench.analyze_results(q_path, c_path)

    q_rmse = float(summary["quantum_rmse"])
    c_rmse = float(summary["classical_rmse"])
    gain = float(summary["rmse_gain"])
    mode_rmse = {
        mode: rmse([bench._point_error(edge, idx) for idx, edge in enumerate(path)])
        for mode, path in paths.items()
    }
    advanced = {
        mode: bench.compute_advanced_metrics(path)
        for mode, path in paths.items()
    }
    per_fix_frames = []
    per_fix_by_mode = {}
    for mode, path in paths.items():
        frame = bench.per_fix_diagnostics(mode, path)
        frame.insert(0, "seed", int(seed))
        frame.insert(0, "run_index", int(run_index) if run_index is not None else -1)
        per_fix_frames.append(frame)
        per_fix_by_mode[mode] = frame

    if temp_path is not None:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

    record = {
        "quantum_rmse": q_rmse,
        "classical_rmse": c_rmse,
        "rmse_gain": gain,
        "latency_cache_policy": "cold mode-isolated caches; common geometry precomputed",
    }
    for mode in paths:
        record[f"{mode}_rmse"] = mode_rmse[mode]
        record[f"{mode}_latency_ms_per_fix"] = latency_ms[mode]
        record[f"{mode}_edge_accuracy"] = advanced[mode]["edge_accuracy"]
        record[f"{mode}_layer_accuracy"] = advanced[mode]["z_layer_accuracy"]
        record[f"{mode}_route_mismatch_rate"] = 100.0 - advanced[mode]["edge_accuracy"]
        record[f"{mode}_impossible_jumps"] = advanced[mode]["impossible_jumps"]
        record[f"{mode}_relaxed_adjacent_accuracy"] = advanced[mode]["relaxed_adjacent_accuracy"]
        record[f"{mode}_same_carriageway_accuracy"] = advanced[mode]["same_carriageway_accuracy"]
        record[f"{mode}_physical_segment_accuracy"] = advanced[mode]["physical_segment_accuracy"]
        record[f"{mode}_reverse_directed_rate"] = advanced[mode]["reverse_directed_rate"]
        record[f"{mode}_layer_aware_rmse"] = advanced[mode]["layer_aware_rmse"]
        mode_fix = per_fix_by_mode[mode]
        record[f"{mode}_candidate_oracle_recall"] = float(
            100.0 * mode_fix["candidate_contains_true"].mean()
        ) if not mode_fix.empty else 0.0
        record[f"{mode}_strict_radius_oracle_recall"] = float(
            100.0 * mode_fix["strict_radius_contains_true"].mean()
        ) if not mode_fix.empty else 0.0
        record[f"{mode}_post_beam_oracle_recall"] = float(
            100.0 * mode_fix["beam_contains_true"].mean()
        ) if not mode_fix.empty else 0.0
        record[f"{mode}_candidate_fallback_rate"] = float(
            100.0 * mode_fix["candidate_fallback_used"].mean()
        ) if not mode_fix.empty else 0.0
        record[f"{mode}_candidate_unlocalized_rate"] = float(
            100.0 * mode_fix["candidate_unlocalized"].mean()
        ) if not mode_fix.empty else 0.0
        record[f"{mode}_adjacent_nonexact_rate"] = float(
            100.0 * mode_fix["adjacent_nonexact_match"].mean()
        ) if not mode_fix.empty else 0.0
        record[f"{mode}_trellis_breaks"] = int(mode_fix["trellis_break"].sum()) if not mode_fix.empty else 0
        record[f"{mode}_connectivity_reanchors"] = int(
            mode_fix["connectivity_reanchor"].sum()
        ) if not mode_fix.empty else 0
        record[f"{mode}_imputed_fix_rate"] = float(
            100.0 * mode_fix["position_imputed"].mean()
        ) if not mode_fix.empty else 0.0
    record["_per_fix_rows"] = pd.concat(per_fix_frames, ignore_index=True).to_dict("records")
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Run repeated calibrated quantum-vs-classical held-out subsample benchmark with 95% confidence intervals."
    )
    parser.add_argument("--runs", type=int, default=20, help="Number of repeated runs with distinct seeds.")
    parser.add_argument("--base-seed", type=int, default=REPRO_BASE_SEED, help="Starting seed; run i uses base-seed + i.")
    parser.add_argument("--graph", type=str, default=None, help="GraphML filename for a single-case run.")
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help="Trajectory CSV filename for a single-case run.",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default=None,
        help="Case label used to resolve Step06 artifacts in single-case mode; inferred from a canonical graph filename when omitted.",
    )
    parser.add_argument("--radius", type=float, default=INFERENCE_CANDIDATE_RADIUS_M, help="Candidate edge search radius (meters).")
    parser.add_argument("--fallback-k", type=int, default=10, help="Fallback candidate count.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=350,
        help="Fallback sample cap when truck IDs are unavailable; production runs retain one complete held-out truck.",
    )
    parser.add_argument(
        "--regenerate-trajectories-per-run",
        action="store_true",
        help="Reserved for future full-pipeline trajectory regeneration; current runs use seeded held-out subsamples.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_max_workers(),
        help="Maximum parallel workers to use (default: all CPU cores).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    ensure_project_dirs()

    bench_module = load_module(base_dir / "step05_quantum_classical_benchmarking.py", "benchmark_module")
    benchmarker_cls = bench_module.MapMatchingBenchmarker

    if (args.graph is None) ^ (args.trajectory is None):
        raise ValueError("Provide both --graph and --trajectory for single-case mode, or neither to run all hard cases.")

    if args.graph and args.trajectory:
        single_case_name = (
            args.case_name
            or Path(args.graph).stem.removeprefix("step01_").removesuffix("_unified")
        )
        targets = [(single_case_name, args.graph, args.trajectory)]
    else:
        targets = [
            (case_name, f"{case_name}_unified.graphml", f"{case_name}_trajectories_v5.csv")
            for case_name in CASE_NAMES
        ]

    records = []
    per_fix_records = []
    case_targets = []
    for case_name, graph_name, traj_name in targets:
        if args.graph and args.trajectory:
            graph_arg = Path(graph_name)
            traj_arg = Path(traj_name)
            graph_path = graph_arg if graph_arg.is_absolute() else Path(__file__).resolve().parent / graph_arg
            traj_path = traj_arg if traj_arg.is_absolute() else Path(__file__).resolve().parent / traj_arg
        else:
            case_name = graph_name.replace("_unified.graphml", "")
            graph_path = unified_graph_path(case_name)
            traj_path = trajectory_path(case_name)
        if not graph_path.exists() or not traj_path.exists():
            print(f"Skipping {case_name}: missing graph or trajectory file.")
            continue
        source_paths = (
            graph_path,
            traj_path,
            base_dir / "step05_quantum_classical_benchmarking.py",
            base_dir / "step06_quantum_calibration_corrected.py",
            base_dir / "step07_reproducibility_and_ci_evaluation.py",
            base_dir / "pipeline_config.py",
        )
        calibrated_weights = _require_fresh_step06_artifact(
            weights_path(case_name), case_name, "QMM weights", source_paths
        )
        _validate_weight_shape(calibrated_weights, (2, 9, 3), case_name, "QMM weights")

        threshold_file = _require_fresh_step06_artifact(
            confidence_threshold_path(case_name), case_name, "confidence threshold", source_paths
        )
        with open(threshold_file, "r", encoding="utf-8") as handle:
            threshold_payload = json.load(handle)
        expected_threshold_semantics = (
            "maximum QMM expectation over inference-style candidates at noisy/causally-imputed XY"
        )
        if threshold_payload.get("score_semantics") != expected_threshold_semantics:
            raise RuntimeError(
                f"Stale confidence calibration for {case_name}: rerun "
                "step06_quantum_calibration_corrected.py before Step07."
            )
        if threshold_payload.get("case") != case_name:
            raise RuntimeError(
                f"Step06 confidence calibration case mismatch for {case_name}: "
                f"found {threshold_payload.get('case')!r} in {threshold_file}."
            )
        confidence_threshold = float(threshold_payload["threshold"])
        # Retain a disabled sentinel for chance-level confidence diagnostics.
        # Confidence does not alter candidates or variances in headline runs.
        if threshold_payload.get("chance_level_detector", False):
            confidence_threshold = -1.000001
        calibrated_qts_weights = _require_fresh_step06_artifact(
            qts_weights_path(case_name), case_name, "QTS weights", source_paths
        )
        _validate_weight_shape(calibrated_qts_weights, (2, 4, 3), case_name, "QTS weights")
        calibrated_classical_matched = _require_fresh_step06_artifact(
            classical_matched_weights_path(case_name),
            case_name,
            "matched-classical emission weights",
            source_paths,
        )
        _validate_weight_shape(
            calibrated_classical_matched, (56,), case_name, "matched-classical emission weights"
        )
        calibrated_classical_matched_qts = _require_fresh_step06_artifact(
            classical_matched_qts_weights_path(case_name),
            case_name,
            "matched-classical transition weights",
            source_paths,
        )
        _validate_weight_shape(
            calibrated_classical_matched_qts,
            (25,),
            case_name,
            "matched-classical transition weights",
        )
        calibrated_mlp = _require_fresh_step06_artifact(
            mlp_ablation_model_path(case_name), case_name, "MLP ablation model", source_paths
        )
        _validate_ablation_model(calibrated_mlp, case_name, "MLP ablation model")
        calibrated_svm = _require_fresh_step06_artifact(
            svm_ablation_model_path(case_name), case_name, "SVM ablation model", source_paths
        )
        _validate_ablation_model(calibrated_svm, case_name, "SVM ablation model")
        case_targets.append((
            case_name, graph_path, traj_path, calibrated_weights, calibrated_qts_weights, confidence_threshold,
            calibrated_classical_matched, calibrated_classical_matched_qts, calibrated_mlp, calibrated_svm,
        ))

    jobs = []
    # Interleave seeds across all cases so progress is balanced per-case.
    # run_index (0..runs-1) selects a distinct held-out truck per run via a
    # fixed permutation of the case's holdout pool —
    # independent of `seed`, which still drives other run-to-run randomness
    # (e.g. classical-baseline restart selection).
    for i in range(args.runs):
        seed = int(args.base_seed) + i
        for (case_name, graph_path, traj_path, calibrated_weights, calibrated_qts_weights, confidence_threshold,
             calibrated_classical_matched, calibrated_classical_matched_qts, calibrated_mlp, calibrated_svm) in case_targets:
            jobs.append((
                case_name, graph_path, traj_path, calibrated_weights, calibrated_qts_weights, confidence_threshold,
                seed, i, calibrated_classical_matched, calibrated_classical_matched_qts, calibrated_mlp, calibrated_svm,
            ))

    if jobs:
        max_workers = max(1, min(int(args.max_workers), len(jobs)))

        # Avoid multiprocessing spawn overhead/hangs on constrained environments.
        if max_workers == 1:
            for (case_name, graph_path, traj_path, calibrated_weights, calibrated_qts_weights, confidence_threshold,
                 seed, run_index, calibrated_classical_matched, calibrated_classical_matched_qts, calibrated_mlp, calibrated_svm) in jobs:
                result = _run_single_eval(
                    graph_path,
                    traj_path,
                    calibrated_weights,
                    calibrated_qts_weights,
                    confidence_threshold,
                    seed,
                    args.radius,
                    args.fallback_k,
                    args.max_points,
                    run_index=run_index,
                    classical_matched_weights_file=calibrated_classical_matched,
                    classical_matched_qts_weights_file=calibrated_classical_matched_qts,
                    mlp_model_file=calibrated_mlp,
                    svm_model_file=calibrated_svm,
                )
                print(f"[{case_name}] completed seed={seed}")
                per_fix = result.pop("_per_fix_rows", [])
                per_fix_records.extend({"case": case_name, **row} for row in per_fix)
                records.append({"case": case_name, "seed": seed, **result})
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                future_map = {
                    ex.submit(
                        _run_single_eval,
                        graph_path,
                        traj_path,
                        calibrated_weights,
                        calibrated_qts_weights,
                        confidence_threshold,
                        seed,
                        args.radius,
                        args.fallback_k,
                        args.max_points,
                        run_index,
                        calibrated_classical_matched,
                        calibrated_classical_matched_qts,
                        calibrated_mlp,
                        calibrated_svm,
                    ): (case_name, seed)
                    for (case_name, graph_path, traj_path, calibrated_weights, calibrated_qts_weights, confidence_threshold,
                         seed, run_index, calibrated_classical_matched, calibrated_classical_matched_qts, calibrated_mlp, calibrated_svm) in jobs
                }
                for fut in as_completed(future_map):
                    case_name, seed = future_map[fut]
                    result = fut.result()
                    print(f"[{case_name}] completed seed={seed}")
                    per_fix = result.pop("_per_fix_rows", [])
                    per_fix_records.extend({"case": case_name, **row} for row in per_fix)
                    records.append({"case": case_name, "seed": seed, **result})

    out_df = pd.DataFrame(records)

    if out_df.empty:
        print("No valid case data found. Nothing to summarize.")
        return

    grouped = out_df.groupby("case", as_index=False).agg(
        quantum_rmse_mean=("quantum_rmse", "mean"),
        classical_rmse_mean=("classical_rmse", "mean"),
        level_ekf_hmm_rmse_mean=("level_ekf_hmm_rmse", "mean"),
        mlp_rmse_mean=("mlp_rmse", "mean"),
        svm_rmse_mean=("svm_rmse", "mean"),
        rmse_gain_mean=("rmse_gain", "mean"),
        quantum_edge_accuracy_mean=("quantum_edge_accuracy", "mean"),
        quantum_physical_segment_accuracy_mean=("quantum_physical_segment_accuracy", "mean"),
        quantum_relaxed_adjacent_accuracy_mean=("quantum_relaxed_adjacent_accuracy", "mean"),
        quantum_layer_accuracy_mean=("quantum_layer_accuracy", "mean"),
        quantum_reverse_directed_rate_mean=("quantum_reverse_directed_rate", "mean"),
        quantum_candidate_oracle_recall_mean=("quantum_candidate_oracle_recall", "mean"),
        quantum_strict_radius_oracle_recall_mean=("quantum_strict_radius_oracle_recall", "mean"),
        quantum_post_beam_oracle_recall_mean=("quantum_post_beam_oracle_recall", "mean"),
        quantum_candidate_fallback_rate_mean=("quantum_candidate_fallback_rate", "mean"),
        quantum_candidate_unlocalized_rate_mean=("quantum_candidate_unlocalized_rate", "mean"),
        quantum_connectivity_reanchors_mean=("quantum_connectivity_reanchors", "mean"),
        classical_edge_accuracy_mean=("classical_edge_accuracy", "mean"),
        classical_physical_segment_accuracy_mean=("classical_physical_segment_accuracy", "mean"),
        classical_relaxed_adjacent_accuracy_mean=("classical_relaxed_adjacent_accuracy", "mean"),
        classical_layer_accuracy_mean=("classical_layer_accuracy", "mean"),
        classical_reverse_directed_rate_mean=("classical_reverse_directed_rate", "mean"),
        classical_candidate_oracle_recall_mean=("classical_candidate_oracle_recall", "mean"),
        classical_strict_radius_oracle_recall_mean=("classical_strict_radius_oracle_recall", "mean"),
        classical_post_beam_oracle_recall_mean=("classical_post_beam_oracle_recall", "mean"),
        classical_connectivity_reanchors_mean=("classical_connectivity_reanchors", "mean"),
        level_ekf_hmm_edge_accuracy_mean=("level_ekf_hmm_edge_accuracy", "mean"),
        level_ekf_hmm_physical_segment_accuracy_mean=("level_ekf_hmm_physical_segment_accuracy", "mean"),
        level_ekf_hmm_relaxed_adjacent_accuracy_mean=("level_ekf_hmm_relaxed_adjacent_accuracy", "mean"),
        level_ekf_hmm_layer_accuracy_mean=("level_ekf_hmm_layer_accuracy", "mean"),
        level_ekf_hmm_route_mismatch_rate_mean=("level_ekf_hmm_route_mismatch_rate", "mean"),
        level_ekf_hmm_latency_ms_per_fix_mean=("level_ekf_hmm_latency_ms_per_fix", "mean"),
        classical_latency_ms_per_fix_mean=("classical_latency_ms_per_fix", "mean"),
        quantum_latency_ms_per_fix_mean=("quantum_latency_ms_per_fix", "mean"),
    )

    ci_rows = []
    for case_name, grp in out_df.groupby("case"):
        ci_rows.append(
            {
                "case": case_name,
                "quantum_rmse_ci95": ci95(grp["quantum_rmse"]),
                "classical_rmse_ci95": ci95(grp["classical_rmse"]),
                "level_ekf_hmm_rmse_ci95": ci95(grp["level_ekf_hmm_rmse"]),
                "mlp_rmse_ci95": ci95(grp["mlp_rmse"]),
                "svm_rmse_ci95": ci95(grp["svm_rmse"]),
                "rmse_gain_ci95": ci95(grp["rmse_gain"]),
                "gain_positive_rate": float((grp["rmse_gain"] > 0).mean()),
            }
        )
    ci_df = pd.DataFrame(ci_rows)
    summary_df = grouped.merge(ci_df, on="case", how="left")

    print("\n=== REPEATED-RUN SUMMARY (95% CI) BY CASE ===")
    print(summary_df.to_string(index=False))

    out_csv = REPORTS_DIR / "step07_benchmark_reproducibility_model_seed_summary.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved per-run metrics: {out_csv}")

    per_fix_csv = REPORTS_DIR / "step07_benchmark_per_fix_paths.csv"
    pd.DataFrame(per_fix_records).to_csv(per_fix_csv, index=False)
    print(f"Saved per-fix paths and diagnostics: {per_fix_csv}")

    out_summary_csv = REPORTS_DIR / "step07_benchmark_reproducibility_model_seed_summary_by_case.csv"
    summary_df.to_csv(out_summary_csv, index=False)
    print(f"Saved per-case summary: {out_summary_csv}")

    metadata = {
        "runs": int(args.runs),
        "base_seed": int(args.base_seed),
        "radius": float(args.radius),
        "fallback_k": int(args.fallback_k),
        "max_points": int(args.max_points) if args.max_points is not None else None,
        "regenerate_trajectories_per_run": bool(args.regenerate_trajectories_per_run),
        "mode": "validation-selected-calibrated-weights-with-seeded-complete-held-out-truck-trajectories-and-complementary-classical-training",
        "weights_source": "Step06 minimum held-out validation pairwise hinge-ranking-loss restart per case",
        "confidence_threshold_source": "per-case Step06 held-out validation balanced-accuracy selection; benchmark trajectories excluded",
        "confidence_score_semantics": "maximum QMM expectation over inference-style candidates at noisy/causally-imputed XY",
        "per_fix_output": str(per_fix_csv),
        "classical_selection": "five MLP configurations and five RBF-SVM configurations selected by balanced accuracy using trucks disjoint from each complete held-out evaluation trajectory",
        "level_ekf_hmm": "causal graph-constrained soft road-level probability filter feeding a constant-velocity EKF and HMM/Viterbi decoder; true_layer is evaluation-only",
    }
    metadata_json = REPORTS_DIR / "step07_benchmark_reproducibility_model_seed_metadata.json"
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved reproducibility metadata: {metadata_json}")


if __name__ == "__main__":
    main()
