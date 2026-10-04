"""Persisted statistical analysis of the paired benchmark results.

Computes the Wilcoxon signed-rank tests with Holm correction, plus
matched-pairs rank-biserial correlation, a
Hodges-Lehmann estimator with a bootstrap CI, and a post-hoc
detectable-effect-at-80%-power analysis — plus an explicit, reported
definition of the Holm family, instead of silently correcting only the
headline comparison.

Reads results/reports/step07_benchmark_reproducibility_model_seed_summary.csv
(the per-seed records written by step07_reproducibility_and_ci_evaluation.py,
which includes a `{mode}_rmse` column for every ablation/control mode in
EVAL_MODES).
"""
import numpy as np
import pandas as pd
from scipy import stats

from pipeline_config import REPORTS_DIR

CASE_NAMES = [
    "Rozelle_Interchange_NSW",
    "West_Gate_Tunnel_VIC",
    "NorthConnex_NSW",
    "Light_Horse_Interchange_NSW",
    "Domain_Tunnel_VIC",
    "M80_Princes_Freeway_VIC",
]

BASELINE_MODE = "classical"
# Every mode compared against the plain spatial HMM baseline. "quantum" (full
# ST-QMM) is the paper's headline comparison; the rest are the ablation grid,
# the controls, and the other baselines in the benchmark.
TREATMENT_MODES = [
    "quantum", "level_ekf_hmm", "level_ekf_hmm_quantum", "mlp", "svm",
    "qmm_entanglement", "qts_only", "frozen", "classical_matched",
]
DIRECT_COMPARISONS = [
    ("level_ekf_hmm", "level_ekf_hmm_quantum"),
]
ALPHA = 0.05
POWER_TARGET = 0.8
BOOTSTRAP_SAMPLES = 2000


def holm_correction(p_values):
    """Holm step-down correction. Returns adjusted p-values in the original
    (unsorted) order, each already accumulated with the step-down max so
    adjusted p-values are monotone non-decreasing in the sorted order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def rank_biserial_correlation(gains):
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank
    test: r = (W+ - W-) / (W+ + W-) over signed ranks of the nonzero gains."""
    gains = np.asarray(gains, dtype=float)
    nonzero = gains[gains != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    w_pos = ranks[nonzero > 0].sum()
    w_neg = ranks[nonzero < 0].sum()
    total = w_pos + w_neg
    return float((w_pos - w_neg) / total) if total > 0 else 0.0


def hodges_lehmann_estimator(gains, n_bootstrap=BOOTSTRAP_SAMPLES, seed=0):
    """Median of all pairwise Walsh averages (x_i + x_j)/2, i <= j — the
    standard one-sample/paired Hodges-Lehmann location estimator — with a
    percentile bootstrap CI over the paired sample."""
    gains = np.asarray(gains, dtype=float)
    n = len(gains)
    i_idx, j_idx = np.triu_indices(n)
    estimate = float(np.median((gains[i_idx] + gains[j_idx]) / 2.0))

    rng = np.random.default_rng(seed)
    boot_estimates = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = gains[rng.integers(0, n, size=n)]
        si, sj = np.triu_indices(n)
        boot_estimates[b] = np.median((sample[si] + sample[sj]) / 2.0)
    lo, hi = np.percentile(boot_estimates, [2.5, 97.5])
    return estimate, float(lo), float(hi)


def _power_at_shift(residuals, shift, trials, alpha, rng):
    n = len(residuals)
    successes = 0
    for _ in range(trials):
        sample = residuals[rng.integers(0, n, size=n)] + shift
        if np.allclose(sample, 0.0):
            continue
        try:
            _, p = stats.wilcoxon(sample, alternative="two-sided")
        except ValueError:
            continue
        if p < alpha:
            successes += 1
    return successes / trials


def detectable_effect_at_power(gains, power=POWER_TARGET, alpha=ALPHA, trials=200, max_bisections=16, seed=0):
    """Post-hoc detectable-effect analysis: the smallest constant shift
    added to the observed paired-difference distribution's residuals (gains
    minus their own median, so the search adds a pure location shift on top
    of the actually-observed noise/variability) that would push the
    Wilcoxon test's power to `power` at this sample size — found by
    bisection over simulated power. Reports NaN if not reached within a
    generous bracket (power does not reach target even at a large shift).
    """
    gains = np.asarray(gains, dtype=float)
    n = len(gains)
    if n < 2:
        return float("nan")
    residuals = gains - np.median(gains)
    spread = float(np.std(gains)) + 1e-9
    rng = np.random.default_rng(seed)

    lo, hi = 0.0, 4.0 * spread
    if _power_at_shift(residuals, hi, trials, alpha, rng) < power:
        return float("nan")
    for _ in range(max_bisections):
        mid = 0.5 * (lo + hi)
        if _power_at_shift(residuals, mid, trials, alpha, rng) >= power:
            hi = mid
        else:
            lo = mid
    return float(hi)


def analyze():
    seed_csv = REPORTS_DIR / "step07_benchmark_reproducibility_model_seed_summary.csv"
    if not seed_csv.exists():
        raise FileNotFoundError(
            f"{seed_csv} not found — run step07_reproducibility_and_ci_evaluation.py first."
        )
    df = pd.read_csv(seed_csv)

    rows = []
    for case in CASE_NAMES:
        case_df = df[df["case"] == case]
        if case_df.empty:
            continue
        comparisons = [(BASELINE_MODE, mode) for mode in TREATMENT_MODES]
        comparisons.extend(DIRECT_COMPARISONS)
        for baseline_mode, mode in comparisons:
            baseline_col = f"{baseline_mode}_rmse"
            treatment_col = f"{mode}_rmse"
            if baseline_col not in case_df.columns or treatment_col not in case_df.columns:
                continue
            baseline_rmse = case_df[baseline_col].to_numpy(dtype=float)
            gains = baseline_rmse - case_df[treatment_col].to_numpy(dtype=float)
            if np.allclose(gains, gains[0]) and np.allclose(gains, 0.0):
                w_stat, p_value = np.nan, 1.0
            else:
                try:
                    w_stat, p_value = stats.wilcoxon(gains, alternative="two-sided")
                except ValueError:
                    # All-zero or otherwise degenerate paired differences.
                    w_stat, p_value = np.nan, 1.0
            r_rb = rank_biserial_correlation(gains)
            hl_estimate, hl_lo, hl_hi = hodges_lehmann_estimator(gains)
            detectable = detectable_effect_at_power(gains)
            rows.append({
                "case": case,
                "comparison": f"{baseline_mode}_vs_{mode}",
                "baseline_mode": baseline_mode,
                "treatment_mode": mode,
                "n": len(gains),
                "mean_gain_m": float(np.mean(gains)),
                "median_gain_m": float(np.median(gains)),
                "wilcoxon_w": float(w_stat) if np.isfinite(w_stat) else np.nan,
                "wilcoxon_p": float(p_value),
                "rank_biserial_r": r_rb,
                "hodges_lehmann_m": hl_estimate,
                "hodges_lehmann_ci95_lo": hl_lo,
                "hodges_lehmann_ci95_hi": hl_hi,
                "detectable_gain_at_80pct_power_m": detectable,
                "positive_rate": float(np.mean(gains > 0)),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        print("No comparisons could be computed — check that the seed summary CSV has the expected columns.")
        return result

    # Two explicitly reported Holm families: the primary ST-QMM vs HMM
    # comparison, and the full family including every baseline and control.
    primary_mask = result["comparison"] == f"{BASELINE_MODE}_vs_quantum"
    result["holm_p_primary_family"] = np.nan
    result.loc[primary_mask, "holm_p_primary_family"] = holm_correction(
        result.loc[primary_mask, "wilcoxon_p"].to_numpy()
    )
    result["holm_p_full_family"] = holm_correction(result["wilcoxon_p"].to_numpy())
    ekf_mask = result["comparison"] == "level_ekf_hmm_vs_level_ekf_hmm_quantum"
    result["holm_p_ekf_family"] = np.nan
    result.loc[ekf_mask, "holm_p_ekf_family"] = holm_correction(
        result.loc[ekf_mask, "wilcoxon_p"].to_numpy()
    )
    result["significant_primary_family_alpha05"] = result["holm_p_primary_family"] < ALPHA
    result["significant_ekf_family_alpha05"] = result["holm_p_ekf_family"] < ALPHA
    result["significant_full_family_alpha05"] = result["holm_p_full_family"] < ALPHA

    out_csv = REPORTS_DIR / "step07_statistical_analysis.csv"
    result.to_csv(out_csv, index=False)
    pd.set_option("display.width", 220)
    print(result.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(
        f"\nFamily sizes — primary (ST-QMM vs HMM, {ALPHA} FWER): {primary_mask.sum()}; "
        f"Level-EKF quantum add-on: {ekf_mask.sum()}; full: {len(result)}"
    )
    return result


if __name__ == "__main__":
    analyze()
