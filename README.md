# ST-QMM: Quantum Feature Maps for 3D Vehicle Positioning

Reference implementation and benchmark pipeline for **"Quantum Feature Maps for 3D
Vehicle Positioning: A Hybrid Framework and Controlled Benchmark"** (Larry M.
Arjomandi), submitted to *IEEE Transactions on Intelligent Transportation Systems*.

The paper introduces **ST-QMM**, a Spatio-Temporal Quantum Map Matcher that combines
a 9-qubit edge-semantic circuit (QMM) and a 4-qubit edge-pair compatibility circuit
(QTS) with a graph-constrained Viterbi decoder, and benchmarks it against spatial and
level-aware probabilistic baselines and a capacity-matched classical control across
six complex Australian road networks.

This repository contains the full pipeline code: graph construction and auditing,
synthetic trajectory generation, quantum circuit implementation (PennyLane), offline
calibration, Viterbi decoding, and the reproducibility/statistical evaluation used to
produce every reported number and figure in the paper.

## What's included / not included

This repository ships **code only**. The audited GraphML road networks and the
trajectories generated from them are **not included**, because the networks are built
by fusing OpenStreetMap data with a commercially licensed Geoscape roads dataset
(accessed via an Austroads subscription); redistribution rights for that enrichment
have not been confirmed. If you have your own OSM and/or Geoscape access, the
`step00`–`step03` scripts below reproduce the graph and trajectory generation
end-to-end.

## Pipeline overview

Scripts are organized as a numbered pipeline. Each stage reads the previous stage's
output from `data/` or `results/` (paths are centralized in `pipeline_config.py`).

| Stage | Script | Purpose |
|---|---|---|
| 00 | `step00_pull_hard_case_maps.py` | Defines the six hard-case study areas (precise lat/lon centers) and pulls their OSM extracts. |
| 01 | `step01_reproduce_unified_graph.py` | Builds the audited, unified 3D road graphs (OSM topology + Geoscape attribute enrichment, layer/tunnel/bridge semantics). |
| 02 | `step02_check_graphml_file.py` | Validates the generated GraphML files (connectivity, directed reachability, schema). |
| 03 | `step03_generate_quantum_trajectories.py` | Generates deterministic synthetic truck trajectories with GNSS noise, blackout, and recovery modeling. |
| 03 | `step03_validate_trajectories.py` | Validates trajectories against graph topology and GNSS recovery rules. |
| 04 | `step04_viterbi_quantum_integration.py` | Standalone deterministic quantum-Viterbi matcher (single-case integration/demo path). |
| 05 | `step05_quantum_classical_benchmarking.py` | Main benchmarking module: QMM/QTS circuits, spatial HMM, Level-EKF–HMM, MLP/SVM ablation controls, matched-classical control, full ablation grid. |
| 06 | `step06_quantum_calibration_corrected.py` | Offline calibration: trains QMM/QTS and matched-classical weights via pairwise margin ranking loss; selects confidence-dip thresholds. |
| 07 | `step07_reproducibility_and_ci_evaluation.py` | Repeated held-out evaluation (20 seeds/case) producing the headline reproducibility tables and bootstrap confidence intervals. |
| 07 | `step07_statistical_analysis.py` | Wilcoxon signed-rank / Holm–Bonferroni / Hodges–Lehmann statistical analysis over the Step 07 results. |
| 08 | `step08_generate_candidates.py` | Generates edge candidate sets for all hard-case maps. |
| 09 | `step09_quantum_transition_scorer.py` | Transition-scoring visual analytics using Step 06-trained QTS weights. |

Supporting scripts:

- `pipeline_config.py` — shared paths, seeds, negative-sampling radii, and helper functions used throughout.
- `diagnostic_barren_plateau_scan.py` — gradient-variance barren-plateau diagnostic (McClean et al. protocol), referenced in Sec. III-A of the paper.
- `test_step06_key_aware_training.py` — unit tests for parallel-edge calibration identity handling.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.9.6. See `requirements.txt` for pinned dependency versions
(PennyLane 0.38, NumPy 1.26, SciPy 1.13, scikit-learn 1.6).

## Reproducing the benchmark

With your own audited GraphML networks and trajectories in place under `data/`:

```bash
python3 step06_quantum_calibration_corrected.py --seed 42 --restarts 5
python3 step07_reproducibility_and_ci_evaluation.py --runs 20 --base-seed 42 --radius 60
python3 step07_statistical_analysis.py
```

## Citation

If you use this code, please cite the paper (citation details to be added once the
DOI is assigned).

## License

Code in this repository is released under the MIT License (see `LICENSE`). This
license covers the original pipeline code only; it does not grant any rights to
third-party map data (OpenStreetMap, licensed under the Open Database License, or
Geoscape Australia data, licensed separately).
