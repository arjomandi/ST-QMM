"""Barren-plateau diagnostic: gradient-variance scan across qubit counts and
circuit depths, following the McClean et al. (2018) protocol (cited as ref22
in the paper). For each (n_qubits, n_layers) combination, sample many random
parameter initializations, compute d<Z0>/d(theta of a fixed reference gate)
via PennyLane autodiff, and report the variance across samples. Exponential
decay of this variance with n_qubits is the standard barren-plateau signature.

This informs circuit-sizing decisions for the QMM (9-qubit) / QTS (4-qubit)
redesign: if gradient variance has already collapsed at 9 qubits / 2 layers,
adding depth or qubits will not help and a shallower/local-readout design is
warranted instead.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

QUBIT_COUNTS = [4, 6, 9, 12]
LAYER_COUNTS = [1, 2, 3]
N_SAMPLES = 60
FEATURE_SEED = 7

results = []

for n_qubits in QUBIT_COUNTS:
    dev = qml.device("default.qubit", wires=n_qubits)
    feat_rng = np.random.default_rng(FEATURE_SEED)
    features = feat_rng.uniform(0.0, 1.0, size=n_qubits)

    for n_layers in LAYER_COUNTS:
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(weights, features=features, n_qubits=n_qubits):
            qml.AngleEmbedding(pnp.pi * pnp.array(features), wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        grad_fn = qml.grad(circuit, argnum=0)

        # Track the gradient of a fixed reference parameter: layer 0, wire 0,
        # rotation index 0 (the standard single-parameter probe in the
        # McClean et al. protocol).
        rng = np.random.default_rng(1000 + n_qubits * 13 + n_layers)
        ref_grads = []
        for _ in range(N_SAMPLES):
            w = pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)
            g = grad_fn(w)
            ref_grads.append(float(np.asarray(g)[0, 0, 0]))

        ref_grads = np.asarray(ref_grads)
        variance = float(np.var(ref_grads))
        results.append({
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "grad_variance": variance,
            "grad_mean_abs": float(np.mean(np.abs(ref_grads))),
        })
        print(f"n_qubits={n_qubits:2d}  n_layers={n_layers}  "
              f"Var[dZ0/dtheta]={variance:.3e}  mean|grad|={np.mean(np.abs(ref_grads)):.3e}")

import pandas as pd
df = pd.DataFrame(results)
df.to_csv("results/reports/diagnostic_barren_plateau_scan.csv", index=False)
print("\nSaved: results/reports/diagnostic_barren_plateau_scan.csv")

print("\n--- Decay check: variance ratio vs 4-qubit baseline, per layer count ---")
for n_layers in LAYER_COUNTS:
    sub = df[df["n_layers"] == n_layers].sort_values("n_qubits")
    base = sub.iloc[0]["grad_variance"]
    ratios = [f"{row.n_qubits}q:{row.grad_variance/base:.3f}x" for row in sub.itertuples()]
    print(f"layers={n_layers}: " + "  ".join(ratios))
