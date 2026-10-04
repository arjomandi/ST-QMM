"""Focused regressions for parallel-edge calibration identity.

A Rozelle trajectory fixture contains a real parallel-edge case (429 fixes on
(7455648344, 7452255175, key 1)) that a first-key fallback would mislabel.
Final graph generation deduplicates that pair, so the fixture test runs only
when both keys are available; the synthetic tests below always cover
key-aware behaviour.
"""

from pathlib import Path
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from step05_quantum_classical_benchmarking import (
    MapMatchingBenchmarker,
    extract_9_features,
    transition_features,
)
from step06_quantum_calibration_corrected import (
    add_causal_effective_observations,
    build_ranking_pairs,
    build_transition_ranking_pairs,
    edge_catalog,
    trajectory_edge_id,
    true_edge_data,
)


ROOT = Path(__file__).resolve().parent
ROZELLE_GRAPH = ROOT / "data/graphs/step01_Rozelle_Interchange_NSW_unified.graphml"
ROZELLE_TRAJECTORY = ROOT / "data/trajectories/step03_Rozelle_Interchange_NSW_trajectories_v5.csv"
PARALLEL_EDGE = ("7455648344", "7452255175", 1)


class KeyAwareTrainingRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = nx.read_graphml(ROZELLE_GRAPH)
        cls.rows = pd.read_csv(
            ROZELLE_TRAJECTORY,
            dtype={"edge_u": str, "edge_v": str},
        )

    def test_final_rozelle_graph_has_no_duplicate_directed_pairs(self):
        if not self.graph.is_multigraph():
            # A DiGraph enforces this invariant structurally.
            return
        pairs = [(str(u), str(v)) for u, v, _key in self.graph.edges(keys=True)]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_historical_429_rozelle_key1_rows_keep_exact_identity(self):
        if not self.graph.is_multigraph():
            self.skipTest("final deduplicated Rozelle graph is a DiGraph")
        if not (
            self.graph.has_edge(PARALLEL_EDGE[0], PARALLEL_EDGE[1], 0)
            and self.graph.has_edge(*PARALLEL_EDGE)
        ):
            self.skipTest("historical Rozelle parallel-edge fixture is absent")

        rows = self.rows[
            (self.rows["edge_u"] == PARALLEL_EDGE[0])
            & (self.rows["edge_v"] == PARALLEL_EDGE[1])
            & (self.rows["edge_key"].astype(int) == PARALLEL_EDGE[2])
        ]
        self.assertEqual(len(rows), 429)
        self.assertTrue(self.graph.has_edge(*PARALLEL_EDGE))
        self.assertTrue(self.graph.has_edge(PARALLEL_EDGE[0], PARALLEL_EDGE[1], 0))

        for _, row in rows.iterrows():
            self.assertEqual(trajectory_edge_id(row, self.graph), PARALLEL_EDGE)
            self.assertIs(true_edge_data(row, self.graph), self.graph.edges[PARALLEL_EDGE])

        # Step05's downstream oracle/mismatch diagnostics must use the same
        # exact truth and may not fall back to key 0 or to the reverse edge.
        bench = MapMatchingBenchmarker.__new__(MapMatchingBenchmarker)
        bench.G = self.graph
        bench.edge_ids = [
            (str(u), str(v), int(key))
            for u, v, key in self.graph.edges(keys=True)
        ]
        bench.edges_by_uv = {}
        for edge in bench.edge_ids:
            bench.edges_by_uv.setdefault(edge[:2], []).append(edge)
        bench.true_edge_col = None
        self.assertTrue(all(bench._resolve_true_edge(row) == PARALLEL_EDGE for _, row in rows.iterrows()))

    def test_parallel_key0_is_a_negative_not_the_key1_positive(self):
        graph = nx.MultiDiGraph()
        graph.add_node("u", x=0.0, y=0.0)
        graph.add_node("v", x=100.0, y=0.0)
        common = {"geometry": "LINESTRING (0 0, 100 0)", "length": 100.0}
        graph.add_edge("u", "v", key=0, **common, maxspeed=30, layer=0)
        graph.add_edge("u", "v", key=1, **common, maxspeed=100, layer=-1, tunnel=1)
        row = pd.DataFrame(
            [{
                "truck_id": "truck",
                "timestamp": 0.0,
                "edge_u": "u",
                "edge_v": "v",
                "edge_key": 1,
                "obs_x": 50.0,
                "obs_y": 0.0,
                "time_qubit_val": 0.5,
            }]
        )
        row = add_causal_effective_observations(row)
        true_rows, negative_rows, audit = build_ranking_pairs(
            row,
            graph,
            edge_catalog(graph),
            radius=60.0,
            max_negatives=1,
            return_audit=True,
        )
        self.assertEqual(audit["candidate_oracle_hits"], 1)
        self.assertEqual(audit["ranking_pairs"], 1)
        np.testing.assert_allclose(
            true_rows[0],
            extract_9_features(graph.edges["u", "v", 1], traffic_regime=0.5),
        )
        np.testing.assert_allclose(
            negative_rows[0],
            extract_9_features(graph.edges["u", "v", 0], traffic_regime=0.5),
        )

    def test_digraph_truth_uses_canonical_key_zero(self):
        graph = nx.DiGraph()
        graph.add_node("u", x=0.0, y=0.0)
        graph.add_node("v", x=1.0, y=0.0)
        graph.add_edge("u", "v", length=1.0)
        row = pd.Series({"edge_u": "u", "edge_v": "v"})
        self.assertEqual(trajectory_edge_id(row, graph), ("u", "v", 0))
        self.assertIs(true_edge_data(row, graph), graph.edges["u", "v"])

        multi = nx.MultiDiGraph(graph)
        self.assertIsNone(trajectory_edge_id(row, multi))

    def test_qts_true_transition_and_parallel_negative_are_key_aware(self):
        graph = nx.MultiDiGraph()
        graph.add_node("u", x=0.0, y=0.0)
        graph.add_node("v", x=100.0, y=0.0)
        common = {"geometry": "LINESTRING (0 0, 100 0)", "length": 100.0}
        graph.add_edge("u", "v", key=0, **common, maxspeed=30, layer=0)
        graph.add_edge("u", "v", key=1, **common, maxspeed=100, layer=-1, tunnel=1)
        graph.add_edge(
            "v",
            "u",
            key=0,
            geometry="LINESTRING (100 0, 0 0)",
            length=100.0,
            maxspeed=50,
        )
        rows = pd.DataFrame(
            [
                {"truck_id": "truck", "timestamp": 0.0, "edge_u": "u", "edge_v": "v", "edge_key": 1, "obs_x": 10.0, "obs_y": 0.0},
                {"truck_id": "truck", "timestamp": 1.0, "edge_u": "u", "edge_v": "v", "edge_key": 1, "obs_x": 20.0, "obs_y": 0.0},
            ]
        )
        true_rows, negative_rows, audit = build_transition_ranking_pairs(
            rows,
            graph,
            edge_catalog(graph),
            ["truck"],
            radius=60.0,
            max_negatives=1,
            cap=None,
            return_audit=True,
        )
        self.assertEqual(audit["previous_candidate_oracle_hits"], 1)
        self.assertEqual(audit["next_candidate_oracle_hits"], 1)
        self.assertEqual(audit["ranking_pairs"], 1)
        np.testing.assert_allclose(
            true_rows[0],
            transition_features(
                graph.edges["u", "v", 1],
                graph.edges["u", "v", 1],
                10.0,
            ),
        )
        # The same (u,v) pair at key 0 remains a distinct candidate/negative.
        self.assertFalse(np.allclose(true_rows[0], negative_rows[0]))


if __name__ == "__main__":
    unittest.main()
