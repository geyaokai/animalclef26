from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.animalclef_analysis.graph_clustering import (
    GRAPH_METHOD_CHINESE_WHISPERS,
    GRAPH_METHOD_CONNECTED_COMPONENTS,
    build_graph_adjacency,
    cluster_labels_from_score_graph,
    run_graph_threshold_sweep,
)


class GraphClusteringTest(unittest.TestCase):
    def test_build_graph_adjacency_respects_threshold_and_mutual_topk(self) -> None:
        score = np.asarray(
            [
                [1.0, 0.95, 0.90, 0.20],
                [0.95, 1.0, 0.40, 0.10],
                [0.90, 0.40, 1.0, 0.85],
                [0.20, 0.10, 0.85, 1.0],
            ],
            dtype=np.float32,
        )

        indices, _weights = build_graph_adjacency(score, threshold=0.8, top_k=2, mutual_top_k=True)

        self.assertEqual(indices[0].tolist(), [1, 2])
        self.assertEqual(indices[1].tolist(), [0])
        self.assertEqual(indices[2].tolist(), [0, 3])
        self.assertEqual(indices[3].tolist(), [2])

    def test_connected_components_finds_two_clusters(self) -> None:
        score = np.asarray(
            [
                [1.0, 0.95, 0.10, 0.05],
                [0.95, 1.0, 0.08, 0.05],
                [0.10, 0.08, 1.0, 0.92],
                [0.05, 0.05, 0.92, 1.0],
            ],
            dtype=np.float32,
        )

        labels = cluster_labels_from_score_graph(
            score_matrix=score,
            threshold=0.8,
            method=GRAPH_METHOD_CONNECTED_COMPONENTS,
        )

        self.assertEqual(labels.tolist(), [0, 0, 1, 1])

    def test_chinese_whispers_keeps_disconnected_components_separate(self) -> None:
        score = np.asarray(
            [
                [1.0, 0.95, 0.10, 0.05],
                [0.95, 1.0, 0.08, 0.05],
                [0.10, 0.08, 1.0, 0.92],
                [0.05, 0.05, 0.92, 1.0],
            ],
            dtype=np.float32,
        )

        labels = cluster_labels_from_score_graph(
            score_matrix=score,
            threshold=0.8,
            method=GRAPH_METHOD_CHINESE_WHISPERS,
            iterations=10,
            seed=42,
        )

        self.assertEqual(labels.tolist(), [0, 0, 1, 1])

    def test_run_graph_threshold_sweep_reports_metrics(self) -> None:
        df = pd.DataFrame(
            {
                "image_id": ["1", "2", "3", "4"],
                "dataset": ["SalamanderID2025"] * 4,
            }
        )
        score = np.asarray(
            [
                [1.0, 0.95, 0.10, 0.05],
                [0.95, 1.0, 0.08, 0.05],
                [0.10, 0.08, 1.0, 0.92],
                [0.05, 0.05, 0.92, 1.0],
            ],
            dtype=np.float32,
        )

        sweep_df, prediction_df = run_graph_threshold_sweep(
            df=df,
            score_matrix=score,
            thresholds=[0.8],
            method=GRAPH_METHOD_CONNECTED_COMPONENTS,
        )

        self.assertEqual(len(sweep_df), 1)
        self.assertEqual(int(sweep_df.iloc[0]["cluster_count"]), 2)
        self.assertEqual(len(prediction_df), 4)


if __name__ == "__main__":
    unittest.main()
