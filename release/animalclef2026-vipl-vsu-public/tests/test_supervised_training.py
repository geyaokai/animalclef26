from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    HAS_TORCH = False

if HAS_TORCH:
    from src.animalclef_analysis.supervised_training import (
        ArcFaceHead,
        attach_training_labels,
        build_best_metric_summary,
        compute_masked_supcon_loss,
        compute_relation_distillation_loss,
        scale_learning_rate,
        write_training_monitor_plots,
    )


@unittest.skipUnless(HAS_TORCH, "torch is required for supervised training tests")
class SupervisedTrainingHelpersTest(unittest.TestCase):
    def test_attach_training_labels_builds_dataset_local_indices(self) -> None:
        fit_df = pd.DataFrame(
            {
                "image_id": ["1", "2", "3", "4", "5"],
                "dataset": ["LynxID2025", "LynxID2025", "SalamanderID2025", "SalamanderID2025", "SalamanderID2025"],
                "identity": ["a", "b", "x", "x", "y"],
                "recommended_model_input_path_v1": [f"img_{i}.jpg" for i in range(5)],
            }
        )
        labeled_df, label_maps, class_summary_df = attach_training_labels(fit_df)
        self.assertEqual(label_maps["LynxID2025"], {"a": 0, "b": 1})
        self.assertEqual(label_maps["SalamanderID2025"], {"x": 0, "y": 1})
        self.assertEqual(set(labeled_df["dataset_index"].tolist()), {0, 1})
        self.assertEqual(int(class_summary_df[class_summary_df["dataset"] == "SalamanderID2025"]["classes"].iloc[0]), 2)

    def test_attach_training_labels_only_builds_heads_for_present_datasets(self) -> None:
        fit_df = pd.DataFrame(
            {
                "image_id": ["1", "2"],
                "dataset": ["LynxID2025", "LynxID2025"],
                "identity": ["a", "b"],
                "recommended_model_input_path_v1": ["img_1.jpg", "img_2.jpg"],
            }
        )
        labeled_df, label_maps, class_summary_df = attach_training_labels(fit_df)
        self.assertEqual(list(label_maps.keys()), ["LynxID2025"])
        self.assertEqual(class_summary_df["dataset"].tolist(), ["LynxID2025"])
        self.assertEqual(set(labeled_df["dataset_index"].tolist()), {0})

    def test_arcface_head_outputs_expected_shape(self) -> None:
        head = ArcFaceHead(in_features=4, out_features=3, scale=30.0, margin=0.3)
        embeddings = torch.randn(5, 4)
        labels = torch.tensor([0, 1, 2, 1, 0])
        logits = head(embeddings, labels)
        self.assertEqual(tuple(logits.shape), (5, 3))

    def test_relation_distillation_zero_when_same(self) -> None:
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        dataset_indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        loss = compute_relation_distillation_loss(embeddings, embeddings, dataset_indices)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_masked_supcon_ignores_singletons_without_positive_pairs(self) -> None:
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        dataset_indices = torch.tensor([0, 0, 0], dtype=torch.long)
        global_label_indices = torch.tensor([0, 0, 1], dtype=torch.long)
        counts = torch.tensor([2, 2, 1], dtype=torch.long)
        loss = compute_masked_supcon_loss(
            embeddings=embeddings,
            dataset_indices=dataset_indices,
            global_label_indices=global_label_indices,
            identity_image_counts_fit=counts,
            temperature=0.1,
        )
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_write_training_monitor_plots_creates_pngs(self) -> None:
        training_log_df = pd.DataFrame(
            {
                "epoch": [1, 2],
                "train_loss": [10.0, 8.0],
                "train_arcface_loss": [9.0, 7.0],
                "train_relation_distill_loss": [0.5, 0.4],
                "train_feature_distill_loss": [0.3, 0.2],
                "train_supcon_loss": [0.0, 0.0],
                "macro_ari": [0.2, 0.3],
                "macro_recall_at_1": [0.4, 0.5],
                "LynxID2025_ari": [0.1, 0.2],
                "SalamanderID2025_ari": [0.2, 0.3],
            }
        )
        alignment_history_df = pd.DataFrame(
            {
                "epoch": [1, 2],
                "dataset": ["ALL_VAL", "ALL_VAL"],
                "relation_mse": [0.05, 0.03],
                "relation_mae": [0.1, 0.08],
                "relation_corr": [0.6, 0.7],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_paths = write_training_monitor_plots(
                plots_dir=Path(tmp_dir),
                training_log_df=training_log_df,
                alignment_history_df=alignment_history_df,
            )
            self.assertTrue(plot_paths["loss"].exists())
            self.assertTrue(plot_paths["validation"].exists())
            self.assertTrue(plot_paths["alignment"].exists())

    def test_scale_learning_rate_linear(self) -> None:
        scaled = scale_learning_rate(base_lr=1e-4, effective_batch_size=16, reference_batch_size=4, mode="linear")
        self.assertAlmostEqual(scaled, 4e-4)

    def test_build_best_metric_summary_keeps_earliest_best_epoch(self) -> None:
        training_log_df = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "macro_ari": [0.4, 0.5, 0.5],
                "LynxID2025_ari": [0.2, 0.1, 0.3],
                "SalamanderID2025_ari": [0.1, 0.4, 0.35],
                "SeaTurtleID2022_ari": [0.8, 0.75, 0.9],
            }
        )
        best_df = build_best_metric_summary(training_log_df)
        macro_row = best_df[best_df["selection_key"] == "macro"].iloc[0]
        lynx_row = best_df[best_df["selection_key"] == "LynxID2025"].iloc[0]
        self.assertEqual(int(macro_row["best_epoch"]), 2)
        self.assertEqual(str(macro_row["checkpoint_name"]), "best_macro.pt")
        self.assertEqual(int(lynx_row["best_epoch"]), 3)


if __name__ == "__main__":
    unittest.main()
