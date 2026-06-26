from __future__ import annotations

import unittest
from typing import Any
from types import SimpleNamespace

from src.animalclef_analysis.body_orientation_probe import rotation_to_vertical
from src.animalclef_analysis.qualitative_preprocessing_review import (
    _resolve_vertical_alignment_decision,
    _run_yolo_prompt_backoff,
    parameter_explanation_rows,
)


class QualitativePreprocessingReviewTest(unittest.TestCase):
    def test_parameter_explanations_cover_scale_and_fallback(self) -> None:
        rows = parameter_explanation_rows()
        parameters = set(rows["parameter"].tolist())
        self.assertIn("axis_confidence", parameters)
        self.assertIn("scale_factor", parameters)
        self.assertIn("fallback_reason", parameters)
        self.assertIn("target_extent_ratio / target_major_extent_ratio", parameters)
        self.assertIn("yolo_imgsz / yolo_max_det", parameters)
        self.assertIn("sam_threshold / sam_mask_threshold", parameters)
        self.assertIn("orientation_target=vertical", parameters)
        self.assertIn("forced_vertical_alignment", parameters)

    def test_rotation_to_vertical_prefers_smallest_vertical_alignment_turn(self) -> None:
        self.assertEqual(rotation_to_vertical(5.0), 85.0)
        self.assertEqual(rotation_to_vertical(80.0), 10.0)
        self.assertEqual(rotation_to_vertical(-20.0), -70.0)

    def test_resolve_vertical_alignment_decision_forces_dataset_branch(self) -> None:
        should_rotate, reason = _resolve_vertical_alignment_decision(
            dataset="TexasHornedLizards",
            decision=SimpleNamespace(reason="low_axis_confidence", should_apply=False),
            axis_stats={"axis_angle_deg": 15.0},
        )
        self.assertTrue(should_rotate)
        self.assertEqual(reason, "forced_vertical:low_axis_confidence")

        should_rotate, reason = _resolve_vertical_alignment_decision(
            dataset="UnknownDataset",
            decision=SimpleNamespace(reason="low_axis_confidence", should_apply=False),
            axis_stats={"axis_angle_deg": 15.0},
        )
        self.assertFalse(should_rotate)
        self.assertEqual(reason, "low_axis_confidence")

    def test_yolo_prompt_backoff_syncs_cached_clip_device_and_recovers(self) -> None:
        class DummyParameter:
            @property
            def device(self) -> str:
                return "cuda:0"

        class DummyClipModel:
            def __init__(self) -> None:
                self.device = "cpu"

        class DummyInnerModel:
            def parameters(self):
                yield DummyParameter()

        class DummyWorldModel:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []
                self.names: list[str] = []
                self.clip_model = DummyClipModel()
                self.model = DummyInnerModel()

            def set_classes(self, classes: list[str]) -> None:
                self.calls.append(classes)

        class DummyPredictorModel:
            def __init__(self) -> None:
                self.names: list[str] = []

        class DummyPredictor:
            def __init__(self) -> None:
                self.model = DummyPredictorModel()

        class DummyYOLO:
            def __init__(self) -> None:
                self.model = DummyWorldModel()
                self.predictor = DummyPredictor()
                self.predict_calls = 0

            def set_classes(self, classes: list[str]) -> None:
                self.model.set_classes(classes)
                self.model.names = classes
                self.predictor.model.names = classes

            def predict(self, image: Any, **_: Any) -> list[Any]:
                self.predict_calls += 1
                return []

        dummy_model = DummyYOLO()
        _, payload = _run_yolo_prompt_backoff(
            dummy_model,
            image=None,
            prompt_candidates=("head", "body"),
            conf=0.15,
            iou=0.5,
            imgsz=640,
            max_det=8,
        )
        self.assertEqual(dummy_model.model.calls, [["head"], ["body"]])
        self.assertEqual(dummy_model.model.clip_model.device, "cuda:0")
        self.assertEqual(payload["fallback_reason"], "no_detection")


if __name__ == "__main__":
    unittest.main()
