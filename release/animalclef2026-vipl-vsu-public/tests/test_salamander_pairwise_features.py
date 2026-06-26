from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.animalclef_analysis.salamander_pairwise_features import (
    FEATURE_SET_BASIC,
    FEATURE_SET_DUAL_VIEW_V1,
    FEATURE_SET_META_GRAPH_V1,
    FEATURE_SET_META_GRAPH_DUAL_VIEW_V1,
    FEATURE_SET_META_GRAPH_YELLOW_V1,
    FEATURE_SET_YELLOW_V1,
    append_feature_set,
    append_dual_view_pair_features,
    append_metadata_pair_features,
    append_route_graph_pair_features,
    resolve_pair_feature_columns,
)


class SalamanderPairwiseFeaturesTest(unittest.TestCase):
    def test_append_metadata_pair_features_adds_orientation_and_date_signals(self) -> None:
        pair_df = pd.DataFrame(
            {
                "left_index": [0, 0],
                "right_index": [1, 2],
                "image_id": ["10", "10"],
                "neighbor_image_id": ["11", "12"],
            }
        )
        metadata_df = pd.DataFrame(
            {
                "image_id": ["10", "11", "12"],
                "orientation": ["top", "right", ""],
                "date": ["2020-01-01", "2020-01-01", ""],
            }
        )

        enriched = append_metadata_pair_features(pair_df=pair_df, metadata_df=metadata_df)

        self.assertEqual(int(enriched.iloc[0]["orientation_known_pair"]), 1)
        self.assertEqual(int(enriched.iloc[0]["same_orientation"]), 0)
        self.assertEqual(int(enriched.iloc[0]["orientation_pair_is_right__top"]), 1)
        self.assertEqual(int(enriched.iloc[0]["capture_date_known_pair"]), 1)
        self.assertEqual(int(enriched.iloc[0]["same_capture_date"]), 1)
        self.assertAlmostEqual(float(enriched.iloc[0]["capture_day_gap"]), 0.0)
        self.assertEqual(int(enriched.iloc[1]["orientation_known_pair"]), 0)
        self.assertEqual(int(enriched.iloc[1]["orientation_pair_is_unknown"]), 1)
        self.assertEqual(int(enriched.iloc[1]["capture_date_known_pair"]), 0)

    def test_append_route_graph_pair_features_adds_rank_and_shared_neighbor_support(self) -> None:
        pair_df = pd.DataFrame(
            {
                "left_index": [0],
                "right_index": [1],
                "image_id": ["10"],
                "neighbor_image_id": ["11"],
            }
        )
        route_score = np.asarray(
            [
                [1.0, 0.95, 0.80, 0.10],
                [0.95, 1.0, 0.70, 0.20],
                [0.80, 0.70, 1.0, 0.60],
                [0.10, 0.20, 0.60, 1.0],
            ],
            dtype=np.float32,
        )

        enriched = append_route_graph_pair_features(pair_df=pair_df, route_score=route_score, top_k=2)

        self.assertAlmostEqual(float(enriched.iloc[0]["route_rank_pct_left_to_right"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(enriched.iloc[0]["route_rank_pct_right_to_left"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(enriched.iloc[0]["route_reciprocal_rank_mean"]), 1.0, places=6)
        self.assertEqual(int(enriched.iloc[0]["route_mutual_topk"]), 1)
        self.assertEqual(int(enriched.iloc[0]["route_shared_neighbor_count"]), 1)
        self.assertAlmostEqual(float(enriched.iloc[0]["route_shared_neighbor_ratio"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(enriched.iloc[0]["route_shared_neighbor_mean_score"]), 0.75, places=6)

    def test_append_feature_set_meta_graph_extends_basic_columns(self) -> None:
        pair_df = pd.DataFrame(
            {
                "left_index": [0],
                "right_index": [1],
                "image_id": ["10"],
                "neighbor_image_id": ["11"],
                "route_global_score": [0.9],
                "fusion_global_score": [0.8],
                "student_global_score": [0.85],
                "distill_global_score": [0.82],
                "route_minus_fusion": [0.1],
                "route_minus_student": [0.05],
                "student_minus_distill": [0.03],
                "local_score": [0.7],
                "local_raw_score": [0.2],
                "inliers": [12],
                "good_matches": [20],
                "left_keypoints": [100],
                "right_keypoints": [120],
                "keypoint_min": [100],
                "keypoint_max": [120],
                "inlier_ratio": [0.12],
                "match_density": [0.2],
            }
        )
        metadata_df = pd.DataFrame(
            {
                "image_id": ["10", "11"],
                "orientation": ["top", "right"],
                "date": ["2020-01-01", "2020-01-05"],
            }
        )
        route_score = np.asarray(
            [
                [1.0, 0.95],
                [0.95, 1.0],
            ],
            dtype=np.float32,
        )

        basic_df = append_feature_set(
            pair_df=pair_df,
            metadata_df=metadata_df,
            route_score=route_score,
            feature_set=FEATURE_SET_BASIC,
            graph_top_k=1,
        )
        meta_df = append_feature_set(
            pair_df=pair_df,
            metadata_df=metadata_df,
            route_score=route_score,
            feature_set=FEATURE_SET_META_GRAPH_V1,
            graph_top_k=1,
        )

        self.assertNotIn("same_orientation", basic_df.columns)
        self.assertIn("same_orientation", meta_df.columns)
        self.assertIn("route_shared_neighbor_ratio", meta_df.columns)
        self.assertEqual(resolve_pair_feature_columns(FEATURE_SET_BASIC)[0], "route_global_score")
        self.assertTrue(set(resolve_pair_feature_columns(FEATURE_SET_META_GRAPH_V1)).issubset(set(meta_df.columns)))
        self.assertGreater(meta_df.shape[1], basic_df.shape[1])

    def test_append_dual_view_pair_features_adds_masked_and_cross_scores(self) -> None:
        pair_df = pd.DataFrame(
            {
                "left_index": [0],
                "right_index": [1],
                "image_id": ["10"],
                "neighbor_image_id": ["11"],
                "student_global_score": [0.80],
                "distill_global_score": [0.70],
            }
        )
        masked_student_score = np.asarray([[1.0, 0.60], [0.60, 1.0]], dtype=np.float32)
        masked_distill_score = np.asarray([[1.0, 0.50], [0.50, 1.0]], dtype=np.float32)
        student_cross_score = np.asarray([[0.95, 0.55], [0.65, 0.90]], dtype=np.float32)
        distill_cross_score = np.asarray([[0.96, 0.45], [0.75, 0.91]], dtype=np.float32)

        enriched = append_dual_view_pair_features(
            pair_df=pair_df,
            masked_student_score=masked_student_score,
            masked_distill_score=masked_distill_score,
            student_cross_score=student_cross_score,
            distill_cross_score=distill_cross_score,
        )

        self.assertAlmostEqual(float(enriched.iloc[0]["masked_student_global_score"]), 0.60, places=6)
        self.assertAlmostEqual(float(enriched.iloc[0]["student_masked_score_mean"]), 0.70, places=6)
        self.assertAlmostEqual(float(enriched.iloc[0]["student_masked_score_gap"]), 0.20, places=6)
        self.assertAlmostEqual(float(enriched.iloc[0]["student_cross_score_mean"]), 0.60, places=6)
        self.assertAlmostEqual(float(enriched.iloc[0]["student_cross_score_max"]), 0.65, places=6)
        self.assertAlmostEqual(float(enriched.iloc[0]["dual_view_cross_score_mean"]), 0.60, places=6)

    def test_append_feature_set_meta_graph_dual_view_extends_basic_columns(self) -> None:
        pair_df = pd.DataFrame(
            {
                "left_index": [0],
                "right_index": [1],
                "image_id": ["10"],
                "neighbor_image_id": ["11"],
                "route_global_score": [0.9],
                "fusion_global_score": [0.8],
                "student_global_score": [0.85],
                "distill_global_score": [0.82],
                "route_minus_fusion": [0.1],
                "route_minus_student": [0.05],
                "student_minus_distill": [0.03],
                "local_score": [0.7],
                "local_raw_score": [0.2],
                "inliers": [12],
                "good_matches": [20],
                "left_keypoints": [100],
                "right_keypoints": [120],
                "keypoint_min": [100],
                "keypoint_max": [120],
                "inlier_ratio": [0.12],
                "match_density": [0.2],
            }
        )
        metadata_df = pd.DataFrame(
            {
                "image_id": ["10", "11"],
                "orientation": ["top", "right"],
                "date": ["2020-01-01", "2020-01-05"],
            }
        )
        route_score = np.asarray([[1.0, 0.95], [0.95, 1.0]], dtype=np.float32)
        masked_student_score = np.asarray([[1.0, 0.88], [0.88, 1.0]], dtype=np.float32)
        masked_distill_score = np.asarray([[1.0, 0.84], [0.84, 1.0]], dtype=np.float32)
        student_cross_score = np.asarray([[0.97, 0.80], [0.86, 0.94]], dtype=np.float32)
        distill_cross_score = np.asarray([[0.96, 0.79], [0.83, 0.93]], dtype=np.float32)

        dual_df = append_feature_set(
            pair_df=pair_df,
            metadata_df=metadata_df,
            route_score=route_score,
            feature_set=FEATURE_SET_DUAL_VIEW_V1,
            graph_top_k=1,
            masked_student_score=masked_student_score,
            masked_distill_score=masked_distill_score,
            student_cross_score=student_cross_score,
            distill_cross_score=distill_cross_score,
        )
        meta_dual_df = append_feature_set(
            pair_df=pair_df,
            metadata_df=metadata_df,
            route_score=route_score,
            feature_set=FEATURE_SET_META_GRAPH_DUAL_VIEW_V1,
            graph_top_k=1,
            masked_student_score=masked_student_score,
            masked_distill_score=masked_distill_score,
            student_cross_score=student_cross_score,
            distill_cross_score=distill_cross_score,
        )

        self.assertIn("masked_student_global_score", dual_df.columns)
        self.assertNotIn("same_orientation", dual_df.columns)
        self.assertIn("same_orientation", meta_dual_df.columns)
        self.assertIn("route_shared_neighbor_ratio", meta_dual_df.columns)
        self.assertTrue(
            set(resolve_pair_feature_columns(FEATURE_SET_META_GRAPH_DUAL_VIEW_V1)).issubset(set(meta_dual_df.columns))
        )

    def test_append_feature_set_yellow_uses_precomputed_yellow_columns(self) -> None:
        pair_df = pd.DataFrame(
            {
                "left_index": [0],
                "right_index": [1],
                "image_id": ["10"],
                "neighbor_image_id": ["11"],
                "route_global_score": [0.9],
                "fusion_global_score": [0.8],
                "student_global_score": [0.85],
                "distill_global_score": [0.82],
                "route_minus_fusion": [0.1],
                "route_minus_student": [0.05],
                "student_minus_distill": [0.03],
                "local_score": [0.7],
                "local_raw_score": [0.2],
                "inliers": [12],
                "good_matches": [20],
                "left_keypoints": [100],
                "right_keypoints": [120],
                "keypoint_min": [100],
                "keypoint_max": [120],
                "inlier_ratio": [0.12],
                "match_density": [0.2],
            }
        )
        yellow_pair_df = pair_df.assign(
            left_yellow_quality_flag_v1=True,
            right_yellow_quality_flag_v1=True,
            left_yellow_focus_available_v1=True,
            right_yellow_focus_available_v1=True,
            yellow_focus_pair_valid_v1=True,
            yellow_orb_pair_valid_v1=True,
            yellow_roi_left_keypoints=50,
            yellow_roi_right_keypoints=49,
            yellow_roi_good_matches=17,
            yellow_roi_inliers=9,
            yellow_roi_local_raw_score=0.21,
            yellow_roi_local_score=0.44,
            yellow_roi_keypoint_min=49,
            yellow_patch_pair_valid_v1=True,
            yellow_patch_overlap_pixels_v1=240,
            yellow_patch_gray_corr_v1=0.93,
            yellow_patch_gray_absdiff_v1=0.07,
            yellow_patch_mask_iou_v1=0.61,
            yellow_patch_mask_dice_v1=0.76,
            yellow_patch_profile_corr_v1=0.88,
            yellow_patch_profile_l1_v1=0.18,
            yellow_orb_support_v1=True,
            yellow_patch_support_v1=True,
            yellow_pair_support_v1=True,
            yellow_orb_fail_v1=False,
            yellow_patch_hard_fail_v1=False,
            yellow_patch_extreme_fail_v1=False,
            yellow_patch_soft_fail_v1=False,
            yellow_hard_veto_v1=False,
            yellow_soft_veto_v1=False,
            yellow_veto_applied_v1=False,
        )
        metadata_df = pd.DataFrame({"image_id": ["10", "11"], "orientation": ["top", "top"], "date": ["2020-01-01", "2020-01-02"]})
        route_score = np.asarray([[1.0, 0.95], [0.95, 1.0]], dtype=np.float32)

        yellow_df = append_feature_set(
            pair_df=pair_df,
            metadata_df=metadata_df,
            route_score=route_score,
            feature_set=FEATURE_SET_YELLOW_V1,
            graph_top_k=1,
            yellow_pair_df=yellow_pair_df,
        )

        self.assertIn("yellow_roi_local_score", yellow_df.columns)
        self.assertIn("yellow_patch_gray_corr_v1", yellow_df.columns)
        self.assertTrue(set(resolve_pair_feature_columns(FEATURE_SET_YELLOW_V1)).issubset(set(yellow_df.columns)))

    def test_append_feature_set_meta_graph_yellow_extends_yellow_columns(self) -> None:
        pair_df = pd.DataFrame(
            {
                "left_index": [0],
                "right_index": [1],
                "image_id": ["10"],
                "neighbor_image_id": ["11"],
                "route_global_score": [0.9],
                "fusion_global_score": [0.8],
                "student_global_score": [0.85],
                "distill_global_score": [0.82],
                "route_minus_fusion": [0.1],
                "route_minus_student": [0.05],
                "student_minus_distill": [0.03],
                "local_score": [0.7],
                "local_raw_score": [0.2],
                "inliers": [12],
                "good_matches": [20],
                "left_keypoints": [100],
                "right_keypoints": [120],
                "keypoint_min": [100],
                "keypoint_max": [120],
                "inlier_ratio": [0.12],
                "match_density": [0.2],
            }
        )
        yellow_pair_df = pair_df.assign(
            left_yellow_quality_flag_v1=True,
            right_yellow_quality_flag_v1=True,
            left_yellow_focus_available_v1=True,
            right_yellow_focus_available_v1=True,
            yellow_focus_pair_valid_v1=True,
            yellow_orb_pair_valid_v1=True,
            yellow_roi_left_keypoints=50,
            yellow_roi_right_keypoints=49,
            yellow_roi_good_matches=17,
            yellow_roi_inliers=9,
            yellow_roi_local_raw_score=0.21,
            yellow_roi_local_score=0.44,
            yellow_roi_keypoint_min=49,
            yellow_patch_pair_valid_v1=True,
            yellow_patch_overlap_pixels_v1=240,
            yellow_patch_gray_corr_v1=0.93,
            yellow_patch_gray_absdiff_v1=0.07,
            yellow_patch_mask_iou_v1=0.61,
            yellow_patch_mask_dice_v1=0.76,
            yellow_patch_profile_corr_v1=0.88,
            yellow_patch_profile_l1_v1=0.18,
            yellow_orb_support_v1=True,
            yellow_patch_support_v1=True,
            yellow_pair_support_v1=True,
            yellow_orb_fail_v1=False,
            yellow_patch_hard_fail_v1=False,
            yellow_patch_extreme_fail_v1=False,
            yellow_patch_soft_fail_v1=False,
            yellow_hard_veto_v1=False,
            yellow_soft_veto_v1=False,
            yellow_veto_applied_v1=False,
        )
        metadata_df = pd.DataFrame({"image_id": ["10", "11"], "orientation": ["top", "right"], "date": ["2020-01-01", "2020-01-05"]})
        route_score = np.asarray([[1.0, 0.95], [0.95, 1.0]], dtype=np.float32)

        meta_yellow_df = append_feature_set(
            pair_df=pair_df,
            metadata_df=metadata_df,
            route_score=route_score,
            feature_set=FEATURE_SET_META_GRAPH_YELLOW_V1,
            graph_top_k=1,
            yellow_pair_df=yellow_pair_df,
        )

        self.assertIn("yellow_roi_local_score", meta_yellow_df.columns)
        self.assertIn("same_orientation", meta_yellow_df.columns)
        self.assertIn("route_shared_neighbor_ratio", meta_yellow_df.columns)
        self.assertTrue(set(resolve_pair_feature_columns(FEATURE_SET_META_GRAPH_YELLOW_V1)).issubset(set(meta_yellow_df.columns)))


if __name__ == "__main__":
    unittest.main()
