# Texas Self-Train Summary

## Experiment Card

- `experiment_id`: `ft_texas_miew_tcuwarmup_trusted_views_v1`
- `goal`: `Texas formal self-train: TCU warmup init + trusted supervision + all-image single-view pseudo-positives + no distill`
- `dataset`: `TexasHornedLizards`
- `student_backbone`: `miew`
- `student_model_id`: `conservationxlabs/miewid-msv3`
- `teacher_source_dirs`: `[]`
- `student_init_checkpoint`: `/home/hechen/gyk/animalclef/artifacts/training/experiments/ft_texas_external_warmup_v1/checkpoints/best.pt`
- `student_init_info`: `{'checkpoint_path': '/home/hechen/gyk/animalclef/artifacts/training/experiments/ft_texas_external_warmup_v1/checkpoints/best.pt', 'checkpoint_epoch': 11, 'loaded_key_count': 1216, 'skipped_key_count': 0, 'loaded_keys_preview': ['backbone.backbone.conv_stem.weight', 'backbone.backbone.bn1.weight', 'backbone.backbone.bn1.bias', 'backbone.backbone.bn1.running_mean', 'backbone.backbone.bn1.running_var', 'backbone.backbone.bn1.num_batches_tracked', 'backbone.backbone.blocks.0.0.conv_exp.weight', 'backbone.backbone.blocks.0.0.bn1.weight', 'backbone.backbone.blocks.0.0.bn1.bias', 'backbone.backbone.blocks.0.0.bn1.running_mean', 'backbone.backbone.blocks.0.0.bn1.running_var', 'backbone.backbone.blocks.0.0.bn1.num_batches_tracked', 'backbone.backbone.blocks.0.0.conv_pwl.weight', 'backbone.backbone.blocks.0.0.bn2.weight', 'backbone.backbone.blocks.0.0.bn2.bias', 'backbone.backbone.blocks.0.0.bn2.running_mean', 'backbone.backbone.blocks.0.0.bn2.running_var', 'backbone.backbone.blocks.0.0.bn2.num_batches_tracked', 'backbone.backbone.blocks.0.1.conv_exp.weight', 'backbone.backbone.blocks.0.1.bn1.weight'], 'skipped_keys_preview': []}`
- `anchor_threshold`: `0.38`
- `trusted_membership_path`: `/home/hechen/gyk/animalclef/artifacts/analysis/texas_trusted_batch_v1/tables/trusted_membership_v1.csv`
- `pseudo_positive_pairs_path`: `/home/hechen/gyk/animalclef/artifacts/training/cache/texas_pseudo_positive_views_v1/tables/pseudo_positive_pairs_v1.csv`

## Pseudo Labels

- `total_images`: `274`
- `seed_images`: `46`
- `seed_coverage_ratio`: `0.167883`
- `seed_clusters`: `18`
- `uncertain_images`: `228`
- `candidate_pairs`: `165`
- `mutual_topk_pairs`: `70`
- `trusted_images_after_override`: `46`
- `untrusted_images_after_override`: `228`

| pseudo_identity | size | mean_component_density | mean_component_size |
| --- | --- | --- | --- |
| trusted_comp_003 | 7 | 0.2857144285714286 | 1.8571428571428572 |
| trusted_comp_006 | 4 | 0.3333335 | 2.0 |
| trusted_comp_004 | 3 | 0.3333333333333333 | 1.3333333333333333 |
| trusted_comp_011 | 3 | 0.5 | 2.6666666666666665 |
| trusted_comp_014 | 3 | 0.3333333333333333 | 1.3333333333333333 |
| trusted_comp_001 | 2 | 0.2 | 3.0 |
| trusted_comp_002 | 2 | 0.2 | 3.0 |
| trusted_comp_005 | 2 | 0.6 | 5.0 |
| trusted_comp_007 | 2 | 1.0 | 2.0 |
| trusted_comp_008 | 2 | 0.666667 | 3.0 |
| trusted_comp_009 | 2 | 1.0 | 3.0 |
| trusted_comp_010 | 2 | 0.0 | 1.0 |
| trusted_comp_012 | 2 | 0.0 | 1.0 |
| trusted_comp_013 | 2 | 0.6 | 5.0 |
| trusted_comp_015 | 2 | 0.3 | 3.0 |
| trusted_comp_016 | 2 | 0.6 | 5.0 |
| trusted_comp_017 | 2 | 0.8333335 | 2.5 |
| trusted_comp_018 | 2 | 1.0 | 2.0 |

## Training Config

- `input_size`: `440`
- `student_feature_shape`: `B x 2152`
- `student_embedding_shape`: `B x 512`
- `teacher_fused_shape`: `B x 0`
- `train_augmentation`: `['DatasetSpecificPreprocess(dataset=TexasHornedLizards, stage=train, mode=gray_percentile_rgb, low=2.0, high=98.0)', 'RandomResizedCrop(size=(440, 440), scale=(0.88, 1.0))', 'RandomHorizontalFlip(p=0.5)', 'RandomRotation(degrees=[-10.0, 10.0])', 'ToTensor', 'Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))', 'RandomErasing(p=0.1, scale=(0.02, 0.1))']`
- `eval_preprocess`: `['DatasetSpecificPreprocess(dataset=TexasHornedLizards, stage=eval, mode=gray_percentile_rgb, low=2.0, high=98.0)', 'Resize(size=(440, 440))', 'ToTensor', 'Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))']`
- `classification_head`: `arcface`
- `pseudo_head_shape`: `18 x 512`
- `seed`: `42`
- `seed_oversample_factor`: `2.0`
- `view_pair_weight`: `0.5`
- `view_pair_temperature`: `0.07`
- `train_rows_after_view_expand`: `1142`
- `trusted_view_rows`: `230`
- `untrusted_view_rows`: `912`
- `trusted_view_images`: `46`
- `untrusted_view_images`: `228`
- `view_row_sources`: `{'auto_single_view_fallback_unmatched': 912, 'intra_image_pseudo_positive': 230}`
- `per_device_train_batch`: `8`
- `per_device_eval_batch`: `12`
- `effective_batch_size`: `8`
- `optimizer`: `AdamW`
- `reference_backbone_lr / reference_head_lr`: `1e-05 / 0.0001`
- `resolved_backbone_lr / resolved_head_lr`: `2e-05 / 0.0002`
- `weight_decay`: `0.01`
- `scheduler`: `linear warmup + cosine decay`
- `warmup_ratio`: `0.1`
- `epochs`: `12`
- `amp_enabled`: `True`
- `grad_clip_norm`: `1.0`
- `loss_weights`: `pseudo=1.0, relation=0.0, feature=0.0, view_pair=0.5`

## Best Proxy Result

- `best_epoch`: `3`
- `best_threshold`: `0.44`
- `best_proxy_score`: `0.703000`
- `best_seed_pair_agreement`: `1.000000`
- `best_mutual_topk_pair_keep_ratio`: `0.171429`
- `best_seed_recall_at_1`: `1.000000`
- `best_cluster_count`: `242`
- `best_largest_cluster_size`: `7`
- `peak_cuda_memory_mb`: `9751.53`

## Chosen Threshold Summary

| threshold | samples | clusters | largest_cluster_size | singleton_clusters | singleton_ratio | non_singleton_images | non_singleton_image_ratio | p90_cluster_size | seed_pair_agreement | seed_recall_at_1 | candidate_pair_keep_ratio | mutual_topk_pair_keep_ratio | student_teacher_topk_overlap | teacher_anchor_clusters | cluster_delta_vs_teacher_anchor | pair_agreement_vs_teacher_anchor | pair_agreement_vs_student_anchor | proxy_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.44 | 274.0 | 242.0 | 7.0 | 220.0 | 0.909091 | 54.0 | 0.19708 | 1.0 | 1.0 | 1.0 | 0.084848 | 0.171429 | nan | nan | nan | nan | 0.999893 | 0.703 |

## Full Threshold Sweep

| threshold | samples | clusters | largest_cluster_size | singleton_clusters | singleton_ratio | non_singleton_images | non_singleton_image_ratio | p90_cluster_size | seed_pair_agreement | seed_recall_at_1 | candidate_pair_keep_ratio | mutual_topk_pair_keep_ratio | student_teacher_topk_overlap | teacher_anchor_clusters | cluster_delta_vs_teacher_anchor | pair_agreement_vs_teacher_anchor | pair_agreement_vs_student_anchor | proxy_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.34 | 274.0 | 248.0 | 7.0 | 231.0 | 0.931452 | 43.0 | 0.156934 | 1.0 | 0.996135 | 1.0 | 0.078788 | 0.157143 | nan | nan | nan | nan | 0.99992 | 0.696261 |
| 0.36 | 274.0 | 247.0 | 7.0 | 229.0 | 0.927126 | 45.0 | 0.164234 | 1.0 | 0.997101 | 1.0 | 0.078788 | 0.157143 | nan | nan | nan | nan | 0.999947 | 0.696696 |
| 0.38 | 274.0 | 246.0 | 7.0 | 228.0 | 0.926829 | 46.0 | 0.167883 | 1.0 | 0.999034 | 1.0 | 0.078788 | 0.157143 | nan | nan | nan | nan | 1.0 | 0.697565 |
| 0.4 | 274.0 | 245.0 | 7.0 | 226.0 | 0.922449 | 48.0 | 0.175182 | 1.0 | 0.999034 | 1.0 | 0.078788 | 0.157143 | nan | nan | nan | nan | 0.999973 | 0.697565 |
| 0.42 | 274.0 | 243.0 | 7.0 | 222.0 | 0.91358 | 52.0 | 0.189781 | 1.0 | 1.0 | 1.0 | 0.078788 | 0.157143 | nan | nan | nan | nan | 0.99992 | 0.698 |
| 0.44 | 274.0 | 242.0 | 7.0 | 220.0 | 0.909091 | 54.0 | 0.19708 | 1.0 | 1.0 | 1.0 | 0.084848 | 0.171429 | nan | nan | nan | nan | 0.999893 | 0.703 |

## Resource Snapshot

- `device`: `cuda:2`
- `selected_gpu`: `{'index': 2, 'name': 'NVIDIA TITAN RTX', 'memory_total_mb': 24576, 'memory_used_mb': 165, 'memory_free_mb': 24046, 'utilization_gpu_pct': 1}`
- `resource_decision`: `Formal Texas self-train on GPU2; batch8 is the historical stable setting for miew trusted-views Texas.`
- `probe_reuse_note`: `Warmup->selftrain checkpoint handoff and source-dir submission override were both smoke-tested in wildfusion before this formal run.`
- `tmux_sessions_at_launch`: `['A2: 1 windows (created Tue Apr 14 02:29:07 2026)', 'lgsp_21shot: 1 windows (created Wed Apr 15 08:27:16 2026)', 'qwen_train: 1 windows (created Sun Apr 12 20:15:18 2026)', 'texas_review: 1 windows (created Mon Apr 13 13:55:49 2026)', 'texas_tcuwarmup_v1: 1 windows (created Fri Apr 17 17:06:36 2026)', 'texas_yes_expand_review_v2: 1 windows (created Tue Apr 14 21:43:26 2026)']`

## Monitoring Figures

![Training loss curves](plots/training_loss_curves.png)

- 读图方式：先看总 `train_loss` 是否下降，再看 `pseudo / relation / feature / view_pair` 哪个分量在主导优化。

![Pseudo proxy curves](plots/proxy_metric_curves.png)

- 读图方式：上图越高越好；下图先看选中阈值是否稳定，再看 `largest_cluster_size` 是否出现塌缩。

## Epoch Log

| epoch | train_loss | train_pseudo_loss | train_relation_distill_loss | train_feature_distill_loss | train_view_pair_loss | mean_seed_fraction | peak_cuda_memory_mb | best_threshold | best_proxy_score | best_seed_pair_agreement | best_mutual_topk_pair_keep_ratio | best_seed_recall_at_1 | best_cluster_count | best_largest_cluster_size | best_cluster_delta_vs_teacher_anchor | best_pair_agreement_vs_teacher_anchor | best_student_teacher_topk_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 6.444935 | 6.428562 | 0.0 | 0.0 | 0.032747 | 0.317308 | 9751.53 | 0.44 | 0.671783 | 0.966184 | 0.114286 | 1.0 | 259.0 | 3.0 | nan | nan | nan |
| 2.0 | 0.069913 | 0.058175 | 0.0 | 0.0 | 0.023477 | 0.334207 | 9751.54 | 0.42 | 0.697131 | 0.998068 | 0.157143 | 1.0 | 244.0 | 7.0 | nan | nan | nan |
| 3.0 | 0.016799 | 0.003891 | 0.0 | 0.0 | 0.025817 | 0.341492 | 9751.53 | 0.44 | 0.703 | 1.0 | 0.171429 | 1.0 | 242.0 | 7.0 | nan | nan | nan |
| 4.0 | 0.025773 | 0.010629 | 0.0 | 0.0 | 0.030289 | 0.318182 | 9751.53 | 0.38 | 0.698 | 1.0 | 0.157143 | 1.0 | 245.0 | 7.0 | nan | nan | nan |
| 5.0 | 0.009234 | 0.000217 | 0.0 | 0.0 | 0.018034 | 0.347028 | 9751.54 | 0.42 | 0.698 | 1.0 | 0.157143 | 1.0 | 244.0 | 7.0 | nan | nan | nan |
| 6.0 | 0.007314 | 0.00026 | 0.0 | 0.0 | 0.014108 | 0.325466 | 9751.53 | 0.36 | 0.698 | 1.0 | 0.157143 | 1.0 | 245.0 | 7.0 | nan | nan | nan |
| 7.0 | 0.012163 | 0.000729 | 0.0 | 0.0 | 0.022869 | 0.33683 | 9751.53 | 0.4 | 0.698 | 1.0 | 0.157143 | 1.0 | 244.0 | 7.0 | nan | nan | nan |
| 8.0 | 0.009863 | 9.5e-05 | 0.0 | 0.0 | 0.019537 | 0.330128 | 9751.53 | 0.4 | 0.698 | 1.0 | 0.157143 | 1.0 | 244.0 | 7.0 | nan | nan | nan |
| 9.0 | 0.010011 | 0.00029 | 0.0 | 0.0 | 0.019443 | 0.340618 | 9751.53 | 0.38 | 0.698 | 1.0 | 0.157143 | 1.0 | 245.0 | 7.0 | nan | nan | nan |
| 10.0 | 0.011549 | 0.000264 | 0.0 | 0.0 | 0.022571 | 0.316434 | 9751.53 | 0.38 | 0.698 | 1.0 | 0.157143 | 1.0 | 245.0 | 7.0 | nan | nan | nan |
| 11.0 | 0.009255 | 2.3e-05 | 0.0 | 0.0 | 0.018463 | 0.337121 | 9751.53 | 0.4 | 0.698 | 1.0 | 0.157143 | 1.0 | 244.0 | 7.0 | nan | nan | nan |
| 12.0 | 0.012495 | 0.000101 | 0.0 | 0.0 | 0.024789 | 0.34324 | 9751.53 | 0.38 | 0.698 | 1.0 | 0.157143 | 1.0 | 245.0 | 7.0 | nan | nan | nan |

## Conclusion And Next Decision

- `current_best_judgment`: `Use this checkpoint only if a Texas-only submission variant beats the frozen `miew@0.38` public baseline `0.37277`.`
