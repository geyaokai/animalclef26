# Supervised Training Summary

## Experiment Card

- `experiment_id`: `ft_mega_arcface_distill_v1`
- `status`: `completed`
- `goal`: `Run the second formal supervised training mainline with Mega initialization, per-dataset ArcFace heads, and Mega+Miew teacher distillation under the validated batch size.`
- `student_backbone`: `mega`
- `student_model_id`: `BVRA/MegaDescriptor-L-384`
- `teacher_sources`: `['mega', 'miew']`

## Data Split

- `split_protocol`: `identity-level holdout`
- `val_identity_fraction`: `0.1`
- `seed`: `42`
- `fit_ids / fit_images`: `991 / 11739`
- `val_ids / val_images`: `111 / 1335`
- `fit_classes_by_dataset`: `{'LynxID2025': 69, 'SalamanderID2025': 528, 'SeaTurtleID2022': 394}`

## Training Config

- `input_size`: `384`
- `student_feature_shape`: `B x 1536`
- `student_embedding_shape`: `B x 512`
- `teacher_fused_shape`: `B x 3688`
- `train_augmentation`: `['RandomResizedCrop(size=(384, 384), scale=(0.75, 1.0))', 'ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.02, 0.02))', 'ToTensor', 'Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))', 'RandomErasing(p=0.25, scale=(0.02, 0.1))']`
- `eval_preprocess`: `['Resize(size=(384, 384))', 'ToTensor', 'Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))']`
- `head_shapes`: `{'LynxID2025': '69 x 512', 'SalamanderID2025': '528 x 512', 'SeaTurtleID2022': '394 x 512'}`
- `per_device_train_batch`: `20`
- `per_device_eval_batch`: `20`
- `gradient_accumulation_steps`: `1`
- `world_size`: `1`
- `effective_batch_size`: `20`
- `optimizer`: `AdamW`
- `reference_batch_size`: `4`
- `reference_backbone_lr / reference_head_lr`: `1e-05 / 0.0001`
- `lr_scaling`: `linear`
- `resolved_backbone_lr / resolved_head_lr`: `5e-05 / 0.0005`
- `weight_decay`: `0.01`
- `scheduler`: `linear warmup + cosine decay`
- `warmup_ratio`: `0.1`
- `epochs`: `8`
- `amp_enabled`: `True`
- `grad_clip_norm`: `1.0`
- `losses`: `ArcFace + relation distill + feature distill + masked SupCon(optional)`
- `loss_weights`: `arcface=1.0, relation=0.2, feature=0.05, supcon=0.0`
- `teacher_cache_dir`: `/home/hechen/gyk/animalclef/artifacts/training/cache/shared_teacher_cache_seed42_val0p1_full_v2`

## Best Result

- `best_epoch`: `5`
- `best_macro_ari`: `0.5017`
- `best_macro_recall_at_1`: `0.7069`
- `peak_cuda_memory_mb`: `19910.81`

## Teacher Components

| teacher_source | model_id | fit_dim | val_dim |
| --- | --- | --- | --- |
| mega | BVRA/MegaDescriptor-L-384 | 1536 | 1536 |
| miew | conservationxlabs/miewid-msv3 | 2152 | 2152 |

## Resource Decision

- `device`: `cuda:2`
- `selected_gpu`: `{'index': 2, 'name': 'NVIDIA TITAN RTX', 'memory_total_mb': 24576, 'memory_used_mb': 165, 'memory_free_mb': 24046, 'utilization_gpu_pct': 0}`
- `resource_decision`: `Use single GPU cuda:2 for the formal run because probe_mega_bs20_full_v1 reached a stable 19.9 GB peak on a 24 GB card; keep experiments independent and comparable rather than switching to multi-GPU.`
- `probe_reuse_note`: `Reusing probe_mega_bs20_full_v1 is valid because backbone, input size, loss branches, AMP mode, world_size, per-device batch, and gradient accumulation are unchanged.`
- `tmux_sessions_at_launch`: `['codex-ft-mega-v1: 1 windows (created Wed Mar 25 18:15:05 2026)', 'codex-ft-miew-v1: 1 windows (created Wed Mar 25 18:08:43 2026)', 'download: 1 windows (created Wed Mar 25 01:07:31 2026)', 'flmm-rt-b-221849: 1 windows (created Tue Mar 24 22:20:41 2026)', 'flmm-rt-w-221849: 1 windows (created Tue Mar 24 22:20:47 2026)', 'p1_train: 1 windows (created Tue Mar 24 21:19:50 2026)']`

## Best Validation Thresholds

| dataset | threshold | ari | nmi | pairwise_f1 | cluster_count | singleton_cluster_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| LynxID2025 | 0.8 | 0.493685 | 0.539451 | 0.659873 | 16 | 0.25 |
| SalamanderID2025 | 0.7 | 0.136382 | 0.820603 | 0.15 | 72 | 0.555556 |
| SeaTurtleID2022 | 0.7 | 0.875089 | 0.926905 | 0.879691 | 109 | 0.330275 |

## Validation Recall

| dataset | recall_at_1 | recall_at_5 |
| --- | --- | --- |
| LynxID2025 | 0.836879 | 0.957447 |
| SalamanderID2025 | 0.329897 | 0.57732 |
| SeaTurtleID2022 | 0.953861 | 0.981168 |

## Teacher / Student Alignment

| dataset | samples | relation_mse | relation_mae | relation_corr | epoch |
| --- | --- | --- | --- | --- | --- |
| ALL_VAL | 1335 | 0.016604 | 0.105647 | 0.778249 | 5 |
| LynxID2025 | 143 | 0.046181 | 0.174044 | 0.581493 | 5 |
| SalamanderID2025 | 130 | 0.09845 | 0.29023 | 0.453996 | 5 |
| SeaTurtleID2022 | 1062 | 0.017768 | 0.11177 | 0.843555 | 5 |

## Monitoring Figures

![Training loss curves](plots/training_loss_curves.png)

- 读图方式：横轴是 epoch。上半图先看总 `train_loss` 是否稳定下降；下半图再看 `ArcFace / relation distill / feature distill / SupCon` 各分量谁在主导训练。

![Validation metric curves](plots/validation_metric_curves.png)

- 读图方式：先看 `macro ARI` 和 `macro Recall@1` 的走势，再看每个 dataset 的 `ARI` 是否同步提升，避免被单一数据集掩盖。

![Alignment curves](plots/alignment_curves.png)

- 读图方式：`relation_mse / relation_mae` 越低越好，`relation_corr` 越高越好。它反映 student 是否还在贴近 teacher 的局部相似度结构。

## Epoch Log

| epoch | train_loss | train_arcface_loss | train_relation_distill_loss | train_feature_distill_loss | train_supcon_loss | peak_cuda_memory_mb | macro_ari | macro_recall_at_1 | alignment_relation_mse | alignment_relation_mae | alignment_relation_corr | LynxID2025_ari | LynxID2025_threshold | SalamanderID2025_ari | SalamanderID2025_threshold | SeaTurtleID2022_ari | SeaTurtleID2022_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 9.366877 | 9.32446 | 0.026589 | 0.741982 | 0.606325 | 19910.81 | 0.390316 | 0.679355 | 0.014733 | 0.099472 | 0.808147 | 0.246141 | 0.7 | 0.073299 | 0.35 | 0.851508 | 0.7 |
| 2.0 | 5.455749 | 5.421823 | 0.041743 | 0.511542 | 0.350946 | 19910.81 | 0.486806 | 0.707176 | 0.017068 | 0.107829 | 0.779178 | 0.484745 | 0.8 | 0.123336 | 0.5 | 0.852337 | 0.7 |
| 3.0 | 3.538036 | 3.50348 | 0.053143 | 0.478537 | 0.19813 | 19910.81 | 0.457152 | 0.708396 | 0.016571 | 0.106291 | 0.791108 | 0.351646 | 0.8 | 0.140568 | 0.55 | 0.879243 | 0.7 |
| 4.0 | 2.377841 | 2.342758 | 0.059957 | 0.461839 | 0.154324 | 19910.81 | 0.495518 | 0.700676 | 0.016455 | 0.105315 | 0.780162 | 0.48014 | 0.8 | 0.13914 | 0.7 | 0.867275 | 0.7 |
| 5.0 | 1.417638 | 1.382743 | 0.062843 | 0.446532 | 0.113159 | 19910.81 | 0.501719 | 0.706879 | 0.016604 | 0.105647 | 0.778249 | 0.493685 | 0.8 | 0.136382 | 0.7 | 0.875089 | 0.7 |

## Qualitative Review

### LynxID2025 predicted clusters

![LynxID2025 predicted clusters](../qualitative/predicted_clusters_LynxID2025.jpg)

- 读图方式：随机看几行 cluster 内样本是否真的像同一只个体，优先观察是否存在明显背景主导或视角断裂。

### LynxID2025 hard negatives

![LynxID2025 hard negatives](../qualitative/hard_negatives_LynxID2025.jpg)

- 读图方式：左边是 query，右边是高相似但不同 identity 的样本，优先看模型是否被局部纹理、遮挡或姿态误导。

### SalamanderID2025 predicted clusters

![SalamanderID2025 predicted clusters](../qualitative/predicted_clusters_SalamanderID2025.jpg)

- 读图方式：随机看几行 cluster 内样本是否真的像同一只个体，优先观察是否存在明显背景主导或视角断裂。

### SalamanderID2025 hard negatives

![SalamanderID2025 hard negatives](../qualitative/hard_negatives_SalamanderID2025.jpg)

- 读图方式：左边是 query，右边是高相似但不同 identity 的样本，优先看模型是否被局部纹理、遮挡或姿态误导。

### SeaTurtleID2022 predicted clusters

![SeaTurtleID2022 predicted clusters](../qualitative/predicted_clusters_SeaTurtleID2022.jpg)

- 读图方式：随机看几行 cluster 内样本是否真的像同一只个体，优先观察是否存在明显背景主导或视角断裂。

### SeaTurtleID2022 hard negatives

![SeaTurtleID2022 hard negatives](../qualitative/hard_negatives_SeaTurtleID2022.jpg)

- 读图方式：左边是 query，右边是高相似但不同 identity 的样本，优先看模型是否被局部纹理、遮挡或姿态误导。

## Conclusion And Next Decision

- `current_best_judgment`: `Inspect best epoch summary, compare against frozen fusion and rerank baselines, then decide whether to continue the current backbone or switch to the next matrix item.`
