# Supervised Training Summary

## Experiment Card

- `experiment_id`: `ft_miew_arcface_masked_supcon_v1`
- `status`: `completed`
- `goal`: `Add masked SupCon on top of the current Miew ArcFace+distill baseline.`
- `student_backbone`: `miew`
- `student_model_id`: `conservationxlabs/miewid-msv3`
- `teacher_sources`: `['mega', 'miew']`

## Data Split

- `split_protocol`: `identity-level holdout`
- `val_identity_fraction`: `0.1`
- `seed`: `42`
- `fit_ids / fit_images`: `991 / 11739`
- `val_ids / val_images`: `111 / 1335`
- `fit_classes_by_dataset`: `{'LynxID2025': 69, 'SalamanderID2025': 528, 'SeaTurtleID2022': 394}`

## Training Config

- `input_size`: `440`
- `student_feature_shape`: `B x 2152`
- `student_embedding_shape`: `B x 512`
- `teacher_fused_shape`: `B x 3688`
- `train_augmentation`: `['RandomResizedCrop(size=(440, 440), scale=(0.75, 1.0))', 'ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.02, 0.02))', 'ToTensor', 'Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))', 'RandomErasing(p=0.25, scale=(0.02, 0.1))']`
- `eval_preprocess`: `['Resize(size=(440, 440))', 'ToTensor', 'Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))']`
- `head_shapes`: `{'LynxID2025': '69 x 512', 'SalamanderID2025': '528 x 512', 'SeaTurtleID2022': '394 x 512'}`
- `per_device_train_batch`: `32`
- `per_device_eval_batch`: `32`
- `gradient_accumulation_steps`: `1`
- `world_size`: `1`
- `effective_batch_size`: `32`
- `optimizer`: `AdamW`
- `reference_batch_size`: `4`
- `reference_backbone_lr / reference_head_lr`: `1e-05 / 0.0001`
- `lr_scaling`: `linear`
- `resolved_backbone_lr / resolved_head_lr`: `8e-05 / 0.0008`
- `weight_decay`: `0.01`
- `scheduler`: `linear warmup + cosine decay`
- `warmup_ratio`: `0.1`
- `epochs`: `8`
- `amp_enabled`: `True`
- `grad_clip_norm`: `1.0`
- `losses`: `ArcFace + relation distill + feature distill + masked SupCon(optional)`
- `loss_weights`: `arcface=1.0, relation=0.2, feature=0.05, supcon=0.1`
- `teacher_cache_dir`: `/home/hechen/gyk/animalclef/artifacts/training/cache/shared_teacher_cache_seed42_val0p1_full_v2`

## Best Result

- `best_epoch`: `2`
- `best_macro_ari`: `0.4989`
- `best_macro_recall_at_1`: `0.7967`
- `peak_cuda_memory_mb`: `18782.92`

## Teacher Components

| teacher_source | model_id | fit_dim | val_dim |
| --- | --- | --- | --- |
| mega | BVRA/MegaDescriptor-L-384 | 1536 | 1536 |
| miew | conservationxlabs/miewid-msv3 | 2152 | 2152 |

## Resource Decision

- `device`: `cuda:3`
- `selected_gpu`: `{'index': 3, 'name': 'NVIDIA TITAN RTX', 'memory_total_mb': 24576, 'memory_used_mb': 165, 'memory_free_mb': 24046, 'utilization_gpu_pct': 0}`
- `resource_decision`: `Single-GPU formal run on a free 24 GB card after a fresh probe because the loss branch changed.`
- `probe_reuse_note`: `Probe probe_miew_masked_supcon_bs32_v1 passed before this formal run; do not reuse old bs32 probe because SupCon changes the active loss branch.`
- `tmux_sessions_at_launch`: `['codex-subcenter-v1: 1 windows (created Thu Mar 26 01:47:07 2026)', 'codex-supcon-v1: 1 windows (created Thu Mar 26 01:47:07 2026)', 'flmm-rt-b-221849: 1 windows (created Tue Mar 24 22:20:41 2026)', 'flmm-rt-w-221849: 1 windows (created Tue Mar 24 22:20:47 2026)', 'p1_train: 1 windows (created Tue Mar 24 21:19:50 2026)', 'qwen_train: 1 windows (created Wed Mar 25 19:43:23 2026)']`

## Best Validation Thresholds

| dataset | threshold | ari | nmi | pairwise_f1 | cluster_count | singleton_cluster_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| LynxID2025 | 0.8 | 0.351766 | 0.501268 | 0.535382 | 19 | 0.368421 |
| SalamanderID2025 | 0.6 | 0.249973 | 0.866129 | 0.259843 | 77 | 0.532468 |
| SeaTurtleID2022 | 0.7 | 0.895005 | 0.939228 | 0.898953 | 90 | 0.288889 |

## Validation Recall

| dataset | recall_at_1 | recall_at_5 |
| --- | --- | --- |
| LynxID2025 | 0.879433 | 0.929078 |
| SalamanderID2025 | 0.546392 | 0.690722 |
| SeaTurtleID2022 | 0.964218 | 0.984934 |

## Teacher / Student Alignment

| dataset | samples | relation_mse | relation_mae | relation_corr | epoch |
| --- | --- | --- | --- | --- | --- |
| ALL_VAL | 1335 | 0.018582 | 0.111592 | 0.741458 | 2 |
| LynxID2025 | 143 | 0.036775 | 0.155554 | 0.59181 | 2 |
| SalamanderID2025 | 130 | 0.072041 | 0.249159 | 0.69307 | 2 |
| SeaTurtleID2022 | 1062 | 0.020918 | 0.119846 | 0.79176 | 2 |

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
| 1.0 | 9.408506 | 9.260094 | 0.036626 | 0.723991 | 1.048874 | 18781.95 | 0.497723 | 0.75658 | 0.019339 | 0.114946 | 0.755305 | 0.437037 | 0.8 | 0.167148 | 0.3 | 0.888983 | 0.7 |
| 2.0 | 3.913407 | 3.822126 | 0.058886 | 0.511777 | 0.539154 | 18782.92 | 0.498915 | 0.796681 | 0.018582 | 0.111592 | 0.741458 | 0.351766 | 0.8 | 0.249973 | 0.6 | 0.895005 | 0.7 |

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
