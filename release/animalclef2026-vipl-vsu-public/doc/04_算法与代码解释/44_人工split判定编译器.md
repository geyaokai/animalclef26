# 人工 split 判定编译器

## 这个编译器解决什么问题

- 现在人工复核台已经能稳定产出 `pair judgments`：
  - `yes` = 这两张图应该是同一个体
  - `no` = 这两张图不应该在同一簇
  - `uncertain` = 暂时不下结论
- 但 Kaggle 提交链路吃的不是样本对判断，而是可执行的 cluster overlay 规则。
- 所以中间还缺一层“翻译器”：
  - 输入是人工 pair 判断
  - 输出是 `split_to_singletons / attach_to_anchor` 这类真正能改 submission 的操作

当前这层就落在：

- 编译逻辑：`src/animalclef_analysis/manual_split_compiler.py`
- 命令入口：`scripts/compile_manual_split_judgments.py`

## 输入是什么

### 1. base submission 预测表

- 路径例子：
  - `artifacts/submissions/kaggle_variant_lynx_seedsmooth_alpha0p15_onxgb_v1/tables/test_predictions_v1.csv`
- 关键列：
  - `dataset`
  - `image_id`
  - `pred_cluster_id`
- 作用：
  - 先知道当前官方主路把每张 test 图分到了哪个 cluster
  - 后面的 split 只能在这个现有 cluster 结构上做小修补

### 2. 人工 pair judgments

- 路径例子：
  - `artifacts/analysis/manual_review_sessions/autosave/manual_pair_review_v1.json`
- 每条 judgment 里最关键的是：
  - `dataset`
  - `candidate_key`
  - `candidate_type`
  - `image_id`
  - `neighbor_image_id`
  - `label`
  - `xgb_same_identity_prob`

这里的 `candidate_key` 在当前 `split` 任务里，等价于“当前被审的 base cluster id”。

## 它是怎么工作的

## 第 1 步：只拿 `split` judgment

- 当前这版编译器只处理：
  - `candidate_type == split`
- `merge` judgment 先不编进来。
- 原因很简单：
  - 你当前最强信号来自人工发现“这个簇里谁明显不该在一起”
  - `merge` 目前大多是明显 `no`，还不适合直接转成稳健 merge

## 第 2 步：在每个候选簇内部，给每张图记一张“人工得分表”

假设某个 base cluster 里有 `M` 张图，人工实际看过其中 `P` 对 pair。

编译器会把每张图都映射成一行 `image score`：

- `judged_pair_count`
  - 这张图参与过多少个人工判断 pair
- `yes_degree`
  - 它和多少张图被判成 `yes`
- `no_degree`
  - 它和多少张图被判成 `no`
- `uncertain_degree`
  - 它和多少张图被判成 `uncertain`
- `net_no_margin = no_degree - yes_degree`

通俗理解：

- `yes_degree` 高，说明这张图在当前簇里更像“有自己同伴”
- `no_degree` 高，说明它和当前簇里很多图都冲突
- `net_no_margin` 高，说明“反对票明显多于支持票”

## 第 3 步：决定哪些图应该先拆出去

当前默认门槛是：

- `no_degree >= 2`
- `no_degree - yes_degree >= 1`

也就是只有当一张图：

- 至少被两个已审 pair 指向“它不该在这里”
- 而且反对票比支持票更多

才会被选进 `selected_for_split=True`。

这一步是故意保守的。

- 不是只要出现一个 `no` 就拆
- 也不是看到一条低分 pair 就拆
- 而是要求“人工冲突证据足够成形”

## 第 4 步：给原簇保留一个 anchor

如果一个簇里有些图被选中拆出，那剩余图里要挑一个 anchor 留在原簇。

anchor 选择规则是：

- 优先 `yes_degree - no_degree` 更高的图
- 再看 `yes_degree` 更高
- 再看 `no_degree` 更低
- 最后按 `image_id` 稳定打破平局

为什么一定要留 anchor：

- `split_to_singletons` 不是“整簇炸掉重建”
- 它是“保留原簇，再把选中的成员挪出去”

如果一个簇里所有图都被人工判得很可疑，这时编译器也不会把整簇清空，而是会：

- 强制从全簇里留一个最像 anchor 的图
- 其余图再拆出去

所以当前 v1 的原则不是“完全重聚类”，而是“最小可执行修补”。

## 第 5 步：先生成一条 `split_to_singletons`

如果某个候选簇最终有 `K` 张图被选中拆出，就生成一条：

- `action = split_to_singletons`
- `anchor_image_id = 保留下来的 anchor`
- `member_image_ids = 这 K 张要被拆出的图`

执行结果是：

- anchor 留在原 cluster
- `member_image_ids` 里的每张图，各自变成一个新的 singleton

## 第 6 步：再看拆出去的图里，有没有应该重新抱团的小组

有时候人工判断不是“这几张都该各自单飞”，而是：

- 它们不该留在原大簇里
- 但其中两三张彼此又明显是同一个体

所以编译器会在“已经被拆出去的图”里面，再建一个 `yes graph`：

- 节点：被拆出的图
- 边：人工判成 `yes` 的 pair

然后找这个图里的连通分量。

如果某个分量满足：

- 大小大于 `1`
- 分量内部没有任何人工 `no` 冲突

就再补一条：

- `attach_to_anchor`

把这个小组重新挂到同一个 anchor 上。

通俗理解就是：

- 先保守拆开
- 再把人工明确说“它们彼此是一类”的小团体重新并回去

## 为什么当前 real run 里几乎没有 regroup

你这次实际标注的 `split` 信号，大部分是“很多 pair 都是 `no`”。

所以当前真实编译结果里：

- 被选中的 moved image 大多互相没有稳定 `yes`
- 或者即便有 `yes`，也伴随内部冲突

于是当前这版 `component_summary_v1.csv` 基本都是 singleton component，没有额外的 `attach_after_split=True`。

这说明你当前人工 split 更像是在做“错并切除”，而不是“拆后重组”。

## 输出是什么

编译器会产出两层结果。

### 第一层：分析产物

- 输出目录：
  - `artifacts/analysis/manual_split_compile_v1/`
- 关键文件：
  - `compiled_overlay_spec.json`
  - `tables/candidate_summary_v1.csv`
  - `tables/image_summary_v1.csv`
  - `tables/component_summary_v1.csv`
  - `tables/compiled_operations_v1.csv`
  - `reports/summary.md`

### 第二层：submission-ready 产物

- 由 overlay 执行器继续消费上面的 `compiled_overlay_spec.json`
- 输出目录：
  - `artifacts/submissions/manual_split_compiled_v1/`
- 关键文件：
  - `submission.csv`
  - `tables/test_predictions_v1.csv`
  - `tables/changed_images_v1.csv`
  - `tables/overlay_operations_v1.csv`
  - `tables/cluster_summary_v1.csv`
  - `reports/summary.md`

## 运行命令

```bash
python scripts/compile_manual_split_judgments.py \
  --base-submission-dir artifacts/submissions/kaggle_variant_lynx_seedsmooth_alpha0p15_onxgb_v1 \
  --pair-judgments artifacts/analysis/manual_review_sessions/autosave/manual_pair_review_v1.json \
  --output-dir artifacts/analysis/manual_split_compile_v1 \
  --submission-output-dir artifacts/submissions/manual_split_compiled_v1 \
  --submission-description "Manual split compiled overlay v1"
```

## 这次真实数据跑出来是什么样

基于你当前 autosave 的真实结果：

- 总 judgment：`415`
- 其中 `split judgment`：`304`
- 进入编译视野的 `split candidate`：`103`
- 真正生成 `split_to_singletons` 操作的 candidate：`42`
- 最终被挪动的图像数：`117`

对应的 `SalamanderID2025` cluster 形态变化是：

- `base_clusters: 310 -> overlay_clusters: 427`
- `base_singletons: 124 -> overlay_singletons: 260`

这说明两件事：

- 这条链路已经打通了，人工 split 可以稳定转成 submission
- 但这版 `split-only v1` 仍然偏激进，当前更像“把明显错并切掉很多”，还不是最终适合直接 official 的收敛形态

## 这条线下一步该怎么收

如果后面要继续把它做成更稳的 official 候选，优先级应该是：

1. 先收紧 split 门槛
   - 例如提高 `min_no_degree`
   - 或要求更高的 `net_no_margin`
2. 再引入更强的二次筛选
   - 例如只允许 `A/B` 级 candidate 编译
   - 或限制单个 cluster 最多可拆比例
3. 最后再考虑 merge / regroup
   - 也就是把你后面补看的 `merge` judgment 接进来

当前这版最重要的价值，不是“它已经是最优 submission”，而是：

- 我们已经把“人工 pair 判断 -> 可执行 overlay -> 可提交 submission”这条通路完全落地了。
