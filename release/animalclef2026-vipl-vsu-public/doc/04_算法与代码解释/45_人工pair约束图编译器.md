# 人工 pair 约束图编译器

## 这个编译器解决什么问题

- 旧的 `manual_split_compiler` 已经能把人工 `split` judgment 编译成 `split_to_singletons / attach_to_anchor`。
- 但它本质上还是“按图像记票”：
  - 某张图被多少个 `no` 指到
  - 某张图是否该先拆出去
- 你后面提的关键想法更直接：
  - 人工说这两张图 `no`，那它们就不该再通过图聚类被并回同一类
  - 人工说这两张图 `yes`，那它们应该被当成高优先级支撑边

所以新的一层不是再做一版“更复杂的 split 记票”，而是：

- 把人工 `yes / no / uncertain` 直接翻译成 pair-level 图约束
- 在候选子图里重新做一次“受约束的连通块划分”
- 最后再把结果映射回现有 overlay 执行器能吃的操作

当前代码位置：

- 编译逻辑：`src/animalclef_analysis/manual_constraint_graph_compiler.py`
- 命令入口：`scripts/compile_manual_constraint_graph.py`

## 它和旧 `manual_split_compiler` 有什么本质区别

旧编译器的核心问题是：

- 它主要决定“哪张图该拆出去”
- 但一旦拆出去以后，原来那条 `no` pair 本身并没有变成硬约束

新编译器的核心变化是：

- `no` 不是“拆票”
- `no` 是真正的 `cannot-link`
- `yes` 不是“参考意见”
- `yes` 是高优先级并边

通俗说：

- 旧版更像“先按人的感觉挑嫌疑人，再做保守拆分”
- 新版更像“把人给出的禁并边和支撑边直接写进图规则里，然后再决定最后怎么分块”

## 输入是什么

### 1. 当前 base submission 的预测表

- 当前默认是：
  - `artifacts/submissions/manual_split_compiled_v1/tables/test_predictions_v1.csv`
- 也就是当前 public best 的那条 `Salamander manual split compiled overlay v1`

关键列：

- `dataset`
- `image_id`
- `pred_cluster_id`

作用：

- 它定义了“当前要在谁的基础上再做局部重编译”
- 新编译器不是从零重建整个 submission，而是在这条 base route 上再修 `SalamanderID2025`

### 2. pair graph 表

- 当前默认是：
  - `artifacts/analysis/salamander_ambiguity_map_probe_official_aligned_v1/tables/test_pair_disagreement_v1.csv`

关键列：

- `image_id`
- `neighbor_image_id`
- `xgb_same_identity_prob`
- `ambiguity_score`
- `base_cluster_left`
- `base_cluster_right`

作用：

- 给每个候选簇提供“局部 pair 图”
- 其中真正拿来排序并边强度的是 `xgb_same_identity_prob`

### 3. 人工 pair judgments

- 当前默认是：
  - `artifacts/analysis/manual_review_sessions/autosave/manual_pair_review_v1.json`

关键列：

- `dataset`
- `candidate_type`
- `candidate_key`
- `image_id`
- `neighbor_image_id`
- `label`
- `xgb_same_identity_prob`
- `ambiguity_score`

当前这版只消费：

- `candidate_type == split`

## 它是怎么工作的

## 第 1 步：只保留 `split` 候选里的人工 pair judgment

- 这版先不处理 `merge` candidate
- 也不把 `support` pair 单独编成规则

原因不是它们没价值，而是当前最成熟的数据仍然来自：

- 你已经人工确认“这个 base cluster 里面谁不能在一起”

所以先把最稳定的 `split` 证据落成可执行编译器。

## 第 2 步：对每个候选簇，收集局部子图

假设某个 `candidate_key=12`，对应当前 `SalamanderID2025` 的一个 base cluster。

编译器会收两类 pair：

- probe 表里的局部 pair
- 人工 judgment 里实际看过的 pair

然后把它们并成一张局部 pair 表：

- 节点数 = 这个候选簇涉及到的图像数，记作 `M`
- 边数 = probe 或 judgment 覆盖到的 pair 数，记作 `E`

每条边会带这些信息：

- `score = xgb_same_identity_prob`
- `manual_label in {yes, no, uncertain, ""}`
- `from_probe`
- `from_judgment`

这里要注意：

- 同一对图如果 probe 和 judgment 都出现，最终会合并成一条边
- `manual_label` 优先保留人工结果

## 第 3 步：把人工 `no` 变成真正的 `cannot-link`

对每条人工 `no` 边：

- 左右两个节点互相加入 `cannot-link` 邻接表

这一步的意思非常直接：

- 后面无论 `score` 多高
- 只要一次 merge 会让某条人工 `no` pair 落到同一个连通块里
- 这次 merge 就必须被挡掉

这就是“人工 `no` 直接作用在相似度图上”的具体实现。

## 第 4 步：按边分数从高到低尝试并块

编译器会把局部 pair 边按优先级排序：

1. `manual yes`
2. 其他 `score >= graph_threshold` 的边
3. 分数更低的边直接不考虑并块

当前默认：

- `graph_threshold = 0.25`

然后从高到低依次尝试 `union`。

如果某条边连接的两个节点已经在同一个分量里：

- 记成 `skip_same_component`

如果这条边是人工 `no`：

- 直接记成 `skip_manual_no`

如果这条边会触发某条 `cannot-link` 冲突：

- 记成 `block_due_cannot_link`
- 如果它原本还是人工 `yes`，会单独记成 `block_even_yes_due_cannot_link`

否则：

- 执行 `union`

## 第 5 步：把最终分量翻译回 overlay 规则

连通块跑完后，一个候选簇可能被切成多个 component。

例如当前真实 `cluster 12`：

- anchor component：`5565|5566|5567|5568`
- regroup component：`5569|5570|5594`
- singleton：`5937`

overlay 执行器并不认识“component”这个概念，所以要重新翻译成两步：

1. 先来一条 `split_to_singletons`
   - anchor 留在原簇
   - 其余 component 里的图先全部拆成 singleton
2. 对于每个大小大于 `1` 的非 anchor component
   - 再补一条 `attach_to_anchor`
   - 把这个 component 内的成员重新并到它自己的 anchor 上

所以最终还是落回现有的：

- `split_to_singletons`
- `attach_to_anchor`

这样就不需要重写 overlay 执行器。

## 第 6 步：为什么还要有 review gate

如果把所有看过一两个 pair 的 candidate 都直接编进去，会很激进。

所以新编译器加了两道门：

- `min_judged_pairs`
- `min_no_pairs`

当前更可信的一版是：

- `min_judged_pairs >= 2`
- `min_no_pairs >= 2`

它的意思是：

- 至少看过两对
- 而且至少已经确认两条 `no`

才允许这个 candidate 真正进入 constrained graph 重编译。

这本质上是在问：

- 这个簇的“不能在一起”证据是不是已经成形
- 还是只是刚看了一眼，信息量还不够

## 当前真实数据的两版结果

## 1. full 版

- 分析目录：
  - `artifacts/analysis/manual_constraint_graph_v1/`
- 提交包：
  - `artifacts/submissions/manual_constraint_graph_v1/`

结果：

- `99` 个 candidate 被编译
- `146` 张 `SalamanderID2025` 图被改动
- `117` 个 overlay operation
- `Salamander` cluster：`427 -> 469`
- singleton：`260 -> 336`

结论：

- 这版太激进
- 相对当前 public best 的 `split-only` 底座又继续把簇切得更碎

## 2. gate2 版

- 分析目录：
  - `artifacts/analysis/manual_constraint_graph_v1_gate2/`
- 提交包：
  - `artifacts/submissions/manual_constraint_graph_v1_gate2/`

运行参数：

- `graph_threshold=0.25`
- `min_judged_pairs=2`
- `min_no_pairs=2`

结果：

- `42` 个 candidate 被编译
- `61` 个 candidate 因为 review gate 被跳过
- `89` 张 `SalamanderID2025` 图被改动
- `60` 个 overlay operation
- `Salamander` cluster：`427 -> 412`
- singleton：`260 -> 229`

这版和 full 版最关键的区别是：

- 它不是继续把 `split-only` 结果切得更碎
- 而是把过于碎的部分，借助 pair-level 约束重新收回来一部分

所以如果后面要 official 验证：

- 应该优先送 `gate2`
- 不应该送 full

## 当前它为什么比旧 `split-only` 更像“修补”而不是“爆破”

旧 `manual_split_compiled_v1` 的形态是：

- `Salamander` `310 -> 427`
- singleton `124 -> 260`

这条线 public 的确已经证真，但它非常依赖：

- 人工把可疑成员果断拆出去

新 `gate2` 编译器是在同一份人工 judgment 上，进一步利用：

- `no` 的硬约束
- `yes` 的局部 regroup

把一部分“本来被拆成 singleton，但其实彼此还有稳定支撑关系”的小团体并回来。

所以它更像：

- 先承认 `split-only` 的 public 价值
- 再尝试把里面明显过碎的部分修补回来

## 输出是什么

### 第一层：分析产物

- 输出目录：
  - `artifacts/analysis/manual_constraint_graph_v1_gate2/`
- 关键文件：
  - `compiled_overlay_spec.json`
  - `tables/candidate_summary_v1.csv`
  - `tables/component_summary_v1.csv`
  - `tables/edge_summary_v1.csv`
  - `tables/compiled_operations_v1.csv`
  - `reports/summary.md`

### 第二层：submission-ready 产物

- 输出目录：
  - `artifacts/submissions/manual_constraint_graph_v1_gate2/`
- 关键文件：
  - `submission.csv`
  - `tables/test_predictions_v1.csv`
  - `tables/changed_images_v1.csv`
  - `tables/overlay_operations_v1.csv`
  - `tables/cluster_summary_v1.csv`
  - `reports/summary.md`

## 运行命令

```bash
python scripts/compile_manual_constraint_graph.py \
  --base-submission-dir artifacts/submissions/manual_split_compiled_v1 \
  --pair-graph-path artifacts/analysis/salamander_ambiguity_map_probe_official_aligned_v1/tables/test_pair_disagreement_v1.csv \
  --pair-judgments artifacts/analysis/manual_review_sessions/autosave/manual_pair_review_v1.json \
  --output-dir artifacts/analysis/manual_constraint_graph_v1_gate2 \
  --submission-output-dir artifacts/submissions/manual_constraint_graph_v1_gate2 \
  --submission-description "Manual constraint graph overlay v1 gate2" \
  --min-judged-pairs 2 \
  --min-no-pairs 2
```

## 当前这条线最重要的结论

不是：

- “它已经打赢当前 public best”

而是：

- 我们已经把“人工 pair judgment 直接写进图约束”这件事真正实现了
- 并且已经能在真实 autosave 上稳定产出 submission-ready 结果

接下来真正值得验证的问题只有一个：

- 相对已经 official 证真的 `manual_split_compiled_v1`
- 这种 pair-level `cannot-link / must-link` 重编译
- 能不能在 public 上把过碎的形态修回来一点，同时继续保住增益

如果要继续消耗 Kaggle 配额，当前只值得验证：

- `manual_constraint_graph_v1_gate2`

不值得验证：

- `manual_constraint_graph_v1` full 版
