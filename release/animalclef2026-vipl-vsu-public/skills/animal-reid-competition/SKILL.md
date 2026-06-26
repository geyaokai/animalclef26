---
name: animal-reid-competition
description: Use this skill for Kaggle-style animal re-identification or wildlife individual clustering competitions, especially when the task has multiple species/datasets, open-set or test-only clustering, foundation descriptors, reranking, pseudo-labeling, manual review, and limited official submissions.
---

# Animal Re-ID Competition Workflow

Use this workflow when helping with an animal individual identification, wildlife re-ID, or unsupervised clustering competition.

## Core Principle

Treat the system as a dataset-routed clustering pipeline, not as one universal classifier. First maximize candidate recall with strong global descriptors, then improve edge quality with dataset-specific rerank, pairwise features, graph constraints, pseudo-labeling, or human review.

## 1. Audit Before Modeling

Start by answering:

- What are the datasets/species, splits, image counts, and identity counts?
- Which datasets have local labels and which are test-only?
- How many singleton identities exist?
- Are there duplicate images, leaks, path issues, corrupt images, or train/test resolution shifts?
- Does test cluster count have a prior from identity numbering, image density, metadata, or domain knowledge?
- Is the benchmark open-set retrieval, closed-set classification, or clustering? Do not reuse thresholds from a different task type without validation.

Save audit outputs as tables plus a short report. Important conclusions should be durable and easy to cite later.

## 2. Build Strong Global Baselines

Prefer foundation wildlife descriptors before training from scratch.

Default baseline order:

1. Run each descriptor independently.
2. L2-normalize embeddings.
3. Try simple early fusion such as normalized concat or weighted concat.
4. Cluster per dataset/species, not across all species together.
5. Sweep thresholds and report ARI, NMI, pairwise precision/recall/F1, Recall@k, cluster count, singleton ratio, and largest cluster size where applicable.

Do not treat Recall@k as the final metric. Use it to diagnose whether a reranker has enough candidate recall to work.

## 3. Route by Dataset

For each dataset/species, decide its own route:

- Backbone or descriptor.
- Preprocessing view.
- Threshold range.
- Whether to use local rerank.
- Whether to use pseudo-labeling or transductive postprocess.
- Whether manual review is justified.

Do not force one backbone, one threshold, or one postprocess across all datasets unless evidence supports it.

## 4. Candidate Recall Then Rerank

Before adding local matching or pairwise models, run a candidate recall audit:

- For each query with at least one positive in validation, compute whether a same-identity image appears in top-1, top-5, top-10, top-20, top-50, etc.
- Find the smallest k that reaches the target recall band.
- If top-K recall is poor, fix the global embedding before blaming local rerank.

Use expensive local matching only on shortlisted candidate pairs.

## 5. Convert Domain Priors Into Soft Features

Domain priors can be powerful but brittle. Prefer using them as features before using them as hard rules.

Examples:

- Pattern masks.
- Color bands.
- Body alignment confidence.
- Local matcher inliers.
- Patch similarity.
- Same seed cluster flags.
- View consistency.

Feed these into a pairwise model or scoring layer where possible. Hard vetoes, forced merges, and global penalties should be treated as high-risk until validated.

## 6. Use Human Review Deliberately

Use human effort on ambiguity zones:

- Candidate splits where one cluster may contain multiple identities.
- Candidate merges where two clusters may be the same identity.
- High-disagreement pairs across routes.
- Pairs where model score and visual evidence conflict.

Prefer pair-first judgments:

- `must-link`
- `cannot-link`
- `uncertain`

Compile judgments into graph constraints or overlays, and record the exact number of changed images, clusters, and singleton changes.

## 7. Pseudo-Labeling And Test-Only Datasets

For datasets with no local labels:

- Start with cluster-shape sanity checks.
- Build high-confidence seed clusters from conservative thresholds or trusted human-reviewed examples.
- Keep pseudo-labels small and clean before expanding coverage.
- Use view consistency or intra-image augmentations to train uncertain images without pretending all uncertain clusters are true labels.
- Build proxy metrics, but audit failure modes where the proxy can overvalue bad merges.

Do not let pseudo-labeling outrun the evidence quality.

## 8. Official Submission Discipline

When submissions are limited, each official run should be a single primary factor against a known base.

Record:

- Base submission and score.
- Intended single change.
- Dataset affected.
- Expected mechanism.
- Risk.
- Result score and delta.
- Whether the change is retained, rejected, or needs another validation.

Avoid spending slots on changes that cannot be attributed.

## 9. Stop-Loss Rules

Keep negative results. They prevent repeated mistakes.

Common stop-loss signals:

- Local validation improves but public LB drops with a clear distribution mismatch.
- Proxy score improves while seed clusters gain outsiders.
- Hard veto creates too many singletons.
- Ensemble increases merge errors around high-value seed clusters.
- A local matcher helps one dataset but hurts others.
- A unified backbone improves macro but loses the dataset that matters for the current route.

When a result fails, write whether the idea is invalid or whether only the current integration method failed.

## 10. Report Template

A good experiment or competition summary should include:

- Task type and metric.
- Data audit.
- Baseline routes.
- Dataset-specific route table.
- Validation protocol and split caveats.
- Main improvements with before/after metrics.
- Negative results and why they failed.
- Manual review or human prior usage.
- Final system architecture.
- Reusable lessons.
- Next actions if continuing.
