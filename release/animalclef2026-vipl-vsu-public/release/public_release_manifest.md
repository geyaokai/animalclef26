# Public Release Manifest

This manifest records what should be included in the public code release for
the AnimalCLEF 2026 VIPL-VSU working notes.

## Include

- `src/animalclef_analysis/`: reusable pipeline modules.
- `scripts/`: runnable scripts except private local audit scripts and one-off
  staging scripts that depend on private local artifacts.
- `tests/`: smoke and unit tests.
- `working_note/main.pdf`: working note PDF.
- `doc/04_算法与代码解释/43_人工歧义复核台.md`
- `doc/04_算法与代码解释/44_人工split判定编译器.md`
- `doc/04_算法与代码解释/45_人工pair约束图编译器.md`
- `sample_submission.csv`
- `metadata.csv`
- Release artifacts:
  - `artifacts/submissions/kaggle_variant_salamander_top10_manual_graph_on_062817_bestpublic_v1/tables/manual_top10_*.csv`
  - `artifacts/submissions/kaggle_variant_salamander_crossview_manual_yes_on_067578_v1/reports/summary.*`
  - `artifacts/submissions/kaggle_variant_salamander_crossview_manual_yes_on_067578_v1/tables/additional_manual_yes_*.csv`
  - `artifacts/submissions/kaggle_variant_salamander_crossview_manual_yes_on_067578_v1/tables/final_cluster_summary_v1.csv`
  - `artifacts/submissions/kaggle_variant_salamander_crossview_manual_yes_on_067578_v1/submission.csv`
  - `artifacts/analysis/texas_pair_registry_v3/summary.json`
  - `artifacts/analysis/texas_pair_registry_v3/texas_pair_registry_v3.csv`
- Selected route checkpoints and summaries:
  - `artifacts/training/experiments/ft_miew_arcface_masked_supcon_v1/checkpoints/last.pt`
  - `artifacts/training/experiments/ft_texas_miew_tcuwarmup_trusted_views_v1/checkpoints/best_checkpoint.pt`
  - `artifacts/training/experiments/ft_mega_arcface_distill_v1/` config and reports are included; the `best.pt` checkpoint is larger than GitHub LFS per-object limits and should be distributed through a separate artifact host.

## Exclude

- `images/`
- `artifacts.bak/`
- Large generated artifacts and caches.
- Python bytecode and local IDE/session files.
- Unselected exploratory or failed-run model weights.
- `submission_staging/` except no files by default.
- Private local staging/audit artifacts and the scripts that generate them.

## Notes For The Paper

The final constrained submission is human-in-the-loop. The release should state
that manual pair constraints are part of the reproduced final system. If raw
review boards are omitted, release the compact pair/constraint tables and the
changed-image summaries.
