# VIPL-VSU AnimalCLEF 2026 Public Code Release

This directory is the public release package for the VIPL-VSU AnimalCLEF 2026
working note. It contains code, the paper PDF, the final submission CSV,
selected checkpoints, and compact pair-constraint artifacts.

Main entry points:

- `src/animalclef_analysis/`: reusable pipeline modules.
- `scripts/`: runnable training, inference, review, and submission utilities.
- `working_note/main.pdf`: working note PDF.
- `artifacts/submissions/kaggle_variant_salamander_crossview_manual_yes_on_067578_v1/submission.csv`: final submitted CSV.
- `artifacts/analysis/texas_pair_registry_v3/`: compact Texas pair registry.
- `artifacts/training/experiments/`: selected configs, summaries, and checkpoints.

Notes:

- Two checkpoints are included through Git LFS.
- The Mega route `best.pt` checkpoint is larger than GitHub LFS per-object
  limits, so only its config and reports are included here.
- Raw images, large caches, generated review boards, and private local
  staging/audit artifacts are not included.

See `release/public_release_manifest.md` for the detailed file policy.
