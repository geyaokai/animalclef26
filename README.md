# AnimalCLEF 2026 VIPL-VSU Release

Public release for the AnimalCLEF 2026 working note:

**Dataset-Routed Clustering with Candidate Pair Constraints for AnimalCLEF 2026**

All release files are under:

```text
release/animalclef2026-vipl-vsu-public/
```

Quick map:

- `src/animalclef_analysis/`: reusable pipeline code.
- `scripts/`: experiment, inference, submission, and review utilities.
- `working_note/`: LaTeX source for the working note.
- `artifacts/submissions/.../submission.csv`: final submitted CSV.
- `artifacts/submissions/.../tables/`: compact Salamander constraint summaries.
- `artifacts/analysis/texas_pair_registry_v3/`: Texas reviewed pair registry.
- `artifacts/training/experiments/`: selected route configs, summaries, and available checkpoints.

Two checkpoints are tracked with Git LFS. The Mega route checkpoint is larger
than GitHub LFS per-object limits, so only its config and reports are included
here; the checkpoint should be hosted separately.

Raw images, large caches, generated review boards, and private local
staging/audit artifacts are not included.
