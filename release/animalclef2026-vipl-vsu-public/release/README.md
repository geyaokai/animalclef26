# AnimalCLEF 2026 Public Release Staging

This directory contains release metadata and helper scripts. It is not the
competition workspace itself.

Use `scripts/prepare_public_release.py` from the repository root to build a
clean public package under `release/animalclef2026-vipl-vsu-public/`.

Release policy:

- Public: source code, tests, working-note LaTeX source, route summaries,
  final submission CSV, selected route checkpoints, and compact manual
  constraint summaries for Salamander and Texas.
- Excluded: raw images, large caches, generated HTML review boards, local
  environment files, and private local staging/audit artifacts.
