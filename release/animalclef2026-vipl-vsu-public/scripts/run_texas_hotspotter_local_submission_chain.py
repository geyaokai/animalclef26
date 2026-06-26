#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _seconds_until_next_utc_reset() -> int:
    now = datetime.now(timezone.utc)
    next_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
    return max(1, int((next_day - now).total_seconds()))


def _candidate_output_dir(base_dir: Path, rank: int) -> Path:
    return base_dir.parent / f"{base_dir.name}_rank{rank}"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    import sys

    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_hotspotter_local_clustering import (
        DEFAULT_BASE_PREDICTIONS_PATH,
        DEFAULT_BEST_PUBLIC_SCORE,
        DEFAULT_OUTPUT_DIR,
        build_texas_override_submission,
        count_kaggle_submissions_today,
        kaggle_submit_and_poll,
        load_distinct_saved_texas_local_candidates,
    )

    parser = argparse.ArgumentParser(
        description="Export and optionally submit the top distinct Texas HotSpotter local-clustering candidates."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-predictions-path", type=Path, default=DEFAULT_BASE_PREDICTIONS_PATH)
    parser.add_argument(
        "--submission-output-root",
        type=Path,
        default=Path("artifacts/submissions/kaggle_variant_texas_hotspotter_local_chain_v1"),
    )
    parser.add_argument("--num-distinct-candidates", type=int, default=2)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--wait-for-slot", action="store_true")
    parser.add_argument("--best-public-score", type=float, default=DEFAULT_BEST_PUBLIC_SCORE)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--poll-timeout-seconds", type=int, default=900)
    parser.add_argument("--description-prefix", type=str, default="Texas local chain")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    analysis_dir = (args.analysis_dir if args.analysis_dir.is_absolute() else (repo_root / args.analysis_dir)).resolve()
    output_root = (
        args.submission_output_root
        if args.submission_output_root.is_absolute()
        else (repo_root / args.submission_output_root)
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_dir = output_root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_distinct_saved_texas_local_candidates(
        analysis_dir=analysis_dir,
        top_n=max(1, int(args.num_distinct_candidates)),
    )
    if not candidates:
        raise ValueError(f"No distinct Texas local candidates found under {analysis_dir}")

    exported: list[dict[str, object]] = []
    for rank, candidate in enumerate(candidates, start=1):
        route_name = f"hotspotter_local_{candidate.score_matrix_name}_{candidate.method}_rank{rank}"
        candidate_output_dir = _candidate_output_dir(output_root, rank)
        outputs = build_texas_override_submission(
            repo_root=repo_root,
            base_predictions_path=args.base_predictions_path,
            texas_pred_df=candidate.prediction_df,
            output_dir=candidate_output_dir,
            route_name=route_name,
            route_summary=candidate.summary_row,
        )
        exported.append(
            {
                "rank": int(rank),
                "route_name": route_name,
                "candidate_name": candidate.name,
                "summary_row": candidate.summary_row,
                "output_dir": str(candidate_output_dir),
                "submission_path": str(outputs["submission_path"]),
                "summary_path": str(outputs["summary_path"]),
            }
        )

    submission_log: list[dict[str, object]] = []
    current_best = float(args.best_public_score)

    if args.submit:
        while True:
            used_slots, submission_listing = count_kaggle_submissions_today(repo_root=repo_root)
            if used_slots < 5:
                break
            if not args.wait_for_slot:
                raise RuntimeError(
                    f"Kaggle daily quota already exhausted ({used_slots}/5). Re-run with --wait-for-slot inside tmux."
                )
            sleep_seconds = _seconds_until_next_utc_reset()
            wait_record = {
                "event": "wait_for_utc_reset",
                "used_slots": int(used_slots),
                "sleep_seconds": int(sleep_seconds),
                "utc_now": datetime.now(timezone.utc).isoformat(),
            }
            submission_log.append(wait_record)
            (runtime_dir / "submission_chain_progress.json").write_text(
                json.dumps({"exported": exported, "submission_log": submission_log}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            time.sleep(sleep_seconds)

        for item in exported:
            rank = int(item["rank"])
            candidate_summary = item["summary_row"]
            description = (
                f"{args.description_prefix} r{rank} "
                f"{candidate_summary.get('score_matrix_name', '')} "
                f"{candidate_summary.get('method', '')} "
                f"{candidate_summary.get('param_key', '')}"
            )
            description = description.replace(",", ";")[:180]
            submit_result = kaggle_submit_and_poll(
                repo_root=repo_root,
                submission_path=Path(item["submission_path"]),
                description=description,
                poll_seconds=int(args.poll_seconds),
                timeout_seconds=int(args.poll_timeout_seconds),
            )
            record = {
                "rank": rank,
                "description": description,
                "candidate_name": item["candidate_name"],
                "submission_path": item["submission_path"],
                "submit_result": submit_result,
                "best_public_score_before_submit": current_best,
            }
            public_score = submit_result.get("public_score")
            improved = public_score is not None and float(public_score) > current_best
            record["improved"] = improved
            submission_log.append(record)
            (runtime_dir / "submission_chain_progress.json").write_text(
                json.dumps({"exported": exported, "submission_log": submission_log}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if public_score is not None:
                current_best = max(current_best, float(public_score))
            if rank == 1 and not improved:
                break

    report = {
        "analysis_dir": str(analysis_dir),
        "best_public_score_input": float(args.best_public_score),
        "exported": exported,
        "submission_log": submission_log,
    }
    report_path = runtime_dir / "submission_chain_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[texas_hotspotter_chain] report: {report_path}")
    for item in exported:
        print(f"[texas_hotspotter_chain] rank{item['rank']} submission: {item['submission_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
