#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _resolve_path(repo_root: Path, value: Path) -> Path:
    return (value if value.is_absolute() else (repo_root / value)).resolve()


def _slugify(value: str) -> str:
    chars: list[str] = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")
    compact = "".join(chars)
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact.strip("_")


def _threshold_slug(value: float | int | str | None) -> str:
    if value is None:
        return "na"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _slugify(str(value))
    text = f"{number:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _load_candidate_rows(candidate_summary_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(candidate_summary_path).copy()
    if frame.empty:
        raise ValueError(f"No candidate rows found in {candidate_summary_path}")
    frame.insert(0, "candidate_rank", range(1, len(frame) + 1))
    return frame


def _prediction_table_path(analysis_output_dir: Path, candidate_row: pd.Series) -> Path:
    matrix_name = str(candidate_row["score_matrix_name"])
    method = str(candidate_row["method"])
    safe_name = f"{matrix_name}__{method}".replace(":", "__")
    return analysis_output_dir / "tables" / f"{safe_name}_predictions_v1.csv"


def _match_candidate_predictions(prediction_df: pd.DataFrame, candidate_row: pd.Series) -> pd.DataFrame:
    matched = prediction_df.copy()
    if "score_matrix_name" in matched.columns:
        matched = matched[matched["score_matrix_name"].astype(str) == str(candidate_row["score_matrix_name"])].copy()
    if "clustering_method" in matched.columns:
        matched = matched[matched["clustering_method"].astype(str) == str(candidate_row["method"])].copy()
    if "chosen_threshold" in matched.columns and not pd.isna(candidate_row.get("threshold")):
        threshold = float(candidate_row["threshold"])
        matched = matched[pd.to_numeric(matched["chosen_threshold"], errors="coerce").sub(threshold).abs() <= 1e-9].copy()
    if "graph_top_k" in matched.columns and not pd.isna(candidate_row.get("graph_top_k")):
        graph_top_k = int(candidate_row["graph_top_k"])
        matched = matched[pd.to_numeric(matched["graph_top_k"], errors="coerce").fillna(-1).astype(int) == graph_top_k].copy()
    if "mutual_top_k" in matched.columns and not pd.isna(candidate_row.get("mutual_top_k")):
        desired = bool(candidate_row["mutual_top_k"])
        matched = matched[matched["mutual_top_k"].fillna(False).astype(bool) == desired].copy()
    if matched.empty:
        raise ValueError(
            "Candidate prediction filter produced no rows: "
            f"score_matrix={candidate_row.get('score_matrix_name')} method={candidate_row.get('method')} "
            f"threshold={candidate_row.get('threshold')} graph_top_k={candidate_row.get('graph_top_k')} "
            f"mutual_top_k={candidate_row.get('mutual_top_k')}"
        )
    matched["image_id"] = matched["image_id"].astype(str)
    if matched["image_id"].duplicated().any():
        dupes = matched.loc[matched["image_id"].duplicated(), "image_id"].astype(str).head(5).tolist()
        raise ValueError(f"Matched candidate predictions still contain duplicate image_id rows, examples: {dupes}")
    return matched.reset_index(drop=True)


def _candidate_route_name(candidate_row: pd.Series) -> str:
    return (
        f"hotspotter_auto_r{int(candidate_row['candidate_rank']):02d}_"
        f"{_slugify(str(candidate_row['score_matrix_name']))}_"
        f"{_slugify(str(candidate_row['method']))}_"
        f"thr{_threshold_slug(candidate_row.get('threshold'))}"
    )


def _candidate_description(candidate_row: pd.Series, prefix: str) -> str:
    return (
        f"{prefix} r{int(candidate_row['candidate_rank'])} "
        f"{candidate_row['score_matrix_name']} {candidate_row['method']} "
        f"thr={float(candidate_row.get('threshold', 0.0)):.4f} "
        f"ps={float(candidate_row.get('proxy_score', 0.0)):.5f}"
    )


def _build_candidate_submission(
    *,
    repo_root: Path,
    analysis_output_dir: Path,
    base_predictions_path: Path,
    output_root: Path,
    candidate_row: pd.Series,
) -> dict[str, Any]:
    from animalclef_analysis.texas_hotspotter_local_clustering import build_texas_override_submission

    prediction_path = _prediction_table_path(analysis_output_dir, candidate_row)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Candidate prediction table not found: {prediction_path}")
    prediction_df = pd.read_csv(prediction_path).copy()
    selected_prediction_df = _match_candidate_predictions(prediction_df=prediction_df, candidate_row=candidate_row)
    route_name = _candidate_route_name(candidate_row)
    candidate_dir = output_root / f"candidate_r{int(candidate_row['candidate_rank']):02d}_{route_name}"
    build_outputs = build_texas_override_submission(
        repo_root=repo_root,
        base_predictions_path=base_predictions_path,
        texas_pred_df=selected_prediction_df,
        output_dir=candidate_dir,
        route_name=route_name,
        route_summary=_json_ready(candidate_row.to_dict()),
    )
    return {
        "rank": int(candidate_row["candidate_rank"]),
        "description": _candidate_description(candidate_row, prefix="Texas local-clustering auto"),
        "route_name": route_name,
        "candidate_dir": candidate_dir,
        "prediction_table_path": prediction_path,
        "selected_rows": int(len(selected_prediction_df)),
        "summary_row": _json_ready(candidate_row.to_dict()),
        "build_outputs": _json_ready(build_outputs),
    }


def _submit_candidate(
    *,
    repo_root: Path,
    candidate_bundle: dict[str, Any],
    poll_seconds: int,
    timeout_seconds: int,
    daily_limit: int,
) -> dict[str, Any]:
    from animalclef_analysis.texas_hotspotter_local_clustering import (
        count_kaggle_submissions_today,
        kaggle_submit_and_poll,
    )

    before_used, before_listing = count_kaggle_submissions_today(repo_root=repo_root)
    if before_used >= daily_limit:
        return {
            "description": candidate_bundle["description"],
            "status": "skipped_daily_limit",
            "used_slots_before": int(before_used),
            "used_slots_after": int(before_used),
            "daily_limit": int(daily_limit),
            "listing_before": before_listing,
            "listing_after": before_listing,
            "public_score": None,
        }

    submit_result = kaggle_submit_and_poll(
        repo_root=repo_root,
        submission_path=Path(candidate_bundle["build_outputs"]["submission_path"]),
        description=str(candidate_bundle["description"]),
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
    after_used, after_listing = count_kaggle_submissions_today(repo_root=repo_root)
    stdout_text = str(submit_result.get("submit_stdout", ""))
    stderr_text = str(submit_result.get("submit_stderr", ""))
    probable_quota = (
        int(submit_result.get("submit_returncode", 0)) != 0
        and int(after_used) == int(before_used)
        and ("400" in stdout_text or "400" in stderr_text)
    )
    status = str(submit_result.get("status", ""))
    if probable_quota and not status:
        status = "submit_failed_probable_quota"
    submit_result.update(
        {
            "description": candidate_bundle["description"],
            "used_slots_before": int(before_used),
            "used_slots_after": int(after_used),
            "daily_limit": int(daily_limit),
            "listing_before": before_listing,
            "listing_after": after_listing,
            "probable_quota_exhaustion": bool(probable_quota),
            "status": status or "submitted",
        }
    )
    return _json_ready(submit_result)


def _write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Texas Local-Clustering Auto Submit",
        "",
        f"- Analysis output dir: `{summary['analysis_output_dir']}`",
        f"- Base predictions path: `{summary['base_predictions_path']}`",
        f"- Output root: `{summary['output_root']}`",
        f"- Runtime dir: `{summary['runtime_dir']}`",
        f"- Score gate for candidate 2: `{summary['submit_second_if_score_gt']}`",
        f"- Dry run: `{summary['dry_run']}`",
        "",
        "## Candidate 1",
        "",
        f"- Rank: `{summary['candidate_1']['rank']}`",
        f"- Description: `{summary['candidate_1']['description']}`",
        f"- Submission path: `{summary['candidate_1']['build_outputs']['submission_path']}`",
        f"- Proxy score: `{summary['candidate_1']['summary_row']['proxy_score']}`",
        "",
        "## Candidate 2",
        "",
        f"- Rank: `{summary['candidate_2']['rank']}`",
        f"- Description: `{summary['candidate_2']['description']}`",
        f"- Submission path: `{summary['candidate_2']['build_outputs']['submission_path']}`",
        f"- Proxy score: `{summary['candidate_2']['summary_row']['proxy_score']}`",
        "",
        "## Kaggle",
        "",
        f"- Candidate 1 status: `{summary['candidate_1_submit']['status']}`",
        f"- Candidate 1 public score: `{summary['candidate_1_submit']['public_score']}`",
        f"- Candidate 2 decision: `{summary['candidate_2_decision']}`",
        f"- Candidate 2 status: `{summary['candidate_2_submit']['status']}`",
        f"- Candidate 2 public score: `{summary['candidate_2_submit']['public_score']}`",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_hotspotter_local_clustering import (
        DEFAULT_BASE_PREDICTIONS_PATH,
        DEFAULT_BEST_PUBLIC_SCORE,
    )

    parser = argparse.ArgumentParser(
        description="Build two Texas local-clustering candidate submissions, submit candidate 1, poll Kaggle, and gate candidate 2 on candidate 1 public score."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        default=Path("artifacts/analysis/texas_hotspotter_local_clustering_v1"),
    )
    parser.add_argument("--base-predictions-path", type=Path, default=DEFAULT_BASE_PREDICTIONS_PATH)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/submissions/kaggle_variant_texas_hotspotter_local_autosubmit_v1"),
    )
    parser.add_argument("--first-rank", type=int, default=1)
    parser.add_argument("--second-rank", type=int, default=2)
    parser.add_argument("--submit-second-if-score-gt", type=float, default=DEFAULT_BEST_PUBLIC_SCORE)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--poll-timeout-seconds", type=int, default=900)
    parser.add_argument("--daily-limit", type=int, default=5)
    parser.add_argument("--proxy-url", type=str, default="http://127.0.0.1:9999")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("http_proxy", args.proxy_url)
    os.environ.setdefault("https_proxy", args.proxy_url)

    repo_root = args.repo_root.resolve()
    analysis_output_dir = _resolve_path(repo_root, args.analysis_output_dir)
    base_predictions_path = _resolve_path(repo_root, args.base_predictions_path)
    output_root = _resolve_path(repo_root, args.output_root)
    runtime_dir = output_root / "runtime"
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    candidate_summary_path = analysis_output_dir / "tables" / "candidate_summary_v1.csv"
    if not candidate_summary_path.exists():
        raise FileNotFoundError(f"Candidate summary not found: {candidate_summary_path}")

    candidate_df = _load_candidate_rows(candidate_summary_path)
    if args.first_rank < 1 or args.second_rank < 1:
        raise ValueError("Candidate ranks must be >= 1.")
    if args.first_rank == args.second_rank:
        raise ValueError("--first-rank and --second-rank must be different.")
    if args.first_rank > len(candidate_df) or args.second_rank > len(candidate_df):
        raise ValueError(f"Candidate rank exceeds available rows: total={len(candidate_df)}")

    candidate_1_row = candidate_df.iloc[args.first_rank - 1].copy()
    candidate_2_row = candidate_df.iloc[args.second_rank - 1].copy()
    candidate_1 = _build_candidate_submission(
        repo_root=repo_root,
        analysis_output_dir=analysis_output_dir,
        base_predictions_path=base_predictions_path,
        output_root=output_root,
        candidate_row=candidate_1_row,
    )
    candidate_2 = _build_candidate_submission(
        repo_root=repo_root,
        analysis_output_dir=analysis_output_dir,
        base_predictions_path=base_predictions_path,
        output_root=output_root,
        candidate_row=candidate_2_row,
    )

    candidate_1_submit: dict[str, Any]
    candidate_2_submit: dict[str, Any]
    candidate_2_decision = "not_evaluated"
    if args.dry_run:
        candidate_1_submit = {"status": "dry_run", "public_score": None}
        candidate_2_submit = {"status": "dry_run", "public_score": None}
        candidate_2_decision = "dry_run"
    else:
        candidate_1_submit = _submit_candidate(
            repo_root=repo_root,
            candidate_bundle=candidate_1,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.poll_timeout_seconds,
            daily_limit=args.daily_limit,
        )
        first_score = candidate_1_submit.get("public_score")
        if isinstance(first_score, (int, float)) and float(first_score) > float(args.submit_second_if_score_gt):
            candidate_2_decision = "submit_candidate_2"
            candidate_2_submit = _submit_candidate(
                repo_root=repo_root,
                candidate_bundle=candidate_2,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.poll_timeout_seconds,
                daily_limit=args.daily_limit,
            )
        else:
            candidate_2_decision = (
                f"skip_candidate_2_first_score={first_score}"
                if first_score is not None
                else f"skip_candidate_2_first_status={candidate_1_submit.get('status', '')}"
            )
            candidate_2_submit = {"status": "skipped_gate", "public_score": None}

    summary = _json_ready(
        {
            "analysis_output_dir": analysis_output_dir,
            "base_predictions_path": base_predictions_path,
            "output_root": output_root,
            "runtime_dir": runtime_dir,
            "submit_second_if_score_gt": float(args.submit_second_if_score_gt),
            "dry_run": bool(args.dry_run),
            "candidate_1": candidate_1,
            "candidate_2": candidate_2,
            "candidate_1_submit": candidate_1_submit,
            "candidate_2_decision": candidate_2_decision,
            "candidate_2_submit": candidate_2_submit,
        }
    )
    summary_json_path = runtime_dir / "autosubmit_summary.json"
    summary_md_path = runtime_dir / "autosubmit_summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(summary_md_path, summary)

    print(f"[texas_local_autosubmit] summary_json={summary_json_path}")
    print(f"[texas_local_autosubmit] summary_md={summary_md_path}")
    print(f"[texas_local_autosubmit] candidate_1_submission={candidate_1['build_outputs']['submission_path']}")
    print(f"[texas_local_autosubmit] candidate_2_submission={candidate_2['build_outputs']['submission_path']}")
    print(f"[texas_local_autosubmit] candidate_1_status={candidate_1_submit['status']}")
    print(f"[texas_local_autosubmit] candidate_2_status={candidate_2_submit['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
