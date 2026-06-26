#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import gradio as gr
import pandas as pd


DEFAULT_BASE_PREDICTIONS = Path("artifacts/submissions/kaggle_variant_lynx_seedsmooth_alpha0p15_onxgb_v1")
DEFAULT_PROBE_DIR = Path("artifacts/analysis/salamander_ambiguity_map_probe_official_aligned_v1")
DEFAULT_EXPORT_JUDGMENTS = Path("artifacts/analysis/manual_review_sessions/pair_judgments_v1.json")
DEFAULT_AUTOSAVE_DIR = Path("artifacts/analysis/manual_review_sessions/autosave")
DEFAULT_SESSION_NAME = "manual_pair_review_v1"

PAIR_TABLE_COLUMNS = [
    "image_id",
    "neighbor_image_id",
    "yes_priority_score",
    "yes_candidate_reason",
    "xgb_same_identity_prob",
    "local_score",
    "route_global_score",
    "ambiguity_score",
    "vote_direction",
    "merge_votes",
    "split_votes",
]
TASK_STATE_COLUMNS = [
    "status_code",
    "status",
    "dataset",
    "candidate_type",
    "candidate_key",
    "progress",
    "judged_pairs",
    "pair_total",
    "yes_count",
    "no_count",
    "uncertain_count",
    "priority_score",
    "size_hint",
    "preview",
    "priority_rank",
    "status_rank",
]


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _empty_pair_table() -> pd.DataFrame:
    return _empty_df(PAIR_TABLE_COLUMNS)


def _empty_task_state_df() -> pd.DataFrame:
    return _empty_df(TASK_STATE_COLUMNS)


def _format_task_table(task_df: pd.DataFrame) -> pd.DataFrame:
    if task_df is None or task_df.empty:
        return pd.DataFrame(
            columns=[
                "状态",
                "进度",
                "数据集",
                "类型",
                "候选",
                "已判",
                "总 pair",
                "yes",
                "no",
                "uncertain",
                "优先分",
                "规模",
                "说明",
            ]
        )
    display_df = task_df.loc[
        :,
        [
            "status",
            "progress",
            "dataset",
            "candidate_type",
            "candidate_key",
            "judged_pairs",
            "pair_total",
            "yes_count",
            "no_count",
            "uncertain_count",
            "priority_score",
            "size_hint",
            "preview",
        ],
    ].copy()
    return display_df.rename(
        columns={
            "status": "状态",
            "progress": "进度",
            "dataset": "数据集",
            "candidate_type": "类型",
            "candidate_key": "候选",
            "judged_pairs": "已判",
            "pair_total": "总 pair",
            "yes_count": "yes",
            "no_count": "no",
            "uncertain_count": "uncertain",
            "priority_score": "优先分",
            "size_hint": "规模",
            "preview": "说明",
        }
    )


def _task_board_markdown(task_df: pd.DataFrame) -> str:
    if task_df is None or task_df.empty:
        return "- 当前没有 candidate task。"
    status_counts = task_df["status_code"].astype(str).value_counts().to_dict()
    return "\n".join(
        [
            f"- `total_tasks`: `{len(task_df)}`",
            f"- `in_progress`: `{int(status_counts.get('in_progress', 0))}`",
            f"- `pending`: `{int(status_counts.get('pending', 0))}`",
            f"- `completed`: `{int(status_counts.get('completed', 0))}`",
            f"- `empty`: `{int(status_counts.get('empty', 0))}`",
            "- 点下面任务表的某一行，就会加载对应 candidate。",
        ]
    )


def _selected_candidate_text(task_df: pd.DataFrame, candidate_value: str | None) -> str:
    if task_df is None or task_df.empty or not candidate_value:
        return "未选中任务。"
    subset = task_df[task_df["candidate_key"].astype(str).eq(str(candidate_value))].copy()
    if subset.empty:
        return "未选中任务。"
    row = subset.iloc[0]
    return (
        f"当前任务：`{row['candidate_type']}` `{row['candidate_key']}` | "
        f"`{row['dataset']}` | `{row['status']}` | "
        f"`{row['progress']}`"
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.manual_review_workbench import (
        PAIR_LABEL_NO,
        PAIR_LABEL_UNCERTAIN,
        PAIR_LABEL_YES,
        build_candidate_task_table,
        build_judgment_preview_json,
        clear_pair_judgments,
        export_pair_judgments,
        find_next_unjudged_pair_index,
        judgments_to_dataframe,
        load_pair_judgments,
        load_review_bundle,
        remove_pair_judgment_at,
        render_candidate_view,
        render_pair_detail,
        suggest_candidate_value,
        summarize_pair_judgments,
        upsert_pair_judgment,
    )

    parser = argparse.ArgumentParser(description="Launch pair-first manual ambiguity review workbench.")
    parser.add_argument("--base-predictions", type=Path, default=DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    autosave_dir = (repo_root / DEFAULT_AUTOSAVE_DIR).resolve()
    autosave_dir.mkdir(parents=True, exist_ok=True)
    last_session_path = autosave_dir / "last_session_name.txt"
    bundle_cache: dict[str, object] = {"token": "", "candidate_payloads": {}}

    def _session_cache_tag(session_name: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9]+", "_", str(session_name).strip()).strip("_").lower()
        return text or DEFAULT_SESSION_NAME

    def _autosave_path(session_name: str) -> Path:
        return autosave_dir / f"{_session_cache_tag(session_name)}.json"

    def _remember_last_session(session_name: str) -> None:
        resolved_session_name = str(session_name).strip() or DEFAULT_SESSION_NAME
        last_session_path.write_text(resolved_session_name, encoding="utf-8")

    def _read_last_session() -> str:
        if not last_session_path.exists():
            return ""
        return last_session_path.read_text(encoding="utf-8").strip()

    def _resolve_restore_session_name(requested_session_name: str) -> str:
        requested = str(requested_session_name).strip()
        if requested:
            return requested
        last_session_name = _read_last_session()
        if last_session_name:
            return last_session_name
        return DEFAULT_SESSION_NAME

    def _restore_cached_judgments(requested_session_name: str) -> tuple[str, list[dict], Path, bool]:
        resolved_session_name = _resolve_restore_session_name(requested_session_name)
        cache_path = _autosave_path(resolved_session_name)
        loaded_session_name, judgments = load_pair_judgments(cache_path)
        effective_session_name = loaded_session_name or resolved_session_name
        return effective_session_name, judgments, cache_path, cache_path.exists()

    def _autosave_judgments(session_name: str, judgments: list[dict]) -> Path:
        resolved_session_name = str(session_name).strip() or DEFAULT_SESSION_NAME
        path = export_pair_judgments(
            session_name=resolved_session_name,
            judgments=judgments,
            output_path=_autosave_path(resolved_session_name),
        )
        _remember_last_session(resolved_session_name)
        return path

    def _selected_judgment_text(judgments: list[dict], selected_index: int | None) -> str:
        if selected_index is None or selected_index < 0 or selected_index >= len(judgments):
            return "未选中 judgment。"
        item = judgments[selected_index]
        return (
            f"当前选中：row `{selected_index}` | "
            f"`{item.get('candidate_type', '')}` `{item.get('candidate_key', '')}` | "
            f"`{item.get('image_id', '')}` vs `{item.get('neighbor_image_id', '')}` | "
            f"`{item.get('label', '')}`"
        )

    def _judgment_outputs(
        judgments: list[dict],
        *,
        session_name: str = DEFAULT_SESSION_NAME,
        selected_index: int | None = None,
    ) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, str, int | None, str]:
        return (
            judgments,
            judgments_to_dataframe(judgments),
            summarize_pair_judgments(judgments),
            build_judgment_preview_json(session_name, judgments),
            selected_index,
            _selected_judgment_text(judgments, selected_index),
        )

    def _pair_table_for_display(pair_df: pd.DataFrame) -> pd.DataFrame:
        if pair_df is None or pair_df.empty:
            return _empty_pair_table()
        columns = [column for column in PAIR_TABLE_COLUMNS if column in pair_df.columns]
        return pair_df.loc[:, columns].copy()

    def _bundle_token(bundle) -> str:
        if bundle is None:
            return ""
        return f"{bundle.predictions_path}|{bundle.probe_dir}"

    def _ensure_bundle_cache(bundle) -> None:
        token = _bundle_token(bundle)
        if str(bundle_cache.get("token", "")) != token:
            bundle_cache["token"] = token
            bundle_cache["candidate_payloads"] = {}

    def _get_candidate_payload(bundle, direction: str, candidate_value: str):
        _ensure_bundle_cache(bundle)
        cache_key = (str(direction), str(candidate_value))
        payloads = bundle_cache["candidate_payloads"]
        if cache_key not in payloads:
            payloads[cache_key] = render_candidate_view(bundle, direction, str(candidate_value))
        return payloads[cache_key]

    def _empty_candidate_outputs():
        empty_pair_df = _empty_pair_table()
        return (
            "",
            "",
            [],
            empty_pair_df,
            gr.update(choices=[], value=None),
            "",
            empty_pair_df,
            None,
            None,
            "_No support pair selected._",
            "",
            "",
        )

    def _candidate_outputs(
        bundle,
        direction: str,
        candidate_value: str | None,
        *,
        judgments=None,
        pair_value: str | None = None,
        note_value: str | None = None,
        action_status: str = "",
    ):
        if bundle is None or not candidate_value:
            return _empty_candidate_outputs()
        payload = _get_candidate_payload(bundle, direction, str(candidate_value))
        pair_df = payload["pair_df"]
        pair_choices = payload["pair_choices"]
        valid_pair_values = {str(value) for _, value in pair_choices}
        resolved_pair_value = str(pair_value) if pair_value is not None and str(pair_value) in valid_pair_values else None
        if resolved_pair_value is None:
            next_unjudged_idx = find_next_unjudged_pair_index(
                pair_df,
                judgments or [],
                dataset=str(payload["dataset"]),
                candidate_type=str(direction),
                candidate_key=str(candidate_value),
                start_after_index=None,
                treat_dataset_pair_as_judged=str(direction) == "yes",
            )
            if next_unjudged_idx is not None and str(next_unjudged_idx) in valid_pair_values:
                resolved_pair_value = str(next_unjudged_idx)
            else:
                resolved_pair_value = pair_choices[0][1] if pair_choices else None
        pair_detail = render_pair_detail(bundle, pair_df, resolved_pair_value)
        return (
            payload["dataset"],
            payload["summary_markdown"],
            payload["gallery_items"],
            _pair_table_for_display(pair_df),
            gr.update(choices=pair_choices, value=resolved_pair_value),
            str(resolved_pair_value or ""),
            pair_df,
            pair_detail["left_image"],
            pair_detail["right_image"],
            pair_detail["pair_markdown"],
            payload["default_note"] if note_value is None else str(note_value),
            action_status,
        )

    def _task_outputs(bundle, direction: str, judgments, preferred_candidate: str | None = None):
        if bundle is None:
            empty_task_df = _empty_task_state_df()
            return (
                _task_board_markdown(empty_task_df),
                _format_task_table(empty_task_df),
                empty_task_df,
                "",
                "未选中任务。",
            )
        task_df = build_candidate_task_table(bundle, direction, judgments or [])
        candidate_value = str(preferred_candidate) if preferred_candidate and not task_df.empty and task_df["candidate_key"].astype(str).eq(str(preferred_candidate)).any() else suggest_candidate_value(task_df)
        return (
            _task_board_markdown(task_df),
            _format_task_table(task_df),
            task_df,
            candidate_value or "",
            _selected_candidate_text(task_df, candidate_value),
        )

    def _full_candidate_panel(bundle, direction: str, judgments, preferred_candidate: str | None = None):
        task_summary, task_display_df, task_state_df, candidate_value, selected_candidate_text = _task_outputs(
            bundle,
            direction,
            judgments,
            preferred_candidate=preferred_candidate,
        )
        candidate_outputs = _candidate_outputs(bundle, direction, candidate_value, judgments=judgments)
        return (
            task_summary,
            task_display_df,
            task_state_df,
            candidate_value,
            selected_candidate_text,
            *candidate_outputs,
        )

    def load_workspace(base_predictions_path: str, probe_dir: str, direction: str, requested_session_name: str):
        try:
            bundle = load_review_bundle(
                repo_root=repo_root,
                base_predictions_path=base_predictions_path,
                probe_dir=probe_dir or None,
            )
            bundle_cache["token"] = ""
            bundle_cache["candidate_payloads"] = {}
        except Exception as exc:
            empty_task_df = _empty_task_state_df()
            judgment_state, judgment_df, summary_df, preview_json, selected_idx, selected_text = _judgment_outputs(
                [],
                session_name=requested_session_name or DEFAULT_SESSION_NAME,
                selected_index=None,
            )
            return (
                None,
                f"加载失败：`{exc}`",
                requested_session_name or DEFAULT_SESSION_NAME,
                judgment_state,
                judgment_df,
                summary_df,
                preview_json,
                selected_idx,
                selected_text,
                "未恢复任何缓存。",
                _task_board_markdown(empty_task_df),
                _format_task_table(empty_task_df),
                empty_task_df,
                "",
                "未选中任务。",
                *_empty_candidate_outputs(),
            )

        session_name_value, restored_judgments, autosave_path, cache_found = _restore_cached_judgments(requested_session_name)
        judgment_state, judgment_df, summary_df, preview_json, selected_idx, selected_text = _judgment_outputs(
            restored_judgments,
            session_name=session_name_value,
            selected_index=None,
        )
        _remember_last_session(session_name_value)
        status = "\n".join(
            [
                f"- `predictions`: `{bundle.predictions_path}`",
                f"- `probe_dir`: `{bundle.probe_dir}`",
                f"- `rows`: `{len(bundle.pred_df)}`",
                f"- `split_candidates`: `{len(bundle.split_candidate_df)}`",
                f"- `merge_candidates`: `{len(bundle.merge_candidate_df)}`",
                f"- `autosave_path`: `{autosave_path}`",
                f"- `restored_judgments`: `{len(restored_judgments)}`",
                "- 当前模式：人工只判断 pair 是 `yes / no / uncertain`。",
            ]
        )
        queue_status = (
            f"已从缓存恢复 `{len(restored_judgments)}` 条 judgment。"
            if cache_found
            else f"未找到缓存，当前从空白开始；后续会自动缓存到：`{autosave_path}`"
        )
        return (
            bundle,
            status,
            session_name_value,
            judgment_state,
            judgment_df,
            summary_df,
            preview_json,
            selected_idx,
            selected_text,
            queue_status,
            *_full_candidate_panel(bundle, direction, restored_judgments),
        )

    def refresh_candidate_panel(bundle, direction: str, judgments, selected_candidate_value: str | None):
        return _full_candidate_panel(bundle, direction, judgments, preferred_candidate=selected_candidate_value)

    def select_candidate_task(bundle, direction: str, task_df: pd.DataFrame, judgments, evt: gr.SelectData):
        if bundle is None or task_df is None or task_df.empty:
            return (
                "",
                "未选中任务。",
                *_empty_candidate_outputs(),
            )
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        row_index = int(row_index)
        if row_index < 0 or row_index >= len(task_df):
            return (
                "",
                "未选中任务。",
                *_empty_candidate_outputs(),
            )
        candidate_value = str(task_df.iloc[row_index]["candidate_key"])
        return (
            candidate_value,
            _selected_candidate_text(task_df, candidate_value),
            *_candidate_outputs(bundle, direction, candidate_value, judgments=judgments),
        )

    def go_to_next_candidate(bundle, direction: str, task_df: pd.DataFrame, judgments, current_candidate_value: str | None):
        if bundle is None or task_df is None or task_df.empty:
            return (
                "",
                "未选中任务。",
                *_empty_candidate_outputs(),
            )
        task_keys = task_df["candidate_key"].astype(str).tolist()
        unfinished_df = task_df[~task_df["status_code"].astype(str).isin(["completed", "empty"])].copy()
        unfinished_keys = unfinished_df["candidate_key"].astype(str).tolist()
        if current_candidate_value and unfinished_keys:
            if str(current_candidate_value) in unfinished_keys:
                current_pos = unfinished_keys.index(str(current_candidate_value))
                candidate_value = unfinished_keys[(current_pos + 1) % len(unfinished_keys)]
            else:
                candidate_value = unfinished_keys[0]
        elif unfinished_keys:
            candidate_value = unfinished_keys[0]
        else:
            if current_candidate_value and str(current_candidate_value) in task_keys:
                current_pos = task_keys.index(str(current_candidate_value))
                candidate_value = task_keys[(current_pos + 1) % len(task_keys)]
            else:
                candidate_value = task_keys[0]
        return (
            candidate_value,
            _selected_candidate_text(task_df, candidate_value),
            *_candidate_outputs(bundle, direction, candidate_value, judgments=judgments),
        )

    def _resolve_next_review_target(
        bundle,
        direction: str,
        judgments,
        current_candidate_value: str | None,
        current_candidate_dataset: str,
        current_pair_df: pd.DataFrame,
        current_pair_index: str | None,
    ) -> tuple[str, str | None]:
        if bundle is None:
            return "", None

        current_pair_idx = None if current_pair_index in (None, "") else int(current_pair_index)
        if current_candidate_value and current_pair_df is not None and len(current_pair_df) > 0:
            next_pair_idx = find_next_unjudged_pair_index(
                current_pair_df,
                judgments or [],
                dataset=str(current_candidate_dataset),
                candidate_type=str(direction),
                candidate_key=str(current_candidate_value),
                start_after_index=current_pair_idx,
                treat_dataset_pair_as_judged=str(direction) == "yes",
            )
            if next_pair_idx is not None:
                return str(current_candidate_value), str(next_pair_idx)

        task_df = build_candidate_task_table(bundle, direction, judgments or [])
        if task_df.empty:
            return "", None

        unfinished_keys = task_df[
            ~task_df["status_code"].astype(str).isin(["completed", "empty"])
        ]["candidate_key"].astype(str).tolist()
        if not unfinished_keys:
            current_value = str(current_candidate_value) if current_candidate_value else suggest_candidate_value(task_df)
            return current_value or "", None

        if current_candidate_value and str(current_candidate_value) in unfinished_keys:
            start_pos = unfinished_keys.index(str(current_candidate_value))
            candidate_scan = unfinished_keys[start_pos + 1 :] + unfinished_keys[: start_pos + 1]
        else:
            candidate_scan = unfinished_keys

        for candidate_value in candidate_scan:
            payload = render_candidate_view(bundle, direction, candidate_value)
            pair_df = payload["pair_df"]
            next_pair_idx = find_next_unjudged_pair_index(
                pair_df,
                judgments or [],
                dataset=str(payload["dataset"]),
                candidate_type=str(direction),
                candidate_key=str(candidate_value),
                start_after_index=None,
                treat_dataset_pair_as_judged=str(direction) == "yes",
            )
            if next_pair_idx is not None:
                return str(candidate_value), str(next_pair_idx)

        fallback_candidate = suggest_candidate_value(task_df)
        return str(fallback_candidate or ""), None

    def show_pair(bundle, pair_df: pd.DataFrame, pair_index: str):
        if bundle is None:
            return "", None, None, "_No support pair selected._"
        detail = render_pair_detail(bundle, pair_df, pair_index)
        return str(pair_index or ""), detail["left_image"], detail["right_image"], detail["pair_markdown"]

    def show_pair_from_table(bundle, pair_df: pd.DataFrame, evt: gr.SelectData):
        if bundle is None or pair_df is None or pair_df.empty:
            return gr.update(value=None), "", None, None, "_No support pair selected._"
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        row_index = int(row_index)
        detail = render_pair_detail(bundle, pair_df, row_index)
        return gr.update(value=str(row_index)), str(row_index), detail["left_image"], detail["right_image"], detail["pair_markdown"]

    def _task_outputs_for_judgment(bundle, direction: str, judgments, selected_candidate_value: str | None):
        task_summary, task_display_df, task_state_df, candidate_value, selected_candidate_text = _task_outputs(
            bundle,
            direction,
            judgments,
            preferred_candidate=selected_candidate_value,
        )
        return task_summary, task_display_df, task_state_df, candidate_value, selected_candidate_text

    def _refresh_after_judgment_queue_change(
        bundle,
        direction: str,
        judgments,
        *,
        session_name: str,
        selected_candidate_value: str | None,
        queue_status: str,
        pair_action_status: str | None = None,
    ):
        judgment_state, judgment_df, summary_df, preview_json, selected_idx, selected_text = _judgment_outputs(
            judgments,
            session_name=session_name or DEFAULT_SESSION_NAME,
            selected_index=None,
        )
        task_summary, task_display_df, task_state_df, next_candidate_value, selected_candidate_text = _task_outputs_for_judgment(
            bundle,
            direction,
            judgments,
            selected_candidate_value=selected_candidate_value,
        )
        candidate_outputs = _candidate_outputs(
            bundle,
            direction,
            next_candidate_value,
            judgments=judgments,
            action_status=pair_action_status or queue_status,
        )
        return (
            judgment_state,
            judgment_df,
            summary_df,
            preview_json,
            selected_idx,
            selected_text,
            queue_status,
            task_summary,
            task_display_df,
            task_state_df,
            next_candidate_value,
            selected_candidate_text,
            *candidate_outputs,
        )

    def record_pair_judgment(
        bundle,
        judgments,
        session_name: str,
        direction: str,
        candidate_value: str,
        candidate_dataset: str,
        pair_df: pd.DataFrame,
        pair_index: str,
        note: str,
        label: str,
    ):
        try:
            if bundle is None or not candidate_value:
                raise ValueError("No candidate selected")
            if pair_df is None or len(pair_df) == 0:
                raise ValueError("No support pair available")
            pair_idx = 0 if pair_index in (None, "") else int(pair_index)
            pair_idx = max(0, min(pair_idx, len(pair_df) - 1))
            pair_row = pair_df.iloc[pair_idx]
            next_judgments = upsert_pair_judgment(
                judgments or [],
                dataset=str(candidate_dataset),
                candidate_type=str(direction),
                candidate_key=str(candidate_value),
                pair_row=pair_row,
                label=label,
                note=note,
            )
            judgment_state, judgment_df, summary_df, preview_json, selected_idx, selected_text = _judgment_outputs(
                next_judgments,
                session_name=session_name or DEFAULT_SESSION_NAME,
                selected_index=None,
            )
            autosave_path = _autosave_judgments(session_name or DEFAULT_SESSION_NAME, next_judgments)
            task_summary, task_display_df, task_state_df, selected_candidate_value, selected_candidate_text = _task_outputs_for_judgment(
                bundle,
                direction,
                next_judgments,
                selected_candidate_value=candidate_value,
            )
            next_candidate_value, next_pair_value = _resolve_next_review_target(
                bundle,
                direction,
                next_judgments,
                current_candidate_value=candidate_value,
                current_candidate_dataset=candidate_dataset,
                current_pair_df=pair_df,
                current_pair_index=pair_index,
            )
            if next_candidate_value:
                selected_candidate_value = next_candidate_value
                selected_candidate_text = _selected_candidate_text(task_state_df, next_candidate_value)
            if next_candidate_value == str(candidate_value) and next_pair_value is not None:
                status_suffix = "已跳到当前任务下一对。"
            elif next_candidate_value and str(next_candidate_value) != str(candidate_value):
                status_suffix = "当前任务已做完，已跳到下一条未完成任务。"
            else:
                status_suffix = "当前没有更多未完成 pair。"
            status = (
                f"已记录：`{pair_row['image_id']}` vs `{pair_row['neighbor_image_id']}` -> "
                f"`{label}`；{status_suffix} 已自动缓存到：`{autosave_path}`"
            )
            candidate_outputs = _candidate_outputs(
                bundle,
                direction,
                selected_candidate_value,
                judgments=next_judgments,
                pair_value=next_pair_value,
                note_value="",
                action_status=status,
            )
            return (
                judgment_state,
                judgment_df,
                summary_df,
                preview_json,
                selected_idx,
                selected_text,
                task_summary,
                task_display_df,
                task_state_df,
                selected_candidate_value,
                selected_candidate_text,
                *candidate_outputs,
            )
        except Exception as exc:
            judgment_state, judgment_df, summary_df, preview_json, selected_idx, selected_text = _judgment_outputs(
                judgments or [],
                session_name=session_name or DEFAULT_SESSION_NAME,
                selected_index=None,
            )
            task_summary, task_display_df, task_state_df, selected_candidate_value, selected_candidate_text = _task_outputs_for_judgment(
                bundle,
                direction,
                judgments or [],
                selected_candidate_value=candidate_value,
            )
            return (
                judgment_state,
                judgment_df,
                summary_df,
                preview_json,
                selected_idx,
                selected_text,
                task_summary,
                task_display_df,
                task_state_df,
                selected_candidate_value,
                selected_candidate_text,
                *_candidate_outputs(
                    bundle,
                    direction,
                    selected_candidate_value,
                    judgments=judgments or [],
                    pair_value=pair_index,
                    note_value=note,
                    action_status=f"记录失败：`{exc}`",
                ),
            )

    def record_yes(*args):
        return record_pair_judgment(*args, label=PAIR_LABEL_YES)

    def record_no(*args):
        return record_pair_judgment(*args, label=PAIR_LABEL_NO)

    def record_uncertain(*args):
        return record_pair_judgment(*args, label=PAIR_LABEL_UNCERTAIN)

    def select_judgment_row(judgments, session_name: str, evt: gr.SelectData):
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        _, _, _, _, selected_idx, selected_text = _judgment_outputs(
            judgments or [],
            session_name=session_name or DEFAULT_SESSION_NAME,
            selected_index=int(row_index),
        )
        return selected_idx, selected_text

    def remove_selected_judgment(bundle, direction: str, judgments, session_name: str, selected_index, selected_candidate_value: str | None):
        next_judgments = remove_pair_judgment_at(judgments or [], selected_index)
        autosave_path = _autosave_judgments(session_name or DEFAULT_SESSION_NAME, next_judgments)
        if selected_index is None:
            queue_status = f"没有选中任何 judgment；已自动缓存到：`{autosave_path}`"
        else:
            queue_status = f"已删除选中 judgment；已自动缓存到：`{autosave_path}`"
        if selected_index is None:
            return _refresh_after_judgment_queue_change(
                bundle,
                direction,
                judgments or [],
                session_name=session_name or DEFAULT_SESSION_NAME,
                selected_candidate_value=selected_candidate_value,
                queue_status=queue_status,
            )
        return _refresh_after_judgment_queue_change(
            bundle,
            direction,
            next_judgments,
            session_name=session_name or DEFAULT_SESSION_NAME,
            selected_candidate_value=selected_candidate_value,
            queue_status=queue_status,
        )

    def remove_last_judgment(bundle, direction: str, judgments, session_name: str, selected_candidate_value: str | None):
        next_judgments = list((judgments or [])[:-1]) if judgments else []
        autosave_path = _autosave_judgments(session_name or DEFAULT_SESSION_NAME, next_judgments)
        return _refresh_after_judgment_queue_change(
            bundle,
            direction,
            next_judgments,
            session_name=session_name or DEFAULT_SESSION_NAME,
            selected_candidate_value=selected_candidate_value,
            queue_status=f"已删除最后一个 judgment；已自动缓存到：`{autosave_path}`",
        )

    def clear_judgments(bundle, direction: str, session_name: str, selected_candidate_value: str | None):
        next_judgments = clear_pair_judgments()
        autosave_path = _autosave_judgments(session_name or DEFAULT_SESSION_NAME, next_judgments)
        return _refresh_after_judgment_queue_change(
            bundle,
            direction,
            next_judgments,
            session_name=session_name or DEFAULT_SESSION_NAME,
            selected_candidate_value=selected_candidate_value,
            queue_status=f"已清空 judgment 队列；已自动缓存到：`{autosave_path}`",
        )

    def export_judgments(session_name: str, judgments, export_path: str):
        try:
            exported = export_pair_judgments(
                session_name=session_name or DEFAULT_SESSION_NAME,
                judgments=judgments or [],
                output_path=export_path,
            )
            preview_json = build_judgment_preview_json(session_name or DEFAULT_SESSION_NAME, judgments or [])
            return preview_json, f"已导出：`{exported}`"
        except Exception as exc:
            preview_json = build_judgment_preview_json(session_name or DEFAULT_SESSION_NAME, judgments or [])
            return preview_json, f"导出失败：`{exc}`"

    with gr.Blocks(title="AnimalCLEF Pair Review Workbench") as demo:
        bundle_state = gr.State(None)
        candidate_task_df_state = gr.State(_empty_task_state_df())
        selected_candidate_value_state = gr.State("")
        candidate_pair_df_state = gr.State(_empty_pair_table())
        candidate_dataset_state = gr.State("")
        current_pair_value_state = gr.State("")
        pair_judgments_state = gr.State([])
        selected_judgment_index_state = gr.State(None)

        gr.Markdown(
            "# AnimalCLEF Pair Review Workbench\n"
            "- 现在人工阶段只做一件事：判断一对图是不是同一个体。\n"
            "- candidate 不再用下拉，而是改成任务面板，直接看 `待处理 / 进行中 / 已完成`。\n"
            "- 按钮语义：`同一个体=yes`，`不是同一个体=no`，`不确定=uncertain`。\n"
            "- judgment 会自动缓存到磁盘；页面刷新后会自动恢复最近 session。\n"
            "- 后续 `split / merge` 作为第二阶段，再根据这些 pair judgment 生成。"
        )

        with gr.Row():
            base_predictions_input = gr.Textbox(label="Base Predictions / Submission Dir", value=str(args.base_predictions))
            probe_dir_input = gr.Textbox(label="Probe Dir", value=str(args.probe_dir))
        with gr.Row():
            direction_radio = gr.Radio(
                choices=[("Yes Candidate", "yes"), ("Split Candidate", "split"), ("Merge Candidate", "merge")],
                value="yes",
                label="Candidate Type",
            )
            load_button = gr.Button("加载工作区", variant="primary")
        load_status = gr.Markdown()

        with gr.Tab("Pair Review"):
            candidate_task_summary = gr.Markdown("- 当前还没有加载 candidate task。")
            with gr.Row():
                next_candidate_button = gr.Button("下一条未完成任务", variant="secondary")
                selected_candidate_markdown = gr.Markdown("未选中任务。")
            candidate_task_table = gr.Dataframe(label="Candidate Tasks", interactive=False, wrap=True)
            candidate_summary = gr.Markdown()
            candidate_gallery = gr.Gallery(label="Candidate Images", columns=4, height=280, allow_preview=True)
            pair_dropdown = gr.Dropdown(label="Support Pair", choices=[], value=None, interactive=True)
            pair_table = gr.Dataframe(label="Support Pair Table", interactive=False)
            with gr.Row():
                left_image = gr.Image(label="Left Pair Image", type="filepath", height=320)
                right_image = gr.Image(label="Right Pair Image", type="filepath", height=320)
            pair_markdown = gr.Markdown()
            pair_note = gr.Textbox(label="Pair Note", lines=2, value="")
            with gr.Row():
                mark_yes_button = gr.Button("同一个体", variant="primary")
                mark_no_button = gr.Button("不是同一个体", variant="secondary")
                mark_uncertain_button = gr.Button("不确定")
            pair_action_status = gr.Markdown()

        with gr.Tab("Judgments / Export"):
            judgment_summary_df = gr.Dataframe(label="Candidate Judgment Summary", interactive=False)
            pair_judgments_df = gr.Dataframe(label="Pair Judgments Queue", interactive=False)
            selected_judgment_markdown = gr.Markdown("未选中 judgment。")
            with gr.Row():
                remove_selected_button = gr.Button("删除选中 judgment")
                remove_last_button = gr.Button("删除最后一个")
                clear_button = gr.Button("清空 judgment 队列")
            queue_status = gr.Markdown()
            pair_judgments_json = gr.Code(
                label="Pair Judgment JSON Preview",
                language="json",
                value=build_judgment_preview_json(DEFAULT_SESSION_NAME, []),
            )
            with gr.Row():
                session_name_input = gr.Textbox(label="Session Name", value=DEFAULT_SESSION_NAME)
                export_path_input = gr.Textbox(label="Export Path", value=str(DEFAULT_EXPORT_JUDGMENTS))
            export_button = gr.Button("导出 Pair Judgments", variant="primary")
            export_status = gr.Markdown()

        load_button.click(
            fn=load_workspace,
            inputs=[base_predictions_input, probe_dir_input, direction_radio, session_name_input],
            outputs=[
                bundle_state,
                load_status,
                session_name_input,
                pair_judgments_state,
                pair_judgments_df,
                judgment_summary_df,
                pair_judgments_json,
                selected_judgment_index_state,
                selected_judgment_markdown,
                queue_status,
                candidate_task_summary,
                candidate_task_table,
                candidate_task_df_state,
                selected_candidate_value_state,
                selected_candidate_markdown,
                candidate_dataset_state,
                candidate_summary,
                candidate_gallery,
                pair_table,
                pair_dropdown,
                current_pair_value_state,
                candidate_pair_df_state,
                left_image,
                right_image,
                pair_markdown,
                pair_note,
                pair_action_status,
            ],
            show_progress="minimal",
        )
        demo.load(
            fn=load_workspace,
            inputs=[base_predictions_input, probe_dir_input, direction_radio, session_name_input],
            outputs=[
                bundle_state,
                load_status,
                session_name_input,
                pair_judgments_state,
                pair_judgments_df,
                judgment_summary_df,
                pair_judgments_json,
                selected_judgment_index_state,
                selected_judgment_markdown,
                queue_status,
                candidate_task_summary,
                candidate_task_table,
                candidate_task_df_state,
                selected_candidate_value_state,
                selected_candidate_markdown,
                candidate_dataset_state,
                candidate_summary,
                candidate_gallery,
                pair_table,
                pair_dropdown,
                current_pair_value_state,
                candidate_pair_df_state,
                left_image,
                right_image,
                pair_markdown,
                pair_note,
                pair_action_status,
            ],
            show_progress="minimal",
        )
        direction_radio.change(
            fn=refresh_candidate_panel,
            inputs=[bundle_state, direction_radio, pair_judgments_state, selected_candidate_value_state],
            outputs=[
                candidate_task_summary,
                candidate_task_table,
                candidate_task_df_state,
                selected_candidate_value_state,
                selected_candidate_markdown,
                candidate_dataset_state,
                candidate_summary,
                candidate_gallery,
                pair_table,
                pair_dropdown,
                current_pair_value_state,
                candidate_pair_df_state,
                left_image,
                right_image,
                pair_markdown,
                pair_note,
                pair_action_status,
            ],
            show_progress="minimal",
        )
        candidate_task_table.select(
            fn=select_candidate_task,
            inputs=[bundle_state, direction_radio, candidate_task_df_state, pair_judgments_state],
            outputs=[
                selected_candidate_value_state,
                selected_candidate_markdown,
                candidate_dataset_state,
                candidate_summary,
                candidate_gallery,
                pair_table,
                pair_dropdown,
                current_pair_value_state,
                candidate_pair_df_state,
                left_image,
                right_image,
                pair_markdown,
                pair_note,
                pair_action_status,
            ],
            show_progress="minimal",
        )
        next_candidate_button.click(
            fn=go_to_next_candidate,
            inputs=[bundle_state, direction_radio, candidate_task_df_state, pair_judgments_state, selected_candidate_value_state],
            outputs=[
                selected_candidate_value_state,
                selected_candidate_markdown,
                candidate_dataset_state,
                candidate_summary,
                candidate_gallery,
                pair_table,
                pair_dropdown,
                current_pair_value_state,
                candidate_pair_df_state,
                left_image,
                right_image,
                pair_markdown,
                pair_note,
                pair_action_status,
            ],
            show_progress="minimal",
        )
        pair_dropdown.change(
            fn=show_pair,
            inputs=[bundle_state, candidate_pair_df_state, pair_dropdown],
            outputs=[current_pair_value_state, left_image, right_image, pair_markdown],
            show_progress="minimal",
        )
        pair_table.select(
            fn=show_pair_from_table,
            inputs=[bundle_state, candidate_pair_df_state],
            outputs=[pair_dropdown, current_pair_value_state, left_image, right_image, pair_markdown],
            show_progress="minimal",
        )
        mark_yes_button.click(
            fn=record_yes,
            inputs=[
                bundle_state,
                pair_judgments_state,
                session_name_input,
                direction_radio,
                selected_candidate_value_state,
                candidate_dataset_state,
                candidate_pair_df_state,
                current_pair_value_state,
                pair_note,
            ],
            outputs=[
                pair_judgments_state,
                pair_judgments_df,
                judgment_summary_df,
                pair_judgments_json,
                selected_judgment_index_state,
                selected_judgment_markdown,
                candidate_task_summary,
                candidate_task_table,
                candidate_task_df_state,
                selected_candidate_value_state,
                selected_candidate_markdown,
                candidate_dataset_state,
                candidate_summary,
                candidate_gallery,
                pair_table,
                pair_dropdown,
                current_pair_value_state,
                candidate_pair_df_state,
                left_image,
                right_image,
                pair_markdown,
                pair_note,
                pair_action_status,
            ],
            show_progress="minimal",
        )
        mark_no_button.click(
            fn=record_no,
            inputs=[
                bundle_state,
                pair_judgments_state,
                session_name_input,
                direction_radio,
                selected_candidate_value_state,
                candidate_dataset_state,
                candidate_pair_df_state,
                current_pair_value_state,
                pair_note,
            ],
            outputs=[
                pair_judgments_state,
                pair_judgments_df,
                judgment_summary_df,
                pair_judgments_json,
                selected_judgment_index_state,
                selected_judgment_markdown,
                candidate_task_summary,
                candidate_task_table,
                candidate_task_df_state,
                selected_candidate_value_state,
                selected_candidate_markdown,
                candidate_dataset_state,
                candidate_summary,
                candidate_gallery,
                pair_table,
                pair_dropdown,
                current_pair_value_state,
                candidate_pair_df_state,
                left_image,
                right_image,
                pair_markdown,
                pair_note,
                pair_action_status,
            ],
            show_progress="minimal",
        )
        mark_uncertain_button.click(
            fn=record_uncertain,
            inputs=[
                bundle_state,
                pair_judgments_state,
                session_name_input,
                direction_radio,
                selected_candidate_value_state,
                candidate_dataset_state,
                candidate_pair_df_state,
                current_pair_value_state,
                pair_note,
            ],
            outputs=[
                pair_judgments_state,
                pair_judgments_df,
                judgment_summary_df,
                pair_judgments_json,
                selected_judgment_index_state,
                selected_judgment_markdown,
                candidate_task_summary,
                candidate_task_table,
                candidate_task_df_state,
                selected_candidate_value_state,
                selected_candidate_markdown,
                candidate_dataset_state,
                candidate_summary,
                candidate_gallery,
                pair_table,
                pair_dropdown,
                current_pair_value_state,
                candidate_pair_df_state,
                left_image,
                right_image,
                pair_markdown,
                pair_note,
                pair_action_status,
            ],
            show_progress="minimal",
        )
        pair_judgments_df.select(
            fn=select_judgment_row,
            inputs=[pair_judgments_state, session_name_input],
            outputs=[selected_judgment_index_state, selected_judgment_markdown],
            show_progress="minimal",
        )
        remove_selected_button.click(
            fn=remove_selected_judgment,
            inputs=[
                bundle_state,
                direction_radio,
                pair_judgments_state,
                session_name_input,
                selected_judgment_index_state,
                selected_candidate_value_state,
            ],
            outputs=[
                pair_judgments_state,
                pair_judgments_df,
                judgment_summary_df,
                pair_judgments_json,
                selected_judgment_index_state,
                selected_judgment_markdown,
                queue_status,
                candidate_task_summary,
                candidate_task_table,
                candidate_task_df_state,
                selected_candidate_value_state,
                selected_candidate_markdown,
                candidate_dataset_state,
                candidate_summary,
                candidate_gallery,
                pair_table,
                pair_dropdown,
                current_pair_value_state,
                candidate_pair_df_state,
                left_image,
                right_image,
                pair_markdown,
                pair_note,
                pair_action_status,
            ],
            show_progress="minimal",
        )
        remove_last_button.click(
            fn=remove_last_judgment,
            inputs=[
                bundle_state,
                direction_radio,
                pair_judgments_state,
                session_name_input,
                selected_candidate_value_state,
            ],
            outputs=[
                pair_judgments_state,
                pair_judgments_df,
                judgment_summary_df,
                pair_judgments_json,
                selected_judgment_index_state,
                selected_judgment_markdown,
                queue_status,
                candidate_task_summary,
                candidate_task_table,
                candidate_task_df_state,
                selected_candidate_value_state,
                selected_candidate_markdown,
                candidate_dataset_state,
                candidate_summary,
                candidate_gallery,
                pair_table,
                pair_dropdown,
                current_pair_value_state,
                candidate_pair_df_state,
                left_image,
                right_image,
                pair_markdown,
                pair_note,
                pair_action_status,
            ],
            show_progress="minimal",
        )
        clear_button.click(
            fn=clear_judgments,
            inputs=[bundle_state, direction_radio, session_name_input, selected_candidate_value_state],
            outputs=[
                pair_judgments_state,
                pair_judgments_df,
                judgment_summary_df,
                pair_judgments_json,
                selected_judgment_index_state,
                selected_judgment_markdown,
                queue_status,
                candidate_task_summary,
                candidate_task_table,
                candidate_task_df_state,
                selected_candidate_value_state,
                selected_candidate_markdown,
                candidate_dataset_state,
                candidate_summary,
                candidate_gallery,
                pair_table,
                pair_dropdown,
                current_pair_value_state,
                candidate_pair_df_state,
                left_image,
                right_image,
                pair_markdown,
                pair_note,
                pair_action_status,
            ],
            show_progress="minimal",
        )
        export_button.click(
            fn=export_judgments,
            inputs=[session_name_input, pair_judgments_state, export_path_input],
            outputs=[pair_judgments_json, export_status],
            show_progress="minimal",
        )

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(repo_root)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
