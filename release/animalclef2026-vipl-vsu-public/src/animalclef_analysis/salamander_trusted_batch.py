from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .manual_review_workbench import PAIR_LABEL_NO, PAIR_LABEL_YES, load_pair_judgments
from .texas_trusted_batch import dataframe_to_markdown_table


SALAMANDER_DATASET = "SalamanderID2025"


@dataclass(frozen=True)
class SalamanderTrustedBatchArtifacts:
    trusted_membership_path: Path
    clean_trusted_membership_path: Path
    trusted_pairs_path: Path
    clean_trusted_pairs_path: Path
    cannot_link_pairs_path: Path
    trusted_components_path: Path
    clean_trusted_components_path: Path
    conflict_pairs_path: Path
    summary_path: Path
    review_html_path: Path


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item in self.parent:
            return
        self.parent[item] = item
        self.rank[item] = 0

    def find(self, item: str) -> str:
        self.add(str(item))
        parent = self.parent[str(item)]
        if parent != str(item):
            self.parent[str(item)] = self.find(parent)
        return self.parent[str(item)]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(str(left))
        right_root = self.find(str(right))
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def components(self) -> list[list[str]]:
        groups: dict[str, list[str]] = {}
        for item in sorted(self.parent):
            groups.setdefault(self.find(item), []).append(item)
        return sorted([sorted(values) for values in groups.values()], key=lambda values: (len(values), values), reverse=True)


def _canonical_pair(left: object, right: object) -> tuple[str, str, str]:
    left_text = str(left)
    right_text = str(right)
    ordered = sorted([left_text, right_text])
    return ordered[0], ordered[1], f"{ordered[0]}|{ordered[1]}"


def _normalize_text_list(values: list[object]) -> str:
    items = sorted({str(value) for value in values if str(value).strip() and str(value).lower() != "nan"})
    return "|".join(items)


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _resolve_repo_path(repo_root: Path, path_value: object) -> Path | None:
    value = str(path_value or "").strip()
    if not value:
        return None
    path = Path(value)
    absolute = path if path.is_absolute() else repo_root / path
    absolute = absolute.resolve()
    return absolute if absolute.exists() else None


def _html_escape(value: object) -> str:
    import html

    return html.escape(str(value))


def _dataframe_to_html_table(frame: pd.DataFrame, *, max_rows: int = 80) -> str:
    if frame.empty:
        return "<p class=\"muted\">empty</p>"
    preview = frame.head(int(max_rows)).copy()
    headers = "".join(f"<th>{_html_escape(column)}</th>" for column in preview.columns)
    rows = []
    for _, row in preview.iterrows():
        rows.append("<tr>" + "".join(f"<td>{_html_escape(row[column])}</td>" for column in preview.columns) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _judgments_to_frame(pair_judgments_path: Path, *, dataset: str = SALAMANDER_DATASET) -> pd.DataFrame:
    _session_name, judgments = load_pair_judgments(pair_judgments_path)
    frame = pd.DataFrame(judgments)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "candidate_type",
                "candidate_key",
                "pair_key",
                "image_id",
                "neighbor_image_id",
                "label",
                "note",
                "base_cluster_left",
                "base_cluster_right",
                "xgb_same_identity_prob",
                "ambiguity_score",
            ]
        )
    frame["dataset"] = frame["dataset"].fillna("").astype(str)
    frame = frame[frame["dataset"].eq(str(dataset))].copy().reset_index(drop=True)
    if frame.empty:
        return frame
    for column in ["image_id", "neighbor_image_id", "candidate_type", "candidate_key", "label"]:
        frame[column] = frame[column].fillna("").astype(str)
    if "note" not in frame.columns:
        frame["note"] = ""
    frame["note"] = frame["note"].fillna("").astype(str)
    frame["label"] = frame["label"].str.strip().str.lower()
    ordered = frame.apply(lambda row: _canonical_pair(row["image_id"], row["neighbor_image_id"]), axis=1, result_type="expand")
    frame["image_id"] = ordered[0]
    frame["neighbor_image_id"] = ordered[1]
    frame["pair_key"] = ordered[2]
    for column in ["base_cluster_left", "base_cluster_right"]:
        if column not in frame.columns:
            frame[column] = -1
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(-1).astype(int)
    for column in ["xgb_same_identity_prob", "ambiguity_score"]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def _aggregate_label_pairs(frame: pd.DataFrame, *, label: str, provenance: str) -> pd.DataFrame:
    subset = frame[frame["label"].astype(str).eq(str(label))].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "pair_key",
                "image_id",
                "neighbor_image_id",
                "provenance",
                "candidate_types",
                "candidate_keys",
                "base_cluster_left_values",
                "base_cluster_right_values",
                "notes",
                "manual_pair_count",
                "max_xgb_same_identity_prob",
                "max_ambiguity_score",
            ]
        )
    rows: list[dict[str, Any]] = []
    for pair_key, group in subset.groupby("pair_key", sort=True):
        rows.append(
            {
                "pair_key": str(pair_key),
                "image_id": str(group["image_id"].iloc[0]),
                "neighbor_image_id": str(group["neighbor_image_id"].iloc[0]),
                "provenance": str(provenance),
                "candidate_types": _normalize_text_list(group["candidate_type"].tolist()),
                "candidate_keys": _normalize_text_list(group["candidate_key"].tolist()),
                "base_cluster_left_values": _normalize_text_list(group["base_cluster_left"].tolist()),
                "base_cluster_right_values": _normalize_text_list(group["base_cluster_right"].tolist()),
                "notes": _normalize_text_list(group["note"].tolist()),
                "manual_pair_count": int(len(group)),
                "max_xgb_same_identity_prob": round(float(group["xgb_same_identity_prob"].max()), 6),
                "max_ambiguity_score": round(float(group["ambiguity_score"].max()), 6),
            }
        )
    return pd.DataFrame(rows).sort_values(["image_id", "neighbor_image_id"]).reset_index(drop=True)


def _load_salamander_metadata(metadata_path: Path, *, dataset: str = SALAMANDER_DATASET) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path)
    frame["dataset"] = frame["dataset"].astype(str)
    frame["image_id"] = frame["image_id"].astype(str)
    subset = frame[frame["dataset"].eq(str(dataset))].copy().reset_index(drop=True)
    if "original_rgb_path_v1" not in subset.columns:
        subset["original_rgb_path_v1"] = subset.get("path", "")
    return subset


def _build_trusted_membership(
    *,
    trusted_pairs_df: pd.DataFrame,
    cannot_link_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trusted_pairs_df.empty:
        empty_membership = pd.DataFrame(
            columns=[
                "component_id",
                "image_id",
                "component_size",
                "path",
                "original_rgb_path_v1",
                "split",
                "dataset",
                "has_manual_yes_pair",
                "manual_yes_degree",
                "manual_no_degree",
                "candidate_types",
                "candidate_keys",
                "conflict_no_pair_count",
            ]
        )
        return empty_membership, pd.DataFrame(), pd.DataFrame()

    union_find = _UnionFind()
    for row in trusted_pairs_df.itertuples(index=False):
        union_find.union(str(row.image_id), str(row.neighbor_image_id))

    component_by_image: dict[str, str] = {}
    for component_index, members in enumerate(union_find.components(), start=1):
        component_id = f"salamander_trusted_comp_{component_index:03d}"
        for image_id in members:
            component_by_image[str(image_id)] = component_id

    metadata_lookup = metadata_df.drop_duplicates(subset=["image_id"]).set_index("image_id", drop=False)
    yes_degree: dict[str, int] = {}
    for row in trusted_pairs_df.itertuples(index=False):
        yes_degree[str(row.image_id)] = yes_degree.get(str(row.image_id), 0) + 1
        yes_degree[str(row.neighbor_image_id)] = yes_degree.get(str(row.neighbor_image_id), 0) + 1
    no_degree: dict[str, int] = {}
    conflict_rows: list[dict[str, Any]] = []
    for row in cannot_link_df.itertuples(index=False):
        left = str(row.image_id)
        right = str(row.neighbor_image_id)
        no_degree[left] = no_degree.get(left, 0) + 1
        no_degree[right] = no_degree.get(right, 0) + 1
        left_component = component_by_image.get(left, "")
        right_component = component_by_image.get(right, "")
        if left_component and left_component == right_component:
            conflict_rows.append(
                {
                    "pair_key": str(row.pair_key),
                    "image_id": left,
                    "neighbor_image_id": right,
                    "component_id": left_component,
                    "candidate_types": str(row.candidate_types),
                    "candidate_keys": str(row.candidate_keys),
                    "manual_pair_count": int(row.manual_pair_count),
                    "max_xgb_same_identity_prob": float(row.max_xgb_same_identity_prob),
                    "max_ambiguity_score": float(row.max_ambiguity_score),
                }
            )

    pair_context = {}
    for row in trusted_pairs_df.itertuples(index=False):
        for image_id in [str(row.image_id), str(row.neighbor_image_id)]:
            state = pair_context.setdefault(image_id, {"candidate_types": [], "candidate_keys": []})
            state["candidate_types"].extend(str(row.candidate_types).split("|"))
            state["candidate_keys"].extend(str(row.candidate_keys).split("|"))

    rows: list[dict[str, Any]] = []
    component_sizes = pd.Series(component_by_image).value_counts().to_dict()
    conflict_df = pd.DataFrame(conflict_rows)
    conflict_count_by_component = (
        conflict_df.groupby("component_id").size().to_dict() if not conflict_df.empty else {}
    )
    for image_id, component_id in sorted(component_by_image.items(), key=lambda item: (item[1], item[0])):
        meta = metadata_lookup.loc[image_id] if image_id in metadata_lookup.index else pd.Series(dtype=object)
        context = pair_context.get(image_id, {"candidate_types": [], "candidate_keys": []})
        rows.append(
            {
                "component_id": component_id,
                "image_id": image_id,
                "component_size": int(component_sizes.get(component_id, 1)),
                "path": str(meta.get("path", "")),
                "original_rgb_path_v1": str(meta.get("original_rgb_path_v1", meta.get("path", ""))),
                "split": str(meta.get("split", "")),
                "dataset": str(meta.get("dataset", SALAMANDER_DATASET)),
                "has_manual_yes_pair": True,
                "manual_yes_degree": int(yes_degree.get(image_id, 0)),
                "manual_no_degree": int(no_degree.get(image_id, 0)),
                "candidate_types": _normalize_text_list(context["candidate_types"]),
                "candidate_keys": _normalize_text_list(context["candidate_keys"]),
                "conflict_no_pair_count": int(conflict_count_by_component.get(component_id, 0)),
            }
        )
    membership_df = pd.DataFrame(rows).sort_values(["component_id", "image_id"]).reset_index(drop=True)

    component_rows: list[dict[str, Any]] = []
    for component_id, group in membership_df.groupby("component_id", sort=True):
        component_rows.append(
            {
                "component_id": str(component_id),
                "component_size": int(len(group)),
                "image_ids": "|".join(group["image_id"].astype(str).tolist()),
                "candidate_types": _normalize_text_list(group["candidate_types"].tolist()),
                "candidate_keys": _normalize_text_list(group["candidate_keys"].tolist()),
                "manual_yes_degree_sum": int(group["manual_yes_degree"].sum()),
                "manual_no_degree_sum": int(group["manual_no_degree"].sum()),
                "conflict_no_pair_count": int(group["conflict_no_pair_count"].max()),
            }
        )
    component_df = pd.DataFrame(component_rows).sort_values(["component_size", "component_id"], ascending=[False, True]).reset_index(drop=True)
    return membership_df, component_df, conflict_df


def _build_clean_trusted_tables(
    *,
    membership_df: pd.DataFrame,
    component_df: pd.DataFrame,
    trusted_pairs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if membership_df.empty:
        return membership_df.copy(), component_df.copy(), trusted_pairs_df.copy()

    clean_component_ids = set(
        membership_df[membership_df["conflict_no_pair_count"].astype(int).eq(0)]["component_id"].astype(str)
    )
    clean_membership_df = membership_df[membership_df["component_id"].astype(str).isin(clean_component_ids)].copy()
    clean_components_df = component_df[component_df["component_id"].astype(str).isin(clean_component_ids)].copy()

    image_to_clean_component = dict(
        zip(clean_membership_df["image_id"].astype(str), clean_membership_df["component_id"].astype(str), strict=False)
    )
    if trusted_pairs_df.empty:
        clean_pairs_df = trusted_pairs_df.copy()
    else:
        clean_pairs_df = trusted_pairs_df[
            trusted_pairs_df["image_id"].astype(str).map(image_to_clean_component).fillna("").ne("")
            & trusted_pairs_df["neighbor_image_id"].astype(str).map(image_to_clean_component).fillna("").ne("")
            & trusted_pairs_df["image_id"].astype(str).map(image_to_clean_component).eq(
                trusted_pairs_df["neighbor_image_id"].astype(str).map(image_to_clean_component)
            )
        ].copy()

    return (
        clean_membership_df.sort_values(["component_id", "image_id"]).reset_index(drop=True),
        clean_components_df.sort_values(["component_size", "component_id"], ascending=[False, True]).reset_index(drop=True),
        clean_pairs_df.sort_values(["image_id", "neighbor_image_id"]).reset_index(drop=True),
    )


def _write_trusted_review_html(
    *,
    repo_root: Path,
    output_dir: Path,
    membership_df: pd.DataFrame,
    clean_components_df: pd.DataFrame,
    conflict_df: pd.DataFrame,
) -> Path:
    html_dir = output_dir / "review_html"
    html_dir.mkdir(parents=True, exist_ok=True)
    staged_dir = html_dir / "staged_assets"

    cards: list[str] = []
    if not membership_df.empty:
        sorted_membership = membership_df.sort_values(
            ["conflict_no_pair_count", "component_size", "component_id", "image_id"],
            ascending=[False, False, True, True],
        )
        for component_id, group in sorted_membership.groupby("component_id", sort=False):
            conflict_count = int(group["conflict_no_pair_count"].astype(int).max())
            status = "conflicted" if conflict_count > 0 else "clean"
            image_tiles: list[str] = []
            for row in group.itertuples(index=False):
                image_id = str(row.image_id)
                src = _resolve_repo_path(repo_root, getattr(row, "original_rgb_path_v1", ""))
                if src is None:
                    src = _resolve_repo_path(repo_root, getattr(row, "path", ""))
                if src is None:
                    image_tiles.append(
                        f"""
                        <div class="tile missing">
                          <div class="missing-box">missing</div>
                          <div class="meta">{_html_escape(image_id)}</div>
                        </div>
                        """
                    )
                    continue
                suffix = src.suffix.lower() or ".jpg"
                dst = staged_dir / str(component_id) / f"{image_id}{suffix}"
                _copy_or_link(src, dst)
                rel = os.path.relpath(str(dst), str(html_dir))
                image_tiles.append(
                    f"""
                    <div class="tile">
                      <a href="{_html_escape(rel)}" target="_blank"><img src="{_html_escape(rel)}" loading="lazy" /></a>
                      <div class="meta">id={_html_escape(image_id)}</div>
                      <div class="meta">yes={int(row.manual_yes_degree)} no={int(row.manual_no_degree)}</div>
                    </div>
                    """
                )

            component_conflicts = conflict_df[conflict_df["component_id"].astype(str).eq(str(component_id))].copy() if not conflict_df.empty else pd.DataFrame()
            cards.append(
                f"""
                <section class="component-card {status}">
                  <div class="component-head">
                    <div>
                      <h2>{_html_escape(component_id)}</h2>
                      <p>size={int(group["component_size"].iloc[0])} | status={status} | internal_manual_no_conflicts={conflict_count}</p>
                    </div>
                    <span class="badge {status}">{status}</span>
                  </div>
                  <div class="image-grid">{''.join(image_tiles)}</div>
                  <details {'open' if conflict_count > 0 else ''}>
                    <summary>conflict pairs</summary>
                    {_dataframe_to_html_table(component_conflicts, max_rows=40)}
                  </details>
                </section>
                """
            )

    clean_component_ids = set(clean_components_df["component_id"].astype(str)) if not clean_components_df.empty else set()
    summary_rows = [
        {"metric": "trusted_components_total", "value": int(membership_df["component_id"].nunique()) if not membership_df.empty else 0},
        {"metric": "trusted_members_total", "value": int(len(membership_df))},
        {"metric": "clean_components", "value": int(len(clean_component_ids))},
        {"metric": "clean_members", "value": int(membership_df["component_id"].astype(str).isin(clean_component_ids).sum()) if not membership_df.empty else 0},
        {"metric": "conflict_components", "value": int(membership_df["component_id"].nunique() - len(clean_component_ids)) if not membership_df.empty else 0},
        {"metric": "conflict_pairs", "value": int(len(conflict_df))},
    ]
    summary_df = pd.DataFrame(summary_rows)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Salamander Trusted Batch Review</title>
  <style>
    :root {{
      --bg: #f7f3ea;
      --ink: #211d18;
      --muted: #746d63;
      --clean: #1d7a46;
      --warn: #a53f2b;
      --card: #fffaf0;
      --line: #d9cdbb;
    }}
    body {{ margin: 0; padding: 28px; font-family: Georgia, 'Times New Roman', serif; background: var(--bg); color: var(--ink); }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    h2 {{ margin: 0; font-size: 20px; }}
    p {{ margin: 6px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
    th, td {{ border: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #efe3d0; }}
    .muted {{ color: var(--muted); }}
    .component-card {{ margin: 20px 0; padding: 16px; border: 1px solid var(--line); border-radius: 16px; background: var(--card); box-shadow: 0 8px 24px rgba(52, 43, 31, 0.08); }}
    .component-card.conflicted {{ border-color: rgba(165, 63, 43, 0.65); }}
    .component-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }}
    .badge {{ display: inline-block; padding: 5px 10px; border-radius: 999px; color: white; font-size: 13px; }}
    .badge.clean {{ background: var(--clean); }}
    .badge.conflicted {{ background: var(--warn); }}
    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; margin-top: 14px; }}
    .tile {{ background: white; border: 1px solid var(--line); border-radius: 12px; padding: 8px; }}
    .tile img {{ width: 100%; height: 130px; object-fit: contain; background: #15130f; border-radius: 8px; }}
    .missing-box {{ height: 130px; display: grid; place-items: center; background: #ddd3c3; border-radius: 8px; color: var(--muted); }}
    .meta {{ color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    details {{ margin-top: 12px; }}
    summary {{ cursor: pointer; color: var(--muted); }}
  </style>
</head>
<body>
  <h1>Salamander Trusted Batch Review</h1>
  <p class="muted">Clean components are safe positive seeds by default. Conflicted components contain at least one internal manual no pair and should be excluded or resolved before training.</p>
  {_dataframe_to_html_table(summary_df)}
  {''.join(cards)}
</body>
</html>
"""
    index_path = html_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def compile_salamander_trusted_batch(
    *,
    repo_root: Path,
    pair_judgments_path: Path,
    output_dir: Path,
    metadata_path: Path,
    dataset: str = SALAMANDER_DATASET,
) -> SalamanderTrustedBatchArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    judgment_df = _judgments_to_frame(pair_judgments_path, dataset=dataset)
    trusted_pairs_df = _aggregate_label_pairs(judgment_df, label=PAIR_LABEL_YES, provenance="manual_yes_pair")
    cannot_link_df = _aggregate_label_pairs(judgment_df, label=PAIR_LABEL_NO, provenance="manual_no_pair")
    metadata_df = _load_salamander_metadata(metadata_path, dataset=dataset)
    membership_df, component_df, conflict_df = _build_trusted_membership(
        trusted_pairs_df=trusted_pairs_df,
        cannot_link_df=cannot_link_df,
        metadata_df=metadata_df,
    )
    clean_membership_df, clean_components_df, clean_pairs_df = _build_clean_trusted_tables(
        membership_df=membership_df,
        component_df=component_df,
        trusted_pairs_df=trusted_pairs_df,
    )

    trusted_membership_path = tables_dir / "trusted_membership_v1.csv"
    clean_trusted_membership_path = tables_dir / "trusted_membership_clean_v1.csv"
    trusted_pairs_path = tables_dir / "trusted_pairs_v1.csv"
    clean_trusted_pairs_path = tables_dir / "trusted_pairs_clean_v1.csv"
    cannot_link_pairs_path = tables_dir / "cannot_link_pairs_v1.csv"
    trusted_components_path = tables_dir / "trusted_components_v1.csv"
    clean_trusted_components_path = tables_dir / "trusted_components_clean_v1.csv"
    conflict_pairs_path = tables_dir / "conflict_pairs_v1.csv"
    trusted_pairs_df.to_csv(trusted_pairs_path, index=False)
    clean_pairs_df.to_csv(clean_trusted_pairs_path, index=False)
    cannot_link_df.to_csv(cannot_link_pairs_path, index=False)
    membership_df.to_csv(trusted_membership_path, index=False)
    clean_membership_df.to_csv(clean_trusted_membership_path, index=False)
    component_df.to_csv(trusted_components_path, index=False)
    clean_components_df.to_csv(clean_trusted_components_path, index=False)
    conflict_df.to_csv(conflict_pairs_path, index=False)
    review_html_path = _write_trusted_review_html(
        repo_root=repo_root,
        output_dir=output_dir,
        membership_df=membership_df,
        clean_components_df=clean_components_df,
        conflict_df=conflict_df,
    )

    label_counts = (
        judgment_df.groupby(["candidate_type", "label"]).size().reset_index(name="count")
        if not judgment_df.empty
        else pd.DataFrame(columns=["candidate_type", "label", "count"])
    )
    summary_payload = {
        "dataset": str(dataset),
        "pair_judgments_path": str(pair_judgments_path),
        "judgments": int(len(judgment_df)),
        "trusted_pairs": int(len(trusted_pairs_df)),
        "cannot_link_pairs": int(len(cannot_link_df)),
        "trusted_components": int(len(component_df)),
        "trusted_members": int(len(membership_df)),
        "conflict_pairs": int(len(conflict_df)),
        "clean_trusted_pairs": int(len(clean_pairs_df)),
        "clean_trusted_components": int(len(clean_components_df)),
        "clean_trusted_members": int(len(clean_membership_df)),
        "trusted_membership_path": str(trusted_membership_path),
        "clean_trusted_membership_path": str(clean_trusted_membership_path),
        "trusted_pairs_path": str(trusted_pairs_path),
        "clean_trusted_pairs_path": str(clean_trusted_pairs_path),
        "cannot_link_pairs_path": str(cannot_link_pairs_path),
        "trusted_components_path": str(trusted_components_path),
        "clean_trusted_components_path": str(clean_trusted_components_path),
        "conflict_pairs_path": str(conflict_pairs_path),
        "review_html_path": str(review_html_path),
    }
    summary_json_path = reports_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Salamander Trusted Batch v1",
        "",
        f"- dataset: `{dataset}`",
        f"- pair_judgments_path: `{pair_judgments_path}`",
        f"- judgments: `{len(judgment_df)}`",
        f"- trusted yes pairs: `{len(trusted_pairs_df)}`",
        f"- cannot-link no pairs: `{len(cannot_link_df)}`",
        f"- trusted components: `{len(component_df)}`",
        f"- trusted members: `{len(membership_df)}`",
        f"- conflict pairs inside trusted components: `{len(conflict_df)}`",
        f"- clean trusted pairs: `{len(clean_pairs_df)}`",
        f"- clean trusted components: `{len(clean_components_df)}`",
        f"- clean trusted members: `{len(clean_membership_df)}`",
        f"- review HTML: `{review_html_path}`",
        "",
        "## Label Counts",
        "",
        dataframe_to_markdown_table(label_counts),
        "",
        "## Trusted Components Preview",
        "",
        dataframe_to_markdown_table(component_df.head(20)) if not component_df.empty else "_No trusted components._",
        "",
        "## Clean Trusted Components Preview",
        "",
        dataframe_to_markdown_table(clean_components_df.head(20)) if not clean_components_df.empty else "_No clean trusted components._",
        "",
        "## Conflict Preview",
        "",
        dataframe_to_markdown_table(conflict_df.head(20)) if not conflict_df.empty else "_No internal yes/no conflicts._",
        "",
    ]
    summary_path = reports_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    return SalamanderTrustedBatchArtifacts(
        trusted_membership_path=trusted_membership_path,
        clean_trusted_membership_path=clean_trusted_membership_path,
        trusted_pairs_path=trusted_pairs_path,
        clean_trusted_pairs_path=clean_trusted_pairs_path,
        cannot_link_pairs_path=cannot_link_pairs_path,
        trusted_components_path=trusted_components_path,
        clean_trusted_components_path=clean_trusted_components_path,
        conflict_pairs_path=conflict_pairs_path,
        summary_path=summary_path,
        review_html_path=review_html_path,
    )
