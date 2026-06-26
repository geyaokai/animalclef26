#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_BASE_DIR = Path("artifacts/submissions/kaggle_variant_salamander_top10_manual_graph_on_062817_bestpublic_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/submissions/kaggle_variant_salamander_crossview_manual_yes_on_067578_v1")


class DisjointSet:
    def __init__(self, items: list[str]) -> None:
        self.parent = {str(item): str(item) for item in items}

    def find(self, item: str) -> str:
        item = str(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(str(left))
        root_right = self.find(str(right))
        if root_left == root_right:
            return
        if root_right < root_left:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _load_yes_pairs(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_rows = payload.get("rows", payload if isinstance(payload, list) else [])
            frame = pd.DataFrame(raw_rows)
        else:
            frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        if "label" in frame.columns:
            frame = frame[frame["label"].fillna("").astype(str).str.lower().eq("yes")].copy()
        for left_col, right_col in [
            ("image_id", "neighbor_image_id"),
            ("canonical_image_id", "canonical_neighbor_image_id"),
        ]:
            if {left_col, right_col}.issubset(frame.columns):
                for row in frame.itertuples(index=False):
                    rows.append(
                        {
                            "image_id": str(getattr(row, left_col)),
                            "neighbor_image_id": str(getattr(row, right_col)),
                            "source_path": str(path),
                        }
                    )
                break
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["image_id", "neighbor_image_id", "source_path"])
    result["pair_key"] = result.apply(
        lambda row: "|".join(sorted([str(row["image_id"]), str(row["neighbor_image_id"])], key=lambda value: int(value) if value.isdigit() else value)),
        axis=1,
    )
    return result.drop_duplicates("pair_key").reset_index(drop=True)


def _choose_component_cluster(members: list[str], cluster_by_image: dict[str, int]) -> int:
    counts: dict[int, int] = {}
    for image_id in members:
        cluster = int(cluster_by_image[image_id])
        counts[cluster] = counts.get(cluster, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from animalclef_analysis.descriptor_baselines import build_submission

    parser = argparse.ArgumentParser(description="Union additional Salamander manual YES pairs into an existing submission.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--yes-labels", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-submission-path", type=Path, default=Path("sample_submission.csv"))
    parser.add_argument("--rule-name", type=str, default="salamander_crossview_manual_yes_overlay_v1")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    base_dir = _resolve(repo_root, args.base_dir).resolve()
    output_dir = _resolve(repo_root, args.output_dir).resolve()
    sample_submission_path = _resolve(repo_root, args.sample_submission_path).resolve()
    yes_paths = [_resolve(repo_root, path).resolve() for path in args.yes_labels]

    base_predictions_path = base_dir / "tables" / "test_predictions_v1.csv"
    pred_df = pd.read_csv(base_predictions_path, low_memory=False)
    pred_df["image_id"] = pred_df["image_id"].astype(str)
    pred_df["dataset"] = pred_df["dataset"].astype(str)
    pred_df["pred_cluster_id"] = pd.to_numeric(pred_df["pred_cluster_id"], errors="raise").astype(int)
    pred_df["cluster_label"] = pred_df["cluster_label"].astype(str)

    sal_mask = pred_df["dataset"].eq(SALAMANDER_DATASET)
    sal_df = pred_df[sal_mask].copy()
    image_ids = sal_df["image_id"].tolist()
    image_id_set = set(image_ids)
    cluster_by_image = dict(zip(sal_df["image_id"], sal_df["pred_cluster_id"], strict=True))

    yes_df = _load_yes_pairs(yes_paths)
    yes_df = yes_df[yes_df["image_id"].isin(image_id_set) & yes_df["neighbor_image_id"].isin(image_id_set)].copy()
    dsu = DisjointSet(image_ids)
    for row in yes_df.itertuples(index=False):
        dsu.union(str(row.image_id), str(row.neighbor_image_id))

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for image_id in image_ids:
        members_by_root[dsu.find(image_id)].append(image_id)

    final_cluster_by_image = cluster_by_image.copy()
    component_rows = []
    for root, members in members_by_root.items():
        if len(members) < 2:
            continue
        base_clusters = sorted({int(cluster_by_image[image_id]) for image_id in members})
        if len(base_clusters) < 2:
            continue
        target_cluster = _choose_component_cluster(members, cluster_by_image)
        for image_id in members:
            final_cluster_by_image[image_id] = int(target_cluster)
        component_rows.append(
            {
                "component_root": root,
                "component_size": len(members),
                "target_cluster": int(target_cluster),
                "source_clusters": "|".join(str(value) for value in base_clusters),
                "members": "|".join(sorted(members, key=lambda value: int(value) if value.isdigit() else value)),
            }
        )

    result_df = pred_df.copy()
    changed_rows = []
    for idx, row in result_df[sal_mask].iterrows():
        image_id = str(row["image_id"])
        old_cluster = int(row["pred_cluster_id"])
        new_cluster = int(final_cluster_by_image[image_id])
        if old_cluster == new_cluster:
            continue
        result_df.at[idx, "pred_cluster_id"] = new_cluster
        result_df.at[idx, "cluster_label"] = f"cluster_{SALAMANDER_DATASET}_{new_cluster}"
        result_df.at[idx, "manual_overlay_enabled"] = True
        result_df.at[idx, "manual_overlay_rule"] = str(args.rule_name)
        changed_rows.append({"image_id": image_id, "old_cluster": old_cluster, "new_cluster": new_cluster})

    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)
    yes_df.to_csv(tables_dir / "additional_manual_yes_pairs_v1.csv", index=False)
    pd.DataFrame(component_rows).to_csv(tables_dir / "additional_manual_yes_components_v1.csv", index=False)
    pd.DataFrame(changed_rows).to_csv(tables_dir / "additional_manual_yes_changed_images_v1.csv", index=False)
    build_submission(result_df, sample_submission_path=sample_submission_path, output_path=output_dir / "submission.csv")

    def cluster_summary(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for dataset, group in frame.groupby("dataset", sort=True):
            sizes = group.groupby("pred_cluster_id").size()
            rows.append(
                {
                    "dataset": dataset,
                    "rows": int(len(group)),
                    "clusters": int(sizes.size),
                    "singletons": int((sizes == 1).sum()),
                    "max_cluster_size": int(sizes.max()),
                }
            )
        return pd.DataFrame(rows)

    summary = {
        "base_dir": str(base_dir),
        "base_predictions_path": str(base_predictions_path),
        "yes_label_paths": [str(path) for path in yes_paths],
        "output_dir": str(output_dir),
        "additional_yes_pairs": int(len(yes_df)),
        "merged_components": int(len(component_rows)),
        "changed_images": int(len(changed_rows)),
        "submission_path": str(output_dir / "submission.csv"),
    }
    cluster_summary(pred_df).to_csv(tables_dir / "base_cluster_summary_v1.csv", index=False)
    cluster_summary(result_df).to_csv(tables_dir / "final_cluster_summary_v1.csv", index=False)
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Salamander Additional Manual YES Overlay",
        "",
        f"- Additional yes pairs: `{len(yes_df)}`",
        f"- Merged components: `{len(component_rows)}`",
        f"- Changed images: `{len(changed_rows)}`",
        f"- Submission: `{output_dir / 'submission.csv'}`",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
