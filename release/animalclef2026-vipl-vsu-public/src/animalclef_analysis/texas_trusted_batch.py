from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TEXAS_DATASET = "TexasHornedLizards"
DEFAULT_APPROVED_CLASS_INDICES = [2, 5, 8, 10, 11, 13, 14, 15, 17, 21, 22, 24, 26, 29, 30]
DEFAULT_CLASS_EXCLUSIONS = {
    22: {"15466"},
    24: {"15371"},
}


@dataclass(frozen=True)
class TrustedBatchArtifacts:
    trusted_membership_path: Path
    trusted_pairs_path: Path
    trusted_components_path: Path
    summary_path: Path


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item in self.parent:
            return
        self.parent[item] = item
        self.rank[item] = 0

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank_left = self.rank[root_left]
        rank_right = self.rank[root_right]
        if rank_left < rank_right:
            self.parent[root_left] = root_right
        elif rank_left > rank_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def _canonical_pair(image_id: object, neighbor_image_id: object) -> tuple[str, str, str]:
    left = str(image_id)
    right = str(neighbor_image_id)
    ordered = sorted([left, right])
    return ordered[0], ordered[1], f"{ordered[0]}|{ordered[1]}"


def _normalize_text_list(values: list[object]) -> str:
    items = sorted({str(value) for value in values if str(value).strip() and str(value).lower() != "nan"})
    return "|".join(items)


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(row[column]) else str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])


def load_texas_manual_yes_pairs(manual_pairs_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(manual_pairs_path)
    frame["dataset"] = frame["dataset"].astype(str)
    frame["label"] = frame["label"].astype(str)
    subset = frame[(frame["dataset"] == TEXAS_DATASET) & (frame["label"] == "yes")].copy().reset_index(drop=True)
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "neighbor_image_id",
                "pair_key",
                "provenance",
                "manual_candidate_type",
                "manual_candidate_key",
                "manual_note",
                "xgb_same_identity_prob",
                "ambiguity_score",
            ]
        )
    subset["image_id"] = subset["image_id"].astype(str)
    subset["neighbor_image_id"] = subset["neighbor_image_id"].astype(str)
    ordered = subset.apply(lambda row: _canonical_pair(row["image_id"], row["neighbor_image_id"]), axis=1, result_type="expand")
    subset["image_id"] = ordered[0]
    subset["neighbor_image_id"] = ordered[1]
    subset["pair_key"] = ordered[2]
    subset["provenance"] = "manual_yes_pair"
    subset["manual_candidate_type"] = subset["candidate_type"].astype(str)
    subset["manual_candidate_key"] = subset["candidate_key"].astype(str)
    subset["manual_note"] = subset["note"].fillna("").astype(str)
    keep_columns = [
        "image_id",
        "neighbor_image_id",
        "pair_key",
        "provenance",
        "manual_candidate_type",
        "manual_candidate_key",
        "manual_note",
        "xgb_same_identity_prob",
        "ambiguity_score",
    ]
    return subset[keep_columns].drop_duplicates(subset=["pair_key", "manual_candidate_type", "manual_candidate_key"]).reset_index(drop=True)


def load_approved_seed_members(
    review_package_dir: Path,
    *,
    approved_class_indices: list[int] | None = None,
    class_exclusions: dict[int, set[str]] | None = None,
) -> pd.DataFrame:
    # `approved_class_index` is interpreted as the 1-based order shown in the
    # review package contact-sheet table. This matches how the user reviewed classes.
    approved_class_indices = approved_class_indices or list(DEFAULT_APPROVED_CLASS_INDICES)
    class_exclusions = class_exclusions or DEFAULT_CLASS_EXCLUSIONS
    review_tables_dir = review_package_dir / "tables"
    contact_sheet_path = review_tables_dir / "seed_cluster_contact_sheets_v1.csv"
    pseudo_manifest_path = review_tables_dir / "pseudo_manifest_v1.csv"

    contact_df = pd.read_csv(contact_sheet_path)
    contact_df = contact_df.reset_index(drop=True)
    contact_df["approved_class_index"] = contact_df.index + 1
    contact_df["pseudo_identity"] = contact_df["pseudo_identity"].astype(str)

    pseudo_df = pd.read_csv(pseudo_manifest_path)
    pseudo_df["dataset"] = pseudo_df["dataset"].astype(str)
    pseudo_df["image_id"] = pseudo_df["image_id"].astype(str)
    pseudo_df["pseudo_identity"] = pseudo_df["pseudo_identity"].astype(str)
    pseudo_df = pseudo_df[pseudo_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)

    approved_df = contact_df[contact_df["approved_class_index"].isin([int(value) for value in approved_class_indices])].copy()
    approved_members = pseudo_df.merge(approved_df[["approved_class_index", "pseudo_identity"]], on="pseudo_identity", how="inner")
    approved_members["excluded_by_user"] = approved_members.apply(
        lambda row: str(row["image_id"]) in class_exclusions.get(int(row["approved_class_index"]), set()),
        axis=1,
    )
    approved_members["approved_class_exclusion_reason"] = approved_members.apply(
        lambda row: (
            f"class_{int(row['approved_class_index'])}_explicit_exclusion"
            if bool(row["excluded_by_user"])
            else ""
        ),
        axis=1,
    )
    approved_members = approved_members[~approved_members["excluded_by_user"]].copy().reset_index(drop=True)
    approved_members["provenance"] = "approved_seed_class"
    approved_members["approved_class_index"] = approved_members["approved_class_index"].astype(int)
    keep_columns = [
        "image_id",
        "pseudo_identity",
        "approved_class_index",
        "provenance",
        "path",
        "preferred_path_v1",
        "recommended_model_input_path_v1",
        "original_rgb_path_v1",
        "texas_center_body_repaired_fallback_stage_v1",
    ]
    return approved_members[[column for column in keep_columns if column in approved_members.columns]].copy()


def build_approved_seed_edges(approved_members_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if approved_members_df.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "neighbor_image_id",
                "pair_key",
                "provenance",
                "approved_class_index",
                "pseudo_identity",
            ]
        )
    for (approved_class_index, pseudo_identity), group in approved_members_df.groupby(
        ["approved_class_index", "pseudo_identity"],
        sort=True,
    ):
        # Once a seed class is approved, we trust every remaining pair inside it.
        image_ids = sorted(group["image_id"].astype(str).tolist())
        for left, right in itertools.combinations(image_ids, 2):
            ordered_left, ordered_right, pair_key = _canonical_pair(left, right)
            rows.append(
                {
                    "image_id": ordered_left,
                    "neighbor_image_id": ordered_right,
                    "pair_key": pair_key,
                    "provenance": "approved_seed_class",
                    "approved_class_index": int(approved_class_index),
                    "pseudo_identity": str(pseudo_identity),
                }
            )
    return pd.DataFrame(rows)


def load_texas_metadata(metadata_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path)
    frame["dataset"] = frame["dataset"].astype(str)
    frame["image_id"] = frame["image_id"].astype(str)
    subset = frame[frame["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    if subset.empty:
        return pd.DataFrame(columns=["image_id", "path", "split", "dataset"])
    return subset


def _aggregate_trusted_pairs(manual_yes_df: pd.DataFrame, approved_edge_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not manual_yes_df.empty:
        frames.append(manual_yes_df.copy())
    if not approved_edge_df.empty:
        frames.append(approved_edge_df.copy())
    if not frames:
        return pd.DataFrame(
            columns=[
                "pair_key",
                "image_id",
                "neighbor_image_id",
                "provenance",
                "manual_yes_pair",
                "approved_seed_class",
                "manual_candidate_types",
                "manual_candidate_keys",
                "approved_class_indices",
                "approved_pseudo_identities",
            ]
        )

    all_pairs = pd.concat(frames, ignore_index=True)
    grouped_rows: list[dict[str, object]] = []
    for pair_key, group in all_pairs.groupby("pair_key", sort=True):
        left = str(group["image_id"].iloc[0])
        right = str(group["neighbor_image_id"].iloc[0])
        provenances = _normalize_text_list(group["provenance"].tolist())
        grouped_rows.append(
            {
                "pair_key": pair_key,
                "image_id": left,
                "neighbor_image_id": right,
                "provenance": provenances,
                "manual_yes_pair": bool((group["provenance"].astype(str) == "manual_yes_pair").any()),
                "approved_seed_class": bool((group["provenance"].astype(str) == "approved_seed_class").any()),
                "manual_candidate_types": _normalize_text_list(group.get("manual_candidate_type", pd.Series(dtype=str)).tolist()),
                "manual_candidate_keys": _normalize_text_list(group.get("manual_candidate_key", pd.Series(dtype=str)).tolist()),
                "approved_class_indices": _normalize_text_list(group.get("approved_class_index", pd.Series(dtype=str)).tolist()),
                "approved_pseudo_identities": _normalize_text_list(group.get("pseudo_identity", pd.Series(dtype=str)).tolist()),
                "manual_pair_count": int((group["provenance"].astype(str) == "manual_yes_pair").sum()),
                "approved_seed_pair_count": int((group["provenance"].astype(str) == "approved_seed_class").sum()),
            }
        )
    trusted_pairs_df = pd.DataFrame(grouped_rows).sort_values(["image_id", "neighbor_image_id"]).reset_index(drop=True)
    return trusted_pairs_df


def _build_membership(
    *,
    trusted_pairs_df: pd.DataFrame,
    approved_members_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    uf = UnionFind()
    member_sources: dict[str, dict[str, object]] = {}

    def ensure_member(image_id: str) -> None:
        uf.add(image_id)
        member_sources.setdefault(
            image_id,
            {
                "image_id": image_id,
                "has_manual_yes_pair": False,
                "has_approved_seed_class": False,
                "approved_class_indices": set(),
                "approved_pseudo_identities": set(),
            },
        )

    for row in trusted_pairs_df.itertuples(index=False):
        ensure_member(str(row.image_id))
        ensure_member(str(row.neighbor_image_id))
        # Union-find is the actual trusted-batch compiler: every trusted positive
        # edge merges two images into the same connected component.
        uf.union(str(row.image_id), str(row.neighbor_image_id))
        if bool(row.manual_yes_pair):
            member_sources[str(row.image_id)]["has_manual_yes_pair"] = True
            member_sources[str(row.neighbor_image_id)]["has_manual_yes_pair"] = True
        if bool(row.approved_seed_class):
            member_sources[str(row.image_id)]["has_approved_seed_class"] = True
            member_sources[str(row.neighbor_image_id)]["has_approved_seed_class"] = True

    for row in approved_members_df.itertuples(index=False):
        image_id = str(row.image_id)
        ensure_member(image_id)
        member_sources[image_id]["has_approved_seed_class"] = True
        member_sources[image_id]["approved_class_indices"].add(int(row.approved_class_index))
        member_sources[image_id]["approved_pseudo_identities"].add(str(row.pseudo_identity))

    component_members: dict[str, list[str]] = {}
    for image_id in sorted(member_sources):
        root = uf.find(image_id)
        component_members.setdefault(root, []).append(image_id)

    component_keys = sorted(component_members.keys(), key=lambda root: (min(int(image_id) for image_id in component_members[root]), root))
    root_to_component_id = {root: f"trusted_comp_{index:03d}" for index, root in enumerate(component_keys, start=1)}

    metadata_lookup = metadata_df.set_index("image_id", drop=False) if not metadata_df.empty else pd.DataFrame()
    approved_lookup = approved_members_df.drop_duplicates(subset=["image_id"]).set_index("image_id", drop=False) if not approved_members_df.empty else pd.DataFrame()

    membership_rows: list[dict[str, object]] = []
    for root in component_keys:
        members = sorted(component_members[root], key=lambda value: int(value))
        component_id = root_to_component_id[root]
        for image_id in members:
            metadata_row = metadata_lookup.loc[image_id] if not metadata_lookup.empty and image_id in metadata_lookup.index else None
            approved_row = approved_lookup.loc[image_id] if not approved_lookup.empty and image_id in approved_lookup.index else None
            source_state = member_sources[image_id]
            membership_rows.append(
                {
                    "component_id": component_id,
                    "image_id": image_id,
                    "component_size": int(len(members)),
                    "path": (
                        str(approved_row["path"])
                        if approved_row is not None and "path" in approved_row.index
                        else (str(metadata_row["path"]) if metadata_row is not None and "path" in metadata_row.index else "")
                    ),
                    "original_rgb_path_v1": (
                        str(approved_row["original_rgb_path_v1"])
                        if approved_row is not None and "original_rgb_path_v1" in approved_row.index
                        else (str(metadata_row["path"]) if metadata_row is not None and "path" in metadata_row.index else "")
                    ),
                    "split": str(metadata_row["split"]) if metadata_row is not None and "split" in metadata_row.index else "",
                    "dataset": TEXAS_DATASET,
                    "has_manual_yes_pair": bool(source_state["has_manual_yes_pair"]),
                    "has_approved_seed_class": bool(source_state["has_approved_seed_class"]),
                    "approved_class_indices": _normalize_text_list(list(source_state["approved_class_indices"])),
                    "approved_pseudo_identities": _normalize_text_list(list(source_state["approved_pseudo_identities"])),
                }
            )
    membership_df = pd.DataFrame(membership_rows).sort_values(["component_id", "image_id"]).reset_index(drop=True)

    component_rows: list[dict[str, object]] = []
    for component_id, group in membership_df.groupby("component_id", sort=True):
        member_ids = group["image_id"].astype(str).tolist()
        member_set = set(member_ids)
        pair_subset = trusted_pairs_df[
            trusted_pairs_df["image_id"].astype(str).isin(member_set)
            & trusted_pairs_df["neighbor_image_id"].astype(str).isin(member_set)
        ].copy()
        component_rows.append(
            {
                "component_id": component_id,
                "member_count": int(len(group)),
                "pair_count": int(len(pair_subset)),
                "manual_yes_pair_edge_count": int(pair_subset["manual_yes_pair"].fillna(False).astype(bool).sum()) if not pair_subset.empty else 0,
                "approved_seed_class_edge_count": int(pair_subset["approved_seed_class"].fillna(False).astype(bool).sum()) if not pair_subset.empty else 0,
                "has_manual_yes_pair": bool(group["has_manual_yes_pair"].fillna(False).astype(bool).any()),
                "has_approved_seed_class": bool(group["has_approved_seed_class"].fillna(False).astype(bool).any()),
                "approved_class_indices": _normalize_text_list(group["approved_class_indices"].astype(str).tolist()),
                "approved_pseudo_identities": _normalize_text_list(group["approved_pseudo_identities"].astype(str).tolist()),
                "image_ids": "|".join(member_ids),
            }
        )
    components_df = pd.DataFrame(component_rows).sort_values(["member_count", "component_id"], ascending=[False, True]).reset_index(drop=True)
    return membership_df, components_df


def compile_texas_trusted_batch(
    *,
    repo_root: Path,
    manual_pairs_path: Path,
    review_package_dir: Path,
    output_dir: Path,
    approved_class_indices: list[int] | None = None,
    class_exclusions: dict[int, set[str]] | None = None,
    metadata_path: Path | None = None,
) -> TrustedBatchArtifacts:
    repo_root = repo_root.resolve()
    manual_pairs_path = manual_pairs_path.resolve()
    review_package_dir = review_package_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = metadata_path.resolve() if metadata_path is not None else repo_root / "metadata.csv"

    manual_yes_df = load_texas_manual_yes_pairs(manual_pairs_path)
    approved_members_df = load_approved_seed_members(
        review_package_dir,
        approved_class_indices=approved_class_indices,
        class_exclusions=class_exclusions,
    )
    approved_edge_df = build_approved_seed_edges(approved_members_df)
    trusted_pairs_df = _aggregate_trusted_pairs(manual_yes_df=manual_yes_df, approved_edge_df=approved_edge_df)
    metadata_df = load_texas_metadata(metadata_path) if metadata_path.exists() else pd.DataFrame()
    membership_df, components_df = _build_membership(
        trusted_pairs_df=trusted_pairs_df,
        approved_members_df=approved_members_df,
        metadata_df=metadata_df,
    )

    trusted_membership_path = tables_dir / "trusted_membership_v1.csv"
    trusted_pairs_path = tables_dir / "trusted_pairs_v1.csv"
    trusted_components_path = tables_dir / "trusted_components_v1.csv"
    membership_df.to_csv(trusted_membership_path, index=False)
    trusted_pairs_df.to_csv(trusted_pairs_path, index=False)
    components_df.to_csv(trusted_components_path, index=False)

    summary_df = pd.DataFrame(
        [
            {
                "trusted_components": int(len(components_df)),
                "trusted_members": int(len(membership_df)),
                "trusted_pairs": int(len(trusted_pairs_df)),
                "manual_yes_pairs": int(len(manual_yes_df)),
                "approved_seed_members": int(len(approved_members_df)),
                "approved_seed_edges": int(len(approved_edge_df)),
            }
        ]
    )
    approved_class_df = (
        approved_members_df[["approved_class_index", "pseudo_identity"]]
        .drop_duplicates()
        .sort_values(["approved_class_index", "pseudo_identity"])
        .reset_index(drop=True)
        if not approved_members_df.empty
        else pd.DataFrame(columns=["approved_class_index", "pseudo_identity"])
    )
    exclusion_rows = []
    for class_index in sorted((class_exclusions or DEFAULT_CLASS_EXCLUSIONS).keys()):
        for image_id in sorted((class_exclusions or DEFAULT_CLASS_EXCLUSIONS)[class_index]):
            exclusion_rows.append({"approved_class_index": int(class_index), "excluded_image_id": str(image_id)})
    exclusions_df = pd.DataFrame(exclusion_rows) if exclusion_rows else pd.DataFrame(columns=["approved_class_index", "excluded_image_id"])

    summary_lines = [
        "# Texas Trusted Batch",
        "",
        "## Inputs",
        "",
        f"- Manual positive pairs: `{manual_pairs_path.relative_to(repo_root)}`",
        f"- Review package: `{review_package_dir.relative_to(repo_root)}`",
        f"- Metadata: `{metadata_path.relative_to(repo_root) if metadata_path.exists() else metadata_path}`",
        "",
        "## Summary",
        "",
        dataframe_to_markdown_table(summary_df),
        "",
        "## Approved Seed Classes",
        "",
        dataframe_to_markdown_table(approved_class_df),
        "",
        "## Explicit Exclusions",
        "",
        dataframe_to_markdown_table(exclusions_df),
        "",
        "## Largest Trusted Components",
        "",
        dataframe_to_markdown_table(components_df.head(15)),
        "",
    ]
    summary_path = reports_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return TrustedBatchArtifacts(
        trusted_membership_path=trusted_membership_path,
        trusted_pairs_path=trusted_pairs_path,
        trusted_components_path=trusted_components_path,
        summary_path=summary_path,
    )
