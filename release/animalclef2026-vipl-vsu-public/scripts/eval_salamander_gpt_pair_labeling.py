#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_YES_PAIRS = Path("artifacts/analysis/salamander_trusted_batch_v1/tables/trusted_pairs_v1.csv")
DEFAULT_NO_PAIRS = Path("artifacts/analysis/salamander_trusted_batch_v1/tables/cannot_link_pairs_v1.csv")
DEFAULT_METADATA = Path("metadata.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_gpt_pair_label_eval_v1")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate GPT-style vision pair labeling against curated Salamander pair judgments."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--yes-pairs", type=Path, default=DEFAULT_YES_PAIRS)
    parser.add_argument("--no-pairs", type=Path, default=DEFAULT_NO_PAIRS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default="gpt-5.5")
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-file", type=Path, default=None, help="Optional provider env file with exported API keys.")
    parser.add_argument("--ccs-provider", type=str, default=None, help="Read API key/base URL from ~/.cc-switch for this provider id.")
    parser.add_argument("--base-url", type=str, default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--dry-run", action="store_true", help="Only build the sampled pair table and metrics shell.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls.")
    return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _load_env_file(path: Path | None) -> None:
    if path is None:
        return
    if not path.exists():
        raise FileNotFoundError(path)
    pattern = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        key, value = match.groups()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def _first_available_env(keys: list[str]) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return None


def _load_ccs_provider(provider_id: str, *, app_type: str = "codex") -> tuple[str | None, str | None]:
    db_path = Path.home() / ".cc-switch" / "cc-switch.db"
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "select settings_config from providers where id = ? and app_type = ?",
            (provider_id, app_type),
        ).fetchone()
    if row is None:
        raise KeyError(f"ccs provider not found: {provider_id!r} for app={app_type!r}")
    settings = json.loads(str(row[0]))
    auth = settings.get("auth", {}) if isinstance(settings, dict) else {}
    api_key = auth.get("OPENAI_API_KEY") or auth.get("API_KEY")
    config_text = str(settings.get("config", ""))
    base_url = None
    match = re.search(r'base_url\s*=\s*"([^"]+)"', config_text)
    if match:
        base_url = match.group(1)
    return api_key, base_url


def _canonical_pair(left: object, right: object) -> tuple[str, str, str]:
    ordered = sorted([str(left), str(right)])
    return ordered[0], ordered[1], f"{ordered[0]}|{ordered[1]}"


def _load_pair_frame(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image_id", "neighbor_image_id"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        left, right, pair_key = _canonical_pair(row.image_id, row.neighbor_image_id)
        rows.append(
            {
                "pair_key": pair_key,
                "image_id": left,
                "neighbor_image_id": right,
                "ground_truth": label,
                "source_path": str(path),
                "candidate_types": str(getattr(row, "candidate_types", "")),
                "candidate_keys": str(getattr(row, "candidate_keys", "")),
                "manual_pair_count": int(getattr(row, "manual_pair_count", 1)),
                "xgb_same_identity_prob": float(getattr(row, "max_xgb_same_identity_prob", 0.0)),
                "ambiguity_score": float(getattr(row, "max_ambiguity_score", 0.0)),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["pair_key"]).reset_index(drop=True)


def _sample_pairs(yes_df: pd.DataFrame, no_df: pd.DataFrame, *, n_pairs: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    n_yes = min(len(yes_df), n_pairs // 2)
    n_no = min(len(no_df), n_pairs - n_yes)
    if n_yes + n_no < n_pairs and len(yes_df) > n_yes:
        n_yes = min(len(yes_df), n_yes + (n_pairs - n_yes - n_no))
    yes_idx = rng.sample(range(len(yes_df)), n_yes) if n_yes else []
    no_idx = rng.sample(range(len(no_df)), n_no) if n_no else []
    sampled = pd.concat([yes_df.iloc[yes_idx], no_df.iloc[no_idx]], ignore_index=True)
    order = list(range(len(sampled)))
    rng.shuffle(order)
    sampled = sampled.iloc[order].reset_index(drop=True)
    sampled.insert(0, "sample_id", [f"salamander_pair_{idx + 1:03d}" for idx in range(len(sampled))])
    return sampled


def _load_image_lookup(metadata_path: Path, repo_root: Path) -> dict[str, str]:
    metadata = pd.read_csv(metadata_path, low_memory=False)
    metadata["image_id"] = metadata["image_id"].astype(str)
    metadata = metadata[metadata["dataset"].astype(str).eq(SALAMANDER_DATASET)].copy()
    path_col = "path"
    lookup: dict[str, str] = {}
    for row in metadata.itertuples(index=False):
        image_id = str(getattr(row, "image_id"))
        rel_path = str(getattr(row, path_col))
        path = (repo_root / rel_path).resolve()
        if path.exists():
            lookup[image_id] = str(path)
    return lookup


def _image_to_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["yes", "no", "uncertain"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "brief_reason": {"type": "string", "maxLength": 240},
        },
        "required": ["label", "confidence", "brief_reason"],
        "additionalProperties": False,
    }


def _extract_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return str(payload["output_text"])
    chunks: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                chunks.append(str(content["text"]))
    return "\n".join(chunks).strip()


def _post_responses(*, api_key: str, base_url: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    root = (base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{root}/responses"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "OpenAI/Python animalclef-pair-eval",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {text[:800]}") from exc


def _call_model(*, api_key: str, base_url: str | None, model: str, left_path: str, right_path: str) -> dict[str, Any]:
    prompt = (
        "You are judging individual animal re-identification for fire salamanders. "
        "Compare the two images and decide whether they show the same individual. "
        "Focus on stable dorsal yellow spot patterns, relative pattern geometry, and body markings. "
        "Ignore pose, crop, lighting, background, and scale when possible. "
        "Return yes if they are the same individual, no if different, uncertain if the images are not sufficient."
    )
    payload = {
        "model": model,
        "instructions": "Return only valid JSON matching the requested schema.",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_text", "text": "Image A:"},
                    {"type": "input_image", "image_url": _image_to_data_url(left_path), "detail": "high"},
                    {"type": "input_text", "text": "Image B:"},
                    {"type": "input_image", "image_url": _image_to_data_url(right_path), "detail": "high"},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "salamander_pair_judgment",
                "strict": True,
                "schema": _schema(),
            }
        },
    }
    response = _post_responses(api_key=api_key, base_url=base_url, payload=payload)
    return json.loads(_extract_output_text(response))


def _metrics(pred_df: pd.DataFrame) -> dict[str, Any]:
    evaluated = pred_df[pred_df["pred_label"].isin(["yes", "no"])].copy()
    labels = ["yes", "no"]
    confusion = {
        gt: {pred: int(((evaluated["ground_truth"] == gt) & (evaluated["pred_label"] == pred)).sum()) for pred in labels}
        for gt in labels
    }
    total = int(len(evaluated))
    correct = int((evaluated["ground_truth"] == evaluated["pred_label"]).sum()) if total else 0
    per_label: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[gt][label] for gt in labels if gt != label)
        fn = sum(confusion[label][pred] for pred in labels if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "n_total": int(len(pred_df)),
        "n_evaluated_yes_no": total,
        "n_uncertain_or_failed": int(len(pred_df) - total),
        "accuracy": correct / total if total else None,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "confusion_matrix": confusion,
        "per_label": per_label,
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = _resolve(repo_root, args.output_dir).resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    yes_df = _load_pair_frame(_resolve(repo_root, args.yes_pairs), "yes")
    no_df = _load_pair_frame(_resolve(repo_root, args.no_pairs), "no")
    sampled = _sample_pairs(yes_df, no_df, n_pairs=int(args.n_pairs), seed=int(args.seed))

    image_lookup = _load_image_lookup(_resolve(repo_root, args.metadata), repo_root)
    sampled["image_path"] = sampled["image_id"].map(image_lookup)
    sampled["neighbor_image_path"] = sampled["neighbor_image_id"].map(image_lookup)
    missing = sampled[sampled["image_path"].isna() | sampled["neighbor_image_path"].isna()]
    if not missing.empty:
        raise FileNotFoundError(f"Missing image paths for {len(missing)} sampled pairs.")
    sampled.to_csv(tables_dir / "sampled_pairs_v1.csv", index=False)

    if args.dry_run:
        metrics = {"status": "dry_run", "n_sampled": int(len(sampled)), "label_counts": sampled["ground_truth"].value_counts().to_dict()}
        (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"[dry-run] sampled pairs: {tables_dir / 'sampled_pairs_v1.csv'}")
        print(f"[dry-run] metrics: {reports_dir / 'metrics.json'}")
        return 0

    _load_env_file(args.env_file)
    ccs_api_key = None
    ccs_base_url = None
    if args.ccs_provider:
        ccs_api_key, ccs_base_url = _load_ccs_provider(str(args.ccs_provider))
    api_key = (
        os.environ.get(args.api_key_env)
        or ccs_api_key
        or _first_available_env(["OPENAI_API_KEY", "CODE78_API_KEY", "GMN_API_KEY"])
    )
    if not api_key:
        raise RuntimeError("No API key found. Set OPENAI_API_KEY or pass --env-file with a provider key.")

    base_url = args.base_url or ccs_base_url or os.environ.get("OPENAI_BASE_URL")
    if args.ccs_provider == "code78" and base_url == "https://www.78code.cc/v1":
        base_url = "https://api.78code.cc/v1"
    rows: list[dict[str, Any]] = []
    progress_path = tables_dir / "gpt_pair_predictions_progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    predictions_path = tables_dir / "gpt_pair_predictions_v1.csv"
    metrics_path = reports_dir / "metrics.json"
    total_pairs = int(len(sampled))
    for pair_index, row in enumerate(sampled.itertuples(index=False), start=1):
        result: dict[str, Any]
        error = ""
        started = time.time()
        try:
            result = _call_model(
                api_key=api_key,
                base_url=base_url,
                model=str(args.model),
                left_path=str(row.image_path),
                right_path=str(row.neighbor_image_path),
            )
        except Exception as exc:  # noqa: BLE001
            result = {"label": "uncertain", "confidence": 0.0, "brief_reason": ""}
            error = repr(exc)
        rows.append(
            {
                "sample_id": row.sample_id,
                "pair_key": row.pair_key,
                "image_id": row.image_id,
                "neighbor_image_id": row.neighbor_image_id,
                "ground_truth": row.ground_truth,
                "pred_label": str(result.get("label", "uncertain")).lower(),
                "pred_confidence": float(result.get("confidence", 0.0)),
                "brief_reason": str(result.get("brief_reason", "")),
                "error": error,
                "xgb_same_identity_prob": row.xgb_same_identity_prob,
                "ambiguity_score": row.ambiguity_score,
                "image_path": row.image_path,
                "neighbor_image_path": row.neighbor_image_path,
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
        with progress_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(predictions_path, index=False)
        metrics = _metrics(pred_df)
        metrics["model"] = str(args.model)
        metrics["sample_seed"] = int(args.seed)
        metrics["errors"] = int(pred_df["error"].astype(str).ne("").sum())
        metrics["completed"] = int(len(pred_df))
        metrics["total_requested"] = total_pairs
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(
            f"[{pair_index:03d}/{total_pairs:03d}] {row.sample_id} "
            f"gt={row.ground_truth} pred={rows[-1]['pred_label']} "
            f"conf={rows[-1]['pred_confidence']:.2f} "
            f"err={'yes' if error else 'no'} "
            f"elapsed={rows[-1]['elapsed_sec']:.1f}s",
            flush=True,
        )
        if float(args.sleep) > 0:
            time.sleep(float(args.sleep))

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(predictions_path, index=False)
    metrics = _metrics(pred_df)
    metrics["model"] = str(args.model)
    metrics["sample_seed"] = int(args.seed)
    metrics["errors"] = int(pred_df["error"].astype(str).ne("").sum())
    metrics["completed"] = int(len(pred_df))
    metrics["total_requested"] = total_pairs
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[eval] predictions: {predictions_path}")
    print(f"[eval] progress_jsonl: {progress_path}")
    print(f"[eval] metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
