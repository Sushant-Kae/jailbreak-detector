"""Fetch and merge external Hugging Face jailbreak-detection datasets with
the project's synthetic starter dataset.

Requires network access to huggingface.co, which most restricted/offline
sandboxes will NOT have. If `pd.read_parquet("hf://...")` fails with a host
allowlist / 403 error, run this script on your own machine or in an
environment with normal internet access instead.

Usage:
    pip install datasets huggingface_hub pyarrow
    python scripts/fetch_external_datasets.py

Output:
    data/dataset_combined.csv  (same text,label,category schema as
    data/dataset.csv, with an added `source` column)

IMPORTANT: unlike data/dataset.csv (synthetic, abstract placeholders only),
the two external datasets contain REAL adversarial and sometimes sensitive
text (e.g. weapons, self-harm, hate speech, sexual-content-related prompts —
the llm-semantic-router dataset explicitly documents categories like
S5_weapons_cbrne and S3_sex_crimes). Read the README section on this before
committing data/dataset_combined.csv anywhere, especially a public repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import src`

from src.normalize import normalize_text

OUTPUT_PATH = Path("data/dataset_combined.csv")
STARTER_PATH = Path("data/dataset.csv")

# Label values (lowercased) that map to each of our two classes. Extend
# these sets if a dataset uses different vocabulary than what's below.
SUSPICIOUS_LABEL_WORDS = {
    "jailbreak", "unsafe", "malicious", "injection", "attack", "adversarial", "1",
}
NORMAL_LABEL_WORDS = {
    "benign", "safe", "normal", "legit", "legitimate", "0",
}

# Candidate column names to look for text/label, in priority order.
TEXT_COLUMN_CANDIDATES = ["text", "prompt", "instruction", "input", "query", "message"]
LABEL_COLUMN_CANDIDATES = ["label_text", "label", "class", "target", "category"]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_cols:
            return lower_cols[candidate]
    return None


def _map_label(raw_label) -> str | None:
    """Map a raw label value to 'normal' or 'suspicious'. Returns None if
    the value is unrecognized (caller should drop or raise on these)."""
    value = str(raw_label).strip().lower()
    if value in SUSPICIOUS_LABEL_WORDS:
        return "suspicious"
    if value in NORMAL_LABEL_WORDS:
        return "normal"
    return None


def standardize_dataframe(df: pd.DataFrame, source_name: str, category_prefix: str) -> pd.DataFrame:
    """Convert an arbitrary external dataframe into our text,label,category,source schema.

    Raises ValueError with a clear message if text/label columns can't be
    identified, so the caller can inspect df.columns and extend the
    candidate lists above rather than silently mis-mapping data.
    """
    text_col = _find_column(df, TEXT_COLUMN_CANDIDATES)
    label_col = _find_column(df, LABEL_COLUMN_CANDIDATES)

    if text_col is None or label_col is None:
        raise ValueError(
            f"[{source_name}] Could not auto-detect text/label columns. "
            f"Available columns: {list(df.columns)}. "
            "Add the correct names to TEXT_COLUMN_CANDIDATES / "
            "LABEL_COLUMN_CANDIDATES in this script."
        )

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    out["label"] = df[label_col].apply(_map_label)
    out["category"] = f"{category_prefix}"
    out["source"] = source_name

    unmapped = out["label"].isna().sum()
    if unmapped:
        print(
            f"[{source_name}] Warning: dropping {unmapped} rows with an "
            f"unrecognized label value in column '{label_col}'."
        )
    out = out.dropna(subset=["label", "text"])
    out = out[out["text"].str.strip().str.len() > 0]
    return out.reset_index(drop=True)


def load_llm_semantic_router() -> pd.DataFrame:
    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }
    frames = []
    for split_name, path in splits.items():
        df = pd.read_parquet(f"hf://datasets/llm-semantic-router/jailbreak-detection-dataset/{path}")
        std = standardize_dataframe(
            df, source_name="llm-semantic-router", category_prefix=f"external_llm_semantic_router_{split_name}"
        )
        frames.append(std)
    return pd.concat(frames, ignore_index=True)


def load_sentinel() -> pd.DataFrame:
    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }
    frames = []
    for split_name, path in splits.items():
        df = pd.read_parquet(f"hf://datasets/Chgdz/sentinel-jailbreak-detection/{path}")
        std = standardize_dataframe(
            df, source_name="sentinel-jailbreak-detection", category_prefix=f"external_sentinel_{split_name}"
        )
        frames.append(std)
    return pd.concat(frames, ignore_index=True)


def load_starter_dataset() -> pd.DataFrame:
    df = pd.read_csv(STARTER_PATH)
    df["source"] = "starter_synthetic"
    return df


def merge_and_dedupe(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    combined["text"] = combined["text"].astype(str)
    # Dedupe on normalized text so near-identical rows across sources collapse.
    combined["_dedupe_key"] = combined["text"].apply(normalize_text)
    before = len(combined)
    combined = combined.drop_duplicates(subset="_dedupe_key").drop(columns="_dedupe_key")
    after = len(combined)
    if before != after:
        print(f"Deduplicated {before - after} rows with duplicate normalized text.")
    return combined.reset_index(drop=True)


def main() -> None:
    print("Loading starter synthetic dataset...")
    starter = load_starter_dataset()

    print("Loading llm-semantic-router/jailbreak-detection-dataset ...")
    try:
        semantic_router = load_llm_semantic_router()
        print(f"  -> {len(semantic_router)} rows")
    except Exception as exc:  # noqa: BLE001 - surface any network/schema error clearly
        print(f"  FAILED: {exc}")
        print("  Skipping this source. See the module docstring for troubleshooting.")
        semantic_router = pd.DataFrame(columns=["text", "label", "category", "source"])

    print("Loading Chgdz/sentinel-jailbreak-detection ...")
    try:
        sentinel = load_sentinel()
        print(f"  -> {len(sentinel)} rows")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}")
        print("  Skipping this source. See the module docstring for troubleshooting.")
        sentinel = pd.DataFrame(columns=["text", "label", "category", "source"])

    combined = merge_and_dedupe([starter, semantic_router, sentinel])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined[["text", "label", "category", "source"]].to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(combined)} total rows to {OUTPUT_PATH}")
    print(combined["label"].value_counts())
    print(combined["source"].value_counts())
    print(
        "\nReminder: this file contains REAL adversarial/sensitive text from "
        "external sources. See README before "
        "sharing or committing it."
    )


if __name__ == "__main__":
    main()
