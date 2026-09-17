"""Controlled human-feedback loop.

Design principle (deliberately NOT automatic): the detector never trusts
its own predictions as ground truth. A new example only enters the
training data after a human confirms its correct label via
`python main.py review`. This file only handles storage; main.py's
`review` command handles the interactive confirmation step, and `train`
optionally merges reviewed examples back in.

    prediction -> signals -> explanation -> HUMAN reviews & confirms label
        -> reviewed example stored here -> periodic retrain -> new model

This intentionally does not implement automatic self-training, online
learning, or any mechanism that updates the model from unverified
predictions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.normalize import normalize_text

REVIEWED_PATH = Path("data/reviewed_examples.csv")
VALID_LABELS = {"normal", "suspicious"}
COLUMNS = ["text", "label", "source"]


def load_reviewed_examples(path: Path = REVIEWED_PATH) -> pd.DataFrame:
    """Return reviewed examples, or an empty (correctly-shaped) DataFrame
    if none have been recorded yet."""
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def save_reviewed_example(text: str, label: str, source: str = "manual_review", path: Path = REVIEWED_PATH) -> None:
    """Append one human-confirmed example to the reviewed dataset.

    Raises ValueError for an invalid label rather than silently storing
    bad data -- this file is meant to be trustworthy training input.
    """
    label = label.strip().lower()
    if label not in VALID_LABELS:
        raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}")
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_reviewed_examples(path)
    new_row = pd.DataFrame([{"text": text, "label": label, "source": source}])
    combined = pd.concat([existing, new_row], ignore_index=True)
    combined.to_csv(path, index=False)


def merge_with_base_dataset(base_path: str, reviewed_path: Path = REVIEWED_PATH) -> pd.DataFrame:
    """Combine the base dataset with any reviewed examples, deduplicating
    by normalized text (reviewed examples win on conflicts, since they
    reflect the most recent human-confirmed judgment)."""
    base_df = pd.read_csv(base_path)
    reviewed_df = load_reviewed_examples(reviewed_path)

    if reviewed_df.empty:
        return base_df

    base_df = base_df.copy()
    reviewed_df = reviewed_df.copy()
    base_df["_dedupe_key"] = base_df["text"].astype(str).apply(normalize_text)
    reviewed_df["_dedupe_key"] = reviewed_df["text"].astype(str).apply(normalize_text)

    # Reviewed examples take priority: drop any base rows whose normalized
    # text also appears in the reviewed set, then append reviewed rows.
    base_df = base_df[~base_df["_dedupe_key"].isin(set(reviewed_df["_dedupe_key"]))]

    combined = pd.concat([base_df, reviewed_df], ignore_index=True)
    combined = combined.drop(columns="_dedupe_key")
    return combined
