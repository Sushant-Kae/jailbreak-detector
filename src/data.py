"""Dataset loading and splitting."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
VALID_LABELS = {"normal", "suspicious"}


@dataclass
class Splits:
    x_train: list[str]
    y_train: list[str]
    x_val: list[str]
    y_val: list[str]
    x_test: list[str]
    y_test: list[str]


def validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema/labels on an already-loaded DataFrame and do light
    cleanup only (cast to str, strip whitespace, drop empty rows).

    Deliberately does NOT lowercase/collapse whitespace here (that was the
    old behavior) -- full normalization (src.normalize.normalize_text) is
    applied later, at the point each text is actually fed to the ML model
    or rule signals (see src/model.py train_model and
    src/pipeline.py analyze_text). Lowercasing at load time silently
    corrupted case-sensitive Base64 payloads stored in the dataset (Base64
    is case-sensitive), which broke evaluation of the encoded-example
    category. Keeping raw case through splitting and normalizing only at
    point-of-use fixes that while keeping train/predict preprocessing
    identical (both normalize immediately before use).
    """
    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    invalid = set(df["label"].unique()) - VALID_LABELS
    if invalid:
        raise ValueError(f"Dataset contains unexpected labels: {invalid}. Expected {VALID_LABELS}")

    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    return df


def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate a dataset CSV from disk."""
    return validate_and_normalize(pd.read_csv(path))


def make_splits(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15) -> Splits:
    """Stratified train/val/test split with a fixed seed (no leakage: split
    happens before any vectorizer is fit)."""
    x_temp, x_test, y_temp, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df["label"].tolist(),
    )

    # val_size is expressed relative to the full dataset; convert to a
    # fraction of the remaining (temp) split.
    relative_val_size = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    return Splits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

