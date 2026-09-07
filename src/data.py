"""Dataset loading and splitting."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.normalize import normalize_text

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


def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the dataset CSV."""
    df = pd.read_csv(path)

    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    invalid = set(df["label"].unique()) - VALID_LABELS
    if invalid:
        raise ValueError(f"Dataset contains unexpected labels: {invalid}. Expected {VALID_LABELS}")

    # Normalize text once, consistently, at load time.
    df["text"] = df["text"].apply(normalize_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    return df


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

