"""TF-IDF + Logistic Regression classifier.

Kept deliberately simple and explainable per project requirements:
no deep learning, single sklearn Pipeline so vectorizer and classifier
always travel together and can never go out of sync.
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import RANDOM_SEED

MODEL_PATH = Path("models/jailbreak_model.joblib")

# Label order used everywhere probabilities are reported.
POSITIVE_LABEL = "suspicious"


def build_pipeline() -> Pipeline:
    """Build an untrained TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,  # normalization already lowercases
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_model(x_train: list[str], y_train: list[str]) -> Pipeline:
    """Fit a fresh pipeline on already-normalized training text."""
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    return pipeline


def predict_proba_suspicious(pipeline: Pipeline, texts: list[str]) -> list[float]:
    """Return P(label == 'suspicious') for each text, using the same
    normalized text convention as training."""
    classes = list(pipeline.classes_)
    idx = classes.index(POSITIVE_LABEL)
    probs = pipeline.predict_proba(texts)
    return [row[idx] for row in probs]


def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python main.py train` first."
        )
    return joblib.load(path)
