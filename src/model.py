"""TF-IDF + Logistic Regression classifier.

Kept deliberately simple and explainable per project requirements:
no deep learning, single sklearn Pipeline so vectorizer and classifier
always travel together and can never go out of sync.

Feature representation: word-level, character-level, or combined TF-IDF
was compared experimentally (see README "Feature representation
comparison" section for the actual numbers from that run) rather than
assumed. The winner is set as DEFAULT_FEATURE_MODE below.
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data import RANDOM_SEED
from src.normalize import normalize_text

MODEL_PATH = Path("models/jailbreak_model.joblib")

# Label order used everywhere probabilities are reported.
POSITIVE_LABEL = "suspicious"

# Chosen after comparing word/char/combined on validation data -- see
# README. "combined" adds character 3-5 grams alongside word 1-2 grams,
# which helps with minor spelling/obfuscation variants without needing a
# bigger model.
DEFAULT_FEATURE_MODE = "combined"


def _build_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,  # normalization already lowercases
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )


def _build_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_df=0.95,
    )


def build_pipeline(feature_mode: str = DEFAULT_FEATURE_MODE) -> Pipeline:
    """Build an untrained TF-IDF + Logistic Regression pipeline.

    feature_mode: "word" (word 1-2 grams only), "char" (character 3-5
    grams only), or "combined" (both, concatenated via FeatureUnion).
    """
    if feature_mode == "word":
        vectorizer_step = ("tfidf", _build_word_vectorizer())
    elif feature_mode == "char":
        vectorizer_step = ("tfidf", _build_char_vectorizer())
    elif feature_mode == "combined":
        vectorizer_step = (
            "tfidf",
            FeatureUnion(
                [
                    ("word", _build_word_vectorizer()),
                    ("char", _build_char_vectorizer()),
                ]
            ),
        )
    else:
        raise ValueError(f"Unknown feature_mode '{feature_mode}'. Expected 'word', 'char', or 'combined'.")

    return Pipeline(
        steps=[
            vectorizer_step,
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


def train_model(x_train: list[str], y_train: list[str], feature_mode: str = DEFAULT_FEATURE_MODE) -> Pipeline:
    """Fit a fresh pipeline on training text.

    Normalizes each text immediately before fitting -- x_train is expected
    to be raw-ish text (e.g. straight from src.data.load_dataset, which no
    longer lowercases at load time so Base64 payloads keep their case for
    transformation analysis). Predict-time (src/pipeline.py) normalizes at
    the same point, so train/predict preprocessing stays identical.
    """
    normalized_x_train = [normalize_text(t) for t in x_train]
    pipeline = build_pipeline(feature_mode)
    pipeline.fit(normalized_x_train, y_train)
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
