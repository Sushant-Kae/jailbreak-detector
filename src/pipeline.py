"""End-to-end prediction pipeline: normalization -> ML -> signals -> risk.

This is the single place that assembles a prediction, so training-time and
predict-time preprocessing can never drift apart.
"""

from sklearn.pipeline import Pipeline

from src.model import predict_proba_suspicious
from src.normalize import normalize_text
from src.risk import RiskResult, compute_risk


def analyze_text(pipeline: Pipeline, raw_text: str) -> RiskResult:
    """Run the full pipeline on a single raw input string.

    Handles empty input as an explicit edge case: an empty/whitespace-only
    prompt is classified as normal with a risk score of 0 and is never sent
    into the vectorizer.
    """
    normalized = normalize_text(raw_text)

    if not normalized:
        return RiskResult(
            classification="normal",
            risk_score=0.0,
            ml_probability=0.0,
            signals=[],
            reasons=["empty input"],
        )

    ml_probability = predict_proba_suspicious(pipeline, [normalized])[0]
    return compute_risk(normalized, ml_probability)


def analyze_batch(pipeline: Pipeline, raw_texts: list[str]) -> list[RiskResult]:
    """Vectorized-friendly batch version of analyze_text.

    Non-empty texts are normalized and scored together in one model call
    for efficiency; empty texts are still short-circuited per-item.
    """
    normalized_texts = [normalize_text(t) for t in raw_texts]
    non_empty_idx = [i for i, t in enumerate(normalized_texts) if t]

    ml_probs_by_idx: dict[int, float] = {}
    if non_empty_idx:
        probs = predict_proba_suspicious(pipeline, [normalized_texts[i] for i in non_empty_idx])
        ml_probs_by_idx = dict(zip(non_empty_idx, probs))

    results: list[RiskResult] = []
    for i, normalized in enumerate(normalized_texts):
        if not normalized:
            results.append(
                RiskResult(
                    classification="normal",
                    risk_score=0.0,
                    ml_probability=0.0,
                    signals=[],
                    reasons=["empty input"],
                )
            )
        else:
            results.append(compute_risk(normalized, ml_probs_by_idx[i]))

    return results
