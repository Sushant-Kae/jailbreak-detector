"""End-to-end prediction pipeline:

    raw text
      -> transformation/obfuscation analysis (Base64, spaced letters, etc.)
      -> normalization
      -> ML classifier
      -> rule signals
      -> risk scoring

This is the single place that assembles a prediction, so training-time and
predict-time preprocessing can never drift apart.

How transformation analysis is combined (documented weights):

- Finding an encoded/transformed segment at all is NOT itself evidence of
  a jailbreak (Base64 is used constantly for entirely benign reasons) --
  it contributes a small "transform_detected" signal (weight 0.05) purely
  for transparency in the explanation.
- If the *revealed* content (decoded Base64 / collapsed spaced letters)
  itself triggers a rule signal or scores highly on the ML classifier, that
  is meaningful evidence of a hidden manipulation attempt -- it adds a
  much larger "encoded_instruction_manipulation" signal (weight 0.45).
- If, in addition, the surrounding text explicitly asks the assistant to
  act on ("follow", "obey", "comply with") the decoded/transformed
  content rather than just discuss or analyze it, that intent adds a
  further "follow_intent_on_transformed_content" signal (weight 0.15).
- Revealed content is only ever analyzed as data (run back through the
  same rule/ML scoring as any other text) -- it is never executed, and
  decoding is not recursive (depth 1 only).
"""

import re

from sklearn.pipeline import Pipeline

from src.model import predict_proba_suspicious
from src.normalize import normalize_text
from src.risk import RiskResult, compute_risk
from src.signals import Signal, extract_signals
from src.transform_analysis import analyze_transformations

TRANSFORM_DETECTED_WEIGHT = 0.05
ENCODED_MANIPULATION_WEIGHT = 0.45
ENCODED_MANIPULATION_ANALYSIS_ONLY_WEIGHT = 0.20
FOLLOW_INTENT_WEIGHT = 0.15
STRUCTURAL_ANOMALY_WEIGHT = 0.10

# A revealed candidate counts as "suspicious" if its own rule signals fire
# or its own ML score clears this bar. Kept the same as the main
# classification threshold for consistency and explainability.
CANDIDATE_ML_THRESHOLD = 0.5

# Phrases suggesting the person wants the *revealed* content analyzed or
# explained, not acted on -- e.g. "just explain what it says", "don't act
# on it". This cannot give perfect context understanding (distinguishing
# a genuine analysis request from an attacker pre-emptively disclaiming
# intent is a hard, unsolved problem), but it's a reasonable, explainable
# signal to dampen -- not eliminate -- the encoded-manipulation weight
# rather than ignoring the stated intent entirely.
_ANALYSIS_ONLY_INTENT_PATTERN = re.compile(
    r"\b(just |only )?explain what it (says|means)\b"
    r"|\bdo not (act on|follow|obey|execute) (it|this|that)\b"
    r"|\bwithout (acting on|following|executing) it\b"
    r"|\bfor (analysis|educational) purposes only\b"
)


def _transformation_signals(pipeline: Pipeline, raw_text: str) -> list[Signal]:
    """Analyze raw (pre-normalization) text for obfuscation/transformation
    and return any extra Signal objects it justifies. Revealed text is
    only ever scored, never executed or recursed into further."""
    analysis = analyze_transformations(raw_text)
    signals: list[Signal] = []

    if analysis.has_invisible_unicode:
        signals.append(
            Signal(
                name="invisible_unicode",
                description="invisible/zero-width Unicode formatting characters detected",
                weight=STRUCTURAL_ANOMALY_WEIGHT,
            )
        )

    if analysis.has_long_opaque_token:
        signals.append(
            Signal(
                name="long_opaque_token",
                description="unusually long, low-vowel opaque token detected",
                weight=STRUCTURAL_ANOMALY_WEIGHT,
            )
        )

    if not analysis.candidates:
        return signals

    # A transformed/encoded segment exists at all -- small signal only.
    signals.append(
        Signal(
            name="transform_detected",
            description=f"{analysis.candidates[0].kind} segment detected in input",
            weight=TRANSFORM_DETECTED_WEIGHT,
        )
    )

    any_candidate_suspicious = False
    for candidate in analysis.candidates:
        revealed_normalized = normalize_text(candidate.revealed_text)
        if not revealed_normalized:
            continue

        revealed_rule_signals = extract_signals(revealed_normalized)
        revealed_ml_prob = predict_proba_suspicious(pipeline, [revealed_normalized])[0]

        if revealed_rule_signals or revealed_ml_prob >= CANDIDATE_ML_THRESHOLD:
            any_candidate_suspicious = True
            break

    if any_candidate_suspicious:
        analysis_only = bool(_ANALYSIS_ONLY_INTENT_PATTERN.search(raw_text.lower())) and not analysis.has_follow_intent
        weight = ENCODED_MANIPULATION_ANALYSIS_ONLY_WEIGHT if analysis_only else ENCODED_MANIPULATION_WEIGHT
        description = (
            "decoded/transformed segment contains a manipulation-style phrase "
            "(reduced weight: request appears to ask for analysis only)"
            if analysis_only
            else "decoded/transformed segment contains an instruction-manipulation signal"
        )
        signals.append(
            Signal(
                name="encoded_instruction_manipulation",
                description=description,
                weight=weight,
            )
        )
        if analysis.has_follow_intent:
            signals.append(
                Signal(
                    name="follow_intent_on_transformed_content",
                    description="input asks the assistant to act on decoded/transformed content",
                    weight=FOLLOW_INTENT_WEIGHT,
                )
            )

    return signals


def analyze_text(pipeline: Pipeline, raw_text: str) -> RiskResult:
    """Run the full pipeline on a single raw input string.

    Handles empty input as an explicit edge case: an empty/whitespace-only
    prompt is classified as normal with a risk score of 0 and is never sent
    into the vectorizer.
    """
    if not raw_text or not raw_text.strip():
        return RiskResult(
            classification="normal",
            risk_score=0.0,
            ml_probability=0.0,
            signals=[],
            reasons=["empty input"],
        )

    normalized = normalize_text(raw_text)
    ml_probability = predict_proba_suspicious(pipeline, [normalized])[0]
    transform_signals = _transformation_signals(pipeline, raw_text)

    return compute_risk(normalized, ml_probability, extra_signals=transform_signals)


def analyze_batch(pipeline: Pipeline, raw_texts: list[str]) -> list[RiskResult]:
    """Batch version of analyze_text. Kept simple (loops analyze_text)
    rather than re-implementing batching for transformation analysis,
    since transformation checks are cheap regex/decoding operations."""
    return [analyze_text(pipeline, t) for t in raw_texts]
