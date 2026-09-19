"""Combine the ML classifier's probability with rule-based signals into a
single, documented, explainable risk score.

Scoring formula (intentionally simple):

    risk_score = clamp(
        ML_WEIGHT * ml_probability
        + sum(weight for each fired rule/authority/persona signal)
        + sum(weight for each fired transformation/obfuscation signal),
        0.0,
        1.0,
    )

All signal weights live next to their pattern definitions (src/signals.py
for text patterns, src/pipeline.py for transformation-derived signals) so
the full picture of "what contributes how much" stays close to the code
that decides *whether* something fired. This function just sums whatever
Signal objects it's given.

This is a heuristic risk indicator for triage, NOT a calibrated probability
that a jailbreak attempt would actually succeed against any real system.

Technique labeling: each fired signal is mapped to a broad technique
bucket (ROLEPLAY, EMOTIONAL_MANIPULATION, OBFUSCATION,
INSTRUCTION_SANDWICH, INSTRUCTION_OVERRIDE) purely to make the explanation
more useful -- it does NOT affect the risk score or classification, and
the main task remains binary normal vs. suspicious as required.
"""

from dataclasses import dataclass, field

from src.signals import Signal, extract_signals

ML_WEIGHT = 0.6
SUSPICIOUS_THRESHOLD = 0.5

# Maps a fired signal's name to a human-facing technique label. Signals
# not listed here (there shouldn't be any) fall back to "OTHER".
_SIGNAL_TO_TECHNIQUE = {
    "instruction_override": "INSTRUCTION_OVERRIDE",
    "authority_escalation": "INSTRUCTION_OVERRIDE",
    "delimited_directive": "INSTRUCTION_OVERRIDE",
    "stacked_imperatives": "INSTRUCTION_OVERRIDE",
    "persona_roleplay_bypass": "ROLEPLAY",
    "roleplay_persona_manipulation": "ROLEPLAY",
    "hypothetical_bypass": "ROLEPLAY",
    "emotional_manipulation": "EMOTIONAL_MANIPULATION",
    "instruction_sandwiching": "INSTRUCTION_SANDWICH",
    "obfuscation": "OBFUSCATION",
    "transform_detected": "OBFUSCATION",
    "encoded_instruction_manipulation": "OBFUSCATION",
    "follow_intent_on_transformed_content": "OBFUSCATION",
    "invisible_unicode": "OBFUSCATION",
    "long_opaque_token": "OBFUSCATION",
}


def _derive_technique(fired_signals: list[Signal]) -> str:
    """Best-effort technique label for the explanation, derived from
    which signal(s) fired. Purely informational (see module docstring)."""
    techniques = {_SIGNAL_TO_TECHNIQUE.get(s.name, "OTHER") for s in fired_signals}
    if not techniques:
        return "NORMAL"
    if len(techniques) > 1:
        return "MIXED"
    return techniques.pop()


@dataclass
class RiskResult:
    classification: str
    risk_score: float
    ml_probability: float
    signals: list[Signal] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    technique: str = "NORMAL"

    def to_dict(self) -> dict:
        return {
            "classification": self.classification,
            "risk_score": round(self.risk_score, 4),
            "ml_probability": round(self.ml_probability, 4),
            "signals": [s.name for s in self.signals],
            "reasons": self.reasons,
            "technique": self.technique,
        }


def compute_risk(
    normalized_text: str,
    ml_probability: float,
    extra_signals: list[Signal] | None = None,
) -> RiskResult:
    """Combine ML probability + rule signals (+ optional extra signals,
    e.g. from transformation/obfuscation analysis) into a RiskResult.

    `normalized_text` must already be normalized (see src.normalize) so
    signal patterns match consistently. `extra_signals` lets callers (like
    src/pipeline.py) fold in signals derived from decoded/revealed content
    without this function needing to know anything about transformations.
    """
    fired_signals = extract_signals(normalized_text) + list(extra_signals or [])
    signal_contribution = sum(s.weight for s in fired_signals)

    raw_score = ML_WEIGHT * ml_probability + signal_contribution
    risk_score = max(0.0, min(1.0, raw_score))

    classification = "suspicious" if risk_score >= SUSPICIOUS_THRESHOLD else "normal"

    reasons: list[str] = []
    if ml_probability >= 0.5:
        reasons.append(f"elevated classifier score ({ml_probability:.2f})")
    for s in fired_signals:
        reasons.append(s.description)
    if not reasons:
        reasons.append("no manipulation signals detected; low classifier score")

    technique = _derive_technique(fired_signals) if classification == "suspicious" else "NORMAL"

    return RiskResult(
        classification=classification,
        risk_score=risk_score,
        ml_probability=ml_probability,
        signals=fired_signals,
        reasons=reasons,
        technique=technique,
    )
