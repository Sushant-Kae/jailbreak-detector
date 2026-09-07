"""Lightweight, explainable rule-based signals.

These are intentionally small and pattern-level (not an exhaustive attack
database). They exist to catch obvious structural cues and to give the
risk-scoring layer something human-readable to point at. They are combined
with the ML classifier's output rather than used alone.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    name: str
    description: str
    weight: float


# Each pattern is a small, generic regex representing a *category* of
# manipulation language, not a copy of any specific real-world payload.
_INSTRUCTION_OVERRIDE_PATTERNS = [
    r"\bignore (all |any |previous |prior )+instructions?\b",
    r"\bdisregard (all |any |previous |prior )+(rules|instructions?)\b",
    r"\bforget (everything|what) you (were told|know)\b",
    r"\boverride your (safety|guidelines|configuration|rules)\b",
    r"\bnew instructions? supersede\b",
    r"\btreat .*(instructions?|rules) as void\b",
    r"\bauthoriz(e|ing|ation) (to|you to) bypass\b",
]

_PERSONA_ROLEPLAY_PATTERNS = [
    r"\bact as an? (unrestricted|jailbroken|rule[- ]?free)\b",
    r"\bpretend you are\b.*\bno (rules|restrictions|limitations)\b",
    r"\bdeveloper mode\b",
    r"\bunrestricted mode\b",
    r"\bdo anything now\b",
    r"\byou are now dan\b",
]

_OBFUSCATION_PATTERNS = [
    r"\bbase64\b",
    r"\bleetspeak\b",
    r"\bfirst letter of every word\b",
    r"\brandom punctuation between letters\b",
    r"\bso (the )?(filter|moderation|detection)\b.*\b(can't|cannot|won't)\b",
]

_HYPOTHETICAL_BYPASS_PATTERNS = [
    r"\bhypothetically,? if you had no restrictions\b",
    r"\bin a fictional story with no real[- ]world rules\b",
    r"\bwhat would your unrestricted answer be\b",
]


def _any_match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def extract_signals(normalized_text: str) -> list[Signal]:
    """Return the list of rule-based signals that fired for this text.

    `normalized_text` should already be produced by normalize.normalize_text
    so behavior is consistent across the pipeline.
    """
    signals: list[Signal] = []

    if _any_match(_INSTRUCTION_OVERRIDE_PATTERNS, normalized_text):
        signals.append(
            Signal(
                name="instruction_override",
                description="instruction-manipulation signal detected",
                weight=0.35,
            )
        )

    if _any_match(_PERSONA_ROLEPLAY_PATTERNS, normalized_text):
        signals.append(
            Signal(
                name="persona_roleplay_bypass",
                description="persona/roleplay bypass signal detected",
                weight=0.30,
            )
        )

    if _any_match(_OBFUSCATION_PATTERNS, normalized_text):
        signals.append(
            Signal(
                name="obfuscation",
                description="obfuscation/evasion signal detected",
                weight=0.25,
            )
        )

    if _any_match(_HYPOTHETICAL_BYPASS_PATTERNS, normalized_text):
        signals.append(
            Signal(
                name="hypothetical_bypass",
                description="hypothetical/fictional-framing bypass signal detected",
                weight=0.20,
            )
        )

    # Simple structural signal: unusually long prompt with many imperative
    # verbs can indicate a stacked/compound manipulation attempt. Kept
    # intentionally coarse.
    word_count = len(normalized_text.split())
    imperative_hits = len(
        re.findall(r"\b(ignore|disregard|forget|override|act as|pretend|simulate)\b", normalized_text)
    )
    if word_count > 40 and imperative_hits >= 2:
        signals.append(
            Signal(
                name="stacked_imperatives",
                description="unusually long prompt with multiple stacked directives",
                weight=0.15,
            )
        )

    return signals
