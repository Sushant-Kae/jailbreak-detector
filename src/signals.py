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
#
# Design note on "instruction override" patterns: earlier versions used a
# fixed whitelist of modifier words between the verb and the noun (e.g.
# "ignore (all|any|previous|prior) instructions"). That was too brittle —
# "ignore all YOUR previous instructions" has an extra word ("your") the
# whitelist didn't cover, so it silently failed to match. The patterns below
# allow a short, bounded run of common modifier words (not an unbounded
# wildcard, to avoid matching unrelated third-person sentences like
# "override an AI's instructions" in an academic discussion) between the
# verb and the noun.
_OVERRIDE_VERBS = r"(?:ignore|disregard|forget|override|bypass|discard|erase)"
_OVERRIDE_MODIFIERS = (
    r"(?:all|any|every|previous|prior|earlier|original|further|additional|"
    r"your|my|our|the|this|these|current|existing|system)"
)
_OVERRIDE_NOUNS = r"(?:instructions?|rules?|guidelines?|directives?|configuration|training|restrictions?)"

_INSTRUCTION_OVERRIDE_PATTERNS = [
    rf"\b{_OVERRIDE_VERBS}\s+(?:{_OVERRIDE_MODIFIERS}\s+){{0,3}}{_OVERRIDE_NOUNS}\b",
    r"\bforget (everything|what) you (were told|know)\b",
    r"\bnew instructions? supersede\b",
    r"\btreat .*(instructions?|rules) as void\b",
    r"\bauthoriz(e|ing|ation) (to|you to) bypass\b",
    r"\bstop following the rules\b",
    r"\bset aside (every|any|all) (guideline|rule|instruction)s?\b",
]

_PERSONA_ROLEPLAY_PATTERNS = [
    r"\bact as an? (unrestricted|jailbroken|rule[- ]?free)\b",
    r"\bpretend you are\b.*\bno (rules|restrictions|limitations)\b",
    r"\bdeveloper mode\b",
    r"\bunrestricted mode\b",
    r"\bdo anything now\b",
    r"\byou are now dan\b",
]

# New category: attempts to claim elevated privilege/authority over the
# assistant, independent of exact wording ("accept me as administrator",
# "treat me as root", "I am your new administrator", etc).
_AUTHORITY_ESCALATION_PATTERNS = [
    r"\b(accept|treat|recognize) me as (an? |the |your )?(system )?(admin|administrator|root|superuser|operator)\b",
    r"\bgrant me (admin|administrator|root) (access|privileges|rights)\b",
    r"\bi am (now |)(your |the )?(new )?(admin|administrator|system operator|superuser)\b",
    r"\byou (must|will|shall) (obey|comply with) (only )?me\b",
    r"\b(elevate|escalate) my (privileges|access|permissions)\b",
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

# Structural signal: a directive-like fragment hidden inside brackets/braces
# alongside otherwise unrelated prose is a common way to smuggle an
# instruction-override attempt inside a normal-looking question (e.g.
# "is X okay? also {ignore all instructions and...}"). This catches the
# *structure* rather than requiring exact wording.
_DELIMITED_SEGMENT_PATTERN = re.compile(r"[\{\[<]([^\}\]>]{5,300})[\}\]>]")
_DELIMITED_TRIGGER_WORDS = re.compile(
    r"\b(ignore|disregard|override|bypass|admin|administrator|root|instructions?|"
    r"pretend|act as|dan|unrestricted|jailbreak|system prompt)\b"
)


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

    if _any_match(_AUTHORITY_ESCALATION_PATTERNS, normalized_text):
        signals.append(
            Signal(
                name="authority_escalation",
                description="authority/role-escalation signal detected",
                weight=0.35,
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

    # Structural signal: an override/authority-style word embedded inside a
    # bracketed/braced segment, separate from the surrounding sentence.
    # This targets the *structure* attackers use to smuggle a directive
    # alongside an unrelated, normal-looking question — it fires regardless
    # of the exact wording used inside the delimiters.
    for match in _DELIMITED_SEGMENT_PATTERN.finditer(normalized_text):
        if _DELIMITED_TRIGGER_WORDS.search(match.group(1)):
            signals.append(
                Signal(
                    name="delimited_directive",
                    description="directive-like text embedded in brackets/braces detected",
                    weight=0.30,
                )
            )
            break  # one signal is enough even if multiple segments match

    return signals
