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
# Word separator that tolerates excessive punctuation/separators used as a
# simple obfuscation technique ("ignore, all, your, previous, instructions"),
# not just whitespace.
_SEP = r"[\s,;]+"

_INSTRUCTION_OVERRIDE_PATTERNS = [
    rf"\b{_OVERRIDE_VERBS}{_SEP}(?:{_OVERRIDE_MODIFIERS}{_SEP}){{0,3}}{_OVERRIDE_NOUNS}\b",
    r"\bforget (everything|what) you (were told|know)\b",
    r"\bnew instructions? supersede\b",
    r"\btreat .*(instructions?|rules) as void\b",
    r"\bauthoriz(e|ing|ation) (to|you to) bypass\b",
    r"\bstop following the rules\b",
    r"\bset aside (every|any|all) (guideline|rule|instruction)s?\b",
    # Same concept, reversed word order ("set your rules aside").
    r"\bset (your |)(rules?|guidelines?|instructions?) aside\b",
]

_PERSONA_ROLEPLAY_PATTERNS = [
    r"\bact as an? (unrestricted|jailbroken|rule[- ]?free)\b",
    r"\bpretend you are\b.*\bno (rules|restrictions|limitations)\b",
    r"\bdeveloper mode\b",
    r"\bunrestricted mode\b",
    r"\bdo anything now\b",
    r"\byou are now dan\b",
]

# --- Roleplay / persona manipulation (combination-based) ---------------
#
# Roleplay itself is completely ordinary ("act as a history professor")
# and must NOT be flagged alone. The signal below only fires when BOTH
# (a) the text asks the assistant to adopt a persona/fictional role, AND
# (b) that persona is explicitly framed as having no rules/policies, so
# the combination -- not roleplay by itself -- is what's suspicious.
_ROLEPLAY_FRAMING_PATTERNS = [
    r"\b(act as|roleplay as|speak as|pretend (that )?you are|imagine you are|"
    r"play the (role|character|persona) of|take on the (role|character|persona) of|"
    r"become the (role|character|persona) of|you are now an?)\b",
    r"\blet'?s roleplay\b",
]
_RESTRICTION_FREE_CLAIM_PATTERNS = [
    r"\b(no|without any|has no) (\w+\s+){0,2}"
    r"(polic(y|ies)|restrictions?|rules?|guidelines?|filters?|limits?|constraints?)\b",
    r"\bcan (answer|do|say) anything\b",
    r"\b(answer|answers|answering|do|does|doing|say|says|saying)\s+anything\b",
    r"\bmust answer any question\b",
    r"\bdifferent rules? (than|from) (you|your normal)\b",
    r"\bwithout (any )?restrictions?\b",
    # "operates/exists/lives/functions outside every guideline/rule/policy"
    r"\b(operates?|exists?|lives?|functions?)\s+outside\s+(any|every|all)"
    r"\s+(of (your|its|these) )?(guidelines?|rules?|polic(y|ies)|restrictions?)\b",
]

# --- Emotional manipulation (combination-based) -------------------------
#
# Emotional language itself is completely ordinary and must NOT be
# flagged alone (grief, nostalgia, and urgency are normal parts of human
# communication). The signal below only fires when emotional framing is
# combined with an attempt to obtain restricted behavior -- reusing the
# instruction-override / restriction-free-claim / authority-escalation
# patterns already defined above as the "attempted bypass" side.
_EMOTIONAL_FRAMING_PATTERNS = [
    r"\b(late |dying |dear |beloved )?(grandmother|grandma|grandpa|grandfather|mother|father|mom|dad)"
    r"\b.{0,20}\b(used to|would always|told me|always said)\b",
    r"\bbefore (she|he) (died|passed away)\b",
    r"\bpassed away (last|this) (year|month|week)\b",
    r"\bi miss (her|him|them) so much\b",
    r"\bplease,? i'?m begging you\b",
    r"\byou'?re the only one who can help\b",
    r"\bi (don'?t|do not) have anyone else to (ask|turn to)\b",
    r"\bnowhere else to turn\b",
    r"\bjust this once,? please\b",
    r"\bi'?ll be (devastated|heartbroken) if\b",
    r"\bit would mean (so much|the world) to me\b",
]

# A generic "asking for content that is restricted/forbidden" phrase, used
# as one of the bypass-side signals for the emotional-manipulation combo
# below (an emotionally-framed request can imply wanting restricted
# content without using an explicit override verb like "ignore").
_RESTRICTED_CONTENT_REQUEST_PATTERNS = [
    r"\b(restricted|forbidden|off-?limits|banned|prohibited)\s+(things?|content|information|instructions?|topics?)\b",
]

# --- Instruction sandwiching / multi-step escalation ---------------------
#
# A sequence of individually-reasonable steps is completely ordinary
# (tutorials, recipes, lesson plans) and must NOT be flagged alone. The
# signal below only fires when a multi-step/enumerated structure is
# combined with a final step asking for restricted operational output --
# the escalation pattern, not step-by-step writing in general.
_SEQUENCE_MARKER_PATTERN = re.compile(
    r"\b(first|second|third|fourth|fifth|next|then|after that|finally|lastly|"
    r"start(?:ing)? by|begin(?:ning)? with)\b",
    re.IGNORECASE,
)
_OPERATIONAL_REQUEST_PATTERNS = [
    r"\bwrite (a |the )?(working |functional )?(exploit|malware|virus|ransomware)\b",
    r"\bprovide (the )?(exact|full|complete) (steps|instructions|code) to\b",
    r"\b(give|provide) me (a |the )?(step[- ]by[- ]step )?(guide|instructions) to (bypass|hack|exploit|attack)\b",
    r"\b(write|generate) (operational|working) (attack|exploit) code\b",
    r"\bprovide operational instructions for\b",
    r"\b(hand|give|send) me a (ready-to-use |working |functional )?(payload|exploit)\b",
    r"\b(payload|exploit) that bypasses\b",
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
    # Note: this specifically targets asking the assistant to ENCODE its
    # own output in Base64 to evade a filter -- a real evasion technique.
    # Merely mentioning Base64 (e.g. "decode this Base64 string") is NOT
    # itself suspicious and must not match here; that case is handled by
    # src/transform_analysis.py + src/pipeline.py, which only escalate
    # risk if the *decoded* content is itself suspicious.
    r"\bencode\b[\w\s]{0,30}\bin base64\b",
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

    if _any_match(_ROLEPLAY_FRAMING_PATTERNS, normalized_text) and _any_match(
        _RESTRICTION_FREE_CLAIM_PATTERNS, normalized_text
    ):
        signals.append(
            Signal(
                name="roleplay_persona_manipulation",
                description="roleplay/persona framing combined with a restriction-free claim detected",
                weight=0.35,
            )
        )

    if _any_match(_EMOTIONAL_FRAMING_PATTERNS, normalized_text) and (
        _any_match(_INSTRUCTION_OVERRIDE_PATTERNS, normalized_text)
        or _any_match(_RESTRICTION_FREE_CLAIM_PATTERNS, normalized_text)
        or _any_match(_AUTHORITY_ESCALATION_PATTERNS, normalized_text)
        or _any_match(_RESTRICTED_CONTENT_REQUEST_PATTERNS, normalized_text)
    ):
        signals.append(
            Signal(
                name="emotional_manipulation",
                description="emotional framing combined with an attempt to bypass restrictions detected",
                weight=0.35,
            )
        )

    if (
        len(_SEQUENCE_MARKER_PATTERN.findall(normalized_text)) >= 3
        and _any_match(_OPERATIONAL_REQUEST_PATTERNS, normalized_text)
    ):
        signals.append(
            Signal(
                name="instruction_sandwiching",
                description="multi-step structure escalating to a restricted operational request detected",
                weight=0.40,
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
