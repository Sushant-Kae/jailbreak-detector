"""Transformation / obfuscation analysis.

Detects text that has been transformed in a way that could hide an
instruction-manipulation attempt from simple word-level matching:
Base64-like segments, letters spaced out one-per-character, and invisible/
unusual Unicode formatting characters.

IMPORTANT SAFETY NOTE: decoded/revealed text produced here is treated
strictly as *data to analyze* (fed back into the same rule-signal and ML
scoring used for ordinary text). It is never executed, never used to change
this program's own behavior, and this module never recurses into content
found inside an already-decoded candidate (decode depth is exactly 1).

This module intentionally does NOT decide suspicious/normal on its own —
it only reports what it found. src/pipeline.py decides how much weight to
give a transformation based on what the *revealed* content looks like.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass

# --- Base64 detection -------------------------------------------------

# A candidate must look like a real Base64 block: only base64 alphabet
# characters, length a multiple of 4 (with optional padding), and at
# least 16 characters -- short alnum tokens ("Hello123") are common
# English/code fragments and would otherwise cause false positives.
#
# Uses lookaround instead of \b: Python's \b does not treat "=" as a word
# boundary at end-of-string, which silently drops Base64 padding and
# breaks decoding.
_BASE64_CANDIDATE_PATTERN = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])")

# Words/phrases that reference decoding/transforming and often precede a
# request for the assistant to *act on* the revealed content, rather than
# just discuss or analyze it. Used only as a secondary weighting signal in
# src/pipeline.py, not to decide suspiciousness by itself.
FOLLOW_INTENT_PATTERN = re.compile(
    r"\b(and|then)\s+(follow|obey|comply with|execute|act on|do what it says)\b"
)

# Decode-related verbs, used to gate the follow-intent check so an
# unrelated "and follow up with X" doesn't get miscounted.
DECODE_CONTEXT_PATTERN = re.compile(
    r"\b(decode|decrypt|translate|convert|unscramble|interpret)\b"
)


@dataclass(frozen=True)
class TransformCandidate:
    kind: str  # "base64" or "spaced_letters"
    original_segment: str
    revealed_text: str


@dataclass(frozen=True)
class TransformAnalysis:
    candidates: list[TransformCandidate]
    has_invisible_unicode: bool
    has_long_opaque_token: bool
    has_follow_intent: bool


def _try_decode_base64(candidate: str) -> str | None:
    """Attempt to decode a candidate string as Base64.

    Returns the decoded text only if it decodes cleanly to mostly-printable
    UTF-8 text of a plausible length -- otherwise returns None. This keeps
    incidental alnum tokens (identifiers, hashes, etc.) from being treated
    as hidden instructions.
    """
    # Real Base64 content is a multiple of 4 characters long.
    if len(candidate) % 4 != 0:
        return None
    try:
        decoded_bytes = base64.b64decode(candidate, validate=True)
    except (binascii.Error, ValueError):
        return None

    try:
        decoded_text = decoded_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None

    if len(decoded_text) < 6:
        return None

    printable = sum(1 for ch in decoded_text if ch.isprintable())
    if printable / len(decoded_text) < 0.9:
        return None

    return decoded_text


def find_base64_candidates(raw_text: str) -> list[TransformCandidate]:
    """Find and safely decode Base64-like segments in raw (un-normalized)
    text. Case is preserved on purpose -- Base64 is case-sensitive, and
    lowercasing before this step would corrupt real Base64 content."""
    candidates: list[TransformCandidate] = []
    for match in _BASE64_CANDIDATE_PATTERN.finditer(raw_text):
        segment = match.group(0)
        decoded = _try_decode_base64(segment)
        if decoded is not None:
            candidates.append(
                TransformCandidate(kind="base64", original_segment=segment, revealed_text=decoded)
            )
    return candidates


# --- Spaced-out letters ("i g n o r e") --------------------------------

# Four or more single letters each separated by exactly one space is an
# uncommon, deliberate-looking pattern in ordinary prose (unlike "a b c"
# lists, which are short); it's a known simple technique for evading
# whole-word pattern matching.
_SPACED_LETTERS_PATTERN = re.compile(r"\b(?:[A-Za-z]\s){4,}[A-Za-z]\b")


def find_spaced_letter_candidates(raw_text: str) -> list[TransformCandidate]:
    """Find runs of single letters separated by spaces and collapse them
    back into words, so the revealed text can be analyzed normally.

    When multiple spaced-out words appear in the same text (common when
    an entire phrase is spaced out, with extra whitespace between words),
    an additional candidate joining all revealed words together is also
    returned -- otherwise each word is only checked in isolation and a
    multi-word rule pattern (e.g. "ignore ... instructions") could never
    match even though the full revealed phrase clearly would.
    """
    candidates: list[TransformCandidate] = []
    for match in _SPACED_LETTERS_PATTERN.finditer(raw_text):
        segment = match.group(0)
        revealed = segment.replace(" ", "")
        candidates.append(
            TransformCandidate(kind="spaced_letters", original_segment=segment, revealed_text=revealed)
        )

    if len(candidates) >= 2:
        joined_segment = " ".join(c.original_segment for c in candidates)
        joined_revealed = " ".join(c.revealed_text for c in candidates)
        candidates.append(
            TransformCandidate(kind="spaced_letters", original_segment=joined_segment, revealed_text=joined_revealed)
        )

    return candidates


# --- Structural anomalies (informational only, small weight) ----------

# Zero-width / invisible formatting characters are a known way to break up
# words so they don't match filters while still rendering normally.
_INVISIBLE_UNICODE_CHARS = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # zero-width no-break space / BOM
    "\u2060",  # word joiner
}


def has_invisible_unicode(raw_text: str) -> bool:
    return any(ch in _INVISIBLE_UNICODE_CHARS for ch in raw_text)


def has_long_opaque_token(raw_text: str) -> bool:
    """Flag a single very long token with no spaces and a low vowel ratio
    (e.g. a long opaque blob that isn't valid Base64 but also isn't
    ordinary prose). Purely structural, small weight only."""
    for token in raw_text.split():
        letters = [ch for ch in token if ch.isalpha()]
        if len(token) >= 40 and len(letters) >= 20:
            vowels = sum(1 for ch in letters if ch.lower() in "aeiou")
            if vowels / len(letters) < 0.15:
                return True
    return False


def analyze_transformations(raw_text: str) -> TransformAnalysis:
    """Run all transformation checks on raw (pre-normalization) text.

    `raw_text` should be the original, un-lowercased input, since Base64
    decoding depends on exact case.
    """
    candidates = find_base64_candidates(raw_text) + find_spaced_letter_candidates(raw_text)

    return TransformAnalysis(
        candidates=candidates,
        has_invisible_unicode=has_invisible_unicode(raw_text),
        has_long_opaque_token=has_long_opaque_token(raw_text),
        has_follow_intent=bool(
            FOLLOW_INTENT_PATTERN.search(raw_text.lower())
            or (
                DECODE_CONTEXT_PATTERN.search(raw_text.lower())
                and re.search(r"\b(and|then)\s+(follow|do what it says|comply)\b", raw_text.lower())
            )
        ),
    )
