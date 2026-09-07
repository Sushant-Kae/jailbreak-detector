"""Text normalization used identically at train time and predict time.

Keeping normalization in one shared function is what prevents
train/predict skew (a common source of silent bugs in ML pipelines).
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize raw input text before feature extraction.

    Steps:
    1. Handle None / non-string input safely.
    2. Unicode-normalize (NFKC) to collapse look-alike characters.
    3. Lowercase.
    4. Collapse repeated whitespace.
    5. Strip leading/trailing whitespace.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
