"""
Splits a raw eligibility criteria blob into individual criterion sentences.

The ClinicalTrials.gov criteria field is a single free-text block mixing
inclusion and exclusion criteria under headers. This module segments it
into a list of discrete, classifiable criterion strings.

Handles the following real-world formatting patterns observed in the API:
  - Standard bullet format:  "* criterion text"
  - Numbered bullet format:  "1. criterion text"
  - Sub-points (indented):   "  * sub" or "  a. sub" — folded into parent
  - All-caps headers:        "INCLUSION CRITERIA:"
  - Lowercase headers:       "Inclusion criteria:"
  - Colon-optional headers:  "Exclusion Criteria" (no colon)
  - Inline headers:          "Inclusion Criteria:Text starts immediately"
  - Preamble sentences:      prose before first bullet — filtered out
  - Old-style headers:       "DISEASE CHARACTERISTICS:" — section = "unknown"
  - Escaped comparators:     "\\<" and "\\>" — cleaned to "<" and ">"
"""

import re


# ---------------------------------------------------------------------------
# Header patterns
# ---------------------------------------------------------------------------

# Matches "Inclusion Criteria", "INCLUSION CRITERIA:", "Inclusion Criterion",
# and bullet-prefixed variants like "* INCLUSION CRITERIA:"
_INCLUSION_RE = re.compile(
    r'^\s*[\*\-]?\s*inclusion\s+criteri[ao]n?s?\s*:?',
    re.IGNORECASE
)
_EXCLUSION_RE = re.compile(
    r'^\s*[\*\-]?\s*exclusion\s+criteri[ao]n?s?\s*:?',
    re.IGNORECASE
)
# Old-style section headers that don't map to inclusion/exclusion
_UNKNOWN_SECTION_RE = re.compile(
    r'^\s*(disease|patient|prior\s+concurrent|other)\s+(characteristics?|therapy|criteria)\s*:?',
    re.IGNORECASE
)

# A line is a top-level bullet if it starts with "* " or a number + period/dot
_TOP_BULLET_RE = re.compile(r'^\s{0,2}(\*|\d+[\.\)])\s+')

# A line is a sub-point if it's indented more than 2 spaces before a bullet
_SUB_BULLET_RE = re.compile(r'^\s{3,}(\*|[a-z\d][\.\)])\s+')

MIN_CRITERION_LENGTH = 10  # characters — shorter than this is noise


def _clean_text(text: str) -> str:
    """Clean API escape sequences and normalise whitespace."""
    text = text.replace('\\<', '<').replace('\\>', '>')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _is_header(line: str) -> tuple[str | None, str]:
    """
    Returns (section_label, remainder_text) if line is a section header,
    else (None, line).
    remainder_text is whatever text follows the header on the same line.
    """
    for pattern, label in [
        (_INCLUSION_RE, 'inclusion'),
        (_EXCLUSION_RE, 'exclusion'),
        (_UNKNOWN_SECTION_RE, 'unknown'),
    ]:
        m = pattern.match(line)
        if m:
            remainder = line[m.end():].strip()
            return label, remainder
    return None, line


def split_criteria(raw_text: str) -> list[dict]:
    """
    Split a raw eligibility criteria blob into individual criterion dicts.

    Each dict has:
        text     (str)  — the cleaned criterion sentence
        section  (str)  — "inclusion", "exclusion", or "unknown"
        position (int)  — 0-based index within the full criteria blob
    """
    if not raw_text:
        return []

    lines = raw_text.splitlines()

    current_section = 'unknown'
    current_bullet_lines: list[str] = []
    results: list[dict] = []
    position = 0

    def flush_bullet():
        nonlocal position
        if not current_bullet_lines:
            return
        combined = ' '.join(current_bullet_lines)
        cleaned = _clean_text(combined)
        if len(cleaned) >= MIN_CRITERION_LENGTH:
            results.append({
                'text': cleaned,
                'section': current_section,
                'position': position,
            })
            position += 1
        current_bullet_lines.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for a section header (possibly with inline text after it)
        label, remainder = _is_header(stripped)
        if label is not None:
            flush_bullet()
            current_section = label
            # If text follows immediately on the same line as the header, treat
            # it as the start of a new bullet only if it looks like one
            if remainder and _TOP_BULLET_RE.match(remainder):
                current_bullet_lines.append(remainder.lstrip('*0123456789.) '))
            continue

        # Sub-point — append to the current bullet
        if _SUB_BULLET_RE.match(line):
            sub_text = re.sub(r'^\s+(\*|[a-z\d][\.\)])\s+', '', line).strip()
            if current_bullet_lines:
                current_bullet_lines.append(sub_text)
            # If no current bullet yet, treat as top-level
            else:
                current_bullet_lines.append(sub_text)
            continue

        # Top-level bullet — flush previous, start new
        if _TOP_BULLET_RE.match(line):
            flush_bullet()
            bullet_text = re.sub(r'^\s{0,2}(\*|\d+[\.\)])\s+', '', line).strip()
            current_bullet_lines.append(bullet_text)
            continue

        # Continuation line (no bullet marker, not a header) — append to current
        if current_bullet_lines:
            current_bullet_lines.append(stripped)
            continue

        # Preamble prose before any bullet in this section — skip it

    flush_bullet()
    return results
