"""Shared request-normalization and identifier parsing helpers."""

from __future__ import annotations

import re


def normalize_text(user_request: str) -> str:
    """Normalize whitespace and basic Russian spelling variations."""
    return re.sub(r"\s+", " ", str(user_request).lower().replace("ё", "е")).strip()


def extract_target_id(user_request: str) -> int | None:
    """Extract ``target_id`` / ``id`` / block number from a free-form request."""
    patterns = (
        r"target_id\s*[:=]?\s*(\d+)",
        r"\bid\s*[:=]?\s*(\d+)\b",
        r"\bквартал(?:а|у|ом)?\s*(?:с\s*)?(?:id\s*)?[:=]?\s*(\d+)\b",
        r"\bблок(?:а|у|ом)?\s*(?:с\s*)?(?:id\s*)?[:=]?\s*(\d+)\b",
        r"(квартал|блок)\D{0,20}(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, user_request, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(match.lastindex or 1)
        return int(value)
    return None
