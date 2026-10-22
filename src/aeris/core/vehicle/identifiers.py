"""Shared identifier validation logic for the vehicle domain."""

import re


def validate_domain_identifier(v: str, field_name: str) -> str:
    """
    Enforce strict formatting for stable domain identifiers.

    Rules:
    - Trims whitespace.
    - Rejects empty strings.
    - Requires only alphanumeric characters, dashes, and underscores.
    """
    v_stripped = v.strip()
    if not v_stripped:
        raise ValueError(f"{field_name} cannot be empty or solely whitespace.")

    # Allow alphanumeric characters, dashes, and underscores
    if not re.match(r"^[A-Za-z0-9_-]+$", v_stripped):
        raise ValueError(
            f"{field_name} must contain only alphanumeric characters, dashes, and underscores."
        )
    return v_stripped
