from __future__ import annotations

import re

from .schemas import PolicyDecision

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,40}\b"
            r"(previous|prior|system|developer|instructions?|rules?|policy)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"\b(reveal|print|show|return|extract|leak)\b.{0,40}\b"
            r"(secret|password|credential|api[ -]?key|system prompt|token)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "tool_abuse",
        re.compile(
            r"\b(run|execute|launch|invoke)\b.{0,30}\b"
            r"(shell|command|powershell|bash|python code|arbitrary code)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "role_manipulation",
        re.compile(r"\b(you are now|act as|switch roles?)\b.{0,40}\b(unrestricted|admin|root|developer)\b", re.IGNORECASE),
    ),
)


def inspect_untrusted_text(text: str) -> PolicyDecision:
    """Check text against the injection rules and report the first rule that matches.

    Returns allowed=True when no rule matches.
    """
    for rule_name, pattern in _RULES:
        if pattern.search(text):
            return PolicyDecision(
                allowed=False,
                reason_code="prompt_injection_detected",
                matched_rule=rule_name,
            )
    return PolicyDecision(allowed=True, reason_code="allowed")

