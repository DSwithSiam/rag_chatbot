import re

INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"disregard\s+all\s+rules",
    r"system\s+prompt",
    r"developer\s+message",
    r"jailbreak",
    r"override\s+instructions",
]


def is_prompt_injection_attempt(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)
