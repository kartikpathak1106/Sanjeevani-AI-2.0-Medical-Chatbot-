"""
Medical Safety Layer for Sanjeevani AI.
- Blocks clearly dangerous queries (self-harm, overdose instructions, etc.)
- Adds a standard medical disclaimer to every AI response.
This is a basic keyword-based filter — it is NOT a substitute for a real
content-moderation system, but it catches the obvious dangerous cases.
"""

import re

DISCLAIMER = (
    "\n\n---\n"
    "⚠️ *This is AI-generated information for educational purposes only. "
    "It is not a medical diagnosis. Please consult a qualified doctor for "
    "any real health concern.*"
)

CRISIS_MESSAGE = (
    "I can't help with that request. If you or someone you know is in "
    "distress or thinking about self-harm, please reach out right away:\n\n"
    "- **India — AASRA**: +91-9820466726 (24x7)\n"
    "- **India — iCall**: +91-9152987821\n"
    "- Or contact your nearest hospital emergency department.\n\n"
    "You deserve support from a real person — please talk to someone you trust or a professional."
)

# Keyword patterns that should never get a direct AI answer.
UNSAFE_PATTERNS = [
    r"\bhow (many|much).{0,40}(pills?|tablets?|dose).{0,40}(die|kill|overdose)",
    r"\bhow to (kill myself|end my life|commit suicide)",
    r"\bways? to (kill myself|harm myself|hurt myself)",
    r"\blethal dose\b",
    r"\bpainless way to die\b",
    r"\bsuicide method",
]


def check_safety(text: str):
    """
    Returns (is_safe: bool, message: str or None).
    If not safe, `message` is what should be shown to the user instead
    of passing the query to the LLM.
    """
    lowered = text.lower()
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, lowered):
            return False, CRISIS_MESSAGE
    return True, None


def add_disclaimer(answer: str) -> str:
    return answer + DISCLAIMER
