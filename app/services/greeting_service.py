import re
from openai import OpenAI

client = OpenAI()

GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|sup|yo)[\s!.,]*$",
    r"^good\s+(morning|afternoon|evening|night)[\s!.,]*$",
    r"^what'?s\s+up[\s!.,]*$",
    r"^greetings[\s!.,]*$",
    r"^(thanks|thank you|thx)[\s!.,]*$",
]

QUESTION_STARTERS = {
    "what", "how", "when", "where", "why", "who", "which",
    "can", "could", "does", "is", "are", "tell", "explain", "describe"
}

GREETING_RESPONSE = "Hey! I'm Han's AI assistant. Feel free to ask me anything about his background, experience, or projects!"


def is_greeting_rule_based(message: str) -> bool:
    msg = message.strip().lower()
    return any(re.match(pattern, msg) for pattern in GREETING_PATTERNS)


def _needs_llm_check(message: str) -> bool:
    words = message.strip().lower().split()
    if not words:
        return False
    return len(words) < 8 and words[0] not in QUESTION_STARTERS


def _classify_with_llm(message: str) -> str:
    """Returns 'greeting' or 'question'. Sync — call via run_in_threadpool."""
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "Classify the user message as either 'greeting' (small talk, pleasantries, "
            "thanks, expressions with no real question) or 'question' (requests for "
            "information or help). Reply with only one word: greeting or question."
        ),
        input=message,
    )
    label = response.output_text.strip().lower()
    return "greeting" if label == "greeting" else "question"


def classify_message(message: str) -> str:
    """Returns 'greeting' or 'question'. Sync — call via run_in_threadpool."""
    if is_greeting_rule_based(message):
        return "greeting"
    if _needs_llm_check(message):
        return _classify_with_llm(message)
    return "question"
