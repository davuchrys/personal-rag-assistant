"""Shared helpers for managing conversation context passed to the LLM."""


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token) — good enough for budgeting, not exact."""
    return max(1, len(text) // 4)


def truncate_history(chat_history: list[dict], max_tokens: int = 1500) -> list[dict]:
    """Keeps the most recent messages that fit within a token budget.

    Walks backward from the newest message and stops once adding another
    message would exceed max_tokens (always keeps at least the newest one).
    """
    if not chat_history:
        return []

    selected = []
    total = 0
    for msg in reversed(chat_history):
        role = msg.get("role", "")
        content = msg.get("content", "") if role == "user" else msg.get("answer", "")
        tokens = estimate_tokens(content)
        if selected and total + tokens > max_tokens:
            break
        selected.append(msg)
        total += tokens

    selected.reverse()
    return selected
