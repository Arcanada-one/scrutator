"""Simple token estimation for chunking — no tiktoken needed."""


def token_count(text: str) -> int:
    """Estimate token count: word count * 1.3 (approximation for multilingual text)."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    if not text:
        return ""
    words = text.split()
    max_words = int(max_tokens / 1.3)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])
