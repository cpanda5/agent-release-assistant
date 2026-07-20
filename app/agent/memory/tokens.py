"""
Token estimation for context-window budget management.

tiktoken's cl100k_base is not calibrated for Qwen, but it's a reasonable
approximation — within ~10% for mixed Chinese/English text at the lengths
we use (hundreds to thousands of tokens). Accurate enough to prevent
ContextTooLong; we keep a 20% buffer anyway.
"""
import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _enc = None
    _TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available; using character-count fallback for token estimation")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE and _enc is not None:
        return len(_enc.encode(text))
    # Fallback: UTF-8 bytes / 3 is accurate to ~15% for mixed CJK+Latin text
    return max(1, len(text.encode("utf-8")) // 3)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate token cost of a messages list including role overhead."""
    total = 0
    for m in messages:
        total += estimate_tokens(m.get("content", ""))
        total += 4  # role + formatting overhead per message
    total += 2  # reply priming
    return total
