"""Rendering recent conversation turns into prompt text, under a hard budget.

The transcript exists so a thread is a conversation; this is what makes the model
actually read it. Both halves need bounding, and they are different problems:

* **Storage** is bounded by the reducer (``CONVERSATION_MAX_STORED_TURNS``).
* **Context** is bounded here, and much tighter. Storage cares about row size;
  context costs money and latency on every single request, and competes for space
  with the retrieved excerpts that the answer is actually grounded in.

Two limits, because one is not enough. A turn cap alone lets six long advisor
answers blow the budget; a character cap alone would admit fifty one-word turns and
spend the budget on noise. So: take the last N turns, then evict oldest-first until
the rendered block fits the character budget.

Why characters and not tokens: the provider's own truncation
(``INPUT_DEFAULT_MAX_CHARACTERS``) is measured in characters, so budgeting in the
same unit is what actually prevents the excerpts being silently cut out of the
prompt. A tokeniser would be more precise and would not protect against that.
"""

from customer_support.models.enums.MessageRoleEnum import MessageRole
from customer_support.models.graph.conversation import ConversationMessage

from .template_parser import TemplateParser

RAG_GROUP = "rag"

#: Roles rendered with their own label key in each locale.
_ROLE_KEYS = {
    MessageRole.STUDENT: "history_student",
    MessageRole.ASSISTANT: "history_assistant",
    MessageRole.ADVISOR: "history_advisor",
}


def _truncate(text: str, limit: int) -> str:
    """One long turn must not consume the whole budget on its own."""
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "…"


def format_history(
    messages: list[ConversationMessage],
    parser: TemplateParser,
    language: str | None = None,
    max_turns: int = 6,
    max_chars: int = 2000,
    max_chars_per_turn: int = 400,
) -> str:
    """Recent turns as prompt text, or "" when there is no usable history.

    Returns "" rather than an empty block for a first turn, so the answer prompt
    carries no history section at all instead of an awkward empty heading.
    """
    if not messages or max_turns <= 0 or max_chars <= 0:
        return ""

    # Newest-first while evicting, so the oldest turns are the ones dropped.
    recent = list(messages[-max_turns:])

    rendered: list[str] = []
    budget = max_chars

    for message in reversed(recent):
        key = _ROLE_KEYS.get(message.role)
        if key is None:
            continue

        line = parser.get(
            RAG_GROUP,
            key,
            {"content": _truncate(message.content, max_chars_per_turn)},
            language=language,
        )
        if not line:
            continue

        # +1 for the newline this line will be joined with.
        cost = len(line) + 1
        if cost > budget:
            # Budget spent. Everything older than this is dropped too, which keeps
            # the history contiguous — a transcript with a hole in the middle reads
            # as if turns were deleted rather than trimmed.
            break

        budget -= cost
        rendered.append(line)

    if not rendered:
        return ""

    rendered.reverse()  # back to chronological order for the model to read
    return "\n".join(rendered)
