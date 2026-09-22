"""Conversation history must be bounded in two independent places.

* **Storage** — the ``messages`` reducer trims, so a long-lived thread's checkpoint
  cannot grow forever.
* **Context** — the renderer budgets, so history cannot crowd the retrieved
  excerpts out of the answer prompt. That one is the expensive failure: the provider
  truncates the whole prompt at ``INPUT_DEFAULT_MAX_CHARACTERS``, so an unbounded
  history would silently drop the evidence the answer is supposed to be grounded in.

Both limits are tested here because both are invisible when they break — nothing
errors, the answers just quietly get worse.
"""

from customer_support.models.graph.conversation import (
    CONVERSATION_MAX_STORED_TURNS,
    ConversationMessage,
    append_and_trim,
)
from customer_support.stores.llm.templates import TemplateParser
from customer_support.stores.llm.templates.history import format_history

TEMPLATES = TemplateParser(primary_language="en", default_language="en")


def exchange(count: int) -> list[ConversationMessage]:
    """`count` student/assistant pairs, numbered so order is checkable."""
    turns: list[ConversationMessage] = []
    for index in range(count):
        turns.append(ConversationMessage.student(f"question {index}"))
        turns.append(ConversationMessage.assistant(f"answer {index}"))
    return turns


# --------------------------------------------------------------- the reducer


def test_the_reducer_appends_below_the_cap():
    existing = exchange(2)
    result = append_and_trim(existing, [ConversationMessage.student("next")])

    assert len(result) == 5
    assert result[-1].content == "next"


def test_the_reducer_trims_to_the_cap():
    """Without this the checkpoint grows forever, and every run rewrites the whole
    list."""
    far_too_many = exchange(CONVERSATION_MAX_STORED_TURNS)  # twice the cap

    result = append_and_trim(far_too_many, [ConversationMessage.student("newest")])

    assert len(result) == CONVERSATION_MAX_STORED_TURNS
    # Oldest-first eviction: the newest turn survives, turn 0 does not.
    assert result[-1].content == "newest"
    assert all(message.content != "question 0" for message in result)


def test_the_reducer_handles_an_empty_starting_state():
    """A brand new thread has no messages key at all."""
    result = append_and_trim(None, [ConversationMessage.student("first")])
    assert [m.content for m in result] == ["first"]


def test_the_reducer_preserves_chronological_order():
    result = append_and_trim(exchange(1), exchange(1))
    assert [m.content for m in result] == ["question 0", "answer 0", "question 0", "answer 0"]


# -------------------------------------------------------------- the renderer


def test_no_history_renders_as_empty_string():
    """A first turn must produce no history section, not an empty heading."""
    assert format_history([], TEMPLATES) == ""


def test_turns_are_labelled_by_role():
    rendered = format_history(
        [
            ConversationMessage.student("how many hours?"),
            ConversationMessage.assistant("135 credit hours."),
            ConversationMessage.advisor("The fee is 300 EGP.", ticket_id=1, author="nour"),
        ],
        TEMPLATES,
    )

    assert "Student: how many hours?" in rendered
    assert "You: 135 credit hours." in rendered
    # An advisor's turn is labelled distinctly: it is authoritative in a way the
    # assistant's own earlier turn is not.
    assert "Academic advisor: The fee is 300 EGP." in rendered


def test_only_the_most_recent_turns_are_kept():
    rendered = format_history(exchange(10), TEMPLATES, max_turns=4)

    assert "question 9" in rendered
    assert "question 0" not in rendered
    assert len(rendered.splitlines()) == 4


def test_the_character_budget_evicts_oldest_first():
    """A turn cap alone is not enough — six long advisor answers would still blow
    the prompt. The char budget is the backstop."""
    long_turns = [ConversationMessage.assistant("x" * 300) for _ in range(6)]

    rendered = format_history(
        long_turns, TEMPLATES, max_turns=6, max_chars=400, max_chars_per_turn=300
    )

    assert len(rendered) <= 400
    assert rendered  # not empty: at least the newest turn fits


def test_one_enormous_turn_cannot_eat_the_whole_budget():
    rendered = format_history(
        [ConversationMessage.student("y" * 5000)],
        TEMPLATES,
        max_chars=2000,
        max_chars_per_turn=120,
    )

    assert len(rendered) < 200
    assert rendered.endswith("…")  # truncated, and visibly so


def test_history_renders_in_the_requested_locale():
    """The labels are student-facing text, so they follow the resolved locale."""
    messages = [ConversationMessage.student("كم عدد الساعات؟")]

    arabic = format_history(messages, TEMPLATES, language="ar")
    english = format_history(messages, TEMPLATES, language="en")

    assert arabic.startswith("الطالب:")
    assert english.startswith("Student:")


def test_a_zero_budget_disables_history_entirely():
    """An operator turning history off must get no history, not a crash."""
    assert format_history(exchange(3), TEMPLATES, max_turns=0) == ""
    assert format_history(exchange(3), TEMPLATES, max_chars=0) == ""


def test_whitespace_is_collapsed():
    """Multi-line answers are flattened so one turn stays one line and the budget
    arithmetic holds."""
    rendered = format_history(
        [ConversationMessage.assistant("line one\n\nline  two")], TEMPLATES
    )

    assert rendered.count("\n") == 0
    assert "line one line two" in rendered
