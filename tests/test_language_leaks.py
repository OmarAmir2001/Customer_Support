"""Language must follow the reader, not the data.

Two leaks found by running the system, not by the suite — which is why they are
pinned here. Both had the same shape: a string whose language nobody chose, so it
picked the language of the *content* rather than the *reader*.

1. `escalation_reason` came back in Arabic for an English question with
   `language=en` explicitly set. The judge was told "write the reason in the same
   language as the question" and ignored it, because the prompt it was reading was
   dominated by Arabic handbook excerpts.
2. The holding message a student reads on escalation was a single Settings string,
   so an Arabic student got an English refusal.
"""

from string import Template

import pytest

from customer_support.stores.llm.templates import (
    JUDGE_SYSTEM_PROMPT,
    RAG_GROUP,
    TemplateParser,
    judge_system_prompt,
)

# ------------------------------------------------- the judge's reason language


def test_the_judge_is_told_a_named_language_not_asked_to_infer_one():
    """ "the same language as the question" is an inference; a name is an instruction.

    The model is reading a prompt full of Arabic excerpts. Asking it to work out
    which language the question was in is asking it to ignore the bulk of its own
    context, and it did not.
    """
    assert isinstance(JUDGE_SYSTEM_PROMPT, Template)
    assert "$reason_language" in JUDGE_SYSTEM_PROMPT.template
    assert "same language as the question" not in JUDGE_SYSTEM_PROMPT.template


@pytest.mark.parametrize(("language", "expected"), [("en", "English"), ("ar", "Arabic")])
def test_the_resolved_locale_names_the_reason_language(language, expected):
    prompt = judge_system_prompt(language)
    assert f"Write the reason in {expected}" in prompt
    # And it says to ignore the excerpts' language, which is what actually went wrong.
    assert "regardless of the language of the excerpts" in prompt


def test_an_unknown_locale_falls_back_to_english_rather_than_leaving_a_placeholder():
    """A raw `$reason_language` in a live prompt would be worse than a wrong language."""
    for value in (None, "", "fr", "zz"):
        prompt = judge_system_prompt(value)
        assert "Write the reason in English" in prompt
        assert "$reason_language" not in prompt


# ------------------------------------------- the student-facing holding message


@pytest.fixture
def templates():
    return TemplateParser(primary_language="en", default_language="en")


def test_the_escalation_message_exists_in_both_locales(templates):
    english = templates.get(RAG_GROUP, "escalation_message", {}, language="en")
    arabic = templates.get(RAG_GROUP, "escalation_message", {}, language="ar")

    assert english and arabic
    assert english != arabic, "both locales returned the same string — ar is not translated"


def test_the_arabic_message_is_actually_arabic(templates):
    """Parity only proves a key exists. This proves someone translated it.

    A copy-pasted English value would pass every structural check in
    test_locale_parity.py.
    """
    arabic = templates.get(RAG_GROUP, "escalation_message", {}, language="ar")
    arabic_letters = sum(1 for char in arabic if "؀" <= char <= "ۿ")
    assert arabic_letters > 10, f"no Arabic script in the Arabic message: {arabic!r}"

    english = templates.get(RAG_GROUP, "escalation_message", {}, language="en")
    assert not any("؀" <= char <= "ۿ" for char in english)


def test_the_message_does_not_promise_the_drafted_answer(templates):
    """An answer that failed a gate must never reach the student, not even hedged."""
    for language in ("en", "ar"):
        message = templates.get(RAG_GROUP, "escalation_message", {}, language=language)
        assert "$" not in message, "unsubstituted placeholder in student-facing copy"


# ----------------------------------------------------------- the wiring itself


@pytest.mark.asyncio
async def test_escalate_node_reads_the_locale_not_the_setting():
    """The node must consult the parser, so a locale change takes effect.

    Asserted through the node rather than by reading its source, because the bug
    being prevented is the node quietly going back to `settings.ESCALATION_...`.
    """
    from types import SimpleNamespace

    from customer_support.graph.nodes import make_escalate_node

    class _Escalation:
        async def create_ticket(self, state):
            return SimpleNamespace(ticket_id=99)

    deps = SimpleNamespace(
        escalation=_Escalation(),
        templates=TemplateParser(primary_language="en", default_language="en"),
        settings=SimpleNamespace(
            ESCALATION_STUDENT_MESSAGE="THE SETTING — must not be used when a template exists"
        ),
    )

    node = make_escalate_node(deps)
    result = await node({"question": "q", "language": "ar", "thread_id": "t"})

    assert result["ticket_id"] == 99
    assert "THE SETTING" not in result["answer"]
    assert any("؀" <= char <= "ۿ" for char in result["answer"]), result["answer"]


@pytest.mark.asyncio
async def test_escalate_node_falls_back_to_the_setting_when_a_template_is_missing():
    """Already the failure path — a missing template must not become a 503."""
    from types import SimpleNamespace

    from customer_support.graph.nodes import make_escalate_node

    class _Escalation:
        async def create_ticket(self, state):
            return SimpleNamespace(ticket_id=7)

    class _NoTemplates:
        def get(self, *_args, **_kwargs):
            return None

    deps = SimpleNamespace(
        escalation=_Escalation(),
        templates=_NoTemplates(),
        settings=SimpleNamespace(ESCALATION_STUDENT_MESSAGE="fallback copy"),
    )

    result = await make_escalate_node(deps)({"question": "q", "language": "en", "thread_id": "t"})
    assert result["answer"] == "fallback copy"
