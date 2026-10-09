"""The streaming path, and the one property that keeps it honest.

Streaming exists because a long answer feels better arriving in pieces. It costs
something real: the post-generation gates cannot run, because faithfulness and
answer relevance can only be scored on a COMPLETE answer, and by then the student
has already read it. So a streamed answer is an unverified answer, and `/chat`
remains the gated default.

What these tests pin down is that the streamed answer is built from the SAME prompt
as the gated one. Two copies of that assembly would diverge the first time a section
was added, and the streamed answer would then be grounded differently than the
answer the judges were tuned against — with nothing to reveal it.
"""

import pytest

from customer_support.controllers.GenerationController import GenerationController
from customer_support.models.db_schemas import RetrievedDocument
from customer_support.stores.llm.templates import TemplateParser

TEMPLATES = TemplateParser(primary_language="en", default_language="en")


class FakeSettings:
    ASSISTANT_NAME = "Murshid"
    GENERATION_DEFAULT_MAX_TOKENS = 800
    GENERATION_DEFAULT_TEMPERATURE = 0.1
    CONVERSATION_HISTORY_TURNS = 6
    CONVERSATION_HISTORY_MAX_CHARS = 2000
    CONVERSATION_HISTORY_MAX_CHARS_PER_TURN = 400
    ASSETS_DIR = "assets"


class FakeClient:
    """Records what it was asked to stream, and in how many pieces."""

    def __init__(self, pieces=("135 ", "credit ", "hours.")):
        self.pieces = pieces
        self.streamed_prompt = None
        self.blocking_prompt = None

    def construct_prompt(self, prompt, role):
        return {"role": role, "content": prompt}

    def generate_text(self, prompt, chat_history=None, max_output_tokens=None, temperature=None):
        self.blocking_prompt = prompt
        return "".join(self.pieces)

    def stream_text(self, prompt, chat_history=None, max_output_tokens=None, temperature=None):
        self.streamed_prompt = prompt
        yield from self.pieces


def chunk(text="A student needs 135 credit hours.", source="CS_2023"):
    return RetrievedDocument(
        text=text, score=0.9, metadata={"source": source, "section": "Article 24"}
    )


def controller(client=None):
    return GenerationController(
        generation_client=client or FakeClient(), templates=TEMPLATES, settings=FakeSettings()
    )


def test_streaming_yields_pieces_not_one_block():
    client = FakeClient(pieces=("a", "b", "c"))
    pieces = list(controller(client).stream_answer(question="q", chunks=[chunk()]))

    assert pieces == ["a", "b", "c"], "the generator must pass pieces through unjoined"


@pytest.mark.asyncio
async def test_the_streamed_and_gated_paths_send_an_identical_prompt():
    """The property this file exists for.

    If these ever differ, the streamed answer is grounded in different evidence than
    the one the judges scored — and nothing anywhere would say so.
    """
    client = FakeClient()
    gen = controller(client)
    chunks = [chunk(), chunk("Warnings are issued below 2.0.", "CS_2023")]

    await gen.generate_answer(question="how many hours?", chunks=chunks)
    list(gen.stream_answer(question="how many hours?", chunks=chunks))

    assert client.blocking_prompt is not None
    assert client.streamed_prompt == client.blocking_prompt


def test_build_prompt_includes_the_excerpts_and_ends_with_the_question():
    """The footer stays last on purpose — an instruction at the very end of a long
    prompt is followed more reliably than the same one in the system message."""
    system, prompt = controller().build_prompt(question="how many credit hours?", chunks=[chunk()])

    assert "Murshid" in system
    assert "135 credit hours" in prompt
    assert prompt.rstrip().endswith("?") or "how many credit hours?" in prompt[-400:]


def test_streaming_refuses_an_empty_context_rather_than_inventing_one():
    """Mirrors generate_answer. Reaching here with no chunks means gate 1 did not run,
    which is a wiring bug and should be loud."""
    with pytest.raises(ValueError):
        list(controller().stream_answer(question="q", chunks=[]))


def test_a_provider_without_streaming_still_works():
    """`stream_text` is deliberately NOT abstract: streaming is an optimisation of a
    user's wait, not part of what makes a provider usable. The default yields the
    whole answer as one chunk so callers can always treat streaming as available."""
    from customer_support.stores.llm.LLMInterface import LLMInterface

    class MinimalProvider(LLMInterface):
        def set_generation_model(self, model_id): ...
        def set_embedding_model(self, model_id, embedding_size): ...
        def generate_text(
            self, prompt, chat_history=None, max_output_tokens=None, temperature=None
        ):
            return "the whole answer"

        def generate_json(
            self, prompt, system_prompt=None, max_output_tokens=None, temperature=None
        ):
            return "{}"

        def embed_text(self, text, document_type=None):
            return [[0.0]]

        def construct_prompt(self, prompt, role):
            return {"role": role, "content": prompt}

    assert list(MinimalProvider().stream_text("q")) == ["the whole answer"]
