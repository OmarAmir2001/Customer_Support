"""Retrieval ranking tests: department filtering and source precedence.

These are pure-logic tests with a fake vector store — no database, no embeddings.
The bug they exist to prevent is silent: retrieval returns chunks either way, and
nothing errors. It only shows up as "the learning loop never closes."
"""

import pytest

from customer_support.controllers.RetrievalController import RetrievalController
from customer_support.models.db_schemas import RetrievedDocument


class FakeSettings:
    RETRIEVAL_OVERFETCH_FACTOR = 3
    ASSETS_DIR = "assets"


class FakeEmbedding:
    def embed_text(self, text, document_type=None):
        return [[0.1] * 8]


class FakeVectorDB:
    """Returns rows in similarity order, as a real vector store does."""

    def __init__(self, rows):
        self.rows = rows

    async def search_by_vector(self, collection_name, vector, limit):
        return self.rows[:limit]


def doc(source, score, department="CS", text=None):
    return RetrievedDocument(
        text=text or f"chunk from {source}",
        score=score,
        metadata={"source": source, "department": department},
    )


def controller(rows):
    return RetrievalController(
        vectordb_client=FakeVectorDB(rows),
        embedding_client=FakeEmbedding(),
        collection_name="collection_1",
        settings=FakeSettings(),
    )


def sources(docs):
    return [(d.metadata or {}).get("source") for d in docs]


@pytest.mark.asyncio
async def test_other_department_is_filtered_out():
    """A CS student must never be answered from the IS handbook."""
    rows = [doc("IS_2023", 0.99, department="IS"), doc("CS_2023", 0.80)]

    result = await controller(rows).retrieve(question="q", department="CS", limit=5)

    assert sources(result) == ["CS_2023"]


@pytest.mark.asyncio
async def test_departmentless_chunks_are_shared_content():
    """Chunks with no department belong to everyone and must survive the filter."""
    rows = [doc("shared_policy", 0.9, department=None), doc("CS_2023", 0.8)]

    result = await controller(rows).retrieve(question="q", department="CS", limit=5)

    assert sources(result) == ["CS_2023", "shared_policy"]


@pytest.mark.asyncio
async def test_handbook_is_presented_before_a_ticket_answer():
    """Precedence: when both are present, handbook text comes first even though the
    ticket answer scored higher — ticket answers never override the handbook."""
    rows = [doc("instructor_resolved", 0.99), doc("CS_2023", 0.70)]

    result = await controller(rows).retrieve(question="q", department="CS", limit=5)

    assert sources(result) == ["CS_2023", "instructor_resolved"]


@pytest.mark.asyncio
async def test_a_promoted_answer_still_makes_the_cut_when_the_handbook_is_full():
    """The regression this file exists for.

    Ranking by source BEFORE truncating to `limit` starves instructor-resolved chunks:
    every handbook chunk sorts ahead of them, so once `limit` handbook chunks survive
    the filter a promoted ticket answer can never reach the generator, however well it
    matches. Section 5 wants ticket answers to FILL GAPS the handbook does not cover,
    so relevance must decide inclusion and precedence only the ordering.
    """
    # Five handbook chunks the vector store considers less similar than the promoted
    # answer — exactly the shape of "the handbook does not cover this question".
    rows = [doc("instructor_resolved", 0.95)] + [doc("CS_2023", 0.30 - i / 100) for i in range(5)]

    result = await controller(rows).retrieve(question="q", department="CS", limit=5)

    assert "instructor_resolved" in sources(result), (
        "a promoted ticket answer was ranked out of the results entirely; "
        "the learning loop cannot close"
    )
    # Precedence still holds among what was selected.
    assert sources(result)[0] == "CS_2023"


@pytest.mark.asyncio
async def test_empty_retrieval_returns_a_list_not_false():
    """Gate 1 turns "empty" into an escalation, so the type must stay honest."""
    result = await controller([]).retrieve(question="q", department="CS", limit=5)

    assert result == []


@pytest.mark.asyncio
async def test_blank_question_fails_loudly():
    with pytest.raises(ValueError):
        await controller([doc("CS_2023", 0.9)]).retrieve(question="   ", department="CS")
