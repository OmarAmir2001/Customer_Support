"""Handbook precedence must survive a handbook the code has never heard of.

The reported bug: `RetrievalController` ranked by
`source.startswith(("CS_2023", "IS_2023"))`, so when the institute publishes a
CS_2025 handbook it falls through to rank 2 — BELOW `instructor_resolved` — silently
inverting the system's core rule that the handbook always outranks a promoted ticket
answer. Nothing errors. The learning loop just quietly starts overruling the
regulations.

Fixed by stamping `authority: "handbook"` at ingest and ranking on that, so a chunk
carries what it IS rather than having its nature re-derived from its name by every
reader.
"""

import pytest

from customer_support.controllers.RetrievalController import RetrievalController
from customer_support.models.db_schemas import RetrievedDocument


class FakeSettings:
    RETRIEVAL_OVERFETCH_FACTOR = 3
    ASSETS_DIR = "assets"
    HANDBOOK_SOURCES = {"CS_2023": "CS", "IS_2023": "IS", "CS_SAMPLE": "CS"}


class FakeEmbedding:
    def embed_text(self, text, document_type=None):
        return [[0.1] * 8]


class FakeVectorDB:
    def __init__(self, rows):
        self.rows = rows

    async def search_by_vector(self, collection_name, vector, limit):
        return self.rows[:limit]


def doc(source, score, *, authority=None, department="CS"):
    meta = {"source": source, "department": department}
    if authority is not None:
        meta["authority"] = authority
    return RetrievedDocument(text=f"from {source}", score=score, metadata=meta)


def controller(rows):
    return RetrievalController(
        vectordb_client=FakeVectorDB(rows),
        embedding_client=FakeEmbedding(),
        collection_name="collection_3",
        settings=FakeSettings(),
    )


@pytest.mark.asyncio
async def test_a_future_handbook_still_outranks_a_promoted_ticket_answer():
    """The reviewer's exact scenario: a 2025 handbook nobody hardcoded.

    Ranked above `instructor_resolved` because the chunk declares itself a handbook,
    not because its name matched a list someone remembered to update.
    """
    rows = [
        # The promoted answer retrieves BETTER — which is the whole reason precedence
        # exists: ticket answers are phrased in student language and out-retrieve
        # formal regulation text.
        doc("instructor_resolved", 0.95),
        doc("CS_2025", 0.80, authority="handbook"),
    ]
    chunks = await controller(rows).retrieve(question="how many hours?", department="CS")

    assert [c.metadata["source"] for c in chunks] == ["CS_2025", "instructor_resolved"]


@pytest.mark.asyncio
async def test_rows_indexed_before_authority_existed_still_rank_first():
    """The migration case, and why the legacy prefix check cannot be deleted yet.

    An existing collection holds handbook rows with no `authority` key. Ranking
    purely on that field would demote every one of them on the next request —
    causing the exact bug this change prevents.
    """
    rows = [
        doc("instructor_resolved", 0.95),
        doc("CS_2023", 0.80),  # no authority key, as an older index has
    ]
    chunks = await controller(rows).retrieve(question="how many hours?", department="CS")

    assert [c.metadata["source"] for c in chunks] == ["CS_2023", "instructor_resolved"]


@pytest.mark.asyncio
async def test_the_committed_sample_handbook_outranks_a_ticket_answer():
    """The samples are what a fresh clone demos with, so their precedence matters."""
    rows = [
        doc("instructor_resolved", 0.99),
        doc("CS_SAMPLE", 0.50, authority="handbook"),
    ]
    chunks = await controller(rows).retrieve(question="how many hours?", department="CS")

    assert chunks[0].metadata["source"] == "CS_SAMPLE"


@pytest.mark.asyncio
async def test_similarity_still_orders_within_the_handbook_group():
    """Precedence decides groups; relevance decides order inside one."""
    rows = [
        doc("CS_2023", 0.60, authority="handbook"),
        doc("CS_2025", 0.90, authority="handbook"),
    ]
    chunks = await controller(rows).retrieve(question="q", department="CS")

    assert [c.score for c in chunks] == [0.60, 0.90], (
        "the stable sort must preserve the vector store's similarity order"
    )


@pytest.mark.asyncio
async def test_an_unattributed_chunk_ranks_last():
    rows = [
        doc("something_else", 0.99),
        doc("instructor_resolved", 0.10),
        doc("CS_2023", 0.05, authority="handbook"),
    ]
    chunks = await controller(rows).retrieve(question="q", department="CS")

    assert [c.metadata["source"] for c in chunks] == [
        "CS_2023",
        "instructor_resolved",
        "something_else",
    ]


# ---------------------------------------------------------------- ingest side


def test_ingest_stamps_authority_on_handbook_chunks():
    """Retrieval can only rank on the field if ingestion writes it."""
    from customer_support.controllers.ProcessController import (
        HANDBOOK_AUTHORITY,
        ProcessController,
    )

    controller = ProcessController(project_id="1", settings=FakeSettings())
    metadata = controller._chunk_metadata("3EsFAHwA_CS_2023.md", {"h3": "مادة (٢٤)"})

    assert metadata["source"] == "CS_2023"
    assert metadata["department"] == "CS"
    assert metadata["authority"] == HANDBOOK_AUTHORITY


def test_a_source_added_to_settings_is_recognised_without_a_code_change():
    """The point of moving the map into configuration.

    Publishing next year's handbook should be an .env edit, not a patch in two
    files that have to agree.
    """
    from customer_support.controllers.ProcessController import ProcessController

    class _WithNextYear(FakeSettings):
        HANDBOOK_SOURCES = {"CS_2025": "CS"}

    controller = ProcessController(project_id="1", settings=_WithNextYear())
    metadata = controller._chunk_metadata("CS_2025.md", {})

    assert metadata["source"] == "CS_2025"
    assert metadata["department"] == "CS"
    assert metadata["authority"] == "handbook"


def test_a_non_handbook_file_gets_no_authority():
    """An arbitrary upload must not inherit handbook precedence."""
    from customer_support.controllers.ProcessController import ProcessController

    controller = ProcessController(project_id="1", settings=FakeSettings())
    metadata = controller._chunk_metadata("some_random_upload.pdf", {})

    assert metadata.get("authority") is None
    assert metadata["department"] is None
