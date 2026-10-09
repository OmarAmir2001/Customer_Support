"""The shape of the HTTP contract, independent of whether the agent works.

Rubric 02 asks for three specific things — ``sources[]`` that are citations rather
than chunk ids, a ``/health`` that reports ``documents_indexed``, and an endpoint
reachable as ``/ask``. Each is checked here against the real app objects, not
against a description of them, because all three are the kind of claim a README can
make while the code does something else.

No database, no model calls, no lifespan. The routers are mounted on a bare app and
the one dependency that would read configuration is overridden.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from customer_support.helpers.citations import citations_for
from customer_support.helpers.config import get_settings
from customer_support.models.db_schemas import RetrievedDocument
from customer_support.routers.chat import ask_router, chat_router
from customer_support.routers.health import base_router
from customer_support.routers.schemas.chat import ChatResponse


def _doc(**metadata) -> RetrievedDocument:
    return RetrievedDocument(text="...", score=0.5, metadata=metadata)


def _settings():
    """The get_settings override.

    Zero-argument on purpose: FastAPI reads the signature of a dependency override
    and turns any parameter it finds into a request parameter, so a `**overrides`
    convenience here made every call a 422 with no sign of why.
    """
    return SimpleNamespace(
        APP_NAME="test-app",
        APP_VERSION="9.9.9",
        KB_COLLECTION_NAME="collection_3",
    )


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(base_router)
    app.include_router(chat_router)
    app.include_router(ask_router)
    app.dependency_overrides[get_settings] = _settings
    return TestClient(app)


# --------------------------------------------------------------- sources[]


def test_citations_are_human_checkable_not_chunk_ids():
    """The distinction the rubric actually asks for.

    A chunk id identifies a row in our vector store. A citation names a section of a
    document the reader can open — so the id must not leak into the field even
    though it is sitting right there in the same metadata dict.
    """
    docs = [_doc(citation="CS_2023 — مادة (٢٤)", chunk_id="CS_2023::الفصل الثالث > مادة (٢٤)::0")]

    assert citations_for(docs) == ["CS_2023 — مادة (٢٤)"]
    assert "::" not in citations_for(docs)[0]


def test_citations_deduplicate_but_keep_retrieval_order():
    """Two parts of one section are one citation; the first one stays first.

    Order carries meaning — RetrievalController ranks handbook text ahead of promoted
    ticket answers, so sorting here would discard the precedence it establishes.
    """
    docs = [
        _doc(citation="CS_2023 — مادة (٢٤)"),
        _doc(citation="CS_2023 — مادة (٢٤)"),  # part 1 of the same section
        _doc(citation="CS_2023 — مادة (١٢)"),
    ]
    assert citations_for(docs) == ["CS_2023 — مادة (٢٤)", "CS_2023 — مادة (١٢)"]


def test_promoted_ticket_answer_cites_its_ticket():
    """A promoted answer has no handbook section, so it cites what it does have."""
    docs = [_doc(source="instructor_resolved", ticket_id="42", department="CS")]
    assert citations_for(docs) == ["Advisor-resolved ticket #42"]


def test_citation_is_built_from_source_and_section_when_corpus_field_is_absent():
    """The HTTP ingest path predates the corpus builder and writes no `citation`."""
    docs = [_doc(source="IS_2023", section="الفصل الثاني > مادة (٩)")]
    assert citations_for(docs) == ["IS_2023 — مادة (٩)"]


def test_unattributable_chunk_is_omitted_rather_than_guessed():
    """A citation a student cannot follow is worse than none: it looks like evidence."""
    assert citations_for([_doc(), _doc(section="(untitled)")]) == []


def test_chat_response_carries_sources_and_defaults_to_empty():
    response = ChatResponse(thread_id="t", answer="a", escalated=False)
    assert response.sources == []
    # Not a shared mutable default: two responses must not alias one list.
    response.sources.append("CS_2023 — مادة (٢٤)")
    assert ChatResponse(thread_id="t2", answer="a", escalated=False).sources == []


# ------------------------------------------------------------------ /health


def test_health_reports_documents_indexed():
    app = FastAPI()
    app.include_router(base_router)
    app.dependency_overrides[get_settings] = _settings

    class _Vectordb:
        async def get_collection_info(self, collection_name):
            assert collection_name == "collection_3"
            return {"record_count": 200}

    app.state.vectordb_client = _Vectordb()

    body = TestClient(app).get("/health").json()
    assert body["status"] == "healthy"
    assert body["documents_indexed"] == 200


def test_health_is_degraded_rather_than_lying_when_the_count_fails(client):
    """No vector store wired at all — the readiness probe must still answer."""
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["documents_indexed"] is None


def test_liveness_root_still_works(client):
    """`/` is what the compose healthcheck probes; adding /health must not move it."""
    body = client.get("/").json()
    assert body["APP_VERSION"] == "9.9.9"


# -------------------------------------------------------------------- /ask


def test_ask_is_the_same_handler_as_chat_not_a_copy(client):
    """One function on two paths.

    Asserted on the route table rather than by calling both, because two endpoints
    that merely *behave* the same today are exactly what drifts later.
    """
    # Asserted on the routers themselves rather than on app.routes: FastAPI 0.138
    # mounts an included router as one opaque _IncludedRouter object instead of
    # flattening its routes into the app, so app.routes shows neither path.
    chat_endpoints = {route.path: route.endpoint for route in chat_router.routes}
    ask_endpoints = {route.path: route.endpoint for route in ask_router.routes}

    assert ask_endpoints["/ask"] is chat_endpoints["/api/v1/chat"]

    # And both are really reachable, which the identity check alone would not prove.
    paths = client.app.openapi()["paths"]
    assert "/ask" in paths
    assert "/api/v1/chat" in paths


@pytest.mark.parametrize("path", ["/ask", "/api/v1/chat"])
def test_short_question_is_rejected_with_422_on_both_paths(client, path):
    """Validation happens before the handler, so this needs no graph.

    The checklist asks for it explicitly, and it is also what proves the alias is
    really the same handler: a hand-written second endpoint would need its own
    schema and could accept what this one rejects.
    """
    response = client.post(path, json={"question": "hi", "student_id": "s1"})
    assert response.status_code == 422


@pytest.mark.parametrize("path", ["/ask", "/api/v1/chat"])
def test_missing_student_id_is_rejected_with_422(client, path):
    assert client.post(path, json={"question": "a real question"}).status_code == 422
