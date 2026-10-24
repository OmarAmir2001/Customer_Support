"""Advisor and operator routes require a credential, and the actor comes from it.

The reported hole: nginx proxies every path on the one port bound to 0.0.0.0, none of
these routes authenticated, and `advisor_id` was a request field. So anyone who could
reach port 80 could claim a ticket, resolve it with `promote_to_kb: true`, and write
their own text into what future students are told — with `ticket_status_history`
recording whichever name they typed as fact. "Human-gated promotion" was gated by
nothing.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from customer_support.routers.escalation import escalation_router
from customer_support.routers.profile import profile_router

KEY = "test-key"
ADVISOR = "advisor_under_test"


class _TicketModel:
    def __init__(self):
        self.transitions = []

    async def get_ticket(self, _ticket_id):
        return SimpleNamespace(
            ticket_id=1,
            thread_id="t-1",
            status="pending",
            question="q",
            advisor_answer=None,
            department="CS",
            promoted_to_kb=False,
            promoted_at=None,
            resolved_by=None,
            student_id="s1",
            failed_gate="context_relevance",
            gate_reason="none found",
            created_at="2026-01-01T00:00:00Z",
        )

    async def list_tickets(self, **_kwargs):
        return []


class _Escalation:
    """Records the actor every lifecycle call was given."""

    def __init__(self):
        self.calls = []

    async def claim(self, ticket_id, advisor_id):
        self.calls.append(("claim", advisor_id))
        return await _TicketModel().get_ticket(ticket_id)


def build_app(*, keys=None):
    app = FastAPI()
    app.include_router(escalation_router)
    app.include_router(profile_router)
    app.state.settings = SimpleNamespace(
        ADVISOR_API_KEYS={KEY: ADVISOR} if keys is None else keys,
        ADVISOR_API_KEY_HEADER="X-API-Key",
    )
    app.state.ticket_model = _TicketModel()
    app.state.escalation_controller = _Escalation()
    app.state.profile_model = SimpleNamespace()
    app.state.memory = SimpleNamespace()
    return app


# --------------------------------------------------------------------- 401


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/v1/escalation/tickets"),
        ("get", "/api/v1/escalation/tickets/1"),
        ("post", "/api/v1/escalation/tickets/1/claim"),
        ("post", "/api/v1/escalation/tickets/1/resolve"),
        ("post", "/api/v1/escalation/tickets/1/reject"),
        ("post", "/api/v1/escalation/tickets/1/close"),
        ("post", "/api/v1/escalation/tickets/1/reopen"),
        ("post", "/api/v1/escalation/tickets/1/release"),
        ("get", "/api/v1/profile/202202"),
        ("delete", "/api/v1/profile/202202"),
    ],
)
def test_no_key_is_rejected_on_every_operator_route(method, path):
    """The test the review asked for, across every route rather than one."""
    client = TestClient(build_app())
    # Only POST takes a body; TestClient.get/delete reject a `json` kwarg.
    kwargs = {"json": {}} if method == "post" else {}
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 401, f"{method.upper()} {path} did not require a key"


def test_a_wrong_key_is_rejected():
    client = TestClient(build_app())
    response = client.get("/api/v1/escalation/tickets", headers={"X-API-Key": "not-the-key"})
    assert response.status_code == 401


def test_the_failure_does_not_reveal_whether_a_key_exists():
    """Identical responses for "no key" and "wrong key".

    Distinguishing them tells an attacker their guess was the right shape.
    """
    client = TestClient(build_app())
    absent = client.get("/api/v1/escalation/tickets")
    wrong = client.get("/api/v1/escalation/tickets", headers={"X-API-Key": "nope"})

    assert absent.status_code == wrong.status_code == 401
    assert absent.json() == wrong.json()


def test_an_empty_key_map_denies_everything():
    """Fail closed. A missing credential set must not read as "no auth required",
    which is how most of these ship open."""
    client = TestClient(build_app(keys={}))
    assert client.get("/api/v1/escalation/tickets").status_code == 401


# ---------------------------------------------------- the actor is the credential


def test_a_valid_key_is_accepted():
    client = TestClient(build_app())
    response = client.get("/api/v1/escalation/tickets", headers={"X-API-Key": KEY})
    assert response.status_code == 200


def test_the_recorded_actor_comes_from_the_key_not_the_body():
    """The impersonation that used to be possible.

    The body asks to act as someone else; the controller must still be told the
    identity the credential authenticated as.
    """
    app = build_app()
    client = TestClient(app)

    response = client.post(
        "/api/v1/escalation/tickets/1/claim",
        headers={"X-API-Key": KEY},
        json={"advisor_id": "someone_else_entirely"},
    )

    assert response.status_code == 200
    assert app.state.escalation_controller.calls == [("claim", ADVISOR)]


def test_the_request_bodies_no_longer_accept_an_actor():
    """Not just ignored — absent from the schema.

    An ignored-but-accepted field reads as supported, and the next person wires a UI
    to it.
    """
    from customer_support.routers.schemas.escalation import (
        CloseRequest,
        RejectRequest,
        ReleaseRequest,
        ReopenRequest,
        ResolveRequest,
    )

    for model in (ResolveRequest, ReleaseRequest, RejectRequest, CloseRequest, ReopenRequest):
        fields = set(model.model_fields)
        assert "advisor_id" not in fields, f"{model.__name__} still takes advisor_id"
        assert "actor" not in fields, f"{model.__name__} still takes actor"


def test_claim_needs_no_body_at_all():
    """The actor was the only thing in it."""
    client = TestClient(build_app())
    response = client.post("/api/v1/escalation/tickets/1/claim", headers={"X-API-Key": KEY})
    assert response.status_code == 200


# ------------------------------------------------------------ the student path


def test_the_student_chat_route_is_not_gated():
    """Students have no credential, so /chat must stay open.

    Gating it would be the wrong fix for the wrong problem — and the unauthenticated
    `student_id` there is a separate, recorded finding.
    """
    from customer_support.routers.chat import chat_router

    assert not chat_router.dependencies, "the student chat router must not require an advisor key"
