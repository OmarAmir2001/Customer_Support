"""The HTTP client for the backend. The only module that knows the API exists.

Three rules shaped it:

**Every failure becomes one readable sentence.** A UI that renders a stack trace or a
raw `httpx.ConnectError` teaches the viewer nothing. `ApiError` carries text a person
can act on — "the API is not reachable at http://...; is the stack up?" — and every
view renders `str(exc)` without needing to know what went wrong.

**It is synchronous.** Gradio event handlers run in a worker thread, so blocking I/O
there is correct and an async client would need its own loop management for no gain.
The one exception is streaming, which is a generator and streams fine synchronously.

**It knows the API's error contract.** The backend answers 404 for a missing ticket,
409 for a refused transition *or* a lost race, 422 for a validation failure. Those
mean different things to an advisor, so they are translated rather than collapsed
into "request failed".
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from settings import settings


class ApiError(RuntimeError):
    """Anything that went wrong, already phrased for a human."""


class ApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        stream_base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.api_base_url).rstrip("/")
        self.stream_base_url = (stream_base_url or settings.stream_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout

    # ------------------------------------------------------------- internals

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiError(
                f"The API is not reachable at {self.base_url}. Is the stack up? "
                f"Try `make up`, then `make ps`. ({exc.__class__.__name__})"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ApiError(
                f"The API did not answer within {self.timeout:.0f}s. A cold question "
                "runs retrieval plus up to three judge calls, so this usually means a "
                "rate-limited or unreachable model provider — check the app logs."
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"HTTP error talking to the API: {exc}") from exc

        return self._unwrap(response)

    @staticmethod
    def _unwrap(response: httpx.Response) -> Any:
        if response.is_success:
            if not response.content:
                return None
            try:
                return response.json()
            except ValueError:
                return response.text

        detail = ""
        try:
            body = response.json()
            detail = body.get("detail") if isinstance(body, dict) else str(body)
            if isinstance(detail, list):  # FastAPI validation errors
                detail = "; ".join(
                    f"{'.'.join(str(p) for p in item.get('loc', []))}: {item.get('msg', '')}"
                    for item in detail
                )
        except ValueError:
            detail = response.text[:400]

        # Translated, not collapsed: these mean different things to an advisor.
        if response.status_code == 404:
            raise ApiError(f"Not found. {detail}")
        if response.status_code == 409:
            raise ApiError(
                "Refused (409). Either the ticket is not in a state that allows this "
                f"move, or another advisor got there first — reload the queue. {detail}"
            )
        if response.status_code == 422:
            raise ApiError(f"The API rejected the input (422). {detail}")
        if response.status_code == 503:
            raise ApiError(
                "The assistant is temporarily unavailable (503). The graph run failed — "
                f"usually the model provider. {detail}"
            )
        raise ApiError(f"API returned {response.status_code}. {detail}")

    # ----------------------------------------------------------------- chat

    def ask(
        self,
        question: str,
        student_id: str,
        department: str | None = None,
        language: str | None = None,
        thread_id: str | None = None,
    ) -> dict:
        """The GATED path. Returns `{thread_id, answer, sources, escalated, ...}`."""
        payload: dict[str, Any] = {"question": question, "student_id": student_id}
        if department:
            payload["department"] = department
        if language:
            payload["language"] = language
        if thread_id:
            payload["thread_id"] = thread_id
        return self._request("POST", "/api/v1/chat", json=payload)

    def stream(
        self,
        question: str,
        department: str | None = None,
        language: str | None = None,
    ) -> Iterator[str]:
        """The UNGATED path, on the BentoML service.

        A generator, so Gradio can render tokens as they arrive. Note this hits a
        different base url and a different process; it is optional, and a connection
        error here must not look like the whole system being down.
        """
        payload: dict[str, Any] = {"question": question}
        if department:
            payload["department"] = department
        if language:
            payload["language"] = language

        url = f"{self.stream_base_url}/ask"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, json=payload) as response:
                    if not response.is_success:
                        response.read()
                        raise ApiError(
                            f"The streaming service returned {response.status_code}. "
                            f"{response.text[:300]}"
                        )
                    for piece in response.iter_text():
                        if piece:
                            yield piece
        except httpx.ConnectError as exc:
            raise ApiError(
                f"The streaming service is not reachable at {self.stream_base_url}. "
                "It is a separate process — start it with "
                "`uv run --extra dev bentoml serve "
                "customer_support.bento_service:HandbookAssistant --port 3001`. "
                "The verified path does not need it."
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"Streaming failed: {exc}") from exc

    def conversation(self, thread_id: str) -> dict:
        return self._request("GET", f"/api/v1/chat/{thread_id}")

    # ------------------------------------------------------------- tickets

    def list_tickets(
        self,
        ticket_status: str | None = None,
        department: str | None = None,
        promotion_held: bool | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> list[dict]:
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if ticket_status:
            params["ticket_status"] = ticket_status
        if department:
            params["department"] = department
        if promotion_held is not None:
            params["promotion_held"] = str(promotion_held).lower()
        return self._request("GET", "/api/v1/escalation/tickets", params=params) or []

    def ticket(self, ticket_id: int) -> dict:
        return self._request("GET", f"/api/v1/escalation/tickets/{ticket_id}")

    def claim(self, ticket_id: int, advisor_id: str) -> dict:
        return self._request(
            "POST",
            f"/api/v1/escalation/tickets/{ticket_id}/claim",
            json={"advisor_id": advisor_id},
        )

    def release(self, ticket_id: int, advisor_id: str, note: str | None = None) -> dict:
        return self._request(
            "POST",
            f"/api/v1/escalation/tickets/{ticket_id}/release",
            json={"advisor_id": advisor_id, "note": note or None},
        )

    def resolve(
        self, ticket_id: int, advisor_id: str, advisor_answer: str, promote_to_kb: bool
    ) -> dict:
        return self._request(
            "POST",
            f"/api/v1/escalation/tickets/{ticket_id}/resolve",
            json={
                "advisor_id": advisor_id,
                "advisor_answer": advisor_answer,
                "promote_to_kb": promote_to_kb,
            },
        )

    def reject(self, ticket_id: int, advisor_id: str, reason: str) -> dict:
        return self._request(
            "POST",
            f"/api/v1/escalation/tickets/{ticket_id}/reject",
            json={"advisor_id": advisor_id, "reason": reason},
        )

    def close(self, ticket_id: int, advisor_id: str, note: str | None = None) -> dict:
        return self._request(
            "POST",
            f"/api/v1/escalation/tickets/{ticket_id}/close",
            json={"advisor_id": advisor_id, "note": note or None},
        )

    def reopen(self, ticket_id: int, actor: str, note: str | None = None) -> dict:
        return self._request(
            "POST",
            f"/api/v1/escalation/tickets/{ticket_id}/reopen",
            json={"actor": actor, "note": note or None},
        )

    def assess_promotion(self, ticket_id: int, advisor_answer: str) -> dict:
        """What the Section 5 judges advise about a draft, before it is saved."""
        return self._request(
            "POST",
            f"/api/v1/escalation/tickets/{ticket_id}/promotion-assessment",
            json={"advisor_answer": advisor_answer},
        )

    # ------------------------------------------------------ knowledge base

    def ingest(self, project_id: int, file_path: str) -> dict:
        """Upload one file. Returns `{file_id, asset_id, asset_name}`.

        `file_id` is the asset's primary key and is exactly what /process accepts —
        the endpoint returns both it and `asset_name` so a caller never has to guess
        which of the two it was handed.
        """
        path = Path(file_path)
        with path.open("rb") as handle:
            return self._request(
                "POST",
                f"/api/v1/admin/ingest/{project_id}",
                files={"file": (path.name, handle, "application/octet-stream")},
            )

    def process(
        self,
        project_id: int,
        file_id: str | None,
        chunk_size: int,
        overlap: int,
        do_reset: bool,
    ) -> dict:
        """Chunk uploaded files into Postgres. `file_id` None means every file."""
        return self._request(
            "POST",
            f"/api/v1/admin/process/{project_id}",
            json={
                "file_id": file_id or None,
                "chunk_size": int(chunk_size),
                "overlap": int(overlap),
                "do_reset": bool(do_reset),
            },
        )

    def push(self, project_id: int, do_reset: bool) -> dict:
        """Embed and index into pgvector. COSTS MONEY.

        `do_reset` is sent as the int the endpoint's schema declares, not a bool.
        """
        return self._request(
            "POST",
            f"/api/v1/admin/knowledge_base/push/{project_id}",
            json={"do_reset": 1 if do_reset else 0},
        )

    def index_info(self, project_id: int) -> dict:
        return self._request("GET", f"/api/v1/admin/index_info/info/{project_id}")

    def kb_search(
        self, project_id: int, query: str, limit: int = 10, department: str | None = None
    ) -> dict:
        """Retrieval exactly as the agent sees it — same controller, same ranking."""
        return self._request(
            "POST",
            f"/api/v1/admin/knowledge_base/search/{project_id}",
            json={"query": query, "limit": int(limit), "department": department or None},
        )

    # ------------------------------------------------------- profile & health

    def profile(self, student_id: str) -> dict:
        return self._request("GET", f"/api/v1/profile/{student_id}")

    def delete_profile(self, student_id: str) -> dict:
        return self._request("DELETE", f"/api/v1/profile/{student_id}")

    def health(self) -> dict:
        """Readiness: `{status, documents_indexed, collection, version}`."""
        return self._request("GET", "/health")

    def liveness(self) -> dict:
        return self._request("GET", "/")
