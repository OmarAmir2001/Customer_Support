"""The UI builds, and its client speaks the API's error contract.

Skipped entirely when the `ui` extra is not installed, which is the case in CI — the
API image does not carry Gradio and there is no reason to install it to test the
backend. Run it with:

    uv run --extra dev --extra ui pytest tests/test_ui_smoke.py

**Why a build test is worth having.** Most of what breaks in a Gradio app is wiring:
an event handler whose `outputs` list no longer matches what the function returns, a
component renamed in one place and not the other, a tab that raises on import.
Constructing every tab catches all of that without a browser and without a running
API — the components are built, the handlers are registered, nothing is called.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

gr = pytest.importorskip("gradio", reason="the `ui` extra is not installed")

UI_DIR = Path(__file__).resolve().parent.parent / "app" / "gradio"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))


def test_the_whole_interface_builds():
    """Every tab, every component, every event wiring — without a server."""
    import main

    demo = main.build_app()
    assert isinstance(demo, gr.Blocks)
    # Blocks keeps its children; a tab that silently failed to build would leave
    # this far smaller.
    assert len(demo.blocks) > 50, f"suspiciously few components: {len(demo.blocks)}"


def test_the_ui_imports_nothing_from_the_backend():
    """The architectural rule, asserted rather than documented.

    The console is a separate deployable that talks HTTP. If someone imports
    `customer_support` here for convenience, the UI silently acquires a database
    handle, provider SDKs and the whole dependency tree — and stops being able to
    point at a remote deployment.
    """
    offenders = []
    for source in sorted(UI_DIR.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import customer_support", "from customer_support")):
                offenders.append(f"{source.relative_to(UI_DIR)}: {stripped}")
    assert not offenders, "the UI must not import the backend:\n" + "\n".join(offenders)


def test_no_init_py_in_the_ui_root():
    """Naming a package `gradio` would shadow the library it imports.

    Guarded because adding an `__init__.py` is the obvious "tidy-up" someone makes
    later, and the resulting failure (`AttributeError` on `gradio.Blocks`) points
    nowhere near the cause.
    """
    assert not (UI_DIR / "__init__.py").exists()


# ----------------------------------------------------- the API error contract


def _client():
    from api_client import ApiClient

    return ApiClient(base_url="http://testserver", timeout=1)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (404, "Not found"),
        (409, "another advisor got there first"),
        (422, "rejected the input"),
        (503, "temporarily unavailable"),
        (500, "API returned 500"),
    ],
)
def test_http_errors_become_sentences_an_advisor_can_act_on(status, expected):
    """409 is the one that matters: it means two different things to a person.

    Either the ticket is not in a state that allows the move, or another advisor
    claimed it first. Collapsing both into "request failed" loses the only
    information that tells the advisor whether to reload or to give up.
    """
    from api_client import ApiClient, ApiError

    response = httpx.Response(status, json={"detail": "nope"})
    with pytest.raises(ApiError) as caught:
        ApiClient._unwrap(response)
    assert expected in str(caught.value)


def test_a_validation_error_list_is_flattened_into_readable_text():
    """FastAPI returns `detail` as a LIST for 422, not a string.

    Rendering that list raw puts `[{'loc': ['body', 'question'], ...}]` on screen.
    """
    from api_client import ApiClient, ApiError

    response = httpx.Response(
        422,
        json={
            "detail": [
                {
                    "loc": ["body", "question"],
                    "msg": "String should have at least 3 characters",
                }
            ]
        },
    )
    with pytest.raises(ApiError) as caught:
        ApiClient._unwrap(response)
    message = str(caught.value)
    assert "body.question" in message
    assert "at least 3 characters" in message


def test_an_unreachable_api_names_the_command_that_fixes_it():
    """A connection error is the most likely failure, so it gets the best message."""
    from api_client import ApiError

    client = _client()
    with pytest.raises(ApiError) as caught:
        client.health()
    assert "make up" in str(caught.value)


def test_a_successful_empty_body_is_not_a_parse_error():
    """Some lifecycle endpoints could answer 200 with no content."""
    from api_client import ApiClient

    assert ApiClient._unwrap(httpx.Response(200)) is None
