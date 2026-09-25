"""Test-wide guards.

The one thing here exists because of a failure that is worth not repeating: a test
constructed a controller without passing `settings=`, so BaseController fell back to
`get_settings()`, which reads the developer's `.env`. The suite passed locally and
failed in CI — the wrong way round for a unit test, and a full CI cycle to learn
something the machine already knew.

The suite's defining property is that it needs no database, no API keys and no
configuration. That is what makes it fast and what makes CI meaningful. A silent
fallback to real Settings quietly breaks that property, so this turns the fallback
into an immediate, explicit failure instead.
"""

import importlib

import pytest

# importlib, deliberately. `from customer_support.controllers import BaseController`
# returns the CLASS, because controllers/__init__.py re-exports it — and setattr on
# the class raises, since the class has no `get_settings` attribute. This gets the
# module that actually holds the name.
base_controller_module = importlib.import_module("customer_support.controllers.BaseController")


@pytest.fixture(autouse=True)
def _no_real_settings_in_tests(monkeypatch):
    """Make the implicit `get_settings()` fallback fail loudly.

    Patched on BaseController's own module reference, not on
    customer_support.helpers.config: BaseController did `from ... import get_settings`,
    so it holds its own name and patching the source module would miss it.
    """

    def _refuse():
        raise AssertionError(
            "A controller was built without `settings=`, so it fell back to real "
            "Settings and would read .env. Pass a FakeSettings with the attributes "
            "that controller needs (ASSETS_DIR at minimum). Tests must not depend on "
            "local configuration — that is the difference between passing here and "
            "passing in CI."
        )

    monkeypatch.setattr(base_controller_module, "get_settings", _refuse)
