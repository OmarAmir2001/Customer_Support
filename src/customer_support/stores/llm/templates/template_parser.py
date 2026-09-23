"""Locale-aware prompt loader.

Layout it expects::

    locales/
      en/rag.py        <- "group" is the module name, "key" is a name inside it
      ar/rag.py

Two decisions make this production-shaped rather than a tutorial parser:

1. **It is stateless.** The language is an argument to every call, never instance
   state. The obvious design — one parser on ``app.state`` with a ``set_language()``
   called per request — is a data race under an async server: two concurrent
   requests share the instance, and an Arabic request can read the language an
   English request just set. That bug is invisible with one user and impossible to
   reproduce on demand, so it is designed out rather than tested for. One instance
   is safe to share precisely because nothing mutates.
2. **A missing locale falls back and says so.** Falling back keeps a half-translated
   deployment serving answers instead of 500ing. Logging the fallback is what stops
   that from becoming "our Arabic students silently got English prompts for a
   month" — the one failure mode a tolerant fallback introduces.

A missing *variable*, by contrast, raises. That is a programming error, and a prompt
containing a literal "$question" is worse than a loud failure: the model would
answer a question it never received.
"""

import importlib
from functools import cache, lru_cache
from pathlib import Path
from string import Template

from customer_support.helpers.logging_config import get_logger

logger = get_logger(__name__)

LOCALES_DIR = Path(__file__).parent / "locales"
LOCALES_PACKAGE = f"{__package__}.locales"

# The locale every lookup falls back to. It must be complete; the parity test
# treats it as the reference every other locale is compared against.
FALLBACK_LANGUAGE = "en"


@cache
def _load_group(language: str, group: str):
    """Import locales/<language>/<group>.py, or return None if it does not exist.

    Cached: prompt modules are immutable at run time, and importing one per request
    would put filesystem work on the hot path. Module-level rather than a method, so
    the cache is not keyed on a parser instance.
    """
    if not (LOCALES_DIR / language / f"{group}.py").is_file():
        return None
    try:
        return importlib.import_module(f"{LOCALES_PACKAGE}.{language}.{group}")
    except ImportError:
        # A group file that exists but cannot import is a real defect, not a
        # missing translation — surface it instead of quietly using English.
        logger.error("locale_group_import_failed", language=language, group=group, exc_info=True)
        raise


@lru_cache(maxsize=1)
def _discover_languages() -> tuple[str, ...]:
    """Language codes that actually have a directory on disk."""
    if not LOCALES_DIR.is_dir():
        return ()
    return tuple(
        sorted(
            d.name
            for d in LOCALES_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_") and (d / "__init__.py").is_file()
        )
    )


class TemplateParser:
    def __init__(self, primary_language: str | None = None, default_language: str | None = None):
        self.default_language = default_language or FALLBACK_LANGUAGE
        self.primary_language = primary_language or self.default_language

    # ------------------------------------------------------------- languages

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return _discover_languages()

    def resolve_language(self, requested: str | None = None) -> str:
        """Pick the locale to render in, tolerantly.

        ``requested`` may be None (no preference expressed), a supported code, or
        something this deployment has no translation for. Only the first supported
        candidate wins; anything else lands on the default.
        """
        candidates = [requested, self.primary_language, self.default_language, FALLBACK_LANGUAGE]
        supported = self.supported_languages

        for candidate in candidates:
            if candidate and candidate in supported:
                if requested and candidate != requested:
                    logger.warning(
                        "locale_fallback", requested=requested, resolved=candidate
                    )
                return candidate

        # Nothing matched — including the configured default. That is a deployment
        # error, so it is loud, but still non-fatal for the request in flight.
        logger.error(
            "no_supported_locale",
            requested=requested,
            primary=self.primary_language,
            default=self.default_language,
            available=list(supported),
        )
        return self.default_language

    # --------------------------------------------------------------- lookups

    def get(
        self,
        group: str,
        key: str,
        variables: dict | None = None,
        language: str | None = None,
    ) -> str | None:
        """Render one template. Returns None only if the group or key does not exist.

        Falls back a group at a time, not a key at a time: if Arabic has the group
        but is missing this one key, that is a parity bug and the parity test is the
        place it should surface, not a silent per-key language mix inside a single
        prompt.
        """
        if not group or not key:
            return None

        resolved = self.resolve_language(language)

        module = _load_group(resolved, group)
        if module is None and resolved != self.default_language:
            logger.warning(
                "locale_group_missing", language=resolved, group=group,
                falling_back_to=self.default_language,
            )
            resolved = self.default_language
            module = _load_group(resolved, group)

        if module is None:
            logger.error("locale_group_not_found", group=group, language=resolved)
            return None

        template = getattr(module, key, None)
        if template is None:
            logger.error("locale_key_not_found", group=group, key=key, language=resolved)
            return None

        if not isinstance(template, Template):
            # Every value in a group file is meant to be a Template, so this is a
            # defect in the locale file rather than something to paper over.
            logger.error(
                "locale_key_not_a_template",
                group=group, key=key, language=resolved, actual_type=type(template).__name__,
            )
            return None

        # substitute, not safe_substitute: a missing variable is a caller bug, and a
        # prompt that reaches the model with "$question" still in it is worse than
        # a traceback. The parity test renders every template to catch these early.
        return template.substitute(variables or {})

    def version(self, group: str, language: str | None = None) -> str | None:
        """The group's VERSION string, for logging alongside an eval run."""
        module = _load_group(self.resolve_language(language), group)
        return getattr(module, "VERSION", None) if module else None
