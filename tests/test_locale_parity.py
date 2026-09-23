"""Locale parity: every translation must expose the same names AND the same
placeholders as the reference locale.

The parser falls back tolerantly when a locale is missing, which is right for
availability and dangerous for correctness — a forgotten Arabic key would quietly
serve English to Arabic students, and nothing would ever fail. These tests move that
failure to CI, where it is cheap.

Placeholder parity is the subtler half. A locale can have every key and still be
broken: if the Arabic ``document_prompt`` forgets ``$chunk_text``, it renders
perfectly and silently drops the evidence out of the prompt.
"""

import importlib
from string import Template

import pytest

from customer_support.stores.llm.templates.template_parser import (
    FALLBACK_LANGUAGE,
    LOCALES_DIR,
    LOCALES_PACKAGE,
    TemplateParser,
)

# Dummy values used to smoke-render every template. Any placeholder a locale
# declares must appear here, otherwise the render test fails and tells you to add it.
SAMPLE_VARIABLES = {
    "assistant_name": "Murshid",
    "question": "how many credit hours?",
    "chunks": "[1] (student handbook CS_2023) 135 credit hours.",
    "answer": "135 credit hours.",
    "doc_num": 1,
    "doc_label": "student handbook CS_2023",
    "chunk_text": "The programme requires 135 credit hours.",
    "source": "CS_2023",
    "section": "4.2",
    "profile": "name: Omar, department: CS",
    "history": "Student: and for IS?\nYou: the IS handbook requires 132 hours.",
    "content": "and what about for IS students?",
}


def discover_languages() -> list[str]:
    return sorted(
        d.name
        for d in LOCALES_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "__init__.py").is_file()
    )


def discover_groups(language: str) -> list[str]:
    return sorted(
        p.stem for p in (LOCALES_DIR / language).glob("*.py") if not p.stem.startswith("_")
    )


def load(language: str, group: str):
    return importlib.import_module(f"{LOCALES_PACKAGE}.{language}.{group}")


def public_values(module) -> dict:
    """Names a locale file is declaring — its templates plus its VERSION.

    Imported helpers are excluded implicitly: ``Template`` is a class, so it is
    neither a Template instance nor a str.
    """
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_") and isinstance(value, (Template, str))
    }


OTHER_LANGUAGES = [lang for lang in discover_languages() if lang != FALLBACK_LANGUAGE]
REFERENCE_GROUPS = discover_groups(FALLBACK_LANGUAGE)


def test_the_reference_locale_exists():
    """Everything else is compared against it, and the parser falls back to it."""
    assert FALLBACK_LANGUAGE in discover_languages()
    assert REFERENCE_GROUPS, "the reference locale declares no groups"


def test_more_than_one_locale_is_installed():
    """Not a correctness requirement — a canary. If this fails, either the Arabic
    locale was deleted or locale discovery broke, and every other test in this file
    would vacuously pass."""
    assert OTHER_LANGUAGES, "only the reference locale was found; parity is untested"


@pytest.mark.parametrize("language", OTHER_LANGUAGES)
def test_locale_declares_every_group(language):
    assert discover_groups(language) == REFERENCE_GROUPS


@pytest.mark.parametrize("language", OTHER_LANGUAGES)
@pytest.mark.parametrize("group", REFERENCE_GROUPS)
def test_locale_declares_every_key(language, group):
    reference = set(public_values(load(FALLBACK_LANGUAGE, group)))
    translated = set(public_values(load(language, group)))

    missing = reference - translated
    extra = translated - reference
    assert not missing, f"{language}/{group} is missing: {sorted(missing)}"
    assert not extra, f"{language}/{group} declares keys no other locale has: {sorted(extra)}"


@pytest.mark.parametrize("language", discover_languages())
@pytest.mark.parametrize("group", REFERENCE_GROUPS)
def test_every_prompt_is_a_template(language, group):
    """Mixing plain strings and Templates is what makes a prompt blow up with
    AttributeError at request time instead of at import."""
    for name, value in public_values(load(language, group)).items():
        if name.isupper():
            continue  # VERSION and friends are metadata, not prompts
        assert isinstance(value, Template), f"{language}/{group}.{name} is not a Template"


@pytest.mark.parametrize("language", OTHER_LANGUAGES)
@pytest.mark.parametrize("group", REFERENCE_GROUPS)
def test_placeholders_match_the_reference(language, group):
    """The half that key parity misses: a template can exist and still have lost a
    variable, which renders fine and silently drops content from the prompt."""
    reference = public_values(load(FALLBACK_LANGUAGE, group))
    translated = public_values(load(language, group))

    for name, ref_template in reference.items():
        if not isinstance(ref_template, Template):
            continue
        expected = set(ref_template.get_identifiers())
        actual = set(translated[name].get_identifiers())
        assert actual == expected, (
            f"{language}/{group}.{name} placeholders differ: "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )


@pytest.mark.parametrize("language", discover_languages())
@pytest.mark.parametrize("group", REFERENCE_GROUPS)
def test_every_template_renders(language, group):
    """substitute() is strict, so this catches a typo'd placeholder at CI time
    rather than as a KeyError on a live request."""
    for name, value in public_values(load(language, group)).items():
        if not isinstance(value, Template):
            continue
        unknown = set(value.get_identifiers()) - set(SAMPLE_VARIABLES)
        assert not unknown, (
            f"{language}/{group}.{name} uses placeholders this test has no sample "
            f"for: {sorted(unknown)} — add them to SAMPLE_VARIABLES"
        )
        rendered = value.substitute(SAMPLE_VARIABLES)
        assert rendered.strip(), f"{language}/{group}.{name} rendered empty"


@pytest.mark.parametrize("language", discover_languages())
@pytest.mark.parametrize("group", REFERENCE_GROUPS)
def test_group_declares_a_version(language, group):
    """A logged eval run has to be traceable to the exact prompt text behind it."""
    version = getattr(load(language, group), "VERSION", None)
    assert isinstance(version, str) and version, f"{language}/{group} has no VERSION"


# ------------------------------------------------------------------ the parser


def test_parser_resolves_a_supported_language():
    parser = TemplateParser(primary_language=FALLBACK_LANGUAGE)
    for language in discover_languages():
        assert parser.resolve_language(language) == language


def test_parser_falls_back_for_an_unknown_language():
    """Tolerant, not fatal: a half-translated deployment keeps answering."""
    parser = TemplateParser(primary_language=FALLBACK_LANGUAGE, default_language=FALLBACK_LANGUAGE)
    assert parser.resolve_language("tlh") == FALLBACK_LANGUAGE


def test_parser_uses_the_primary_language_when_nothing_is_requested():
    parser = TemplateParser(primary_language=FALLBACK_LANGUAGE)
    assert parser.resolve_language(None) == FALLBACK_LANGUAGE


def test_parser_returns_none_for_a_missing_key_rather_than_raising():
    parser = TemplateParser(primary_language=FALLBACK_LANGUAGE)
    assert parser.get("rag", "no_such_key") is None
    assert parser.get("no_such_group", "system_prompt") is None


def test_parser_renders_a_real_prompt():
    parser = TemplateParser(primary_language=FALLBACK_LANGUAGE)
    rendered = parser.get("rag", "footer_prompt", {"question": "when are exams?"})
    assert "when are exams?" in rendered
