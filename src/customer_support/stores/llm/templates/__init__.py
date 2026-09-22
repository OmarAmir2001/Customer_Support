"""Prompt templates.

Sits under ``stores/llm/`` rather than in a controller because a prompt is the wire
format for a model, and because both GradingController and GenerationController need
the same excerpt rendering — a module two controllers share is owned by neither.
Controllers importing from stores is the allowed dependency direction.

Split by audience:

* ``locales/<lang>/<group>.py`` — student-facing text, one file per language.
* ``judges.py`` — grading rubrics, single-language (only a verdict's ``reason``
  reaches a student, and it is asked for in the question's language).
* ``excerpts.py`` — chunk rendering shared by both, so their labels cannot drift.
* ``template_parser.py`` — stateless loader with tolerant, logged locale fallback.
"""

from .excerpts import JUDGE_LANGUAGE, RAG_GROUP, describe_source, format_excerpts
from .judges import (
    ANSWER_RELEVANCE_TEMPLATE,
    CONTEXT_RELEVANCE_TEMPLATE,
    FAITHFULNESS_TEMPLATE,
    JUDGE_SYSTEM_PROMPT,
)
from .template_parser import FALLBACK_LANGUAGE, TemplateParser

__all__ = [
    "ANSWER_RELEVANCE_TEMPLATE",
    "CONTEXT_RELEVANCE_TEMPLATE",
    "FAITHFULNESS_TEMPLATE",
    "FALLBACK_LANGUAGE",
    "JUDGE_LANGUAGE",
    "JUDGE_SYSTEM_PROMPT",
    "RAG_GROUP",
    "TemplateParser",
    "describe_source",
    "format_excerpts",
]
