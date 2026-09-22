"""Rendering retrieved chunks into prompt text.

Lives beside the templates rather than on a controller because BOTH the generator
and the judges show excerpts to a model, and they must label them identically. If
the generator called an excerpt "an answer an advisor gave" and the faithfulness
judge saw the same row unlabelled, the judge would treat a perfectly grounded
answer as hearsay and the learning loop would die at gate 2a. One function, so the
two cannot drift.

The only difference between the two callers is the language they render in, and that
is a parameter:

* the generator renders in the student's resolved locale;
* the judges render in ``JUDGE_LANGUAGE``, because the rubrics are single-language
  by design — only a judge's ``reason`` is ever shown to a student, and translating
  the rubrics would double the surface you have to keep in step while tuning.
"""

from customer_support.models.db_schemas import RetrievedDocument

from .template_parser import TemplateParser

RAG_GROUP = "rag"

#: Judge rubrics are English-only, so excerpts shown to a judge are labelled in
#: English too — a prompt that mixes rubric and label languages reads worse to the
#: model than one that is consistently one language.
JUDGE_LANGUAGE = "en"

INSTRUCTOR_RESOLVED_SOURCE = "instructor_resolved"


def describe_source(
    doc: RetrievedDocument, parser: TemplateParser, language: str | None = None
) -> str:
    """Say what an excerpt IS, in the reader's language.

    Section 5 deliberately feeds two kinds of text into one excerpt list — handbook
    sections and answers advisors gave earlier students — and a model that cannot
    tell them apart discounts the second kind.
    """
    meta = doc.metadata or {}
    source = meta.get("source")
    section = meta.get("section")

    if source == INSTRUCTOR_RESOLVED_SOURCE:
        key, variables = "source_instructor", {}
    elif not source:
        key, variables = "source_unknown", {}
    elif section:
        key, variables = "source_handbook_with_section", {"source": source, "section": section}
    else:
        key, variables = "source_handbook", {"source": source}

    label = parser.get(RAG_GROUP, key, variables, language=language)
    # parser.get only returns None for a missing group/key, which the parity test
    # rules out. Degrade to the raw source rather than putting "None" in a prompt.
    return label if label is not None else (source or "")


def format_excerpts(
    chunks: list[RetrievedDocument], parser: TemplateParser, language: str | None = None
) -> str:
    """Numbered, provenance-labelled excerpts, joined.

    Numbering is 1-based so a judge's reason ("excerpt 2 does not support...") reads
    the way a person would say it.
    """
    rendered = [
        parser.get(
            RAG_GROUP,
            "document_prompt",
            {
                "doc_num": index,
                "doc_label": describe_source(chunk, parser, language=language),
                "chunk_text": chunk.text,
            },
            language=language,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(part for part in rendered if part)
