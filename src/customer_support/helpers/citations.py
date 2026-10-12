"""Turning retrieved chunks into citations a client can check.

The rubric asks for ``sources[]`` to be *citations, not chunk ids* — and that
distinction is the whole reason this module exists rather than a list comprehension
at the call site. ``CS_2023::الفصل الثالث > مادة (٢٤)::0`` identifies a row in our
vector store; it tells a student nothing and a reviewer less. ``CS_2023 — مادة (٢٤)``
names a section of a handbook they can open and verify.

``build_corpus.py`` already computes exactly that string into each record's
``citation`` field, so the normal path here is a lookup, not a construction. The
fallbacks below exist for the two kinds of row that do not come from the corpus
builder:

* **promoted ticket answers** (``source: instructor_resolved``) have no handbook
  section to point at — their provenance is a ticket, so that is what they cite;
* **rows indexed through the HTTP ingest path** carry source and section but no
  pre-built citation, because that chunker predates the corpus builder.
"""

from customer_support.models.db_schemas import RetrievedDocument

INSTRUCTOR_RESOLVED_SOURCE = "instructor_resolved"


def citation_for(doc: RetrievedDocument) -> str | None:
    """One checkable provenance string, or None if the row cannot name its origin.

    None rather than a placeholder: a citation a student cannot follow is worse than
    no citation, because it looks like evidence.
    """
    meta = doc.metadata or {}

    # The corpus builder's own value, which is what the handbook path should hit.
    citation = meta.get("citation")
    if citation:
        return str(citation)

    source = meta.get("source")

    if source == INSTRUCTOR_RESOLVED_SOURCE:
        ticket_id = meta.get("ticket_id")
        # English even on an Arabic answer. This is a provenance label in a
        # machine-readable API field, not prose shown to the student, and the
        # ticket number is the part that carries the meaning.
        return f"Advisor-resolved ticket #{ticket_id}" if ticket_id else "Advisor-resolved answer"

    if not source:
        return None

    section = meta.get("section")
    if section and section != "(untitled)":
        # Mirrors build_corpus.citation_for: the LEAF of the section path, because
        # "الفصل الثالث > مادة (٢٤)" is how we store it and "مادة (٢٤)" is how a
        # student would look it up.
        leaf = str(section).split(" > ")[-1]
        return f"{source} — {leaf}"

    return str(source)


def citations_for(chunks: list[RetrievedDocument] | None) -> list[str]:
    """Deduplicated citations, in the order the chunks were given.

    Order is preserved rather than sorted because the chunk order is meaningful:
    RetrievalController ranks handbook text ahead of promoted ticket answers, so the
    first citation is the most authoritative one. Duplicates are dropped because two
    parts of one long section are one citation to a reader.
    """
    seen: set[str] = set()
    citations: list[str] = []
    for doc in chunks or []:
        citation = citation_for(doc)
        if citation and citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations
