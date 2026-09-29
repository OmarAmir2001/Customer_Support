#!/usr/bin/env python
"""Build the structured corpus from the raw handbooks. Deterministic, offline, free.

    uv run python scripts/build_corpus.py
    uv run python scripts/build_corpus.py --chunk-size 1500 --overlap 100

This is the first stage of the ingestion pipeline, split out from the HTTP path for
three reasons that all needed it:

* **DVC** drives stages by running a command. `dvc repro` cannot POST to a running
  API, so the corpus has to be buildable by a script from tracked inputs.
* **MLflow** experiments compare chunking configurations. Each run needs to rebuild
  the corpus with different parameters, cheaply and without touching a database.
* **Evaluation** has to run against a known corpus version. A corpus assembled by a
  sequence of HTTP calls is not a version; a file with a content hash is.

It reuses `ProcessController.process_file_content` rather than reimplementing the
splitting. That method is pure — Documents in, chunks out, no filesystem or database
— so the script and the API cannot drift into chunking things differently, which is
the failure this would otherwise invite.

**Deliberately does NOT embed or index.** Embedding costs money and needs a running
Postgres; chunking costs nothing and needs neither. Keeping them separate means a
chunking experiment is free to repeat, and it makes the expensive stage the only one
that has to be skipped when iterating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from customer_support.controllers.ProcessController import (  # noqa: E402
    HANDBOOK_DEPARTMENTS,
    ProcessController,
)

DEFAULT_INPUT = REPO_ROOT / "data" / "handbooks"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "corpus"

#: A chunk longer than this means the overflow splitter did not run — the symptom of
#: a section that was never split, which retrieves poorly and blows the answer prompt.
#: Checked rather than assumed, because a silent chunking bug surfaces three steps
#: later as a bad answer and the model gets the blame.
HARD_MAX_CHARS = 4000


class CorpusError(Exception):
    """Raised when the built corpus fails validation. Deliberately fatal: a corpus
    with a parsing bug in it produces hallucinations downstream, and by then the
    cause is invisible."""


def load_handbook(path: Path) -> list[Document]:
    """Read one handbook file into a single Document.

    Read directly rather than through ProcessController's loader, because that one
    resolves paths inside the assets/ upload directory. The raw handbooks live in
    data/ under version control, which is the whole point of a reproducible build.
    """
    return [Document(page_content=path.read_text(encoding="utf-8"), metadata={})]


def citation_for(source: str, section: str) -> str:
    """A human-checkable reference, not an internal id.

    This is what an answer cites and what the API returns in `sources`. A chunk id
    tells a reader nothing; "CS_2023 — مادة (٢٤)" can be looked up in the handbook
    and verified, which is the only kind of citation worth returning.
    """
    leaf = section.split(" > ")[-1] if section else ""
    return f"{source} — {leaf}" if leaf and leaf != "(untitled)" else source


def build(input_dir: Path, chunk_size: int, overlap: int, settings=None) -> tuple[list[dict], dict]:
    """Chunk every handbook in `input_dir`. Returns (records, manifest).

    `settings` is injectable so this is testable without a .env — the controller
    underneath needs ASSETS_DIR, and nothing else. Left as None, it resolves the real
    Settings, which is what the CLI wants.
    """
    files = sorted(input_dir.glob("*.md"))
    if not files:
        raise CorpusError(f"no .md handbooks found in {input_dir}")

    # project_id is required by the constructor but unused here: nothing in this
    # script reads from the project directory.
    processor = ProcessController(project_id="corpus-build", settings=settings)

    records: list[dict] = []
    for path in files:
        chunks = processor.process_file_content(
            file_content=load_handbook(path),
            file_id=path.name,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # Number the parts within each section so a split section stays identifiable
        # as one logical unit — the same grouping the vector re-sync deletes on.
        per_section: Counter[str] = Counter()
        for chunk in chunks:
            source = chunk.metadata.get("source") or path.stem
            section = chunk.metadata.get("section") or "(untitled)"
            part = per_section[section]
            per_section[section] += 1

            records.append(
                {
                    # Stable and derived from position, not from a counter that
                    # shifts when an unrelated file is added. This is the key the
                    # idempotent delete-then-insert sync uses.
                    "chunk_id": f"{source}::{section}::{part}",
                    "source": source,
                    "department": chunk.metadata.get("department"),
                    "section": section,
                    "part": part,
                    "text": chunk.page_content,
                    "char_count": len(chunk.page_content),
                    "citation": citation_for(source, section),
                }
            )

    # Fill in how many parts each section ended up with, so a consumer can tell a
    # whole section from a fragment of one without re-scanning.
    totals = Counter(record["section"] for record in records)
    for record in records:
        record["parts_in_section"] = totals[record["section"]]

    payload = json.dumps(records, ensure_ascii=False, sort_keys=True)
    manifest = {
        "built_at": datetime.now(UTC).isoformat(),
        # The corpus version. Two runs with the same inputs and parameters produce
        # the same hash, which is what makes it usable as a data version in MLflow
        # lineage rather than a timestamp that changes on every rebuild.
        "content_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "source_files": [path.name for path in files],
        "records": len(records),
        "sections": len(totals),
        "by_source": dict(Counter(record["source"] for record in records)),
        "by_department": dict(Counter(str(record["department"]) for record in records)),
        "char_count_total": sum(record["char_count"] for record in records),
        "char_count_max": max(record["char_count"] for record in records),
    }
    return records, manifest


def validate(records: list[dict]) -> list[str]:
    """Check the corpus before anything embeds it. Returns a list of problems.

    Every check here is a bug that is silent at build time and expensive later: an
    empty chunk wastes an embedding call and can never match; a missing source breaks
    retrieval RANKING, since handbook chunks are supposed to outrank promoted
    answers; a missing department breaks the filter that stops a CS student being
    answered out of the IS handbook; an oversized chunk means the splitter did not
    run.
    """
    problems: list[str] = []

    if not records:
        problems.append("corpus is empty")
        return problems

    seen: set[str] = set()
    for record in records:
        ref = record.get("chunk_id", "<no id>")

        if not (record.get("text") or "").strip():
            problems.append(f"{ref}: empty text")
        if record["chunk_id"] in seen:
            problems.append(f"{ref}: duplicate chunk_id")
        seen.add(record["chunk_id"])

        if record.get("source") not in HANDBOOK_DEPARTMENTS:
            problems.append(
                f"{ref}: source {record.get('source')!r} is not a known handbook "
                f"(expected one of {sorted(HANDBOOK_DEPARTMENTS)}) — retrieval "
                f"precedence keys on this"
            )
        if not record.get("department"):
            problems.append(f"{ref}: no department — the retrieval filter needs it")
        if not record.get("section") or record["section"] == "(untitled)":
            problems.append(f"{ref}: no section label — nothing to cite")
        if record["char_count"] > HARD_MAX_CHARS:
            problems.append(
                f"{ref}: {record['char_count']} chars exceeds {HARD_MAX_CHARS} — "
                f"the overflow splitter did not run"
            )

    # Every handbook that exists should have contributed. A file that produced no
    # chunks is a parsing failure, and it is invisible in a total record count.
    contributed = {record["source"] for record in records}
    for handbook in HANDBOOK_DEPARTMENTS:
        if handbook not in contributed:
            problems.append(f"{handbook}: contributed no chunks at all")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="max chars per chunk before the overflow splitter runs (MLflow param)",
    )
    parser.add_argument(
        "--overlap", type=int, default=50, help="overlap for oversized sections (MLflow param)"
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="write the corpus even if validation fails (for inspecting a bad build)",
    )
    args = parser.parse_args()

    records, manifest = build(args.input_dir, args.chunk_size, args.overlap)
    problems = validate(records)
    manifest["valid"] = not problems

    corpus_path = args.output_dir / "corpus.json"
    manifest_path = args.output_dir / "manifest.json"

    # The validation gate comes before mkdir on purpose: a failed build should leave
    # no trace at all, not an empty directory that looks like a half-finished run.
    if problems and not args.allow_invalid:
        print(f"corpus FAILED validation with {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  - {problem}", file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more", file=sys.stderr)
        print("\nnothing written. Pass --allow-invalid to inspect the output.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {corpus_path.relative_to(REPO_ROOT)}")
    print(f"  records   {manifest['records']} across {manifest['sections']} sections")
    print(f"  by source {manifest['by_source']}")
    print(f"  chars     {manifest['char_count_total']} total, {manifest['char_count_max']} max")
    print(f"  version   {manifest['content_sha256'][:16]}")
    if problems:
        print(f"  WARNING   written with {len(problems)} validation problem(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
