#!/usr/bin/env python
"""Index the built corpus into pgvector. Idempotent. COSTS MONEY (embeddings).

    uv run --extra dev python scripts/index_corpus.py
    uv run --extra dev python scripts/index_corpus.py --dry-run     # free
    uv run --extra dev python scripts/index_corpus.py --project-id 3

The second half of the ingestion pipeline. `build_corpus.py` turns handbooks into a
validated corpus for free; this puts that corpus into the vector store, which is the
part that costs Cohere calls and needs a database.

Two things needed it, and neither could use the HTTP path:

* **DVC** drives stages by running a command, so the `index` stage cannot POST to
  `/knowledge_base/push`.
* **The chunking experiment** re-indexes once per configuration. Going through HTTP
  would mean the indexed content came from the API's own chunker rather than from the
  DVC-tracked corpus whose hash the experiment records.

**It writes no rows to the `chunks` table.** The HTTP path persists chunks there and
passes their ids as `record_id`; this passes None, which the column allows and the
promoted-ticket path already relies on. The corpus file is the source of truth here,
so a second copy in Postgres would just be something else to keep in sync.

**Idempotent without a reset.** `sync_sections` deletes by `(source, section)` and
re-inserts, so running it twice leaves the same rows — and changing `chunk_size` is
handled correctly, because the section set is unchanged even when the number of parts
per section is not. Promoted ticket answers carry `source: instructor_resolved` and no
`section`, so no handbook group's criteria can match them: the learning loop survives
a re-index.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CORPUS = REPO_ROOT / "data" / "corpus" / "corpus.json"


@dataclass
class _Chunk:
    """Only the two attributes the vector path reads.

    Deliberately not a real `DataChunk`: that is a SQLAlchemy row with foreign keys to
    `project` and `assets`, and constructing one would mean creating rows this script
    has no reason to own.
    """

    chunk_text: str
    chunk_metadata: dict = field(default_factory=dict)


@dataclass
class _Project:
    """`create_collection_name` reads exactly this."""

    project_id: int


def load_corpus(path: Path) -> list[_Chunk]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise SystemExit(f"{path} is empty — run scripts/build_corpus.py first")

    return [
        _Chunk(
            chunk_text=record["text"],
            # source and section are what the sync keys on, and what retrieval ranks
            # and filters by. Passing the corpus record's own values through keeps the
            # indexed metadata identical to what validation already checked.
            chunk_metadata={
                "source": record["source"],
                "section": record["section"],
                "department": record.get("department"),
                "citation": record.get("citation"),
                "chunk_id": record.get("chunk_id"),
            },
        )
        for record in records
    ]


async def main_async(args) -> int:
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from customer_support.controllers.KBController import KBController
    from customer_support.helpers.config import get_settings
    from customer_support.stores.llm.LLMProviderFactory import LLMProviderFactory
    from customer_support.stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

    settings = get_settings()
    chunks = load_corpus(args.corpus)

    manifest = args.corpus.parent / "manifest.json"
    version = "unknown"
    if manifest.exists():
        version = json.loads(manifest.read_text(encoding="utf-8")).get("content_sha256", "?")

    sections = {(c.chunk_metadata["source"], c.chunk_metadata["section"]) for c in chunks}
    print(f"corpus {version[:16]}  ·  {len(chunks)} chunks across {len(sections)} sections")
    print(f"target collection_{args.project_id}")

    if args.dry_run:
        print("\n--dry-run: nothing embedded, nothing written.")
        return 0

    dsn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    engine = create_async_engine(dsn, pool_pre_ping=True)
    db_client = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    llm = LLMProviderFactory(settings)
    embedding_client = llm.create(provider_name=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID, embedding_size=settings.EMBEDDING_MODEL_SIZE
    )
    generation_client = llm.create(provider_name=settings.GENERATION_BACKEND)

    vectordb = VectorDBProviderFactory(settings, db_client=db_client).create(
        provider=settings.VECTOR_DB_BACKEND
    )
    await vectordb.connect()

    kb = KBController(
        vectordb_client=vectordb,
        generation_client=generation_client,
        embedding_client=embedding_client,
    )

    try:
        print(f"\nembedding {len(chunks)} chunks and syncing by section...")
        ok = await kb.sync_sections(
            project=_Project(project_id=args.project_id),
            chunks=chunks,
            # None, not fabricated ids: these chunks have no row in the `chunks`
            # table, and collection.chunk_id is a nullable FK to it.
            chunks_ids=[None] * len(chunks),
            do_reset=args.reset,
        )
        if not ok:
            print("sync_sections reported failure — see the logs above", file=sys.stderr)
            return 1

        info = await kb.get_vector_db_collection_info(project=_Project(project_id=args.project_id))
        print(f"\ncollection now: {info}")
    finally:
        await vectordb.disconnect()
        await engine.dispose()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--project-id",
        type=int,
        default=3,
        help="collection_<id>. 3 matches KB_COLLECTION_NAME, which is what the agent reads",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "rebuild the whole collection. Rarely wanted: the per-section sync is "
            "already idempotent, and a reset also deletes every promoted ticket answer"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would happen, call nothing"
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
