#!/usr/bin/env python
"""Build the query-drift baseline from the evaluation set. COSTS MONEY (embeddings).

    uv run --extra dev python scripts/build_drift_baseline.py
    uv run --extra dev python scripts/build_drift_baseline.py --dry-run   # free

Writes `data/eval/drift_baseline.json`: the centroid of the evaluation set's question
embeddings, plus enough provenance to tell whether it still applies.

**Why the eval set is the right reference.** It is the same 62 questions the gate
thresholds and the chunking sweep were tuned against, so "far from this centroid"
means precisely "outside the range where the measured quality numbers hold". A
baseline built from production traffic would instead drift along WITH the traffic and
could never report anything.

**One number is recorded that matters more than the centroid:** the 5th percentile of
the eval questions' own similarity to their centroid. That is the spread of known-good
questions, so it is the honest threshold for "unusual" — picking 0.5 out of the air
would be a number about nothing. It is printed here and belongs in the Grafana panel.

Re-run this whenever EMBEDDING_MODEL_ID or EMBEDDING_MODEL_SIZE changes. A baseline
from a different model is not comparable, and the app logs `drift_not_comparable`
and disables the metric rather than reporting a dimension mismatch as drift.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_QUESTIONS = REPO_ROOT / "data" / "eval" / "questions.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "eval" / "drift_baseline.json"


def load_questions(path: Path) -> list[str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    questions = [record["question"] for record in records if record.get("question")]
    if not questions:
        raise SystemExit(f"{path} has no questions")
    return questions


async def main_async(args) -> int:
    from customer_support.helpers.config import get_settings
    from customer_support.helpers.drift import centroid, cosine
    from customer_support.stores.llm.LLMEnum import DocumentTypeEnum
    from customer_support.stores.llm.LLMProviderFactory import LLMProviderFactory

    settings = get_settings()
    questions = load_questions(args.questions)

    print(f"{len(questions)} questions from {args.questions.name}")
    print(f"embedding with {settings.EMBEDDING_BACKEND} / {settings.EMBEDDING_MODEL_ID}")

    if args.dry_run:
        print("\n--dry-run: nothing embedded, nothing written.")
        return 0

    embedding_client = LLMProviderFactory(settings).create(provider_name=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID, embedding_size=settings.EMBEDDING_MODEL_SIZE
    )

    # Embedded as QUERY, not as DOCUMENT. Cohere's asymmetric models place the two
    # input types in different regions of the space, and the vector this baseline is
    # compared against at request time is a query — so a document-side baseline would
    # report a constant offset as drift.
    vectors = await asyncio.to_thread(
        embedding_client.embed_text, questions, DocumentTypeEnum.QUERY.value
    )
    if not vectors or len(vectors) != len(questions):
        raise SystemExit(f"embedding returned {len(vectors or [])} vectors for {len(questions)}")

    reference = centroid(vectors)
    if reference is None:
        raise SystemExit("could not build a centroid from the returned vectors")

    # The spread of the known-good questions themselves, which is what makes a
    # threshold defensible instead of invented.
    similarities = sorted(value for v in vectors if (value := cosine(v, reference)) is not None)
    index = max(0, int(0.05 * len(similarities)) - 1)
    p5 = similarities[index]

    baseline = {
        "built_at": datetime.now(UTC).isoformat(),
        "model": settings.EMBEDDING_MODEL_ID,
        "dimensions": len(reference),
        "questions": len(questions),
        "source": str(args.questions.relative_to(REPO_ROOT)),
        "centroid": reference,
        # Read these before trusting a drift panel: they say what "normal" is.
        "self_similarity": {
            "min": similarities[0],
            "p5": p5,
            "median": similarities[len(similarities) // 2],
            "max": similarities[-1],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(baseline, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {args.output.relative_to(REPO_ROOT)}")
    print(f"  dimensions      {len(reference)}")
    print(
        f"  self-similarity min {similarities[0]:.3f}  p5 {p5:.3f}  median "
        f"{baseline['self_similarity']['median']:.3f}  max {similarities[-1]:.3f}"
    )
    print(f"\nUse p5 = {p5:.2f} as the 'unusual question' line on the Grafana drift panel.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would happen, call nothing"
    )
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
