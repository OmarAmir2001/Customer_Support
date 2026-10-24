#!/usr/bin/env python
"""Upload the evaluation question set to Langfuse as a dataset. Idempotent, free.

    uv run --extra dev python scripts/upload_eval_dataset.py --dry-run   # show, change nothing
    uv run --extra dev python scripts/upload_eval_dataset.py

**What this buys.** Tracing an eval run already groups its questions into one Langfuse
session, which makes a single run browsable. It does NOT make two runs comparable: to
see "q003 scored 0 last week and 0.8 today" you would still diff two JSON files. A
dataset is the fixed list both runs attach to, so Langfuse can line them up item by
item.

**No ground truth is invented here.** Every item comes from `data/eval/questions.json`,
which is human-written. The script only reshapes it into Langfuse's
input/expected_output/metadata split — the question goes in `input`, the answer in
`expected_output`, and everything that is neither (the article it should cite, whether
it is answerable at all, review notes) goes in `metadata`, never smuggled into the
other two.

**Idempotent.** Each item carries a stable id (`handbook-eval-q001`), and the v2 SDK
upserts on id. Re-running after editing a question updates that item in place rather
than appending a second copy — which matters because a dataset with two versions of
q003 in it silently halves the value of every comparison built on it.

The 8 unanswerable questions get **no** expected_output. That is not missing data: the
correct behaviour for them is to escalate rather than answer, and an empty expected
output beside `answerable: false` says so. Their `note` field explains why.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_QUESTIONS = REPO_ROOT / "data" / "eval" / "questions.json"

#: One dataset, named for what it evaluates rather than when it was made.
DATASET_NAME = "handbook-eval"

#: Prefix for item ids. The SDK requires ids to be globally unique across the
#: project, not just within the dataset, so a bare "q001" would collide with the
#: first other dataset that numbers its items the same obvious way.
ITEM_ID_PREFIX = "handbook-eval"

#: Fields that describe the question rather than being the question or its answer.
#: Anything here goes to metadata; see the module docstring on keeping the three
#: responsibilities separate.
METADATA_FIELDS = (
    "expected_article",
    "expected_sources",
    "answerable",
    "category",
    "note",
    "bilingual_pair",
    "mirrors",
    "discriminates_department",
    "needs_review",
)


def item_id(question_id: str) -> str:
    return f"{ITEM_ID_PREFIX}-{question_id}"


def build_item(question: dict) -> dict:
    """One Langfuse dataset item from one eval question."""
    return {
        "id": item_id(question["id"]),
        # Exactly what a run needs to reproduce the question, and nothing else.
        # department and language are inputs because they change what is retrieved
        # and what language the answer comes back in.
        "input": {
            "question": question["question"],
            "department": question.get("department"),
            "language": question["language"],
        },
        # Empty for the unanswerable ones, deliberately — see the module docstring.
        "expected_output": question.get("ground_truth") or None,
        "metadata": {
            "question_id": question["id"],
            **{field: question[field] for field in METADATA_FIELDS if field in question},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would be uploaded, call nothing"
    )
    args = parser.parse_args()

    from customer_support.helpers.config import get_settings

    settings = get_settings()
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    items = [build_item(question) for question in questions]

    answerable = sum(1 for question in questions if question["answerable"])
    print(f"{len(items)} items from {args.questions.relative_to(REPO_ROOT)}")
    print(f"  answerable        {answerable}   (expected_output set)")
    print(f"  unanswerable      {len(items) - answerable}   (expected_output empty, on purpose)")
    print(f"  languages         {dict(Counter(q['language'] for q in questions))}")
    print(f"  departments       {dict(Counter(q.get('department') for q in questions))}")
    flagged = [q["id"] for q in questions if q.get("needs_review")]
    if flagged:
        print(f"  flagged for review {flagged}  <- ground truth you marked as unsettled")

    print("\nfirst item as it will be sent:")
    print(json.dumps(items[0], ensure_ascii=False, indent=2)[:700])

    if args.dry_run:
        print("\n--dry-run: nothing created, nothing uploaded.")
        return 0

    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        raise SystemExit(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set, so there is "
            "nowhere to upload to. They live in .env."
        )

    from langfuse import Langfuse

    client = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST or "http://localhost:3002",
    )

    client.create_dataset(
        name=args.dataset,
        description=(
            "Bilingual Arabic/English questions over the CS and IS department "
            "handbooks, with human-written ground truth. 54 answerable; 8 "
            "deliberately out of scope, where the correct behaviour is to escalate "
            "rather than answer."
        ),
        metadata={
            "source": str(args.questions.relative_to(REPO_ROOT)),
            "answerable": answerable,
            "unanswerable": len(items) - answerable,
            "needs_review": flagged,
        },
    )
    print(f"\ndataset {args.dataset!r} ensured")

    for index, item in enumerate(items, start=1):
        client.create_dataset_item(dataset_name=args.dataset, **item)
        if index % 20 == 0 or index == len(items):
            print(f"  upserted {index}/{len(items)}")

    # Without this the script can exit before the SDK's background thread has sent
    # anything — the commonest way to instrument something and then see nothing.
    client.flush()
    print(f"\ndone. {len(items)} items in {args.dataset!r}")
    print(f"  {settings.LANGFUSE_HOST or 'http://localhost:3002'} -> Datasets -> {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
