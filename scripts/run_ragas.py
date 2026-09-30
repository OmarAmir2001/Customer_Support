#!/usr/bin/env python
"""Score the RAG pipeline with RAGAS against the evaluation set.

    uv run --extra dev python scripts/run_ragas.py                 # the whole set
    uv run --extra dev python scripts/run_ragas.py --limit 20      # the CI subset
    uv run --extra dev python scripts/run_ragas.py --dry-run       # no model calls

Four metrics, which map onto the pipeline like this:

    context_precision   did retrieval put the right chunks at the top?
    context_recall      did retrieval find everything the reference answer needs?
    faithfulness        is every claim in the answer supported by those chunks?
    answer_relevancy    does the answer address the question that was asked?

**The judge gates are bypassed on purpose.** Running the full graph would escalate
every low-confidence question, leaving no answer to score — RAGAS would end up
measuring the gates rather than the retrieval and generation underneath them. Scoring
the ungated answer is what tells you what the gates are protecting you from.

The gates' own verdicts are recorded alongside, without being allowed to suppress
anything. Your three gates and RAGAS measure nearly the same things by different
means — one online and hand-written, one offline and standard — so having both on the
same questions turns "do they agree?" into a number instead of an opinion.

**Cost.** One generation call per question, plus several RAGAS judge calls per metric
per question. The free Groq tier caps the judge model at 8,000 tokens/minute and that
was reached at five concurrent users, so this runs SEQUENTIALLY and leans on the
provider backoff. `--limit` exists for that reason.
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
DEFAULT_MANIFEST = REPO_ROOT / "data" / "corpus" / "manifest.json"
DEFAULT_REPORTS = REPO_ROOT / "reports"


def corpus_version() -> str:
    """The corpus hash, so a score is attributable to the data that produced it.

    A faithfulness number with no data version attached cannot be explained later —
    "it scored 0.86 last month and 0.81 today" becomes a shrug rather than a diff.
    """
    try:
        return json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))["content_sha256"]
    except (OSError, KeyError, json.JSONDecodeError):
        return "unknown"


async def collect_samples(questions: list[dict], deps, *, top_k: int) -> list[dict]:
    """Run retrieval and generation for each question, recording what RAGAS needs.

    Sequential, not gathered. The provider's token budget is the binding constraint
    here — concurrency buys nothing when the limiter is per-minute, and it turns a
    slow run into a failed one.
    """
    rows: list[dict] = []

    for index, item in enumerate(questions, start=1):
        qid, question = item["id"], item["question"]
        print(f"  [{index:>3}/{len(questions)}] {qid} {question[:56]}", flush=True)

        chunks = await deps.retrieval.retrieve(
            question=question, department=item.get("department"), limit=top_k
        )

        row = {
            "id": qid,
            "question": question,
            "language": item["language"],
            "department": item.get("department"),
            "answerable": item["answerable"],
            "reference": item.get("ground_truth") or "",
            "expected_article": item.get("expected_article"),
            "expected_sources": item.get("expected_sources") or [],
            "retrieved_contexts": [c.text for c in chunks],
            "retrieved_sources": [str((c.metadata or {}).get("source")) for c in chunks],
            "retrieved_sections": [str((c.metadata or {}).get("section")) for c in chunks],
        }

        if not chunks:
            # Nothing retrieved is a real outcome, and for the unanswerable questions
            # it is the CORRECT one. Recorded rather than skipped.
            row["answer"] = ""
            row["generation_skipped"] = "no chunks retrieved"
            rows.append(row)
            continue

        try:
            row["answer"] = (
                await deps.generation.generate_answer(
                    question=question,
                    chunks=chunks,
                    # Passing the language explicitly is not optional here. In the
                    # running app the graph negotiates it from the request; this
                    # script has no graph, so leaving it unset falls back to
                    # PRIMARY_LANG and every Arabic question gets an English answer.
                    #
                    # That silently wrecks answer_relevancy, which embeds a question
                    # generated FROM the answer and compares it to the original: an
                    # English answer to an Arabic question scores badly for a reason
                    # that has nothing to do with the system being evaluated.
                    language=item["language"],
                )
                or ""
            )
        except Exception as exc:  # a provider outage should not lose the whole run
            row["answer"] = ""
            row["generation_error"] = str(exc)

        rows.append(row)

    return rows


async def record_gate_verdicts(rows: list[dict], deps) -> None:
    """What the online judges would have said, without letting them suppress anything.

    This is the raw material for comparing a cheap hand-written gate against the
    standard offline metric. It costs three extra judge calls per question, so it is
    opt-in.
    """
    for row in rows:
        if not row.get("answer"):
            continue
        try:
            from customer_support.models.db_schemas import RetrievedDocument

            chunks = [
                RetrievedDocument(text=t, score=1.0, metadata={}) for t in row["retrieved_contexts"]
            ]
            faith = await deps.grading.check_faithfulness(answer=row["answer"], chunks=chunks)
            relev = await deps.grading.check_answer_relevance(
                question=row["question"], answer=row["answer"]
            )
            row["gate_faithfulness"] = faith.score
            row["gate_answer_relevance"] = relev.score
            row["gate_would_escalate"] = not (faith.passed and relev.passed)
        except Exception as exc:
            row["gate_error"] = str(exc)


def score_with_ragas(rows: list[dict], settings, embedding_client) -> dict:
    """Hand the collected rows to RAGAS.

    Import inside the function on purpose: ragas pulls a large dependency tree, and
    `--dry-run` should not pay for it just to inspect what would be scored.
    """
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # Import path note: these live at `ragas.metrics` in 0.4.x and are deprecated in
    # favour of `ragas.metrics.collections` at v1.0 — but answer relevancy is not in
    # collections yet, so all four come from one place rather than two.
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    # Only answerable questions are scored. An unanswerable one has no reference
    # answer, so faithfulness and recall are undefined for it — averaging in a zero
    # would quietly drag the score down and make the number mean nothing.
    scorable = [r for r in rows if r["answerable"] and r.get("answer")]
    if not scorable:
        raise SystemExit("nothing scorable: every answerable question produced no answer")

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["retrieved_contexts"],
                reference=r["reference"],
            )
            for r in scorable
        ]
    )

    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.RAGAS_MODEL_ID or settings.GENERATION_MODEL_ID,
            api_key=settings.GROQ_API_KEY,
            base_url=settings.OPENAI_API_URL,
            temperature=0.0,
            # LangChain's own retry. RAGAS does not use this project's provider, so
            # the Retry-After handling in stores/llm/rate_limit.py never sees these
            # calls — this is the only backoff they get.
            max_retries=6,
        )
    )
    # answer_relevancy embeds generated questions, so an embeddings model is needed
    # even though nothing else here does.
    #
    # Wrapping the app's OWN embedding client rather than reaching for
    # langchain-cohere. Two reasons: no extra dependency, and more importantly the
    # evaluation then measures the same embedding model the system actually retrieves
    # with. Scoring retrieval through a different embedder would be measuring a
    # system nobody runs.
    #
    # The first version pointed OpenAIEmbeddings at Groq's base_url, which fails with
    # "input must be a string or an array of strings" — Groq serves chat completions,
    # not embeddings, and this project's embeddings come from Cohere.
    embeddings = LangchainEmbeddingsWrapper(_AppEmbeddings(embedding_client))

    result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
            # strictness=1 is required, not tuning. ResponseRelevancy asks the model
            # for N generations in one call; Groq rejects n>1 outright with
            # "'n' : number must be at most 1", and the metric then degrades to a
            # single sample anyway. Asking for one keeps it honest instead of
            # silently failing some jobs.
            ResponseRelevancy(strictness=1),
        ],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(
            # The single most important line in this function on a free tier.
            max_workers=settings.RAGAS_MAX_WORKERS,
            timeout=settings.RAGAS_TIMEOUT_SECONDS,
        ),
    )
    # to_pandas() is the public accessor. `dict(result)` looks like it should work
    # and does not: EvaluationResult.__getitem__ takes a METRIC NAME and returns the
    # per-sample list, so dict() iterates integers and dies with KeyError: 0.
    frame = result.to_pandas()
    metric_columns = [
        c
        for c in frame.columns
        if c not in {"user_input", "response", "retrieved_contexts", "reference"}
    ]
    means = {c: round(float(frame[c].mean()), 4) for c in metric_columns}
    per_question = frame[metric_columns].to_dict(orient="records")
    return {"scored": len(scorable), "means": means, "per_question": per_question}


class _AppEmbeddings:
    """LangChain's embeddings interface over this project's own provider.

    LangChain (and therefore RAGAS) needs exactly two methods. Our provider already
    speaks to Cohere with the right model and batching, so this is an adapter, not an
    implementation.
    """

    def __init__(self, client):
        self._client = client

    def embed_query(self, text: str) -> list[float]:
        vectors = self._client.embed_text(text, "query")
        return list(vectors[0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(v) for v in self._client.embed_text(list(texts), "document")]


def retrieval_accuracy(rows: list[dict]) -> dict:
    """Cheap checks RAGAS does not make, computed from what we already collected.

    Worth having because they are free and they test things specific to this system:
    whether the department filter held, and whether the expected article was found at
    all. A high context_precision with the wrong handbook cited is still wrong.
    """
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    article_hits = sum(
        1
        for r in answerable
        if r.get("expected_article")
        and any(r["expected_article"] in s for s in r["retrieved_sections"])
    )
    # A leak is the OTHER handbook appearing, not any unexpected source.
    #
    # The first version flagged every question, because `instructor_resolved` chunks —
    # promoted answers from the learning loop — are in the collection and are not
    # department-scoped by design. Counting those as leaks made the check fire on
    # everything, and a check that always fires gets ignored, which is worse than not
    # having it.
    handbooks = {"CS_2023", "IS_2023"}
    source_leaks = [
        r["id"]
        for r in answerable
        if r["expected_sources"]
        and any(s in handbooks and s not in r["expected_sources"] for s in r["retrieved_sources"])
    ]

    return {
        "answerable": len(answerable),
        "expected_article_retrieved": article_hits,
        "expected_article_hit_rate": (
            round(article_hits / len(answerable), 3) if answerable else None
        ),
        # A leak means the department filter let the other handbook through. Because
        # the two handbooks are near-identical, the answer can still be right while
        # the citation points at the wrong programme.
        "department_filter_leaks": source_leaks,
        "unanswerable_that_retrieved_nothing": sum(
            1 for r in unanswerable if not r["retrieved_contexts"]
        ),
        "unanswerable_total": len(unanswerable),
    }


async def main_async(args) -> int:
    from customer_support.controllers.GenerationController import GenerationController
    from customer_support.controllers.GradingController import GradingController
    from customer_support.controllers.RetrievalController import RetrievalController
    from customer_support.helpers.config import get_settings
    from customer_support.stores.llm.LLMProviderFactory import LLMProviderFactory
    from customer_support.stores.llm.templates import TemplateParser
    from customer_support.stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

    settings = get_settings()
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    print(f"corpus version {corpus_version()[:16]}  ·  {len(questions)} questions")

    if args.dry_run:
        print("\n--dry-run: nothing was called. Composition of the selected set:")
        print(f"  answerable   {sum(1 for q in questions if q['answerable'])}")
        print(f"  unanswerable {sum(1 for q in questions if not q['answerable'])}")
        print(f"  arabic       {sum(1 for q in questions if q['language'] == 'ar')}")
        return 0

    # Build only what evaluation needs. No graph, no checkpointer, no HTTP — the run
    # has to be repeatable from a script for DVC and MLflow to mean anything.
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    dsn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    engine = create_async_engine(dsn, pool_pre_ping=True)
    db_client = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    llm = LLMProviderFactory(settings)
    generation_client = llm.create(provider_name=settings.GENERATION_BACKEND)
    generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)
    embedding_client = llm.create(provider_name=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID, embedding_size=settings.EMBEDDING_MODEL_SIZE
    )
    judge_client = llm.create(provider_name=settings.GENERATION_BACKEND)
    judge_client.set_generation_model(model_id=settings.JUDGE_MODEL_ID)

    vectordb = VectorDBProviderFactory(settings, db_client=db_client).create(
        provider=settings.VECTOR_DB_BACKEND
    )
    await vectordb.connect()

    templates = TemplateParser(
        primary_language=settings.PRIMARY_LANG, default_language=settings.DEFAULT_LANG
    )

    class Deps:
        retrieval = RetrievalController(
            vectordb_client=vectordb,
            embedding_client=embedding_client,
            collection_name=settings.KB_COLLECTION_NAME,
            settings=settings,
        )
        generation = GenerationController(
            generation_client=generation_client, templates=templates, settings=settings
        )
        grading = GradingController(
            generation_client=judge_client, templates=templates, settings=settings
        )

    try:
        print("\nretrieval + generation (gates bypassed):")
        rows = await collect_samples(questions, Deps, top_k=settings.RETRIEVAL_TOP_K)

        if args.with_gates:
            print("\nrecording what the online gates would have said:")
            await record_gate_verdicts(rows, Deps)
    finally:
        await vectordb.disconnect()
        await engine.dispose()

    accuracy = retrieval_accuracy(rows)
    print("\nretrieval checks (free, computed locally):")
    for key, value in accuracy.items():
        print(f"  {key:38} {value}")

    report = {
        "run_at": datetime.now(UTC).isoformat(),
        "corpus_version": corpus_version(),
        "questions": len(questions),
        "generation_model": settings.GENERATION_MODEL_ID,
        "embedding_model": settings.EMBEDDING_MODEL_ID,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "retrieval_accuracy": accuracy,
        "rows": rows,
    }

    if not args.no_ragas:
        print("\nscoring with RAGAS (this is the part that costs tokens):")
        outcome = score_with_ragas(rows, settings, embedding_client)
        report["ragas_scored"] = outcome["scored"]
        report["ragas"] = outcome["means"]
        # Per-question scores too, not just the averages. An average of 0.8 hides
        # whether that is every question at 0.8 or half at 1.0 and half at 0.6 — and
        # only the second tells you which questions to go and look at.
        for row, scores in zip(
            [r for r in rows if r["answerable"] and r.get("answer")],
            outcome["per_question"],
            strict=False,
        ):
            row["ragas"] = {k: (None if v != v else round(float(v), 4)) for k, v in scores.items()}
        print("\nRAGAS:")
        for name, value in report["ragas"].items():
            print(f"  {name:38} {value}")

    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / f"ragas_{corpus_version()[:12]}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--limit", type=int, default=None, help="first N questions (CI subset)")
    parser.add_argument(
        "--no-ragas", action="store_true", help="collect and check retrieval, skip the scoring"
    )
    parser.add_argument(
        "--with-gates",
        action="store_true",
        help="also record the online judges' verdicts, for the judge-vs-RAGAS comparison",
    )
    parser.add_argument("--dry-run", action="store_true", help="show what would run, call nothing")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
