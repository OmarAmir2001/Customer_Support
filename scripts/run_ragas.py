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
per question. Scoring is sequential, checkpointed per metric, and stops on provider
errors. Resuming reuses valid scores. Concurrency does not increase daily quota;
use `--limit` to control workload and check your account's remaining budget.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.embeddings import Embeddings

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_QUESTIONS = REPO_ROOT / "data" / "eval" / "questions.json"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "corpus" / "manifest.json"
DEFAULT_REPORTS = REPO_ROOT / "reports"
RAGAS_METRICS = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")


def select_questions(questions: list[dict], limit: int | None) -> list[dict]:
    """A fixed round-robin over language/department/answerability strata.

    Taking the first 20 silently tests only answerable CS questions. Every config
    must instead see the same representative subset, independent of provider luck.
    """
    if not limit or limit >= len(questions):
        return questions
    if limit < 0:
        raise ValueError("limit must be non-negative")
    groups = defaultdict(deque)
    for question in questions:
        groups[(question["language"], question.get("department"), question["answerable"])].append(
            question
        )
    selected = []
    while len(selected) < limit:
        for group in groups.values():
            if group and len(selected) < limit:
                selected.append(group.popleft())
    return selected


def write_report(report: dict, path: Path) -> None:
    """Keep the answers even when a later metric job fails or the process stops."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_scores(report: dict) -> None:
    """An empty answer or missing metric must never make a successful MLflow run.

    RAGAS returns NaN on failed jobs by default, and pandas.mean silently omits it.
    A deceptively high average over the surviving rows is worse than a failed run.
    """
    answerable = [row for row in report["rows"] if row["answerable"]]
    if not answerable:
        raise ValueError("evaluation contains no answerable questions")
    for row in answerable:
        if not row.get("answer", "").strip():
            raise ValueError(f"{row['id']}: answerable question has no generated answer")
        for metric in RAGAS_METRICS:
            value = row.get("ragas", {}).get(metric)
            if value is None or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{row['id']}: missing or invalid {metric} score ({value})")
    for metric in RAGAS_METRICS:
        value = report.get("ragas", {}).get(metric)
        if value is None or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"missing or invalid aggregate {metric}")


# Text normalisation for the free answer-containment check. Arabic needs it: the same
# word differs by diacritics, tatweel and alef variant, and a lexical comparison that
# ignores that reports a miss for text that is plainly there.
_AR_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u0640]")
_AR_ALEF = re.compile("[\u0623\u0625\u0622]")
_AR_DIGITS = str.maketrans(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669", "0123456789"
)


def corpus_version(manifest: Path = DEFAULT_MANIFEST) -> str:
    """The corpus hash, so a score is attributable to the data that produced it.

    A faithfulness number with no data version attached cannot be explained later —
    "it scored 0.86 last month and 0.81 today" becomes a shrug rather than a diff.
    """
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))["content_sha256"]
    except (OSError, KeyError, json.JSONDecodeError):
        return "unknown"


async def collect_samples(
    questions: list[dict],
    deps,
    *,
    top_k: int,
    generate: bool = True,
    checkpoint=None,
    existing_rows: list[dict] | None = None,
) -> list[dict]:
    """Run retrieval and generation for each question, recording what RAGAS needs.

    With `generate=False` only retrieval runs: no generation call, no answer, nothing
    for a judge to score. That is the mode the chunking sweep uses, because chunking
    changes what gets retrieved and everything after it is the generator's doing.

    Sequential, not gathered. The provider's token budget is the binding constraint
    here — concurrency buys nothing when the limiter is per-minute, and it turns a
    slow run into a failed one.
    """
    rows: list[dict] = []
    saved = {row["id"]: row for row in existing_rows or []}

    for index, item in enumerate(questions, start=1):
        qid, question = item["id"], item["question"]
        print(f"  [{index:>3}/{len(questions)}] {qid} {question[:56]}", flush=True)
        if qid in saved and (saved[qid].get("answer") or not generate):
            rows.append(saved[qid])
            if checkpoint:
                checkpoint(rows)
            continue

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
            "expected_citations": [
                f"{source} — {item['expected_article']}"
                for source in item.get("expected_sources", [])
                if item.get("expected_article")
            ],
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
            if checkpoint:
                checkpoint(rows)
            continue

        if not generate:
            row["answer"] = ""
            row["generation_skipped"] = "retrieval-only run"
            rows.append(row)
            if checkpoint:
                checkpoint(rows)
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
        if checkpoint:
            checkpoint(rows)

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
            context = await deps.grading.check_context_relevance(
                question=row["question"], chunks=chunks
            )
            faith = await deps.grading.check_faithfulness(answer=row["answer"], chunks=chunks)
            relev = await deps.grading.check_answer_relevance(
                question=row["question"], answer=row["answer"]
            )
            row["gate_faithfulness"] = faith.score
            row["gate_context_relevance"] = context.score
            row["gate_answer_relevance"] = relev.score
            row["gate_would_escalate"] = not (context.passed and faith.passed and relev.passed)
        except Exception as exc:
            row["gate_error"] = str(exc)


def valid_score(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1


def score_pending_metrics(rows: list[dict], evaluate_metric, *, checkpoint=None) -> dict:
    """Commit each successful metric before starting another paid operation."""
    scorable = [row for row in rows if row["answerable"] and row.get("answer")]
    if not scorable:
        raise ValueError("nothing scorable: every answerable question produced no answer")
    pending = sum(
        not valid_score(row.get("ragas", {}).get(metric))
        for row in scorable
        for metric in RAGAS_METRICS
    )
    print(f"  {pending} missing metric scores; saved valid scores will be reused", flush=True)
    completed = 0
    for row in scorable:
        scores = row.setdefault("ragas", {})
        for metric in RAGAS_METRICS:
            if valid_score(scores.get(metric)):
                continue
            try:
                value = evaluate_metric(row, metric)
            except Exception as exc:
                raise RuntimeError(
                    f"RAGAS stopped at {row['id']} / {metric}: {exc}. "
                    "Saved answers and successful scores are preserved. "
                    "For HTTP 429, wait for the provider quota to recover before resuming."
                ) from exc
            if not valid_score(value):
                raise ValueError(f"{row['id']}: missing or invalid {metric} score ({value})")
            scores[metric] = round(float(value), 4)
            if checkpoint:
                checkpoint(rows)
            completed += 1
            print(f"  [{completed}/{pending}] {row['id']} {metric}={scores[metric]}", flush=True)
    return {
        "scored": len(scorable),
        "means": {
            metric: round(sum(row["ragas"][metric] for row in scorable) / len(scorable), 4)
            for metric in RAGAS_METRICS
        },
        "per_question": [row["ragas"].copy() for row in scorable],
    }


def score_with_ragas(rows: list[dict], settings, embedding_client, *, checkpoint=None) -> dict:
    """Hand the collected rows to RAGAS.

    Import inside the function on purpose: ragas pulls a large dependency tree, and
    `--dry-run` should not pay for it just to inspect what would be scored.
    """
    import httpx
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

    run_config = RunConfig(
        max_workers=1,
        timeout=settings.RAGAS_TIMEOUT_SECONDS,
        # RAGAS otherwise retries each LLM call ten times, on top of SDK retries.
        # One attempt lets daily exhaustion stop the run immediately.
        max_retries=1,
    )

    def make_judge():
        """A FRESH judge — and therefore a fresh async HTTP client — per metric call.

        This must not be hoisted out of the loop, however wasteful it looks.
        `ragas.evaluate()` calls `asyncio.run()` internally, so each metric runs in a
        brand new event loop. A ChatOpenAI built once lazily creates an AsyncOpenAI
        whose httpx connection pool belongs to whichever loop first used it; the next
        metric then reaches into that pool from a different loop and httpx dies with
        `RuntimeError: Event loop is closed`, surfacing as the thoroughly misleading
        `openai.APIConnectionError: Connection error`.

        That is not hypothetical: a sweep scored exactly one metric
        (`q001 context_precision=0.75`) and then failed on the very next one. The
        object is cheap; the connection has to be re-established either way, because
        the old pool is already dead.
        """
        return LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.RAGAS_MODEL_ID or settings.GENERATION_MODEL_ID,
                api_key=settings.GROQ_API_KEY,
                base_url=settings.RAGAS_API_URL or settings.OPENAI_API_URL,
                temperature=0.0,
                max_tokens=settings.RAGAS_MAX_OUTPUT_TOKENS,
                reasoning_effort=settings.RAGAS_REASONING_EFFORT,
                max_retries=0,
                # httpx.Timeout, NOT a float — and this is the whole fix.
                #
                # langchain_openai hands out its async httpx client from an
                # @lru_cache keyed on (base_url, timeout, socket_options). With a
                # plain number the key is hashable, so every ChatOpenAI sharing a
                # base_url gets the SAME AsyncClient — bound to whichever event loop
                # touched it first. Since ragas.evaluate() runs each metric inside
                # its own asyncio.run(), metric two reaches into a pool belonging to
                # a closed loop and httpx raises "Event loop is closed", which
                # surfaces as "APIConnectionError: Connection error".
                #
                # An httpx.Timeout is unhashable, so the cache lookup raises
                # TypeError and langchain falls back to building a fresh client.
                # That is the library's own documented escape hatch, not a trick.
                # Do not "simplify" this back to a number.
                timeout=httpx.Timeout(float(settings.RAGAS_TIMEOUT_SECONDS)),
            ),
            run_config=run_config,
        )

    if (settings.RAGAS_BACKEND or "").upper() == "COHERE":
        # Built once and reused: a synchronous client has no event-loop affinity, so
        # unlike the langchain_openai path there is nothing to rebuild per metric.
        from customer_support.stores.llm.LLMProviderFactory import LLMProviderFactory

        cohere_judge = _make_app_judge(
            provider=LLMProviderFactory(settings).create(provider_name="COHERE"),
            model_id=settings.RAGAS_MODEL_ID or settings.JUDGE_MODEL_ID,
            max_output_tokens=settings.RAGAS_MAX_OUTPUT_TOKENS,
            run_config=run_config,
        )

        def judge_factory():
            return cohere_judge

    else:
        # Groq. A FRESH judge per call is mandatory here — see make_judge's docstring.
        judge_factory = make_judge

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

    metric_factories = {
        "context_precision": lambda: LLMContextPrecisionWithReference(name="context_precision"),
        "context_recall": LLMContextRecall,
        "faithfulness": Faithfulness,
        # Groq rejects n>1; asking for one avoids unsupported parallel generations.
        "answer_relevancy": lambda: ResponseRelevancy(strictness=1),
    }

    def evaluate_metric(row, metric):
        # A single job prevents another queued paid request from starting after
        # an error, and lets us persist the result before starting the next job.
        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=row["question"],
                    response=row["answer"],
                    retrieved_contexts=row["retrieved_contexts"],
                    reference=row["reference"],
                )
            ]
        )
        result = evaluate(
            dataset=dataset,
            metrics=[metric_factories[metric]()],
            llm=judge_factory(),
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=True,
            allow_nest_asyncio=False,
            show_progress=False,
        )
        return float(result.to_pandas().iloc[0][metric])

    return score_pending_metrics(rows, evaluate_metric, checkpoint=checkpoint)


def _make_app_judge(provider, model_id: str, max_output_tokens: int, run_config):
    """A RAGAS judge backed by this project's own Cohere provider.

    Built as a function rather than a module-level class because `BaseRagasLLM` has
    to be imported lazily — ragas pulls a large dependency tree and `--dry-run`
    should not pay for it.

    **Subclasses `BaseRagasLLM` rather than duck-typing it.** The first attempt
    implemented only the three abstract methods and failed with
    `'_AppJudge' object has no attribute 'generate'`: RAGAS calls `generate()`, which
    is concrete on the base class and wraps `agenerate_text` in the retry policy from
    `run_config`. Duck-typing skipped the retries as well as the method.

    Why not `langchain-cohere`: it cannot be installed here at all — it pins
    `langchain-core<0.4` while langgraph needs `>=1.0`.

    Why this also kills a whole bug class: our provider is SYNCHRONOUS, so it has no
    event-loop affinity. The `langchain_openai` route serves its async httpx client
    from an `@lru_cache`, so every judge sharing a base_url shared one client bound to
    the first loop that used it — and since `ragas.evaluate()` runs each metric inside
    its own `asyncio.run()`, the second metric reliably died with "Event loop is
    closed" wearing an "APIConnectionError: Connection error" mask.
    """
    import asyncio

    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM

    class _AppJudge(BaseRagasLLM):
        def __post_init__(self):  # BaseRagasLLM is a dataclass; keep its contract
            pass

        def get_temperature(self, n: int) -> float:
            # Deterministic judging. RAGAS raises the temperature when it wants
            # variety across n samples; a repeatable score is worth more here.
            return 0.0

        def is_finished(self, response: LLMResult) -> bool:
            return True

        def _one(self, text: str, temperature: float | None) -> str:
            provider.set_generation_model(model_id=model_id)
            # generate_json, not generate_text: every RAGAS metric prompt asks for a
            # JSON verdict, and generate_json is the path that does NOT run the prompt
            # through process_text — truncating a metric's evidence would invalidate
            # its score exactly as it would a judge's.
            return (
                provider.generate_json(
                    text, None, max_output_tokens, 0.0 if temperature is None else temperature
                )
                or ""
            )

        def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            text = prompt.to_string()
            # n>1 by looping: the provider returns one completion per call and
            # `multiple_completion_supported` is False, so RAGAS knows not to expect
            # batching. In practice the metrics we run ask for one.
            return LLMResult(
                generations=[
                    [Generation(text=self._one(text, temperature)) for _ in range(max(n, 1))]
                ]
            )

        async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            # to_thread, not a direct call: RAGAS awaits this from inside its own loop,
            # and blocking that loop on a network round trip would serialise the run.
            return await asyncio.to_thread(
                self.generate_text, prompt, n, temperature, stop, callbacks
            )

    judge = _AppJudge()
    judge.run_config = run_config
    judge.multiple_completion_supported = False
    judge.cache = None
    return judge


class _AppEmbeddings(Embeddings):
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

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


def _tokens(text: str) -> set[str]:
    """Normalised word set. Digits count: '132' is usually the whole answer."""
    text = _AR_MARKS.sub("", text.translate(_AR_DIGITS).lower())
    text = _AR_ALEF.sub("\u0627", text)
    return {t for t in re.findall(r"\w+", text) if len(t) > 2 or t.isdigit()}


def ground_truth_recall(reference: str, contexts: list[str]) -> float | None:
    """Share of the reference answer's words that appear in the retrieved text.

    A crude, free stand-in for context_recall. It is lexical, so it undercounts a
    correct paraphrase and its absolute value means little; it exists to compare
    configurations on the same questions without spending a single judge token.
    Returns None when the reference has nothing to compare (an unanswerable question).
    """
    wanted = _tokens(reference)
    if not wanted:
        return None
    return len(wanted & _tokens(" ".join(contexts))) / len(wanted)


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

    recalls = [
        value
        for r in answerable
        if (value := ground_truth_recall(r.get("reference") or "", r["retrieved_contexts"]))
        is not None
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
        # Both of these reward bigger chunks by construction: more text per hit means
        # more chances to contain the answer. mean_context_chars is reported next to
        # them so that trade is visible instead of being mistaken for quality.
        "ground_truth_token_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
        "mean_context_chars": (
            round(sum(len("".join(r["retrieved_contexts"])) for r in answerable) / len(answerable))
            if answerable
            else None
        ),
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
    data_version = corpus_version(getattr(args, "manifest", DEFAULT_MANIFEST))
    questions = select_questions(json.loads(args.questions.read_text(encoding="utf-8")), args.limit)

    print(f"corpus version {data_version[:16]}  ·  {len(questions)} questions")

    if args.dry_run:
        print("\n--dry-run: nothing was called. Composition of the selected set:")
        print(f"  answerable   {sum(1 for q in questions if q['answerable'])}")
        print(f"  unanswerable {sum(1 for q in questions if not q['answerable'])}")
        print(f"  arabic       {sum(1 for q in questions if q['language'] == 'ar')}")
        return 0

    from importlib.metadata import version

    if data_version == "unknown":
        raise ValueError("evaluation requires a valid corpus manifest for data lineage")
    mode = "retrieval" if args.no_generation else "ragas"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = getattr(args, "output", None)
    resume = getattr(args, "resume", None)
    out = output or resume or args.reports / f"{mode}_{data_version[:12]}_{stamp}.json"
    provenance = {
        "corpus_version": data_version,
        "eval_sha256": hashlib.sha256(
            json.dumps(questions, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "question_ids": [question["id"] for question in questions],
        "questions": len(questions),
        "generation_model": settings.GENERATION_MODEL_ID,
        "generation_max_tokens": settings.GENERATION_DEFAULT_MAX_TOKENS,
        "generation_temperature": settings.GENERATION_DEFAULT_TEMPERATURE,
        "embedding_model": settings.EMBEDDING_MODEL_ID,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "ragas_model": settings.RAGAS_MODEL_ID or settings.GENERATION_MODEL_ID,
        "ragas_max_output_tokens": settings.RAGAS_MAX_OUTPUT_TOKENS,
        "ragas_reasoning_effort": settings.RAGAS_REASONING_EFFORT,
        "evaluation_mode": mode,
    }
    report = {
        "run_at": datetime.now(UTC).isoformat(),
        **provenance,
        "versions": {package: version(package) for package in ("ragas", "mlflow", "openai")},
        "report_path": str(out),
        "status": "collecting",
        "rows": [],
    }
    if resume:
        saved = json.loads(resume.read_text(encoding="utf-8"))
        for key, value in provenance.items():
            if saved.get(key) != value:
                raise ValueError(f"cannot resume: {key} changed")
        report["rows"] = saved["rows"]
        if saved.get("status") == "complete" and saved.get("ragas"):
            validate_scores(saved)
            print(f"reusing complete evaluation {resume}")
            return saved

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

        def checkpoint(rows):
            report["rows"] = rows
            write_report(report, out)

        rows = await collect_samples(
            questions,
            Deps,
            top_k=settings.RETRIEVAL_TOP_K,
            generate=not args.no_generation,
            checkpoint=checkpoint,
            existing_rows=report["rows"],
        )

        if args.with_gates and not args.no_generation:
            print("\nrecording what the online gates would have said:")
            await record_gate_verdicts(rows, Deps)
    finally:
        await vectordb.disconnect()
        await engine.dispose()

    accuracy = retrieval_accuracy(rows)
    print("\nretrieval checks (free, computed locally):")
    for key, value in accuracy.items():
        print(f"  {key:38} {value}")

    report.update(retrieval_accuracy=accuracy, rows=rows, status="generated")
    write_report(report, out)

    if not (args.no_ragas or args.no_generation):
        print("\nscoring with RAGAS (this is the part that costs tokens):")
        missing_answers = [r["id"] for r in rows if r["answerable"] and not r.get("answer")]
        if missing_answers:
            error = f"answerable questions without generated answers: {missing_answers}"
            report.update(status="failed", evaluation_error=error)
            write_report(report, out)
            raise ValueError(error)
        # evaluate() owns its event loop; run it off the app's loop rather than
        # applying nest_asyncio globally to make nested event loops appear to work.
        try:
            outcome = await asyncio.to_thread(
                score_with_ragas, rows, settings, embedding_client, checkpoint=checkpoint
            )
        except Exception as exc:
            report.update(status="failed", evaluation_error=str(exc))
            write_report(report, out)
            raise
        report["ragas_scored"] = outcome["scored"]
        report["ragas"] = outcome["means"]
        # Per-question scores too, not just the averages. An average of 0.8 hides
        # whether that is every question at 0.8 or half at 1.0 and half at 0.6 — and
        # only the second tells you which questions to go and look at.
        for row, scores in zip(
            [r for r in rows if r["answerable"] and r.get("answer")],
            outcome["per_question"],
            strict=True,
        ):
            row["ragas"] = scores
        try:
            validate_scores(report)
        except ValueError as exc:
            report.update(status="failed", evaluation_error=str(exc))
            write_report(report, out)
            raise
        print("\nRAGAS:")
        for name, value in report["ragas"].items():
            print(f"  {name:38} {value}")

    report["status"] = "complete"
    write_report(report, out)
    print(f"\nwrote {out}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument(
        "--output", type=Path, help="explicit report path; default is unique per run"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="reuse saved answers and valid scores; score only missing metrics",
    )
    parser.add_argument("--limit", type=int, default=None, help="balanced subset; 0 means all")
    parser.add_argument(
        "--no-ragas", action="store_true", help="collect and check retrieval, skip the scoring"
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="retrieval only: no generation calls, no judge, no RAGAS (implies --no-ragas)",
    )
    parser.add_argument(
        "--with-gates",
        action="store_true",
        help="also record the online judges' verdicts, for the judge-vs-RAGAS comparison",
    )
    parser.add_argument("--dry-run", action="store_true", help="show what would run, call nothing")
    args = parser.parse_args()
    asyncio.run(main_async(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
