#!/usr/bin/env python
"""Chunking experiments, tracked in MLflow. COSTS MONEY.

    uv run --extra dev python scripts/run_experiment.py --dry-run
    uv run --extra dev python scripts/run_experiment.py --limit 20
    uv run --extra dev python scripts/run_experiment.py --limit 20 --register
    uv run --extra dev python scripts/run_experiment.py --retrieval-only --register

One MLflow run per chunking configuration. Each one:

    1. rebuilds the corpus at that chunk_size / overlap   (free, offline)
    2. re-indexes it into pgvector                        (Cohere calls)
    3. scores it with RAGAS against the eval set          (Groq calls — the expensive part)
    4. logs params, metrics and artifacts

**What "the model" is on this track.** There are no weights to register. The thing
that varies, and that makes the system measurably better or worse, is the *retrieval
configuration* — chunk_size, overlap, the embedding model. So that is what gets
compared and what gets promoted, with RAGAS faithfulness as the deciding metric.

**The first configuration is not arbitrary.** `POST /process` still defaults to
`chunk_size=100`, which produces 828 chunks from the same handbooks that 1000 turns
into 199. At 100 characters a handbook article is shredded across many fragments.
That default is very likely wrong, but "likely" is not a measurement, so it goes in
the grid as a run rather than being quietly changed.

**Lineage.** Every run records the corpus `content_sha256`, so a metric is attributable
to the exact data that produced it. A faithfulness score with no data version attached
cannot be explained a month later.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Quieten the "load the tracing skill" banner; this script does tracking, not tracing.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

CORPUS_DIR = REPO_ROOT / "data" / "corpus"
REPORTS = REPO_ROOT / "reports"

#: The grid. chunk_size is the axis that matters; the last entry isolates overlap so a
#: difference cannot be attributed to two changes at once.
GRID = [
    {"chunk_size": 100, "overlap": 50},  # the current /process default
    {"chunk_size": 500, "overlap": 50},
    {"chunk_size": 1000, "overlap": 50},  # the current params.yaml value
    {"chunk_size": 1500, "overlap": 50},
    {"chunk_size": 1000, "overlap": 200},  # same size, more overlap
]

#: The metric promotion is decided on. Faithfulness, because the failure this system
#: must not have is a confidently wrong answer to a student — a grounded answer that
#: reads a little awkwardly is recoverable, an ungrounded one is not.
PRIMARY_METRIC = "faithfulness"

#: The metric promotion is decided on when --retrieval-only is set. No generator and no
#: judge are involved, so the number is deterministic: two identical runs agree exactly.
#: It is lexical and it rewards bigger chunks, which is why mean_context_chars is logged
#: beside it. Read the pair, not the winner alone.
RETRIEVAL_METRIC = "ground_truth_token_recall"

REGISTERED_MODEL = "HandbookRetrieval"


def run(cmd: list[str], label: str) -> None:
    """Run a step, failing loudly. A half-completed configuration would log metrics
    that do not correspond to what is actually in the vector store."""
    print(f"    {label}...", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"{label} failed with exit {result.returncode}")


def corpus_manifest() -> dict:
    return json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))


def git_commit() -> str:
    """The code version, so a metric is attributable to the code as well as the data."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


async def score(limit: int | None, with_gates: bool, retrieval_only: bool = False) -> dict:
    """Reuse the RAGAS harness in-process rather than shelling out to it.

    In-process so the returned scores are objects rather than something scraped back
    out of stdout.
    """
    import run_ragas

    args = argparse.Namespace(
        questions=REPO_ROOT / "data" / "eval" / "questions.json",
        reports=REPORTS,
        limit=limit,
        no_ragas=retrieval_only,
        no_generation=retrieval_only,
        with_gates=with_gates,
        dry_run=False,
    )
    await run_ragas.main_async(args)

    # main_async writes the report keyed by corpus hash; read back the one it just wrote.
    version = corpus_manifest()["content_sha256"][:12]
    return json.loads((REPORTS / f"ragas_{version}.json").read_text(encoding="utf-8"))


def log_one(mlflow, config: dict, report: dict, manifest: dict, metric: str) -> str:
    """One MLflow run. Params and metrics together, never one without the other."""
    with mlflow.start_run(run_name=f"chunk{config['chunk_size']}_ov{config['overlap']}") as run:
        mlflow.set_tags({"git_commit": git_commit(), "framework": "chunking-sweep"})
        mlflow.log_params(
            {
                **config,
                "embedding_model": report["embedding_model"],
                "generation_model": report["generation_model"],
                "retrieval_top_k": report["retrieval_top_k"],
                # The data version. Without it a metric is unreproducible.
                "corpus_sha256": manifest["content_sha256"],
                "corpus_records": manifest["records"],
                "corpus_sections": manifest["sections"],
                "eval_questions": report["questions"],
            }
        )

        metrics = dict(report.get("ragas") or {})
        accuracy = report["retrieval_accuracy"]
        # Cheap local checks alongside the RAGAS ones. A high context_precision with
        # the wrong handbook cited is still wrong, and only this catches it.
        if accuracy.get("expected_article_hit_rate") is not None:
            metrics["expected_article_hit_rate"] = accuracy["expected_article_hit_rate"]
        for key in ("ground_truth_token_recall", "mean_context_chars"):
            if accuracy.get(key) is not None:
                metrics[key] = accuracy[key]
        metrics["department_filter_leaks"] = len(accuracy.get("department_filter_leaks") or [])
        metrics["corpus_chunks"] = manifest["records"]
        metrics["max_chunk_chars"] = manifest["char_count_max"]
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})

        # Artifacts: the corpus itself and the full report, so a run can be inspected
        # rather than just compared.
        mlflow.log_artifact(str(CORPUS_DIR / "corpus.json"), artifact_path="corpus")
        mlflow.log_artifact(str(CORPUS_DIR / "manifest.json"), artifact_path="corpus")
        report_path = REPORTS / f"ragas_{manifest['content_sha256'][:12]}.json"
        if report_path.exists():
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")

        print(f"    logged run {run.info.run_id[:12]}  {metric}={metrics.get(metric)}")
        return run.info.run_id


def register_best(mlflow, experiment_name: str, metric: str) -> None:
    """Register the winning configuration and promote it to Production.

    The 'model' is a configuration, so what gets registered is the RUN — its params
    are the thing you would apply. Promotion moves a pointer; nothing is rebuilt,
    which is what makes a rollback a decision rather than a deploy.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=50,
    )
    scored = [r for r in runs if r.data.metrics.get(metric) is not None]
    if not scored:
        print("  nothing to register: no run recorded the primary metric")
        return

    best, *rest = scored
    best_value = best.data.metrics[metric]
    runner_up = rest[0].data.metrics[metric] if rest else None

    print(f"\n  best by {metric}: {best_value:.4f}")
    print(
        f"    chunk_size={best.data.params.get('chunk_size')} "
        f"overlap={best.data.params.get('overlap')}"
    )
    if runner_up is not None:
        margin = best_value - runner_up
        print(f"    runner-up {runner_up:.4f}  ·  margin {margin:.4f}")
        # A margin smaller than run-to-run noise is a coin flip wearing a lab coat.
        # RAGAS uses an LLM judge, so its own variance is not negligible; this is a
        # warning rather than a refusal because the noise floor has not been measured
        # yet (two identical runs would measure it).
        if margin < 0.05:
            print(
                "    WARNING: margin is within plausible judge noise. Treat this as "
                "'no clear winner' until you have measured run-to-run variance."
            )

    try:
        client.create_registered_model(REGISTERED_MODEL)
    except Exception:
        pass  # already exists

    version = client.create_model_version(
        name=REGISTERED_MODEL,
        # The run is the artifact: its params ARE the configuration.
        source=f"runs:/{best.info.run_id}/corpus",
        run_id=best.info.run_id,
        description=(
            f"chunk_size={best.data.params.get('chunk_size')}, "
            f"overlap={best.data.params.get('overlap')}, "
            f"{metric}={best_value:.4f}, "
            f"corpus={best.data.params.get('corpus_sha256', '?')[:12]}"
        ),
    )
    # An alias, and the legacy stage as well: the rubric asks for "promoted to
    # Production", and 3.x prefers aliases while the UI still shows stages.
    client.set_registered_model_alias(REGISTERED_MODEL, "production", version.version)
    try:
        client.transition_model_version_stage(REGISTERED_MODEL, version.version, "Production")
    except Exception as exc:
        print(f"    (stage transition unavailable: {exc})")

    print(f"  registered {REGISTERED_MODEL} v{version.version} -> alias 'production'")


async def main_async(args) -> int:
    import mlflow

    grid = GRID[: args.configs] if args.configs else GRID
    # Retrieval-only runs cost no generation or judge tokens, so the whole question set
    # is the sensible default there; the RAGAS run keeps its small default.
    limit = args.limit if args.limit is not None else (0 if args.retrieval_only else 20)
    metric = RETRIEVAL_METRIC if args.retrieval_only else PRIMARY_METRIC

    print(f"{len(grid)} configurations · eval limit {limit or 'all'} · decided on {metric}")
    for config in grid:
        print(f"  chunk_size={config['chunk_size']:>5}  overlap={config['overlap']:>4}")

    if args.dry_run:
        print("\n--dry-run: nothing built, indexed, scored or logged.")
        if args.retrieval_only:
            print(
                "Per configuration this would cost: the embedding calls to index the corpus "
                f"plus one per question ({limit or 62}). No generation, no judge."
            )
        else:
            print(
                "Per configuration this would cost: ~3 embedding calls, "
                f"{limit or 62} generations, and ~{4 * (limit or 62)} RAGAS judge calls."
            )
        return 0

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    for index, config in enumerate(grid, start=1):
        print(
            f"\n[{index}/{len(grid)}] chunk_size={config['chunk_size']} overlap={config['overlap']}"
        )

        run(
            [
                "uv",
                "run",
                "--extra",
                "dev",
                "python",
                "scripts/build_corpus.py",
                "--chunk-size",
                str(config["chunk_size"]),
                "--overlap",
                str(config["overlap"]),
            ],
            "building corpus",
        )
        manifest = corpus_manifest()
        print(f"    {manifest['records']} chunks, version {manifest['content_sha256'][:12]}")

        run(["uv", "run", "--extra", "dev", "python", "scripts/index_corpus.py"], "indexing")

        if args.retrieval_only:
            print("    measuring retrieval (no generation, no judge)...", flush=True)
        else:
            print("    scoring with RAGAS (slow)...", flush=True)
        report = await score(limit, args.with_gates, retrieval_only=args.retrieval_only)

        log_one(mlflow, config, report, manifest, metric)

    if args.register:
        register_best(mlflow, args.experiment, metric)

    print(f"\nCompare the runs at {args.tracking_uri}/#/experiments")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="handbook-chunking")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="eval questions per run; 0 means all. Default 20, or all with --retrieval-only",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="no generation and no judge: rank configs on free retrieval metrics",
    )
    parser.add_argument("--configs", type=int, default=None, help="only the first N configs")
    parser.add_argument("--with-gates", action="store_true", help="also record the judge verdicts")
    parser.add_argument(
        "--register", action="store_true", help="register the winner and promote it to Production"
    )
    parser.add_argument("--dry-run", action="store_true", help="show the plan and the cost")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())