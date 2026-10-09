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

The grid keeps `chunk_size=100` as a historical baseline. `/process` now defaults
to 1000; comparing it with smaller chunks gives that change measurable evidence.

**Lineage.** Every run records the corpus `content_sha256`, so a metric is attributable
to the exact data that produced it. A faithfulness score with no data version attached
cannot be explained a month later.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
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
    {"chunk_size": 100, "overlap": 50},  # historical baseline
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


def corpus_manifest(directory: Path = CORPUS_DIR) -> dict:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def has_saved_samples(report_path: Path, questions: list[dict], manifest: dict, *, generate: bool):
    """A matching, fully collected report needs no fresh index or embedding calls."""
    if not report_path.exists():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("corpus_version") != manifest["content_sha256"]:
        return False
    if report.get("question_ids") != [question["id"] for question in questions]:
        return False
    saved = {row["id"]: row for row in report.get("rows", [])}
    return all(
        question["id"] in saved and (not generate or saved[question["id"]].get("answer"))
        for question in questions
    )


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


async def score(
    limit: int | None,
    with_gates: bool,
    retrieval_only: bool = False,
    output: Path | None = None,
    resume: Path | None = None,
    corpus_dir: Path = CORPUS_DIR,
) -> dict:
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
        output=output,
        resume=resume,
        manifest=corpus_dir / "manifest.json",
    )
    return await run_ragas.main_async(args)


def log_one(
    mlflow,
    config: dict,
    report: dict,
    manifest: dict,
    metric: str,
    corpus_dir: Path = CORPUS_DIR,
) -> str:
    """One MLflow run. Params and metrics together, never one without the other."""
    from run_ragas import validate_scores

    if metric == PRIMARY_METRIC:
        validate_scores(report)
    if report["corpus_version"] != manifest["content_sha256"]:
        raise ValueError("evaluation report does not match this corpus")
    active = mlflow.active_run()
    context = (
        contextlib.nullcontext(active)
        if active
        else mlflow.start_run(run_name=f"chunk{config['chunk_size']}_ov{config['overlap']}")
    )
    with context as run:
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "framework": "chunking-sweep",
                "evaluation_mode": report["evaluation_mode"],
                "eval_sha256": report["eval_sha256"],
            }
        )
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
                "eval_sha256": report["eval_sha256"],
                "ragas_model": report["ragas_model"],
                "generation_max_tokens": report["generation_max_tokens"],
                "generation_temperature": report["generation_temperature"],
                "ragas_max_output_tokens": report["ragas_max_output_tokens"],
                "ragas_reasoning_effort": report["ragas_reasoning_effort"],
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
        metrics["mean_chunk_chars"] = manifest["char_count_total"] / manifest["records"]
        metrics["generated_answers"] = sum(bool(row.get("answer")) for row in report["rows"])
        if report.get("ragas_scored") is not None:
            metrics["ragas_scored"] = report["ragas_scored"]
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})

        # Artifacts: the corpus itself and the full report, so a run can be inspected
        # rather than just compared.
        mlflow.log_artifact(str(corpus_dir / "corpus.json"), artifact_path="corpus")
        mlflow.log_artifact(str(corpus_dir / "manifest.json"), artifact_path="corpus")
        mlflow.log_artifact(report["report_path"], artifact_path="evaluation")
        mlflow.log_artifact(str(REPO_ROOT / "data/eval/questions.json"), artifact_path="evaluation")
        mlflow.log_dict(
            {
                **config,
                **{
                    key: report[key]
                    for key in (
                        "embedding_model",
                        "generation_model",
                        "retrieval_top_k",
                        "corpus_version",
                        "eval_sha256",
                        "generation_max_tokens",
                        "generation_temperature",
                    )
                },
            },
            "configuration/retrieval_config.json",
        )
        mlflow.log_dict(report["versions"], "environment/versions.json")
        for filename in ("scripts/run_ragas.py", "scripts/run_experiment.py", "uv.lock"):
            mlflow.log_artifact(str(REPO_ROOT / filename), artifact_path="code")

        print(f"    logged run {run.info.run_id[:12]}  {metric}={metrics.get(metric)}")
        return run.info.run_id


def register_best(
    mlflow,
    experiment_name: str,
    metric: str,
    *,
    run_ids: list[str] | None = None,
    tie_margin: float = 0.05,
    minimum_faithfulness: float = 0.75,
) -> dict:
    """Register the winning configuration and promote it to Production.

    The 'model' is a configuration, so what gets registered is the RUN — its params
    are the thing you would apply. Promotion moves a pointer; nothing is rebuilt,
    which is what makes a rollback a decision rather than a deploy.
    """
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=50,
    )
    scored = [
        r
        for r in runs
        if r.info.status == "FINISHED"
        and (run_ids is None or r.info.run_id in run_ids)
        and math.isfinite(r.data.metrics.get(metric, float("nan")))
    ]
    if metric == PRIMARY_METRIC:
        from run_ragas import RAGAS_METRICS

        scored = [
            r
            for r in scored
            if all(math.isfinite(r.data.metrics.get(key, float("nan"))) for key in RAGAS_METRICS)
        ]
    if not scored:
        raise ValueError("nothing to register: no complete run recorded the required metrics")

    peak = scored[0].data.metrics[metric]
    # This is a conservative tie policy, not a measured noise floor. Within the
    # margin prefer less context, rather than claiming a tiny gain is significant.
    tied = [
        r
        for r in scored
        if peak - r.data.metrics[metric] <= tie_margin
        and (metric != PRIMARY_METRIC or r.data.metrics[metric] >= minimum_faithfulness)
    ]
    if not tied:
        raise ValueError(f"faithfulness {peak:.4f} < {minimum_faithfulness}; not promoted")
    best = min(
        tied,
        key=lambda r: (
            r.data.metrics.get("mean_context_chars", float("inf")),
            int(r.data.params.get("overlap", "0")),
        ),
    )
    rest = [r for r in scored if r.info.run_id != best.info.run_id]
    best_value = best.data.metrics[metric]
    if metric == PRIMARY_METRIC and best_value < minimum_faithfulness:
        raise ValueError(f"faithfulness {best_value:.4f} < {minimum_faithfulness}; not promoted")
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
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise

    version = client.create_model_version(
        name=REGISTERED_MODEL,
        # The run is the artifact: its params ARE the configuration.
        source=f"runs:/{best.info.run_id}/configuration",
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
    client.transition_model_version_stage(
        REGISTERED_MODEL, version.version, "Production", archive_existing_versions=True
    )
    client.set_registered_model_alias(REGISTERED_MODEL, "production", version.version)
    promoted = client.get_model_version_by_alias(REGISTERED_MODEL, "production")
    if promoted.version != version.version or promoted.current_stage != "Production":
        raise RuntimeError("registry stage and production alias do not agree")

    print(f"  registered {REGISTERED_MODEL} v{version.version} -> alias 'production'")
    return {
        "run_id": best.info.run_id,
        "version": version.version,
        "metric": metric,
        "value": best_value,
        "tie_margin": tie_margin,
        "config": {key: int(best.data.params[key]) for key in ("chunk_size", "overlap")},
    }


async def main_async(args) -> int:
    import mlflow
    from run_ragas import DEFAULT_QUESTIONS, select_questions

    grid = GRID[: args.configs] if args.configs else GRID
    if args.chunk_size is not None:
        grid = [{"chunk_size": args.chunk_size, "overlap": args.overlap}]
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
                "Each configuration embeds its corpus and selected questions, generates "
                f"{limit or 62} answers, and scores four metrics per answerable question. "
                "Each metric may require multiple judge calls."
            )
        return 0

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    # Experiments use their own rebuildable index. They must not change the live
    # assistant's handbook chunks or include its promoted ticket answers in scores.
    from customer_support.helpers.config import get_settings

    if get_settings().KB_COLLECTION_NAME == f"collection_{args.project_id}":
        raise ValueError("experiment project-id points at the live collection")
    os.environ["KB_COLLECTION_NAME"] = f"collection_{args.project_id}"
    get_settings.cache_clear()
    session_dir = args.resume_dir or REPORTS / "experiments" / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    summary = {"experiment": args.experiment, "metric": metric, "runs": []}

    for index, config in enumerate(grid, start=1):
        print(
            f"\n[{index}/{len(grid)}] chunk_size={config['chunk_size']} overlap={config['overlap']}"
        )

        trial_dir = session_dir / f"chunk{config['chunk_size']}_ov{config['overlap']}"
        corpus_dir = trial_dir / "corpus"
        report_path = trial_dir / "evaluation.json"
        run(
            [
                sys.executable,
                "scripts/build_corpus.py",
                "--output-dir",
                str(corpus_dir),
                "--chunk-size",
                str(config["chunk_size"]),
                "--overlap",
                str(config["overlap"]),
            ],
            "building corpus",
        )
        manifest = corpus_manifest(corpus_dir)
        print(f"    {manifest['records']} chunks, version {manifest['content_sha256'][:12]}")

        questions = select_questions(
            json.loads(DEFAULT_QUESTIONS.read_text(encoding="utf-8")), limit
        )
        if has_saved_samples(report_path, questions, manifest, generate=not args.retrieval_only):
            print("    reusing saved samples; skipping indexing and embedding calls", flush=True)
        else:
            run(
                [
                    sys.executable,
                    "scripts/index_corpus.py",
                    "--corpus",
                    str(corpus_dir / "corpus.json"),
                    "--project-id",
                    str(args.project_id),
                    "--reset",
                ],
                "indexing experiment collection",
            )

        if args.retrieval_only:
            print("    measuring retrieval (no generation, no judge)...", flush=True)
        else:
            print("    scoring with RAGAS (slow)...", flush=True)
        with mlflow.start_run(run_name=f"chunk{config['chunk_size']}_ov{config['overlap']}"):
            mlflow.log_params(config)
            mlflow.set_tag("git_commit", git_commit())
            try:
                report = await score(
                    limit,
                    args.with_gates,
                    retrieval_only=args.retrieval_only,
                    output=report_path,
                    resume=report_path if report_path.exists() else None,
                    corpus_dir=corpus_dir,
                )
                run_id = log_one(mlflow, config, report, manifest, metric, corpus_dir)
            except BaseException:
                # A rate-limited or malformed evaluation is a FAILED run with its
                # collected answers attached, never a successful metric-only run.
                if report_path.exists():
                    mlflow.log_artifact(str(report_path), artifact_path="evaluation")
                raise
        summary["runs"].append(
            {
                "run_id": run_id,
                "config": config,
                "report": str(report_path),
                "metrics": report.get("ragas", {}),
            }
        )
        (session_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.register:
        summary["production"] = register_best(
            mlflow,
            args.experiment,
            metric,
            run_ids=[entry["run_id"] for entry in summary["runs"]],
            tie_margin=args.tie_margin,
            minimum_faithfulness=args.minimum_faithfulness,
        )
        (session_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nCompare the runs at {args.tracking_uri}/#/experiments")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="handbook-ragas-v1")
    parser.add_argument(
        "--project-id", type=int, default=9001, help="isolated experiment collection"
    )
    parser.add_argument("--resume-dir", type=Path, help="reuse saved answers from a previous sweep")
    parser.add_argument("--chunk-size", type=int, help="evaluate one configuration instead of grid")
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--tie-margin", type=float, default=0.05)
    parser.add_argument("--minimum-faithfulness", type=float, default=0.75)
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
    if args.project_id <= 0:
        parser.error("project-id must be positive and separate from the live collection")
    if args.limit is not None and args.limit < 0:
        parser.error("limit must be non-negative")
    if args.configs is not None and not 1 <= args.configs <= len(GRID):
        parser.error("configs must be between 1 and 5")
    if not 0 <= args.tie_margin <= 1 or not 0 <= args.minimum_faithfulness <= 1:
        parser.error("tie margin and minimum faithfulness must be between 0 and 1")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
