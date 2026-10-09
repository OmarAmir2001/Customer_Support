"""Evaluation failures must preserve evidence and never become successful runs."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_experiment import PRIMARY_METRIC, has_saved_samples, log_one, register_best  # noqa: E402
from run_ragas import (  # noqa: E402
    RAGAS_METRICS,
    _AppEmbeddings,
    collect_samples,
    score_pending_metrics,
    score_with_ragas,
    select_questions,
    validate_scores,
    write_report,
)


def question(identifier="q1", **overrides):
    return {
        "id": identifier,
        "question": "What are the graduation requirements?",
        "language": "en",
        "department": "CS",
        "answerable": True,
        "ground_truth": "135 credits",
        "expected_article": "Article 24",
        "expected_sources": ["CS_2023"],
        **overrides,
    }


def scored_report():
    scores = dict.fromkeys(RAGAS_METRICS, 0.8)
    return {
        "rows": [{"id": "q1", "answerable": True, "answer": "135 credits", "ragas": scores.copy()}],
        "ragas": scores.copy(),
    }


def test_subset_represents_both_languages_departments_and_unanswerables():
    # Ordered like the real data: the leading block is all answerable CS questions.
    questions = [question(f"cs{i}") for i in range(20)] + [
        question("ar", language="ar"),
        question("is", department="IS"),
        question("unknown", answerable=False),
    ]
    selected = select_questions(questions, 8)
    assert len(selected) == len({q["id"] for q in selected}) == 8
    assert {q["language"] for q in selected} == {"ar", "en"}
    assert {q["department"] for q in selected} == {"CS", "IS"}
    assert any(not q["answerable"] for q in selected)
    assert selected == select_questions(questions, 8)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1, 1.1])
def test_missing_or_invalid_per_question_metric_rejects_a_good_average(value):
    report = scored_report()
    report["rows"][0]["ragas"]["faithfulness"] = value
    with pytest.raises(ValueError, match="q1.*faithfulness"):
        validate_scores(report)


def test_generation_failure_cannot_be_silently_excluded_from_scores():
    report = scored_report()
    report["rows"].append({"id": "q2", "answerable": True, "answer": ""})
    with pytest.raises(ValueError, match="q2.*no generated answer"):
        validate_scores(report)


def test_an_unanswerable_question_has_no_fake_ragas_score():
    report = scored_report()
    report["rows"].append({"id": "q2", "answerable": False, "answer": "Not covered"})
    validate_scores(report)
    assert "ragas" not in report["rows"][1]


def test_invalid_evaluation_is_rejected_before_mlflow_is_touched():
    report = scored_report()
    report["rows"][0]["ragas"]["context_recall"] = None
    with pytest.raises(ValueError, match="context_recall"):
        log_one(None, {}, report, {}, PRIMARY_METRIC)


@pytest.mark.asyncio
async def test_ragas_adapter_supports_the_async_methods_the_wrapper_calls():
    class FakeEmbeddings:
        def embed_text(self, texts, mode):
            return [[1.0, 2.0] for _ in (texts if isinstance(texts, list) else [texts])]

    adapter = _AppEmbeddings(FakeEmbeddings())
    assert await adapter.aembed_query("question") == [1.0, 2.0]
    assert await adapter.aembed_documents(["first", "second"]) == [[1.0, 2.0], [1.0, 2.0]]


@pytest.mark.asyncio
async def test_collected_answers_are_checkpointed_before_later_failures(tmp_path):
    class Retrieval:
        async def retrieve(self, **kwargs):
            if kwargs["question"] == "second":
                raise RuntimeError("retrieval unavailable")
            return [SimpleNamespace(text="135 credits", metadata={"source": "CS_2023"})]

    class Generation:
        async def generate_answer(self, **kwargs):
            return "135 credits"

    out = tmp_path / "answers.json"
    deps = SimpleNamespace(retrieval=Retrieval(), generation=Generation())
    with pytest.raises(RuntimeError, match="retrieval unavailable"):
        await collect_samples(
            [question(), question("q2", question="second")],
            deps,
            top_k=5,
            checkpoint=lambda rows: write_report({"rows": rows}, out),
        )
    saved = json.loads(out.read_text())
    assert saved["rows"][0]["answer"] == "135 credits"
    assert saved["rows"][0]["expected_citations"] == ["CS_2023 — Article 24"]


def test_reports_cannot_write_nonstandard_nan_json(tmp_path):
    out = tmp_path / "result.json"
    write_report({"status": "generated"}, out)
    with pytest.raises(ValueError):
        write_report({"faithfulness": float("nan")}, out)
    assert json.loads(out.read_text()) == {"status": "generated"}


def test_quota_failure_stops_scoring_and_resume_reuses_successful_metrics(tmp_path):
    rows = [
        {"id": "q1", "answerable": True, "answer": "135 credits"},
        {"id": "q2", "answerable": True, "answer": "Other answer"},
    ]
    calls = []
    out = tmp_path / "evaluation.json"

    def evaluate_metric(row, metric):
        calls.append((row["id"], metric))
        if metric == "faithfulness":
            raise RuntimeError("HTTP 429: tokens per day limit reached")
        return 0.8

    with pytest.raises(RuntimeError, match="q1 / faithfulness.*429"):
        score_pending_metrics(
            rows, evaluate_metric, checkpoint=lambda rows: write_report({"rows": rows}, out)
        )
    assert calls == [("q1", metric) for metric in RAGAS_METRICS[:3]]
    saved = json.loads(out.read_text())["rows"]
    assert saved[0]["ragas"] == {"context_precision": 0.8, "context_recall": 0.8}

    resumed_calls = []

    def resumed_evaluation(row, metric):
        resumed_calls.append((row["id"], metric))
        return 0.9

    outcome = score_pending_metrics(saved, resumed_evaluation)
    assert ("q1", "context_precision") not in resumed_calls
    assert ("q1", "context_recall") not in resumed_calls
    assert len(resumed_calls) == 6
    assert outcome["means"]["context_precision"] == 0.85
    validate_scores({"rows": saved, "ragas": outcome["means"]})


def test_invalid_saved_metric_is_rescored_but_unanswerables_are_excluded():
    report = scored_report()
    report["rows"][0]["ragas"]["context_recall"] = None
    report["rows"].append({"id": "q2", "answerable": False, "answer": "Not covered"})
    calls = []

    def evaluate_metric(row, metric):
        calls.append((row["id"], metric))
        return 0.9

    outcome = score_pending_metrics(report["rows"], evaluate_metric)
    assert calls == [("q1", "context_recall")]
    assert outcome["scored"] == 1


def test_real_ragas_and_openai_sdk_do_not_retry_daily_quota_errors(monkeypatch):
    import httpx
    import langchain_openai

    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    requests = []

    def quota_error(request):
        requests.append(request)
        return httpx.Response(
            429,
            json={"error": {"message": "tokens per day limit reached", "type": "tokens"}},
            headers={"retry-after": "600"},
        )

    original_chat = langchain_openai.ChatOpenAI
    transport = httpx.MockTransport(quota_error)

    def offline_chat(**kwargs):
        return original_chat(
            **kwargs,
            http_async_client=httpx.AsyncClient(transport=transport),
            http_client=httpx.Client(transport=transport),
        )

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", offline_chat)
    settings = SimpleNamespace(
        # Names the backend explicitly: this test is about the OpenAI-SDK judge path
        # and its no-retry behaviour, so it must not follow the Cohere branch that is
        # now the default.
        RAGAS_BACKEND="OPENAI",
        RAGAS_MODEL_ID="test-model",
        GENERATION_MODEL_ID="test-model",
        GROQ_API_KEY="fake-test-key",
        RAGAS_API_URL="https://judge.example/v1",
        OPENAI_API_URL="https://judge.example/v1",
        RAGAS_MAX_OUTPUT_TOKENS=4096,
        RAGAS_REASONING_EFFORT="low",
        RAGAS_TIMEOUT_SECONDS=2,
    )
    row = {
        "id": "q1",
        "question": "How many credits?",
        "answerable": True,
        "answer": "135 credits",
        "reference": "135 credits",
        "retrieved_contexts": ["Graduation requires 135 credits."],
    }
    with pytest.raises(RuntimeError, match="429"):
        score_with_ragas([row], settings, None)
    assert len(requests) == 1


def test_resume_skips_indexing_only_for_matching_fully_collected_samples(tmp_path):
    out = tmp_path / "evaluation.json"
    report = {"corpus_version": "abc", "question_ids": ["q1"], **scored_report()}
    write_report(report, out)
    assert has_saved_samples(out, [question()], {"content_sha256": "abc"}, generate=True)
    assert not has_saved_samples(out, [question()], {"content_sha256": "def"}, generate=True)
    assert not has_saved_samples(out, [question("q2")], {"content_sha256": "abc"}, generate=True)
    report["rows"] = []
    write_report(report, out)
    assert not has_saved_samples(out, [question()], {"content_sha256": "abc"}, generate=True)


def registry_run(identifier, score, *, status="FINISHED", context_chars=1000):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=identifier, status=status),
        data=SimpleNamespace(
            metrics={**dict.fromkeys(RAGAS_METRICS, score), "mean_context_chars": context_chars},
            params={"chunk_size": "1000", "overlap": "50", "corpus_sha256": "abc"},
        ),
    )


@pytest.fixture
def registry(monkeypatch):
    import mlflow.tracking

    class Registry:
        runs = []
        writes = []

        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="1")

        def search_runs(self, *args, **kwargs):
            return sorted(self.runs, key=lambda run: run.data.metrics[PRIMARY_METRIC], reverse=True)

        def create_registered_model(self, name):
            self.writes.append(("model", name))

        def create_model_version(self, **kwargs):
            self.writes.append(("version", kwargs["run_id"]))
            return SimpleNamespace(version="3")

        def transition_model_version_stage(self, *args, **kwargs):
            assert kwargs["archive_existing_versions"] is True
            self.writes.append(("stage", args[1]))

        def set_registered_model_alias(self, *args):
            self.writes.append(("alias", args[2]))

        def get_model_version_by_alias(self, *args):
            return SimpleNamespace(version="3", current_stage="Production")

    fake = Registry()
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: fake)
    return fake


def test_below_threshold_run_cannot_move_production(registry):
    registry.runs = [registry_run("bad", 0.7)]
    with pytest.raises(ValueError, match="not promoted"):
        register_best(None, "experiment", PRIMARY_METRIC)
    assert registry.writes == []


def test_partial_failed_and_unrelated_runs_cannot_win(registry):
    partial = registry_run("partial", 1.0)
    partial.data.metrics.pop("context_recall")
    registry.runs = [
        partial,
        registry_run("failed", 1.0, status="FAILED"),
        registry_run("other-sweep", 1.0),
        registry_run("valid", 0.85),
    ]
    result = register_best(
        None, "experiment", PRIMARY_METRIC, run_ids=["partial", "failed", "valid"]
    )
    assert result["run_id"] == "valid"
    assert registry.writes[-2:] == [("stage", "3"), ("alias", "3")]


def test_tie_policy_cannot_select_a_candidate_below_the_quality_floor(registry):
    registry.runs = [registry_run("good", 0.78), registry_run("bad", 0.74, context_chars=100)]
    result = register_best(None, "experiment", PRIMARY_METRIC)
    assert result["run_id"] == "good"
