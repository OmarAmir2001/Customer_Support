"""Query drift and token cost — the two things a dashboard claims and code must do.

Both are instrumentation on a path that has already succeeded, so the property that
matters most is that neither can fail a request. That is asserted explicitly, not
assumed from the try/except being visible in the source.
"""

import json
import math

import pytest

from customer_support.controllers.RetrievalController import RetrievalController
from customer_support.helpers.drift import centroid, cosine, load_baseline, normalise
from customer_support.models.db_schemas import RetrievedDocument
from customer_support.utils.metrics import QUERY_DRIFT, TOKENS, record_tokens


class FakeSettings:
    RETRIEVAL_OVERFETCH_FACTOR = 3
    ASSETS_DIR = "assets"


class FakeEmbedding:
    def __init__(self, vector=None):
        self.vector = vector or [0.1] * 8

    def embed_text(self, text, document_type=None):
        return [self.vector]


class FakeVectorDB:
    async def search_by_vector(self, collection_name, vector, limit):
        return [
            RetrievedDocument(
                text="t", score=0.9, metadata={"source": "CS_2023", "department": "CS"}
            )
        ]


# -------------------------------------------------------------------- cosine


def test_cosine_of_a_vector_with_itself_is_one():
    vector = [0.3, -0.4, 0.5]
    assert cosine(vector, vector) == pytest.approx(1.0)


def test_cosine_ignores_magnitude():
    """Embedding magnitude varies with text length, so drift must not read it.

    If this were a dot product instead, a long question would look like drift.
    """
    assert cosine([1.0, 0.0], [7.0, 0.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_refuses_mismatched_dimensions_instead_of_guessing():
    """A baseline from a different EMBEDDING_MODEL_SIZE is not comparable.

    None here is what makes the controller log `drift_not_comparable` instead of
    reporting a configuration mismatch as a change in student behaviour.
    """
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) is None


def test_cosine_and_normalise_survive_a_zero_vector():
    assert normalise([0.0, 0.0]) is None
    assert cosine([0.0, 0.0], [1.0, 1.0]) is None


# ------------------------------------------------------------------ centroid


def test_centroid_normalises_before_averaging():
    """Otherwise the longest questions dominate the reference point.

    Two vectors pointing the same way with very different magnitudes must produce a
    centroid pointing that way — not one pulled towards the larger.
    """
    reference = centroid([[1.0, 0.0], [100.0, 0.0]])
    assert reference == pytest.approx([1.0, 0.0])


def test_centroid_of_opposing_directions_is_the_bisector():
    reference = centroid([[1.0, 0.0], [0.0, 1.0]])
    assert reference == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2)])


def test_centroid_rejects_ragged_input():
    with pytest.raises(ValueError):
        centroid([[1.0, 0.0], [1.0, 0.0, 0.0]])


def test_centroid_of_nothing_is_none():
    assert centroid([]) is None
    assert centroid([[0.0, 0.0]]) is None


# ------------------------------------------------------------------ baseline


def test_a_missing_baseline_disables_drift_rather_than_failing_boot(tmp_path):
    """A fresh clone has no baseline: building one costs embedding calls."""
    assert load_baseline(tmp_path / "nope.json") is None


def test_an_unreadable_baseline_is_reported_as_absent(tmp_path):
    bad = tmp_path / "drift_baseline.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_baseline(bad) is None

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"built_at": "now"}), encoding="utf-8")
    assert load_baseline(empty) is None


def test_the_committed_baseline_matches_the_configured_embedding_size():
    """The baseline in the repo must be usable by the app as configured.

    384 is VECTOR_DB_DEFAULT_VECTOR_SIZE and EMBEDDING_MODEL_SIZE for
    embed-multilingual-light-v3.0. A baseline of any other width silently disables
    the metric, and this is the cheapest place to notice.
    """
    baseline = load_baseline()
    if baseline is None:
        pytest.skip("no baseline committed")
    assert baseline["dimensions"] == 384
    assert len(baseline["centroid"]) == 384
    # Unit length, which is what makes the request-time cosine a plain dot product.
    magnitude = math.sqrt(sum(value * value for value in baseline["centroid"]))
    assert magnitude == pytest.approx(1.0, abs=1e-9)


# ----------------------------------------------------- drift in the hot path


def _histogram_count() -> float:
    for metric in QUERY_DRIFT.collect():
        for sample in metric.samples:
            if sample.name.endswith("_count"):
                return sample.value
    return 0.0


@pytest.mark.asyncio
async def test_retrieval_records_drift_when_a_baseline_is_configured():
    before = _histogram_count()

    controller = RetrievalController(
        vectordb_client=FakeVectorDB(),
        embedding_client=FakeEmbedding([1.0] + [0.0] * 7),
        collection_name="collection_3",
        settings=FakeSettings(),
        drift_centroid=[1.0] + [0.0] * 7,
    )
    await controller.retrieve(question="how many credit hours?", department="CS")

    assert _histogram_count() == before + 1


@pytest.mark.asyncio
async def test_a_mismatched_baseline_does_not_fail_the_question():
    """The point of the whole exercise: instrumentation cannot break retrieval.

    An 8-dim query against a 3-dim baseline is a misconfiguration. It must produce a
    log line and an answer, not an exception on a student's question.
    """
    controller = RetrievalController(
        vectordb_client=FakeVectorDB(),
        embedding_client=FakeEmbedding([0.1] * 8),
        collection_name="collection_3",
        settings=FakeSettings(),
        drift_centroid=[1.0, 0.0, 0.0],
    )
    chunks = await controller.retrieve(question="how many credit hours?", department="CS")
    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_no_baseline_means_no_drift_and_no_error():
    before = _histogram_count()
    controller = RetrievalController(
        vectordb_client=FakeVectorDB(),
        embedding_client=FakeEmbedding(),
        collection_name="collection_3",
        settings=FakeSettings(),
    )
    await controller.retrieve(question="how many credit hours?", department="CS")
    assert _histogram_count() == before


# ---------------------------------------------------------------- token cost


def _tokens(model: str, kind: str) -> float:
    return TOKENS.labels(model, kind)._value.get()


def test_tokens_are_counted_by_model_and_direction():
    before_prompt = _tokens("test-model", "prompt")
    before_completion = _tokens("test-model", "completion")

    record_tokens("test-model", 100, 42)

    assert _tokens("test-model", "prompt") == before_prompt + 100
    assert _tokens("test-model", "completion") == before_completion + 42


def test_an_unnamed_model_is_bucketed_not_dropped():
    before = _tokens("unknown", "prompt")
    record_tokens(None, 7, None)
    assert _tokens("unknown", "prompt") == before + 7


def test_absent_usage_records_nothing_rather_than_zero():
    """A provider that reports no usage must not look like a free call.

    Incrementing by zero would be indistinguishable from a counted call on the
    dashboard; not incrementing leaves the gap visible against the request count.
    """
    before = _tokens("test-model", "prompt")
    record_tokens("test-model", None, None)
    record_tokens("test-model", 0, 0)
    assert _tokens("test-model", "prompt") == before


# ---------------------------------------------- usage extraction per provider


class _Usage:
    prompt_tokens = 11
    completion_tokens = 5


class _OpenAIResponse:
    usage = _Usage()


class _CohereUnits:
    input_tokens = 13
    output_tokens = 7


class _CohereMeta:
    billed_units = _CohereUnits()


class _CohereResponse:
    meta = _CohereMeta()


def test_openai_usage_extraction_reads_the_sdk_shape():
    from customer_support.stores.llm.providers.OpenAIProvider import OpenAIProvider

    provider = OpenAIProvider.__new__(OpenAIProvider)  # no client, no network
    before_in = _tokens("groq-model", "prompt")
    before_out = _tokens("groq-model", "completion")
    provider._record_usage(_OpenAIResponse(), "groq-model")
    assert _tokens("groq-model", "prompt") == before_in + 11
    assert _tokens("groq-model", "completion") == before_out + 5


def test_cohere_usage_comes_from_meta_billed_units_not_usage():
    """Cohere reports under `meta`, and billed units are what you pay for.

    Reading `usage` (the OpenAI shape) would silently record nothing for the
    provider that was the paid one — a cost panel that is flat because it is blind.
    """
    from customer_support.stores.llm.providers.CohereProvider import CohereProvider

    provider = CohereProvider.__new__(CohereProvider)
    before_in = _tokens("cohere-model", "prompt")
    before_out = _tokens("cohere-model", "completion")
    provider._record_usage(_CohereResponse(), "cohere-model")
    assert _tokens("cohere-model", "prompt") == before_in + 13
    assert _tokens("cohere-model", "completion") == before_out + 7


@pytest.mark.parametrize("response", [None, object(), _CohereResponse()])
def test_usage_extraction_never_raises_on_an_unexpected_shape(response):
    """A renamed SDK field must not turn a good answer into an error."""
    from customer_support.stores.llm.providers.OpenAIProvider import OpenAIProvider

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._record_usage(response, "whatever")
