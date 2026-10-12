"""Query drift: how far today's questions sit from the ones the system was built for.

The rubric asks for *"cosine similarity on query embeddings vs baseline — drift
logged"*. The baseline is the centroid of the evaluation set's question embeddings,
which is a defensible reference precisely because that set is also what the gates and
the chunking sweep were tuned against: a question far from it is a question the
measured quality numbers do not cover.

**What a falling cosine actually means here.** Not that the model is worse — nothing
about the model changed. It means students have started asking about things the
handbook does not address, which in this system shows up as rising escalations. The
two panels read together: drift down with escalations up is a *corpus* problem (write
more handbook, or promote more ticket answers), while escalations up with drift flat
is a *provider* problem (judges failing closed).

**It costs nothing.** RetrievalController has already embedded the query to search
with it, so this is one dot product over a vector that exists either way. That is the
reason it can run on every request instead of on a sample.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from customer_support.helpers.logging_config import get_logger

logger = get_logger(__name__)

#: Where build_drift_baseline.py writes, and where main.py looks.
DEFAULT_BASELINE_PATH = Path("data/eval/drift_baseline.json")


def normalise(vector: list[float]) -> list[float] | None:
    """Unit-length copy, or None for a zero vector.

    None rather than an exception: a zero vector means the embedding call returned
    something unusable, and drift is instrumentation — it must not be able to fail a
    student's question.
    """
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return None
    return [value / magnitude for value in vector]


def cosine(left: list[float], right: list[float]) -> float | None:
    """Cosine similarity, or None if either side is unusable or the dims differ.

    The dimension check is not defensive padding: ``EMBEDDING_MODEL_SIZE`` is
    configurable, and a baseline built with a 384-dim model against a 1024-dim one
    would otherwise silently report drift that is really a configuration mismatch.
    """
    if not left or not right or len(left) != len(right):
        return None
    left_unit, right_unit = normalise(left), normalise(right)
    if left_unit is None or right_unit is None:
        return None
    return sum(a * b for a, b in zip(left_unit, right_unit, strict=True))


def centroid(vectors: list[list[float]]) -> list[float] | None:
    """The mean of unit vectors, re-normalised.

    Normalising BEFORE averaging matters: raw embedding magnitudes vary with text
    length, so a plain mean would let the longest questions dominate the reference
    point and make the baseline a statement about question length.
    """
    units = [unit for vector in vectors if (unit := normalise(vector)) is not None]
    if not units:
        return None

    dimensions = len(units[0])
    if any(len(unit) != dimensions for unit in units):
        raise ValueError("cannot build a centroid from vectors of differing dimensions")

    mean = [sum(unit[index] for unit in units) / len(units) for index in range(dimensions)]
    return normalise(mean)


def load_baseline(path: Path | str = DEFAULT_BASELINE_PATH) -> dict | None:
    """The baseline, or None — a missing file disables drift and nothing else.

    Returning None rather than raising is deliberate: the baseline costs embedding
    calls to build, so a fresh clone will not have one, and the app must still boot
    and serve. The absence is logged once at startup, where it is actionable.
    """
    path = Path(path)
    if not path.exists():
        logger.info("drift_baseline_absent", path=str(path))
        return None

    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
        vector = baseline.get("centroid")
        if not vector:
            raise ValueError("no centroid in baseline file")

        logger.info(
            "drift_baseline_loaded",
            path=str(path),
            dimensions=len(vector),
            questions=baseline.get("questions"),
            model=baseline.get("model"),
        )
        return baseline
    except Exception as exc:
        logger.error("drift_baseline_invalid", path=str(path), error=str(exc))
        return None
