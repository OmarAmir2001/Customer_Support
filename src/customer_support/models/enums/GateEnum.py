from enum import StrEnum


class GateEnum(StrEnum):
    """The three orthogonal checks of Section 2.

    Orthogonal on purpose: each judges a different thing against different evidence,
    so they do not share blind spots the way several rephrasings of "how sure are you"
    would. Escalate if ANY of them fails.
    """

    CONTEXT_RELEVANCE = "context_relevance"  # pre-generation: did retrieval find anything usable?
    FAITHFULNESS = "faithfulness"  # post-generation: is every claim supported by the chunks?
    ANSWER_RELEVANCE = "answer_relevance"  # post-generation: does it answer the question asked?

    # Not a judge, and not one of the three orthogonal checks: a failure ORIGIN, for
    # when generation itself produced nothing so there was no answer to judge. It is
    # separate so it cannot contaminate the faithfulness score distribution that the
    # `gate_evaluated` log lines are mined for when tuning thresholds.
    GENERATION = "generation"


class GateFailureReason(StrEnum):
    """Fallback reason strings. A judge usually returns something more specific;
    these are used when it cannot."""

    NO_RELEVANT_CONTEXT = "No relevant handbook content was found for this question."
    NOT_GROUNDED = "The drafted answer is not fully supported by the handbook."
    OFF_TOPIC = "The drafted answer does not address the question that was asked."
    JUDGE_UNAVAILABLE = "The answer could not be verified automatically."
    GENERATION_FAILED = "The assistant could not draft an answer."