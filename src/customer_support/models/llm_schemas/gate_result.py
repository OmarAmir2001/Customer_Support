"""Structured output of a judge. Never trust an LLM's text: parse it into this."""

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """Exactly what the judge model is asked to return: a score and its reasoning.

    The judge does NOT decide pass/fail. Thresholds are configuration and live in
    Settings, so they can be tuned against the eval set without touching a prompt.
    """

    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500)


class GateResult(BaseModel):
    """A judge verdict plus the threshold decision applied to it."""

    gate: str
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str
