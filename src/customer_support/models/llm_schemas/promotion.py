"""The result of assessing whether a resolved answer belongs in the knowledge base.

Section 5's promotion flow produces three separate facts, and collapsing them into
one boolean loses the thing that matters:

* ``suggested_promote`` — the checkbox DEFAULT. Advice, always overridable.
* ``held`` — a veto. Only a contradiction sets it, and the advisor cannot tick past
  it, because the question is no longer "should this be added" but "which of these
  two is wrong".
* the reasons — what an advisor reads to decide, and what a handbook reviewer
  needs in order to act.
"""

from pydantic import BaseModel, Field


class PromotionAssessment(BaseModel):
    """What the machine advises about promoting one resolved answer."""

    generalizable: bool
    generalizability_score: float = Field(ge=0.0, le=1.0)
    generalizability_reason: str

    contradicts_handbook: bool
    contradiction_score: float = Field(ge=0.0, le=1.0)
    contradiction_reason: str

    #: Excerpts the contradiction check was shown, so an advisor can see what the
    #: verdict was actually based on rather than trusting a bare score.
    compared_against: list[str] = Field(default_factory=list)

    @property
    def held(self) -> bool:
        """A contradiction blocks promotion regardless of the advisor's checkbox.

        The single exception to "the machine only pre-fills". A conflicting answer
        must not become a competing chunk — the handbook gets reviewed instead.
        """
        return self.contradicts_handbook

    @property
    def suggested_promote(self) -> bool:
        """What the checkbox should default to when the advisor opens the ticket."""
        return self.generalizable and not self.contradicts_handbook

    def hold_reason(self) -> str | None:
        return self.contradiction_reason if self.held else None


class ContradictionVerdict(BaseModel):
    """Just the veto, for the path that only needs the veto.

    Deliberately NOT a PromotionAssessment with the generalizability fields left
    empty: those fields are non-optional, so filling them would mean inventing a
    verdict no judge gave. A caller that only asks "may this be promoted" gets a
    type that can only answer that.
    """

    #: True when a contradiction blocks promotion regardless of the advisor's box.
    held: bool
    score: float = Field(ge=0.0, le=1.0)
    #: The judge's wording, and only when it actually holds — mirrors
    #: PromotionAssessment.hold_reason(), which returns None when nothing is held.
    reason: str | None = None
