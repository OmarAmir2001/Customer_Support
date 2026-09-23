"""The profile's shape, and the shape of an extraction.

Two models, and the difference between them is the whole safety story:

* ``StudentProfile`` — what is known about a student. Every field optional, because
  a profile is accumulated over many turns and is never complete at first contact.
* ``StudentProfileUpdate`` — what one extraction claims to have learned. Also all
  optional, but here ``None`` means "this message said nothing about that field",
  NOT "clear it".

That second reading is what makes the merge non-destructive. The extractor is never
asked to re-emit the whole profile, so it cannot drop a field by forgetting it — the
classic failure of regenerate-the-document memory. It reports only what it saw, and
``MemoryController`` merges field by field, never writing a null over a known value.
Deletion is structurally impossible rather than merely discouraged.
"""

from pydantic import BaseModel, Field, field_validator

#: Section 6's persist-list. Anything not here is transient and belongs in the
#: checkpointer: individual questions, momentary states, session-specific detail.
PROFILE_FIELDS = ("name", "department", "gpa", "preferred_language")

DEPARTMENTS = ("CS", "IS")


class StudentProfileUpdate(BaseModel):
    """What one message revealed. Validated hard, because this is LLM output about
    a student and a wrong fact patched over a right one is the failure mode."""

    name: str | None = Field(default=None, max_length=120)
    department: str | None = None
    gpa: float | None = Field(default=None, ge=0.0, le=4.0)
    preferred_language: str | None = Field(default=None, max_length=16)

    @field_validator("department")
    @classmethod
    def _known_department(cls, value: str | None) -> str | None:
        """Drop an unrecognised department instead of failing the whole extraction.

        One bad field should not discard the good ones in the same response, and
        department drives the retrieval filter — a value outside CS/IS would filter
        every chunk away and silently break answering.
        """
        if value is None:
            return None
        normalised = value.strip().upper()
        return normalised if normalised in DEPARTMENTS else None

    @field_validator("name", "preferred_language")
    @classmethod
    def _no_blanks(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def learned_fields(self) -> dict:
        """Only the fields this extraction actually claims. Null means 'not
        mentioned', so it is excluded here and can never overwrite a known value."""
        return {
            field: value
            for field, value in self.model_dump().items()
            if value is not None and field in PROFILE_FIELDS
        }


class StudentProfile(BaseModel):
    """What is known about a student. Injected into the answer prompt each turn."""

    student_id: str
    name: str | None = None
    department: str | None = None
    gpa: float | None = None
    preferred_language: str | None = None

    model_config = {"from_attributes": True}

    @property
    def is_empty(self) -> bool:
        return not any(getattr(self, field) is not None for field in PROFILE_FIELDS)

    def merge(self, update: StudentProfileUpdate) -> "StudentProfile":
        """Patch, do not regenerate.

        Returns a new profile with the learned fields applied over this one. A field
        the extraction did not mention keeps its existing value — that is the single
        rule that makes memory safe to run repeatedly.
        """
        return self.model_copy(update=update.learned_fields())

    def describe(self) -> str:
        """The profile as prompt text, omitting what is not known.

        Rendered here rather than in a locale file: these are field VALUES a student
        gave us, and translating "Omar" or "CS" would be wrong. The surrounding
        sentence is the locale's job.
        """
        known = {
            "name": self.name,
            "department": self.department,
            "GPA": self.gpa,
            "preferred language": self.preferred_language,
        }
        return ", ".join(f"{label}: {value}" for label, value in known.items() if value is not None)
