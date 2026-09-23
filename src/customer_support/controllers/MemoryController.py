"""Section 6 — long-term student memory.

One patched profile per student. Identity, not history.

**On the deviation from the design notes.** The notes specify Trustcall, whose value
is JSON-patch updates that avoid destructive regeneration. That guarantee is
implemented here instead, in code:

* the extractor is asked for a PARTIAL update, never the whole profile, so it cannot
  drop a field by forgetting to re-emit it;
* ``None`` means "not mentioned", and ``learned_fields()`` strips nulls before the
  merge, so a null can never overwrite a known value;
* the merge itself is deterministic Python, so deletion is structurally impossible
  rather than merely unlikely.

That is a stronger guarantee than patch ops the model could still get wrong, and it
keeps every model call behind the existing ``LLMInterface`` instead of standing up a
second LLM abstraction (LangChain chat model + tool calling) beside the provider
factory. Swapping Trustcall in later touches only ``_extract``.

**The three cost levers, in the order Section 6 ranks them:**

1. *Frequency gating* — the biggest saving, and free. Most traffic is impersonal
   handbook questions that carry no identity fact, and a profile is stable, so the
   naive design pays an LLM call per turn to conclude "nothing new". The gate here
   is a pattern match with no model call at all.
2. *A small model* — extraction is easy, answering handbook questions is not. The
   judge/extraction client is configured separately from the generation client.
3. *Off the critical path* — the answer depends on READING the profile, never on
   writing it. The router schedules ``remember`` after the response is sent.
"""

import asyncio
import json
import re

from pydantic import ValidationError

from customer_support.helpers.logging_config import get_logger
from customer_support.models.llm_schemas.student_profile import (
    StudentProfile,
    StudentProfileUpdate,
)
from customer_support.models.ProfileModel import ProfileModel
from customer_support.stores.llm.templates.memory import (
    EXTRACTION_TEMPLATE,
    MEMORY_SYSTEM_PROMPT,
)

from .BaseController import BaseController

# First-person self-description markers, English and Arabic. Deliberately generous:
# a false positive costs one call on a small model, a false negative loses a fact
# the student volunteered. What it reliably filters is the common case — impersonal
# handbook questions ("when are exams?", "كم عدد الساعات المعتمدة؟") carry none of
# these, and that is the bulk of real traffic.
_MARKER_WORDS = (
    # English
    "my",
    "i am",
    "i'm",
    "im",
    "call me",
    "i study",
    "i major",
    "i moved",
    # Arabic
    "اسمي",
    "انا",
    "أنا",
    "معدلي",
    "تخصصي",
    "قسمي",
    "ادرس",
    "أدرس",
    "لغتي",
    "تحولت",
)

# Word boundaries on EVERY marker, not just the English ones. Without them "انا"
# ("I") matches inside "الامتحانات" ("the exams"), so every question about exam
# dates would pay for an extraction — the gate would leak exactly the traffic it
# exists to filter. Arabic letters are word characters, so \b behaves correctly.
#
# Known limitation: a prefixed conjunction ("وأنا") has no boundary before the
# marker and is missed. That is the safe direction — a missed fact is re-offered by
# the next turn that mentions it, while a leaking gate costs a call on every turn.
_IDENTITY_MARKERS = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in _MARKER_WORDS) + r")\b",
    re.IGNORECASE,
)


class MemoryController(BaseController):
    def __init__(self, extraction_client, profile_model: ProfileModel, settings=None):
        super().__init__(settings)
        self.extraction_client = extraction_client
        self.profile_model = profile_model
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------ read

    async def load_profile(self, student_id: str) -> StudentProfile:
        """Read the profile. A single indexed row, on the critical path.

        Never returns None: an unknown student is an empty profile, so every caller
        can read ``profile.department`` without a None check and a first-time student
        is not a special case.
        """
        if not student_id:
            return StudentProfile(student_id="")

        try:
            profile = await self.profile_model.get_profile(student_id)
        except Exception as exc:
            # Memory is an enhancement, not a prerequisite. A profile lookup that
            # fails must degrade to "we know nothing" rather than fail the answer.
            self.logger.error(
                "profile_load_failed", student_id=student_id, error=str(exc), exc_info=True
            )
            return StudentProfile(student_id=student_id)

        return profile or StudentProfile(student_id=student_id)

    # ----------------------------------------------------------------- write

    def is_worth_extracting(self, message: str) -> bool:
        """The frequency gate. No model call.

        A profile does not change on most turns, so the question is not "what did
        this message say about the student" but "could it plausibly have said
        anything at all".
        """
        return bool(message and _IDENTITY_MARKERS.search(message))

    async def remember(self, student_id: str, message: str) -> StudentProfile | None:
        """Extract and patch, if the message plausibly carries a fact.

        Runs off the critical path, so it swallows its own failures: the student
        already has their answer, and a failed extraction must never surface as an
        error on a request that succeeded. Returns the new profile, or None when
        nothing was written.
        """
        if not student_id:
            return None

        if not self.is_worth_extracting(message):
            # The common path, and the whole point of the gate.
            self.logger.info("memory_extraction_skipped", student_id=student_id, reason="no_marker")
            return None

        try:
            current = await self.load_profile(student_id)
            update = await self._extract(message=message, profile=current)

            if update is None:
                return None

            learned = update.learned_fields()
            if not learned:
                # Recorded even though nothing was learned: it is what makes the
                # gate's hit rate measurable instead of a claim.
                await self.profile_model.touch_extraction(student_id)
                self.logger.info("memory_extraction_empty", student_id=student_id)
                return None

            # Only report fields whose value actually changes, so the log shows
            # real learning and not the extractor echoing what it was shown.
            changed = {
                field: value
                for field, value in learned.items()
                if getattr(current, field, None) != value
            }

            profile = await self.profile_model.apply_patch(student_id=student_id, fields=learned)

            self.logger.info(
                "memory_updated",
                student_id=student_id,
                learned=sorted(learned),
                changed=sorted(changed),
            )
            return profile

        except Exception as exc:
            self.logger.error(
                "memory_extraction_failed", student_id=student_id, error=str(exc), exc_info=True
            )
            return None

    # ------------------------------------------------------------- internals

    async def _extract(self, message: str, profile: StudentProfile) -> StudentProfileUpdate | None:
        """One call on the small model, returning a partial update.

        No retry: unlike a judge, a failed extraction has no consequence worth
        paying for — the fact is still in the transcript and the next turn that
        mentions it will try again. Failing closed here means "learn nothing", which
        is the safe direction.
        """
        prompt = EXTRACTION_TEMPLATE.substitute(
            message=message,
            known_profile=profile.describe() or "nothing yet",
        )

        raw = await asyncio.to_thread(
            self.extraction_client.generate_json,
            prompt,
            MEMORY_SYSTEM_PROMPT,
            self.app_settings.MEMORY_MAX_OUTPUT_TOKENS,
            self.app_settings.MEMORY_TEMPERATURE,
        )
        if not raw:
            self.logger.warning("memory_extraction_empty_response")
            return None

        try:
            return StudentProfileUpdate.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as exc:
            # An unparseable extraction learns nothing. That is a non-event, so it
            # is a warning rather than an error.
            self.logger.warning("memory_extraction_rejected", error=str(exc))
            return None
