"""Long-term memory (Section 6): the merge must never lose a known fact, and the
frequency gate must actually filter.

No database and no model calls — a fake profile store and a fake extraction client.
The two properties under test are the ones that make memory safe to run on every
turn: a patch cannot delete, and most turns cost nothing.
"""

import json

import pytest

from customer_support.controllers.MemoryController import MemoryController
from customer_support.models.llm_schemas.student_profile import (
    StudentProfile,
    StudentProfileUpdate,
)


class FakeSettings:
    MEMORY_MAX_OUTPUT_TOKENS = 600
    MEMORY_TEMPERATURE = 0.0
    ASSETS_DIR = "assets"


class FakeExtractor:
    """Returns a canned JSON extraction and records how often it was asked."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def generate_json(self, prompt, system_prompt=None, max_output_tokens=None, temperature=None):
        self.calls += 1
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


class FakeProfileModel:
    def __init__(self, existing: dict | None = None):
        self.rows: dict[str, dict] = dict(existing or {})
        self.patches: list[tuple[str, dict]] = []

    async def get_profile(self, student_id):
        row = self.rows.get(student_id)
        return StudentProfile(student_id=student_id, **row) if row else None

    async def apply_patch(self, student_id, fields):
        self.patches.append((student_id, dict(fields)))
        row = self.rows.setdefault(student_id, {})
        row.update(fields)
        return StudentProfile(student_id=student_id, **row)

    async def touch_extraction(self, student_id):
        await self.apply_patch(student_id, {})


def controller(extractor, profiles=None):
    return MemoryController(
        extraction_client=extractor,
        profile_model=profiles or FakeProfileModel(),
        settings=FakeSettings(),
    )


# ------------------------------------------------- the non-destructive guarantee


def test_a_patch_cannot_delete_a_known_field():
    """The property Trustcall exists to provide, implemented in the merge.

    An extraction that mentions only the department must not blank the name — that
    is the regenerate-the-whole-document failure mode.
    """
    known = StudentProfile(student_id="s1", name="Omar", department="CS", gpa=3.4)

    merged = known.merge(StudentProfileUpdate(department="IS"))

    assert merged.department == "IS"  # the stated fact is applied
    assert merged.name == "Omar"  # everything unmentioned survives
    assert merged.gpa == 3.4


def test_nulls_are_not_learned_fields():
    """None means 'this message said nothing about that field', never 'clear it'."""
    update = StudentProfileUpdate(name="Omar", department=None, gpa=None)

    assert update.learned_fields() == {"name": "Omar"}


def test_an_all_empty_extraction_learns_nothing():
    assert StudentProfileUpdate().learned_fields() == {}


def test_merging_an_empty_update_changes_nothing():
    known = StudentProfile(student_id="s1", name="Omar", department="CS")
    assert known.merge(StudentProfileUpdate()) == known


# ------------------------------------------------------------ input validation


def test_an_unknown_department_is_dropped_not_fatal():
    """Department drives the retrieval filter, so a value outside CS/IS would filter
    every chunk away. Dropping it keeps the good fields in the same response."""
    update = StudentProfileUpdate(name="Omar", department="Mathematics")

    assert update.department is None
    assert update.learned_fields() == {"name": "Omar"}


def test_department_is_normalised():
    assert StudentProfileUpdate(department="cs").department == "CS"
    assert StudentProfileUpdate(department=" is ").department == "IS"


@pytest.mark.parametrize("bad_gpa", [-0.5, 4.5, 100])
def test_an_out_of_range_gpa_is_rejected(bad_gpa):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        StudentProfileUpdate(gpa=bad_gpa)


def test_blank_strings_are_not_facts():
    assert StudentProfileUpdate(name="   ").name is None


# ------------------------------------------------------------ frequency gating


@pytest.mark.parametrize(
    "message",
    [
        "my name is Omar",
        "I'm in the CS department",
        "I am a CS student",
        "my GPA is 3.4",
        "اسمي عمر",
        "انا في قسم علوم الحاسب",
        "معدلي 3.4",
    ],
)
def test_self_description_is_worth_extracting(message):
    assert controller(FakeExtractor({})).is_worth_extracting(message)


@pytest.mark.parametrize(
    "message",
    [
        "how many credit hours are required to graduate?",
        "when do exams start?",
        "كم عدد الساعات المعتمدة المطلوبة للتخرج؟",
        "ما هي مواعيد الامتحانات؟",
        "",
    ],
)
def test_impersonal_questions_are_skipped(message):
    """The gate's whole value: this is the bulk of real traffic, and it must cost
    no model call at all."""
    assert not controller(FakeExtractor({})).is_worth_extracting(message)


@pytest.mark.asyncio
async def test_a_skipped_turn_never_calls_the_model():
    extractor = FakeExtractor({"name": "Omar"})
    memory = controller(extractor)

    result = await memory.remember("s1", "when are the exams?")

    assert result is None
    assert extractor.calls == 0


# --------------------------------------------------------------------- writes


@pytest.mark.asyncio
async def test_a_stated_fact_is_extracted_and_patched():
    extractor = FakeExtractor({"name": "Omar", "department": "CS"})
    profiles = FakeProfileModel()
    memory = controller(extractor, profiles)

    profile = await memory.remember("s1", "my name is Omar and I'm in CS")

    assert extractor.calls == 1
    assert (profile.name, profile.department) == ("Omar", "CS")
    assert profiles.patches == [("s1", {"name": "Omar", "department": "CS"})]


@pytest.mark.asyncio
async def test_extraction_does_not_clear_existing_fields():
    """End to end through the controller: the student mentions only their GPA."""
    profiles = FakeProfileModel({"s1": {"name": "Omar", "department": "CS"}})
    memory = controller(FakeExtractor({"gpa": 3.9}), profiles)

    profile = await memory.remember("s1", "my GPA is 3.9")

    assert profile.gpa == 3.9
    assert profile.name == "Omar"
    assert profile.department == "CS"


@pytest.mark.asyncio
async def test_an_extraction_about_someone_else_writes_nothing():
    """The model is prompted to return {} here; the controller must not patch."""
    profiles = FakeProfileModel({"s1": {"department": "CS"}})
    memory = controller(FakeExtractor({}), profiles)

    result = await memory.remember("s1", "my friend is in IS")

    assert result is None
    # Only the bookkeeping touch, carrying no profile fields.
    assert profiles.patches == [("s1", {})]
    assert profiles.rows["s1"]["department"] == "CS"


@pytest.mark.asyncio
async def test_unparseable_extraction_is_survivable():
    """Memory runs after the response is sent. A failure must learn nothing and
    raise nothing."""
    profiles = FakeProfileModel()
    memory = controller(FakeExtractor("not json at all"), profiles)

    assert await memory.remember("s1", "my name is Omar") is None
    assert profiles.patches == []


@pytest.mark.asyncio
async def test_a_load_failure_degrades_to_an_empty_profile():
    """Memory is an enhancement, not a prerequisite: a broken profile store must not
    fail the student's answer."""

    class BrokenProfiles(FakeProfileModel):
        async def get_profile(self, student_id):
            raise RuntimeError("database is down")

    memory = controller(FakeExtractor({}), BrokenProfiles())
    profile = await memory.load_profile("s1")

    assert profile.student_id == "s1"
    assert profile.is_empty


@pytest.mark.asyncio
async def test_an_unknown_student_loads_an_empty_profile_not_none():
    memory = controller(FakeExtractor({}))
    profile = await memory.load_profile("never-seen")

    assert profile.is_empty
    assert profile.department is None


# ---------------------------------------------------------------- description


def test_describe_omits_what_is_not_known():
    described = StudentProfile(student_id="s1", name="Omar", department="CS").describe()

    assert "Omar" in described and "CS" in described
    assert "GPA" not in described
    assert "None" not in described


def test_an_empty_profile_describes_as_nothing():
    assert StudentProfile(student_id="s1").describe() == ""
    assert StudentProfile(student_id="s1").is_empty
