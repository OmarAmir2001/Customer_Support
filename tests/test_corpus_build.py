"""The corpus build, and the validation that stands in front of it.

The reason this file exists: a parsing bug in the corpus is silent at build time and
surfaces three steps later as a confidently wrong answer, by which point the model
gets the blame. So the build refuses to write a corpus that fails these checks, and
these tests confirm each check can actually fail — a validator that always passes is
worse than none, because it is trusted.

They also pin the property that makes the corpus usable as a *version*: the same
inputs and parameters must produce the same content hash.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_corpus import (  # noqa: E402
    HARD_MAX_CHARS,
    CorpusError,
    build,
    citation_for,
    validate,
)

CS = (
    "# لائحة\n\n## القسم الأول\n\n"
    # Long enough to be split by a small chunk_size and left whole by a large one.
    # A fixture of only short sections produces identical chunks at every size,
    # which makes the parameter look like it does nothing.
    "يشترط لقبول الطالب أن يكون حاصلا على الثانوية العامة أو ما يعادلها. " * 8 + "\n"
)
IS = "# لائحة نظم المعلومات\n\n## الالتحاق\n\nمدة الدراسة أربع سنوات دراسية.\n"


class FakeSettings:
    """All the controller underneath actually reads. Injected rather than inherited
    from a local .env, so this suite behaves the same on a laptop and in CI."""

    def __init__(self, assets_dir):
        self.ASSETS_DIR = str(assets_dir)


@pytest.fixture
def handbooks(tmp_path):
    """A minimal pair of handbooks named so identify_handbook() recognises them."""
    books = tmp_path / "handbooks"
    books.mkdir()
    (books / "CS_2023.md").write_text(CS, encoding="utf-8")
    (books / "IS_2023.md").write_text(IS, encoding="utf-8")
    return books


@pytest.fixture
def settings(tmp_path):
    return FakeSettings(tmp_path / "assets")


def record(**overrides):
    base = {
        "chunk_id": "CS_2023::A > B::0",
        "source": "CS_2023",
        "department": "CS",
        "section": "A > B",
        "part": 0,
        "text": "some handbook text",
        "char_count": 18,
        "citation": "CS_2023 — B",
        "parts_in_section": 1,
    }
    base.update(overrides)
    return base


def both_handbooks(*records):
    """validate() requires every known handbook to contribute, so tests that are not
    about that rule need a chunk from each."""
    return [*records, record(chunk_id="IS_2023::X::0", source="IS_2023", department="IS")]


# ----------------------------------------------------------------- the build


def test_build_produces_one_record_per_chunk_with_citable_metadata(handbooks, settings):
    records, manifest = build(handbooks, chunk_size=1000, overlap=50, settings=settings)

    assert records
    assert manifest["records"] == len(records)
    assert set(manifest["by_source"]) == {"CS_2023", "IS_2023"}
    for rec in records:
        assert rec["text"].strip()
        assert rec["department"] in {"CS", "IS"}
        assert rec["citation"].startswith(rec["source"])


def test_the_same_inputs_and_params_produce_the_same_version(handbooks, settings):
    """What makes the hash a data version rather than a build timestamp. Without
    this, an MLflow run could not record WHICH corpus produced a metric."""
    _, first = build(handbooks, chunk_size=1000, overlap=50, settings=settings)
    _, second = build(handbooks, chunk_size=1000, overlap=50, settings=settings)

    assert first["content_sha256"] == second["content_sha256"]


def test_a_chunk_size_that_splits_sections_changes_the_version(handbooks, settings):
    """The hash covers CONTENT, not parameters — deliberately, so it answers "did
    the data actually change?" rather than "was a flag different?". A chunk_size
    small enough to split a section changes the content, so it must change the
    hash; one that changes nothing correctly leaves it alone (asserted below)."""
    _, big = build(handbooks, chunk_size=1000, overlap=50, settings=settings)
    _, small = build(handbooks, chunk_size=120, overlap=10, settings=settings)

    assert big["content_sha256"] != small["content_sha256"]
    assert small["records"] >= big["records"]


def test_a_chunk_size_that_splits_nothing_leaves_the_version_alone(handbooks, settings):
    """The other half. Both sizes here are larger than every section, so no split
    happens and the corpus is byte-identical — the hash should say so."""
    _, a = build(handbooks, chunk_size=4000, overlap=50, settings=settings)
    _, b = build(handbooks, chunk_size=8000, overlap=50, settings=settings)

    assert a["content_sha256"] == b["content_sha256"]
    assert a["records"] == b["records"]


def test_an_empty_input_directory_is_an_error_not_an_empty_corpus(tmp_path, settings):
    """Silently writing an empty corpus would leave the vector store intact and the
    agent answering from stale data, with nothing to indicate a failed build."""
    with pytest.raises(CorpusError):
        build(tmp_path, chunk_size=1000, overlap=50, settings=settings)


# ------------------------------------------------------------- the validator
#
# One test per rule, each breaking exactly one thing.


def test_a_valid_corpus_has_no_problems():
    assert validate(both_handbooks(record())) == []


def test_empty_text_is_rejected():
    problems = validate(both_handbooks(record(text="   ")))
    assert any("empty text" in p for p in problems)


def test_an_unknown_source_is_rejected():
    """Retrieval precedence keys on `source`: handbook chunks must outrank promoted
    answers. A source that is not a known handbook silently loses that ranking."""
    problems = validate(both_handbooks(record(source="random_file.md")))
    assert any("not a known handbook" in p for p in problems)


def test_a_missing_department_is_rejected():
    """The department filter is what stops a CS student being answered out of the IS
    handbook."""
    problems = validate(both_handbooks(record(department=None)))
    assert any("no department" in p for p in problems)


def test_an_unlabelled_section_is_rejected():
    problems = validate(both_handbooks(record(section="(untitled)")))
    assert any("nothing to cite" in p for p in problems)


def test_an_oversized_chunk_is_rejected():
    """The symptom of the overflow splitter not running. Such a chunk retrieves
    poorly and eats the answer prompt's budget."""
    problems = validate(both_handbooks(record(char_count=HARD_MAX_CHARS + 1)))
    assert any("overflow splitter did not run" in p for p in problems)


def test_a_duplicate_chunk_id_is_rejected():
    """chunk_id is the key the idempotent vector re-sync deletes on. Two rows sharing
    one means a re-index drops one of them."""
    problems = validate(both_handbooks(record(), record()))
    assert any("duplicate chunk_id" in p for p in problems)


def test_a_handbook_contributing_nothing_is_rejected():
    """A file that parsed to zero chunks is invisible in a total record count."""
    problems = validate([record()])  # CS only
    assert any("IS_2023: contributed no chunks" in p for p in problems)


def test_an_empty_corpus_is_rejected():
    assert validate([]) == ["corpus is empty"]


# --------------------------------------------------------------- citations


def test_a_citation_names_the_section_not_the_chunk_id():
    """A lawyer — or a student — verifies an answer by looking the reference up. A
    chunk id cannot be looked up in anything."""
    assert citation_for("CS_2023", "الباب الأول > مادة (٢٤)") == "CS_2023 — مادة (٢٤)"


def test_a_citation_falls_back_to_the_source_when_there_is_no_section():
    assert citation_for("CS_2023", "(untitled)") == "CS_2023"
    assert citation_for("CS_2023", "") == "CS_2023"


# ------------------------------------------------- the corpus that is committed


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "corpus" / "corpus.json").exists(),
    reason="corpus not built yet — run scripts/build_corpus.py",
)
def test_the_committed_corpus_is_valid():
    """Guards the artifact itself, not just the code that makes it. The corpus is
    DVC-tracked and embedded downstream, so a bad one that got committed would be
    invisible until answers degraded."""
    records = json.loads(
        (REPO_ROOT / "data" / "corpus" / "corpus.json").read_text(encoding="utf-8")
    )
    assert validate(records) == []
