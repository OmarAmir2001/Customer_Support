"""PgVectorProvider — the first tests this file has had.

All four findings here are invisible at the current data size and none of them raises
in normal use, which is why they survived: F8 needs a collection that does not exist,
F9 and F10 need enough rows to feel a sequential scan, and F11 needs a setting nobody
has changed from its default.

No database. A fake session records the SQL, which is enough to assert the two things
that were actually wrong — the ORDER statements execute in, and which statements
execute at all.
"""

import pytest
from sqlalchemy.exc import ProgrammingError

from customer_support.stores.vectordb.providers.PgVectorProvider import PgVectorProvider
from customer_support.stores.vectordb.VectorDBEnum import (
    DistanceMethodEnums,
    PgVectorDistanceMethodEnums,
)


class FakeResult:
    def __init__(self, row=None, scalar=0):
        self._row = row
        self._scalar = scalar

    def fetchone(self):
        return self._row

    def scalar_one(self):
        return self._scalar

    def scalar_one_or_none(self):
        return self._scalar


class FakeSession:
    """Records every statement, and can refuse one the way Postgres would.

    `missing_table` makes any `FROM <name>` raise UndefinedTable, which is what the
    real server does and the behaviour F8 hinges on.
    """

    def __init__(self, table_row=None, missing_table=None):
        self.executed = []
        self.table_row = table_row
        self.missing_table = missing_table

    async def execute(self, statement, params=None):
        text = str(statement)
        self.executed.append(text)

        if self.missing_table and f"FROM {self.missing_table}" in text:
            raise ProgrammingError(
                "SELECT ...", {}, Exception(f'relation "{self.missing_table}" does not exist')
            )

        if "FROM pg_tables" in text:
            # Two different pg_tables queries, distinguished the way the real server
            # would: `is_collection_exists` does SELECT 1 and reads scalar_one_or_none,
            # `get_collection_info` does SELECT schemaname,... and reads fetchone.
            # Modelling only one of them made the fake claim every table was absent.
            exists = 1 if self.table_row else None
            return FakeResult(row=self.table_row, scalar=exists)
        if "COUNT(*)" in text:
            return FakeResult(scalar=7)
        return FakeResult()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def begin(self):
        return self


def client_factory(session):
    def _factory():
        return session

    return _factory


def provider(session, distance_method="cosine"):
    return PgVectorProvider(
        db_client=client_factory(session),
        default_vector_size=384,
        distance_method=distance_method,
    )


# ------------------------------------------------------------------------ F8

TABLE_ROW = ("public", "collection_3", "postgres", None, True)


@pytest.mark.asyncio
async def test_get_collection_info_returns_none_for_a_missing_collection():
    """The guard that could never be reached.

    Both statements used to execute before the existence check, so COUNT(*) raised
    UndefinedTable and `if not table_data: return None` was dead code — the one case
    it was written for was the one case that raised past it.
    """
    session = FakeSession(table_row=None, missing_table="collection_999")
    result = await provider(session).get_collection_info("collection_999")

    assert result is None


@pytest.mark.asyncio
async def test_the_count_is_not_executed_when_the_table_is_absent():
    """Asserted on the statements, because ordering is the actual fix.

    Returning None is achievable while still running the count first and catching
    the error — which would leave the aborted-transaction problem in place.
    """
    session = FakeSession(table_row=None, missing_table="collection_999")
    await provider(session).get_collection_info("collection_999")

    assert not any("COUNT(*)" in sql for sql in session.executed), (
        "the count ran against a table known not to exist"
    )


@pytest.mark.asyncio
async def test_get_collection_info_still_reports_a_real_collection():
    session = FakeSession(table_row=TABLE_ROW)
    info = await provider(session).get_collection_info("collection_3")

    assert info["record_count"] == 7
    assert info["table_info"]["tablename"] == "collection_3"


# ----------------------------------------------------------------- F9 + F10


@pytest.mark.asyncio
async def test_supporting_indexes_are_created_for_an_EXISTING_collection():
    """F10: the bug that may be live in a running database.

    `_create_supporting_indexes` sat inside `if not is_collection_existed`, so an
    existing collection took the early return and never got them. Since
    `create_collection` runs on every push, the idempotent CREATE INDEX IF NOT
    EXISTS statements were unreachable for exactly the collections holding data.
    """
    session = FakeSession(table_row=TABLE_ROW)
    created = await provider(session).create_collection("collection_3", embedding_size=384)

    assert created is False, "it must still report that it did not create the table"
    index_sql = [sql for sql in session.executed if "CREATE INDEX" in sql]
    assert index_sql, "an existing collection got no indexes"
    assert not any("CREATE TABLE" in sql for sql in session.executed)


@pytest.mark.asyncio
async def test_the_foreign_key_column_is_indexed():
    """F9: Postgres indexes the REFERENCED side of an FK, never the referencing one.

    Without this, every cascade from a `chunks` delete is a sequential scan of the
    vector table, once per deleted chunk — O(chunks × rows), so `/process` with
    do_reset degrades quadratically.
    """
    session = FakeSession(table_row=TABLE_ROW)
    await provider(session).create_collection("collection_3", embedding_size=384)

    assert any(
        "collection_3_chunk_id_idx" in sql and "(chunk_id)" in sql for sql in session.executed
    ), f"no chunk_id index created; got {[s for s in session.executed if 'CREATE INDEX' in s]}"


@pytest.mark.asyncio
async def test_a_new_collection_gets_the_table_and_every_index():
    session = FakeSession(table_row=None)
    created = await provider(session).create_collection("collection_9", embedding_size=384)

    assert created is True
    assert any("CREATE TABLE" in sql for sql in session.executed)
    for expected in ("_ticket_id_idx", "_source_section_idx", "_chunk_id_idx"):
        assert any(expected in sql for sql in session.executed), f"missing {expected}"


@pytest.mark.asyncio
async def test_index_creation_is_idempotent():
    """It runs on every push, so it has to be safe to run on every push."""
    session = FakeSession(table_row=TABLE_ROW)
    await provider(session).create_collection("collection_3", embedding_size=384)

    for sql in (s for s in session.executed if "CREATE INDEX" in s):
        assert "IF NOT EXISTS" in sql, sql


# ----------------------------------------------------------------------- F11


def test_dot_means_inner_product_on_pgvector_as_it_does_on_qdrant():
    """One setting must not select two metrics.

    `vector_l2_ops` with `<->` is EUCLIDEAN distance. Qdrant maps the same `dot`
    setting to models.Distance.DOT — an inner product — so the two backends computed
    different things from one config value, silently, because both run fine.
    """
    instance = provider(FakeSession(), distance_method=DistanceMethodEnums.DOT.value)

    assert instance.index_opclass == "vector_ip_ops"
    assert instance.distance_operator == "<#>"
    assert "<#>" in instance.score_expression
    assert "<->" not in instance.score_expression, "still using the L2 operator"


def test_the_opclass_and_the_operator_always_agree():
    """A pgvector index is only used by a query ordering on its matching operator.

    Mismatch them and the HNSW index is silently ignored: correct results, full
    scan, no error.
    """
    pairs = {"vector_cosine_ops": "<=>", "vector_ip_ops": "<#>"}
    for method in (DistanceMethodEnums.COSINE.value, DistanceMethodEnums.DOT.value):
        instance = provider(FakeSession(), distance_method=method)
        assert pairs[instance.index_opclass] == instance.distance_operator


def test_cosine_remains_the_default_and_is_unchanged():
    """The shipped configuration must not move while fixing the other branch."""
    instance = provider(FakeSession(), distance_method=None)

    assert instance.index_opclass == PgVectorDistanceMethodEnums.COSINE.value
    assert instance.distance_operator == "<=>"
    assert instance.score_expression == "1 - ({column} <=> :vector)"
