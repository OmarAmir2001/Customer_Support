import json

from sqlalchemy.sql import text as sql_text

from customer_support.helpers.logging_config import get_logger
from customer_support.models.db_schemas import RetrievedDocument

from ..VectorDBEnum import (
    DistanceMethodEnums,
    PgVectorDistanceMethodEnums,
    PgVectorIndexTypeEnums,
    PgVectorTableSchemeEnums,
)
from ..VectorDBInterface import VectorDBInterface


class PgVectorProvider(VectorDBInterface):
    def __init__(
        self,
        db_client,
        default_vector_size: int = 786,
        distance_method: str = None,
        index_threshold: int = 10000,
    ):
        self.db_client = db_client
        self.default_vector_size = default_vector_size
        self.distance_method = distance_method
        self.index_threshold = index_threshold

        self.pg_vector_table_prefix = PgVectorTableSchemeEnums._PREFIX.value

        # `distance_method` arrives as a DistanceMethodEnums value ("cosine"/"dot"), but
        # pgvector needs an operator-class name for CREATE INDEX and an operator for the
        # ORDER BY. Mapping them here keeps the two in step: an index built with
        # vector_cosine_ops is only used by a query ordering on <=>.
        if self.distance_method == DistanceMethodEnums.DOT.value:
            self.index_opclass = PgVectorDistanceMethodEnums.DOT.value
            self.distance_operator = "<->"
            self.score_expression = "-({column} <-> :vector)"
        else:
            # Cosine is the default: it is what VECTOR_DB_DISTANCE_METHOD ships with.
            self.index_opclass = PgVectorDistanceMethodEnums.COSINE.value
            self.distance_operator = "<=>"
            self.score_expression = "1 - ({column} <=> :vector)"

        self.logger = get_logger(__name__)
        self.default_index_name = lambda collection_name: f"{collection_name}_vector_idx"

    async def connect(self):
        # The pgvector extension, not a schema: the vector(N) column type does not exist
        # until this runs. Alembic also creates it, so this is the belt to that braces.
        async with self.db_client() as session:
            async with session.begin():
                await session.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))

    async def disconnect(self):
        pass

    async def is_collection_exists(self, collection_name: str) -> bool:
        async with self.db_client() as session:
            async with session.begin():
                list_tbl = sql_text("SELECT 1 FROM pg_tables WHERE tablename = :collection_name")
                results = await session.execute(list_tbl, {"collection_name": collection_name})
                return bool(results.scalar_one_or_none())

    async def list_all_collections(self) -> list:
        records = []
        async with self.db_client() as session:
            async with session.begin():
                list_tbl = sql_text("SELECT tablename FROM pg_tables WHERE tablename LIKE :prefix")
                results = await session.execute(
                    list_tbl, {"prefix": f"{self.pg_vector_table_prefix}%"}
                )
                records = results.scalars().all()
        return list(records)

    async def get_collection_info(self, collection_name: str) -> dict:
        async with self.db_client() as session:
            async with session.begin():
                table_info_sql = sql_text("""
                    SELECT schemaname, tablename, tableowner, tablespace, hasindexes
                    FROM pg_tables
                    WHERE tablename = :collection_name
                """)

                count_sql = sql_text(f"SELECT COUNT(*) FROM {collection_name}")

                table_info = await session.execute(
                    table_info_sql, {"collection_name": collection_name}
                )
                record_count = await session.execute(count_sql)

                table_data = table_info.fetchone()
                if not table_data:
                    return None

                return {
                    "table_info": {
                        "schemaname": table_data[0],
                        "tablename": table_data[1],
                        "tableowner": table_data[2],
                        "tablespace": table_data[3],
                        "hasindexes": table_data[4],
                    },
                    "record_count": record_count.scalar_one(),
                }

    async def delete_collection(self, collection_name: str):
        async with self.db_client() as session:
            async with session.begin():
                self.logger.info("collection_delete", collection=collection_name)

                # Use quote_identifier to safely inject the table name
                # or sanitize the input to prevent SQL injection
                safe_table_name = session.bind.dialect.identifier_preparer.quote(collection_name)
                delete_sql = sql_text(f"DROP TABLE IF EXISTS {safe_table_name}")

                await session.execute(delete_sql)

        return True

    async def create_collection(
        self, collection_name: str, embedding_size: int, do_reset: bool = False
    ):
        if do_reset:
            _ = await self.delete_collection(collection_name=collection_name)

        is_collection_existed = await self.is_collection_exists(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.info(
                "collection_create", collection=collection_name, embedding_size=embedding_size
            )
            async with self.db_client() as session:
                async with session.begin():
                    create_sql = sql_text(
                        f"CREATE TABLE {collection_name} ("
                        f"{PgVectorTableSchemeEnums.ID.value} bigserial PRIMARY KEY,"
                        f"{PgVectorTableSchemeEnums.TEXT.value} text, "
                        f"{PgVectorTableSchemeEnums.VECTOR.value} vector({embedding_size}), "
                        f"{PgVectorTableSchemeEnums.METADATA.value} jsonb DEFAULT '{{}}', "
                        f"{PgVectorTableSchemeEnums.CHUNK_ID.value} integer NULL, "
                        # ON DELETE CASCADE encodes Section 1's core claim in the
                        # schema: this table is a DERIVED projection of `chunks`, so
                        # a vector row has no meaning once its source chunk is gone.
                        #
                        # Without it, re-chunking is impossible: /process with
                        # do_reset deletes the project's chunks, the vector rows
                        # still reference them, and Postgres correctly refuses with
                        # a ForeignKeyViolation. Fixing the delete ORDER in the
                        # endpoint would work too, but every future caller would
                        # have to remember it. Cascading makes an orphan
                        # structurally impossible instead of merely avoided.
                        f"FOREIGN KEY ({PgVectorTableSchemeEnums.CHUNK_ID.value}) "
                        f"REFERENCES chunks(chunk_id) ON DELETE CASCADE"
                        ")"
                    )
                    await session.execute(create_sql)

            await self._create_metadata_index(collection_name=collection_name)
            return True

        return False

    async def _create_metadata_index(self, collection_name: str) -> None:
        """Index on metadata ->> 'ticket_id'.

        ``delete_by_metadata`` is called on every ticket resolve and reopen; without
        this index each of those is a sequential scan over the whole collection.
        """
        metadata_column = PgVectorTableSchemeEnums.METADATA.value
        async with self.db_client() as session:
            async with session.begin():
                # Promoted ticket answers: deleted by ticket_id on every resolve/reopen.
                await session.execute(
                    sql_text(
                        f"CREATE INDEX IF NOT EXISTS {collection_name}_ticket_id_idx "
                        f"ON {collection_name} (({metadata_column} ->> 'ticket_id'))"
                    )
                )
                # Handbook chunks: deleted by source+section on every section re-sync.
                await session.execute(
                    sql_text(
                        f"CREATE INDEX IF NOT EXISTS {collection_name}_source_section_idx "
                        f"ON {collection_name} "
                        f"(({metadata_column} ->> 'source'), ({metadata_column} ->> 'section'))"
                    )
                )

    async def is_index_existed(self, collection_name: str) -> bool:
        index_name = self.default_index_name(collection_name)
        async with self.db_client() as session:
            async with session.begin():
                check_sql = sql_text("""
                                    SELECT 1
                                    FROM pg_indexes
                                    WHERE tablename = :collection_name
                                    AND indexname = :index_name
                                    """)
                results = await session.execute(
                    check_sql, {"index_name": index_name, "collection_name": collection_name}
                )

                return bool(results.scalar_one_or_none())

    async def create_vector_index(
        self, collection_name: str, index_type: str = PgVectorIndexTypeEnums.HNSW.value
    ):
        is_index_existed = await self.is_index_existed(collection_name=collection_name)
        if is_index_existed:
            return False

        async with self.db_client() as session:
            async with session.begin():
                count_sql = sql_text(f"SELECT COUNT(*) FROM {collection_name}")
                result = await session.execute(count_sql)
                records_count = result.scalar_one()

                if records_count < self.index_threshold:
                    return False

                self.logger.info(
                    "vector_index_create_start",
                    collection=collection_name,
                    index_type=index_type,
                    record_count=records_count,
                )

                index_name = self.default_index_name(collection_name)
                create_idx_sql = sql_text(
                    f"CREATE INDEX {index_name} ON {collection_name} "
                    f"USING {index_type} "
                    f"({PgVectorTableSchemeEnums.VECTOR.value} {self.index_opclass})"
                )

                await session.execute(create_idx_sql)

                self.logger.info(
                    "vector_index_create_complete",
                    collection=collection_name,
                    index_name=index_name,
                )

        return True

    async def reset_vector_index(
        self, collection_name: str, index_type: str = PgVectorIndexTypeEnums.HNSW.value
    ) -> bool:
        index_name = self.default_index_name(collection_name)
        async with self.db_client() as session:
            async with session.begin():
                drop_sql = sql_text(f"DROP INDEX IF EXISTS {index_name}")
                await session.execute(drop_sql)

        return await self.create_vector_index(
            collection_name=collection_name, index_type=index_type
        )

    async def insert_one(
        self,
        collection_name: str,
        text: str,
        vector: list,
        metadata: dict = None,
        record_id: str = None,
    ):
        is_collection_existed = await self.is_collection_exists(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error("insert_one_unknown_collection", collection=collection_name)
            return False

        # record_id (the chunk_id FK) is deliberately optional: a promoted ticket answer
        # is derived from a Mongo/Postgres ticket, not from a handbook chunk, so it has
        # no chunk to point at. The column is nullable for exactly this row type.
        async with self.db_client() as session:
            async with session.begin():
                insert_sql = sql_text(
                    f"INSERT INTO {collection_name} "
                    f"({PgVectorTableSchemeEnums.TEXT.value}, "
                    f"{PgVectorTableSchemeEnums.VECTOR.value}, "
                    f"{PgVectorTableSchemeEnums.METADATA.value}, "
                    f"{PgVectorTableSchemeEnums.CHUNK_ID.value}) "
                    "VALUES (:text, :vector, :metadata, :chunk_id)"
                )

                metadata_json = (
                    json.dumps(metadata, ensure_ascii=False) if metadata is not None else "{}"
                )
                await session.execute(
                    insert_sql,
                    {
                        "text": text,
                        "vector": "[" + ",".join([str(v) for v in vector]) + "]",
                        "metadata": metadata_json,
                        "chunk_id": record_id,
                    },
                )

        # Outside the transaction above: create_vector_index opens its own session, and
        # nesting one inside an open transaction on the same pool can deadlock.
        await self.create_vector_index(collection_name=collection_name)

        return True

    async def insert_many(
        self,
        collection_name: str,
        texts: list,
        vectors: list,
        metadata: list = None,
        record_ids: list = None,
        batch_size: int = 50,
    ):
        is_collection_existed = await self.is_collection_exists(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error("insert_many_unknown_collection", collection=collection_name)
            return False

        if record_ids is None:
            record_ids = [None] * len(texts)

        if len(vectors) != len(record_ids):
            self.logger.error(
                "insert_many_length_mismatch",
                collection=collection_name,
                vector_count=len(vectors),
                record_id_count=len(record_ids),
            )
            return False

        if not metadata or len(metadata) == 0:
            metadata = [None] * len(texts)

        async with self.db_client() as session:
            async with session.begin():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    batch_vectors = vectors[i : i + batch_size]
                    batch_metadata = metadata[i : i + batch_size]
                    batch_record_ids = record_ids[i : i + batch_size]

                    values = []

                    # strict=True is load-bearing, not a lint fix. These four lists
                    # are parallel slices of one batch; if they ever disagree in
                    # length, plain zip() stops at the shortest and quietly indexes
                    # FEWER rows than were handed to it. Nothing raises, nothing logs,
                    # and the collection ends up missing chunks that the caller
                    # believes it inserted. Raising is the only safe behaviour.
                    for _text, _vector, _metadata, _record_id in zip(
                        batch_texts,
                        batch_vectors,
                        batch_metadata,
                        batch_record_ids,
                        strict=True,
                    ):
                        metadata_json = (
                            json.dumps(_metadata, ensure_ascii=False)
                            if _metadata is not None
                            else "{}"
                        )
                        values.append(
                            {
                                "text": _text,
                                "vector": "[" + ",".join([str(v) for v in _vector]) + "]",
                                "metadata": metadata_json,
                                "chunk_id": _record_id,
                            }
                        )

                    batch_insert_sql = sql_text(
                        f"INSERT INTO {collection_name} "
                        f"({PgVectorTableSchemeEnums.TEXT.value}, "
                        f"{PgVectorTableSchemeEnums.VECTOR.value}, "
                        f"{PgVectorTableSchemeEnums.METADATA.value}, "
                        f"{PgVectorTableSchemeEnums.CHUNK_ID.value}) "
                        f"VALUES (:text, :vector, :metadata, :chunk_id)"
                    )

                    await session.execute(batch_insert_sql, values)

        await self.create_vector_index(collection_name=collection_name)

        return True

    async def delete_by_metadata(self, collection_name: str, criteria: dict) -> int:
        """Delete every row matching ALL of `criteria`. Returns the row count.

        Half of Section 1's delete-then-insert: the sync deletes by stable key and only
        then inserts the current state, so running it twice cannot leave two rows for
        one logical item.

        Refuses an empty criteria dict rather than treating it as "match everything".
        An unguarded DELETE with no WHERE would silently empty the collection, and the
        caller that passed {} by accident would see a plausible-looking row count.
        """
        if not criteria:
            self.logger.error("delete_by_metadata_refused_empty", collection=collection_name)
            return 0

        is_collection_existed = await self.is_collection_exists(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error("delete_by_metadata_unknown_collection", collection=collection_name)
            return 0

        metadata_column = PgVectorTableSchemeEnums.METADATA.value

        # Enumerated bind names: the criteria KEYS are data, so they are bound as
        # parameters too rather than interpolated into the SQL.
        conditions = []
        params: dict = {}
        for index, (key, value) in enumerate(criteria.items()):
            conditions.append(f"{metadata_column} ->> :k{index} = :v{index}")
            params[f"k{index}"] = key
            params[f"v{index}"] = str(value)

        async with self.db_client() as session:
            async with session.begin():
                delete_sql = sql_text(
                    f"DELETE FROM {collection_name} WHERE " + " AND ".join(conditions)
                )
                result = await session.execute(delete_sql, params)
                return result.rowcount

    async def search_by_vector(self, collection_name: str, vector: list, limit: int):
        is_collection_existed = await self.is_collection_exists(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error("search_unknown_collection", collection=collection_name)
            return False

        vector = "[" + ",".join([str(v) for v in vector]) + "]"
        vector_column = PgVectorTableSchemeEnums.VECTOR.value

        async with self.db_client() as session:
            async with session.begin():
                # metadata is SELECTed because RetrievedDocument requires it, and the
                # department filter plus handbook-precedence ranking both read it.
                # ORDER BY is on the distance operator ascending, not on a computed
                # score descending: only the former can use the HNSW index.
                search_sql = sql_text(
                    f"SELECT {PgVectorTableSchemeEnums.TEXT.value} as text, "
                    f"{PgVectorTableSchemeEnums.METADATA.value} as metadata, "
                    f"{self.score_expression.format(column=vector_column)} as score"
                    f" FROM {collection_name}"
                    f" ORDER BY {vector_column} {self.distance_operator} :vector ASC "
                    "LIMIT :limit"
                )

                result = await session.execute(search_sql, {"vector": vector, "limit": limit})

                records = result.fetchall()

                return [
                    RetrievedDocument(
                        text=record.text,
                        score=record.score,
                        metadata=self._as_dict(record.metadata),
                    )
                    for record in records
                ]

    @staticmethod
    def _as_dict(metadata) -> dict:
        """JSONB comes back as a dict or as a raw string depending on the driver's
        codec, so normalise it rather than trusting one shape."""
        if isinstance(metadata, str):
            try:
                return json.loads(metadata)
            except json.JSONDecodeError:
                return {}
        return metadata or {}
