from enum import Enum


class VectorDBEnums(Enum):
    QDRANT = "QDRANT"
    PGVECTOR = "PGVECTOR"


class DistanceMethodEnums(Enum):
    COSINE = "cosine"
    DOT = "dot"


class PgVectorTableSchemeEnums(Enum):
    ID = "id"
    TEXT = "text"
    VECTOR = "vector"
    CHUNK_ID = "chunk_id"
    METADATA = "metadata"
    _PREFIX = "pgvector_"


class PgVectorDistanceMethodEnums(Enum):
    """pgvector operator classes, one per DistanceMethodEnums member.

    DOT was `vector_l2_ops`, which is EUCLIDEAN distance, not a dot product — so
    `VECTOR_DB_DISTANCE_METHOD=dot` selected inner product on Qdrant and L2 on
    pgvector. The same setting meant two different metrics, silently, because both
    configurations run and return plausible results. For an interface whose only
    purpose is swappability that is the abstraction leaking.

    Resolved in favour of the name: `dot` means dot product, and pgvector spells
    that `vector_ip_ops` with the `<#>` operator. If L2 is ever wanted it should be
    its own member, with `models.Distance.EUCLID` on the Qdrant side.
    """

    COSINE = "vector_cosine_ops"
    DOT = "vector_ip_ops"


class PgVectorIndexTypeEnums(Enum):
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"
