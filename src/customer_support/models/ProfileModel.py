"""The only place that issues student-profile SQL.

Controllers decide WHAT a profile should become; this class performs the write. Same
split as TicketModel: the lifecycle rules live in exactly one controller, and the
SQL lives in exactly one model.
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from customer_support.helpers.logging_config import get_logger
from customer_support.models.llm_schemas.student_profile import PROFILE_FIELDS, StudentProfile

from .BaseDataModel import BaseDataModel
from .db_schemas.customer_support.schemes.student_profile import (
    StudentProfile as StudentProfileRow,
)


class ProfileModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.logger = get_logger(__name__)

    @classmethod
    async def create_instance(cls, db_client: object):
        return cls(db_client=db_client)

    async def get_profile(self, student_id: str) -> StudentProfile | None:
        async with self.db_client() as session:
            result = await session.execute(
                select(StudentProfileRow).where(StudentProfileRow.student_id == student_id)
            )
            row = result.scalar_one_or_none()

        return StudentProfile.model_validate(row) if row is not None else None

    async def apply_patch(self, student_id: str, fields: dict) -> StudentProfile:
        """Insert or patch a profile, atomically.

        ON CONFLICT rather than read-then-write: a student can have two requests in
        flight, and the unique constraint on student_id would turn the second insert
        into an IntegrityError. Upserting means concurrent extractions merge instead
        of one of them failing.

        Only the fields passed in are written, so this cannot clear a column the
        caller did not mention. The caller has already stripped nulls
        (``StudentProfileUpdate.learned_fields``); this is the second place the same
        rule holds, because losing a known fact is the failure worth defending twice.
        """
        patch = {field: value for field, value in fields.items() if field in PROFILE_FIELDS}
        now = datetime.now(UTC)

        statement = (
            insert(StudentProfileRow)
            .values(
                student_id=student_id,
                last_extracted_at=now,
                extraction_count=1,
                **patch,
            )
            .on_conflict_do_update(
                index_elements=["student_id"],
                set_={
                    **patch,
                    "last_extracted_at": now,
                    # Counted in SQL, not read-modify-written in Python, so parallel
                    # extractions cannot both read 4 and both write 5.
                    "extraction_count": StudentProfileRow.extraction_count + 1,
                    "updated_at": now,
                },
            )
            .returning(StudentProfileRow)
        )

        async with self.db_client() as session:
            async with session.begin():
                result = await session.execute(statement)
                row = result.scalar_one()
                profile = StudentProfile.model_validate(row)

        self.logger.info(
            "profile_patched",
            student_id=student_id,
            fields=sorted(patch),
        )
        return profile

    async def touch_extraction(self, student_id: str) -> None:
        """Record that extraction RAN and learned nothing.

        Worth a write: without it the gate's hit rate is unmeasurable, and
        "extraction is cheap because it rarely runs" stays an untested claim.
        """
        await self.apply_patch(student_id=student_id, fields={})

    async def delete_profile(self, student_id: str) -> bool:
        """Wipe a student's long-term memory. A real product needs this for privacy
        and reset requests, and it is the honest implementation of the
        /profile DELETE endpoint."""
        async with self.db_client() as session:
            async with session.begin():
                result = await session.execute(
                    select(StudentProfileRow).where(StudentProfileRow.student_id == student_id)
                )
                row = result.scalar_one_or_none()
                if row is None:
                    return False
                await session.delete(row)

        self.logger.info("profile_deleted", student_id=student_id)
        return True
