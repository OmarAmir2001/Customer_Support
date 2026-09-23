"""add student profiles

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-09-22

"""

import sqlalchemy as sa
from alembic import op

revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "student_profiles",
        sa.Column("profile_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("student_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("department", sa.String(), nullable=True),
        sa.Column("gpa", sa.Float(), nullable=True),
        sa.Column("preferred_language", sa.String(), nullable=True),
        sa.Column("last_extracted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extraction_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("profile_id"),
        # One profile per student, enforced by the schema rather than by convention:
        # the "single patched document" decision is the whole design (Section 6).
        sa.UniqueConstraint("student_id"),
    )
    op.create_index("ix_student_profile_student_id", "student_profiles", ["student_id"])


def downgrade() -> None:
    op.drop_index("ix_student_profile_student_id", table_name="student_profiles")
    op.drop_table("student_profiles")
