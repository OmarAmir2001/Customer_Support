"""add escalation tickets

Revision ID: a1b2c3d4e5f6
Revises: fb42b724b8d9
Create Date: 2026-09-19

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "a1b2c3d4e5f6"
down_revision = "fb42b724b8d9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # pgvector's extension belongs in a migration, not in application startup:
    # schema changes are Alembic's job, and a fresh clone must get it too.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tickets",
        sa.Column("ticket_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticket_uuid", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("student_id", sa.String(), nullable=False),
        sa.Column("department", sa.String(), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("failed_gate", sa.String(), nullable=False),
        sa.Column("gate_reason", sa.Text(), nullable=False),
        sa.Column("gate_scores", postgresql.JSONB(), nullable=True),
        sa.Column("retrieved_chunks", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("advisor_answer", sa.Text(), nullable=True),
        sa.Column("resolved_by", sa.String(), nullable=True),
        sa.Column("promoted_to_kb", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duplicate_of", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("ticket_id"),
        sa.UniqueConstraint("ticket_uuid"),
        sa.ForeignKeyConstraint(["duplicate_of"], ["tickets.ticket_id"]),
    )
    op.create_index("ix_ticket_status", "tickets", ["status"])
    op.create_index("ix_ticket_thread_id", "tickets", ["thread_id"])
    op.create_index("ix_ticket_student_id", "tickets", ["student_id"])
    op.create_index(
        "ix_ticket_status_department_created", "tickets", ["status", "department", "created_at"]
    )

    op.create_table(
        "ticket_status_history",
        sa.Column("history_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticket_id", sa.Integer(), nullable=False),
        sa.Column("from_status", sa.String(), nullable=True),
        sa.Column("to_status", sa.String(), nullable=False),
        sa.Column("actor", sa.String(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("history_id"),
        sa.ForeignKeyConstraint(["ticket_id"], ["tickets.ticket_id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_ticket_history_ticket_created", "ticket_status_history", ["ticket_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_ticket_history_ticket_created", table_name="ticket_status_history")
    op.drop_table("ticket_status_history")

    op.drop_index("ix_ticket_status_department_created", table_name="tickets")
    op.drop_index("ix_ticket_student_id", table_name="tickets")
    op.drop_index("ix_ticket_thread_id", table_name="tickets")
    op.drop_index("ix_ticket_status", table_name="tickets")
    op.drop_table("tickets")
    # The extension is deliberately NOT dropped: other tables may use vector columns.