"""add promotion hold

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-09-23

"""

import sqlalchemy as sa
from alembic import op

revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tickets",
        sa.Column(
            "promotion_held", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
    )
    op.add_column("tickets", sa.Column("promotion_hold_reason", sa.Text(), nullable=True))

    # A held ticket is a work queue — "which handbook sections might be wrong" — so
    # it needs to be findable, not just recorded. Partial index: held tickets are the
    # rare case, and indexing only them keeps it small.
    op.create_index(
        "ix_ticket_promotion_held",
        "tickets",
        ["promotion_held"],
        postgresql_where=sa.text("promotion_held"),
    )


def downgrade() -> None:
    op.drop_index("ix_ticket_promotion_held", table_name="tickets")
    op.drop_column("tickets", "promotion_hold_reason")
    op.drop_column("tickets", "promotion_held")
