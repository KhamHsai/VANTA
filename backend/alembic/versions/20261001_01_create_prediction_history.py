"""Create prediction history table.

Revision ID: 20261001_01
Revises:
Create Date: 2026-10-01
"""

from collections.abc import Sequence

from alembic import context, op
import sqlalchemy as sa


revision: str = "20261001_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "prediction_history"
INDEX_NAME = "ix_prediction_history_created_at"
EXPECTED_COLUMNS = {
    "id",
    "predicted_label",
    "confidence",
    "is_uncertain",
    "score_general",
    "score_metal",
    "score_organic",
    "score_paper",
    "score_plastic",
    "source_type",
    "created_at",
}


def upgrade() -> None:
    if context.is_offline_mode():
        _create_prediction_history_table()
        _create_created_at_index()
        return

    inspector = sa.inspect(op.get_bind())
    if inspector.has_table(TABLE_NAME):
        existing_columns = {
            column["name"] for column in inspector.get_columns(TABLE_NAME)
        }
        if existing_columns != EXPECTED_COLUMNS:
            raise RuntimeError(
                "An incompatible prediction_history table already exists; "
                "manual review is required before migration."
            )
    else:
        _create_prediction_history_table()

    # MySQL uses non-transactional DDL. This check makes a retry safe if the
    # connection drops after table creation but before Alembic records the revision.
    inspector = sa.inspect(op.get_bind())
    existing_indexes = {
        index["name"] for index in inspector.get_indexes(TABLE_NAME)
    }
    if INDEX_NAME not in existing_indexes:
        _create_created_at_index()


def _create_prediction_history_table() -> None:
    op.create_table(
        TABLE_NAME,
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("predicted_label", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("is_uncertain", sa.Boolean(), nullable=False),
        sa.Column("score_general", sa.Float(), nullable=False),
        sa.Column("score_metal", sa.Float(), nullable=False),
        sa.Column("score_organic", sa.Float(), nullable=False),
        sa.Column("score_paper", sa.Float(), nullable=False),
        sa.Column("score_plastic", sa.Float(), nullable=False),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "predicted_label IN ('general', 'metal', 'organic', 'paper', 'plastic')",
            name="ck_prediction_history_label",
        ),
        sa.CheckConstraint(
            "source_type IN ('camera', 'upload')",
            name="ck_prediction_history_source_type",
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def _create_created_at_index() -> None:
    op.create_index(
        INDEX_NAME,
        TABLE_NAME,
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name=TABLE_NAME)
    op.drop_table(TABLE_NAME)
