"""phase 4 mitigation strategy generation schema

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-29

Adds the single new table the mitigation layer needs. Like 0008, this is
written to be safely re-runnable from any partial prior state: enum types
are created with checkfirst=True and referenced with create_type=False,
and the table uses if_not_exists=True.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

_mitigation_priority = postgresql.ENUM(
    "low", "medium", "high", "critical", name="mitigation_priority", create_type=False
)
_mitigation_urgency = postgresql.ENUM(
    "immediate", "short_term", "long_term", name="mitigation_urgency", create_type=False
)
_mitigation_validation_status = postgresql.ENUM(
    "valid", "flagged", "rejected", name="mitigation_validation_status", create_type=False
)

_ALL_ENUMS = [_mitigation_priority, _mitigation_urgency, _mitigation_validation_status]


def upgrade() -> None:
    bind = op.get_bind()
    for enum_type in _ALL_ENUMS:
        enum_type.create(bind, checkfirst=True)

    op.create_table(
        "mitigation_strategies",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("investigation_runs.run_id"), nullable=False),
        sa.Column("strategy_key", sa.String(64), nullable=False),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("strategy_type", sa.String(64), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("why_this_mitigation", sa.Text(), nullable=False),
        sa.Column("investigation_basis", sa.JSON(), nullable=False),
        sa.Column("risk_addressed", sa.Text(), nullable=False),
        sa.Column("implementation_steps", sa.JSON(), nullable=False),
        sa.Column("expected_impact", sa.Text(), nullable=False),
        sa.Column("priority", _mitigation_priority, nullable=False),
        sa.Column("urgency", _mitigation_urgency, nullable=False),
        sa.Column("confidence", sa.Numeric(5, 2), nullable=False),
        sa.Column("limitations", sa.Text(), nullable=False),
        sa.Column("dependencies", sa.JSON(), nullable=False),
        sa.Column("success_metrics", sa.JSON(), nullable=False),
        sa.Column("is_confirmed_corrective_action", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("validation_status", _mitigation_validation_status, nullable=False),
        sa.Column("validation_notes", sa.Text()),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("run_id", "strategy_key", name="uq_mitigation_run_strategy_key"),
        if_not_exists=True,
    )
    op.create_index("ix_mitigation_strategies_run", "mitigation_strategies", ["run_id"], if_not_exists=True)


def downgrade() -> None:
    op.drop_table("mitigation_strategies")

    bind = op.get_bind()
    for enum_type in reversed(_ALL_ENUMS):
        enum_type.drop(bind, checkfirst=True)
