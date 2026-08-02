"""Store the exact deduction breakdown with each health snapshot.

Revision ID: 20260725_health_deduction_snapshot
Revises: 20260719_prediction_snapshots
Create Date: 2026-07-25
"""
from alembic import op
import sqlalchemy as sa


revision = "20260725_health_deduction_snapshot"
down_revision = "20260719_prediction_snapshots"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("health_scores", sa.Column("deduction_snapshot", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("health_scores", "deduction_snapshot")
