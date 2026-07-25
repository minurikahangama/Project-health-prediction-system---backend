"""store current Jira evidence and auditable prediction snapshots

Revision ID: 20260719_prediction_snapshots
Revises:
Create Date: 2026-07-19
"""
from alembic import op
import sqlalchemy as sa

revision = "20260719_prediction_snapshots"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("projects", sa.Column("jira_metrics_snapshot", sa.JSON(), nullable=True))
    op.add_column("health_scores", sa.Column("feature_vector", sa.JSON(), nullable=True))
    op.add_column("health_scores", sa.Column("shap_explanation", sa.JSON(), nullable=True))
    op.create_table("jira_evidence_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("collected_at", sa.DateTime(), nullable=False),
    )


def downgrade():
    op.drop_table("jira_evidence_snapshots")
    op.drop_column("health_scores", "shap_explanation")
    op.drop_column("health_scores", "feature_vector")
    op.drop_column("projects", "jira_metrics_snapshot")
