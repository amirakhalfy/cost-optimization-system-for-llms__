"""Add user_project_budgets and model_usage_log tables

Revision ID: 87e8bf98b765
Revises: 38ecc7cdf47a
Create Date: 2025-06-08 17:20:19.871404
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = '87e8bf98b765'
down_revision: Union[str, None] = '38ecc7cdf47a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Do NOT alter 'parameters' column type here because of out-of-range data!
    # Instead, just add new tables.

    op.create_table(
        'user_project_budgets',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer, nullable=False),
        sa.Column('project_id', sa.Integer, nullable=False),
        sa.Column('budget', sa.Numeric(12, 2), nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        mysql_engine='InnoDB',
        mysql_charset='utf8mb4',
    )

    op.create_table(
        'model_usage_log',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('model_id', sa.Integer, nullable=False),
        sa.Column('user_id', sa.Integer, nullable=False),
        sa.Column('usage_timestamp', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('usage_cost', sa.Numeric(12, 2), nullable=True),
        mysql_engine='InnoDB',
        mysql_charset='utf8mb4',
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('model_usage_log')
    op.drop_table('user_project_budgets')

    # Do NOT revert 'parameters' column type change because it was never changed in upgrade.
