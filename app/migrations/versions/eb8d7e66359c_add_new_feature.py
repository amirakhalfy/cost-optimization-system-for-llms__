"""Add cost column to model_usage_log

Revision ID: eb8d7e66359c
Revises: 12348179605b
Create Date: 2025-07-16 21:28:09.648738
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'eb8d7e66359c'
down_revision: Union[str, None] = '12348179605b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema by adding 'cost' column to model_usage_log."""
    op.add_column(
        'model_usage_log',
        sa.Column('cost', sa.Numeric(10, 4), nullable=False, server_default="0.0")
    )


def downgrade() -> None:
    """Downgrade schema by removing 'cost' column from model_usage_log."""
    op.drop_column('model_usage_log', 'cost')
