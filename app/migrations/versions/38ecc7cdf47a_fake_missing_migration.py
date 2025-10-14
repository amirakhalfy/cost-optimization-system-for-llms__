"""Fake missing migration

Revision ID: 38ecc7cdf47a
Revises: a23abb777e0f
Create Date: 2025-06-08 17:19:27.106322

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '38ecc7cdf47a'
down_revision: Union[str, None] = 'a23abb777e0f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
