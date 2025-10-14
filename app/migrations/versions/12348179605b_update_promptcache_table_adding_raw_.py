"""add raw_prompt, model_name, max_tokens to PromptCache

Revision ID: 12348179605b
Revises: fa2cec5e128e
Create Date: 2025-07-09 09:42:24.998741
"""

from alembic import op
import sqlalchemy as sa
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = '12348179605b'
down_revision: Union[str, None] = 'fa2cec5e128e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema by adding new columns to PromptCache."""
    op.add_column('prompt_cache', sa.Column('raw_prompt', sa.Text(), nullable=True))
    op.add_column('prompt_cache', sa.Column('model_name', sa.String(length=100), nullable=True))
    op.add_column('prompt_cache', sa.Column('max_tokens', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema by removing columns from PromptCache."""
    op.drop_column('prompt_cache', 'max_tokens')
    op.drop_column('prompt_cache', 'model_name')
    op.drop_column('prompt_cache', 'raw_prompt')
