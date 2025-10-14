"""Add PromptEmbedding and cost fields

Revision ID: 8c8dd609ba80
Revises: eb8d7e66359c
Create Date: 2025-07-21 17:22:32.448319
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '8c8dd609ba80'
down_revision: Union[str, None] = 'eb8d7e66359c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: create prompt_embeddings table."""
    op.create_table(
        'prompt_embeddings',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('raw_prompt', sa.Text(), nullable=False),
        sa.Column('embedding', sa.Text(), nullable=False),
        sa.Column('response', sa.Text(), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=False),
        sa.Column('output_tokens', sa.Integer(), nullable=False),
        sa.Column('cost', sa.Numeric(10, 4), nullable=False, server_default='0.0'),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now())
    )


def downgrade() -> None:
    """Downgrade schema: drop prompt_embeddings table."""
    op.drop_table('prompt_embeddings')
