"""Add PromptCache table

Revision ID: fa2cec5e128e
Revises: 8395b3d8d7b6
Create Date: 2025-07-08 17:38:46.753332

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'fa2cec5e128e'
down_revision: Union[str, None] = '8395b3d8d7b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "prompt_cache",
        sa.Column("prompt_key", sa.String(length=64), primary_key=True),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("prompt_cache")
