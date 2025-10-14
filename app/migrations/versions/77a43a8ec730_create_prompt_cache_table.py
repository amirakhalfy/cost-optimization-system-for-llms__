"""Create prompt_cache table

Revision ID: 77a43a8ec730
Revises: 8c8dd609ba80
Create Date: 2025-07-21 19:53:18.202985

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '77a43a8ec730'
down_revision = '8c8dd609ba80'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'prompt_cache',
        sa.Column('prompt_key', sa.String(length=64), primary_key=True, nullable=False),
        sa.Column('raw_prompt', sa.Text(), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('max_tokens', sa.Integer(), nullable=False),
        sa.Column('response', sa.Text(), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=False),
        sa.Column('output_tokens', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        mysql_engine='InnoDB',
        mysql_charset='utf8mb4'
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('prompt_cache')
