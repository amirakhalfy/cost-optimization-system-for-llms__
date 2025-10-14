from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'c372c6334c5d'
down_revision: Union[str, None] = '87e8bf98b765'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema by replacing user_project_budgets table and ensuring models.parameters is BigInteger."""

    # Drop the old user_project_budgets table
    op.drop_table('user_project_budgets')

    # Recreate user_project_budgets with correct schema
    op.create_table(
        'user_project_budgets',
        sa.Column('user_mail', sa.String(length=255), nullable=False),
        sa.Column('project_budget_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['user_mail'], ['users.user_mail'], name='fk_user_project_budgets_user_mail'),
        sa.ForeignKeyConstraint(['project_budget_id'], ['project_budgets.id'], name='fk_user_project_budgets_project_budget_id')
    )

    # Ensure models.parameters column type remains BigInteger
    op.alter_column(
        'models',
        'parameters',
        existing_type=sa.Integer(),  # or the current type if different
        type_=sa.BigInteger(),
        existing_nullable=True
    )


def downgrade() -> None:
    """Downgrade schema by restoring previous user_project_budgets and models.parameters."""

    # Drop the new version of the user_project_budgets table
    op.drop_table('user_project_budgets')

    # Recreate the original user_project_budgets table
    op.create_table(
        'user_project_budgets',
        sa.Column('user_mail', sa.String(length=255), nullable=False),
        sa.Column('project_budget_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['user_mail'], ['users.user_mail'], name='fk_user_project_budgets_user_mail'),
        sa.ForeignKeyConstraint(['project_budget_id'], ['project_budgets.id'], name='fk_user_project_budgets_project_budget_id')
    )

    # Revert models.parameters column type if you want (but keep as BigInteger to avoid issues)
    op.alter_column(
        'models',
        'parameters',
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),  # only if safe, else keep as BigInteger
        existing_nullable=True
    )
