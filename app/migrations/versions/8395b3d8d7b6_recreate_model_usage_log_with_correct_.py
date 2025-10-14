"""Recreate model_usage_log with correct schema and ensure users table exists

Revision ID: 8395b3d8d7b6
Revises: c372c6334c5d
Create Date: 2025-06-26 18:07:25.219151
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = '8395b3d8d7b6'
down_revision: Union[str, None] = 'c372c6334c5d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()

    # Vérifie si la table 'users' existe
    result = conn.execute(text("SHOW TABLES LIKE 'users'")).fetchone()
    if not result:
        op.create_table(
            'users',
            sa.Column('user_mail', sa.String(255), primary_key=True),
            sa.Column('role', sa.String(255), nullable=False),
            sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
            mysql_collate='utf8mb4_0900_ai_ci',
            mysql_default_charset='utf8mb4',
            mysql_engine='InnoDB'
        )

    # Supprimer 'model_usage_log' s’il existe
    op.drop_table('model_usage_log')

    # Recréer la table 'model_usage_log' avec le bon schéma
    op.create_table(
        'model_usage_log',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('user_mail', sa.String(256), nullable=False),
        sa.Column('model_name', sa.String(256), nullable=False),
        sa.Column('input_tokens', sa.BigInteger, nullable=False, server_default='0'),
        sa.Column('output_tokens', sa.BigInteger, nullable=False, server_default='0'),
        sa.Column('timestamp', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['user_mail'], ['users.user_mail'], name='fk_model_usage_log_user_mail'),
        mysql_collate='utf8mb4_0900_ai_ci',
        mysql_default_charset='utf8mb4',
        mysql_engine='InnoDB'
    )

    # Ajouter les index nécessaires
    op.create_index('idx_user_mail', 'model_usage_log', ['user_mail'])
    op.create_index('idx_model_name', 'model_usage_log', ['model_name'])
    op.create_index('idx_timestamp', 'model_usage_log', ['timestamp'])

def downgrade() -> None:
    """Downgrade schema."""
    # Supprimer la table recréée
    op.drop_table('model_usage_log')

    # Recréer l’ancienne version si nécessaire
    op.create_table(
        'model_usage_log',
        sa.Column('id', mysql.INTEGER(), autoincrement=True, nullable=False),
        sa.Column('model_id', mysql.INTEGER(), autoincrement=False, nullable=False),
        sa.Column('user_id', mysql.INTEGER(), autoincrement=False, nullable=False),
        sa.Column('usage_timestamp', mysql.DATETIME(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('usage_cost', mysql.DECIMAL(precision=12, scale=2), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        mysql_collate='utf8mb4_0900_ai_ci',
        mysql_default_charset='utf8mb4',
        mysql_engine='InnoDB'
    )

    # Optionnel : supprimer la table users si elle a été créée ici
    # op.drop_table('users')
