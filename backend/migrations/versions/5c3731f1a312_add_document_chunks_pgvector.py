"""add_document_chunks_pgvector

Revision ID: 5c3731f1a312
Revises: 37328f5c42df
Create Date: 2026-07-01 01:31:37.577742

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = '5c3731f1a312'
down_revision: Union[str, Sequence[str], None] = '37328f5c42df'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # We must ensure the vector extension is created
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    op.create_table(
        'document_chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('text', sa.String(), nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=1536), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_document_chunks_id'), 'document_chunks', ['id'], unique=False)
    op.create_index(op.f('ix_document_chunks_source'), 'document_chunks', ['source'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_document_chunks_source'), table_name='document_chunks')
    op.drop_index(op.f('ix_document_chunks_id'), table_name='document_chunks')
    op.drop_table('document_chunks')
    op.execute("DROP EXTENSION IF EXISTS vector;")
