"""create llm_cache table

Revision ID: e9e9f1ccc25d
Revises:
Create Date: 2026-07-27 19:46:45.017743

Table name and vector width are literals, not config lookups. A migration is a
record of what was actually applied, so it has to produce the same schema every
time it runs. Changing the embedding model means a new migration, not a new
value for an environment variable this file happened to read.
"""

from collections.abc import Sequence

from alembic import op

revision: str = 'e9e9f1ccc25d'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
  op.execute('CREATE EXTENSION IF NOT EXISTS vector')
  op.execute("""
    CREATE TABLE llm_cache (
      scope         text          NOT NULL,
      fingerprint   text          NOT NULL,
      query         text          NOT NULL,
      response      text          NOT NULL,
      embedding     vector(1536)  NOT NULL,
      created_at    timestamptz   NOT NULL DEFAULT now(),
      expires_at    timestamptz,
      hits          integer       NOT NULL DEFAULT 0,
      last_used_at  timestamptz,
      metadata      jsonb         NOT NULL DEFAULT '{}',
      PRIMARY KEY (scope, fingerprint)
    )
  """)
  # Approximate nearest-neighbour index for search(). Cosine ops must match the
  # `<=>` operator the query uses, or Postgres silently falls back to a scan.
  op.execute("""
    CREATE INDEX llm_cache_embedding_idx ON llm_cache
      USING hnsw (embedding vector_cosine_ops)
  """)
  # Partial: only rows that can actually expire are worth indexing for
  # evict_expired().
  op.execute("""
    CREATE INDEX llm_cache_expires_at_idx ON llm_cache (expires_at)
      WHERE expires_at IS NOT NULL
  """)


def downgrade() -> None:
  op.execute('DROP TABLE IF EXISTS llm_cache')
