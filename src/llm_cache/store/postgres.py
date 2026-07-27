from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import class_row, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from llm_cache.models import CacheEntry, Embedding

_ENTRY_COLUMN_NAMES = (
  'scope',
  'fingerprint',
  'query',
  'response',
  'created_at',
  'expires_at',
  'hits',
  'last_used_at',
  'metadata',
)

_WRITE_COLUMN_NAMES = (*_ENTRY_COLUMN_NAMES, 'embedding')

_UPSERT_COLUMN_NAMES = (
  'query',
  'response',
  'created_at',
  'expires_at',
  'metadata',
  'embedding',
)

_COLUMNS = sql.SQL(', ').join(sql.Identifier(n) for n in _ENTRY_COLUMN_NAMES)
_WRITE_COLUMNS = sql.SQL(', ').join(sql.Identifier(n) for n in _WRITE_COLUMN_NAMES)
_PLACEHOLDERS = sql.SQL(', ').join(sql.Placeholder() * len(_WRITE_COLUMN_NAMES))
_UPSERT_ASSIGNMENTS = sql.SQL(', ').join(
  sql.SQL('{col} = EXCLUDED.{col}').format(col=sql.Identifier(n))
  for n in _UPSERT_COLUMN_NAMES
)


class Postgres:
  def __init__(
    self, dsn: str, *, table: str = 'llm_cache', max_entries: int = 2000
  ) -> None:
    """Connect to an existing cache table.

    The schema is owned by Alembic — run `alembic upgrade head` before
    constructing this. Nothing here creates or alters tables, so the migration
    files stay the single description of the schema.
    """
    self.max_entries = max_entries
    self._table = sql.Identifier(table)
    self._pool = ConnectionPool(dsn, open=True, configure=register_vector)

  def get_exact(self, scope: str, fingerprint: str) -> CacheEntry | None:
    query = sql.SQL(
      'SELECT {columns} FROM {table} '
      'WHERE scope = %s AND fingerprint = %s '
      'AND (expires_at IS NULL OR expires_at > now())'
    ).format(columns=_COLUMNS, table=self._table)
    with self._pool.connection() as conn:
      with conn.cursor(row_factory=class_row(CacheEntry)) as cur:
        cur.execute(query, (scope, fingerprint))
        return cur.fetchone()

  def search(
    self, scope: str, embedding: Embedding, limit: int
  ) -> list[tuple[CacheEntry, float]]:
    """Nearest neighbours in `scope`, best first, as (entry, similarity).

    Returns the top `limit` regardless of how far away they are — deciding what
    counts as close enough is the caller's threshold to apply, not the store's.
    """
    query = sql.SQL(
      'SELECT {columns}, 1 - (embedding <=> %s::vector) AS similarity '
      'FROM {table} '
      'WHERE scope = %s AND (expires_at IS NULL OR expires_at > now()) '
      'ORDER BY embedding <=> %s::vector '
      'LIMIT %s'
    ).format(columns=_COLUMNS, table=self._table)
    vector = list(embedding)
    with self._pool.connection() as conn:
      with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (vector, scope, vector, limit))
        return [
          (CacheEntry(**{k: row[k] for k in _ENTRY_COLUMN_NAMES}), row['similarity'])
          for row in cur.fetchall()
        ]

  def put(self, entry: CacheEntry, embedding: Embedding) -> None:
    query = sql.SQL(
      'INSERT INTO {table} ({columns}) VALUES ({placeholders}) '
      'ON CONFLICT (scope, fingerprint) DO UPDATE SET '
      '{assignments}'
    ).format(
      table=self._table,
      columns=_WRITE_COLUMNS,
      placeholders=_PLACEHOLDERS,
      assignments=_UPSERT_ASSIGNMENTS,
    )
    with self._pool.connection() as conn:
      conn.execute(
        query,
        (
          entry.scope,
          entry.fingerprint,
          entry.query,
          entry.response,
          entry.created_at,
          entry.expires_at,
          entry.hits,
          entry.last_used_at,
          Jsonb(entry.metadata),
          list(embedding),
        ),
      )

  def touch(self, scope: str, fingerprint: str) -> None:
    query = sql.SQL(
      'UPDATE {table} SET hits = hits + 1, last_used_at = now() '
      'WHERE scope = %s AND fingerprint = %s'
    ).format(table=self._table)
    with self._pool.connection() as conn:
      conn.execute(query, (scope, fingerprint))

  def evict_expired(self) -> int:
    query = sql.SQL(
      """
        DELETE FROM {table}
        WHERE expires_at <= now()
      """
    ).format(table=self._table)

    with self._pool.connection() as conn:
      cursor = conn.execute(query)
      return cursor.rowcount

  def purge(self) -> int:
    query = sql.SQL(
      """
        DELETE FROM {table}
        WHERE (scope, fingerprint) IN (
            SELECT scope, fingerprint
            FROM {table}
            ORDER BY hits DESC, created_at DESC
            OFFSET %s
        )
      """
    ).format(table=self._table)

    with self._pool.connection() as conn:
      cursor = conn.execute(query, (self.max_entries,))
      return cursor.rowcount

  def clear(self, scope: str) -> int:
    query = sql.SQL('DELETE from {table} WHERE scope = %s').format(table=self._table)
    with self._pool.connection() as conn:
      cursor = conn.execute(query, (scope,))
      return cursor.rowcount

  def close(self) -> None:
    self._pool.close()
