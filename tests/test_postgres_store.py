import pytest

from llm_cache.models import CacheEntry
from llm_cache.store.postgres import Postgres

pytestmark = pytest.mark.integration

SCOPE = 'gpt-4o|thread:abc'
VECTOR = [0.01] * 1536


def _insert(store: Postgres, fingerprint: str, expires: str | None = None) -> None:
  """Seed a row directly, until put() exists."""
  expires_at = f"now() + interval '{expires}'" if expires else 'NULL'
  with store._pool.connection() as conn:
    conn.execute(
      'INSERT INTO llm_cache'
      ' (scope, fingerprint, query, response, embedding, expires_at, metadata)'
      f' VALUES (%s, %s, %s, %s, %s, {expires_at}, %s)',
      (SCOPE, fingerprint, 'what is 2+2', '4', VECTOR, '{"tokens": 7}'),
    )


def test_get_exact_returns_entry(store: Postgres) -> None:
  _insert(store, 'ff00')
  entry = store.get_exact(SCOPE, 'ff00')
  assert entry is not None
  assert entry.response == '4'
  assert entry.query == 'what is 2+2'
  assert entry.metadata == {'tokens': 7}
  assert entry.hits == 0


def test_get_exact_misses_on_unknown_fingerprint(store: Postgres) -> None:
  _insert(store, 'ff00')
  assert store.get_exact(SCOPE, 'nope') is None


def test_get_exact_is_scoped(store: Postgres) -> None:
  _insert(store, 'ff00')
  assert store.get_exact('gpt-4o|thread:other', 'ff00') is None


def test_get_exact_skips_expired(store: Postgres) -> None:
  _insert(store, 'stale', expires='-1 hour')
  assert store.get_exact(SCOPE, 'stale') is None


def test_get_exact_returns_unexpired(store: Postgres) -> None:
  _insert(store, 'fresh', expires='1 hour')
  assert store.get_exact(SCOPE, 'fresh') is not None


def _entry(fingerprint: str, response: str = '4') -> CacheEntry:
  return CacheEntry(
    scope=SCOPE,
    fingerprint=fingerprint,
    query='what is 2+2',
    response=response,
    metadata={'tokens': 7},
  )


def test_put_roundtrips_through_get_exact(store: Postgres) -> None:
  store.put(_entry('aa11'), VECTOR)
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.response == '4'
  assert entry.metadata == {'tokens': 7}
  assert entry.created_at is not None


def test_put_twice_upserts_instead_of_raising(store: Postgres) -> None:
  store.put(_entry('aa11', response='4'), VECTOR)
  store.put(_entry('aa11', response='four'), VECTOR)
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.response == 'four'


def test_put_preserves_hit_counters_on_upsert(store: Postgres) -> None:
  store.put(_entry('aa11'), VECTOR)
  with store._pool.connection() as conn:
    conn.execute('UPDATE llm_cache SET hits = 9 WHERE fingerprint = %s', ('aa11',))
  store.put(_entry('aa11', response='four'), VECTOR)
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.hits == 9


def test_put_stores_the_vector(store: Postgres) -> None:
  store.put(_entry('aa11'), VECTOR)
  with store._pool.connection() as conn:
    stored = conn.execute(
      'SELECT embedding FROM llm_cache WHERE fingerprint = %s', ('aa11',)
    ).fetchone()
  assert stored is not None
  assert stored[0].to_list() == pytest.approx(VECTOR)


def test_touch_increments_hits(store: Postgres) -> None:
  store.put(_entry('aa11'), VECTOR)
  store.touch(SCOPE, 'aa11')
  store.touch(SCOPE, 'aa11')
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.hits == 2


def test_touch_sets_last_used_at(store: Postgres) -> None:
  store.put(_entry('aa11'), VECTOR)
  assert store.get_exact(SCOPE, 'aa11').last_used_at is None
  store.touch(SCOPE, 'aa11')
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.last_used_at is not None


def test_touch_is_scoped(store: Postgres) -> None:
  store.put(_entry('aa11'), VECTOR)
  store.touch('gpt-4o|thread:other', 'aa11')
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.hits == 0


def test_touch_on_missing_row_is_a_noop(store: Postgres) -> None:
  store.touch(SCOPE, 'nope')
