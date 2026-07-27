import math
import random
from datetime import datetime, timedelta, timezone

import psycopg
import pytest

from llm_cache.models import CacheEntry
from llm_cache.store.postgres import Postgres

pytestmark = pytest.mark.integration

type Vectors = tuple[list[float], list[float], list[float]]

SCOPE = 'gpt-4o|thread:abc'
VECTOR: list[float] = [0.01] * 1536


def _insert(
  db: psycopg.Connection, fingerprint: str, expires: str | None = None
) -> None:
  """Seed a row with raw SQL, bypassing put().

  `expires` is a Postgres interval like '-1 hour', or None for no expiry —
  `now() + NULL::interval` is NULL, so the two cases need no branching.
  """
  db.execute(
    'INSERT INTO llm_cache'
    ' (scope, fingerprint, query, response, embedding, expires_at, metadata)'
    ' VALUES (%s, %s, %s, %s, %s, now() + %s::interval, %s)',
    (SCOPE, fingerprint, 'what is 2+2', '4', VECTOR, expires, '{"tokens": 7}'),
  )


def test_get_exact_returns_entry(store: Postgres, db: psycopg.Connection) -> None:
  _insert(db, 'ff00')
  entry = store.get_exact(SCOPE, 'ff00')
  assert entry is not None
  assert entry.response == '4'
  assert entry.query == 'what is 2+2'
  assert entry.metadata == {'tokens': 7}
  assert entry.hits == 0


def test_get_exact_misses_on_unknown_fingerprint(
  store: Postgres, db: psycopg.Connection
) -> None:
  _insert(db, 'ff00')
  assert store.get_exact(SCOPE, 'nope') is None


def test_get_exact_is_scoped(store: Postgres, db: psycopg.Connection) -> None:
  _insert(db, 'ff00')
  assert store.get_exact('gpt-4o|thread:other', 'ff00') is None


def test_get_exact_skips_expired(store: Postgres, db: psycopg.Connection) -> None:
  _insert(db, 'stale', expires='-1 hour')
  assert store.get_exact(SCOPE, 'stale') is None


def test_get_exact_returns_unexpired(store: Postgres, db: psycopg.Connection) -> None:
  _insert(db, 'fresh', expires='1 hour')
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


def test_put_preserves_hit_counters_on_upsert(
  store: Postgres, db: psycopg.Connection
) -> None:
  store.put(_entry('aa11'), VECTOR)
  db.execute('UPDATE llm_cache SET hits = 9 WHERE fingerprint = %s', ('aa11',))
  store.put(_entry('aa11', response='four'), VECTOR)
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.hits == 9


def test_put_stores_the_vector(store: Postgres, db: psycopg.Connection) -> None:
  store.put(_entry('aa11'), VECTOR)
  stored = db.execute(
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
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.last_used_at is None
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


def _unit(values: list[float]) -> list[float]:
  norm = math.sqrt(sum(v * v for v in values))
  return [v / norm for v in values]


@pytest.fixture
def vectors() -> Vectors:
  """A base vector, one almost identical to it, and one unrelated."""
  rng = random.Random(0)
  base = _unit([rng.random() for _ in range(1536)])
  near = _unit([v + rng.gauss(0, 0.005) for v in base])
  far = _unit([rng.random() for _ in range(1536)])
  return base, near, far


def test_search_orders_by_similarity(store: Postgres, vectors: Vectors) -> None:
  base, near, far = vectors
  store.put(_entry('identical'), base)
  store.put(_entry('similar'), near)
  store.put(_entry('unrelated'), far)
  results = store.search(SCOPE, base, 5)
  assert [e.fingerprint for e, _ in results] == ['identical', 'similar', 'unrelated']
  assert results[0][1] == pytest.approx(1.0, abs=1e-6)
  assert results[0][1] > results[1][1] > results[2][1]


def test_search_respects_limit(store: Postgres, vectors: Vectors) -> None:
  base, near, far = vectors
  store.put(_entry('identical'), base)
  store.put(_entry('similar'), near)
  store.put(_entry('unrelated'), far)
  assert len(store.search(SCOPE, base, 1)) == 1


def test_search_is_scoped(store: Postgres, vectors: Vectors) -> None:
  base, _, _ = vectors
  other = _entry('elsewhere')
  other.scope = 'gpt-4o|thread:other'
  store.put(other, base)
  assert store.search(SCOPE, base, 5) == []


def test_search_skips_expired(store: Postgres, vectors: Vectors) -> None:
  base, near, _ = vectors
  stale = _entry('stale')
  stale.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
  store.put(stale, base)
  store.put(_entry('live'), near)
  assert [e.fingerprint for e, _ in store.search(SCOPE, base, 5)] == ['live']


def test_search_returns_full_entries(store: Postgres, vectors: Vectors) -> None:
  base, _, _ = vectors
  store.put(_entry('aa11'), base)
  entry, score = store.search(SCOPE, base, 1)[0]
  assert entry.response == '4'
  assert entry.metadata == {'tokens': 7}
  assert isinstance(score, float)


def test_search_on_empty_scope_returns_empty_list(
  store: Postgres, vectors: Vectors
) -> None:
  base, _, _ = vectors
  assert store.search(SCOPE, base, 5) == []


def test_touch_many_applies_deltas_in_one_call(store: Postgres) -> None:
  store.put(_entry('aa11'), VECTOR)
  store.put(_entry('bb22'), VECTOR)
  store.touch_many({(SCOPE, 'aa11'): 5, (SCOPE, 'bb22'): 2})
  first = store.get_exact(SCOPE, 'aa11')
  second = store.get_exact(SCOPE, 'bb22')
  assert first is not None and second is not None
  assert first.hits == 5
  assert second.hits == 2


def test_touch_many_adds_rather_than_assigns(store: Postgres) -> None:
  store.put(_entry('aa11'), VECTOR)
  store.touch_many({(SCOPE, 'aa11'): 3})
  store.touch_many({(SCOPE, 'aa11'): 4})
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.hits == 7


def test_touch_many_sets_last_used_at(store: Postgres) -> None:
  store.put(_entry('aa11'), VECTOR)
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.last_used_at is None
  store.touch_many({(SCOPE, 'aa11'): 1})
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.last_used_at is not None


def test_touch_many_is_scoped(store: Postgres) -> None:
  store.put(_entry('aa11'), VECTOR)
  store.touch_many({('gpt-4o|thread:other', 'aa11'): 9})
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.hits == 0


def test_touch_many_ignores_unknown_rows(store: Postgres) -> None:
  store.put(_entry('aa11'), VECTOR)
  store.touch_many({(SCOPE, 'aa11'): 1, (SCOPE, 'ghost'): 4})
  entry = store.get_exact(SCOPE, 'aa11')
  assert entry is not None
  assert entry.hits == 1


def test_touch_many_with_empty_mapping_is_a_noop(store: Postgres) -> None:
  store.touch_many({})
