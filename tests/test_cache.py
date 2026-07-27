import math
import random

import pytest

from llm_cache.cache import Cache
from llm_cache.models import CacheEntry
from llm_cache.store.postgres import Postgres

pytestmark = pytest.mark.integration

SCOPE = 'gpt-4o|thread:abc'


def _unit(values: list[float]) -> list[float]:
  norm = math.sqrt(sum(v * v for v in values))
  return [v / norm for v in values]


class StubEmbedder:
  """Maps known text to fixed vectors, so similarity is deterministic."""

  dimensions = 1536

  def __init__(self) -> None:
    rng = random.Random(0)
    base = _unit([rng.random() for _ in range(1536)])
    self._vectors = {
      'what is 2+2': base,
      'whats 2 + 2': _unit([v + rng.gauss(0, 0.003) for v in base]),
      'how do i deploy postgres': _unit([rng.random() for _ in range(1536)]),
    }

  def embed(self, text: str) -> list[float]:
    return self._vectors[text]


@pytest.fixture
def cache(store: Postgres) -> Cache:
  return Cache(store=store, embedder=StubEmbedder())


def _entry(query: str, response: str) -> CacheEntry:
  return CacheEntry(scope=SCOPE, fingerprint=query, query=query, response=response)


def test_exact_hit_skips_the_embedder(cache: Cache) -> None:
  cache.set(entry=_entry('what is 2+2', '4'))
  response = cache.get(SCOPE, 'what is 2+2', 'what is 2+2')
  assert response is not None
  assert response.entry is not None
  assert response.entry.response == '4'


def test_similar_query_is_a_semantic_hit(cache: Cache) -> None:
  cache.set(_entry('what is 2+2', '4'))
  response = cache.get(SCOPE, 'whats 2 + 2', None)
  assert response is not None
  assert response.entry is not None
  assert response.entry.response == '4'


def test_unrelated_query_is_a_miss(cache: Cache) -> None:
  cache.set(_entry('what is 2+2', '4'))
  response = cache.get(SCOPE, 'how do i deploy postgres', None)
  assert response.entry is None


def test_threshold_of_one_rejects_near_matches(store: Postgres) -> None:
  cache = Cache(store=store, embedder=StubEmbedder(), similarity_threshold=1.0)
  cache.set(_entry('what is 2+2', '4'))
  response = cache.get(SCOPE, 'whats 2 + 2', None)
  assert response.entry is None


def test_low_threshold_accepts_anything(store: Postgres) -> None:
  cache = Cache(store=store, embedder=StubEmbedder(), similarity_threshold=0.0)
  cache.set(_entry('what is 2+2', '4'))
  response = cache.get(SCOPE, 'how do i deploy postgres', None)
  assert response is not None
  assert response.entry is not None


def test_empty_cache_is_a_miss(cache: Cache) -> None:
  response = cache.get(SCOPE, 'what is 2+2', 'what is 2+2')
  assert response.entry is None
