import threading
from collections.abc import Mapping

from llm_cache.exceptions import StoreError
from llm_cache.models import CacheEntry, Embedding
from llm_cache.store.base import CacheStore


class RecordingStore:
  """An in-memory CacheStore that records what was called.

  Implements the whole protocol rather than only the methods a given test
  needs, so it can be passed anywhere a CacheStore is expected without the
  type checker objecting — and so adding a method to CacheStore fails here
  instead of silently leaving the fake behind.
  """

  def __init__(self, fail_on: set[str] | None = None) -> None:
    self.calls: list[str] = []
    self.batches: list[dict[tuple[str, str], int]] = []
    self.fail_on = fail_on or set()
    self._lock = threading.Lock()

  def _record(self, name: str) -> None:
    if name in self.fail_on:
      raise StoreError(f'{name} unavailable')
    with self._lock:
      self.calls.append(name)

  def get_exact(self, scope: str, fingerprint: str) -> CacheEntry | None:
    self._record('get_exact')
    return None

  def search(
    self, scope: str, embedding: Embedding, limit: int
  ) -> list[tuple[CacheEntry, float]]:
    self._record('search')
    return []

  def put(self, entry: CacheEntry, embedding: Embedding) -> None:
    self._record('put')

  def touch(self, scope: str, fingerprint: str) -> None:
    self._record('touch')

  def touch_many(self, counts: Mapping[tuple[str, str], int]) -> None:
    self._record('touch_many')
    with self._lock:
      self.batches.append(dict(counts))

  def evict_expired(self) -> int:
    self._record('evict_expired')
    return 0

  def purge(self) -> int:
    self._record('purge')
    return 0

  def clear(self, scope: str) -> int:
    self._record('clear')
    return 0


# Fails at import time if RecordingStore drifts from the protocol.
_: CacheStore = RecordingStore()
