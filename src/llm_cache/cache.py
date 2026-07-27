import logging
from dataclasses import dataclass

from llm_cache.embeddings import Embedder
from llm_cache.exceptions import CacheError
from llm_cache.store.base import CacheEntry, CacheStore, Embedding

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GetResponse:
  entry: CacheEntry | None = None
  embedding: Embedding | None = None


class Cache:
  def __init__(
    self,
    store: CacheStore,
    embedder: Embedder,
    *,
    similarity_threshold: float = 0.95,
  ) -> None:
    self.store = store
    self.embedder = embedder
    self.similarity_threshold = similarity_threshold

  def get(self, scope: str, query: str, fingerprint: str | None = None) -> GetResponse:
    """Look the query up, degrading to a miss if the cache itself is broken.

    A cache that raises turns its own outage into the caller's outage, so a
    StoreError or EmbeddingError is logged and reported as an empty result —
    the caller then does what it would have done on any miss.
    """
    try:
      if fingerprint:
        entry = self.store.get_exact(scope=scope, fingerprint=fingerprint)
        if entry:
          return GetResponse(
            entry=entry,
          )

      embedding = self.embedder.embed(query)
      entries = self.store.search(scope=scope, limit=1, embedding=embedding)

      if entries:
        entry, similarity = entries[0]
        if similarity >= self.similarity_threshold:
          return GetResponse(entry=entry, embedding=embedding)

      return GetResponse(embedding=embedding)
    except CacheError:
      log.warning('cache lookup failed, treating as a miss', exc_info=True)
      return GetResponse()

  def set(self, entry: CacheEntry, embedding: Embedding | None = None) -> None:
    """Store the entry, or give up quietly.

    Failing to write a cache entry costs a future hit and nothing else, so it
    never propagates to the caller.
    """
    try:
      self.store.put(
        entry=entry, embedding=embedding or self.embedder.embed(entry.query)
      )
    except CacheError:
      log.warning('cache write failed, entry not stored', exc_info=True)
