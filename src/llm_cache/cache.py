from dataclasses import dataclass

from llm_cache.embeddings import Embedder
from llm_cache.store.base import CacheEntry, CacheStore, Embedding


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

  def set(self, entry: CacheEntry, embedding: Embedding | None = None) -> None:
    self.store.put(entry=entry, embedding=embedding or self.embedder.embed(entry.query))
