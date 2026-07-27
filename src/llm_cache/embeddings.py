from typing import Protocol

from llm_cache.models import Embedding


class Embedder(Protocol):
  """Turns query text into a vector.

  Structural, so anything with a matching `embed` works — an OpenAI client, a
  local sentence-transformers model, or a stub in tests.
  """

  dimensions: int

  def embed(self, text: str) -> Embedding: ...


class OpenAIEmbedder:
  def __init__(
    self,
    *,
    model: str = 'text-embedding-3-small',
    dimensions: int = 1536,
    api_key: str | None = None,
  ) -> None:
    """`dimensions` must match the vector width in the migration.

    pgvector fixes the width at the column, so changing this means a new
    migration, not just a different argument here.
    """
    from openai import OpenAI

    self._client = OpenAI(api_key=api_key)
    self._model = model
    self.dimensions = dimensions

  def embed(self, text: str) -> Embedding:
    response = self._client.embeddings.create(
      model=self._model, input=text, dimensions=self.dimensions
    )
    return response.data[0].embedding
