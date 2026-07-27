from typing import Annotated

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _blank_is_unset(value: object) -> object:
  """Treat `VAR=` as absent.

  Environment variables cannot hold null, so an operator turning a setting off
  leaves it empty. Without this, `LLM_CACHE_TTL_SECONDS=` — exactly what
  .env.example ships — fails to parse as an int and the process won't start.
  """
  return None if value == '' else value


# The `gt` sits on the int arm, not the union: applied to the whole thing it
# would also be checked against None, which raises a TypeError.
type OptionalSeconds = Annotated[
  Annotated[int, Field(gt=0)] | None, BeforeValidator(_blank_is_unset)
]


class Settings(BaseSettings):
  """Environment-backed configuration, read once at the application edge.

  Nothing inside the package constructs this — `Cache`, `Postgres` and the
  embedders all take explicit arguments, so a library user is never subject to
  whatever happens to be in the process environment. Applications build one of
  these at startup and pass the values down.
  """

  model_config = SettingsConfigDict(
    env_prefix='LLM_CACHE_',
    env_file='.env',
    env_file_encoding='utf-8',
    extra='ignore',
  )

  dsn: str = 'postgresql://llmcache:llmcache@localhost:5432/llmcache'
  table: str = 'llm_cache'
  namespace: str = 'default'

  # Must match vector(n) in the migration; pgvector fixes the width at the
  # column, so a change here without a migration fails at the first write.
  embedding_dimensions: int = Field(default=1536, gt=0)

  # Cosine similarity, so bounded at 1.0. Below roughly 0.9 unrelated questions
  # start matching, which is worse than missing.
  similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)

  search_limit: int = Field(default=5, gt=0)

  # Blank in .env means no expiry rather than zero seconds.
  ttl_seconds: OptionalSeconds = None
