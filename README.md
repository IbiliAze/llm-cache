# llm-cache

Semantic cache for LLM responses, backed by Postgres and pgvector.

Lookup is two-tier: an exact hash match on the canonicalised request first,
which costs nothing and skips the embedding call, then vector similarity above
a configurable threshold.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Postgres with the `vector` extension (`docker compose up -d`)

## Setup

```sh
uv sync
docker compose up -d
cp .env.example .env
```

## Tests

```sh
uv run pytest                  # unit tests, in-memory store
uv run pytest -m integration   # needs the Postgres from docker-compose.yml
```

## Layout

- `src/llm_cache/keys.py` — request canonicalisation and scope/fingerprint hashing
- `src/llm_cache/cache.py` — `SemanticCache`, the exact → semantic lookup path
- `src/llm_cache/store/` — `CacheStore` protocol, pgvector and in-memory backends
- `src/llm_cache/embeddings.py` — `Embedder` protocol and adapters
- `src/llm_cache/integrations/` — LangChain `BaseCache` adapter, `@memoize`
