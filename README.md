# llm-cache

Semantic cache for LLM responses, backed by Postgres and pgvector.

Lookup is two-tier. An exact hash match on the canonicalised request comes
first — it costs one indexed lookup and skips the embedding call entirely.
Only on a miss does it embed the query and search by vector similarity, and a
neighbour is returned only if it clears a configurable threshold.

That ordering matters: the exact path measures ~0.5ms, while embedding a query
is 50–150ms. Repeat questions never pay for a model call.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker, for the Postgres in `docker-compose.yml` (image `pgvector/pgvector:pg17`)

## Setup

```sh
uv sync
docker compose up -d
cp .env.example .env
uv run alembic upgrade head
```

The last step is not optional — `Postgres` never creates tables, so it fails
against an unmigrated database.

## Usage

```python
from llm_cache.cache import Cache
from llm_cache.embeddings import OpenAIEmbedder
from llm_cache.keys import fingerprint
from llm_cache.models import CacheEntry
from llm_cache.store.postgres import Postgres

cache = Cache(
    store=Postgres('postgresql://llmcache:llmcache@localhost:5432/llmcache'),
    embedder=OpenAIEmbedder(),
    similarity_threshold=0.95,
)


def ask(question: str, thread_id: str) -> str:
    scope = f'gpt-4o|thread:{thread_id}'
    fp = fingerprint(question, temperature=0.0)

    hit = cache.get(scope, question, fp)
    if hit.entry:
        cache.store.touch(scope, hit.entry.fingerprint)
        return hit.entry.response

    answer = call_the_llm(question)
    cache.set(
        CacheEntry(scope=scope, fingerprint=fp, query=question, response=answer),
        embedding=hit.embedding,      # reuse the vector get() already computed
    )
    return answer
```

`hit.embedding` is why `get` returns a `GetResponse` rather than a bare entry.
On a miss it carries the vector that was just computed, so `set` does not embed
the same text a second time.

## Configuration

`Settings` reads the environment or `.env`, validates it, and is constructed at
your application's edge — the library classes take explicit arguments and never
read the environment themselves:

```python
from llm_cache.config import Settings

settings = Settings()
cache = Cache(
    store=Postgres(settings.dsn, table=settings.table),
    embedder=OpenAIEmbedder(dimensions=settings.embedding_dimensions),
    similarity_threshold=settings.similarity_threshold,
)
```

| Variable | Meaning |
| --- | --- |
| `LLM_CACHE_DSN` | Postgres connection string. Also used by Alembic. |
| `LLM_CACHE_TABLE` | Table name, default `llm_cache`. |
| `LLM_CACHE_NAMESPACE` | Default scope prefix. |
| `LLM_CACHE_EMBEDDING_DIMENSIONS` | Must match `vector(n)` in the migration. |
| `LLM_CACHE_SIMILARITY_THRESHOLD` | Minimum cosine similarity for a semantic hit. |
| `LLM_CACHE_SEARCH_LIMIT` | Candidates fetched before thresholding. |
| `LLM_CACHE_TTL_SECONDS` | Entry lifetime; blank means no expiry. |
| `OPENAI_API_KEY` | Only needed for `OpenAIEmbedder`. |

Changing the embedding model usually changes the vector width, and pgvector
fixes that width at the column. A different width means a new migration, not
just a different env value.

## Concepts

**scope** partitions the cache. Everything that must not share answers goes in
it — model, and usually thread or tenant:

```python
scope = f'{model}|thread:{thread_id}'
```

`get_exact`, `search`, and `clear` all filter on it, so entries cannot leak
between conversations, or from a weak model to a strong one.

**fingerprint** is the exact-match key: a SHA-256 of the normalised query plus
any parameters that should change the answer.

```python
fingerprint('What  is 2+2?', temperature=0.0)
```

Normalisation lowercases and collapses whitespace, so `'what is 2+2'` and
`'What  is\n2+2'` share a key. Parameters are sorted, so argument order does not
matter, and `repr` is used so `0` and `'0'` do not collide.

**embedding** is a lookup key, never part of a `CacheEntry`. It is passed
alongside on write and used for ordering on read, but the store never returns
it — an entry read to bump a counter shouldn't drag 6KB of floats with it.

## Architecture

```
Cache          exact → semantic lookup, threshold, graceful degradation
  ├── Embedder     protocol; OpenAIEmbedder, or bring your own
  └── CacheStore   protocol; Postgres today
```

Both dependencies are `typing.Protocol`, so a replacement needs only matching
methods — no inheritance, no registration. Swapping `OpenAIEmbedder` for a local
sentence-transformers model is a class with an `embed` method.

### Failure behaviour

A cache that raises turns its own outage into the caller's outage. `Cache`
catches `CacheError` and degrades to a miss:

| Failure | Behaviour |
| --- | --- |
| Postgres unreachable | logged, treated as a miss, caller proceeds |
| Embedding provider fails | logged, treated as a miss |
| Missing table, wrong dimensions | raises — these are bugs, not blips |

The split is deliberate. `Postgres._connection` translates only
`OperationalError` and `PoolTimeout` into `StoreError`. A `DataException` from
mismatched vector widths stays a psycopg error, because degrading it silently
would leave a cache that never hits and never complains.

### Schema

One table, primary key `(scope, fingerprint)`, owned entirely by Alembic:

- `llm_cache_embedding_idx` — HNSW, `vector_cosine_ops`. Must match the `<=>`
  operator in `search`, or Postgres drops the index and scans.
- `llm_cache_expires_at_idx` — partial, only rows that can expire.

Verified with `EXPLAIN ANALYZE` at 20k rows: the HNSW index is used, with scope
and expiry applied as a filter on top.

## Layout

- `src/llm_cache/cache.py` — `Cache`, the exact → semantic path and degradation
- `src/llm_cache/buffer.py` — `TouchBuffer`, in-memory hit counting
- `src/llm_cache/maintenance.py` — `Maintenance`, the housekeeping thread
- `src/llm_cache/config.py` — `Settings`, environment-backed configuration
- `src/llm_cache/keys.py` — query normalisation and fingerprint hashing
- `src/llm_cache/models.py` — `CacheEntry`, `Embedding`
- `src/llm_cache/embeddings.py` — `Embedder` protocol, `OpenAIEmbedder`
- `src/llm_cache/exceptions.py` — `CacheError` and friends
- `src/llm_cache/store/base.py` — the `CacheStore` protocol
- `src/llm_cache/store/postgres.py` — pgvector backend
- `migrations/` — Alembic; the only description of the schema

## Tests

```sh
uv run pytest                       # everything
uv run pytest -m unit               # pure logic, no database
uv run pytest -m "not integration"  # the same, by exclusion
```

Integration tests run against the real Postgres from `docker-compose.yml` and
apply the real migrations — a broken migration fails here rather than on deploy.
Only the embedder is stubbed, so no API key is needed.

## Migrations

```sh
uv run alembic upgrade head          # apply
uv run alembic revision -m "..."     # new migration
uv run alembic downgrade -1          # back one
uv run alembic current               # what is applied
```

Migrations are hand-written SQL. There are no SQLAlchemy models, so
`--autogenerate` has nothing to diff and is unused.

## Maintenance

Housekeeping runs on one background thread, off the request path:

```python
from llm_cache.buffer import TouchBuffer
from llm_cache.maintenance import Maintenance

buffer = TouchBuffer(store)

with Maintenance(store, buffer=buffer):
    ...                                  # your application runs here
    buffer.touch(scope, fingerprint)     # in-memory, no I/O
```

`Maintenance` flushes buffered hits every 5s and runs `evict_expired` and
`purge` every 5 minutes. Exiting the block stops the thread and writes out any
pending counts. All three intervals are constructor arguments, and `run_once()`
runs every job immediately.

**Why buffer the hits.** Nothing on the read path consults `hits` or
`last_used_at`, so recording a hit should not sit between the caller and their
response. Counting in memory also collapses repeats: 2000 hits across 50 entries
becomes one statement rather than 2000. Measured on the docker-compose Postgres:

| | request path | round trips |
| --- | --- | --- |
| `store.touch()` per hit | 633 ms | 2000 |
| `buffer.touch()` per hit | 0.4 ms | 0 |
| one background flush | — | 1 |

The trade is that up to one flush interval of counts is lost if the process
dies, and a flush that fails drops its batch rather than retrying. These are
cache statistics; nothing reconciles against them.

**Why eviction is not urgent.** `get_exact` and `search` both filter
`expires_at`, so an expired entry is already invisible. `evict_expired` only
reclaims space — which is exactly why it must not be called from `set()`, where
it would make one unlucky request pay for a table-wide DELETE.

The store methods remain callable directly if you'd rather schedule them
yourself with cron or a k8s CronJob:

```python
store.evict_expired()   # delete rows past expires_at
store.purge()           # trim to max_entries, keeping the most-used
store.clear(scope)      # drop one scope
```

`purge` ranks by `hits` and `created_at`. `last_used_at` is populated by `touch`
and would give truer LRU behaviour if recency matters more than lifetime
popularity.

## Known gaps

- `ConnectionPool` uses psycopg's default 30s timeout, so a down database stalls
  each request that long before degrading. Worth lowering to 1–2s.
- Nothing exercises the degradation path in tests; that needs a store stub that
  raises `StoreError`.
- `purge` is not part of the `CacheStore` protocol, so it is reachable only on a
  concrete `Postgres`.
- `main.py` at the repo root is leftover from another project and unrelated to
  this package.
