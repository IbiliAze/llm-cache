import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from llm_cache.store.postgres import Postgres

ROOT = Path(__file__).resolve().parent.parent
DSN = os.getenv(
  'LLM_CACHE_DSN', 'postgresql://llmcache:llmcache@localhost:5432/llmcache'
)


@pytest.fixture(scope='session')
def migrated_dsn() -> str:
  """Bring the test database up to head, once per run.

  Tests go through the same migrations as production rather than their own
  CREATE TABLE, so a broken migration fails here instead of on deploy.
  """
  config = Config(ROOT / 'alembic.ini')
  config.set_main_option('script_location', str(ROOT / 'migrations'))
  command.upgrade(config, 'head')
  return DSN


@pytest.fixture
def store(migrated_dsn: str) -> Iterator[Postgres]:
  store = Postgres(migrated_dsn)
  with store._pool.connection() as conn:
    conn.execute('TRUNCATE llm_cache')
  yield store
  store.close()
