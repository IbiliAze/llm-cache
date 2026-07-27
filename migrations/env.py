import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context

config = context.config

if config.config_file_name is not None:
  fileConfig(config.config_file_name)

# The schema is plain SQL in the migration files, not SQLAlchemy models, so
# there is no metadata to diff and `alembic revision --autogenerate` is unused.
target_metadata = None

load_dotenv()

_DEFAULT_DSN = 'postgresql://llmcache:llmcache@localhost:5432/llmcache'


def _dsn() -> str:
  """The app's DSN, rewritten for SQLAlchemy's driver naming.

  LLM_CACHE_DSN is a plain libpq URL because that is what psycopg wants at
  runtime. SQLAlchemy reads the same prefix as "use psycopg2", which is not
  installed, so point it at psycopg 3 explicitly.
  """
  dsn = os.getenv('LLM_CACHE_DSN', _DEFAULT_DSN)
  if dsn.startswith('postgresql://'):
    return dsn.replace('postgresql://', 'postgresql+psycopg://', 1)
  if dsn.startswith('postgres://'):
    return dsn.replace('postgres://', 'postgresql+psycopg://', 1)
  return dsn


def run_migrations_offline() -> None:
  context.configure(
    url=_dsn(),
    target_metadata=target_metadata,
    literal_binds=True,
    dialect_opts={'paramstyle': 'named'},
  )
  with context.begin_transaction():
    context.run_migrations()


def run_migrations_online() -> None:
  section = config.get_section(config.config_ini_section, {})
  section['sqlalchemy.url'] = _dsn()
  connectable = engine_from_config(
    section, prefix='sqlalchemy.', poolclass=pool.NullPool
  )
  with connectable.connect() as connection:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
      context.run_migrations()


if context.is_offline_mode():
  run_migrations_offline()
else:
  run_migrations_online()
