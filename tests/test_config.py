import pytest
from pydantic import ValidationError

from llm_cache.config import Settings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
  """Ignore the developer's real .env so these assert on defaults."""
  for key in list(Settings.model_fields):
    monkeypatch.delenv(f'LLM_CACHE_{key.upper()}', raising=False)
  monkeypatch.chdir(tmp_path)


def test_defaults_apply_with_no_environment() -> None:
  settings = Settings()
  assert settings.table == 'llm_cache'
  assert settings.similarity_threshold == 0.95
  assert settings.ttl_seconds is None


def test_env_prefix_maps_to_fields(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv('LLM_CACHE_SIMILARITY_THRESHOLD', '0.80')
  monkeypatch.setenv('LLM_CACHE_TABLE', 'other_cache')
  settings = Settings()
  assert settings.similarity_threshold == 0.80
  assert settings.table == 'other_cache'


def test_values_are_coerced_to_their_type(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv('LLM_CACHE_SEARCH_LIMIT', '10')
  assert Settings().search_limit == 10
  assert isinstance(Settings().search_limit, int)


def test_blank_ttl_means_no_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
  """`LLM_CACHE_TTL_SECONDS=` in .env must not become 0."""
  monkeypatch.setenv('LLM_CACHE_TTL_SECONDS', '')
  assert Settings().ttl_seconds is None


@pytest.mark.parametrize('value', ['1.5', '-0.1'])
def test_threshold_outside_zero_to_one_is_rejected(
  monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
  monkeypatch.setenv('LLM_CACHE_SIMILARITY_THRESHOLD', value)
  with pytest.raises(ValidationError):
    Settings()


def test_unparseable_value_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv('LLM_CACHE_SIMILARITY_THRESHOLD', 'very similar')
  with pytest.raises(ValidationError):
    Settings()


def test_non_positive_dimensions_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv('LLM_CACHE_EMBEDDING_DIMENSIONS', '0')
  with pytest.raises(ValidationError):
    Settings()


def test_unknown_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv('LLM_CACHE_SOMETHING_ELSE', 'x')
  assert Settings().table == 'llm_cache'
