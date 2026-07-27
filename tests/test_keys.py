import pytest

from llm_cache.keys import fingerprint

pytestmark = pytest.mark.unit


def test_is_deterministic() -> None:
  assert fingerprint('what is 2+2') == fingerprint('what is 2+2')


def test_looks_like_a_sha256_digest() -> None:
  digest = fingerprint('what is 2+2')
  assert len(digest) == 64
  assert set(digest) <= set('0123456789abcdef')


@pytest.mark.parametrize(
  'variant',
  [
    'what is 2+2',
    'WHAT IS 2+2',
    '  what is 2+2  ',
    'what  is   2+2',
    'what is\t2+2',
    'what is\n2+2',
  ],
)
def test_normalization_collapses_case_and_whitespace(variant: str) -> None:
  assert fingerprint(variant) == fingerprint('what is 2+2')


def test_different_queries_differ() -> None:
  assert fingerprint('what is 2+2') != fingerprint('what is 3+3')


def test_punctuation_is_significant() -> None:
  """Normalization is whitespace and case only — '?' changes the meaning."""
  assert fingerprint('what is 2+2') != fingerprint('what is 2+2?')


def test_params_change_the_fingerprint() -> None:
  assert fingerprint('q', temperature=0.0) != fingerprint('q', temperature=1.0)


def test_params_are_order_independent() -> None:
  assert fingerprint('q', temperature=0.0, top_p=1.0) == fingerprint(
    'q', top_p=1.0, temperature=0.0
  )


def test_absent_param_differs_from_present_one() -> None:
  assert fingerprint('q') != fingerprint('q', temperature=0.0)


def test_param_types_are_distinguished() -> None:
  """0 and '0' must not collide, or a stringified param reuses a cached answer."""
  assert fingerprint('q', temperature=0) != fingerprint('q', temperature='0')


def test_empty_query_is_stable() -> None:
  assert fingerprint('') == fingerprint('   ')
