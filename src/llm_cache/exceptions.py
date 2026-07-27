class CacheError(Exception):
  """Base for failures the cache can survive.

  Catching this means "the cache didn't work, carry on without it". Bugs — a
  missing table, mismatched vector dimensions — deliberately do not inherit
  from it, so they keep crashing loudly instead of turning into silent misses.
  """


class StoreError(CacheError):
  """The backing store was unreachable or refused the query."""


class EmbeddingError(CacheError):
  """The embedding provider failed: timeout, rate limit, bad response."""
