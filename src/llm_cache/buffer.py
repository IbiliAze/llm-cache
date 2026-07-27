import logging
import threading
from collections import Counter
from types import TracebackType

from llm_cache.exceptions import CacheError
from llm_cache.store.base import CacheStore

log = logging.getLogger(__name__)


class TouchBuffer:
  """Accumulates hit counts in memory and writes them out on a timer.

  Nothing on the read path consults `hits` or `last_used_at`, so recording a hit
  should never sit between the caller and their response. Counting in memory
  also collapses repeat hits on the same entry into a single `hits + n`, which
  matters because each row update Postgres cannot do in place costs an entry in
  the HNSW index.

  The trade is durability: up to one interval of counts is lost if the process
  dies. These are cache statistics, so nothing reconciles against them.
  """

  def __init__(self, store: CacheStore, *, interval: float = 5.0) -> None:
    self._store = store
    self._interval = interval
    self._counts: Counter[tuple[str, str]] = Counter()
    self._lock = threading.Lock()
    # Doubles as the sleep and the shutdown signal, so stop() interrupts the
    # wait instead of leaving the caller to sit out a full interval.
    self._stopping = threading.Event()
    self._thread: threading.Thread | None = None

  def touch(self, scope: str, fingerprint: str) -> None:
    """Record a hit. Never does I/O."""
    with self._lock:
      self._counts[(scope, fingerprint)] += 1

  def flush(self) -> None:
    """Write pending counts. Safe to call directly, e.g. in tests."""
    with self._lock:
      if not self._counts:
        return
      batch = self._counts
      self._counts = Counter()
    try:
      self._store.touch_many(batch)
    except CacheError:
      # Deliberately dropped rather than merged back: a store that stays down
      # would otherwise grow this buffer without bound, and losing hit counts
      # costs nothing but eviction accuracy.
      log.warning('touch flush failed, %d counts dropped', len(batch), exc_info=True)

  def start(self) -> None:
    if self._thread is not None:
      return
    self._stopping.clear()
    self._thread = threading.Thread(
      target=self._run, name='llm-cache-touch-buffer', daemon=True
    )
    self._thread.start()

  def stop(self) -> None:
    """Stop the timer and write out whatever is pending."""
    self._stopping.set()
    if self._thread is not None:
      self._thread.join(timeout=self._interval + 5.0)
      self._thread = None
    self.flush()

  def _run(self) -> None:
    while not self._stopping.wait(self._interval):
      try:
        self.flush()
      except Exception:
        # A flush that raises must not kill the thread, or every later hit is
        # silently uncounted with nothing in the logs to say why.
        log.exception('touch buffer flush raised, continuing')

  def __enter__(self) -> 'TouchBuffer':
    self.start()
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    tb: TracebackType | None,
  ) -> None:
    self.stop()
