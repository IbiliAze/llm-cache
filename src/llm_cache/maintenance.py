import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType

from llm_cache.buffer import TouchBuffer
from llm_cache.store.base import CacheStore

log = logging.getLogger(__name__)


@dataclass
class _Job:
  name: str
  run: Callable[[], object]
  interval: float
  next_run: float = field(default=0.0)


class Maintenance:
  """Runs the cache's periodic housekeeping on one background thread.

  None of this belongs on the request path. `get_exact` and `search` already
  filter on `expires_at`, so an expired entry is invisible whether or not it has
  been deleted — eviction only reclaims space. Calling it from `set()` would
  make one unlucky request pay for a table-wide DELETE.

  When a TouchBuffer is passed in, its flush becomes one of these jobs and the
  buffer should not also be started on its own; one scheduler is easier to
  reason about than three threads with independent lifetimes.
  """

  def __init__(
    self,
    store: CacheStore,
    *,
    buffer: TouchBuffer | None = None,
    flush_interval: float = 5.0,
    evict_interval: float = 300.0,
    purge_interval: float = 300.0,
  ) -> None:
    self._buffer = buffer
    self._jobs: list[_Job] = []
    if buffer is not None:
      self._jobs.append(_Job('flush', buffer.flush, flush_interval))
    self._jobs.append(_Job('evict_expired', store.evict_expired, evict_interval))
    self._jobs.append(_Job('purge', store.purge, purge_interval))
    self._stopping = threading.Event()
    self._thread: threading.Thread | None = None

  def start(self) -> None:
    if self._thread is not None:
      return
    self._stopping.clear()
    now = time.monotonic()
    for job in self._jobs:
      job.next_run = now + job.interval
    self._thread = threading.Thread(
      target=self._run, name='llm-cache-maintenance', daemon=True
    )
    self._thread.start()

  def stop(self) -> None:
    """Stop the thread and flush any buffered hits, so counts are not lost."""
    self._stopping.set()
    if self._thread is not None:
      self._thread.join(timeout=30.0)
      self._thread = None
    if self._buffer is not None:
      self._buffer.flush()

  def run_once(self) -> None:
    """Run every job immediately, regardless of schedule. Useful in tests."""
    for job in self._jobs:
      self._run_job(job)

  def _run_job(self, job: _Job) -> None:
    try:
      result = job.run()
    except Exception:
      # One failing job must not stop the thread, or the others silently stop
      # running too and the table grows without anything in the logs.
      log.exception('maintenance job %r failed', job.name)
    else:
      if isinstance(result, int) and result:
        log.info('maintenance job %r affected %d rows', job.name, result)

  def _run(self) -> None:
    while not self._stopping.is_set():
      now = time.monotonic()
      for job in self._jobs:
        if now >= job.next_run:
          self._run_job(job)
          job.next_run = now + job.interval
      # Sleep only until the next job is due, and wake early if stop() is
      # called. Capped so a long interval still notices shutdown promptly.
      delay = min((job.next_run - now for job in self._jobs), default=1.0)
      self._stopping.wait(max(0.01, min(delay, 1.0)))

  def __enter__(self) -> 'Maintenance':
    self.start()
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    tb: TracebackType | None,
  ) -> None:
    self.stop()
