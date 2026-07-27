import time

import pytest

from llm_cache.buffer import TouchBuffer
from llm_cache.maintenance import Maintenance
from tests.fakes import RecordingStore

pytestmark = pytest.mark.unit


def test_run_once_runs_every_job() -> None:
  store = RecordingStore()
  Maintenance(store).run_once()
  assert store.calls == ['evict_expired', 'purge']


def test_buffer_flush_is_scheduled_when_given() -> None:
  store = RecordingStore()
  buffer = TouchBuffer(store)
  buffer.touch('s', 'fp')
  Maintenance(store, buffer=buffer).run_once()
  assert store.calls == ['touch_many', 'evict_expired', 'purge']


def test_a_failing_job_does_not_stop_the_others() -> None:
  store = RecordingStore()
  store.fail_on.add('evict_expired')
  Maintenance(store).run_once()
  assert store.calls == ['purge']


def test_jobs_run_on_their_interval() -> None:
  store = RecordingStore()
  with Maintenance(store, evict_interval=0.05, purge_interval=0.05):
    time.sleep(0.25)
  assert store.calls.count('evict_expired') >= 2
  assert store.calls.count('purge') >= 2


def test_intervals_are_independent() -> None:
  store = RecordingStore()
  with Maintenance(store, evict_interval=0.05, purge_interval=10.0):
    time.sleep(0.25)
  assert store.calls.count('evict_expired') >= 2
  assert store.calls.count('purge') == 0


def test_nothing_runs_before_its_first_interval() -> None:
  store = RecordingStore()
  with Maintenance(store, evict_interval=30.0, purge_interval=30.0):
    time.sleep(0.1)
  assert store.calls == []


def test_stop_flushes_pending_hits() -> None:
  store = RecordingStore()
  buffer = TouchBuffer(store)
  maintenance = Maintenance(
    store, buffer=buffer, flush_interval=30.0, evict_interval=30.0, purge_interval=30.0
  )
  maintenance.start()
  buffer.touch('s', 'fp')
  maintenance.stop()
  assert store.calls == ['touch_many']


def test_stop_is_prompt_even_with_long_intervals() -> None:
  store = RecordingStore()
  maintenance = Maintenance(store, evict_interval=3600.0, purge_interval=3600.0)
  maintenance.start()
  started = time.monotonic()
  maintenance.stop()
  assert time.monotonic() - started < 2.0


def test_start_is_idempotent() -> None:
  store = RecordingStore()
  maintenance = Maintenance(store, evict_interval=30.0, purge_interval=30.0)
  maintenance.start()
  maintenance.start()
  maintenance.stop()
