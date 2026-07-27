import threading
import time

import pytest

from llm_cache.buffer import TouchBuffer
from tests.fakes import RecordingStore

pytestmark = pytest.mark.unit


def test_touch_does_no_io() -> None:
  store = RecordingStore()
  buffer = TouchBuffer(store)
  for _ in range(100):
    buffer.touch('s', 'fp')
  assert store.batches == []


def test_flush_collapses_repeats_into_one_delta() -> None:
  store = RecordingStore()
  buffer = TouchBuffer(store)
  for _ in range(500):
    buffer.touch('s', 'popular')
  buffer.touch('s', 'rare')
  buffer.flush()
  assert store.batches == [{('s', 'popular'): 500, ('s', 'rare'): 1}]


def test_flush_empties_the_buffer() -> None:
  store = RecordingStore()
  buffer = TouchBuffer(store)
  buffer.touch('s', 'fp')
  buffer.flush()
  buffer.flush()
  assert store.batches == [{('s', 'fp'): 1}]


def test_flush_with_nothing_pending_is_a_noop() -> None:
  store = RecordingStore()
  TouchBuffer(store).flush()
  assert store.batches == []


def test_scopes_are_counted_separately() -> None:
  store = RecordingStore()
  buffer = TouchBuffer(store)
  buffer.touch('a', 'fp')
  buffer.touch('b', 'fp')
  buffer.flush()
  assert store.batches == [{('a', 'fp'): 1, ('b', 'fp'): 1}]


def test_store_failure_is_swallowed_and_counts_dropped() -> None:
  store = RecordingStore(fail_on={'touch_many'})
  buffer = TouchBuffer(store)
  buffer.touch('s', 'fp')
  buffer.flush()  # must not raise
  store.fail_on.clear()
  buffer.flush()
  assert store.batches == []  # dropped, not retried


def test_background_thread_flushes_on_its_interval() -> None:
  store = RecordingStore()
  with TouchBuffer(store, interval=0.05) as buffer:
    buffer.touch('s', 'fp')
    time.sleep(0.2)
  assert store.batches, 'expected at least one background flush'


def test_stop_flushes_what_is_pending() -> None:
  store = RecordingStore()
  buffer = TouchBuffer(store, interval=60.0)  # never fires on its own
  buffer.start()
  buffer.touch('s', 'fp')
  buffer.stop()
  assert store.batches == [{('s', 'fp'): 1}]


def test_concurrent_touches_are_not_lost() -> None:
  store = RecordingStore()
  buffer = TouchBuffer(store)

  def hammer() -> None:
    for _ in range(1000):
      buffer.touch('s', 'fp')

  threads = [threading.Thread(target=hammer) for _ in range(8)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  buffer.flush()
  assert store.batches == [{('s', 'fp'): 8000}]


def test_start_is_idempotent() -> None:
  store = RecordingStore()
  buffer = TouchBuffer(store, interval=60.0)
  buffer.start()
  buffer.start()
  buffer.stop()
