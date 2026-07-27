from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

type Embedding = Sequence[float]


@dataclass(slots=True)
class CacheEntry:
  scope: str
  fingerprint: str
  query: str
  response: str
  created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
  expires_at: datetime | None = None
  hits: int = 0
  last_used_at: datetime | None = None
  metadata: dict[str, Any] = field(default_factory=dict[str, Any])
