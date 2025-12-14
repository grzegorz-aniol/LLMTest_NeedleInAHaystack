from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class EvaluationCache:
    """In-memory cache for evaluation scores.

    The cache key is a strict triplet of strings:
    ``(prediction, reference, input)``.

    This ensures that changing *any* of these values results in a
    different cache entry.
    """

    _store: Dict[Tuple[str, str, str], int] = field(default_factory=dict)

    def get(self, prediction: str, reference: str, input_: str) -> Optional[int]:
        """Return a cached score for the given key, or ``None`` if missing."""
        return self._store.get((prediction, reference, input_))

    def set(self, prediction: str, reference: str, input_: str, score: int) -> None:
        """Store a score for the given (prediction, reference, input_) key."""
        self._store[(prediction, reference, input_)] = score
