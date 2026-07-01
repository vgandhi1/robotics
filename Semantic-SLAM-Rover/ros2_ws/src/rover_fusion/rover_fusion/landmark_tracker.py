"""
Landmark tracker: manages the live semantic map.

Deduplicates detections of the same class that are spatially close,
merges position estimates via exponential moving average, and ages out
landmarks that have not been re-observed within a configurable window.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LandmarkEntry:
    landmark_id: str
    class_label: str
    x_map: float
    y_map: float
    confidence: float
    range_m: float
    observation_count: int
    first_seen: float    # time.time()
    last_seen: float     # time.time()
    _alpha: float = 0.3  # EMA weight for position updates (new = alpha * new + (1-alpha) * old)

    def update(self, x_new: float, y_new: float, conf: float, range_m: float) -> None:
        """Merge a new observation via exponential moving average."""
        self.x_map = self._alpha * x_new + (1 - self._alpha) * self.x_map
        self.y_map = self._alpha * y_new + (1 - self._alpha) * self.y_map
        self.confidence = max(self.confidence, conf)
        self.range_m = self._alpha * range_m + (1 - self._alpha) * self.range_m
        self.observation_count += 1
        self.last_seen = time.time()

    def distance_to(self, x: float, y: float) -> float:
        return math.sqrt((self.x_map - x) ** 2 + (self.y_map - y) ** 2)


class LandmarkTracker:
    """
    In-memory semantic landmark map with deduplication and aging.

    Thread-safety note: this class is not thread-safe. The fusion node
    runs in a single ROS 2 executor thread, so no lock is needed. If
    multiple executor threads are used, wrap public methods with a lock.
    """

    def __init__(
        self,
        merge_radius_m: float = 0.5,
        max_age_seconds: float = 30.0,
        min_confidence: float = 0.40,
    ) -> None:
        self._landmarks: Dict[str, LandmarkEntry] = {}
        self.merge_radius_m = merge_radius_m
        self.max_age_seconds = max_age_seconds
        self.min_confidence = min_confidence

    def observe(
        self,
        class_label: str,
        x_map: float,
        y_map: float,
        confidence: float,
        range_m: float,
    ) -> Optional[LandmarkEntry]:
        """
        Record a new observation. Merges into an existing landmark if
        one of the same class is within merge_radius_m; otherwise creates
        a new landmark. Returns the updated/created landmark or None if
        confidence is below threshold.
        """
        if confidence < self.min_confidence:
            return None

        existing = self._find_nearby(class_label, x_map, y_map)
        if existing is not None:
            existing.update(x_map, y_map, confidence, range_m)
            return existing

        lm = LandmarkEntry(
            landmark_id=str(uuid.uuid4())[:8],
            class_label=class_label,
            x_map=x_map,
            y_map=y_map,
            confidence=confidence,
            range_m=range_m,
            observation_count=1,
            first_seen=time.time(),
            last_seen=time.time(),
        )
        self._landmarks[lm.landmark_id] = lm
        return lm

    def _find_nearby(
        self, class_label: str, x: float, y: float
    ) -> Optional[LandmarkEntry]:
        """Return the nearest landmark of the same class within merge_radius_m."""
        best: Optional[LandmarkEntry] = None
        best_dist = float("inf")
        for lm in self._landmarks.values():
            if lm.class_label != class_label:
                continue
            d = lm.distance_to(x, y)
            if d < self.merge_radius_m and d < best_dist:
                best = lm
                best_dist = d
        return best

    def prune_stale(self) -> int:
        """Remove landmarks not seen within max_age_seconds. Returns count removed."""
        now = time.time()
        stale = [
            lid for lid, lm in self._landmarks.items()
            if (now - lm.last_seen) > self.max_age_seconds
        ]
        for lid in stale:
            del self._landmarks[lid]
        return len(stale)

    def get_all(self) -> List[LandmarkEntry]:
        return list(self._landmarks.values())

    def get_by_class(
        self, class_label: str, max_age_seconds: Optional[float] = None
    ) -> List[LandmarkEntry]:
        now = time.time()
        age_limit = max_age_seconds if max_age_seconds is not None else self.max_age_seconds
        return [
            lm for lm in self._landmarks.values()
            if lm.class_label == class_label
            and (now - lm.last_seen) <= age_limit
        ]

    def get_best_by_class(
        self, class_label: str, max_age_seconds: Optional[float] = None
    ) -> Optional[LandmarkEntry]:
        """Return the most-observed landmark of a given class."""
        candidates = self.get_by_class(class_label, max_age_seconds)
        if not candidates:
            return None
        return max(candidates, key=lambda lm: lm.observation_count)

    def __len__(self) -> int:
        return len(self._landmarks)
