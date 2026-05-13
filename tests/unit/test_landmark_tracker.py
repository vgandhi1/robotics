"""
Unit tests for LandmarkTracker (landmark deduplication and aging).

Run with: pytest tests/unit/test_landmark_tracker.py -v
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "ros2_ws" / "src" / "rover_fusion"))

from rover_fusion.landmark_tracker import LandmarkEntry, LandmarkTracker


class TestLandmarkTracker:
    def setup_method(self):
        self.tracker = LandmarkTracker(
            merge_radius_m=0.5,
            max_age_seconds=1.0,
            min_confidence=0.40,
        )

    def test_first_observation_creates_landmark(self):
        lm = self.tracker.observe("person", 1.0, 0.0, 0.8, 1.5)
        assert lm is not None
        assert lm.class_label == "person"
        assert len(self.tracker) == 1

    def test_nearby_same_class_merges(self):
        self.tracker.observe("person", 1.0, 0.0, 0.8, 1.5)
        self.tracker.observe("person", 1.1, 0.0, 0.8, 1.4)  # 10 cm away
        assert len(self.tracker) == 1

    def test_far_away_same_class_creates_new(self):
        self.tracker.observe("person", 1.0, 0.0, 0.8, 1.5)
        self.tracker.observe("person", 5.0, 0.0, 0.8, 5.0)  # 4 m away
        assert len(self.tracker) == 2

    def test_different_classes_do_not_merge(self):
        self.tracker.observe("person", 1.0, 0.0, 0.8, 1.5)
        self.tracker.observe("chair", 1.0, 0.0, 0.8, 1.5)  # same position, different class
        assert len(self.tracker) == 2

    def test_low_confidence_rejected(self):
        lm = self.tracker.observe("person", 1.0, 0.0, 0.20, 1.5)
        assert lm is None
        assert len(self.tracker) == 0

    def test_observation_count_increments(self):
        lm = self.tracker.observe("bottle", 2.0, 1.0, 0.7, 2.2)
        assert lm.observation_count == 1
        lm2 = self.tracker.observe("bottle", 2.0, 1.0, 0.7, 2.2)
        assert lm2.observation_count == 2

    def test_position_ema_update(self):
        lm = self.tracker.observe("box", 0.0, 0.0, 0.9, 1.0)
        assert lm.x_map == pytest.approx(0.0)
        # Second observation at 0.3 m (within merge_radius=0.5 m)
        lm2 = self.tracker.observe("box", 0.3, 0.0, 0.9, 1.0)  # merges (0.3 m < 0.5 m)
        # EMA: alpha=0.3 → new_x = 0.3*0.3 + 0.7*0.0 = 0.09
        assert abs(lm2.x_map - 0.09) < 0.01

    def test_stale_prune(self):
        self.tracker.observe("cup", 1.0, 0.0, 0.9, 1.0)
        assert len(self.tracker) == 1
        time.sleep(1.1)
        removed = self.tracker.prune_stale()
        assert removed == 1
        assert len(self.tracker) == 0

    def test_get_by_class(self):
        self.tracker.observe("person", 1.0, 0.0, 0.8, 1.5)
        self.tracker.observe("person", 5.0, 0.0, 0.8, 5.0)
        self.tracker.observe("chair", 2.0, 0.0, 0.8, 2.0)
        persons = self.tracker.get_by_class("person")
        assert len(persons) == 2

    def test_get_best_by_class(self):
        lm1 = self.tracker.observe("person", 1.0, 0.0, 0.8, 1.5)
        lm2 = self.tracker.observe("person", 5.0, 0.0, 0.8, 5.0)
        # Observe lm2's position 3 more times to make it higher count
        for _ in range(3):
            self.tracker.observe("person", 5.0, 0.0, 0.8, 5.0)
        best = self.tracker.get_best_by_class("person")
        assert best.observation_count == 4  # lm2 was seen 4 times

    def test_get_best_by_class_none_if_empty(self):
        best = self.tracker.get_best_by_class("nonexistent")
        assert best is None

    def test_get_all(self):
        self.tracker.observe("cat", 1.0, 0.0, 0.9, 1.0)
        self.tracker.observe("dog", 2.0, 0.0, 0.9, 2.0)
        assert len(self.tracker.get_all()) == 2


class TestLandmarkEntry:
    def test_distance_to(self):
        lm = LandmarkEntry(
            landmark_id="abc", class_label="box",
            x_map=0.0, y_map=0.0, confidence=0.9, range_m=1.0,
            observation_count=1, first_seen=0.0, last_seen=0.0,
        )
        assert lm.distance_to(3.0, 4.0) == pytest.approx(5.0)

    def test_update_sets_last_seen(self):
        lm = LandmarkEntry(
            landmark_id="abc", class_label="box",
            x_map=0.0, y_map=0.0, confidence=0.9, range_m=1.0,
            observation_count=1, first_seen=0.0, last_seen=0.0,
        )
        before = lm.last_seen
        time.sleep(0.05)
        lm.update(1.0, 0.0, 0.8, 1.0)
        assert lm.last_seen > before
