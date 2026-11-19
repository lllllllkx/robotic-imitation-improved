"""Checks for the phasefocus sampler, without TensorFlow or PyBullet."""

import csv
import io
import os
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import phase_sampling  # noqa: E402


class SamplerTests(unittest.TestCase):

  def test_same_seed_is_reproducible(self):
    a = phase_sampling.PhaseFocusSampler(7)
    b = phase_sampling.PhaseFocusSampler(7)
    self.assertEqual([a.sample() for _ in range(20)],
                     [b.sample() for _ in range(20)])

  def test_band_probability_is_equal_not_width_weighted(self):
    s = phase_sampling.PhaseFocusSampler(
        3, bands=((0.11, 0.14), (0.41, 0.50), (0.95, 0.98)),
        focus_fraction=0.5)
    counts = {"band_0": 0, "band_1": 0, "band_2": 0, "uniform": 0}
    for _ in range(30000):
      _, source = s.sample()
      counts[source] += 1
    # Each band should get roughly 5000 draws despite widths 0.03/0.09/0.03.
    for key in ("band_0", "band_1", "band_2"):
      self.assertGreater(counts[key], 4400)
      self.assertLess(counts[key], 5600)
    self.assertGreater(counts["uniform"], 14000)
    self.assertLess(counts["uniform"], 16000)

  def test_actual_band_frequency_exceeds_focus_fraction(self):
    s = phase_sampling.PhaseFocusSampler(11, focus_fraction=0.5)
    for _ in range(20000):
      phase, source = s.sample()
      s.record(phase, source)
    stats = s.stats()
    # Uniform draws also land in the bands (total width 0.15).
    self.assertGreater(stats["in_focus_band"] / float(stats["total"]), 0.54)
    self.assertLess(stats["in_focus_band"] / float(stats["total"]), 0.61)
    self.assertEqual(stats["total"], 20000)

  def test_audit_csv_records_actual_phase_and_source(self):
    path = tempfile.mktemp(suffix=".csv")
    try:
      s = phase_sampling.PhaseFocusSampler(5, log_path=path, rank=2)
      phase, source = s.sample()
      s.record(phase, source)
      s.close()
      with io.open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
      self.assertEqual(rows[0], ["rank", "episode", "phase", "source",
                                 "focus_band", "in_focus_band", "warmup"])
      self.assertEqual(rows[1][0], "2")
      self.assertEqual(rows[1][3], source)
      self.assertAlmostEqual(float(rows[1][2]), phase, places=8)
    finally:
      if os.path.exists(path):
        os.remove(path)


class FakeMotion(object):

  def get_duration(self):
    return 1.0


class FakeTask(object):

  def __init__(self, warmup=False):
    self._curr_episode_warmup = warmup
    self._motion_time_offset = 0.0
    self.get_active_motion = lambda: FakeMotion()

  def _reset_motion_time_offset(self):
    self._motion_time_offset = 0.25


class InstallTests(unittest.TestCase):

  def test_normal_episode_is_sampled(self):
    task = FakeTask(warmup=False)
    sampler = phase_sampling.PhaseFocusSampler(1, focus_fraction=0.5)
    phase_sampling.install_phase_sampler(task, sampler)
    task._reset_motion_time_offset()
    self.assertGreaterEqual(task._motion_time_offset, 0.0)
    self.assertLess(task._motion_time_offset, 1.0)
    self.assertEqual(sampler.stats()["total"], 1)
    self.assertEqual(sampler.stats()["warmup"], 0)

  def test_warmup_episode_keeps_upstream_offset(self):
    task = FakeTask(warmup=True)
    sampler = phase_sampling.PhaseFocusSampler(1)
    phase_sampling.install_phase_sampler(task, sampler)
    task._reset_motion_time_offset()
    self.assertAlmostEqual(task._motion_time_offset, 0.25)
    self.assertEqual(sampler.stats()["warmup"], 1)
    self.assertEqual(sampler.stats()["total"], 1)


if __name__ == "__main__":
    unittest.main()
