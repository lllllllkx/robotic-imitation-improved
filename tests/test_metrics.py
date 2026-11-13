"""Pure-python checks that run without tensorflow / pybullet / a GPU."""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import env_patch  # noqa: E402
from src import metrics  # noqa: E402


class WeightTests(unittest.TestCase):

  def test_weights_sum_to_one(self):
    self.assertAlmostEqual(sum(env_patch.REWARD_WEIGHTS_V1.values()), 1.0)
    self.assertAlmostEqual(sum(env_patch.REWARD_WEIGHTS_V2.values()), 1.0)

  def test_normalized_return_bounds(self):
    # perfect tracking -> 1.0, total mismatch -> 0.0
    perfect = {k: 1.0 for k in metrics.TERM_NAMES}
    self.assertAlmostEqual(
        metrics.normalized_return(perfect, env_patch.REWARD_WEIGHTS_V1), 1.0)
    worst = {k: 0.0 for k in metrics.TERM_NAMES}
    self.assertAlmostEqual(
        metrics.normalized_return(worst, env_patch.REWARD_WEIGHTS_V1), 0.0)

  def test_normalized_return_matches_weighted_sum(self):
    terms = {"pose": 0.6, "velocity": 0.01, "end_effector": 0.5,
             "root_pose": 0.9, "root_velocity": 0.8}
    w = env_patch.REWARD_WEIGHTS_V1
    expected = (0.5 * 0.6 + 0.05 * 0.01 + 0.2 * 0.5 + 0.15 * 0.9
                + 0.1 * 0.8)
    self.assertAlmostEqual(metrics.normalized_return(terms, w), expected)

  def test_bad_weights_rejected(self):
    with self.assertRaises(ValueError):
      metrics.normalized_return({k: 1.0 for k in metrics.TERM_NAMES},
                                {k: 0.1 for k in metrics.TERM_NAMES})


class MetricHelperTests(unittest.TestCase):

  def test_window_mean_uses_trailing_window(self):
    values = list(np.linspace(0.0, 1.0, 100))
    out = metrics.window_mean(values, dt=0.033, window_s=1.0)
    n = int(round(1.0 / 0.033))
    self.assertAlmostEqual(out, float(np.mean(values[-n:])))

  def test_drift_rate_recovers_slope(self):
    dt = 0.033
    values = 0.05 * np.arange(100) * dt
    self.assertAlmostEqual(metrics.drift_rate(values, dt), 0.05, places=6)

  def test_summarize_episodes(self):
    episodes = [{"mean_reward": 0.5, "steps": 600, "survived": True},
                {"mean_reward": 0.7, "steps": 600, "survived": True}]
    agg = metrics.summarize_episodes(episodes)
    self.assertAlmostEqual(agg["mean_reward_mean"], 0.6)
    self.assertAlmostEqual(agg["mean_reward_std"], 0.1414213562373095)
    self.assertEqual(agg["num_episodes"], 2)


class PatchContractTests(unittest.TestCase):

  class _FakeTask(object):

    def __init__(self):
      self._calc_reward_pose = lambda: 1.0
      self._pose_weight = env_patch.REWARD_WEIGHTS_V1["pose"]
      self._velocity_weight = env_patch.REWARD_WEIGHTS_V1["velocity"]

  class _FakeEnv(object):

    def __init__(self):
      self._task = PatchContractTests._FakeTask()

  def test_reward_v2_sets_weights(self):
    env = self._FakeEnv()
    task = env_patch.apply_velocity_reward_v2(env)
    self.assertAlmostEqual(task._pose_weight, 0.45)
    self.assertAlmostEqual(task._velocity_weight, 0.10)
    self.assertTrue(getattr(task, "_velocity_reward_v2", False))


if __name__ == "__main__":
  unittest.main()
