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


class OrientationTests(unittest.TestCase):
  """yaw extraction and drift must survive roll/pitch and the +-pi wrap."""

  @staticmethod
  def _yaw_quat(yaw, roll=0.0, pitch=0.0):
    """ZYX euler -> pybullet-order quaternion [x, y, z, w]."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return [x, y, z, w]

  def test_yaw_of_pure_yaw_rotation(self):
    for yaw in (-2.0, -0.3, 0.0, 0.4, 2.9):
      self.assertAlmostEqual(
          metrics.quat_to_yaw(self._yaw_quat(yaw)), yaw, places=9)

  def test_yaw_ignores_roll_and_pitch(self):
    # the old upstream formula 2*atan2(qz, qw) drifts by ~1e-3 rad here
    for roll in (-0.1, 0.0, 0.1):
      for pitch in (-0.08, 0.0, 0.08):
        got = metrics.quat_to_yaw(self._yaw_quat(0.7, roll, pitch))
        self.assertAlmostEqual(got, 0.7, places=9)

  def test_wrap_angle_range(self):
    self.assertAlmostEqual(metrics.wrap_angle(3.5), 3.5 - 2 * np.pi)
    self.assertAlmostEqual(metrics.wrap_angle(-3.5), -3.5 + 2 * np.pi)
    self.assertAlmostEqual(metrics.wrap_angle(0.2), 0.2)

  def test_unwrap_removes_two_pi_jumps(self):
    raw = [0.0, 1.5, 3.0, 4.5 - 2 * np.pi, 6.0 - 2 * np.pi]  # continues past pi
    out = metrics.unwrap_angles(raw)
    self.assertAlmostEqual(out[0], 0.0)
    for i in range(1, len(raw)):
      self.assertAlmostEqual(out[i] - out[i - 1], 1.5, places=6)

  def test_drift_on_unwrapped_error_is_monotone(self):
    dt = 0.033
    true_rate = 0.5  # rad/s, large enough to cross +-pi several times
    truth = true_rate * dt * np.arange(400)
    wrapped = np.array([metrics.wrap_angle(v) for v in truth])
    # fitting a wrapped (folded) series misses the accumulation entirely
    self.assertGreater(abs(metrics.drift_rate(wrapped, dt) - true_rate), 0.2)
    self.assertAlmostEqual(
        metrics.drift_rate(metrics.unwrap_angles(wrapped), dt), true_rate,
        places=6)


class VelocityRewardTests(unittest.TestCase):
  """The v2 exponent is an averaging + rescaling, not a no-op averaging."""

  def test_v1_uses_sum(self):
    sum_sq = 57.5
    # n_joints=1 reproduces upstream exp(-0.1 * sum) exactly
    self.assertAlmostEqual(metrics.velocity_reward(sum_sq, 1, 0.1),
                           float(np.exp(-0.1 * sum_sq)), places=12)

  def test_averaging_alone_is_an_identity(self):
    # exp(-0.1*sum) == exp(-1.2*mean) for 12 joints: the scale must change too
    sum_sq = 57.5
    self.assertAlmostEqual(metrics.velocity_reward(sum_sq, 12, 1.2),
                           metrics.velocity_reward(sum_sq, 1, 0.1), places=12)

  def test_v2_scale_gives_a_usable_signal(self):
    # measured error p10/p90 (sum 48.9 .. 146.0) map to v2 term 0.361 / 0.048,
    # which brackets the measured v2 distribution (p90 0.361, p10 0.048)
    self.assertAlmostEqual(metrics.velocity_reward(48.9, 12, 0.25), 0.361,
                           places=2)
    self.assertAlmostEqual(metrics.velocity_reward(146.0, 12, 0.25), 0.048,
                           places=2)

  def test_exp_of_mean_is_below_mean_of_exp(self):
    # IMPROVEMENTS.md: plugging the mean error in gives 1.0e-4, while the
    # realised mean over 5400 steps is 2.6e-3 (Jensen pushes it 26x higher)
    self.assertAlmostEqual(metrics.velocity_reward(92.1, 1, 0.1),
                           1.0e-4, places=6)


class TerminationTests(unittest.TestCase):

  def test_time_limit(self):
    self.assertEqual(metrics.termination_reason(True, 600, 600), "time_limit")

  def test_early_termination(self):
    self.assertEqual(metrics.termination_reason(True, 32, 600),
                     "early_termination")

  def test_unknown(self):
    self.assertEqual(metrics.termination_reason(False, 600, 600), "time_limit")


class PhaseSamplingTests(unittest.TestCase):
  """Start phases are reproducible per (seed, episode) and stay in [0, 1)."""

  def test_reproducible_and_in_range(self):
    for seed in (0, 1, 2):
      for k in range(3):
        p1 = float(np.random.RandomState(1000 * seed + k).uniform())
        p2 = float(np.random.RandomState(1000 * seed + k).uniform())
        self.assertEqual(p1, p2)
        self.assertGreaterEqual(p1, 0.0)
        self.assertLess(p1, 1.0)

  def test_different_pairs_differ(self):
    phases = {float(np.random.RandomState(1000 * s + k).uniform())
              for s in range(3) for k in range(3)}
    self.assertEqual(len(phases), 9)


class SeedAggregationTests(unittest.TestCase):

  def test_two_level_aggregation(self):
    episodes = [
        {"seed": 0, "mean_reward": 0.60},
        {"seed": 0, "mean_reward": 0.80},  # seed 0 mean 0.70
        {"seed": 1, "mean_reward": 0.90},
        {"seed": 1, "mean_reward": 0.90},  # seed 1 mean 0.90
    ]
    out = metrics.aggregate_by_seed(episodes, keys=["mean_reward"])
    self.assertEqual(out["num_seeds"], 2)
    self.assertAlmostEqual(out["mean_reward_mean_across_seeds"], 0.80)
    # std across seeds, not across the four pooled episodes
    self.assertAlmostEqual(out["mean_reward_std_across_seeds"],
                           0.1414213562373095)


if __name__ == "__main__":
  unittest.main()
