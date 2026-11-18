"""Unit tests for the fixed-difficulty curriculum pin.

No pybullet here: the pin only touches the wrapper's bookkeeping fields, so a
minimal stand-in reproduces the one behaviour that matters -- ``reset()``
recomputing the limit whenever the curriculum is still enabled.
"""

from __future__ import absolute_import

import unittest

from src import env_patch


class FakeWrapper(object):

  def __init__(self, start=20, end=600, curriculum_steps=3750000):
    self._episode_length_start = start
    self._episode_length_end = end
    self._curriculum_steps = curriculum_steps
    self._total_step_count = 0
    self._max_episode_steps = start

  def _enable_curriculum(self):
    return self._curriculum_steps > 0

  def _update_time_limit(self):
    t = float(self._total_step_count) / self._curriculum_steps
    t = max(0.0, min(1.0, t)) ** 3
    self._max_episode_steps = int(
        (1.0 - t) * self._episode_length_start + t * self._episode_length_end)

  def reset(self):
    self._total_step_count += 1
    if self._enable_curriculum():
      self._update_time_limit()
    return self._max_episode_steps


class PinEpisodeLengthTests(unittest.TestCase):

  def test_pin_keeps_limit_after_reset(self):
    env = FakeWrapper()
    env._total_step_count = 1250304  # same per-env count as the 1e7 probe
    env._update_time_limit()
    self.assertEqual(env._max_episode_steps, 41)

    env_patch.pin_episode_length(env, 41)
    self.assertEqual(env._max_episode_steps, 41)
    self.assertFalse(env._enable_curriculum())
    self.assertEqual(env._fixed_episode_length, 41)

    # Without the pin, the cubic schedule would immediately raise the limit.
    for _ in range(3):
      self.assertEqual(env.reset(), 41)
    plain = FakeWrapper()
    plain._total_step_count = 1800000  # later curriculum point (t = 0.48)
    plain.reset()
    self.assertGreater(plain._max_episode_steps, 41)

  def test_rejects_non_positive_length(self):
    env = FakeWrapper()
    with self.assertRaises(ValueError):
      env_patch.pin_episode_length(env, 0)
    with self.assertRaises(ValueError):
      env_patch.pin_episode_length(env, -5)


if __name__ == "__main__":
  unittest.main()
