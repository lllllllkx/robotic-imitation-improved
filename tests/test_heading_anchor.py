"""Unit tests for the heading-offset bookkeeping behind the frame comparison.

No pybullet/TensorFlow here: the tracker only needs angles, so the sign
convention, the +-pi wrap and the reconstruction identity are all testable
with synthetic offsets before any evaluation is re-run.
"""

from __future__ import absolute_import
from __future__ import division

import math
import unittest

import numpy as np

from src import heading_anchor
from src import metrics


def quat_from_yaw(yaw):
  """Quaternion for a pure rotation about +z, pybullet order [x, y, z, w]."""
  return np.array([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)])


class WrapTests(unittest.TestCase):

  def test_wrap_keeps_inside_pi(self):
    for raw, expected in ((0.0, 0.0), (0.5, 0.5), (-0.5, -0.5),
                          (math.pi - 1e-9, math.pi - 1e-9),
                          (math.pi + 0.2, -math.pi + 0.2)):
      self.assertAlmostEqual(metrics.wrap_angle(raw), expected, places=9)

  def test_quat_round_trip_matches_wrapped_angle(self):
    for yaw in (-3.0, -1.2, 0.0, 0.7, 3.1, 3.5, -3.5):
      got = heading_anchor.offset_yaw(quat_from_yaw(yaw))
      self.assertAlmostEqual(got, metrics.wrap_angle(yaw), places=9)


class TrackerSignTests(unittest.TestCase):

  def test_first_sync_defines_baseline_without_offset(self):
    t = heading_anchor.AnchorTracker()
    self.assertEqual(t.on_sync(0.3, shared=True), 0.0)
    self.assertEqual(t.cum, 0.0)
    self.assertAlmostEqual(t.theta, 0.3, places=12)

  def test_positive_cycle_offset_accumulates_positively(self):
    t = heading_anchor.AnchorTracker()
    t.on_sync(0.0, shared=True)
    self.assertAlmostEqual(t.on_sync(0.05, shared=False), 0.05, places=12)
    self.assertAlmostEqual(t.on_sync(0.12, shared=False), 0.12, places=12)
    self.assertEqual(t.cycle_events, 2)

  def test_negative_cycle_offset_accumulates_negatively(self):
    t = heading_anchor.AnchorTracker()
    t.on_sync(0.0, shared=True)
    self.assertAlmostEqual(t.on_sync(-0.08, shared=False), -0.08, places=12)
    self.assertAlmostEqual(t.cum, -0.08, places=12)

  def test_offset_crossing_pi_is_a_small_step_not_a_full_turn(self):
    # 3.10 rad -> -3.10 rad is +0.0832 rad across the branch cut, not -6.2 rad
    t = heading_anchor.AnchorTracker()
    t.on_sync(3.10, shared=True)
    cum = t.on_sync(-3.10, shared=False)
    self.assertAlmostEqual(cum, metrics.wrap_angle(-3.10 - 3.10), places=12)
    self.assertAlmostEqual(cum, 2.0 * math.pi - 6.20, places=12)
    self.assertLess(abs(cum), 0.1)

  def test_clip_change_sync_resets_the_offset(self):
    t = heading_anchor.AnchorTracker()
    t.on_sync(0.0, shared=True)
    t.on_sync(0.10, shared=False)
    t.on_sync(0.15, shared=False)
    self.assertAlmostEqual(t.cum, 0.15, places=12)
    # upstream syncs on clip change too, so both frames coincide again
    t.on_sync(0.90, shared=True)
    self.assertEqual(t.cum, 0.0)
    self.assertAlmostEqual(t.clip_sync_delta, 0.75, places=12)

  def test_zero_offset_is_idempotent(self):
    t = heading_anchor.AnchorTracker()
    t.on_sync(0.2, shared=True)
    for _ in range(5):
      t.on_sync(0.2, shared=False)
    self.assertAlmostEqual(t.cum, 0.0, places=12)
    self.assertEqual(t.cycle_events, 5)


class ReconstructionTests(unittest.TestCase):

  def test_subtracting_cum_recovers_the_unanchored_reference(self):
    """psi_ref_unanchored = psi_ref_anchored - cum must restore psi_raw + theta0."""
    rng = np.random.RandomState(7)
    theta0 = 0.4                      # offset installed at reset (frame baseline)
    psi_raw = np.cumsum(rng.uniform(-0.02, 0.02, 600))  # clip's own yaw path

    t = heading_anchor.AnchorTracker()
    t.on_sync(theta0, shared=True)
    cums = []
    for k in range(psi_raw.size):
      if k % 40 == 40 - 1:            # one re-anchor per simulated cycle
        t.on_sync(t.theta + rng.uniform(-0.06, 0.06), shared=False)
      cums.append(t.cum)
    cums = np.asarray(cums)

    ref_anchored = np.array([metrics.wrap_angle(psi_raw[k] + theta0 + cums[k])
                             for k in range(psi_raw.size)])
    # per-step cum, not the tracker's final value
    ref_unanchored = np.array([metrics.wrap_angle(ref_anchored[k] - cums[k])
                               for k in range(psi_raw.size)])
    expected = np.array([metrics.wrap_angle(psi_raw[k] + theta0)
                         for k in range(psi_raw.size)])
    np.testing.assert_allclose(ref_unanchored, expected, atol=1e-12)

  def test_error_difference_equals_cum_for_a_fixed_robot_yaw(self):
    """A robot that never rotates sees e_unanchored - e_anchored = +cum."""
    robot_yaw = 0.0
    t = heading_anchor.AnchorTracker()
    t.on_sync(0.0, shared=True)
    t.on_sync(0.30, shared=False)
    ref_anchored = metrics.wrap_angle(0.30)   # ref yaw after the re-anchor
    e_anch = metrics.wrap_angle(robot_yaw - ref_anchored)
    e_unanch = metrics.wrap_angle(
        robot_yaw - t.ref_yaw_unanchored(ref_anchored))
    self.assertAlmostEqual(e_anch, -0.30, places=12)
    self.assertAlmostEqual(e_unanch, 0.0, places=12)
    self.assertAlmostEqual(e_unanch - e_anch, t.cum, places=12)

  def test_unanchored_series_reproduces_a_never_anchored_run(self):
    """With one sync and no cycle events the two frames are the same series."""
    t = heading_anchor.AnchorTracker()
    t.on_sync(0.25, shared=True)
    values = [0.25, 0.25 + 0.01, 0.25 + 0.02]
    for v in values:
      self.assertAlmostEqual(t.ref_yaw_unanchored(v), v, places=12)


class FrameReconstructionTest(unittest.TestCase):
  """The fixed-frame reference must be rebuilt with the quaternion."""

  def test_yaw_offset_round_trip(self):
    rng = np.random.RandomState(3)
    for _ in range(20):
      q = metrics.normalize_quat(rng.randn(4))
      theta = float(rng.uniform(-1.2, 1.2))
      q_off = metrics.with_yaw_offset(q, theta)
      back = metrics.undo_yaw_offset(q_off, theta)
      np.testing.assert_allclose(back, q, atol=1e-12)

  def test_yaw_angle_adds_exactly(self):
    rng = np.random.RandomState(7)
    for _ in range(20):
      q = metrics.normalize_quat(rng.randn(4))
      theta = float(rng.uniform(-2.0, 2.0))
      got = metrics.quat_to_yaw(metrics.with_yaw_offset(q, theta))
      self.assertAlmostEqual(metrics.wrap_angle(got),
                             metrics.wrap_angle(
                                 metrics.quat_to_yaw(q) + theta), places=10)

  def test_offset_frame_equals_opening_frame(self):
    """theta_cf applied to the raw clip is exactly what upstream would give."""
    rng = np.random.RandomState(11)
    q = metrics.normalize_quat(rng.randn(4))
    theta_cf = 0.37
    anchored = metrics.with_yaw_offset(q, theta_cf)
    fixed = metrics.with_yaw_offset(metrics.undo_yaw_offset(anchored, theta_cf),
                                    theta_cf)
    np.testing.assert_allclose(anchored, fixed, atol=1e-12)

  def test_seed_makes_first_cycle_count(self):
    """A tracker seeded at reset must count the first cycle event."""
    tracker = heading_anchor.AnchorTracker()
    tracker.seed(-0.46)
    tracker.on_sync(-0.735, shared=False)
    self.assertEqual(tracker.cycle_events, 1)
    self.assertAlmostEqual(tracker.cum, metrics.wrap_angle(-0.735 + 0.46),
                           places=10)


class MultiClipFrameTests(unittest.TestCase):
  """theta_cf must follow upstream when the clip changes mid-episode."""

  def test_first_sync_without_seed_still_sets_theta_cf(self):
    t = heading_anchor.AnchorTracker()
    self.assertIsNone(t.theta_cf)
    t.on_sync(0.42, shared=True)
    self.assertAlmostEqual(t.theta_cf, 0.42, places=12)

  def test_clip_change_rebases_theta_cf(self):
    t = heading_anchor.AnchorTracker()
    t.seed(0.40)
    t.on_sync(0.55, shared=False)          # one cycle event inside clip 1
    self.assertAlmostEqual(t.cum, 0.15, places=12)
    t.on_sync(0.90, shared=True)           # upstream re-syncs on clip change
    self.assertAlmostEqual(t.theta_cf, 0.90, places=12)
    self.assertEqual(t.cum, 0.0)
    self.assertEqual(t.clip_events, 1)
    # cycle offsets after the switch accumulate from the new baseline only
    self.assertAlmostEqual(t.on_sync(0.95, shared=False), 0.05, places=12)

  def test_clip_change_across_pi_is_a_small_step(self):
    t = heading_anchor.AnchorTracker()
    t.seed(3.05)
    t.on_sync(-3.05, shared=True)
    self.assertAlmostEqual(t.theta_cf, metrics.wrap_angle(-3.05), places=12)
    self.assertAlmostEqual(t.clip_sync_delta,
                           metrics.wrap_angle(-3.05 - 3.05), places=12)
    self.assertLess(abs(t.clip_sync_delta), 0.2)

  def test_reconstruction_identity_across_two_clips(self):
    """Same-trajectory fixed frame must rebase at the clip boundary."""
    rng = np.random.RandomState(17)
    # clip A: raw quats with pitch/roll, opening offset theta0
    qa = [metrics.normalize_quat(rng.randn(4)) for _ in range(30)]
    qb = [metrics.normalize_quat(rng.randn(4)) for _ in range(30)]

    t = heading_anchor.AnchorTracker()
    t.seed(0.40)
    gaps = []
    for k, q in enumerate(qa):
      if k and k % 10 == 0:
        t.on_sync(t.theta + rng.uniform(-0.05, 0.05), shared=False)
      theta, theta_cf = t.theta, t.theta_cf
      anchored = metrics.with_yaw_offset(q, theta)
      fixed = metrics.with_yaw_offset(q, theta_cf)
      gap = metrics.wrap_angle(metrics.quat_to_yaw(anchored)
                               - metrics.quat_to_yaw(fixed))
      gaps.append(metrics.wrap_angle(gap - t.cum))

    # clip switch: upstream installs a new absolute offset for clip B
    t.on_sync(0.90, shared=True)
    self.assertEqual(t.cum, 0.0)
    for k, q in enumerate(qb):
      if k and k % 10 == 0:
        t.on_sync(t.theta + rng.uniform(-0.05, 0.05), shared=False)
      theta, theta_cf = t.theta, t.theta_cf
      anchored = metrics.with_yaw_offset(q, theta)
      fixed = metrics.with_yaw_offset(q, theta_cf)
      gap = metrics.wrap_angle(metrics.quat_to_yaw(anchored)
                               - metrics.quat_to_yaw(fixed))
      gaps.append(metrics.wrap_angle(gap - t.cum))

    self.assertAlmostEqual(max(abs(g) for g in gaps), 0.0, places=9)
    # the fixed frame of clip B is the frame upstream would track
    q_raw_b0 = qb[0]
    np.testing.assert_allclose(
        metrics.with_yaw_offset(metrics.undo_yaw_offset(
            metrics.with_yaw_offset(q_raw_b0, 0.90), 0.90), 0.90),
        metrics.with_yaw_offset(q_raw_b0, 0.90), atol=1e-12)

  def test_fixed_frame_stays_within_upstream_frame_after_switch(self):
    """Before the switch cum counts; after it the new baseline is clean."""
    t = heading_anchor.AnchorTracker()
    t.seed(-0.20)
    for d in (0.03, 0.06, 0.04):
      t.on_sync(t.theta + d, shared=False)
    self.assertAlmostEqual(t.cum, 0.13, places=12)
    t.on_sync(1.10, shared=True)
    self.assertAlmostEqual(t.cum, 0.0, places=12)
    self.assertAlmostEqual(t.theta_cf, 1.10, places=12)
    self.assertAlmostEqual(
        t.ref_yaw_unanchored(metrics.wrap_angle(1.12)), 1.12, places=12)


if __name__ == "__main__":
  unittest.main()
