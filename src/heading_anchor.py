"""Bookkeeping for the heading offset applied by cycle re-anchoring.

Why this exists
---------------
Upstream ``_sync_ref_origin`` sets ``_origin_offset_rot`` from
``heading_diff = sim_heading - ref_heading``, i.e. it re-defines the *absolute*
world yaw offset of the reference motion instead of accumulating it. Improvement
1 (``env_patch.enable_cycle_heading_anchor``) calls that same routine at every
gait-cycle boundary; upstream itself only calls it on clip changes and at reset.

Everything the reference rotation does is therefore expressible as one angle
``theta``: the world yaw of the reference is ``psi_raw(t) + theta``, where
``psi_raw`` is the clip's own root rotation. When the anchor re-syncs at a cycle
boundary, ``theta`` takes a new value; the *change* is what turns the reference
frame under the robot's feet.

To ask "what would the yaw error have been if this frame had not been rotated",
we need the world-yaw offset our extra anchoring has accumulated since the last
point at which our frame and the upstream frame coincided (reset, or a clip
change where upstream syncs too; both re-base ``theta_cf``):

    psi_ref_unanchored(t) = psi_ref_anchored(t) - cum

This is a *post-hoc* reconstruction along one already-executed trajectory. The
policy observed the re-anchored reference while acting, so the reconstructed
error is a counterfactual measurement of that trajectory, **not** the error the
policy would have produced with anchoring disabled (see
``docs/HEADING_FRAMES.md``).

Sign convention: ``theta`` is the world yaw of ``_origin_offset_rot``
(``metrics.quat_to_yaw``), and increments are taken as wrapped differences
``theta_new - theta_old``, so a clockwise (negative) re-anchoring decreases
``cum``.

The exact statement is a quaternion identity, not an angle sum: with
``q_raw`` the clip's own root rotation at frame t,

    q_ref_anchored = Rz(theta)  * q_raw
    q_ref_fixed    = Rz(theta_cf) * q_raw        (theta_cf: offset at reset)

and ``cum = theta - theta_cf``. Because ``Rz`` acts on the left, the world yaw
really does add, but the *fixed-frame* reference must be rebuilt by composing
the quaternion (``metrics.with_yaw_offset``), otherwise a body with pitch or
roll produces small disagreements. The reconstruction is never assumed from
variable names: it is derived from the quaternions upstream actually installed,
and every evaluation run checks the tracker against it.
"""

from __future__ import absolute_import
from __future__ import division

from src.metrics import quat_multiply
from src.metrics import quat_to_yaw
from src.metrics import undo_yaw_offset
from src.metrics import wrap_angle


class AnchorTracker(object):
  """Accumulates the world-yaw offset added by cycle re-anchoring.

  ``on_sync`` must be called after every ``_sync_ref_origin`` with the yaw of
  the offset quaternion that was just installed and with whether that sync is
  one upstream would also perform (``shared``: reset / clip change) or one that
  only our cycle anchoring adds (``cycle``).
  """

  def __init__(self):
    self.theta = None          # last installed offset yaw (wrapped)
    self.cum = 0.0             # unwrapped offset added since the last shared sync
    self.clip_sync_delta = 0.0  # only for auditing: delta of the last shared sync
    self.shared_events = 0
    self.cycle_events = 0
    self.clip_events = 0       # shared syncs after the first (clip changes)
    self.theta_cf = None   # offset of the last shared sync (upstream's frame)
    self.events = []       # (shared, theta_new, cum_after), for diagnostics

  def on_sync(self, theta_new, shared):
    theta_new = wrap_angle(theta_new)
    if self.theta is None:
      # first sync of the episode: this is the frame baseline, nothing to add
      self.cum = 0.0
      self.theta_cf = theta_new
      self.shared_events += 1
    elif shared:
      # upstream re-synced as well, so its frame and ours coincide again.
      # On a clip change upstream also re-installs the offset from the *new*
      # clip's own heading, so the fixed frame rebases here too; keeping the
      # episode-opening theta_cf would rebuild the fixed reference with the
      # previous clip's baseline and make the comparison meaningless across
      # clips (the single-clip limitation noted in HEADING_FRAMES.md section 8).
      self.clip_sync_delta = wrap_angle(theta_new - self.theta)
      self.cum = 0.0
      self.theta_cf = theta_new
      self.shared_events += 1
      self.clip_events += 1
    else:
      self.cum = self.cum + wrap_angle(theta_new - self.theta)
      self.cycle_events += 1
    self.theta = theta_new
    self.events.append((bool(shared), theta_new, self.cum))
    return self.cum

  def seed(self, theta0):
    """Re-baseline at the start of an episode.

    ``ImitationTask._reset_ref_motion`` installs the opening offset *directly*
    (``self._origin_offset_rot = quaternion_about_axis(sim_heading -
    ref_heading, ...)``) instead of going through ``_sync_ref_origin``, so the
    tracker never sees it. Without this seed the first cycle event would be
    mistaken for the baseline and that whole offset would be dropped from cum.
    """
    self.theta = wrap_angle(theta0)
    self.theta_cf = self.theta       # offset upstream would keep for the episode
    self.cum = 0.0
    self.clip_sync_delta = 0.0
    self.shared_events = 1
    self.cycle_events = 0
    self.clip_events = 0
    self.events = []
    return self.cum

  def ref_yaw_unanchored(self, ref_yaw_anchored):
    """Counterfactual reference yaw: the frame without our cycle re-anchoring."""
    return wrap_angle(ref_yaw_anchored - self.cum)


def offset_yaw(origin_offset_rot):
  """World yaw of an ``_origin_offset_rot`` quaternion (pybullet order)."""
  return quat_to_yaw(origin_offset_rot)


def undo_offset(quat, theta):
  """Remove a world-yaw offset and recover the clip's own rotation."""
  return undo_yaw_offset(quat, theta)
