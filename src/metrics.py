"""Metric helpers shared by the evaluation and training scripts.

Everything here is pure numpy so it can be unit tested without TensorFlow or
pybullet (see ``tests/test_metrics.py``).
"""

from __future__ import absolute_import
from __future__ import division

import numpy as np

TERM_NAMES = ("pose", "velocity", "end_effector", "root_pose", "root_velocity")


def normalized_return(term_means, weights):
  """Paper-style normalised return for one step, from the five reward terms.

  ``ImitationTask.reward`` is a weighted sum of five terms, each of which is
  ``exp(-scale * err)`` and therefore in (0, 1]. Because the upstream weights
  sum to exactly 1.0, the per-step reward is itself already normalised to
  [0, 1] with 1.0 = perfect tracking: this function reproduces
  ``ImitationTask.reward`` from logged terms and is used to cross-check that
  ``mean(reward)`` in our logs is the same quantity the paper reports in
  Table V.
  """
  weights = np.asarray([weights[n] for n in TERM_NAMES], dtype=float)
  terms = np.asarray([term_means[n] for n in TERM_NAMES], dtype=float)
  if abs(weights.sum() - 1.0) > 1e-9:
    raise ValueError("weights must sum to 1, got %.6f" % weights.sum())
  return float(terms.dot(weights))


def quat_to_yaw(quat):
  """Yaw (rotation about +z) of a quaternion in pybullet order ``[x, y, z, w]``.

  The upstream evaluation used ``2 * atan2(qz, qw)``, which is only correct
  when roll and pitch are exactly zero. This is the general extrinsic-ZYX
  yaw: ``atan2(2(wz + xy), 1 - 2(y^2 + z^2))``.
  """
  x, y, z, w = (float(v) for v in quat[:4])
  return float(np.arctan2(2.0 * (w * z + x * y),
                          1.0 - 2.0 * (y * y + z * z)))


def wrap_angle(angle):
  """Wrap an angle to (-pi, pi]."""
  return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def unwrap_angles(angles):
  """Remove 2*pi discontinuities: accumulate the wrapped per-step increments."""
  out = []
  prev_raw = None
  prev_out = 0.0
  for raw in np.asarray(angles, dtype=float):
    if prev_raw is None:
      prev_out = float(raw)
    else:
      prev_out = prev_out + wrap_angle(raw - prev_raw)
    out.append(prev_out)
    prev_raw = raw
  return np.asarray(out, dtype=float)


def velocity_reward(sum_sq_err, n_joints, scale):
  """One velocity reward value, given the summed squared joint-velocity error.

  ``scale`` is the exponent scale, matching upstream ``exp(-0.1*sum)`` and
  ``improvement 2`` ``exp(-0.25*mean)``. Note ``exp(-0.1*sum)`` and
  ``exp(-1.2*mean)`` are the *same* number for 12 joints -- averaging without
  changing the scale is an identity, which is why v2 also changes the scale.
  """
  return float(np.exp(-float(scale) * float(sum_sq_err) / float(n_joints)))


def window_mean(values, dt, window_s):
  """Mean over the trailing ``window_s`` seconds of an episode."""
  values = np.asarray(values, dtype=float)
  if values.size == 0:
    return float("nan")
  n = max(1, min(values.size, int(round(window_s / dt))))
  return float(np.mean(values[-n:]))


def drift_rate(values, dt):
  """Least-squares slope (per second) of a per-step signal.

  Used to separate "bounded periodic error" from "unbounded accumulation":
  a well-tracked cyclic skill has a near-zero slope, an accumulating one has a
  slope equal to the per-second mismatch. Feed it an *unwrapped signed* angle
  series (`unwrap_angles`), never a wrapped absolute error -- the latter
  folds back at +-pi and would hide the accumulation it is meant to measure.
  """
  values = np.asarray(values, dtype=float)
  if values.size < 2:
    return 0.0
  t = np.arange(values.size) * dt
  slope = np.polyfit(t, values, 1)[0]
  return float(slope)


def termination_reason(done, steps, max_steps):
  """Separate the time limit from an early (e.g. falling) termination."""
  if steps >= max_steps:
    return "time_limit"
  return "early_termination" if done else "unknown"


def summarize_episodes(episodes):
  """Aggregate per-episode dicts into mean/std across episodes.

  std uses ddof=1 (sample standard deviation). Episodes are pooled, which
  describes the spread over all evaluated initial conditions; use
  `aggregate_by_seed` when the seeds themselves carry meaning.
  """
  keys = [k for k, v in episodes[0].items()
          if isinstance(v, (int, float)) and not isinstance(v, bool)]
  out = {"num_episodes": len(episodes)}
  for k in keys:
    vals = np.asarray([float(e[k]) for e in episodes], dtype=float)
    out[k + "_mean"] = float(np.mean(vals))
    out[k + "_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
  out["episodes"] = episodes
  return out


def aggregate_by_seed(episodes, keys=None):
  """Two-level aggregation: episodes -> per-seed mean -> mean/std over seeds.

  An evaluation seed here controls *initial conditions* (start phase), not
  policy training; the paper's own seeds are training seeds (three
  independently trained policies). Keeping the levels separate stops a run
  with many episodes per seed from being read as many independent policies.
  """
  if keys is None:
    keys = [k for k, v in episodes[0].items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
  groups = {}
  for episode in episodes:
    groups.setdefault(int(episode["seed"]), []).append(episode)
  out = {"num_seeds": len(groups), "seeds": []}
  per_seed = {}
  for seed, eps in sorted(groups.items()):
    entry = {"seed": seed, "num_episodes": len(eps)}
    for k in keys:
      entry[k] = float(np.mean([float(e[k]) for e in eps]))
    out["seeds"].append(entry)
    per_seed[seed] = entry
  for k in keys:
    vals = np.asarray([per_seed[s][k] for s in sorted(per_seed)], dtype=float)
    out[k + "_mean_across_seeds"] = float(np.mean(vals))
    out[k + "_std_across_seeds"] = (
        float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
  return out
