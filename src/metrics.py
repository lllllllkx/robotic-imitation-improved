"""Metric helpers shared by the evaluation and training scripts."""

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
  slope equal to the per-second mismatch.
  """
  values = np.asarray(values, dtype=float)
  if values.size < 2:
    return 0.0
  t = np.arange(values.size) * dt
  slope = np.polyfit(t, values, 1)[0]
  return float(slope)


def summarize_episodes(episodes):
  """Aggregate per-episode dicts into mean/std across episodes."""
  keys = [k for k, v in episodes[0].items()
          if isinstance(v, (int, float)) and not isinstance(v, bool)]
  out = {"num_episodes": len(episodes)}
  for k in keys:
    vals = np.asarray([float(e[k]) for e in episodes], dtype=float)
    out[k + "_mean"] = float(np.mean(vals))
    out[k + "_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
  out["episodes"] = episodes
  return out
