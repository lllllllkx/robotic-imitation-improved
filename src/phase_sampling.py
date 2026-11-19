"""Initial reference-phase sampling for the 41-step plateau experiments.

The sampler changes only *where in the reference clip an episode starts*. It
does not touch the reward, the randomizer, the action controller or the PPO
hyper-parameters. It is installed as a replacement for the task's
``_reset_motion_time_offset`` hook.

Sampling rule (phasefocus):

* ``focus_fraction`` of episodes pick one of the focus bands, with equal
  probability per band (not proportional to band width);
* the remaining episodes sample the whole clip uniformly.

Episodes where the upstream task enables its warmup period keep the upstream
warmup offset. They are recorded with ``source=warmup`` and are not counted as
phasefocus samples. This keeps the intervention limited to normal reference
initialisation.
"""

from __future__ import absolute_import, division, print_function

import csv
import io

import numpy as np

FOCUS_BANDS = ((0.11, 0.14), (0.41, 0.50), (0.95, 0.98))


def parse_bands(text):
  """Parse ``"0.11:0.14,0.41:0.50"`` into a tuple of (lo, hi) pairs."""
  bands = []
  for item in text.split(","):
    item = item.strip()
    if not item:
      continue
    parts = item.split(":")
    if len(parts) != 2:
      raise ValueError("band must be lo:hi, got %r" % (item,))
    lo, hi = float(parts[0]), float(parts[1])
    if not (0.0 <= lo < hi <= 1.0):
      raise ValueError("band must satisfy 0 <= lo < hi <= 1, got %r" % (item,))
    bands.append((lo, hi))
  if not bands:
    raise ValueError("at least one band is required")
  return tuple(bands)


def band_index(phase, bands):
  for i, (lo, hi) in enumerate(bands):
    if lo <= phase < hi:
      return i
  return None


class PhaseFocusSampler(object):
  """Deterministic per-rank sampler with an on-disk audit trail."""

  def __init__(self, seed, bands=FOCUS_BANDS, focus_fraction=0.5,
               log_path=None, rank=0):
    if not (0.0 <= float(focus_fraction) <= 1.0):
      raise ValueError("focus_fraction must be in [0, 1]")
    self._rng = np.random.RandomState(int(seed))
    self._bands = tuple((float(lo), float(hi)) for lo, hi in bands)
    self._focus_fraction = float(focus_fraction)
    self._rank = int(rank)
    self._count = 0
    self._stats = {"total": 0, "warmup": 0, "uniform": 0,
                   "band_0": 0, "band_1": 0, "band_2": 0,
                   "band_other": 0, "in_focus_band": 0}
    self._fh = None
    self._writer = None
    if log_path:
      self._fh = io.open(log_path, "w", encoding="utf-8", newline="")
      self._writer = csv.writer(self._fh)
      self._writer.writerow(["rank", "episode", "phase", "source",
                             "focus_band", "in_focus_band", "warmup"])

  @property
  def bands(self):
    return self._bands

  def sample(self):
    """Return ``(phase, source)`` for one normal episode."""
    if self._rng.rand() < self._focus_fraction:
      band = int(self._rng.randint(len(self._bands)))
      lo, hi = self._bands[band]
      phase = float(self._rng.uniform(lo, hi))
      source = "band_%d" % band
    else:
      phase = float(self._rng.rand())
      source = "uniform"
    return phase, source

  def record(self, phase, source, warmup=False):
    """Append one episode's actual starting phase to the audit CSV."""
    phase = float(phase)
    idx = band_index(phase, self._bands) if not warmup else None
    self._count += 1
    self._stats["total"] += 1
    if warmup:
      self._stats["warmup"] += 1
    elif source in self._stats:
      self._stats[source] += 1
    if idx is not None:
      self._stats["in_focus_band"] += 1
    if self._writer is not None:
      self._writer.writerow([self._rank, self._count, "%.9f" % phase,
                             source, "" if idx is None else idx,
                             int(idx is not None), int(bool(warmup))])
      self._fh.flush()
    return phase, source

  def stats(self):
    out = dict(self._stats)
    out["focus_fraction_requested"] = self._focus_fraction
    return out

  def close(self):
    if self._fh is not None:
      self._fh.close()
      self._fh = None


def install_phase_sampler(task, sampler):
  """Replace ``_reset_motion_time_offset`` with the sampler.

  Warmup episodes call the original hook, so their offset distribution and
  the subsequent ``warmup_time`` subtraction are exactly as upstream.
  """
  if getattr(task, "_phase_sampler_installed", False):
    raise RuntimeError("phase sampler already installed on this task")
  orig_reset = task._reset_motion_time_offset

  def _reset_motion_time_offset():
    motion = task.get_active_motion()
    duration = float(motion.get_duration())
    if getattr(task, "_curr_episode_warmup", False):
      orig_reset()
      phase = float(task._motion_time_offset) / duration
      sampler.record(phase, "warmup", warmup=True)
      return
    phase, source = sampler.sample()
    task._motion_time_offset = phase * duration
    sampler.record(phase, source, warmup=False)

  task._reset_motion_time_offset_orig = orig_reset
  task._reset_motion_time_offset = _reset_motion_time_offset
  task._phase_sampler_installed = True
  task._phase_sampler = sampler
  return sampler


def clear_phase_sampler(task):
  orig = getattr(task, "_reset_motion_time_offset_orig", None)
  if orig is not None:
    task._reset_motion_time_offset = orig
  task._phase_sampler_installed = False
  task._phase_sampler = None
  return task
