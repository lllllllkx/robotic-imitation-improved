"""Re-encode the eval videos from the already rendered frames (episode 0 only).

The simulation is deterministic, so every episode renders the same motion; the
released videos should represent one 600-step (19.8 s @30 fps) episode rather
than 5 repeats.
"""

import os

import imageio
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for tag in ["dog_pace", "dog_spin"]:
    frame_dir = os.path.join(ROOT, "outputs", "eval", tag, "frames")
    frames = sorted(f for f in os.listdir(frame_dir)
                    if f.startswith("ep00_") and f.endswith(".png"))
    if not frames:
        print("[skip] no episode-0 frames for", tag)
        continue
    imgs = [np.asarray(Image.open(os.path.join(frame_dir, f))) for f in frames]
    out_mp4 = os.path.join(ROOT, "outputs", "eval", tag + ".mp4")
    imageio.mimsave(out_mp4, imgs, fps=30, quality=7)
    picks = np.linspace(0, len(imgs) - 1, 6).astype(int)
    strip = np.concatenate([imgs[i] for i in picks], axis=1)
    Image.fromarray(strip).save(os.path.join(ROOT, "outputs", "eval",
                                             tag + "_filmstrip.png"))
    print("%-9s %d frames -> %.1f s, %.1f MB"
          % (tag, len(imgs), len(imgs) / 30.0, os.path.getsize(out_mp4) / 1e6))
