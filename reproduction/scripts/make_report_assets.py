"""Build figures + summary tables for the reproduction report.

Reads the CSV/JSON produced by eval_pretrained.py, dump_reference_motion.py
and train_short.py, then writes SVG figures and a data JSON into report/.
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linestyle"] = ":"

C_PACE = "#1f77b4"
C_SPIN = "#d62728"
C_REF = "#7f7f7f"
C_TRAIN = "#2ca02c"


def read_csv(path):
  with open(path, newline="") as fh:
    return list(csv.DictReader(fh))


def to_float(rows, key):
  return np.array([float(r[key]) for r in rows])


def rolling(x, w):
  if len(x) < w:
    return x
  k = np.ones(w) / float(w)
  return np.convolve(x, k, mode="valid")


def panel_time(ax, rows, keys, labels, colors, w=30, ylabel="", title=""):
  t = to_float(rows, "time")
  for key, lab, c in zip(keys, labels, colors):
    y = to_float(rows, key)
    yr = rolling(y, w)
    ax.plot(t[w - 1:] if len(y) >= w else t, yr, label=lab, color=c, lw=1.6)
  ax.set_xlabel("时间 (s)")
  ax.set_ylabel(ylabel)
  ax.set_title(title, fontsize=11, loc="left")
  if len(keys) > 1:
    ax.legend(frameon=False, fontsize=9)


def fig_learning_curve(ep_csv, out_svg, title, scale=1.0, xlabel="训练步数 (timesteps)"):
  rows = read_csv(ep_csv)
  steps = to_float(rows, "global_step") * scale
  rets = to_float(rows, "episode_return")
  lens = to_float(rows, "episode_length")
  w = min(15, max(3, len(rets) // 20))
  rs = rolling(rets, w)
  fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
  ax = axes[0]
  ax.plot(steps, rets, color=C_TRAIN, alpha=0.25, lw=0.7)
  ax.plot(steps[w - 1:] if len(rets) >= w else steps, rs, color=C_TRAIN,
          lw=2.0, label="滑动平均 (%d 回合)" % w)
  ax.set_xlabel(xlabel)
  ax.set_ylabel("回合累计奖励")
  ax.set_title(title, fontsize=11, loc="left")
  ax.legend(frameon=False, fontsize=9)

  ax = axes[1]
  ax.plot(steps, lens, color="#9467bd", alpha=0.3, lw=0.7)
  ax.plot(steps[w - 1:] if len(lens) >= w else steps, rolling(lens, w),
          color="#9467bd", lw=2.0)
  ax.set_xlabel(xlabel)
  ax.set_ylabel("回合长度 (env steps @30Hz)")
  ax.set_title("课程学习：回合长度随训练增长", fontsize=11, loc="left")
  fig.tight_layout()
  fig.savefig(out_svg, bbox_inches="tight")
  fig.savefig(out_svg.replace(".svg", ".png"), bbox_inches="tight", dpi=150)
  plt.close(fig)
  return {
      "episodes": int(len(rets)),
      "final_window_return": float(np.mean(rets[-w:])),
      "first_window_return": float(np.mean(rets[:w])),
      "max_return": float(np.max(rets)),
      "final_window_len": float(np.mean(lens[-w:])),
      "first_window_len": float(np.mean(lens[:w])),
  }


def fig_tracking(eval_dir, tags, out_svg, metric_key, ylabel, title, w=30):
  fig, ax = plt.subplots(figsize=(6.4, 3.3))
  colors = {"dog_pace": C_PACE, "dog_spin": C_SPIN}
  for tag in tags:
    rows = read_csv(os.path.join(eval_dir, tag, "steps.csv"))
    rows = [r for r in rows if int(r["episode"]) == 0]
    t = to_float(rows, "time")
    y = rolling(to_float(rows, metric_key), w)
    ax.plot(t[w - 1:], y, lw=1.7, color=colors.get(tag, None), label=tag)
  ax.set_xlabel("时间 (s)")
  ax.set_ylabel(ylabel)
  ax.set_title(title, fontsize=11, loc="left")
  ax.legend(frameon=False, fontsize=9)
  fig.tight_layout()
  fig.savefig(out_svg, bbox_inches="tight")
  fig.savefig(out_svg.replace(".svg", ".png"), bbox_inches="tight", dpi=150)
  plt.close(fig)


def fig_joint_tracking(step_csv, out_svg, joints=(0, 2, 8), t_max=6.0):
  rows = read_csv(step_csv)
  rows = [r for r in rows if int(r["episode"]) == 0]
  t = to_float(rows, "time")
  mask = t <= t_max
  t = t[mask]
  n = len(joints)
  fig, axes = plt.subplots(n, 1, figsize=(6.6, 1.7 * n + 0.6), sharex=True)
  if n == 1:
    axes = [axes]
  names = ["前左髋 (q0)", "前左膝 (q2)", "后左髋 (q8)"]
  for ax, j in zip(axes, joints):
    q = to_float(rows, "q%d" % j)[mask]
    qr = to_float(rows, "qref%d" % j)[mask]
    ax.plot(t, qr, color=C_REF, lw=1.5, label="参考动作 (mocap)")
    ax.plot(t, q, color=C_PACE, lw=1.4, label="策略输出")
    ax.set_ylabel("关节角 (rad)", fontsize=9)
    ax.set_title(names[joints.index(j)] if j in joints else "q%d" % j,
                 fontsize=10, loc="left")
  axes[0].legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
  axes[-1].set_xlabel("时间 (s)")
  fig.tight_layout()
  fig.savefig(out_svg, bbox_inches="tight")
  fig.savefig(out_svg.replace(".svg", ".png"), bbox_inches="tight", dpi=150)
  plt.close(fig)


def fig_reward_terms(eval_dir, tags, out_svg):
  terms = ["pose_reward", "velocity_reward", "end_effector_reward",
           "root_pose_reward", "root_velocity_reward"]
  labels = ["姿态", "关节速度", "足端位置", "根姿态", "根速度"]
  vals = {}
  for tag in tags:
    with open(os.path.join(eval_dir, tag, "summary.json")) as fh:
      summ = json.load(fh)
    vals[tag] = [np.mean([e["mean_term_" + t] for e in summ["episodes"]])
                 for t in terms]
  x = np.arange(len(terms))
  width = 0.36
  fig, ax = plt.subplots(figsize=(6.4, 3.2))
  colors = [C_PACE, C_SPIN, "#2ca02c", "#9467bd"]
  for i, tag in enumerate(tags):
    ax.bar(x + (i - (len(tags) - 1) / 2.0) * width, vals[tag], width,
           label=tag, color=colors[i % len(colors)])
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.set_ylabel("平均奖励分项 (0-1)")
  ax.set_title("模仿奖励分项：预训练策略", fontsize=11, loc="left")
  ax.legend(frameon=False, fontsize=9)
  fig.tight_layout()
  fig.savefig(out_svg, bbox_inches="tight")
  fig.savefig(out_svg.replace(".svg", ".png"), bbox_inches="tight", dpi=150)
  plt.close(fig)
  return vals


def fig_reference_library(ref_dir, out_svg):
  with open(os.path.join(ref_dir, "reference_summary.json")) as fh:
    meta = json.load(fh)
  names = [m["motion_file"].replace(".txt", "") for m in meta]
  dur = [m["duration_s"] for m in meta]
  spd = [m["mean_speed_mps"] for m in meta]
  order = np.argsort(dur)
  names = [names[i] for i in order]
  dur = [dur[i] for i in order]
  spd = [spd[i] for i in order]
  fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
  ax = axes[0]
  ax.barh(names, dur, color="#4c72b0")
  ax.set_xlabel("动作时长 (s)")
  ax.set_title("参考动作库：片段时长", fontsize=11, loc="left")
  ax.grid(axis="y", visible=False)
  ax = axes[1]
  ax.barh(names, spd, color="#dd8452")
  ax.set_xlabel("平均根节点速度 (m/s)")
  ax.set_title("参考动作库：速度", fontsize=11, loc="left")
  ax.grid(axis="y", visible=False)
  fig.tight_layout()
  fig.savefig(out_svg, bbox_inches="tight")
  fig.savefig(out_svg.replace(".svg", ".png"), bbox_inches="tight", dpi=150)
  plt.close(fig)


def fig_root_tracking(step_csv, out_svg):
  rows = read_csv(step_csv)
  rows = [r for r in rows if int(r["episode"]) == 0]
  t = to_float(rows, "time")
  fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
  ax = axes[0]
  ax.plot(t, to_float(rows, "ref_z"), color=C_REF, lw=1.5, label="参考高度")
  ax.plot(t, to_float(rows, "sim_z"), color=C_PACE, lw=1.4, label="仿真高度")
  ax.set_xlabel("时间 (s)")
  ax.set_ylabel("根节点高度 z (m)")
  ax.set_title("根节点高度跟踪", fontsize=11, loc="left")
  ax.legend(frameon=False, fontsize=9)
  ax = axes[1]
  ax.plot(t, to_float(rows, "speed_mps"), color=C_PACE, lw=1.4)
  ax.set_xlabel("时间 (s)")
  ax.set_ylabel("速度 (m/s)")
  ax.set_title("前进速度（参考约 1.09 m/s）", fontsize=11, loc="left")
  fig.tight_layout()
  fig.savefig(out_svg, bbox_inches="tight")
  fig.savefig(out_svg.replace(".svg", ".png"), bbox_inches="tight", dpi=150)
  plt.close(fig)


def fig_vs_paper(eval_dir, tags, out_svg):
  """Paper Table V (sim, normalized return) vs our measured per-step reward."""
  paper = {
      "dog_pace": {"No Rand": 0.839, "Robust": 0.820, "Adaptive": 0.812},
      "dog_spin": {"No Rand": 0.859, "Robust": 0.839, "Adaptive": 0.839},
  }
  label_map = {"No Rand": "论文·无随机化", "Robust": "论文·鲁棒策略",
               "Adaptive": "论文·自适应"}
  ours = {}
  for tag in tags:
    p = os.path.join(eval_dir, tag, "summary.json")
    if os.path.exists(p):
      with open(p) as fh:
        s = json.load(fh)
      ours[tag] = float(np.mean([e["mean_reward"] for e in s["episodes"]]))
  groups = []
  for tag in tags:
    for k in ["No Rand", "Robust", "Adaptive"]:
      groups.append((tag, k, paper[tag][k], label_map[k]))
    groups.append((tag, "ours", ours.get(tag, np.nan), "本次复现 (预训练策略)"))
  fig, ax = plt.subplots(figsize=(8.4, 3.4))
  xs = np.arange(len(groups))
  bars = ax.bar(xs,
                [g[2] for g in groups],
                color=["#9ecae1", "#6baed6", "#3182bd", "#e6550d"][:len(groups)] if False else None,
                width=0.62)
  for bar, g in zip(bars, groups):
    bar.set_color("#e6550d" if g[1] == "ours" else "#6baed6")
    if g[1] == "No Rand":
      bar.set_color("#c6dbef")
    elif g[1] == "Adaptive":
      bar.set_color("#3182bd")
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.012,
            "%.3f" % g[2], ha="center", va="bottom", fontsize=8)
  ax.set_xticks(xs)
  ax.set_xticklabels([g[3] for g in groups], fontsize=8, rotation=18, ha="right")
  ax.axvline(3.5, color="#999999", lw=1.0, ls="--")
  skill_labels = {"dog_pace": "Dog Pace", "dog_spin": "Dog Spin"}
  for i, tag in enumerate(tags):
    ax.text(i * 4 + 1.5, 0.965, skill_labels.get(tag, tag), ha="center",
            va="center", fontsize=10, fontweight="bold",
            transform=ax.get_xaxis_transform())
  ax.set_ylabel("归一化回报 / 每步奖励 (0-1)")
  ax.set_ylim(0, 1.0)
  ax.set_title("与论文 Table V（仿真）对比：预训练策略", fontsize=11, loc="left")
  ax.grid(axis="x", visible=False)
  fig.tight_layout()
  fig.savefig(out_svg, bbox_inches="tight")
  fig.savefig(out_svg.replace(".svg", ".png"), bbox_inches="tight", dpi=150)
  plt.close(fig)
  return {"paper": paper, "ours": ours}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", required=True)
  ap.add_argument("--train_dir", default="")
  ap.add_argument("--out_dir", required=True)
  args = ap.parse_args()
  root = args.root
  out = os.path.join(root, args.out_dir)
  os.makedirs(out, exist_ok=True)
  eval_dir = os.path.join(root, "outputs", "eval")
  ref_dir = os.path.join(root, "outputs", "reference")

  data = {}

  fig_tracking(eval_dir, ["dog_pace", "dog_spin"],
               os.path.join(out, "fig_reward.svg"), "reward",
               "逐步奖励", "每步模仿奖励（1 s 滑动平均）")
  fig_tracking(eval_dir, ["dog_pace", "dog_spin"],
               os.path.join(out, "fig_pose_err.svg"), "pose_err_rad",
               "关节平均绝对误差 (rad)", "关节跟踪误差（1 s 滑动平均）")
  fig_root_tracking(os.path.join(eval_dir, "dog_pace", "steps.csv"),
                    os.path.join(out, "fig_root.svg"))
  fig_joint_tracking(os.path.join(eval_dir, "dog_pace", "steps.csv"),
                     os.path.join(out, "fig_joints.svg"))
  data["reward_terms"] = fig_reward_terms(eval_dir, ["dog_pace", "dog_spin"],
                                          os.path.join(out, "fig_terms.svg"))
  fig_reference_library(ref_dir, os.path.join(out, "fig_reference.svg"))
  data["vs_paper"] = fig_vs_paper(eval_dir, ["dog_pace", "dog_spin"],
                                  os.path.join(out, "fig_vs_paper.svg"))

  if args.train_dir:
    ep = os.path.join(root, "outputs", "train", args.train_dir, "episodes.csv")
    if os.path.exists(ep):
      tag = args.train_dir
      summ_path = os.path.join(root, "outputs", "train", tag, "train_summary.json")
      scale = 1.0
      if os.path.exists(summ_path):
        with open(summ_path) as fh:
          scale = float(json.load(fh).get("num_procs", 1))
      xlabel = "聚合训练步数（%d 进程）" % int(scale)
      data["learning"] = fig_learning_curve(
          ep, os.path.join(out, "fig_learning.svg"), "学习曲线：%s" % tag,
          scale=scale, xlabel=xlabel)

  # summary numbers
  eps = {}
  for tag in ["dog_pace", "dog_spin"]:
    p = os.path.join(eval_dir, tag, "summary.json")
    if not os.path.exists(p):
      continue
    with open(p) as fh:
      s = json.load(fh)
    eps[tag] = {
        "mean_reward_per_step": float(np.mean([e["mean_reward"]
                                               for e in s["episodes"]])),
        "mean_pose_err_rad": float(np.mean([e["mean_pose_err_rad"]
                                            for e in s["episodes"]])),
        "mean_root_err_m": float(np.mean([e["mean_root_err_m"]
                                          for e in s["episodes"]])),
        "max_root_err_m": float(np.max([e["max_root_err_m"]
                                        for e in s["episodes"]])),
        "mean_speed_mps": float(np.mean([e["mean_speed_mps"]
                                         for e in s["episodes"]])),
        "episodes": len(s["episodes"]),
        "steps_per_episode": int(s["episodes"][0]["steps"]),
        "env_time_step": s["env_time_step"],
    }
  data["eval"] = eps

  with open(os.path.join(ref_dir, "reference_summary.json")) as fh:
    data["reference"] = json.load(fh)

  with open(os.path.join(out, "figures_data.json"), "w") as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)
  print(json.dumps(data.get("eval", {}), indent=2, ensure_ascii=False))
  print("[figs] ->", out)


if __name__ == "__main__":
  main()
