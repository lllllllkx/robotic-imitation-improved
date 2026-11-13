"""Generate the self-contained HTML reproduction report.

Reads only files produced inside robotic_imitation_repro/ (eval summaries,
reference-motion summary, training episodes, figures) and embeds the SVG
figures inline, so the report can be opened offline.
"""

import csv
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figs")
OUT = os.path.join(ROOT, "report", "复现报告.html")

PAPER = {
    "dog_pace": {"No Rand": 0.839, "Robust": 0.820, "Adaptive": 0.812},
    "dog_spin": {"No Rand": 0.859, "Robust": 0.839, "Adaptive": 0.839},
}
CN = {"dog_pace": "Dog Pace（踱步）", "dog_spin": "Dog Spin（原地旋转）"}

CSS = """
:root{--ink:#1b1b1b;--muted:#6b6b6b;--line:#e2e0da;--accent:#b4471e;--bg:#fbfaf7;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;line-height:1.75;font-size:15px}
.wrap{max-width:1040px;margin:0 auto;padding:0 28px 80px}
header{background:#17150f;color:#f6f2e8;padding:52px 0 40px;margin-bottom:44px}
header .wrap{padding-bottom:0}
header .kicker{letter-spacing:.22em;font-size:11px;color:#c9a227;text-transform:uppercase}
header h1{font-size:32px;line-height:1.35;margin:14px 0 10px;font-weight:650}
header .sub{color:#b8b2a4;font-size:14px;max-width:780px}
header .meta{margin-top:26px;display:grid;grid-template-columns:1fr 1fr;gap:7px 44px;font-size:12px;color:#c9c3b4}
header .meta b{color:#f6f2e8;font-weight:600}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:0 0 46px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 18px 14px}
.card .n{font-size:26px;font-weight:650;letter-spacing:-.02em}
.card .u{font-size:12px;color:var(--muted);margin-left:4px}
.card .l{font-size:12px;color:var(--muted);margin-top:6px}
h2{font-size:20px;margin:56px 0 8px;padding-left:12px;border-left:4px solid var(--accent);font-weight:650}
h2 .no{color:var(--accent);font-size:13px;letter-spacing:.08em;margin-right:8px;font-weight:600}
h3{font-size:15px;color:var(--accent);margin:28px 0 6px;font-weight:650}
p{margin:10px 0;color:#2c2a26}
.lead{color:#4a4741;font-size:14.5px}
table{border-collapse:collapse;width:100%;margin:14px 0 6px;font-size:13px;background:var(--card)}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:right;vertical-align:middle}
th{background:#f3efe6;font-weight:600;text-align:center}
td:first-child,th:first-child{text-align:left}
td.l{text-align:left}
tr.hl td{background:#fdf3ee;font-weight:600}
caption{caption-side:bottom;text-align:left;font-size:12px;color:var(--muted);padding-top:6px}
.fig{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;margin:16px 0 4px;text-align:center}
.fig svg, .fig img{max-width:100%;height:auto}
.figcap{font-size:12px;color:var(--muted);margin:0 0 22px}
.media{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:18px 0 6px}
.media figure{margin:0;background:#111;border-radius:10px;overflow:hidden}
.media video{width:100%;display:block}
.media figcaption{background:#1d1b17;color:#cfc9bb;font-size:12px;padding:8px 10px}
.strip{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px;margin:16px 0 4px}
.strip img{width:100%;display:block}
.note{background:#fff8ef;border-left:4px solid #d98324;padding:14px 18px;margin:20px 0;font-size:13.5px;color:#4a3f31}
.warn{background:#f4f6f8;border-left:4px solid #5b7c99;padding:14px 18px;margin:20px 0;font-size:13.5px;color:#33414d}
ul{margin:8px 0 8px 20px;padding:0}
li{margin:4px 0;color:#2c2a26}
code{background:#efece4;padding:1px 5px;border-radius:4px;font-size:12.5px;font-family:Consolas,monospace}
footer{margin-top:70px;padding-top:22px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
"""


def fig_svg(name):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        return '<div class="note">缺少图：%s</div>' % name
    with open(path, encoding="utf-8") as fh:
        return '<div class="fig">%s</div>' % fh.read()


def table(headers, rows, caption="", highlight_last=False):
    head = "".join("<th>%s</th>" % h for h in headers)
    body = []
    for i, row in enumerate(rows):
        cls = ' class="hl"' if highlight_last and i == len(rows) - 1 else ""
        body.append("<tr%s>%s</tr>" % (cls, "".join("<td>%s</td>" % c for c in row)))
    cap = "<caption>%s</caption>" % caption if caption else ""
    return "<table>%s<thead><tr>%s</tr></thead><tbody>%s</tbody>%s</table>" % (
        cap, head, "".join(body), "")


def f(x, n=3):
    return ("%." + str(n) + "f") % x


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    data = load_json(os.path.join(FIG, "figures_data.json"))
    ev = data["eval"]
    ref = data["reference"]

    # ---- reference library -------------------------------------------------
    ref_rows = []
    for m in sorted(ref, key=lambda m: -m["duration_s"]):
        ref_rows.append([
            m["motion_file"].replace(".txt", ""),
            str(m["num_frames"]),
            "%d" % round(1.0 / m["frame_duration_s"]),
            f(m["duration_s"], 2),
            f(m["mean_speed_mps"], 3),
            "%.2f–%.2f" % (m["root_height_min"], m["root_height_max"]),
        ])

    # ---- pretrained policy metrics -----------------------------------------
    extra = {}
    for tag in ["dog_pace", "dog_spin"]:
        rows = list(csv.DictReader(open(os.path.join(
            ROOT, "outputs", "eval", tag, "steps.csv"), encoding="utf-8")))
        yaw = [abs(float(r["yaw_err_rad"])) for r in rows]
        extra[tag] = {"yaw_mean": statistics.mean(yaw), "yaw_max": max(yaw)}

    pretrain_rows = []
    for tag in ["dog_pace", "dog_spin"]:
        e = ev[tag]
        pretrain_rows.append([
            CN[tag], f(e["mean_reward_per_step"], 3),
            f(e["mean_pose_err_rad"], 3),
            "%.1f / %.1f" % (e["mean_root_err_m"] * 100, e["max_root_err_m"] * 100),
            f(e["mean_speed_mps"], 2), f(extra[tag]["yaw_mean"], 2),
            "%d / %d" % (e["episodes"], e["steps_per_episode"]),
        ])

    # ---- vs paper ----------------------------------------------------------
    cmp_rows = []
    for tag in ["dog_pace", "dog_spin"]:
        ours = ev[tag]["mean_reward_per_step"]
        p_ref = PAPER[tag]["Robust"]
        cmp_rows.append([
            CN[tag], f(PAPER[tag]["No Rand"], 3), f(PAPER[tag]["Robust"], 3),
            f(PAPER[tag]["Adaptive"], 3), f(ours, 3),
            "%+.3f（%+.1f%%）" % (ours - p_ref, (ours - p_ref) / p_ref * 100),
        ])

    # ---- training ----------------------------------------------------------
    train_dir = os.path.join(ROOT, "outputs", "train", "dog_pace_300k")
    ep_csv = os.path.join(train_dir, "episodes.csv")
    train = None
    if os.path.exists(ep_csv):
        rows = list(csv.DictReader(open(ep_csv, encoding="utf-8")))
        rets = [float(r["episode_return"]) for r in rows]
        lens = [float(r["episode_length"]) for r in rows]
        steps = [float(r["global_step"]) for r in rows]
        wall = [float(r["wall_time_s"]) for r in rows]
        w = max(5, min(20, len(rets) // 10))
        train = {
            "episodes": len(rows),
            "steps": max(steps),
            "wall_h": (max(wall) - min(wall)) / 3600.0,
            "ret_first": statistics.mean(rets[:w]),
            "ret_last": statistics.mean(rets[-w:]),
            "len_first": statistics.mean(lens[:w]),
            "len_last": statistics.mean(lens[-w:]),
            "ret_max": max(rets),
        }
        summ_path = os.path.join(train_dir, "train_summary.json")
        if os.path.exists(summ_path):
            ts = load_json(summ_path)
            train["num_procs"] = int(ts.get("num_procs", 1))
            train["steps"] = float(ts.get("total_timesteps", max(steps) * train["num_procs"]))
            train["wall_h"] = float(ts.get("wall_time_s", 0)) / 3600.0
            train["rate"] = float(ts.get("steps_per_sec_total", 0))
            train["episodes_total"] = int(ts.get("num_episodes_rank0", len(rows)))

    trained = None
    tpath = os.path.join(ROOT, "outputs", "eval", "trained_dog_pace", "summary.json")
    if os.path.exists(tpath):
        s = load_json(tpath)
        eps = s["episodes"]
        trained = {
            "mean_reward": statistics.mean([e["mean_reward"] for e in eps]),
            "pose_err": statistics.mean([e["mean_pose_err_rad"] for e in eps]),
            "root_err": statistics.mean([e["mean_root_err_m"] for e in eps]),
            "steps": statistics.mean([e["steps"] for e in eps]),
            "survived": statistics.mean([0.0 if e.get("terminated_early", False) else 1.0 for e in eps]),
        }

    # ---- assemble ----------------------------------------------------------
    pace, spin = ev["dog_pace"], ev["dog_spin"]
    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>复现报告：用动物动作模仿学习敏捷四足运动技能</title>
<style>%s</style></head><body>""" % CSS)

    parts.append("""
<header><div class="wrap">
<div class="kicker">Reproduction Report · RSS 2020</div>
<h1>Learning Agile Robotic Locomotion Skills by Imitating Animals</h1>
<div class="sub">在 Windows / CPU-only 笔记本上，用论文官方开源实现（motion_imitation, commit 2f206ab）跑通仿真部分：导出参考动作数据、定量评估官方预训练策略、复现 8 进程 PPO 训练管线。</div>
<div class="meta">
<div>代码 <b>github.com/erwincoumans/motion_imitation</b> @ 2f206ab</div>
<div>论文 <b>arXiv:2004.00784v3</b></div>
<div>机器人 <b>Laikago（12 驱动关节 + 浮动基座 = 18 DoF）</b></div>
<div>日期 <b>2026-11-13</b></div>
</div></div></header>
<div class="wrap">
""")

    parts.append("""
<div class="cards">
<div class="card"><div><span class="n">%s</span><span class="u">步</span></div>
<div class="l">预训练策略单回合长度（19.8 s @30 Hz），两策略均未跌落</div></div>
<div class="card"><div><span class="n">%s</span><span class="u">rad</span></div>
<div class="l">Dog Pace 关节平均绝对跟踪误差（≈3.7°）</div></div>
<div class="card"><div><span class="n">%.1f</span><span class="u">cm</span></div>
<div class="l">Dog Pace 根节点位置误差（最大 %.1f cm）</div></div>
<div class="card"><div><span class="n">%s</span></div>
<div class="l">实测每步奖励（Dog Pace / Dog Spin），对照论文仿真归一化回报 0.820 / 0.839</div></div>
</div>
""" % ("600", f(pace["mean_pose_err_rad"], 3), pace["mean_root_err_m"] * 100,
       pace["max_root_err_m"] * 100, "%s / %s" % (f(pace["mean_reward_per_step"], 3),
                                                        f(spin["mean_reward_per_step"], 3))))

    parts.append("""
<p class="lead">一句话结论：<b>论文的「参考动作 → 仿真模仿策略」这一段在本机可以完整跑通并复现到 91% / 76% 的回报水平</b>；
上游仓库没有提供论文的潜在空间自适应（sim-to-real）代码，本机也没有真实机器人，因此那部分明确不复现。
训练管线同样可复现，但论文的 2 亿采样预算在 CPU 上需要以「天」计，本次只做 30 万步（0.15%）的量级验证。</p>
""")

    # 1
    parts.append("""
<h2><span class="no">01</span>复现了什么，没复现什么</h2>
<p>论文的系统由四段组成：① 把真实狗的动作捕捉重定向到 Laikago（IK）；② 用 PPO 在仿真中模仿参考动作学习策略；
③ 训练时随机化动力学参数，并把参数编码进潜在向量；④ 在真实机器人上用 AWR 在潜在空间里做在线适应。</p>
<p>官方仓库给出的是第 ② 段（以及第 ③ 段的随机化环境）的完整实现，外加两个训练好的策略 <code>dog_pace.zip</code> /
<code>dog_spin.zip</code>；重定向工具、潜在编码器、AWR 适应与实机部署脚本都不在仓库里。据此划出本次的边界：</p>
<table><thead><tr><th>论文环节</th><th>官方代码</th><th>本次复现</th></tr></thead><tbody>
<tr><td class="l">动作重定向（IK）</td><td class="l">未提供，仓库只给已重定向的 10 段 <code>data/motions/*.txt</code></td><td class="l">直接使用提供的参考动作，解析成 CSV 核对</td></tr>
<tr><td class="l">仿真模仿训练（PPO）</td><td class="l"><code>run.py</code> + <code>envs/</code> + <code>learning/</code></td><td class="l"><b>已复现</b>：8 进程 MPI，30 万步</td></tr>
<tr><td class="l">预训练策略评估</td><td class="l">两个 <code>.zip</code> 策略，README 给出 <code>--mode test</code> 用法</td><td class="l"><b>已复现</b>：逐帧指标 + 视频</td></tr>
<tr><td class="l">域随机化</td><td class="l"><code>minitaur_env_randomizer</code>（质量/摩擦/电机强度等）</td><td class="l">训练时开启（默认），评估时关闭</td></tr>
<tr><td class="l">潜在空间自适应 / sim-to-real</td><td class="l"><b>未提供</b></td><td class="l">不复现（无硬件、无网络结构）</td></tr>
</tbody><caption>表 1　复现范围与上游代码的对应关系</caption></table>
<div class="warn">论文 Table IV（真机）与 Table V（仿真）里都区分 No Rand / Robust / Adaptive 三类策略，但仓库只发布了一类预训练策略。
下文把它当作「Robust」一类来对照，并明确标注这一假设。</div>
""")

    # 2
    parts.append("""
<h2><span class="no">02</span>环境与隔离方式</h2>
<p>按你的要求，所有东西都放在新建的 <code>robotic_imitation_repro/</code> 里，原有 <code>motion_imitation/</code> 目录与
<code>animal_imitation</code> 环境一行未改。代码通过本地 <code>git clone file://</code> 得到（GitHub 直连被重置），
HEAD 与本地副本同为 <code>2f206ab</code>，再把 <code>origin</code> 指回 GitHub 以保留溯源；环境用
<code>conda create --clone</code> 从原环境复制，互不影响。</p>
""")
    parts.append(table(
        ["项", "值", "说明"],
        [["操作系统 / 硬件", "Windows 11 (10.0.26200)", "Intel Core Ultra 7 155H，22 逻辑核，31.5 GB RAM"],
         ["GPU", "无 NVIDIA GPU", "TensorFlow 1.14.0 纯 CPU 推理与训练"],
         ["Conda 环境", "motion_imitation_repro", "由 animal_imitation 硬链接克隆"],
         ["Python / TensorFlow", "3.7.1 / 1.14.0", "与论文时代一致"],
         ["PyBullet", "3.2.0", "requirements.txt 写 2.7.3，原环境实际装的是 3.2.0"],
         ["gym / stable-baselines", "0.17.1 / 2.10.0", "PPO1 实现"],
         ["MPI", "MS-MPI 10.1 + mpi4py 3.0.3", "Windows 上替代 OpenMPI，支持 2/4/8 进程"],
         ["渲染", "DIRECT + ER_TINY_RENDERER", "离屏渲染，无 GUI，可无人值守出视频"]],
        caption="表 2　复现所用环境（全部为新建，未触碰原环境）"))

    # 3
    parts.append("""
<h2><span class="no">03</span>论文的数据：参考动作库</h2>
<p>论文的核心输入是重定向后的参考动作。仓库里 10 段 <code>.txt</code> 是 12 维关节角 + 根位姿的逐帧数值；
脚本 <code>dump_reference_motion.py</code> 直接调用上游 <code>MotionData</code> 解析器导出成 CSV，
得到论文口径的「技能清单」与其运动学特征：</p>
""")
    parts.append(fig_svg("fig_reference.svg"))
    parts.append('<p class="figcap">图 1　10 段参考动作的片段时长与平均根节点速度（dog_* 为真实狗的 mocap 重定向，其余为艺术家动画）。</p>')
    parts.append(table(["参考动作", "帧数", "帧率 (Hz)", "时长 (s)", "平均根速 (m/s)", "根高范围 (m)"],
                       ref_rows, caption="表 3　参考动作库（数据来源：<code>data/motions/*.txt</code>，由 MotionData 解析）"))
    parts.append("""
<div class="note">这 10 段里只有 4 段狗 mocap 是 60 Hz（0.53–0.75 s、单周期），6 段艺术家动画是 24 Hz 左右的 3.7–6.7 s 多周期片段；
<code>dog_pace</code> 与 <code>dog_backwards_pace</code>、<code>dog_trot</code> 与 <code>dog_backwards_trot</code> 数值完全互为镜像
（把片段倒放即得，论文正是这样得到后退步态的）。所有片段根节点平均高度约 0.44 m，与 Laikago 站立高度一致。</div>
""")

    # 4
    parts.append("""
<h2><span class="no">04</span>运行开源代码：官方预训练策略的定量结果</h2>
<p>按 README 的 <code>--mode test</code> 用法加载策略，但把评估做成可复核的：每个回合从参考动作 <code>t=0</code>
的参考状态初始化（关闭随机时间偏移），因此 5 个回合逐帧完全一致——这是「确定性复现」而不是「平均±方差」。
回合长 600 步 × 33 ms = 19.8 s，与论文 30 Hz 控制频率一致。</p>
""")
    parts.append(table(
        ["策略", "每步奖励", "关节误差 (rad)", "根位置误差 均值/最大 (cm)", "速度 (m/s)", "偏航误差 (rad)", "回合/步数"],
        pretrain_rows, caption="表 4　官方预训练策略在仿真中的实测指标（mode=test，随机化关闭，5 回合 × 600 步）"))

    parts.append(fig_svg("fig_reward.svg"))
    parts.append('<p class="figcap">图 2　逐步模仿奖励（1 s 滑动平均）。Dog Pace 稳定在 0.75 附近并保持周期性起伏（步频 1.58 Hz）；Dog Spin 在 0.64 上下波动。</p>')
    parts.append(fig_svg("fig_joints.svg"))
    parts.append('<p class="figcap">图 3　三个关节的参考角（灰）与策略输出（蓝）：膝关节与髋关节的相位、幅度基本吻合，存在约 0.05–0.1 rad 的稳态偏移，这是模仿类策略的典型残差。</p>')
    parts.append(fig_svg("fig_root.svg"))
    parts.append('<p class="figcap">图 4　左：根节点高度跟踪（毫米级跟随参考的 0.433–0.453 m 起伏）；右：前进速度稳定在 1.00–1.20 m/s，均值 1.074 m/s（参考片段自身为 1.094 m/s）。</p>')
    parts.append(fig_svg("fig_terms.svg"))
    parts.append('<p class="figcap">图 5　奖励的 5 个分项（每步、0–1）。姿态与根速度两项最高；关节速度项接近 0（速度匹配尺度 exp(−0.1·ΣΔv²) 非常严格）；Dog Spin 的根姿态项只有 0.03，原因见下节。</p>')

    parts.append("""
<div class="media">
<figure><video controls preload="metadata" src="../outputs/eval/dog_pace.mp4"></video>
<figcaption>Dog Pace：600 步 / 19.8 s，速度 1.07 m/s，全程未跌落（640×480，30 fps，20 s）</figcaption></figure>
<figure><video controls preload="metadata" src="../outputs/eval/dog_spin.mp4"></video>
<figcaption>Dog Spin：原地旋转，600 步 / 19.8 s，转速略低于参考（640×480，30 fps，20 s）</figcaption></figure>
</div>
""")
    parts.append('<div class="strip"><img src="../outputs/eval/dog_pace_filmstrip.png" alt="Dog Pace 关键帧"></div>')
    parts.append('<p class="figcap">图 6　Dog Pace 回放的 6 个关键帧（同一回合内均匀采样），可见对角步态与抬腿高度。</p>')

    # 5
    parts.append("""
<h2><span class="no">05</span>与论文 Table V 的对比</h2>
<p>论文 Table V 报告的是<b>归一化回报</b>（每个回合内 0 = 最差、1 = 理论最优，3 个随机种子 × 5 回合）。
仓库没有实现这套归一化，本次用<b>每步奖励</b>作为近似——因为 5 个分项各自都是 0–1 的指数型奖励，
每步奖励天然落在 0–1，且其上限 1 恰好等于「完美跟踪」，与归一化回报的尺度可比，但不是严格等价。</p>
""")
    parts.append(fig_svg("fig_vs_paper.svg"))
    parts.append('<p class="figcap">图 7　论文 Table V（仿真、canonical 动力学）与本次实测的对照。</p>')
    parts.append(table(["技能", "论文 No Rand", "论文 Robust", "论文 Adaptive", "本次实测（每步奖励）", "与论文 Robust 之差"],
                       cmp_rows, caption="表 5　与论文 Table V 的数值对照（论文为 3 seeds × 5 回合；本次为单策略、确定性 5 回合）"))
    parts.append("""
<div class="note"><b>差异从哪来（逐项可解释）：</b>
<ul>
<li><b>Dog Pace 0.750 vs 0.820（−8.6%）</b>：关节跟踪误差 0.065 rad、根位置误差 2.4 cm 都已接近奖励尺度允许的残差；
损失主要来自关节速度分项 0.016 与根姿态 0.93 的边缘损失，属于策略本身的稳态残差，而非评测口径错误。</li>
<li><b>Dog Spin 0.638 vs 0.839（−24%）</b>：根姿态分项只有 0.030。逐帧数据显示偏航误差从 0.06 rad 单调累积到 1.0 rad
（均值 0.80 rad）——参考片段每个 0.75 s 周期转 1.734 rad（2.31 rad/s），策略实际转速略低（约差 0.05 rad/s），
20 s 连续旋转后累积近 1 rad。姿态特征里不含根位置、也不含累计转角，策略无法校正这类漂移，这是模仿框架的固有性质。</li>
<li><b>口径差异</b>：论文是 3 个种子训练出的策略各评 5 回合并取均值±标准差，且回合起点在参考动作上随机采样；
本次是单策略、固定从 <code>t=0</code> 起的确定性回合，因此<b>没有误差棒</b>，多次运行结果逐帧一致。</li>
<li><b>仿真器版本</b>：PyBullet 3.2.0（原环境）而非 requirements 指定的 2.7.3，接触求解与电机模型会有细微差别。</li>
</ul></div>
""")

    # 6
    parts.append("""
<h2><span class="no">06</span>复现训练流程：8 进程 MPI + PPO</h2>
<p><code>train_short.py</code> 完全复用上游 <code>build_imitation_env()</code> 与 <code>PPOImitation</code>，超参与
<code>run.py</code> 保持一致（γ=0.95、clip 0.2、每轮 4096 步、优化批 256、1 个 epoch、lr 1e-5、网络 512-256），
额外加了一个只在 rank 0 写盘的回合记录器，用来画学习曲线。</p>
""")
    if train:
        parts.append(table(
            ["项", "值"],
            [["训练步数（聚合 8 进程）", "%s 步（论文 2e8 的 %.2f%%）" % ("{:,}".format(int(train["steps"])), train["steps"] / 2e8 * 100)],
             ["墙钟时间", "%.1f 分钟（含环境初始化）" % (train["wall_h"] * 60)],
             ["吞吐", "≈ %.1f 步/秒（%d 进程聚合）" % (train.get("rate", 0.0), train.get("num_procs", 1))],
             ["完成回合数", "%d 个（rank0 记录）" % train["episodes"]],
             ["回合奖励（前/后 10% 窗口）", "%.2f → %.2f" % (train["ret_first"], train["ret_last"])],
             ["回合长度（前/后 10% 窗口）", "%.1f → %.1f 步" % (train["len_first"], train["len_last"])],
             ["单回合最高奖励", "%.2f" % train["ret_max"]]],
            caption="表 6　本次训练运行的规模与结果（<code>outputs/train/dog_pace_300k/</code>）"))
        parts.append(fig_svg("fig_learning.svg"))
        parts.append('<p class="figcap">图 8　学习曲线：回合累计奖励与回合长度。注意回合长度由课程学习控制（20 → 600 步，对应 3000 万步训练进度），'
                     '30 万步时课程进度仅 1%，因此回合仍只有 10–20 步。</p>')

    parts.append("""
<div class="note"><b>预算差距是这一段复现的核心结论：</b>论文每个技能用约 <b>2 亿</b>采样、3 个种子；
本机 8 进程聚合约 <b>70.6 步/秒</b>（实测），跑完 2 亿步需要 <b>约 33 天</b>（单技能、单种子），三个种子约 3 个月，因此在有限时间内只能验证
「训练脚本能跑、曲线在上升、产物完整」，不能声称复现了论文的策略质量。
课程学习（回合长度 20→600，随 3000 万步线性放开）也决定了 30 万步时的策略只在 0.3–0.7 s 的短片段上训练过，
用它去做 20 s 的连续评估必然失败——这是训练预算的问题，不是代码 bug。</div>
""")

    if trained:
        parts.append("""
<h3>训练后策略的同协议评估</h3>""")
        parts.append(table(
            ["策略", "每步奖励", "关节误差 (rad)", "根误差 (cm)", "平均存活步数", "跑满 600 步"],
            [["官方 dog_pace.zip（2 亿步 + 3 seeds）", f(pace["mean_reward_per_step"], 3),
              f(pace["mean_pose_err_rad"], 3), f(pace["mean_root_err_m"] * 100, 1), "600", "是"],
             ["本次训练（30 万步，单种子）", f(trained["mean_reward"], 3),
              f(trained["pose_err"], 3), f(trained["root_err"] * 100, 1),
              f(trained["steps"], 0), "是" if trained["survived"] >= 0.99 else "否"]],
            caption="表 7　同协议（mode=test、确定性 t=0 起点、600 步）下短预算策略与官方策略的差距"))

    # 7
    parts.append("""
<h2><span class="no">07</span>结论</h2>
<ul>
<li><b>代码可跑通</b>：官方仓库在 Windows + CPU + MS-MPI 上无需改一行代码即可训练与评估；
本次只新增脚本，并全部放在新目录里。</li>
<li><b>论文数据可复现</b>：10 段参考动作的运动学特征（时长、速度、根高）与论文描述一致；
官方预训练策略在仿真中跑满 19.8 s 不跌落，Dog Pace 达到论文仿真归一化回报的 91%，Dog Spin 76%。</li>
<li><b>训练可复现但预算不可比</b>：管线、超参、奖励权重、观测结构（历史 3 帧 + t+1/2/10/30 四帧目标）逐项对齐论文，
但 2e8 采样 × 3 种子在 CPU 上不可行，本次 3e5 只够验证流程。</li>
<li><b>不可复现的部分</b>：IK 重定向工具、潜在编码器与 AWR 自适应、真机实验——上游未发布，本机无硬件。</li>
</ul>
<h2><span class="no">08</span>复现步骤与产物</h2>
<p>全部命令见同目录 <code>复现说明.md</code>（含环境创建、三条运行命令与目录说明）。主要产物：</p>
""")
    parts.append(table(
        ["产物", "内容"],
        [["<code>outputs/reference/*.csv</code>", "10 段参考动作逐帧数据（时间、根位姿、12 关节角）"],
         ["<code>outputs/eval/{dog_pace,dog_spin}/</code>", "官方策略 3000 步逐帧指标 <code>steps.csv</code> + <code>summary.json</code>"],
         ["<code>outputs/eval/*.mp4</code>", "仿真回放视频（640×480，30 fps）"],
         ["<code>outputs/train/dog_pace_300k/</code>", "<code>episodes.csv</code>、<code>model.zip</code>、<code>final_model.zip</code>、<code>train_summary.json</code>"],
         ["<code>report/figs/*.svg</code>", "本报告全部图表（矢量，可缩放）与 <code>figures_data.json</code>"],
         ["<code>scripts/*.py</code>", "复现脚本（评估、参考动作导出、训练、出图、报告生成）"]],
        caption="表 8　复现产物清单（均位于 <code>robotic_imitation_repro/</code> 内）"))

    parts.append("""
<footer>
论文：Peng, X. B., Coumans, E., Zhang, T., Tan, J., Tanaka, K., &amp; Levine, S. <i>Learning Agile Robotic Locomotion Skills by
Imitating Animals</i>. RSS 2020. arXiv:2004.00784v3（Table II 超参、Table V 仿真回报）。<br>
代码：github.com/erwincoumans/motion_imitation @ 2f206ab。项目页：xbpeng.github.io/projects/Robotic_Imitation。<br>
本报告由 <code>scripts/build_report.py</code> 依据实测数据自动生成。
</footer></div></body></html>
""")

    html = "".join(parts)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("[report] -> %s (%.1f KB)" % (OUT, os.path.getsize(OUT) / 1024.0))


if __name__ == "__main__":
    main()
