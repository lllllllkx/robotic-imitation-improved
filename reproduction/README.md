# 论文复现产物（RSS 2020）

本目录是对 **Learning Agile Robotic Locomotion Skills by Imitating Animals**
（Peng et al., RSS 2020）的复现成果：一份可离线打开的自包含 HTML 报告、两段 20 s
回放视频、原始评估数据、参考动作 CSV，以及产出这些内容的脚本。

## 先看哪个

1. `report/复现报告.html` —— 主产物（8 章 / 8 表 / 8 图），直接双击用浏览器打开。
   报告里的视频用相对路径 `../outputs/eval/*.mp4` 引用，**需要保持本目录结构**才能播放。
2. `复现说明.md` —— 同内容的 Markdown 版，便于 diff 与快速浏览。
3. `outputs/eval/dog_pace.mp4`、`dog_spin.mp4` —— 官方预训练策略的单回合回放（20 s、640×480、30 fps）。
   `*_filmstrip.png` 是对应的关键帧拼图。

## 复现的关键数字（官方预训练策略，CPU，确定性 t=0 起点，600 步 = 20 s）

| 技能 | 每步奖励 | 关节误差 (rad) | 根误差 (m) | 偏航误差 (rad) | 水平速度 (m/s) |
| --- | --- | --- | --- | --- | --- |
| dog_pace | 0.7496 | 0.0651 | 0.0241 | 0.0558 | 1.0717 |
| dog_spin | 0.6378 | 0.0495 | 0.0913 | 0.8157 | 0.1372 |

- 论文 Table V（Robust）的口径是**仿真 + canonical 动力学 + 3 个独立训练策略求平均**，
  不是"每策略 5 回合"（那是 Table IV 的真机协议），所以 `0.820 / 0.839` 与上表**不可直接比**。
  本仓库改用随机相位 + 3 个评估初始条件的协议可得到 `dog_pace 0.7481 ± 0.0026`、
  `dog_spin 0.6980 ± 0.0074`（见 `../results/summary_table.md`），但仍不是论文口径：
  策略只有一份，也没有域随机化。
- 本节数字来自本仓库 `src/evaluate.py`（完整 yaw 公式）。早期版本的 `0.0662 / 0.8017`
  是**旧口径（`2·atan2(qz,qw)`）**，仅作历史记录保留在本目录的 CSV 与 HTML 报告里。
- 训练管线验证：8 进程 30 万步 PPO，实测 70.6 步/秒、耗时 70.8 min；自训练策略在评估中
  32 步摔倒（奖励 0.083、关节误差 0.612 rad、根误差 0.143 m）——属于"管线跑通、预算不足"，
  不作为策略质量结论。

## 目录

```
report/复现报告.html        自包含 HTML 报告（8 章 / 8 表 / 8 图）
report/figs/                报告图表的数据源与位图/SVG
复现说明.md                   Markdown 版复现说明
outputs/eval/*.mp4          dog_pace / dog_spin 单回合回放（各 20 s）
outputs/eval/*_filmstrip.png 关键帧拼图
outputs/eval/<tag>/summary.json   每回合指标汇总
outputs/eval/<tag>/steps.csv      [legacy] 逐帧原始数据；分项是在 env.step() 之后重算的，勿用于统计
outputs/eval/trained_dog_pace/    自训练 30 万步策略的评估结果
outputs/reference/*.csv     10 段参考动作的逐帧关节/根状态（由官方 .txt 转出）
outputs/train/dog_pace_300k/ 训练 30 万步的 episodes.csv、训练摘要与最终策略
scripts/                    产出以上内容的脚本（评估、参考动作导出、报告生成、视频重建）
```

## 如何重新生成

环境准备见根目录 `docs/REPRODUCTION.md`（Python 3.7 + TF 1.14 + pybullet 3.2.0）。核心命令：

```powershell
# 官方预训练策略评估（5 回合 × 600 步，同时落盘逐帧 csv 与 mp4）
python scripts/eval_pretrained.py --code_dir third_party/motion_imitation `
    --motion_file third_party/motion_imitation/data/motions/dog_pace.txt `
    --model_file third_party/motion_imitation/data/policies/dog_pace.zip `
    --out_dir outputs/eval --episodes 5 --max_steps 600 --tag dog_pace --video

# 参考动作导出（官方 .txt → 逐帧 csv）
python scripts/dump_reference_motion.py --code_dir third_party/motion_imitation `
    --motion_dir third_party/motion_imitation/data/motions --out_dir outputs/reference

# 报告图表与 HTML 重建
python scripts/make_report_assets.py --root . --train_dir outputs/train/dog_pace_300k --out_dir report/figs
python scripts/build_report.py
```

`rebuild_videos.py` 只用 `dog_pace/ep00` 帧重编码成 20 s 视频（避免把 5 个回合拼成 100 s），
脚本内的路径常量指向本目录的 `outputs/eval`，可直接运行或按需修改。

## 边界

**这批产物是 legacy 口径**：`outputs/eval/*/steps.csv`、HTML 报告与 `复现说明.md` 里的
奖励分项是在 `env.step()` 之后重算的（此时参考状态已推进），每步加权和与实际 `reward`
平均差 0.107；偏航数字用的是 `2·atan2(qz,qw)` 的旧公式。它们保留作为第二阶段的历史记录，
**不要**用它们做统计结论。修正后的指标与重跑结果在仓库根目录 `results/` 下，方法说明见
`../docs/IMPROVEMENTS.md` 的"`reproduction/` 目录里的旧产物"一节。

- 训练只做了 30 万步（约论文预算的 0.15%），用于验证管线与吞吐，不用于比较策略质量；
- 论文 4.2 节的潜在空间自适应 / AWR、Table IV 的真机实验在上游仓库中没有实现，本复现也没有补；
- 上游 `run.py` 的训练超参是 `4096 / 1 epoch / 1e-5`，与论文 Table II 的
  `10000 / 10 epochs / 2e-5` 不同，本复现沿用上游以便对齐。

改进工作（周期朝向重锚定、速度奖励去饱和、多 seed 协议）见上一级目录的
`docs/IMPROVEMENTS.md` 与 `src/`。
