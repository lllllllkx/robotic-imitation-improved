# 机器人模仿学习：论文复现 + 三处改进

对 RSS 2020 论文 **Learning Agile Robotic Locomotion Skills by Imitating Animals**
（Peng, Coumans, Zhang, Lee, Tan, Levine）的复现，以及在此基础上做的三处改进。

- 官方实现（`erwincoumans/motion_imitation`，commit `2f206ab`）以**未修改的副本**放在
  `third_party/motion_imitation/`，包含 10 段参考动作与 2 个官方预训练策略；
- 所有改进都在 `src/env_patch.py` 里以运行期 patch 实现，**开关默认关闭**，
  关闭时与上游逐位一致（可作为基线对照）；
- 无需重新训练即可验证改进 1（朝向重锚定）与改进 3（评估协议）。

## 复现基线（官方预训练策略，CPU）

| 技能 | 单回合每步奖励 | 关节误差 (rad) | 根误差 (m) | 偏航误差 (rad) | 速度 (m/s) |
| --- | --- | --- | --- | --- | --- |
| dog_pace | 0.7496 | 0.0651 | 0.0241 | 0.0662 | 1.074 |
| dog_spin | 0.6378 | 0.0495 | 0.0913 | 0.8017 | — |

（确定性 t=0 起点、单回合 600 步，与上游 `eval_pretrained.py` 逐位一致；跨进程重复运行数值相同。）

论文 Table V（Robust，3 seeds × 5 回合）为 `dog_pace 0.820`、`dog_spin 0.839`；差距来自
随机相位起点、观测噪声与域随机化，属于评估口径差异，不是实现错误（详见
`docs/REPRODUCTION.md`）。

## 复现产物（报告 / 视频 / 原始数据）

复现阶段的完整交付放在 **[reproduction/](reproduction/)**，可离线浏览：

- `reproduction/report/复现报告.html` —— 自包含 HTML 报告（8 章 / 8 表 / 8 图），
  内含两段回放视频（相对路径引用，需保持目录结构）；
- `reproduction/outputs/eval/dog_pace.mp4`、`dog_spin.mp4` —— 各 20 s、640×480、30 fps，
  另有对应的关键帧拼图；
- `reproduction/outputs/eval/*/summary.json` 与 `steps.csv` —— 逐回合汇总与逐帧原始数据；
- `reproduction/outputs/reference/*.csv` —— 10 段参考动作的逐帧关节/根状态；
- `reproduction/outputs/train/dog_pace_300k/` —— 30 万步 PPO 的 `episodes.csv`、训练摘要与最终策略；
- `reproduction/scripts/` —— 产出以上内容的脚本；用法见 `reproduction/README.md`。

## 三处改进

| # | 改进 | 一句话 | 量化结果（3 seeds × 3 回合，随机相位） |
| --- | --- | --- | --- |
| 1 | 周期朝向重锚定 `--yaw-anchor` | 上游每个步态周期只重锚定参考**位置**，朝向仅在切换动作时同步，导致循环转向技能偏航无界累积 | `dog_spin` 奖励 0.6980 → **0.7602**（+8.9%），偏航误差 0.2510 → **0.0473 rad**（−81%）；`dog_pace` 偏航误差 0.0751 → 0.0286（−62%） |
| 2 | 速度奖励去饱和 `--reward-v2` | 上游 `exp(-0.1·ΣΔdq²)` 在 12 关节上求和，实测均值仅 0.0157、p90 1.5e-4，梯度消失 | 该项均值 0.0157 → **0.3376**（确定性回合）/ 0.0026 → **0.1909**（随机相位）；权重和仍为 1 |
| 3 | 多 seed / 随机相位协议 `--random-phase --seeds` | 上游评估是"5 回合同一确定性回放"，无误差棒 | 确定性口径把 `dog_spin` 低估 9.5%（0.6378 vs 0.6980 ± 0.0074）；并给出偏航漂移斜率以区分有界/无界误差 |

完整动机、代码位置、实测数字与边界说明见 **[docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md)**。

## 安装

```powershell
conda env create -f environment.yml     # python 3.7 + TF 1.14 + pybullet 3.2.0
conda activate motion_imitation
# 或手动：pip install -r requirements.txt
```

TensorFlow 1.14 是硬约束（上游用的是 `tf.contrib` 与 stable-baselines 2.10）。
Windows 上跑多进程训练还需 MS-MPI；评估单进程即可，**不需要 GPU**。

## 用法

```powershell
# 基线（上游口径：确定性 t=0 起步，5 回合）
python -m src.evaluate --motion dog_pace --episodes 5 --max-steps 600

# 论文式协议：随机相位 + 3 seeds × 3 回合，输出 mean ± std
python -m src.evaluate --motion dog_spin --episodes 3 --seeds 3 --random-phase

# 改进 1：每个步态周期重锚定参考朝向
python -m src.evaluate --motion dog_spin --episodes 3 --seeds 3 --random-phase --yaw-anchor

# 改进 2：速度项去饱和 + 权重重配平（只改奖励读数，不改变策略行为）
python -m src.evaluate --motion dog_pace --episodes 3 --seeds 3 --random-phase --reward-v2

# 相位栅格（确定性消融）
python -m src.evaluate --motion dog_pace --start-phases "0,0.25,0.5,0.75"

# 汇总成表
python -m src.aggregate
```

每条命令输出 `results/eval/<技能>__<变体>/summary.json`，包含每回合的 5 个奖励分项、
关节/根/偏航误差、漂移斜率、速度、存活步数与 `reward_matches_terms` 自洽校验。
一键跑完整矩阵：`scripts/run_protocol.ps1`。

## 目录

```
third_party/motion_imitation/   官方实现原样副本（Apache-2.0），含 motions / policies
src/env_patch.py                三处改进的运行期 patch（默认关闭）
src/metrics.py                  归一化回报、窗口均值、漂移斜率、多回合聚合
src/evaluate.py                 评估 CLI（多 seed / 随机相位 / 相位栅格）
src/aggregate.py                汇总 results/eval/* → summary_table.csv/.md
tests/                          纯 numpy 契约测试（CI 可跑，不需要 TF/pybullet）
configs/                        报告所用实验配置的镜像（评估协议、奖励权重）
docs/IMPROVEMENTS.md            改进的动机 / 实现 / 数字 / 边界
docs/REPRODUCTION.md            环境、命令、核对清单
scripts/run_protocol.ps1        完整对比矩阵的顺序执行脚本
reproduction/                   复现产物：HTML 报告、回放视频、原始数据与脚本
```

## 结果表（本仓库实测，可重跑）

| 技能 | 变体 | 回合 | 每步奖励 | 关节误差 | 根误差 | 偏航误差 | 漂移 (rad/s) | 速度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dog_pace | 上游基线 | 9 | 0.7481 ± 0.0026 | 0.0651 | 0.0261 | 0.0751 | −0.0006 | 1.074 |
| dog_pace | + 朝向重锚定 | 9 | 0.7521 ± 0.0008 | 0.0649 | 0.0239 | 0.0286 | −0.0008 | 1.079 |
| dog_pace | + 速度项 v2 | 9 | 0.7222 ± 0.0027 | 0.0651 | 0.0261 | 0.0751 | −0.0006 | 1.074 |
| dog_spin | 上游基线 | 9 | 0.6980 ± 0.0074 | 0.0502 | 0.0679 | 0.2510 | 0.0038 | 0.160 |
| dog_spin | + 朝向重锚定 | 9 | 0.7602 ± 0.0014 | 0.0475 | 0.0660 | 0.0473 | −0.0007 | 0.153 |

`+ 速度项 v2` 那一行的行为指标与基线完全一致（同一条轨迹），只有奖励读数变化：
速度项均值从 0.0026 升到 0.1909，奖励从 0.7481 降到 0.7222（姿态项权重让出 0.05）。

（`results/summary_table.md` 由 `src.aggregate` 自动生成，与上表同源。所有数字在固定
线程数 + 确定性相位采样下跨进程可复现：同一条命令跑两次，逐回合数值完全相同。）

## 没做的事（不要当成已完成）

- **没有跑全量 PPO 训练**：论文每技能约 2e8 采样，本机无 GPU、实测 70 步/秒，
  单技能单种子约 33 天。改进 2 属于训练侧改动，目前只有"奖励项不再饱和"的证据，
  没有"重训后策略更好"的证据；
- **没有实现潜在空间自适应 / AWR**（论文 4.2 节）：上游仓库也没有实现；
- **没有真机实验**（论文 Table IV）；
- 训练超参沿用上游 `run.py` 的 `4096 / 1 epoch / 1e-5`，与论文 Table II 的
  `10000 / 10 epochs / 2e-5` 不同。

## 引用与许可

```bibtex
@inproceedings{peng2020imitation,
  title={Learning Agile Robotic Locomotion Skills by Imitating Animals},
  author={Peng, Xue Bin and Coumans, Erwin and Zhang, Tingnan and Lee, Tsang-Wei and Tan, Jie and Levine, Sergey},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2020}
}
```

`third_party/motion_imitation/` 及随附动作/策略数据为 Apache-2.0（见 `LICENSE.txt`），
本仓库新增代码同样以 Apache-2.0 发布，出处见 `NOTICE`。
