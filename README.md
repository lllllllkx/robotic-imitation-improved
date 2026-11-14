# 机器人模仿学习：论文仿真复现 + 三处改进

**定位**：这是 RSS 2020 论文 **Learning Agile Robotic Locomotion Skills by Imitating Animals**
（Peng, Coumans, Zhang, Lee, Tan, Levine）的**仿真复现**，以及在此基础上做的三处改进研究。
它不是论文全部实验的复现——潜在空间自适应 / AWR、真机迁移、全量 PPO 训练都没有实现（见文末）。

- 官方实现（`erwincoumans/motion_imitation`，commit `2f206ab`）以**未修改的副本**放在
  `third_party/motion_imitation/`，包含 10 段参考动作与 2 个官方预训练策略；
- 所有改进都在 `src/env_patch.py` 里以运行期 patch 实现，**开关默认关闭**，
  关闭时走"上游兼容基线路径"（summary 级一致；逐步 `(状态,动作,奖励)` 回归与依赖版本锁未做，
  见 `docs/IMPROVEMENTS.md`）；
- 无需重新训练即可验证改进 1（朝向重锚定）与改进 3（评估协议）。

## 复现基线（官方预训练策略，CPU）

| 技能 | 单回合每步奖励 | 关节误差 (rad) | 根误差 (m) | 偏航误差 (rad) | 水平速度 (m/s) |
| --- | --- | --- | --- | --- | --- |
| dog_pace | 0.7496 | 0.0651 | 0.0241 | 0.0558 | 1.0717 |
| dog_spin | 0.6378 | 0.0495 | 0.0913 | 0.8157 | 0.1372 |

（确定性 t=0 起点、单回合 600 步；同一命令在不同进程重复运行，逐回合数值相同。
偏航用完整公式 `atan2(2(wz+xy), 1-2(y²+z²))`；早期版本的 `0.0662 / 0.8017` 用的是
`2·atan2(qz,qw)`，属旧口径，已废弃。）

**这些数与论文 Table V 不能直接比。** 论文 Table V 是**仿真、canonical 动力学、3 个独立
训练策略求平均**（`dog_pace 0.820`、`dog_spin 0.839`）；这里是**单份发布策略、确定性起点、
单回合**。仓库没有记录这两份预训练策略的训练配置，而上游 `run.py` 用的是
`4096 / 1 epoch / 1e-5`，与论文 Table II 的 `10000 / 10 epochs / 2e-5` 不同，因此差距
可能来自评估口径，也可能来自训练预算，**没有重训就无法判定**（详见 `docs/REPRODUCTION.md`）。

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
| 1 | 周期朝向重锚定 `--yaw-anchor` | 上游每周期只重锚定参考**位置**，朝向仅在切换动作时同步；循环转向技能的朝向误差因此无法清零（起步后一个周期就跳到 −0.65 rad，再以 −0.023 rad/s 累积），`root_pose` 分项被压到 0 | `dog_spin` 奖励 0.6980 → **0.7602**（+8.9%），偏航误差 0.2804 → **0.0537 rad**（−80.9%），`root_pose` 分项 0.5513 → 0.8298；`dog_pace` 偏航误差 0.0631 → 0.0335（−46.9%） |
| 2 | 速度奖励去饱和 `--reward-v2` | 上游 `exp(-0.1·ΣΔdq²)` 在 12 关节上求和，实测中位数只有 1.6e-4、均值 2.6e-3，**奖励信号严重饱和** | 该项均值 2.6e-3 → **0.191**（73×），加权贡献从总奖励的 0.02% 升到 2.6%；权重和仍为 1 |
| 3 | 扩展评估协议 `--random-phase --seeds` | 上游评估是"5 回合同一确定性回放"，无误差棒 | 确定性口径把 `dog_spin` 低估 9.5%（0.6378 vs 0.6980 ± 0.0074）；同时给出种子间/种子内分层统计、带符号漂移斜率、终止原因 |

完整动机、代码位置、实测数字与边界说明见 **[docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md)**；
外部审阅意见的逐条核实、处理与证据见 **[docs/REVIEW_RESPONSE.md](docs/REVIEW_RESPONSE.md)**。

## 安装

```powershell
conda env create -f environment.yml     # python 3.7 + TF 1.14 + pybullet 3.2.0
conda activate motion_imitation
# 或手动：pip install -r requirements.txt
```

TensorFlow 1.14 是硬约束（上游用的是 `tf.contrib` 与 stable-baselines 2.10）。
Windows 上跑多进程训练还需 MS-MPI（`mpiexec`）；评估是单进程的、**不需要 GPU**，
但 `src/evaluate.py` 仍要 `import learning.ppo_imitation`，而它 `from mpi4py import MPI`，
所以 mpi4py 的**导入**必须可用（装 MS-MPI + `pip install mpi4py` 即可）。
依赖版本的两份口径见 `requirements.txt`（本机实测）与 `requirements_upstream.txt`（上游声明），
差别主要在 pybullet：3.2.0 vs 2.7.3。

## 用法

```powershell
# 基线（上游口径：确定性 t=0 起步，5 回合）
python -m src.evaluate --motion dog_pace --episodes 5 --max-steps 600

# 扩展评估协议：随机相位 + 3 个评估种子 × 3 回合，输出 mean ± std 与按种子的分层统计
python -m src.evaluate --motion dog_spin --episodes 3 --seeds 3 --random-phase --run-id demo

# 改进 1：每个步态周期重锚定参考朝向
python -m src.evaluate --motion dog_spin --episodes 3 --seeds 3 --random-phase --yaw-anchor

# 改进 2：速度项去饱和 + 权重重配平（只改奖励读数，不改变策略行为）
python -m src.evaluate --motion dog_pace --episodes 3 --seeds 3 --random-phase --reward-v2

# 相位栅格（确定性消融，4 个相位配对比较）
python -m src.evaluate --motion dog_pace --seeds 1 --episodes 4 --start-phases "0,0.25,0.5,0.75"

# 恢复上游初始状态分布（90% 从参考状态起步 + 随机起始时间），用于检查与官方环境的对齐
python -m src.evaluate --motion dog_pace --episodes 3 --seeds 3 --upstream-init

# 汇总成表（只收同一个 run-id，避免混入历史实验）
python -m src.aggregate --run-id demo
```

每条命令输出 `results/eval/<技能>__<变体>/summary.json`，包含每回合的 5 个奖励分项、
关节/根/偏航误差、带符号漂移斜率、水平/三维速度、终止原因，以及 `reward_from_terms`
与 `mean_reward` 的自洽校验（两者应相等）。
一键跑完整矩阵（含确定性对照、相位栅格与按 run-id 汇总）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_protocol.ps1 `
    -RunId r20261114 -Python D:\anaconda\envs\motion_imitation_repro\python.exe
```

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
docs/REVIEW_RESPONSE.md         审阅意见逐条核实与处理（含仍未做项）
docs/REPRODUCTION.md            环境、命令、核对清单
scripts/run_protocol.ps1        完整对比矩阵的顺序执行脚本
reproduction/                   复现产物：HTML 报告、回放视频、原始数据与脚本
```

## 结果表（本仓库实测，可重跑）

| 技能 | 变体 | 回合 | 每步奖励 | 关节误差 | 根误差 | 偏航误差 | 漂移 (rad/s) | 水平速度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dog_pace | 上游基线 | 9 | 0.7481 ± 0.0026 | 0.0651 | 0.0261 | 0.0631 | +0.0004 | 1.0713 |
| dog_pace | + 朝向重锚定 | 9 | 0.7521 ± 0.0014 | 0.0649 | 0.0239 | 0.0335 | 0.0000 | 1.0765 |
| dog_pace | + 速度项 v2 | 9 | 0.7222 ± 0.0027 | 0.0651 | 0.0261 | 0.0631 | +0.0004 | 1.0713 |
| dog_pace | + 朝向重锚定 + v2 | 9 | 0.7264 ± 0.0014 | 0.0649 | 0.0239 | 0.0335 | 0.0000 | 1.0765 |
| dog_spin | 上游基线 | 9 | 0.6980 ± 0.0074 | 0.0502 | 0.0679 | 0.2804 | −0.0052 | 0.1128 |
| dog_spin | + 朝向重锚定 | 9 | 0.7602 ± 0.0014 | 0.0475 | 0.0660 | 0.0537 | +0.0006 | 0.1069 |
| dog_spin | + 速度项 v2 | 9 | 0.6931 ± 0.0074 | 0.0502 | 0.0679 | 0.2804 | −0.0052 | 0.1128 |
| dog_spin | + 朝向重锚定 + v2 | 9 | 0.7551 ± 0.0015 | 0.0475 | 0.0660 | 0.0537 | +0.0006 | 0.1069 |
| dog_pace | 相位栅格（4 相位） | 4 | 0.7492 ± 0.0004 | 0.0651 | 0.0250 | 0.0583 | −0.0020 | 1.0703 |
| dog_spin | 相位栅格（4 相位） | 4 | 0.6828 ± 0.0313 | 0.0499 | 0.0728 | 0.4150 | −0.0099 | 0.1172 |
| dog_pace | 上游初始分布（对齐对照） | 9 | 0.7488 ± 0.0019 | 0.0651 | 0.0250 | 0.0597 | −0.0010 | 1.0714 |
| dog_spin | 上游初始分布（对齐对照） | 9 | 0.7013 ± 0.0068 | 0.0502 | 0.0668 | 0.2701 | −0.0047 | 0.1126 |

两点读法：① v2 那几行的行为指标与同技能基线完全一致（同一条轨迹，只改奖励读数），
速度项均值从 2.6e-3 升到 0.191，奖励随之从 0.7481 降到 0.7222（姿态项权重让出 0.05）；
② 最后两行用 `--upstream-init` 恢复上游初始状态分布（`ref_state_init_prob=0.9` + 随机起始
时间），成绩与受控模式只差 0.3%–1%（0.7488 vs 0.7481、0.7013 vs 0.6980），说明受控起点没有
系统性抬高成绩；两种模式与论文 Table V 仍不可比（单策略、无域随机化）。
③ `dog_spin` 的相位栅格 std 0.0313 远大于随机相位协议的 0.0074：四个相位分别是
0.6378（t=0）、0.7034、0.6854、0.7047，**t=0 起步是明显的低点**（偏航误差 0.8157 rad），
单点评估的结论会比多相位统计更悲观。

（`results/summary_table.md` 由 `src.aggregate` 自动生成，与上表同源。所有数字在固定
线程数 + 确定性相位采样下跨进程可复现：同一条命令跑两次，逐回合数值完全相同。）

## 没做的事（不要当成已完成）

- **没有跑全量 PPO 训练**：论文每技能约 2e8 采样，本机无 GPU、实测 70 步/秒，
  单技能单种子约 33 天。改进 2 属于训练侧改动，目前只有"奖励项不再饱和"的证据，
  没有"重训后策略更好"的证据；
- **没有实现潜在空间自适应 / AWR**（论文 4.2 节）：上游仓库也没有实现；
- **没有真机实验**（论文 Table IV）；
- 训练超参沿用上游 `run.py` 的 `4096 / 1 epoch / 1e-5`，与论文 Table II 的
  `10000 / 10 epochs / 2e-5` 不同；
- **没有运动重定向（retargeting）流程的复现**：只用了仓库随附的 10 段参考动作，没有从动物
  动作重建到 A1 的部分；
- **没有逐位一致声明**：只有 summary 级对齐（`dog_pace` 0.7496 / `dog_spin` 0.6378），
  没有做逐步 `(状态, 观测, 动作, 奖励)` 回归，pybullet 版本也与上游不同；
- `reproduction/` 里的 CSV 与 HTML 报告是**旧口径**（旧 yaw 公式 + 错位的奖励分项），
  只作历史记录，统计请用 `results/`。

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
