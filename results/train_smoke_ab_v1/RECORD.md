# A/B 两臂 30 万步冒烟训练记录（r20261116_ab_smoke_v1）

- 日期：2026-11-16
- 状态：已完成（两臂顺序训练 + 同协议评估 + 验收清单核对）
- 分支：`exp/ab-retrain-smoke`
- 代码提交：`e26a143`（`feat: 新增变体感知 PPO 训练入口`）
- 训练入口：`src/train_ppo.py`
- 冻结基线保持不动：`9809b20` + `r20261116`（本记录不修改任何 `r20261116` 原始数字）

## 1. 目的与范围

本轮只回答一个问题：**变体感知训练入口能否真正把 `baseline` 与 `yaw_anchor` 两臂分开，并把训练、日志、checkpoint、评估链路跑通。**

- 只做 A/B 两臂，不混入 `reward_v2`。
- 技能固定 `dog_pace`，训练 seed 固定 0，环境随机化开启，PPO 超参沿用上游。
- 两臂**顺序**运行，避免 8+8 进程同时抢占机器。
- 30 万步处于 20→600 episode-length 课程早期，**本轮结果不用于判断方法优劣**。

## 2. 代码与实验来源

| 项目 | 内容 |
| --- | --- |
| 分支 / 提交 | `exp/ab-retrain-smoke` / `e26a143d4f3dc75b21234f4094b4cb905bae5363` |
| 训练入口 | `src/train_ppo.py`（`--variant baseline\|yaw_anchor\|reward_v2`） |
| 上游代码 | vendored `third_party/motion_imitation`，commit `2f206ab` |
| 运动文件 | `third_party/motion_imitation/data/motions/dog_pace.txt`，sha256 `fbf6f94c20c3d5da71aef8ae9322eb3366fe9c6fc15be1bb5dc4519c06e3c581` |
| 训练 manifest | `anchor_off/run_manifest.json`、`anchor_on/run_manifest.json` |
| 工作区状态 | 两臂 manifest 均记录 `working_tree_dirty: false` |

`src/train_ppo.py` 相对用户提供的草案有一处必要改动：`ImitationWrapperEnv` 没有可直接调用的 `seed()` 约定，草案中的 `env.seed(rank_seed)` 会抛 `AttributeError`。实际实现改为：

1. 每进程显式设置 `tf / numpy / random` 种子，规则 `rank_seed = seed + 97 × rank`；
2. 构建环境后调用 `env_patch.pin_task_random(task, rank_seed)`，固定 task 私有 `RandomState`（上游从不播种这个对象）。

这不保证训练逐位可复现，只是让两臂共享同一套种子与随机化规则。

**启动前验证**（8 进程 `--dry-run`，各 8 条 `ENV_READY`）：

| 臂 | patch 标志 | 结果 |
| --- | --- | --- |
| `baseline` | `heading_anchor_installed=false`, `velocity_reward_v2=false`, pose 0.50 / vel 0.05 | 通过 |
| `yaw_anchor` | `heading_anchor_installed=true`, `velocity_reward_v2=false`, pose 0.50 / vel 0.05 | 通过 |

## 3. 训练配置

| 项目 | A 臂（anchor_off） | B 臂（anchor_on） |
| --- | --- | --- |
| `--variant` | `baseline` | `yaw_anchor` |
| 技能 | `dog_pace` | `dog_pace` |
| 聚合步数目标 | 300,000 | 300,000 |
| MPI 进程 | 8 | 8 |
| 训练 seed | 0（rank 规则 `0+97×rank`） | 0（同左） |
| 环境随机化 | 开 | 开 |
| `timesteps_per_actorbatch` | ceil(4096/8)=512/进程 | 同左 |
| `optim_batchsize` | ceil(256/8)=32/进程 | 同左 |
| PPO | gamma 0.95 / clip 0.2 / epochs 1 / lr 1e-5 / lam 0.95 / net [512,256] | 同左 |
| 课程 | episode length 20→600，`curriculum_steps=3e7` | 同左 |
| **唯一变量** | 不安装 heading patch | `enable_cycle_heading_anchor` |

两臂 `run_manifest.json` 逐字段比对，实质差异只有 `variant` 与 `patch_flags.heading_anchor_installed`，另有 `run_id` 与启动时刻不同；其余 commit、motion sha、seed、进程数、随机化、PPO 参数完全一致。

## 4. 运行结果

| 指标 | A 臂 | B 臂 |
| --- | --- | --- |
| 实际聚合步数 | 303,104 | 303,104 |
| 墙钟时间 | 698.2 s（11.6 min） | 652.8 s（10.9 min） |
| 聚合吞吐 | 434.1 步/秒 | 464.3 步/秒 |
| rank0 回合数 | 2,390 | 2,268 |
| `model.zip` | 存在，1,761,222 B | 存在，1,761,222 B |
| model sha256 | `bd46afc3…52f53` | `325ffdf0…32eac` |
| 训练日志末轮 | Iteration 73，`TimestepsSoFar=303104` | 同左 |
| 失败文件 | 无 | 无 |

两臂末轮日志的 `EpLenMean` 分别为 16.5 / 17.5，`EpRewMean` 1.36 / 1.39，处于课程早期频繁跌倒的正常区间。

## 5. 十项验收清单

| # | 检查项 | 结论 | 证据 |
| --- | --- | --- | --- |
| 1 | MPI 启动：8 进程、无 rank 崩溃 | 通过 | 两臂 summary 均记录 `mpi_processes=8`；无 `training_failure.json` |
| 2 | 实验变量：A=`baseline`、B=`yaw_anchor` 且 patch 生效 | 通过 | summary `patch_flags`：false/true；dry-run 8 条 `ENV_READY` 同步确认 |
| 3 | 配对条件：commit / 技能 / seed / 进程 / 随机化 / PPO 一致 | 通过 | manifest 逐字段比对，仅 variant 与 patch 标志不同 |
| 4 | 采样量：每臂约 300,000 聚合步 | 通过 | 两臂均 `actual_aggregate_timesteps=303104`（4096/轮、74 轮） |
| 5 | 训练过程：步数单调、无 NaN/Inf、正常结束 | 通过 | 日志 `TimestepsSoFar` 递增至 303104；loss 数值有限；文本筛到的 8 行是 `terminated` 源码回显，不是数值 NaN |
| 6 | 模型产物：每臂 `model.zip` 非空 | 通过 | 两臂均 1,761,222 B，`model_exists=true` |
| 7 | 元数据：manifest + summary 齐全 | 通过 | 两目录各含 `run_manifest.json`、`training_summary.json` |
| 8 | 性能统计：步数 / 墙钟 / 聚合 steps/s 入库 | 通过 | 434.1 / 464.3 步/秒，见 §8 口径说明 |
| 9 | 评估入口：两份 checkpoint 均可被 `src.evaluate` 加载并产出 summary+CSV | 通过 | 各 6 个 `ep*.csv` + `summary.json`，reward 分项自洽 |
| 10 | 评估指标：reward 分项、误差、速度、终止原因均为有限值 | 通过 | 12 回合 `reward_matches_terms=true`；终止原因均为 `early_termination`；几何审计 `max_abs_cum_tracker_vs_geometry≈4.4e-16` |

## 6. 评估结果（同协议）

评估协议：`--episodes 3 --seeds 2 --random-phase --max-steps 600`，评估 seed 仅表示初始条件，不是训练 seed；A 不加 `--yaw-anchor`，B 加。

| 指标（6 回合均值） | A 臂（无重锚定） | B 臂（重锚定） |
| --- | --- | --- |
| `mean_yaw_err_rad`（周期对齐） | 0.3225 ± 0.0770 | 0.3046 ± 0.1555 |
| `mean_yaw_err_fixed_rad`（固定参考系） | 0.3225 ± 0.0770（按定义相同） | 0.4612 ± 0.1304 |
| `final_yaw_err_fixed_rad` | 0.7087 ± 0.1649 | 0.8358 ± 0.1388 |
| `yaw_drift_rad_per_s`（对齐系） | −1.636 ± 0.355 | −0.602 ± 0.998 |
| `yaw_drift_fixed_rad_per_s`（固定系） | −1.636 ± 0.355 | −1.400 ± 0.561 |
| 累计重锚定偏移均值 | 0.000 | −0.4289 rad |
| 周期事件数 / 回合 | 0.00 | 1.33 |
| clip 同步数 / 回合 | 0.00 | 0.00 |
| 回合步数均值 | 15.0 ± 2.28 | 20.5 ± 4.72 |
| 平均奖励 | 0.0835 ± 0.0144 | 0.0743 ± 0.0139 |
| 水平速度 | 1.014 m/s | 1.018 m/s |
| 存活回合 | 0/6 | 0/6 |

**读数注意**：B 臂周期对齐误差略低（−5.6%），但同一轨迹的固定参考系误差更高（+43%），累计偏移 −0.429 rad，方向与 `docs/HEADING_FRAMES.md` 的机制一致——参考系逐周期吸收了偏航差异。这不是“重锚定提升跟踪能力”的证据，只是参考系效应的又一次复现。

## 7. 能说什么 / 不能说什么

**能说：**
- 变体感知训练入口在 8 进程下可运行，两臂确实分开，且除 heading patch 外条件配对。
- 两臂都能在约 11 分钟量级完成 30 万聚合步，产出可加载的 checkpoint、manifest、summary、episode CSV。
- `src.evaluate` 能读取两臂 checkpoint，`mean_yaw_err_fixed_rad` 在 B 臂确实偏离周期对齐值，几何恒等审计通过。

**不能说：**
- 不能根据 30 万步的奖励/误差差异判断哪种方法更好：课程仍在 20 步附近，12 个评估回合全部 `early_termination`，存活率为 0。
- 不能把 B 臂的固定参考系误差当作“关闭重锚定后策略会怎样”的反事实结果；它是同一条已执行轨迹的事后重建。
- 不能外推 1,000 万步的耗时（见 §8）。

## 8. 与历史 70.6 步/秒的口径差异（重要）

`docs/TRAINING_BUDGET.md` 之前按 **70.6 步/秒** 折算“30 万步 ≈ 70.8 分钟”。本轮实测为 **434.1 / 464.3 步/秒**，30 万步只用了 11.6 / 10.9 分钟，相差约 6 倍。

差异来源需要明确记录，不能直接拿新数字重算长期预算：

- 本轮评估/训练都处于课程最早期，`EpLenMean≈16–17` 步，重置开销占比高但每步物理仿真仍是同一步长；
- 历史 70.6 步/秒的测量条件（是否含渲染、是否 600 步满长回合、是否同机同负载）在文档中没有完整记录，无法确认与本轮同口径；
- 因此本轮只声明“30 万步冒烟的实测耗时”，**不据此重算 1e7 步可行性**；要回答 1,000 万步需要多久，必须在 600 步满长回合的稳态条件下重新标定吞吐。

## 9. 复现命令

```powershell
# 分支与入口检查
git switch exp/ab-retrain-smoke
python -m src.train_ppo --help

# 8 进程 dry-run（先确认 patch 真的装上）
mpiexec -n 8 python -m src.train_ppo --motion dog_pace --variant baseline --seed 0 --dry-run
mpiexec -n 8 python -m src.train_ppo --motion dog_pace --variant yaw_anchor --seed 0 --dry-run

# A 臂
mpiexec -n 8 python -m src.train_ppo --motion dog_pace --variant baseline --seed 0 `
    --total-timesteps 300000 `
    --output-dir outputs\train_smoke\r20261116_ab_smoke_seed0\anchor_off `
    --run-id r20261116_ab_smoke_seed0_anchor_off

# B 臂（必须顺序执行）
mpiexec -n 8 python -m src.train_ppo --motion dog_pace --variant yaw_anchor --seed 0 `
    --total-timesteps 300000 `
    --output-dir outputs\train_smoke\r20261116_ab_smoke_seed0\anchor_on `
    --run-id r20261116_ab_smoke_seed0_anchor_on

# 评估
python -m src.evaluate --motion dog_pace --model_file <anchor_off>\model.zip `
    --episodes 3 --seeds 2 --random-phase --max-steps 600 `
    --out results\train_smoke_eval --tag ab_smoke_anchor_off --run-id r20261116_ab_smoke_eval --csv
python -m src.evaluate --motion dog_pace --model_file <anchor_on>\model.zip `
    --episodes 3 --seeds 2 --random-phase --yaw-anchor --max-steps 600 `
    --out results\train_smoke_eval --tag ab_smoke_anchor_on --run-id r20261116_ab_smoke_eval --csv
```

本机实际解释器为 `D:\anaconda\envs\motion_imitation_repro\python.exe`，`mpiexec` 为 `D:\Microsoft MPI\Bin\mpiexec.exe`。

## 10. 未完成事项

- 单臂 1,000 万步可行性实验未开始；需先在满长回合下重新标定吞吐。
- A/B × 2 seed × 1e7 的最小判断预算未开始。
- `reward_v2` 端到端训练未开始（必须独立 run-id，`baseline` vs `reward_v2`，不与本轮 A/B 混）。
- 关闭/开启重锚定的“策略行为”对照仍缺；本轮只验证训练入口与测量链路。
