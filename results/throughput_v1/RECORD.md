# 满长回合稳态吞吐标定（r20261116_throughput_v1）

- run-id：`r20261116_throughput_v1`
- 日期：2026-11-16 ～ 2026-11-17
- 代码：分支 `exp/ab-retrain-smoke`，新增 `src/benchmark_throughput.py`、`src/parse_ppo_timings.py`，`src/train_ppo.py` 增加 `--init-model` / `--episode-mode`
- 上游：vendored `third_party/motion_imitation`（`2f206ab`）
- 冻结关系：不改 `r20261116`、不覆盖 `ab-retrain-smoke-v1`

## 1. 本轮回答的问题

把两件事分开测：

1. **环境仿真吞吐**：PyBullet 在 600 步满长回合下的 `env.step` 速度；
2. **PPO 端到端吞吐**：rollout、MPI 汇总与 PPO 更新一起算的稳态速度。

两者都不能用“只把 `max_steps` 设为 600”来冒充满长回合，因此每次测量都记录**真实步数**、`completed_600` 与 `terminated`。

## 2. 固定条件

| 项目 | 设置 |
| --- | --- |
| MPI 进程 | 8（`mpiexec -n 8`），单机 22 逻辑核 |
| 动作来源 | 预训练 `dog_pace.zip`；已先验证测试模式下真实跑满 600 步（`time_limit`，reward 0.7496，与冻结 det pace 一致） |
| 环境模式 | `mode="test"`：固定 600 步上限、无课程 |
| 技能 | `dog_pace` |
| 环境随机化 | **关闭**（满长标定统一口径；训练同条件另见 §5 第三档） |
| 策略推理 | 确定性动作；环境档单独计时、不计入 `env.step` |
| 重复 | 环境 3 次 × 每进程 2 回合；PPO 3 次有效 × 每次 10 轮 / 40,960 聚合步 |
| 预热 | 环境档先跑 1 个完整回合再进入计时段；PPO 档丢弃第 1 轮（TF 图构建与首次推理） |

## 3. 环境仿真吞吐

原始数据：`results/throughput_v1/env_randomizer_off/benchmark.json`、`episodes.csv`。

| 重复 | 聚合步数 | 满 600 步回合 | 跌倒 | 环境墙钟 | 环境吞吐 |
| --- | --- | --- | --- | --- | --- |
| 0 | 9,600 | 16/16 | 0 | 16.68 s | 575.7 步每秒 |
| 1 | 9,600 | 16/16 | 0 | 15.17 s | 632.7 步每秒 |
| 2 | 9,600 | 16/16 | 0 | 15.80 s | 607.6 步每秒 |

- 合计 28,800 聚合步、48 个回合，**48/48 真实完成 600 步，跌倒 0**。
- 环境吞吐均值 **605.3 步每秒**（范围 575.7–632.7）。
- 含策略推理的整墙吞吐均值 573.2 步每秒；策略推理每进程每重复仅 0.83–1.02 s，相对环境 15–17 s 可忽略。

验收：`completed_600` 与 `terminated` 逐回合写入 `episodes.csv`，可核对每个回合的真实长度，而不是只看配置里的 600。

## 4. PPO 端到端吞吐

原始数据：`results/throughput_v1/ppo/run{1..5}.console.log`、对应 `training_summary.json`；汇总 `results/throughput_v1/ppo_timings.json`。

每次运行 40,960 聚合步（10 轮 × 4,096），预训练初始化，600 步回合，随机化关闭，8 进程。

| 运行 | 稳态轮次 | 稳态步数 | 稳态耗时 | 稳态吞吐 | 判定 |
| --- | --- | --- | --- | --- | --- |
| run1 | 9/9 | 36,864 | 64.95 s | 567.6 步每秒 | 有效 |
| run4 | 8/9 | 32,768 | 56.00 s | 585.1 步每秒 | 有效（首轮 61 s 为启动抖动，已剔除） |
| run5 | 7/9 | 28,672 | 56.34 s | 508.9 步每秒 | 有效（两次外部负载尖峰 66.5/33 s，已剔除） |
| run2 | — | — | — | — | **无效**：`TimeElapsed` 出现回退（−43.7 s），整墙 35,929.7 s，机器停摆污染 |
| run3 | — | — | — | — | **无效**：稳态单轮退化到 21.9 s（中位数 10.1 s 的 2.2 倍），中途掉到 75 步每秒 |

有效样本合并：**98,304 步 / 177.29 s = 554.5 步每秒**；三次重复均值 553.9，范围 508.9–585.1。

有效性判据：第 1 轮恒剔除；其余轮要求 `TimeElapsed` 严格递增，且稳态单轮不超过中位数的 1.5 倍；不满足则整次运行判无效，原始日志保留、不参与均值。

## 5. 三档口径与预算重算

| 档位 | 实测吞吐 | 口径 | 用途 |
| --- | --- | --- | --- |
| 环境稳态 | 605.3 步每秒 | 随机化关、预训练策略、只计 `env.step` | 解释模拟器上限，**不能直接当训练预算** |
| PPO 稳态 | 554.5 步每秒 | 随机化关、600 步满长、预训练初始化、含 rollout + MPI + 更新 | 前向预算主口径 |
| 实际训练 | 448.7 步每秒 | 随机化开、20→600 课程早期、A/B 冒烟两臂合计 606,208 步 / 1,351 s | 保守上限，解释短回合重置成本 |

按 554.5 步每秒（PPO 稳态）折算：

| 步数 | 墙钟 | 说明 |
| --- | --- | --- |
| 5e6 | 2.50 h | 预训练续训最小判断 |
| 1e7 | 5.01 h | 单臂可行性 |
| 2e7 | 10.02 h | A/B 两臂 × 1 seed |
| 4e7 | 20.04 h | A/B 两臂 × 2 seed（稳健版） |
| 6e8 | 12.52 天 | 论文级：2 臂 × 3 seed × 1e8 |
| 1.2e9 | 25.05 天 | 满预算 |

按 448.7 步每秒（实际训练）折算：1e7 = 6.19 h、2e7 = 12.38 h、4e7 = 24.76 h、6e8 = 15.48 天、1.2e9 = 30.95 天。

按 605.3 步每秒（环境上限）折算：1e7 = 4.59 h、6e8 = 11.47 天、1.2e9 = 22.94 天。

对照旧口径（70.6 步每秒）：1e7 = 1.64 天、6e8 = 98 天、1.2e9 = 197 天。新实测把同一批预算缩短约 7–8 倍。

结论：**前向预算以 PPO 稳态 554.5 为主，448.7 作为保守上界；605.3 只用于解释模拟器上限，不用于训练预算。**

## 6. 边界与不能说的话

- 三档都测自同一台机器、同一时段；run2/run3 的停摆说明本机吞吐会受外部负载显著影响，长时间训练的墙钟应按 448.7 而不是 554.5 预留余量。
- PPO 档使用预训练初始化与随机化关闭，目的是保证回合真的跑满 600 步；它**不是**课程训练曲线，不能用它推断 1e7 步后的策略质量。
- 实际训练档来自 30 万步冒烟，`EpLenMean≈16–17` 步；短回合频繁 reset 是其比 PPO 档慢的主要原因，这一解释尚未通过单独实验拆分。
- 以上均为单次会话实测的外推，未覆盖机器降频、内存压力、磁盘 IO 与长时间运行的热漂移。

## 7. 复现命令

```powershell
# 环境档
mpiexec -n 8 python -m src.benchmark_throughput `
  --motion dog_pace --repeats 3 --episodes-per-rank 2 `
  --out results\throughput_v1\env_randomizer_off --tag env_rand_off `
  --run-id r20261116_throughput_env_v1

# PPO 档（每次重复执行一次，输出目录递增）
mpiexec -n 8 python -m src.train_ppo `
  --motion dog_pace --variant baseline --seed 0 --total-timesteps 40960 `
  --output-dir outputs\throughput_ppo\run1 `
  --run-id r20261116_throughput_ppo_v1_run1 `
  --init-model third_party\motion_imitation\data\policies\dog_pace.zip `
  --episode-mode test --disable-randomizer

# 逐轮解析与有效性判定
python -m src.parse_ppo_timings `
  --run run1 results\throughput_v1\ppo\run1.console.log `
  --run run4 results\throughput_v1\ppo\run4.console.log `
  --run run5 results\throughput_v1\ppo\run5.console.log `
  --out results\throughput_v1\ppo_timings.json
```
