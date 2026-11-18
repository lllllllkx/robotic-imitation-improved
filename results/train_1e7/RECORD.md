# 单臂 1e7 可行性训练记录（baseline，seed 0）

- run-id：`r20261117_single_arm_baseline_1e7`（第 1 段） / `r20261117_single_arm_baseline_seg2_1e7`（第 2 段续跑）
- 分支：`exp/ab-retrain-smoke`
- 上游快照：`third_party/motion_imitation` @ `2f206ab`
- 训练入口：`src/train_ppo.py`（`--resume-step-count` 版本，提交 `62eea30`）
- 运动片段：`data/motions/dog_pace.txt`，sha256 `fbf6f94c20c3d5da71aef8ae9322eb3366fe9c6fc15be1bb5dc4519c06e3c581`
- 最终模型：`outputs/train_1e7/r20261117_single_arm_baseline_seg2/model.zip`，1,761,222 B，sha256 `f23a437552d873b357cf2ae97f5d3e75ae59a2aedd64e1b1051a083fd297947d`

本轮定位是**训练可行性验证**：固定一个实验臂（baseline，无 yaw-anchor、无 reward-v2）、一个训练种子（seed 0），只回答“1e7 步能否稳定跑完、课程能否推进、策略是否脱离早期频繁跌倒”。它不构成 A/B 方法优劣的证据。

## 1. 配置

| 项 | 值 |
| --- | --- |
| 聚合步数目标 | 10,000,000 |
| MPI 进程 | 8 |
| 随机化 | 开（`environment_randomizer=true`） |
| 回合模式 | `train` |
| 课程 | 20 → 600 步，`curriculum_steps_global=3e7`，三次方进度 |
| 课程公式 | `L(t)=20+580t³`，`t=clip(总步数/3e7,0,1)` |
| 学习率 / PPO 超参 | 沿用 `train_ppo.py` 默认（与 A/B 冒烟一致） |
| 中间存档 | 第 1 段 `--int-save-freq 200`；第 2 段 `--int-save-freq 0` |
| 环境步长 | 0.033 s |

课程上限的实际走势（按 `L(t)` 计算并与日志核对）：

| 累计步数 | t | 课程上限 | 日志 `curriculum_limit` |
| --- | --- | --- | --- |
| 4,096 | 0.00014 | 20 | 20 |
| 1e6 | 0.0333 | 20 | 20 |
| 3e6 | 0.1000 | 20 | 20 |
| 5e6 | 0.1667 | 23 | 22 |
| 8,216,576 | 0.2739 | 31 | 31 |
| 1e7 | 0.3333 | 41 | 41 |

`curriculum_limit` 由环境内 `int()` 取整，1e7 时 `int(20+580×0.03704)=41`，与日志一致。

## 2. 执行过程：两段完成

### 2.1 第 1 段（08:26 → 22:00:40，被主机冻结终止）

| 项 | 值 |
| --- | --- |
| 起止 | 2026-11-17 08:26 → 22:00:40（进程树被杀） |
| 完成步数 | 8,216,576（第 2,005 轮结束） |
| 日志 `TimeElapsed` | 48,900 s（`4.89e+04`，三位有效数字） |
| 末轮 `EpLenMean` | 28.8 |
| 末轮 `EpRewMean` | 9.59 |
| 课程上限 | 31 |
| 中间存档 | 2,007 个 zip，合计 3.29 GB（`intermediate/`） |

第 1 段不是正常结束：`console.log` 停在 `Iteration 2006` 开头，`model.zip` 是 21:59:55 的第 2,005 轮周期存档。系统日志在该时段无事件、无重启（上次启动仍为 11-08），22:04 出现的 Windows Update / System Restore 活动晚于进程死亡，因此判定为整机冻结导致后台进程树被一并杀死。

`--int-save-freq 200` 在 Stable-Baselines 的 `CheckpointCallback` 中实测**按迭代**落盘（每 4,096 步一个 zip），不是每 200 步。保存轮及其后 2–5 轮出现 25–64 s 慢轮簇（第 20–24、50–51、69–71、89–90 轮等），这是第 1 段累计吞吐被拉低的主要原因之一。

`TimeElapsed` 由 SB logger 以三位有效数字打印，超过 10,000 s 后变成科学计数法，单轮差值出现 0 或 100 s 量化误差；单轮吞吐不可用，只能看累计墙钟或 checkpoint `LastWriteTime` 间隔。

### 2.2 中断判定与续跑方案

用户选择“回填课程计数器、续跑剩余约 1.78M 步”，而不是从头重跑或只评估 8.2M：

- `src/train_ppo.py` 新增 `--resume-step-count N`：在 `env.reset` 前把 `env._total_step_count` 设为 `round(N / num_procs)`，调用 `_update_time_limit()`，使课程从正确进度继续；manifest 写入 `resume_step_count` 与 `segment=continuation`，summary 增加 `actual_aggregate_timesteps_total`。
- 8,192 步探针（`outputs/train_1e7/_resume_probe`）输出 `RESUME_CURRICULUM aggregate=8216576 per_env=1027072 curriculum_steps=3750000 episode_limit=31`，两轮 `EpLenMean` 29.8 / 29.3，与中断前 28.8 衔接，验证通过。
- 局限：优化器状态、价值函数归一化统计与 RNG 均被重置，因此第 2 段与原连续训练不完全等价；本轮只用于可行性验证，不能据此做细粒度的连续学习曲线分析。

### 2.3 第 2 段（10:18:51 → 11:17:40，正常完成）

| 项 | 值 |
| --- | --- |
| 目标新增步数 | 1,783,424 |
| 实际新增步数 | 1,785,856（按 4,096 步迭代取整） |
| 累计步数 | 10,002,432 |
| 墙钟 | 3,528.13 s（58.8 min） |
| 聚合吞吐 | 506.18 步/秒 |
| rank0 回合数 | 6,970 |
| 状态 | `completed`，8 个 MPI 进程正常退出 |
| 完成时间 | 2026-11-18T03:17:40Z（本地 11:17:40） |

第 2 段关闭了每轮中间存档（`--int-save-freq 0`），没有慢轮簇，吞吐稳定在 506 步/秒，明显高于第 1 段含存档开销时的有效吞吐。

说明：第 2 段的 `run_manifest.json` 记录 `git_commit=32dddaf`、`working_tree_dirty=true`。当时 `--resume-step-count` 的实现还在工作区未提交，随后以 `62eea30` 提交；因此 manifest 里的 commit 指纹早于实际使用的代码版本，两者不是同一个快照。这是本轮可追溯性的一处已知缺口，记录在此而不回填 manifest。

## 3. 课程与训练曲线

合并两段 `progress_snapshot` 后共 2,442 个迭代点，导出 `results/train_1e7/curriculum_curve.csv`。关键节点：

| 累计步数 | 课程上限 | `EpLenMean` | `EpRewMean` |
| --- | --- | --- | --- |
| 4,096 | 20 | 15.1 | 1.12 |
| 1,000,000 | 20 | 18.8 | 1.98 |
| 2,000,000 | 20 | 18.9 | 2.32 |
| 3,000,000 | 20 | 20.0 | 4.31 |
| 5,000,000 | 22 | 21.9 | 6.18 |
| 7,000,000 | 27 | 25.7 | 7.96 |
| 8,216,576 | 31 | 28.8 | 9.59 |
| 9,000,000 | 35 | 31.3 | 10.3 |
| 10,002,432 | 41 | 34.8 | 11.6 |

结论：

- **课程在推进**：上限从 20 升到 41，`EpLenMean` 从 15.1 升到 34.8，`EpRewMean` 从 1.12 升到 11.6，三者同向。
- **策略仍受跌倒限制**：末段 `EpLenMean` 34.8 低于上限 41，且差距在最后 1M 步内由约 2 步扩大到约 6 步。这说明课程上限提高后，策略并未立即跟上；不能把“未达到上限”解释为训练失败，但也不能宣称已具备稳定完成 41 步的能力。
- 早期 0–3M 步 `EpLenMean` 长期在 18–20 步，与课程上限 20 相符，属于课程早期正常现象。

## 4. 固定评估

协议与冻结基线一致：`dog_pace`、`--episodes 3 --seeds 2`（共 6 回合）、`--random-phase`、确定性动作、受控初始状态。两组评估的差别只在 `--max-steps` 与是否 `--yaw-anchor`。

### 4.1 约 41 步课程阶段评估（主要指标）

目录：`results/train_1e7/eval_41/1e7_41__randphase/summary.json`

| 指标 | 值 |
| --- | --- |
| 回合存活 | 4/6 跑满 41 步（`time_limit`），2 个提前终止（29、33 步） |
| 平均步数 | 37.67 ± 5.32 |
| `mean_reward` | 0.4777 ± 0.0674 |
| `mean_pose_err` | 0.0651 ± 0.0078 rad |
| `mean_root_err` | 0.0992 ± 0.0321 m |
| `mean_yaw_err` | 0.1727 ± 0.0444 rad |
| `final_yaw_err` | 0.5722 ± 0.2830 rad |
| `final_yaw_err_signed` | −0.3701 ± 0.5556 rad |
| `mean_speed` | 0.8411 ± 0.0639 m/s |

### 4.2 约 41 步 + 周期偏航重锚定（参考系对照）

目录：`results/train_1e7/eval_41_yaw/1e7_41_yaw__yawanchor+randphase/summary.json`

| 指标 | 值 |
| --- | --- |
| 回合存活 | 4/6 跑满 41 步，2 个提前终止（29、33 步） |
| 平均步数 | 37.67 ± 5.32 |
| `mean_reward` | 0.4812 ± 0.0674 |
| `mean_pose_err` | 0.0654 ± 0.0078 rad |
| `mean_root_err` | 0.0901 ± 0.0321 m（低于不加锚定时的 0.0992） |
| `mean_yaw_err`（对齐参考系） | 0.1185 ± 0.0444 rad |
| `mean_yaw_err_fixed`（固定参考系） | 0.1756 ± 0.0444 rad |
| `final_yaw_err`（对齐） | 0.4147 ± 0.2830 rad |
| `final_yaw_err_fixed`（固定） | 0.5866 ± 0.2830 rad |
| 累计锚定偏航 `final_anchor_cum_yaw` | −0.1487 rad |
| 几何审计 `max_abs_cum_tracker_vs_geometry` | 3.70e-16 rad |
| 周期对齐事件 | 2.00 次/回合 |
| `mean_speed` | 0.8327 ± 0.0639 m/s |

参考系机制再次通过验证：同一批回合里，对齐参考系的 `mean_yaw_err` 0.1185 与固定参考系的 0.1756 不同，几何审计误差 3.7e-16，说明重锚定确实改变了误差定义，**不能只报告对齐指标**。该对照只用于确认测量口径，不用于判断模型优劣。

### 4.3 600 步外推评估（不作为失败判据）

目录：`results/train_1e7/eval_600/1e7_600__randphase/summary.json`

| 指标 | 值 |
| --- | --- |
| 回合存活 | 0/6（全部 `early_termination`） |
| 实际步数 | 81、29、54、50、44、33 |
| 平均步数 | 48.5 |
| `mean_pose_err` | 0.0669 rad |
| `mean_root_err` | 0.1283 m |
| `mean_yaw_err` | 0.2235 rad |
| `mean_speed` | 0.8485 m/s |

600 步是课程当前上限（41 步）的约 15 倍，属于明显分布外推。0/6 存活是预期结果，**不能据此判定本轮训练失败**，也不能据此判定 yaw-anchor 无效。它只说明：以 1e7 步的课程进度，策略尚不具备长时程生存能力。

## 5. 吞吐与预算修订

| 测量 | 结果 | 说明 |
| --- | --- | --- |
| 第 1 段累计 | 8,216,576 / 48,900 s ≈ 168 步/秒 | 含约 12 小时工具通道不可达、多次整机停摆与每轮存档，不能作为训练速度 |
| 第 1 段正常窗口 | 中位 8–9 s/轮，约 455–512 步/秒 | 排除停摆与慢轮簇 |
| 第 2 段 | 1,785,856 / 3,528.13 s = 506.18 步/秒 | 无中间存档，最干净的一段实测 |
| 第 2 段单轮 | 约 8.1 s/轮（4,096 步） | 与第 1 段正常窗口一致 |

按 506 步/秒折算：

| 规模 | 预计墙钟 |
| --- | --- |
| 1e7（本轮实测） | 5.49 h（实际分段合计约 5.7 h 有效计算） |
| 剩余 2e7 | 10.98 h |
| 完整 3e7 | 16.47 h |
| 完整 3e7（按保守 448.7） | 18.57 h |

本轮实测的 506 步/秒比冻结的“实际训练档 448.7”更高，原因是自动存档被关闭、且后段课程回合更长（重置更少）。448.7 仍应作为含存档与调试开销的保守上界。

## 6. 验收结论

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| 训练完整性 | 通过 | 累计 10,002,432 步；`training_summary.json` 状态 `completed`；8 进程退出；无 NaN、无 MPI 崩溃 |
| 模型可加载 | 通过 | 最终 `model.zip` 1,761,222 B；6 回合 ×3 组评估均成功加载 |
| 课程推进 | 通过（有保留） | 上限 20→41，`EpLenMean` 15.1→34.8，`EpRew` 1.12→11.6；末段仍低于上限约 6 步 |
| 生存能力（41 步） | 部分通过 | 4/6 回合跑满 41 步 |
| 生存能力（600 步） | 未通过（外推） | 0/6 存活，最长 81 步；不作为本轮失败判据 |
| 计算效率 | 通过 | 第 2 段 506.18 步/秒；第 1 段正常窗口 455–512 步/秒 |
| A/B 方法优劣 | 不可判定 | 单臂、单种子，且未做正式两臂对照 |

## 7. 限制与注意事项

1. **分段训练不等价于连续训练**：第 2 段重置了优化器状态、价值归一化统计与 RNG；课程计数器是通过 `--resume-step-count` 回填的。可行性结论成立，细粒度学习曲线分析不成立。
2. **`TimeElapsed` 精度不足**：超过 10,000 s 后为三位有效数字科学计数法，单轮差值有 0/100 s 量化误差；本记录的单轮吞吐均来自 checkpoint 时间戳或分段 summary，不用 `TimeElapsed` 差值。
3. **中间存档开销被实测确认**：`--int-save-freq 200` 按迭代落盘，2,007 个 zip / 3.29 GB；长训练应关闭或改为低频，否则墙钟与磁盘都会被显著放大。
4. **主机冻结原因未定**：无系统事件、无重启；22:04 的 Windows Update / System Restore 活动晚于进程死亡。后续长训练建议使用独立会话/服务方式启动，避免进程树挂在会话上。
5. **600 步不能单独作失败判据**：课程在 1e7 时只到 41 步，600 步属于外推。
6. **不能据此宣布 yaw-anchor 改善或损害策略**：本轮是 baseline 臂；参考系对照只验证测量口径。

## 8. 产物

| 类型 | 路径 |
| --- | --- |
| 第 1 段输出 | `outputs/train_1e7/r20261117_single_arm_baseline/`（`console.log`、`episodes.csv`、`model.zip`、`intermediate/`、`progress_snapshot.*`） |
| 第 2 段输出 | `outputs/train_1e7/r20261117_single_arm_baseline_seg2/`（`console.log`、`console.err.log`、`episodes.csv`、`model.zip`、`training_summary.json`、`run_manifest.json`、`progress_snapshot.*`） |
| 续跑探针 | `outputs/train_1e7/_resume_probe/` |
| 课程曲线 | `results/train_1e7/curriculum_curve.csv` |
| 评估 | `results/train_1e7/eval_41/`、`eval_41_yaw/`、`eval_600/` |

## 9. 下一步建议

1. **是否继续剩余 2e7 由训练曲线决定**：当前 `EpLenMean` 与课程上限的差距在扩大，说明策略尚未稳定跟上课程；若继续，应重点观察差距是否收敛，而不是只看 reward。
2. **续训入口需明确**：继续 3e7 时不能用 `--init-model` 从头加载（课程计数器会重置），应整段 3e7 运行，或先实现环境状态/课程计数的完整保存与回填，再做短跑验证。详见 `docs/TRAINING_CONTINUATION.md`。
3. **暂不启动 A/B 正式训练**：单臂单种子不足以比较方法；等长时程生存能力出现可辨识改善后，再投入两臂。
4. **中间存档清理待定**：第 1 段 3.29 GB / 2,007 个 zip 作为证据保留，是否清理需另行决定。
