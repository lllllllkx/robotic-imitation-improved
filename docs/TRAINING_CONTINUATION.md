# 续训、课程计数器与固定难度诊断

## 1. 事实

- 课程回合长度由 `ImitationWrapperEnv._total_step_count / _curriculum_steps` 的三次方决定（`20 → 600`，`curriculum_steps=3e7`，按进程均分）。
- `_total_step_count` 与 `_max_episode_steps` 只存在于环境实例内，**不会写进 `model.zip`**。
- `src/train_ppo.py --init-model` 调用 `PPOImitation.load_parameters`，只恢复网络权重；Stable-Baselines 的该接口明确说明**不加载超参、不更新优化器/动量变量**。
- `PPOImitation.learn` 默认 `reset_num_timesteps=True`，`num_timesteps` 从 0 开始；本仓库使用 `schedule="constant"`，学习率固定 1e-5，没有调度状态需要恢复。

## 2. 实际发生过的分段训练

`r20261117_single_arm_baseline_1e7` 被主机冻结打断后，按用户选择做了**权重恢复 + 课程计数器回填**：

| 段 | 新增步数 | 起止 | 说明 |
| --- | --- | --- | --- |
| 第 1 段 | 8,216,576 | 2026-11-17 08:26 → 22:00:40 | 主机整体冻结，进程树被杀 |
| 第 2 段 | 1,785,856 | 2026-11-18 10:18:51 → 11:17:40 | `--resume-step-count 8216576` 续跑 |
| 合计 | 10,002,432 | — | 超目标 2,432 步（PPO 按 4,096 步迭代取整） |

### 第 2 段到底恢复了什么

| 状态 | 是否恢复 | 依据 |
| --- | --- | --- |
| 策略/价值网络权重 | 是 | `--init-model` → `load_parameters` |
| 课程计数器 | 是（手工回填） | `--resume-step-count` → `env._total_step_count` |
| 回合上限 | 是（由回填计数器重算） | `_update_time_limit()`，探针输出 `episode_limit=31` |
| Adam 优化器状态（一阶/二阶矩） | **否** | `load_parameters` 不更新 trainer/optimizer 变量 |
| `num_timesteps` / 迭代计数 | **否** | `learn()` 默认重置为 0 |
| 学习率调度状态 | 不适用 | `schedule="constant"`，固定 1e-5 |
| Python / NumPy / TensorFlow RNG | **否** | 启动时按 `base_seed + 97 × rank` 重新播种 |
| 环境随机化 RNG 流 | **否**（同 seed 重新播种，不延续原流） | `env.seed()` + `pin_task_random` |
| rollout 缓冲区 | **否** | 新进程重新分配 |

**准确表述**：这是一次“分段训练 + 权重恢复 + 课程计数器回填”，不是完全无缝的 1e7 连续训练。可行性结论不受影响，但不能用它做严格连续学习曲线的细粒度分析，也不能把两段拼成“完整课程”。

第 2 段 `run_manifest.json` 记录 `git_commit=32dddaf`、`working_tree_dirty=true`：当时 `--resume-step-count` 仍在工作区未提交，随后才以 `62eea30` 提交。manifest 的 commit 指纹早于实际运行代码，这是已知的可追溯性缺口。

## 3. 固定难度诊断入口（本轮新增）

原三次方课程在 1e7 后仍会快速抬升：10M 约 41.5 步、11M 48.6 步、12M 57.1 步、13M 67.2 步。若直接续跑，难度和策略能力会同时变化，“差距是否缩小”无法解释。为此新增：

```text
--fixed-episode-length N
```

语义：

- 设置 `env._max_episode_steps = N`；
- 将 `env._curriculum_steps` 置 0，使 `_enable_curriculum()` 返回 False，`reset()` 不再按计数器重算上限；
- 仅允许与 `--episode-mode train` 组合；
- 启动时打印 `FIXED_EPISODE_LENGTH episode_limit=… curriculum_enabled=False`，并断言上限等于请求值。

manifest 与 `training_summary.json` 新增：

```json
"fixed_episode_length": 41,
"curriculum_mode": "fixed"
```

未使用该参数时为 `null` / `"cubic"`，既有 run 的语义不变。

探针验证（`outputs/train_1e7/_plateau_probe`，8,192 步）：

```text
RESUME_CURRICULUM aggregate=10002432 per_env=1250304 curriculum_steps=3750000 episode_limit=41
FIXED_EPISODE_LENGTH episode_limit=41 curriculum_enabled=False total_step_count=1250304
EpLenMean 34.1 (4,096 步) → 35.0 (8,192 步)
```

起始回合长度与 1e7 末段的 34.8 衔接，说明固定 41 步没有把策略放回 20 步起点。

## 4. 当前决定

1. `r20261117_single_arm_baseline_1e7` 冻结，不再改配置、不覆盖归档。
2. 下一步是**41 步课程平台诊断**：新 run-id `r20261118_plateau41_diag_2m`，从 1e7 最终模型初始化，随机化/动作控制/奖励不变，回合上限固定 41，训练约 2M 步。
3. 诊断验收使用与 1e7 相同的固定评估协议（3 回合 × 2 seed、`--random-phase`、`--max-steps 41`），比较训练前后存活率、实际步数、pose/root/yaw 与终止原因。
4. 只有诊断显示策略能在难度不再增加时继续改善，才考虑恢复渐进课程、放缓课程曲线或投入剩余 2e7。

## 5. 平台诊断结果（2026-11-18）

`r20261118_plateau41_diag_2m` 已完成：新增 2,002,944 步、累计 12,005,376、墙钟 5,299.6 s（全程 377.9 步/秒，开头受电池供电限制）。训练口径 `EpLenMean` 从 34.4 到 33.9，没有向 41 步靠近。

固定评估（3 回合 × 2 seed、`--random-phase`、`--max-steps 41`）前后对比：

| 指标 | 前测 | 后测 |
| --- | --- | --- |
| 存活 | 4/6 | 4/6 |
| 平均步数 | 37.67 | 38.33 |
| 奖励 | 0.4777 | 0.5097 |
| pose 误差 | 0.0651 | 0.0597 |
| 平均 yaw 误差 | 0.1727 | 0.1245 |
| 末端 yaw 误差 | 0.5722 | 0.5681 |

两次评估提前终止的都是相同两个相位（0.417、0.129）。结论是“固定难度下未观察到稳定掌握”，只在 pose 误差上有相对稳健的小幅改善。完整数据与边界见 `results/train_1e7/PLATEAU41_DIAG.md`。因此暂不投入剩余 2e7，先扩大评估样本并诊断两个固定失败相位。不能把“固定 41 步诊断”的结果直接当作完整课程或 600 步能力证据。
