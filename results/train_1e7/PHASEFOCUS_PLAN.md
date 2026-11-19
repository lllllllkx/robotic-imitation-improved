# 41 步平台相位聚焦实验（r20261118_plateau41_phasefocus_2m）

## 目标与单变量边界

从 1e7 最终模型继续训练 2M 聚合步，固定 41 步课程平台，单臂、单 seed。
唯一训练干预是 **episode 的初始参考相位采样分布**；奖励函数、随机化、动作控制、
策略网络、优化器、奖励权重和训练难度全部保持不变。

起点模型必须与 `r20261118_plateau41_diag_2m` 一致：

- `outputs/train_1e7/r20261117_single_arm_baseline_seg2/model.zip`
- sha256 `f23a437552d873b357cf2ae97f5d3e75ae59a2aedd64e1b1051a083fd297947d`

这样 2M 诊断与本实验的差异只剩相位采样，可以直接对照。

## 采样规则

| 项目 | 设定 |
| --- | --- |
| 重点区间 | `[0.11,0.14]`、`[0.41,0.50]`、`[0.95,0.98]` |
| 重点区间概率 | 每个区间等概率 1/3，**不按区间宽度加权** |
| 重点采样比例 | 非 warmup episode 的 50% |
| 其余 episode | 从完整片段 `[0,1)` 均匀抽取 |
| warmup episode | 保留上游 warmup 偏移，不参与相位采样 |
| 采样 seed | `phase_seed + 97 * MPI_rank`，base seed `20261118` |

`ref_state_init_prob=0.9`，因此约 10% episode 进入 warmup 分支。以 band 总宽度
0.15 估算，若全部 episode 都参与采样，高风险相位实际覆盖率约
`0.5 + 0.5 * 0.15 = 57.5%`；排除约 10% warmup 后，整体约 51.8%。实际频率以
`training_summary.json` 的 `actual_in_focus_fraction` 为准，不把训练分布简化成
“失败带占 50%”。

每个 rank 写入 `phase_samples_rank{rank}.csv`，字段：

`rank,episode,phase,source,focus_band,in_focus_band,warmup`

其中 `source` 为 `band_0`、`band_1`、`band_2`、`uniform` 或 `warmup`；
`in_focus_band` 按**实际抽到的相位**计算，因此均匀半区落入重点带也会计入。

## 训练命令

```text
mpiexec -n 8 python -m src.train_ppo \
  --motion dog_pace --variant baseline --seed 0 \
  --total-timesteps 2000000 \
  --output-dir outputs/train_1e7/r20261118_plateau41_phasefocus_2m \
  --run-id r20261118_plateau41_phasefocus_2m \
  --fixed-episode-length 41 \
  --resume-step-count 10002432 \
  --init-model outputs/train_1e7/r20261117_single_arm_baseline_seg2/model.zip \
  --phase-sampling phasefocus \
  --phase-bands 0.11:0.14,0.41:0.50,0.95:0.98 \
  --phase-focus-fraction 0.5 --phase-seed 20261118
```

`--total-timesteps` 是新增步数；完成后累计步数应为 12,005,376 附近，和 2M 诊断
同一量级。随机化默认开启，不额外传 `--disable-randomizer`。

## 评估设计

现有 20 对配对数据正式定义为**开发集**。本实验使用独立的 20 个固定留出相位
（`results/train_1e7/phasefocus/holdout_phase_plan.json`）：

- 3 个在 `[0.11,0.14]`，4 个在 `[0.41,0.50]`，3 个在 `[0.95,0.98]`；
- 10 个普通相位，其中 `0.3386` 作为 0.339 退化区的哨兵；
- 前测（1e7 模型）与后测模型使用完全相同的相位、随机化关闭、controlled init、
  `--max-steps 41 --episodes 20 --seeds 1`。

主指标是 41 步干净完成率。次指标按以下顺序报告：

1. 最差关节误差及具体关节编号；
2. 根位置误差；
3. 根朝向误差与末端 yaw；
4. 终止类型和终止前 15–20 步变化；
5. 平均 pose / yaw 误差。

统计保留配对四分类（都成功、前测失败后测成功、前测成功后测失败、都失败）、
配对差标准误和置信区间；不能只比较两个百分比。

## 奖励审计（第一轮不改奖励）

baseline 实际奖励权重为：

```text
pose 0.50, velocity 0.05, end_effector 0.20,
root_pose 0.15, root_velocity 0.10
```

`root_pose` 项已经同时包含根位置平方误差和根朝向角平方误差
（`root_pose_err = root_pos_err + 0.5 * root_rot_err`，见
`imitation_task.py::_calc_reward_root_pose`）。因此第二轮若考虑根位置或根朝向
奖励，必须先处理与现有 0.15 权重的重复计数问题，不能直接叠加。

## 决策规则

| 观察 | 下一步 |
| --- | --- |
| 留出集与三个重点带都改善 | 相位采样值得进一步验证 |
| 重点带改善、留出集不改善 | 可能对重点相位过拟合，不扩大预算 |
| 最差关节改善但 `root_rot_fail` 仍在 | 单独研究根朝向控制或相关奖励 |
| 平均误差下降但最差关节/完成率不变 | 继续失败机制诊断，不追加 2e7 |
| 0.339 等非重点相位退化 | 检查相位采样对其他区域的负迁移 |

## 解释边界

- 单臂、单 seed、20 个留出相位只能支持开发阶段判断，不足以宣称方法优劣。
- 41 步干净完成率是主指标；600 步结果只作长度外推，不单独作为失败判据。
- 本实验不启动剩余 2e7，不修改 `r20261117_single_arm_baseline_1e7`、
  `r20261118_plateau41_diag_2m` 或更早归档。
- 主机若在电池供电下运行，吞吐会显著低于接电状态；墙钟时间必须按实际电源
  状态解释，不能把不同电源状态的 steps/s 直接混用。
