# 多片段 `θ_cf` 验证记录（`r20261116_multiclip_theta_cf_v1`）

独立记录，**不属于冻结基线 `r20261116`**。冻结批次的数字、`docs/BASELINE.md` 与
`docs/REVIEW_RESPONSE.md` 均未改动；本记录只追加。

| 项 | 值 |
| --- | --- |
| run-id | `r20261116_multiclip_theta_cf_v1` |
| 代码来源 | 分支 `exp/multi-clip-theta-cf`，提交 `2717b5d`，标签 `multiclip-theta-cf-v1` |
| 派生自 | `9754225`（冻结基线，`9809b20` + `r20261116`） |
| 环境 | `motion_imitation_repro`（py3.7.1 / TF 1.14 / pybullet 3.2.0），无 GPU |
| 策略 | 官方预训练 `dog_pace.zip`（`5efb1a76…`）、`dog_spin.zip`（`763f4cc7…`），**未重训** |
| 评估口径 | `--yaw-anchor`、`--start-phases 0`、`max_steps 600`、指标在 `task.reward()` 内同步快照 |
| 单测 | `python -m unittest discover -s tests` → 49 例通过 |

## 1. 三组实验

### 1.1 `dup_pace`：同内容双片段（`dog_pace` + 自身副本）

2 seeds × 3 回合，`--clip-change 2:4`（秒）。两段内容相同，策略行为与单片段可比，只触发切换。

| seed | ep | 步数 | clip 同步 | 周期事件 | 终止 | 锚定 yaw | 固定参考系 yaw | `cum_gap` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 600 | 5 | 33 | time_limit | 0.0435 | 0.1059 | 8.9e-16 |
| 0 | 1 | 599 | 6 | 34 | early_termination | 0.0373 | 0.1593 | 4.4e-16 |
| 0 | 2 | 156 | 2 | 8 | early_termination | 0.0739 | 0.0802 | 4.4e-16 |
| 1 | 3 | 218 | 2 | 11 | early_termination | 0.0713 | 0.0770 | 8.9e-16 |
| 1 | 4 | 160 | 1 | 8 | early_termination | 0.1466 | 0.2826 | 8.9e-16 |
| 1 | 5 | 600 | 6 | 32 | time_limit | 0.0464 | 0.0861 | 4.4e-16 |

聚合：奖励 0.6527 ± 0.0755，锚定 yaw 0.0698，固定参考系 yaw 0.1318，平均每个回合 3.7 次
clip 同步、21 次周期重锚定，步数均值 388.8。**`cum_gap` 均值 6.7e-16、最大 8.9e-16**。

### 1.2 `mix_pace_spin`：异内容混片（`dog_pace` + `dog_spin`）

1 seed × 3 回合，`--clip-change 3:6`。

| ep | 步数 | clip 同步 | 周期事件 | 锚定 yaw | 固定参考系 yaw | `cum_gap` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 37 | 0 | 1 | 0.4324 | 0.6777 | 8.9e-16 |
| 1 | 37 | 0 | 1 | 0.4325 | 0.6778 | 8.9e-16 |
| 2 | 473 | 3 | 25 | 0.0841 | 0.2796 | 8.9e-16 |

**3 个回合中 2 个在第 37 步跌倒**（策略与混片不匹配），奖励 0.3308 ± 0.3016、
速度 0.626 m/s。因此本组**只用于参考系算术审计**，行为指标不得引用。

### 1.3 `det_spin_check`：单片段回归

`dog_spin`，1 回合，确定性起点，600 步，0 次 clip 同步：

| 指标 | 本记录 | 冻结批 `r20261116` | 差异 |
| --- | --- | --- | --- |
| 奖励 | 0.7609 | 0.7609 | 0 |
| 锚定 yaw | 0.0602 | 0.0602 | 0 |
| 固定参考系 yaw | 1.9912 | 1.9912 | 0 |
| `cum` | −3.6694 | −3.6694 | 0 |
| 周期事件 | 26 | 26 | 0 |
| `cum_gap` | 2.2e-15 | — | 恒等审计 |

结论：扩展没有改变单片段评估路径。

## 2. 可复现性

`dup_pace` 在本目录与烟测目录（`results/_multiclip/`）各独立跑一次，`aggregate` 中
除 `wall_time_s_mean/std` 与 `episodes[*].wall_time_s` 外**逐字段相同**——奖励、误差、
事件数、步数逐位一致。`--clip-change` 的切换时刻来自 task RNG，已被 `pin_task_random`
钉住，所以同 seed 重跑可复现。

## 3. 结论边界

✅ 可以说

- clip 切换时 `θ_cf` 已按上游同步重新播种，固定参考系重建在 1e-15 量级恒等成立
  （6 个回合、最多 6 次切换/回合、两个独立进程）。
- 同内容双片段下评估可逐位复现。
- 单片段口径与冻结批次逐位一致。

❌ 不可以说

- "已支持任意多片段"——只验证了两段、一个切换机制、单回合上限 600 步。
- 连续多段的长回合基准、混合技能的行为性能**均未验证**；`mix_pace_spin` 的奖励、yaw、
  速度**不得**作为性能数字引用。
- 这不改变 `r20261116` 的任何结论，也不构成对 A/B 重训或 V2 端到端训练已完成的暗示。

## 4. 文件与复现

```
results/multiclip_theta_cf_v1/
  dup_pace__yawanchor+phasegrid/summary.json, ep00..ep05.csv
  mix_pace_spin__yawanchor+phasegrid/summary.json, ep00..ep2.csv
  det_spin_check__yawanchor/summary.json, ep00.csv
  motion_copy/dog_pace_copy.txt   # 第二片段的副本（与 dog_pace.txt 内容相同）
```

```powershell
# 1) 同内容双片段
python -m src.evaluate --motion dog_pace \
    --extra-motions results/multiclip_theta_cf_v1/motion_copy/dog_pace_copy.txt \
    --clip-change 2:4 --policy dog_pace --seeds 2 --episodes 3 \
    --start-phases 0 --yaw-anchor --out results/multiclip_theta_cf_v1 --tag dup_pace \
    --run-id r20261116_multiclip_theta_cf_v1 --csv

# 2) 混合片段（只读审计列）
python -m src.evaluate --motion dog_pace --extra-motions dog_spin \
    --clip-change 3:6 --policy dog_pace --seeds 1 --episodes 3 \
    --start-phases 0 --yaw-anchor --out results/multiclip_theta_cf_v1 --tag mix_pace_spin \
    --run-id r20261116_multiclip_theta_cf_v1 --csv

# 3) 单片段回归
python -m src.evaluate --motion dog_spin --episodes 1 --seeds 1 --yaw-anchor \
    --out results/multiclip_theta_cf_v1 --tag det_spin_check \
    --run-id r20261116_multiclip_theta_cf_v1 --csv
```

逐步复算看 CSV 的 `anchor_cum_yaw_rad`、`yaw_err_fixed_rad`、`ref_yaw_raw_rad` 三列；
聚合值看 `summary.json` 的 `aggregate.max_abs_cum_tracker_vs_geometry_rad` 与
`anchor_clip_syncs`。口径性质见 `protocol.heading_frame`。
