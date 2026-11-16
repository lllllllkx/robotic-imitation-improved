# 多片段 `θ_cf` 扩展（分支 `exp/multi-clip-theta-cf`）

本文件记录在冻结基线 `9754225` 之上、从 `docs/BASELINE.md` 第 4 节"多片段 `θ_cf` 扩展"这一项
派生出来的改动。它**不修改** `r20261116` 的任何数字，也不改变单片段评估的行为。

## 1. 问题

固定参考系等价误差用 `q_ref_fixed = Rz(θ_cf) ⊗ q_raw` 重建。`θ_cf` 原先只在 episode 开始时播种一次
（`AnchorTracker.seed`）。单片段评估里没有 clip 切换，这样是对的；但上游在 clip 切换时**自己也会**
同步朝向（`imitation_task.py:697-700`，`sync_root_rotation=change_clip`），它跟踪的参考系在新片段上
是 `Rz(θ_new) ⊗ q_raw_new`。如果 `θ_cf` 停在旧片段的值，固定参考系就会拿"上一段的坐标系"去量新片段，
重建失去意义。

## 2. 改动

| 位置 | 改动 |
| --- | --- |
| `src/heading_anchor.py` | `on_sync` 在首次同步与每个 `shared`（上游也会做的）同步上更新 `theta_cf = theta_new`；新增 `clip_events` 计数；未调用 `seed` 时首次同步也补上 `theta_cf` |
| `src/env_patch.py` | 新增 `enable_clip_change(env, min_s, max_s)`：设 `_clip_time_min/_clip_time_max` 并抽下一次切换时间；切换时长来自 task RNG，已被 `pin_task_random` 钉住 |
| `src/evaluate.py` | `--extra-motions`（逗号分隔，`data/motions` 名字或路径）、`--clip-change "min:max"`（秒）；`build_env` 接受多片段列表；summary 新增 `anchor_clip_syncs`；`protocol` 记录 `motion_files` 与 `clip_change_s` |
| `tests/test_heading_anchor.py` | 新增 5 例：首次同步补 `theta_cf`、clip 切换重基准、跨 ±π 是小步、两片段逐步恒等重建、切换后周期偏移从新基线累计 |

单片段路径不变：只有一个 `motion` 时上游 `_check_change_clip` 恒 False（`num_motions > 1` 才切换），
`theta_cf` 除 `seed` 外不会被改写，`cum` 仍只累计我们新增的周期同步。

## 3. 验证

**单测**：`python -m unittest discover -s tests` → **49 例通过**（原 44 + 5）。

**同内容双片段对照**（把 `dog_pace.txt` 复制成第二份加载，策略行为与单片段一致，只trigger切换；
2 seeds × 3 回合，`--clip-change 2:4`，600 步上限）：

| 指标 | 值 |
| --- | --- |
| 每回合 clip 同步次数 | 1–6（6 回合共 22 次） |
| 每回合周期重锚定事件 | 8–34 |
| `cum_gap_rad`（tracker 与几何恒等审计） | ≤ **8.9e-16** |
| 重复运行（独立进程、独立目录） | 除 `wall_time` 外**逐位一致**（含每个回合的 reward、误差、事件数） |

**单片段回归**（`dog_spin`，1 回合，确定性起点）：0.0602 rad / 固定参考系 1.9912 rad /
`cum` −3.6694，与 `r20261116` 的确定性对照**逐位相同**，说明扩展没有改变单片段口径。

**混合片段烟雾**（`dog_pace` + `dog_spin`，1 seed × 3 回合，`--clip-change 3:6`）：`cum_gap_rad`
同样 ≤ **8.9e-16**；但 3 个回合中有 2 个在第 37 步跌倒，**行为指标不可比**，仅用于审计恒等。
正式归档见 `results/multiclip_theta_cf_v1/RECORD.md`（run-id `r20261116_multiclip_theta_cf_v1`，含逐步 CSV）；烟测目录 `results/_multiclip/` 被 gitignore 排除，仅作过程留档。

## 4. 能说与不能说

- ✅ "多片段切换时，固定参考系重建在 clip 切换处重基准，恒等审计在 1e-15 量级成立"。
- ✅ "同一内容双片段对照可逐位复现"。
- ❌ "已支持任意多片段"——当前验证只有两个片段、一个切换机制（上游 `_clip_change_time` 抽样），
  没有连续多段的长回合基准，也没有混合技能的性能数字。
- ❌ 混合片段的奖励/误差不能当作策略性能，它测的是参考系算术，不是控制能力。
- ❌ 这不构成对 `r20261116` 的解释升级：原来的单片段结论、A/B 重训的必要性都不变。

## 5. 复现

```powershell
# 同内容双片段审计（把 dog_pace 复制成两个 clip）
python -m src.evaluate --motion dog_pace \
    --extra-motions results/_multiclip/motion_copy/dog_pace_copy.txt \
    --clip-change 2:4 --policy dog_pace --seeds 2 --episodes 3 \
    --start-phases 0 --yaw-anchor --out results/_multiclip --run-id r_smoke_multiclip

# 混合片段（只看恒等审计，不看行为指标）
python -m src.evaluate --motion dog_pace --extra-motions dog_spin \
    --clip-change 3:6 --policy dog_pace --seeds 1 --episodes 3 \
    --start-phases 0 --yaw-anchor --out results/_multiclip --tag mix_pace_spin
```

summary 里看 `max_abs_cum_tracker_vs_geometry_rad` 与 `anchor_clip_syncs`；逐步数据在
`--csv` 输出的 `anchor_cum_yaw_rad` / `yaw_err_fixed_rad` 两列。
