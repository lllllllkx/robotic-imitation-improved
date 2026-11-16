# 固定参考系下的朝向误差：同轨迹事后分解

本文回答一个问题：**改进 1（周期朝向重锚定）让偏航误差从 0.2081 rad 降到 0.0627 rad，
这个改善有多少来自"参考系跟着机器人转"，有多少来自"机器人真的更会跟朝向"？**

结论先说：在我们能做的测量范围内，**几乎全部来自参考系**。同一条已实现轨迹换回固定参考系
重算，`dog_spin` 的平均偏航误差是 **1.4503 rad**（基线 0.2081），`dog_pace` 是 **0.6993 rad**
（基线 0.0630）。所以 README/`IMPROVEMENTS.md` 里"−69.9% / −50.5%"必须带上"周期对齐参考系"
这个限定，不能读成机器人跟踪能力提升。

---

## 1. 这个分解能证明什么、不能证明什么

**能**：在**同一条已经跑完的轨迹**上，把"误差"拆成"参考系转动量"与"参考系内残余误差"，
给出两者各自的数值。

**不能**：证明"如果关掉重锚定，策略会怎么运动"。策略当时观测到的就是**重锚定后的**参考，
参考信息进了观测、进而影响动作；反事实地关掉它，轨迹会完全不同。因此下文一律称

> **固定参考系等价误差（同轨迹事后重建）**

而不称"真实跟踪误差"。

---

## 2. 上游原本怎么处理朝向

| 时机 | 上游行为 | 代码 |
| --- | --- | --- |
| `env.reset()` | `_reset_ref_motion` **直接**把 `_origin_offset_rot` 赋成 `quaternion_about_axis(sim_heading − ref_heading, z)`，**不调用** `_sync_ref_origin` | `imitation_task.py:640-678`（`:648`、`:665-669`） |
| 每周期（相位回绕） | 只同步**位置**（`sync_root_position=True, sync_root_rotation=change_clip`） | `:697-700` |
| 切换动作片段 | 位置 + 朝向一起同步 | `:968-1002`（`:988-992`） |

也就是说：**位置每周期被拉回，朝向只在换片段时才同步**。这就是循环转向技能为什么会在
固定参考系里单向累积偏航误差。

改进 1 把每周期的那次同步改成 `sync_root_rotation=True`——**每周期把参考朝向也对齐到机器人**。

---

## 3. 记号与恒等式

设 `q_raw(t)` 是参考片段自身的根节点旋转（不含任何初始偏移），`θ` 是当前装上的
绕 z 轴偏移角（`_origin_offset_rot` 的世界偏航），则

```
q_ref(t) = Rz(θ) ⊗ q_raw(t)                 # 上游 _calc_ref_pose 左乘 offset：:890-897
```

因为 `Rz` 左乘，世界偏航角确实按加法叠加；但**重建固定参考系必须用四元数复合**，
机身有俯仰/横滚时"两个偏航角相加"与"四元数相乘后取偏航"并不相等，所以 `with_yaw_offset`
走的是四元数路径，而不是把角度相加。

记 `θ_cf` 为 **episode 开始时**（reset 那次直接赋值）装上的偏移，则上游如果**不做**周期重锚定，
参考朝向一直是

```
q_ref_fixed(t) = Rz(θ_cf) ⊗ q_raw(t)
```

而我们的累计偏移量

```
cum(t) = θ(t) − θ_cf
```

两套误差在**同一个时点**（`task.reward()` 内部，参考推进之前）计算：

```
e_anchored = wrap( ψ_sim − ψ(q_ref)        )
e_fixed    = wrap( ψ_sim − ψ(q_ref_fixed)  )
```

`e_fixed` 用的不是"减一个浮点角"，而是把 `q_raw` 用 `Rz(θ_cf)` 真正复合一遍
（`metrics.with_yaw_offset`），所以它等价于"上游那条路径会给出的参考姿态"。

---

## 4. 实现与自检

| 位置 | 作用 |
| --- | --- |
| `src/metrics.py` | `yaw_about_z` / `quat_multiply` / `with_yaw_offset` / `undo_yaw_offset` / `normalize_quat`（纯 numpy） |
| `src/heading_anchor.py` | `AnchorTracker`：`seed(θ_cf)`、`on_sync(θ_new, shared)`、`cum`、`theta_cf`、事件计数；`offset_yaw` / `undo_offset` |
| `src/env_patch.py` | `enable_clip_change(env, min_s, max_s)` 给多片段评估开切换；包住 `_sync_ref_origin` 取事件后的 `_origin_offset_rot` 偏航；用 `_anchor_event_shared = bool(change_clip)` 区分"上游也会做的同步"；快照新增 `ref_orn_raw`（把当前 `θ` 从实际参考姿态里转回去，得到 clip 自身朝向）、`anchor_cum_yaw`、`anchor_theta_cf_yaw`；`reset_anchor_tracker(env)` 在每个 episode 开始时从 `_origin_offset_rot` 播种 |
| `src/evaluate.py` | 逐步写 `anchor_cum_yaw_rad` / `yaw_err_fixed_rad` / `yaw_err_fixed_signed_rad` / `ref_yaw_raw_rad`；per-episode 记 `mean/max/final_yaw_err_fixed_rad`、`yaw_drift_fixed_rad_per_s`、`final_anchor_cum_yaw_rad`、`anchor_cycle_events`、`max_abs_cum_tracker_vs_geometry_rad` |

两个自检都在每次评估时自动跑：

1. **轨迹内部一致性**：`max_abs_cum_tracker_vs_geometry_rad` 比较"tracker 累加的 `cum`"与
   "几何量 `ψ_ref − ψ_raw − θ_cf`"，本轮 5 个锚定 run 全部为 **≤ 2.2e-15 rad**（即两条独立
   路径算同一个量，完全一致）。
2. **单测**：`tests/test_heading_anchor.py` 共 15 例（含四元数往返、左乘偏航严格加和、
   正负偏移、跨 ±π、`seed` 后首个周期事件必须计入、重建后固定帧必须等于 `Rz(θ_cf)⊗q_raw`）。

### 4.1 两个已经修掉的坑（留档，避免以后重踩）

- **基线丢失**：tracker 最初把"episode 里第一次 `_sync_ref_origin`"当作基线，但 reset 那次
  偏移是 `_reset_ref_motion` **直接赋值**、不经过该函数，于是首个周期事件被误判为基线，
  整段偏移被丢掉（冒烟表现为 `cum = 0` 但帧已经转了 0.275 rad）。修法：`reset_anchor_tracker`
  在 `env.reset()` 之后从 `_origin_offset_rot` 播种 `θ_cf`。
- **一帧错位**：`reward()` 里若用 `_calc_ref_pose(_get_motion_time())` 重算 raw 朝向，
   `_get_motion_time()` 已经指到 t+1，重建会整体错一帧（表现为 `anch−fix` 恒差 0.10 rad 左右）。
  修法：不重算，直接把当前 `θ` 从**实际**参考姿态里转回去（`undo_offset`），零时序依赖。

---

## 5. 结果（run `r20261116`，每回合 600 步）

随机相位协议（3 seeds × 3 回合）：

| 技能 | 变体 | 周期对齐偏航误差 | 固定参考系偏航误差 | 累计偏移 `cum` | 周期重锚定事件 | 每周期平均转动 |
| --- | --- | --- | --- | --- | --- | --- |
| dog_spin | 上游基线 | 0.2081 | 0.2081（无重锚定，两口径相同） | 0 | 0 | — |
| dog_spin | + 朝向重锚定 | **0.0627** | **1.4503 ± 0.1021** | −3.0335（−2.78 … −3.18） | 26–27 | ≈ −0.112 rad |
| dog_pace | 上游基线 | 0.0630 | 0.0630 | 0 | 0 | — |
| dog_pace | + 朝向重锚定 | **0.0312** | **0.6993 ± 0.1024** | −1.4060（−1.19 … −1.63） | 31–32 | ≈ −0.045 rad |

确定性单回合（t=0 起点）：

| 技能 | 变体 | 周期对齐偏航误差 | 固定参考系偏航误差 | `cum` |
| --- | --- | --- | --- | --- |
| dog_spin | 上游基线 | 0.7409 | 0.7409 | 0 |
| dog_spin | + 朝向重锚定 | **0.0602** | **1.9912** | −3.6694 |

`+ 朝向重锚定 + v2` 两行的偏航与固定参考系数字与对应的 `+ 朝向重锚定` 行完全相同——
v2 只改奖励读数、不改变动作，这一点在同一提交里已被行为指标逐位一致确认。

### 5.1 怎么读

1. **固定参考系误差 ≈ `cum` 的阶梯**。因为 `ψ_sim ≈ ψ_ref = ψ_raw + θ`（周期对齐后的残余误差
   只有 0.06 rad 量级），所以 `e_fixed ≈ θ − θ_cf = cum`。这不是巧合，正是分解的代数含义：
   **误差在周期对齐口径里被"搬"进了参考系**。
2. **参考系在稳定地转**。`dog_spin` 600 步里发生 26–27 次周期重锚定，累计 −3.03 rad，
   折合每周期约 −0.11 rad；`dog_pace` 31–32 次 → −1.41 rad，每周期约 −0.045 rad。
   各回合终值都同号（spin −2.78 … −3.18、pace −1.19 … −1.63），说明是"机器人每个周期的
   朝向差被逐次吸收进参考系"，而不是随机抖动。
3. **和基线的对照要谨慎**。基线在固定参考系里 0.2081（spin）/ 0.0630（pace），是因为它的
   参考朝向根本不同步，策略必须自己把朝向差压住；重锚定把这股压力撤掉了，参考系随机器人转。
   因此 `0.2081 → 1.4503` 这个"变差 7 倍"既是**测量口径**的结果，也可能是**行为**的结果
   （策略不再被要求跟朝向），二者在同一条轨迹上不可分离。

---

## 6. 研究结论该怎么写

- ✅ "在**每个周期把参考朝向重新对齐**的参考系下，`dog_spin` 平均偏航误差 0.2081 → 0.0627 rad
  （−69.9%），奖励 0.6980 → 0.7602（+8.9%）"。
- ✅ "同一轨迹的**固定参考系等价误差**为 1.4503 rad，累计参考系转动 −3.03 rad，因此该改善
  主要来自参考系转动，不能解释为朝向跟踪能力提升"。
- ❌ "重锚定让机器人的朝向跟踪改善了 69.9%"。
- ❌ "如果不重锚定，误差会是 1.4503 rad"——那是反事实，必须有重训或 A/B 实验才谈得上。

对论文式分析来说，这个结果不是坏消息：它把"奖励上升"拆成了"框架转动 + 残余误差"两部分，
并给出了可复核的数字，比一个孤立的降幅更有说服力。真正要判定改进 1 是否让策略学得更好，
仍需**关闭/开启重锚定的两组重新训练**（本仓库未做，见 `docs/IMPROVEMENTS.md` 的未完成项）。

---

## 7. 复现

```powershell
# 锚定 + 固定参考系两套误差（每次评估自动同时输出）
python -m src.evaluate --motion dog_spin --episodes 3 --seeds 3 --random-phase --yaw-anchor --run-id myrun --csv

# 汇总表新增 yaw_fixed_rad / anchor_cum_rad / cum_gap_rad 三列
python -m src.aggregate --run-id myrun

# 多片段（clip 切换时 θ_cf 重基准）：同内容双片段对照
python -m src.evaluate --motion dog_pace \
    --extra-motions results/_multiclip/aux/dog_pace_copy.txt \
    --clip-change 2:4 --seeds 2 --episodes 3 --yaw-anchor \
    --out results/_multiclip --run-id r_smoke_multiclip
```

CSV 里 `anchor_cum_yaw_rad`、`yaw_err_fixed_rad`、`yaw_err_fixed_signed_rad`、`ref_yaw_raw_rad`
四列可以逐步复算本文所有数字；`summary.json` 的 `protocol.heading_frame` 字段逐字记录了
本口径的性质（"post-hoc fixed-frame reconstruction … not the error a policy would produce
with anchoring disabled"）。

---

## 8. 已知局限（不要当成已完成）

- **多片段切换**：本文所有数字都来自单片段 run，不受影响（`dog_spin` 确定性对照在新分支上重跑
  仍为 0.0602 / 1.9912 / −3.6694，逐位相同）。分支 `exp/multi-clip-theta-cf` 已把 `θ_cf` 在
  上游的 clip 同步处重基准，并用同内容双片段对照与 5 例单测验证恒等审计 ≤ 8.9e-16，
  见 `docs/MULTICLIP_NOTES.md` 与 `results/multiclip_theta_cf_v1/RECORD.md`。仍未验证：连续多段长回合基准、混合技能的行为指标。
- **没有 A/B 重训**：见第 6 节。
- **没有对真机/其它技能做同样分解**：只做了 `dog_spin` / `dog_pace` 两个技能。
