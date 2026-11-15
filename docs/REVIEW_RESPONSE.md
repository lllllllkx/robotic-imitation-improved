# 审阅意见逐条核实与处理

本文件对应外部审阅提出的问题清单，逐条给出**核实结论**、**做了什么**、**证据在哪**。
所有数字来自 run `r20261115`（`scripts/run_protocol.ps1` 一次跑完 15 次评估：8 个变体 +
2 个相位栅格 + 2 个上游初始分布对照 + 3 个确定性对照，再用 `src.aggregate --run-id r20261115`
汇总成 15 行）。旧数字一律标注为"旧口径"，不参与任何结论。

## 1. 评估协议与论文 Table V 不是一回事

**审阅意见**：把本仓库的评估说成"Table V（3 seeds × 5 回合）"是错的，两者协议不同。

**核实**：对。arXiv v3 的 Table V 是**仿真 + canonical 动力学 + 3 个独立训练策略求平均**；
"3 策略 × 每策略 5 回合 = 15 trials" 是 Table IV 的**真机**协议。本仓库只有一份发布策略，
`--seeds` 只改**评估初始条件**。

**处理**：
- README / IMPROVEMENTS / REPRODUCTION / reproduction README 全部改口径；
- `summary.json` 的 `protocol.seed_meaning` 显式写 `initial conditions (not training seeds)`；
- 新增 `--upstream-init`（恢复 `ref_state_init_prob=0.9` + 随机起始时间），用于单独检查"与官方环境
  对齐程度"，与受控模式分开报告；
- 论文 0.820 / 0.839 与本仓库 0.7496 / 0.6378 的差距**只做来源列举**（口径 / 训练预算 / 域随机化），
  不下"实现有错"或"实现正确"的结论——预训练策略的训练配置仓库里没有记录；
- 另做了一次对齐对照：`--upstream-init` 下 `dog_pace` 0.7473 ± 0.0026、`dog_spin` 0.6980 ± 0.0105，
  对受控模式的 0.7481 ± 0.0026 / 0.6980 ± 0.0074，差 0.1%–0.5%，说明"永远从参考状态起步"这个
  受控假设没有系统性抬高成绩。

## 2. 速度奖励项的统计是错的（P10/P50/P90 由误差分位数反推）

**审阅意见**：用误差均值 / 分位数套 `exp(-0.1·E)` 反推奖励分布，并把 `exp(-8.77)` 说成奖励 p90。

**核实**：对，且比审阅指出的更严重——旧 CSV 的奖励分项是在 `env.step()` **之后**重算的
（参考状态已推进），实测每步"加权分项和"与当步 `reward` 平均差 **0.1068**、最大 0.3180。
因此旧文档那一整条分布（v1 均值 0.0157 / p50 0.0043 / p90 0.0476）不可用。

**处理**：
- `src/evaluate.py` 改为在 `task.reward()` 内部 tee 出**该步实际使用**的分项；
  校验：600 行里每步加权和与 `reward` 差 **0.000000**（`reward_from_terms == mean_reward`）；
- 重算 5400 步（9 回合 × 600 步）的真实分布：p10 4.6e-7 / p50 1.6e-4 / 均值 **2.6e-3** / p90 7.5e-3 / max 0.041；
- 反解误差：`ΣΔdq²` p10 48.9 / p50 87.3 / p90 146.0 / 均值 92.1（单关节 RMS 中位 2.70 rad/s）；
- 旧值在 IMPROVEMENTS.md 里明确标为"**旧文档误算值**"，并说明来源与错在哪；
- 保留"误差 p90 → 奖励下尾"的正确对应：`exp(-14.6) ≈ 4.5e-7` 落在奖励 p10（4.6e-7）；
- Jensen：`exp(-0.1×92.1) = 1.0e-4`，实测均值 2.6e-3 比它高 **26 倍**；
- `tests/test_metrics.py` 用这两组数做断言，防止以后再用错口径。

## 3. 速度项"饱和"证据不足，不能从分布直接推优势函数

**审阅意见**："90% 时间低于某值"只是描述，不能单凭它推断对 PPO 优势的贡献。

**核实**：对。现在给出三层证据：① 该项自身分布；② **加权贡献** `0.05 × 2.6e-3 = 1.3e-4`
（占总奖励 0.7481 的 0.02%），v2 后 `0.10 × 0.191 = 1.9e-2`（2.6%）；③ 同回合 pose 加权贡献
0.4488 → 0.4031。统一表述改为"该项的加权贡献只有 0.02%，几乎不参与奖励构成；v2 把它的
数值区间拉开、加权贡献升到 2.6%"，这是**奖励信号数值分布**的陈述，**不能**证明重训后策略更好。

## 4. `--reward-v2` 下的奖励与基线不可比

**审阅意见**：改尺度和权重后不能拿新奖励和旧奖励比大小。

**核实**：对。已写明：五项权重和恒为 1（代码断言），但尺度变了，跨配置只看关节/根/偏航/速度。
README 结果表里 v2 行的行为指标与基线**完全一致**（同一条轨迹，只有奖励读数变），正是为了说明这点。

## 5. yaw 公式错误（`2·atan2(qz, qw)`）

**核实**：对，该式只在 roll=pitch=0 时成立。**处理**：改为 `atan2(2(wz+xy), 1-2(y²+z²))`，
全部 yaw 数字重跑（旧 0.0662 / 0.8017 / 0.2510 / 0.0473 **全部作废**）；
确定性对照新值（跟踪误差已改到与奖励同一时点，见第 17 条）：`dog_pace` 0.0548、
`dog_spin` 0.7409、`dog_spin+yaw-anchor` 0.0602。

## 6. 漂移指标喂的是包裹后的绝对值序列

**核实**：对，误差越过 ±π 会折返，斜率失真。**处理**：拆成两条曲线——
`mean_yaw_err_rad`（包裹绝对误差，跟踪精度）与 `yaw_drift_rad_per_s`（**展开后带符号**误差的
最小二乘斜率）。测试 `test_drift_on_unwrapped_error_is_monotone` 用 0.5 rad/s × 400 步构造
"包裹序列拟合误差 > 0.2、展开后 ≈ 0.5"，锁住这个行为。

## 7. `mean_speed_mps` 用了三维位移

**核实**：对，机身上下起伏被算成"前进速度"。**处理**：`mean_speed_mps` 改为水平速度
（`hypot(dx, dy)/T`），三维速度另存 `mean_speed_3d_mps`。

## 8. `survived` 不含终止原因

**核实**：对。**处理**：新增 `termination`（`time_limit` / `early_termination`），
`survival_pct` 按 `termination == "time_limit"` 统计。

## 9. 线程变量在 numpy 导入之后才设

**核实**：对，`build_env()` 里设已经太晚。**处理**：`evaluate.py` 顶部（**import numpy 之前**）
设 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `NUMEXPR_NUM_THREADS` /
`VECLIB_MAXIMUM_THREADS` = 1，并写明原因。实测同一命令在两个独立进程里逐回合数值相同。

## 10. 单进程评估也 `import learning.ppo_imitation`，依赖 mpi4py

**核实**：对，上游 `PPOImitation` 构造时会 import mpi4py。**处理（部分）**：不去改上游源码，
改为在文档里明确"评估不需要 MPI 运行时，但 mpi4py 必须可导入（Windows 需 MS-MPI）"。
彻底解耦需要重写上游的 `run.py`/`ppo_imitation.py`，属于未做项，已写进"没做的事"。

## 11. `aggregate.py` 扫全部历史目录

**核实**：对，旧表会把不同时期、不同口径的目录混在一起。**处理**：加 `--run-id` 过滤
（`--results-dir` / `--out-dir` 可指定），所有 summary 落盘时带 `run_id`；`run_protocol.ps1`
统一注入同一个 `-RunId`，最后按它汇总。

## 12. `run_protocol.ps1` 只跑 4 个配置

**核实**：对。**处理**：现在跑 2 技能 × 4 变体（base / +yaw / +v2 / +yaw+v2）+ 2 个相位栅格
+ 2 个上游初始分布对照 + 3 个确定性对照 = **15 次评估**（矩阵、相位栅格、上游对照、
确定性对照共 4 段），末尾自动 `aggregate --run-id`。整脚本已端到端试跑
（日志 `logs/r_protocol.out`），并修掉一个真实 bug：`$ErrorActionPreference="Stop"` 会让
Python 的 stderr 警告（TF FutureWarning）被当成 NativeCommandError 直接中断脚本。

## 13. 测试不足（8 例）

**处理**：扩到 **29 例，全 OK**。覆盖：权重和为 1、归一化回报、窗口均值、漂移斜率、
yaw 四元数（含 roll/pitch 干扰）、wrap/unwrap、v1 求和 / v2 平均+尺度 / "仅平均是恒等变换"、
终止原因（含"最后一步摔倒不判 `time_limit`"）、跟踪误差（关节/根/偏航，含形状校验与
wrap 边界）、相位可复现、两层聚合（种子间 vs 池化）、v2 patch 契约。

## 14. 依赖与上游不一致（pybullet 3.2.0 vs 2.7.3）

**核实**：对。**处理**：拆成 `requirements.txt`（本机实测：3.2.0 / numpy 1.17.3 / scipy 1.7.3 /
matplotlib 3.5.3 / imageio 2.31.2 / Pillow 9.5.0）与 `requirements_upstream.txt`（上游声明），
`environment.yml` 与实测版对齐；`summary.json` 的 `provenance.versions` 现在记录真实版本
（pybullet 用 `pkg_resources` 取，模块本身不导出 `__version__`）。

## 15. "逐位一致"声明没有依据

**核实**：对。**处理**：全部降级为"**summary 级一致**（`dog_pace` 0.7496 / `dog_spin` 0.6378）"，
并明确：没有逐步 `(状态, 观测, 动作, 奖励)` 回归，pybullet 版本不同，物理轨迹不可能逐位相同。

## 16. 旧文档与旧 CSV 必须重跑，不得沿用

**处理**：`results/eval/` 12 个目录 + 3 个确定性对照全部用新代码重跑（run `r20261115`；
跟踪指标的时间点改了以后，全部 15 次评估都重跑，没有沿用上一轮数字），
`results/summary_table.md` 由 `aggregate` 重新生成；`reproduction/` 里的旧产物保留但在
README 与 HTML 报告顶部标注 legacy 口径，不作为统计依据。

## 17. 跟踪误差与奖励不在同一时间点（第二轮审阅，最主要的一条）

**核实：对，而且影响结论。** 上游 `LocomotionGymEnv.step()` 的顺序是
`robot.Step(action) → reward = self._reward() → task.update(self)（含 _update_ref_state /
_update_ref_model）→ 返回 obs`。旧 `evaluate.py` 在 `env.step()` 返回之后才读
`task._ref_pose` 与参考模型位姿，读到的是**第 t+1 帧**，而奖励是第 t 帧的；`--yaw-anchor`
的重锚定又发生在 `_update_ref_motion` 之后被读到，边界误差会被"自然抹平"。

**处理**：`env_patch.attach_term_logger` 在包装 `task.reward()` 时同时 tee 一份状态快照
（`ref_time`、参考/仿真关节、根位置与姿态），`evaluate.py` 的关节、根、偏航、漂移全部改用
快照计算，与奖励严格同一时间点；另新增诊断量 `mean_offstep_yaw_gap_rad` /
`mean_offstep_ref_pos_gap_m`（step 返回后再读一次参考状态与快照的差），把这个错位做成
可审计数字：`dog_spin` 偏航 **0.0763 rad（≈4.4°）**、参考根位移 0.0106 m/步；
`dog_pace` 偏航 0.0130 rad、参考根位移 0.0376 m/步。

**后果**：旧数字整批作废，包括此前对外引用的 `0.2804 → 0.0537（−80.9%）`。新口径下
`dog_spin` 偏航 **0.2081 → 0.0627（−69.9%）**、`dog_pace` **0.0630 → 0.0312（−50.5%）**；
确定性单回合 `0.7409 → 0.0602`。改善依然显著，但幅度比旧口径小，因为旧口径把基线误差
虚高了约 0.07 rad。奖励类指标没有变（奖励计算本身没改动）。

## 18. `aggregate.py` 没纳入确定性对照；脚本也没跑上游对照

**核实：对。** 旧 `aggregate.py` 只扫 `results/eval/*/summary.json`，3 个 `results/det_*`
被漏掉；`run_protocol.ps1` 没跑 `--upstream-init` 两组，而表里却有这两行（此前是手工补跑）。

**处理**：`aggregate.py` 改为递归扫描 `results/<group>/<run>/summary.json`，输出新增
`group` 列（`eval` → `main`，`det*` → `det`），跳过 `_` 开头的探针目录，兼容旧的
`--dir/--group`；`run_protocol.ps1` 补第 3 段"上游初始分布对齐"，现在脚本、汇总表、README
三处都是同一组 **15 次评估**，`summary_table.md` 15 行（main 12 + det 3）。

## 19. `termination_reason()` 的边界

**核实：对。** 旧实现先判 `steps >= max_steps`，于是"恰在最后一步摔倒"被记成 `time_limit`。
上游 `ImitationWrapperEnv.step()` 已把原始 `done` 存进 `info["terminated"]`（在叠加自己的
步数上限之前）。**处理**：签名改为
`termination_reason(terminated, done, steps, max_steps)`，先判 `terminated`，返回值集合
`early_termination / time_limit / env_time_cap / unknown`；`survived` 相应改为
`steps >= max_steps and not terminated`。测试里加了"末步摔倒不判 time_limit"一例。

## 20. V2 文档数字矛盾与"梯度"措辞

**核实：部分成立。** 审阅所指的 `[0.53, 0.16]（均值 0.34）` 并不在 `src/env_patch.py`
（grep 确认，`18bdb7d` 也没有），但 `docs/IMPROVEMENTS.md:117` 确实留了这组陈旧数字——
它是早期用错位数据算的，与实测 `0.048 / 0.162 / 0.361`（均值 0.191）矛盾。另外
`env_patch.py`、`IMPROVEMENTS.md`、本文件里的"没有梯度 / 重新变得有梯度"措辞过强。

**处理**：`env_patch.py` 两处 docstring 与 `IMPROVEMENTS.md` 对应句子统一改成
"数值区间被拉开、加权贡献 0.02% → 2.6%"，并明确"这是奖励信号数值分布的陈述，不是训练效果；
没有 v2 的端到端重训"；`IMPROVEMENTS.md` 里的 `[0.53, 0.16]` 已替换为实测分位数，并在
改进 2 的表格里标注旧版 `p10 0.0002 / p50 0.0043 / 均值 0.0157 / p90 0.0476` 属
**旧文档误算值**（那批数据在 `env.step()` 之后重算，加权和与实际 reward 平均差 0.107）。

---

## 仍然存在的限制（不当作已完成）

| 项 | 状态 |
| --- | --- |
| 改进 2 的端到端训练验证 | 未做（2e8 采样 × 70 步/秒 ≈ 33 天/技能/种子，无 GPU） |
| 潜在空间自适应 / AWR（论文 4.2） | 未实现（上游也没有） |
| 真机实验（Table IV） | 无硬件 |
| 运动重定向 pipeline | 未复现，只用仓库随附 10 段参考动作 |
| 逐步 `(状态, 观测, 动作, 奖励)` 回归 | 未做，因此不声明"逐位一致" |
| 固定参考系的跟踪误差对照 | 未做，改进 1 的结论仍限定在"周期对齐参考系"内 |
| mpi4py 与评估的耦合 | 未解耦，仅文档说明 |
| 固定参考系下的"真实跟踪改善"对照 | 未提供，改进 1 的结论限定在"周期对齐参考系"内 |
