# 复现与实验手册

本仓库分两层：

- `third_party/motion_imitation/`：论文官方实现（`erwincoumans/motion_imitation`，commit `2f206ab`）的**未修改副本**，Apache-2.0。
- `src/`：本仓库新增的改进层与实验脚本，只通过运行期 patch 作用在环境对象上。

**定位**：这是论文方法的**仿真复现 + 改进研究**。运动重定向的完整流程、潜在空间自适应 / AWR、
真机实验都没有做（清单见最后一节）。

## 1. 环境

依赖分两份，差别主要在物理引擎：

| 组件 | 上游 paper 分支声明 | 本机实测（本仓库 `requirements.txt`） |
| --- | --- | --- |
| Python | 3.7 / 3.8 | 3.7.1 |
| TensorFlow | 1.14.0 | 1.14.0 |
| PyBullet | **2.7.3** | **3.2.0** |
| NumPy | 1.17.3 | 1.17.3 |
| Gym | 0.17.1 | 0.17.1 |
| Stable-Baselines | 2.10.0 | 2.10.0 |
| mpi4py | 3.0.3 | 3.0.3 |
| SciPy / Matplotlib / imageio / Pillow | 未逐一核对 | 1.7.3 / 3.5.3 / 2.31.2 / 9.5.0 |

所以"官方源代码未改"并不等于"运行环境与官方一致"：**pybullet 版本不同，物理轨迹不可能逐位相同**，
这也是本仓库不声明"逐位一致"的原因（见 `docs/IMPROVEMENTS.md` 的可复现性一节）。

```powershell
# 本机验证过的组合
pip install -r requirements.txt
# 想严格贴近上游物理引擎（未在本机验证）
pip install -r requirements_upstream.txt
```

Windows 上多进程训练还需要 MS-MPI（`mpiexec`）。评估是单进程的，**不需要 GPU**，也不需要 MPI
运行时——但注意 `src/evaluate.py` 仍会 `import learning.ppo_imitation`，而它会 `from mpi4py import MPI`，
所以 mpi4py 的**导入**必须可用（装了 MS-MPI + mpi4py 才能跑）。这是上游结构决定的，尚未解耦。

## 2. 评估（受控协议；不是论文原协议）

```powershell
# 基线：受控模式（固定从参考状态起步、相位 0），单回合
python -m src.evaluate --motion dog_pace --episodes 1 --seeds 1 --max-steps 600

# 扩展协议：随机参考相位 + 3 个评估种子 × 3 回合（输出 mean ± std 与按种子的分层统计）
python -m src.evaluate --motion dog_spin --episodes 3 --seeds 3 --random-phase --run-id demo

# 上游初始状态分布（90% 从参考状态起步 + 随机起始时间），用于检查与官方环境的对齐
python -m src.evaluate --motion dog_pace --episodes 3 --seeds 3 --upstream-init --run-id demo

# 改进 1：每个步态周期重锚定参考朝向
python -m src.evaluate --motion dog_spin --episodes 3 --seeds 3 --random-phase --yaw-anchor --run-id demo

# 改进 2：重平衡速度奖励项（只改变奖励读数，不改变策略行为）
python -m src.evaluate --motion dog_pace --reward-v2 --episodes 3 --seeds 3 --random-phase --run-id demo

# 只汇总本次实验（不带 --run-id 会扫全部历史目录）
python -m src.aggregate --run-id demo
```

输出：`results/eval/<技能>__<变体>/summary.json`，含 `protocol`（是否随机相位、初始状态模式、
权重、`seed_meaning`）、`provenance`（上游 commit、实际版本）、`model_sha256`、`by_seed` 分层统计，
以及每回合的 5 个奖励分项、关节/根/偏航误差、带符号漂移斜率、水平/三维速度、终止原因。

## 3. 训练（可选，CPU 上非常慢）

上游 `run.py` 的默认超参是 `timesteps_per_actorbatch=4096`、`optim_epochs=1`、
`optim_stepsize=1e-5`，与论文 Table II 的 `batch 10000 / epochs 10 / lr 2e-5 / value lr 1e-5`
不同。仓库提供与上游一致的训练入口：

```powershell
mpiexec -n 8 python third_party\motion_imitation\run.py `
    --motion_file third_party\motion_imitation\data\motions\dog_pace.txt `
    --output_dir outputs\train_pace --total_timesteps 300000
```

实测吞吐：8 进程、22 逻辑核、无 GPU 时约 **70 步/秒**（聚合）。论文每个技能约 2 亿采样，
按此吞吐单技能单种子需约 33 天，因此本仓库不做全量训练复现，只做管线验证。

## 4. 结果核对清单

跑完上面的命令后，可逐项核对：

1. 基线单回合评估的**汇总值**应与论文实现的评估工具一致（`dog_pace` 0.7496、`dog_spin` 0.6378，
   重复跑数值相同）。这**不是**逐位一致：没有做逐步 `(状态, 观测, 动作, 奖励)` 回归，
   而且 pybullet 版本与上游不同。
2. `reward_from_terms` 应与 `mean_reward` 一致到 1e-12：说明每步记录的分项就是该步奖励的组成部分
   （早期脚本在 `env.step()` 之后重算分项，两者平均差 0.107，已废弃）。
3. `--yaw-anchor` 应显著降低 `yaw_err_rad` 与 `yaw_drift_rad_per_s`，并抬高 `root_pose` 分项
   （见 `docs/IMPROVEMENTS.md` 的实测表）。
4. 随机相位协议下的 `std` 应明显大于 0；`by_seed` 的 `mean_across_seeds` 才是"跨初始条件"的估计，
   池化 std 会把种子内波动混进来。
5. `--reward-v2` 下奖励数值与基线**不可直接比较**（权重与尺度都变了），要看关节/根/偏航误差。

## 5. 复现阶段产物（`reproduction/`）

`reproduction/` 里是第二阶段（隔离工作区 `robotic_imitation_repro`）的交付物：HTML 报告、
两段 20 s 回放、逐帧 CSV、10 段参考动作 CSV、30 万步训练的 `episodes.csv`。
**注意**：那批 CSV 里的奖励分项是在 `env.step()` 之后记录的，口径不正确（见清单第 2 条），
其中的偏航数值也是旧公式产物。它们保留作为历史记录，**不要**用它们做统计结论；
统计请用本仓库 `src/evaluate.py --csv` 新生成的 `ep*.csv`。
