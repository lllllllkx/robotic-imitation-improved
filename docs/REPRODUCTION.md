# 复现与实验手册

本仓库分两层：

- `third_party/motion_imitation/`：论文官方实现（`erwincoumans/motion_imitation`，commit `2f206ab`）的**未修改副本**，Apache-2.0。
- `src/`：本仓库新增的改进层与实验脚本，只通过运行期 patch 作用在环境对象上。

## 1. 环境

官方代码是 Python 3.7 + TensorFlow 1.14 + pybullet 3.2.0 + gym 0.17.1 + stable-baselines 2.10.0 的栈，本仓库沿用该版本组合：

```powershell
conda create -n motion_imitation python=3.7 -y
conda activate motion_imitation
pip install tensorflow==1.14.0 pybullet==3.2.0 gym==0.17.1 stable-baselines==2.10.0 \
            numpy scipy matplotlib mpi4py pillow imageio
```

Windows 上多进程训练还需要 MS-MPI（`mpiexec`），评估单进程即可。CPU 足够：全部实验在一台无 GPU 的 Intel Core Ultra 7 155H 上完成。

## 2. 评估（论文式协议）

```powershell
# 基线：确定性从 t=0 起步，5 回合
python -m src.evaluate --motion dog_pace --episodes 5 --max-steps 600

# 论文式协议：随机相位 + 3 个随机种子 × 3 回合（给均值和标准差）
python -m src.evaluate --motion dog_spin --episodes 3 --seeds 3 --random-phase

# 改进 1：每个步态周期重锚定参考朝向
python -m src.evaluate --motion dog_spin --episodes 3 --seeds 3 --random-phase \
    --yaw-anchor

# 改进 2：重平衡速度奖励项（只改变奖励读数，不改变策略行为）
python -m src.evaluate --motion dog_pace --reward-v2 --episodes 3 --seeds 3 \
    --random-phase

# 汇总是 CSV + Markdown 表格
python -m src.aggregate
```

输出：`results/eval/<技能>__<变体>/summary.json`，含每回合与聚合的
`mean_reward`、5 个奖励分项、关节/根误差、偏航误差与漂移速率、速度、存活步数。

## 3. 训练（可选，CPU 上非常慢）

上游 `run.py` 的默认超参是 `timesteps_per_actorbatch=4096`、`optim_epochs=1`、
`optim_stepsize=1e-5`，与论文 Table II 的 `batch 10000 / epochs 10 / lr 2e-5 / value lr 1e-5`
不同（见 `docs/IMPROVEMENTS.md`）。仓库提供与上游一致的训练入口：

```powershell
mpiexec -n 8 python third_party\motion_imitation\run.py ^
    --motion_file third_party\motion_imitation\data\motions\dog_pace.txt ^
    --output_dir outputs\train_pace --total_timesteps 300000
```

实测吞吐：8 进程、22 逻辑核、无 GPU 时约 **70 步/秒**（聚合）。论文每个技能约 2 亿采样，
按此吞吐单技能单种子需约 33 天，因此本仓库不做全量训练复现，只做管线验证。

## 4. 结果核对清单

跑完上面的命令后，可逐项核对：

1. 基线单回合评估应与论文实现逐位一致（确定性回合，重复跑结果相同）。
2. `reward_matches_terms == true`：日志中的 `mean(reward)` 与由 5 个分项重建的加权和一致，
   说明"每步奖励"就是论文 Table V 用的归一化回报（权重之和为 1，见 `src/metrics.py`）。
3. `--yaw-anchor` 应显著降低 `yaw_err_rad` 与 `yaw_drift_rad_per_s`（见 `docs/IMPROVEMENTS.md`）。
4. 随机相位协议下的 `std` 应明显大于 0，这是单回合确定性评估无法给出的信息。
