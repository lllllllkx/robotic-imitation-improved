# 续训与课程计数器（重要）

## 事实

- 课程回合长度由 `ImitationWrapperEnv._total_step_count / _curriculum_steps` 的三次方决定（`20 → 600`，`curriculum_steps=3e7`，按进程均分）。
- `_total_step_count` 与 `_max_episode_steps` 只存在于环境实例内，**不会写进 `model.zip`**。
- `src/train_ppo.py --init-model` 只恢复网络权重（`PPOImitation.load_parameters`），环境是新建的，计数器从 0 开始。

## 后果

从 1e7 的 checkpoint 用 `--init-model` 续训 2e7：

- 网络权重是 1e7 时的；
- 但课程会重新从 20 步起步，按 2e7 的进度重新推进；
- 结果**不等于**连续跑完原定 3e7 课程，不能把两段拼成“完整课程”。

## 可选做法

1. **单进程连续跑满 3e7**：唯一不需要改代码、且课程语义正确的做法。若 1e7 结果支持继续，直接从头跑 3e7（约 18.6 h @448.7），已跑的 1e7 不重复计入课程。
2. **给 checkpoint 增加环境状态**：在保存时写入 `total_step_count`、`max_episode_steps`，恢复时回填。需要改 `train_ppo.py` 与 checkpoint 回调，且要断言回填后的回合上限与解析公式一致；改动后必须用短跑验证计数器单调、不重置。
3. **缩短课程到 2e7**：会改变课程时间尺度，和 1e7 的曲线不可直接比较，不推荐作为默认。

## 当前决定

本轮 `r20261117_single_arm_baseline_1e7` 连续跑完 1e7，不中断、不续训。若结果支持继续，优先选做法 1（整段 3e7），而不是从 checkpoint 续训。
