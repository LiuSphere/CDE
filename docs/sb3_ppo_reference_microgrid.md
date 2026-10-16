# 原版 MG 环境的 SB3 PPO 正式训练

独立运行，不修改现有 FACMAC 入口：

```powershell
uv run python tools/train_sb3_microgrid_ppo.py --timesteps 3000000 --seed 0
```

## 环境来源

原文件为 `C:/Users/xyx/Desktop/MG/MG/maenv_sep.py`，
设备配置为 `C:/Users/xyx/Desktop/IPPO4CPT/envs/MG/env_config.json`。
每次运行把这两个文件和原 MG 目录下的三个 CSV 原样快照到结果目录，
保存 SHA256，并直接实例化快照中的 `MyEnv(multi_target=False)`。
不使用当前项目重写的 `src/envs/maenv.py`。

适配器只做：

- 将三个 4 维观察拼成 12 维，PPO 联合控制 6 维连续动作。
- 将对称 `[-1,1]` 动作线性映射到原来的发电、充放电动作范围。
- 使用原单目标奖励的三个智能体平均值，保留 float32 舍入。
- 把原先不结束的流按 24 步标记为截断，正常训练重置继续原日期和 SOC。
- 修正 Gymnasium 空间声明，使其包含标准化后的负观察值，不裁剪观察。
- 为原代码硬编码的配置路径提供模块内文件读取映射，不修改源文件或进程 cwd。
- 评估时保留原生动作/成本记录，把原生 `data_save()` 的外部写入交给运行目录管理。

电池公式、投影、数据标准化、成本和奖励均直接调用原核心。
绕过原 `MGEnv` 多智能体包装器，是因为其接口不符合 SB3，
且单目标分支的奖励索引不能直接用于三智能体的标量列表。

## 正式训练口径

这是一套中心化 PPO 基线，不是 IPPO，也不声称与 FACMAC 的分散策略等价。
300 万步指联合环境步数，不是单个智能体步数之和。

- CPU，seed 0，单个环境流。
- `n_steps=2400`、`batch_size=240`、每批 `n_epochs=10`。
- 学习率 `3e-4`，折扣 `0.99`，GAE `0.95`，clip `0.2`。
- 策略和价值网络各两层 64 单元。
- 初始训练 SOC 为 0.2，日期按原版 0..730 循环；跨天 SOC 不重置。
- 训练侧用 `VecNormalize(norm_obs=False, norm_reward=True)`。
  这不改变原环境奖励公式；Monitor、验证和典型日 CSV 保存原始奖励。
- 每 24000 步验证一次，验证日为 731..738，测试日为 913..920。
- 验证和测试每个指定日独立从 SOC 0.35 开始，无期末目标。
- 只按验证集原始平均回报选择最佳检查点，不按测试表现选优。
- 0 步初始策略也保留为对照，只有验证结果更好才替换最佳指针。
- 每次更优都保存不可覆盖的策略、回报归一化状态和典型日 CSV。

不会另跑短程 PPO 作为冒烟任务。正式训练前只检查独立环境实例的接口。

## 结果目录

所有文件均在新的 `results/sb3_ppo/formal_<时间>_<后缀>/` 内。
可显式用 `--output` 指定未使用的目录。

- `settings.json`：参数、接口适配、源文件哈希与库版本。
- `reference/`：原始环境、配置和数据快照。
- `status.json`：当前步数、最佳路径与运行状态。
- `train.monitor.csv`：训练每段轨迹的原始回报。
- `logs/progress.csv`：SB3 更新指标。
- `validation_history.jsonl`：固定验证日的各日回报。
- `best.json`：当前最佳检查点指针及验证/测试结果。
- `best_checkpoints/step_*/policy.zip`：不可覆盖的获胜策略。
- 同目录 `vecnormalize.pkl`：与检查点配套的训练回报归一化状态。
- 同目录 `typical_day_trajectories.csv` 和逐日 CSV：全部原生记录。
- 正式结束后，在最佳检查点目录生成 8 天共 16 张图。
- `final_policy.zip`、`final_vecnormalize.pkl`：最终模型，不一定是最佳模型。
- `completed.json`：实际完成步数及获胜策略重新加载的逐小时一致性检查。
- `validation_curve.png`：训练结束后的固定验证曲线。

原环境的标准化数据不具备已核实的物理单位，图和成本不能直接解释为实际电费。
本次 PPO 优化完整的原单目标奖励，与仅优化经济成本的 MILP 目标不同。
