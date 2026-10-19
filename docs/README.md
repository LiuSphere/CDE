# 文档索引

当前代码使用 GAVDAC 名称，默认主方法为无交叉的 `gavdac_ea_main`。
安装、配置和运行命令以[项目 README](../README.md)及[算法配置说明](../src/config/algs/README.md)为准。

## 当前说明

| 文档 | 用途 |
| --- | --- |
| [环境与学习接口](facmac_environment_v3.md) | 物理语义、观测、奖励和库存边界 |
| [序列执行与性能](facmac_sequence_execution.md) | 循环历史、梯度边界、等价性验证和计时方法 |
| [W&B 与调度图](wandb_dispatch_showcase.md) | 聚合日志、最终图及展示日筛选限制 |
| [实验计划入口](paper_experiment_plan.md) | 新主方法与旧对照矩阵的适用范围 |
| [实验评估与溯源](paper_execution.md) | 数据使用、模型选择、库存、检查点与结果口径 |
| [交叉探索说明](crossover_parameter_test.md) | 已结束的0.25交叉测试及原始记录入口 |

部分文件沿用 `facmac_` 文件名，以保留既有链接和项目指令中的导航路径；
它们不表示默认算法仍启用交叉。

## 历史审计

以下原始记录放在 `archive/`，仅用于解释模型演变和实验来源，不作为当前执行指令。
旧配置、旧回报及旧环境结果不能直接改名为当前主方法的验证结果。

| 记录 | 保留原因 |
| --- | --- |
| [物理模型 v1](archive/physical_microgrid_validation.md) | 电池能量方程、单位假设及校准来源 |
| [异构微网与互济](archive/heterogeneous_microgrid_validation.md) | 设备参数来源及历史测试集使用证据 |
| [v3 接口开发](archive/environment_v3_development.md) | 动作、观测、奖励表示和采样协议的演变 |
| [环境多种子验证](archive/facmac_multiseed_validation.md) | 保留短训练的有限收益及负面结果 |
| [PPO 奖励设计](archive/sb3_reward_design_validation.md) | 与当前分散策略不同的诊断任务边界 |
| [PPO 奖励迭代](archive/sb3_reward_tuning.md) | 奖励实验及历史测试数据使用披露 |
| [原论文计划](archive/paper_experiment_plan_v03.md) | 含交叉 v03 的原始矩阵，不是新主方法计划 |
| [原执行协议](archive/paper_execution_v03.md) | 已停止队列的预算、筛选和溯源约定 |
| [原交叉探索](archive/crossover_parameter_test_v03.md) | 保留0.25测试的原始假设与比较口径 |

过时的完整源码副本、逐行教学副本、compact 配置说明、旧调参流水和清理日志
已移出文档目录；可从清理备份或 Git 历史恢复。实际运行状态保存在 `results/`，
文档内容本身不授权启动、恢复或扩展实验。
