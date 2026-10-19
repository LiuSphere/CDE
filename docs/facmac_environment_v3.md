# FACMAC + EA 学习接口改进：facmac_dispatch_v3

> 更新：更严格的 5 种子、EA/无 EA 完整对照见
> `docs/facmac_multiseed_validation.md`。纯 FACMAC 只有 3/5 个种子改善，
> 并未学会日内储能循环；下文保留最初两种子短检，不应据此宣称适配已成功。

## 本次范围

目标是改善连续策略的可学习性和 EA 适应度比较的有效性，
不是制造只能由 FACMAC + EA 获胜的环境。
仍为三个异构智能体、每个两个连续动作、一个共享 RNN 策略、CPU 训练。
设备容量、发电成本、电价、储能损耗、退化费、互济限制和源 CSV 均未修改。
没有增加活跃奖励、强制循环、免费预测或环境内优化器。

旧版由 tag `microgrid-heterogeneous-exchange-v2` 保存，commit 为 `09e8348`。
环境观察、动作物理含义和回报数值发生版本变化，旧检查点不能直接加载。
本轮更改未自动提交或推送。

## 检查发现的问题

在旧版前三十个训练日、动作种子 4 的 720 小时均匀随机动作中，
28.89% 的智能体电池请求发生硬裁剪。
这意味着相当一部分不同网络输出对应同一个执行功率。
绝对成本回报同时与动作无法控制的基线电费高度相关，
该样本中相关系数约为 0.89。

此外，旧训练循环的同轮 EA 候选依次推进日期和 SOC：
它们在不同负荷、PV、初始储能上比较整日回报，
还叠加各自的高斯动作噪声。不同回报不一定来自策略差异。
这一点不能仅通过环境数值调参解决，因此同步改动了 runner 和训练循环的调用协议；
EA 变异/交叉、学习器、网络结构及训练强度没有修改。

## 四项改动

### 可行功率比例动作

电池动作现在直接缩放当前可行区间：

```text
upper = min(rated_discharge, (SOC - SOC_min) * E * eta_dis / dt)
lower = -min(rated_charge, (SOC_max - SOC) * E / (eta_ch * dt))
pb = action_battery * (upper if action_battery >= 0 else -lower)
```

同一个 SOC 下完整覆盖原有可行物理功率，没有扩大功率或能量限制。
相比额定功率请求后裁剪，避免了可行区间之外的大面积裁剪平台。
SOC 边界不可用方向仍然只能输出零，这是物理限制，不声称消除了所有平台。
合法动作不再产生不可行请求罚款；越 Box 边界、NaN 等仍报错。
MILP 通过逆映射 `battery_action_from_power()` 回放物理功率。

### 成本基线中心化

```text
reward_i = 0.1 * (baseline_cost_i - full_cost_i)
team_reward = mean(float32(reward_i))
```

`baseline_cost` 是相同小时的零发电、零储能动作基线，
包括同样的互济撮合、损耗和输送费；它只依赖外生输入，不依赖动作或 SOC。
同一个外生场景中，新旧负成本回报仅相差一条固定序列，
因此策略之间的未折扣和折扣回报差保持不变。
回归测试显式检查 `gamma=0.99` 的策略差值。

这不是按每天电费除法归一化，也没有给困难日期更低权重。
日志保留原始 `full_cost`、`cost_reward` 和新增的 `baseline_full_cost`、`full_cost_saving`。
新旧版本的回报数值不可直接比较。

在 v3 相同 720 小时随机轨迹上，仅比较两种回报表示：
负成本回报标准差 1.8654，中心化回报标准差 0.8430，
均值分别为 -4.0743 和 0.1968；合法动作裁剪率为 0。
该样本说明目标数值更居中，不代表所有策略和场景必然降低同样幅度的方差。
记录：`results/facmac_env_diagnostics_20261017_b/diagnostic.json`。

### 因果调度提示

观察从 12 维变为 20 维，状态从 36 维变为 60 维。
新增当前可行充放电功率、下一小时/未来四小时的已公布分时电价，
以及未来四小时本地负荷/PV、其他微网净负荷的均值预测特征。

负荷/PV 预测只使用训练行 0-730 的小时均值模板及当前观测比例修正，
倍率限制为 `[0.25,2.0]`；夜间 PV 没有可用比例时用训练均值。
不读取未来真实负荷/PV，也不使用验证/测试统计量构造模板。
预测是当前信息的可复现特征工程，不能消除未来天气不确定性。
分时电价采用现有数据每天相同、提前公布的周期电价假设，代码显式检查。
若换为未知实时电价，必须重新设计预测，而不是沿用该已公布电价输入。

完整观察索引和公式见 `src/config/envs/microgrid_explained.md`。

### 公共 EA 训练场景

每轮 EA 捕获当前训练日与初始 SOC，五个候选均从该场景重置，
候选 rollout 不额外加入高斯动作噪声；种群参数差异仍提供探索。
RL rollout 随后从同一场景开始，保留正常探索，其末尾状态推进训练日期。
待恢复的训练状态也能在验证之后正确捕获，不把验证日期用于适应度。

所有候选 rollout 照常计入 `t_env`、写入 replay。
相同环境步预算下，EA 批次会重复一个日期，独立日期覆盖量与旧版不同；
这是真实采样协议变化，不能宣称所有训练轨迹都与旧版一致。
测试确认同一网络在同场景重复运行的动作和回报逐项相同，
并在真实训练循环中断言五个候选与 RL 使用同一场景。

## 物理等价与集成验证

输出：`results/oracle_milp/facmac_v3_all_test_20261017_a/`。
在全部 182 个测试日重新求解完整场景、无互济、无电池三个联合 MILP，
共 546 次联合优化。既有电池利用、设备活跃、PV 利用及互济收益门槛全部通过。

与 v2 已保存的 182 个完整场景逐日比较，
最优物理完整成本最大变化约为 `7.96e-12` CU。
这支持“学习接口改变、物理最优调度问题未变”的结论。
完整场景求解/回放最大误差约为 `1.85e-11`。
MILP 是带初末 SOC 相等的完美预测参考，不能等同于非循环 RL 回报上界。

45 项聚焦单元测试通过，覆盖：
可行动作逆映射、折扣回报差值不变、零调度回报为零、
未来数据篡改不影响预测特征、训练场景重复与切分保护、
物理守恒、联络约束、MILP、检查点、CPU 和历史模型隔离。

真实入口通过 512 条轨迹预热后，batch=512 的采样观察末两维为 `(3,20)`，
状态维度 60，动作末两维 `(3,2)`，参数和梯度均在 CPU 且有限。
EA 代数 1/3/无EA分别完成 learner 更新 8/8/5 次，EA 总代数 11/33/0。
日志：`results/facmac_v3_integration_20261017_b.log`。

## 短训练对照

每版本分别运行种子 0、1，每次请求 20,000 环境步。
通过 `src/direct_main.py` 加载配置，batch、warmup、更新强度、学习率、
EA 世代数和种群规模均保持默认值。
使用相同六个验证代表日，不使用测试结果挑选学习接口。
种子 1 的 v2 对照直接运行从 tag 导出的源码快照；
种子 0 的 v2 对照在改动前运行完成。

下表为末次公共固定验证步 **16,224**，不是 20,000 步最终策略评估。
指标排除旧版不可行请求罚款，比较购电、发电、排放、吞吐退化和输送的运行成本，
不能把“少交罚款”计作物理经济收益。

| 种子 | v2 RL | v2 EA | v3 RL | v3 EA |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2342.88 | 2331.26 | 2367.47 | 2239.26 |
| 1 | 2360.16 | 2282.27 | 2262.35 | 2263.17 |

单位为六个验证日平均 CU/天。
按相同规则，在每个种子选择 RL/EA 中平均成本更低的一个候选，
v3 相对 v2 的改善分别为 3.95%、0.87%，比值均值为 2.41%。
单看 EA，两种子都改善；单看 RL，种子 0 略差、种子 1 改善。

这只支持早期学习信号改善，不是统计显著性、收敛性或 EA 优越性证明。
只有两个种子，且 12,288 步用于 replay warmup，训练后验证点较少。
观察维度变化也会改变同种子网络初始化，
公共场景改变了日期覆盖量；四项改动没有逐项因子消融。
这些策略不是已经充分利用电池的最终训练策略。

对照文件：

- `results/facmac_environment/v2_seed0_20261017_a/learning_check.json`
- `results/facmac_environment/v2_seed1_20261017_b/learning_check.json`
- `results/facmac_environment/v3_seed0_20261017_a/learning_check.json`
- `results/facmac_environment/v3_seed1_20261017_a/learning_check.json`
- 汇总与曲线：`results/facmac_environment/v2_v3_comparison_20261017_a/`

Windows 上少数 uv 启动因权限错误未启动 Python，不计作实验。
第二种子对照的最初 `_a` 输出还在配置捕获辅助函数的关键字错误处中止，
修复后用全新 `_b` 输出完成。失败记录没有覆盖或当作成功结果。

## 复现与边界

```powershell
uv run python src/direct_main.py --config gavdac_ea --env-config microgrid --override wandb_mode=disabled --override save_model=False --override t_max=12500
uv run python tests/test_corrected_training.py
uv run python tools/benchmark_gavdac_environment.py --output-id <new-id> --timesteps 20000 --seed 0
uv run python tools/run_physical_microgrid_milp.py --all-test-days --output-id <new-id>
```

`benchmark_gavdac_environment.py --source-root <snapshot>` 可以对照旧源码，
不切换当前分支，也不修改归档源码。后续运行保存入口实际收到的合并配置和参数，
所有实验使用新目录，不覆盖旧结果。

未改变 batch=512、buffer=100000、learner 更新频率或 EA 强度来制造改善。
没有自动正式长训练、推送、提交或创建新 tag。
正式结论仍需要更长训练、更多种子和各改动消融。
