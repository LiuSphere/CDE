# 微电网模型：maenv_sep.py 对齐说明

`src/envs/maenv.py` 复现桌面 MG 中 `maenv_sep.py` 的单目标数值模型，
并适配当前 FACMAC 的观察、团队标量奖励与 24 步轨迹接口。
模型版本为 `maenv_sep_v1`。

## 数据与观察

三个源 CSV 不做修改。价格、负荷、光伏分别按整张表标准化：

```python
std = sqrt(data.var())
if std < 1e-8:
    std = 1.0
normalized = ((data - data.mean()) / std).astype(float32)
```

标准化结果同时用于观察、功率平衡和成本计算，不会再还原成原始值。
因此标准化后的价格、负荷、光伏可以为负；这不表示原始 CSV 是负值。
每个智能体使用相同的完整数据，不使用 `split_ratio` 分摊。

每个智能体的观察固定为：

```text
[标准化价格, 标准化负荷, 标准化光伏, SOC]
```

当前 3 个智能体对应观察维度 4、全局状态维度 12、动作维度 2。
全局状态为观察的拼接；维度由环境接口传递给 runner、回放、控制器和 learner。
与参考代码相同，`step()` 后返回当前小时的数据与更新后的 SOC，
再推进内部时钟，而不是立即将观察切换到下一小时。

## 动作、电池与功率平衡

动作 `[a_dg, a_batt]` 映射为：

```python
pd = a_dg * dg_max
pb_requested = a_batt * battery_cap
```

正电池功率为放电，负电池功率为充电。每步按参考代码隐含的一小时处理。
没有独立的充放电功率配置。

```python
punishment = -battery_punishment_param * (
    max(pb_requested - soc * battery_cap, 0)
    + max(-(1 - soc) * battery_cap - pb_requested, 0)
)

soc_requested = soc + (
    -raw_ch * pb_requested / battery_cap if pb_requested < 0
    else -raw_dis * pb_requested / battery_cap
)
```

越过下界时，SOC 置为 0.2，实际 `pb = (soc_before - 0.2) * battery_cap`；
越过上界时，SOC 置为 0.8，实际 `pb = (soc_before - 0.8) * battery_cap`。
惩罚仍用修正前的请求功率计算。

这些规则是为了复现参考代码，不是对效率能量平衡的修正。

各智能体独立结算，不使用其他智能体的发电或电池功率：

```python
pg_i = load_normalized - pv_normalized - pd_i - pb_i
buy_reward_i = -pg_i * price_normalized if pg_i > 0 else 0
```

`pg_i <= 0` 时没有售电收入，也不额外惩罚反送。

## 奖励

```python
x = pb + 3 * battery_cap * (1 - soc_after)
bess_reward = -(bessa * x**2 + bessb * x + bessc)
if pb > 0:
    bess_reward *= 0.3

gen_reward = -(costa * pd**2 + costb * pd + costc)
eco_reward = 0.5 * gen_reward + bess_reward + buy_reward
env_reward = -pd * env_param
agent_reward = (eco_reward + env_reward + punishment) * 0.1
team_reward = mean(agent_reward)
```

参考包装器先将单体奖励转成 float32，再计算平均；当前接口保留这一舍入。
不加绝对值电池成本、弃电惩罚、SOC 储备惩罚或期末 SOC 惩罚。
不再读取旧版的可选奖励模式、额外权重或额外奖励缩放。
多目标元组奖励分支不在本次适配内，因为当前 FACMAC 使用团队标量奖励。

## 日期与重置

初始 SOC 为 0.2，训练从第 0 行开始。
训练按时间顺序使用第 0 至第 730 行，之后回到第 0 行。
正常 `reset()` 不随机重置 SOC，日末 SOC 延续到下一天。
显式 `reset_all()` 才随机选择训练起始日，并把 SOC 重新设为 0.2。

当前框架仍将每 24 步存为一段轨迹，但标记为时间截断，不是物理终止。
GRU 隐状态按现有 runner 在每段开头初始化；参考文件本身不定义策略隐状态。

为保留项目的固定日选优功能，参考代码训练区间之后的日期按顺序均分为
validation 与 test，两个集合互不重叠。这是评估适配，不是参考文件的原始划分：
参考文件测试从第 730 行开始，存在训练/测试边界行重叠。
`reset_test(row_idx=..., initial_soc=...)` 仍支持指定日期和初始 SOC；
固定日评估依照现有实验配置，每个指定日独立初始化。
评估前暂存训练日期与 SOC，下次训练恢复，不把测试状态带回训练。

## 参数来源与奖励选择

设备参数与空间声明采用用户提供的
`C:/Users/xyx/Desktop/IPPO4CPT/envs/MG/env_config.json`，
并与同目录的 `env_config.toml` 核对。
两份文件的共有字段一致；JSON 另有 `split_ratio = [0.3, 0.3, 0.4]`，
按原值保存到 YAML，但参考 `maenv_sep.py` 不使用这些比例。
运行仍只加载项目内的 `microgrid.yaml`，不依赖桌面文件路径。

| 参数 | agent1 | agent2 | agent3 |
|---|---|---|---|
| 发电上限 | 225 | 225 | 180 |
| 电池容量 | 100 | 100 | 100 |
| `raw_ch / raw_dis` | 0.98 / 0.98 | 0.98 / 0.98 | 0.98 / 0.98 |
| 电池惩罚系数 | 27.5 | 27.5 | 100 |
| 发电成本 `(a,b,c)` | (0.0083, 5.76, 64) | (0.0079, 5.67, 357) | (0.0096, 5.79, 104) |
| 电池成本 `(a,b,c)` | (0.0152, 5.61, 24) | (0.0158, 5.63, 34) | (0.0172, 5.72, 37) |
| 排放系数 | 1.6481 | 1.6481 | 1.6481 |

观察上下界照原文件保存为 `[0,0,0,-1]` 和 `[1,1,1,1]`。
这份空间声明并不包含所有标准化后的实际观察；为复现原版，不裁剪或重新缩放观察。
动作上下界仍为 `[0,-1]` 和 `[1,1]`。
`security_matrix`、`v_slack`、`v_ref`、`adm_matrix` 按原值保存，
但参考 `maenv_sep.py` 中的安全约束代码被注释，本次也不启用。

**奖励模式已确认**：设备参数采用这两份来源，但按用户选择，
FACMAC 继续使用 `maenv_sep.py` 的单目标奖励分支，
不采用来源文件中的 `multi_target = "True"` 设置：

```python
agent_reward = (eco_reward + env_reward + punishment) * 0.1
team_reward = mean(agent_reward)
```

`env_reward` 只计一次。不是把双目标元组相加，也不新增目标合成权重。
FACMAC 的团队标量奖励接口和 learner 的奖励维度保持不变。

## 日志与历史结果

轨迹仍写到 `results/microgrid_trajs/<algorithm>/<run_id>/`：

- `price/load/pv` 为模型实际使用的标准化值。
- `price_raw/load_raw/pv_raw` 为对应 CSV 原值。
- `pg` 为当前智能体交换功率，`pg_total` 为所有智能体交换功率之和。
- `buy_cost` 为当前智能体的购电成本；负的标准化价格可使其为负。
- `reward` 为已乘 0.1 的单体奖励，`team_reward` 为实际传给 learner 的均值。
- `gen_cost` 为未乘 0.5 的发电成本，`economic_reward` 包含这一系数。
- 所有新轨迹携带 `battery_model_version=maenv_sep_v1`。

旧检查点的观察维度和模型版本不兼容，应重新训练。
不修改现有 CSV、历史训练结果或检查点。
`run_perfect_prediction_oracle.py` 的仿真奖励与本模型对齐，但它只是局部数值搜索，
不是全局最优上界。原有 `run_microgrid_milp_oracle.py` 仍是历史原始数据线性调度模型，
不能作为此模型的同条件最优基准。
