# 微电网环境：heterogeneous_exchange_v2

当前分支采用异构、物理耦合的合作场景，不再复现 `maenv_sep_v1` 的数值公式。
修改前源码由 tag `maenv-sep-v1-pre-physical-20261017` 保留。
既有结果、源 CSV 和检查点均不改写；旧策略需要重新训练。

## 数据与量纲

三个 CSV 原样读取，以下映射是明确的**实验场景假设**，不是已经核实的源数据单位：

- 总负荷 = `load.csv × load_scale`，当前 `load_scale = 1`。
- 光伏装机场景 = `pv.csv × pv_scale`，当前 `pv_scale = 4`。
- 购电价格 = `price.csv × price_scale`，当前 `price_scale = 0.01`。
- 场景采用 kW、kWh、CU/kWh，CU 是场景货币单位，不冒充美元或人民币。
- 负荷基础权重为 `[0.55, 0.20, 0.25]`，光伏基础权重为 `[0.10, 0.70, 0.20]`。
- 每小时用不同的平滑时段权重重新分配，总负荷和总光伏仍逐小时守恒。
- 有限功率、有损耗的社区互济用于匹配动作执行后的余缺，不允许从大电网购电再转卖。
- 不把 z-score 标准化结果放入功率平衡或购电结算；价格、负荷、光伏必须非负，
  且当前无售电场景要求购电价格严格为正。

每类资源的基础份额不是每日最终能量份额。具体分配公式为：

```text
w_i(h) = share_i × [1 + amplitude_i × cos(2π(h - peak_hour_i)/24)]
resource_i(day,h) = resource_total(day,h) × w_i(h) / sum_j w_j(h)
```

负荷振幅为 `[0.40, 0.35, 0.30]`，权重峰时为 `[20, 12, 5]`；
光伏振幅为 `[0.25, 0.20, 0.25]`，权重峰时为 `[10, 12, 14]`。
这是可复现的合成区域类型，不代表三个真实站点测量，也不改变原 CSV。

每个智能体观察从 4 维增加到 12 维，顺序如下：

```text
[price/0.3, load_i/300, pv_i/300, SOC_i,
 sin(2πh/24), cos(2πh/24),
 sum_other_load/300, sum_other_pv/300,
 sum_other_usable_AC_energy/1200,
 local_dg_max/300, local_battery_capacity/1200, local_exchange_max/300]
```

观察只包含当前时刻信息，没有完美预测。其他微网的可用储能为
`sum_j≠i (SOC_j - SOC_min) × E_j × eta_dis_j`。
这是明示的当前信息广播假设，策略不再是严格无通信的本地控制。
固定缩放不依赖测试集统计量，不裁剪实际观察。
全局状态是三个观察的拼接，维度为 36；每个智能体的动作仍为 2 维连续 Box。
共享 RNN 策略仍由智能体 ID 和不同观察区分角色，没有增加独立策略网络。
`step()` 后观察推进到下一小时，避免以前的小时错位。

## 设备参数

| 参数 | agent1 | agent2 | agent3 |
| --- | ---: | ---: | ---: |
| 角色 | 傍晚负荷型 | 日间光伏型 | 灵活供给型 |
| 发电上限，场景 kW | 110 | 50 | 240 |
| 电池容量，场景 kWh | 220 | 180 | 600 |
| 充电额定功率，场景 kW | 70 | 45 | 120 |
| 放电额定功率，场景 kW | 70 | 45 | 120 |
| 互济发送/接收各自上限，场景 kW | 160 | 240 | 160 |
| 充电 / 放电效率 | 0.93 / 0.93 | 0.93 / 0.93 | 0.93 / 0.93 |
| 吞吐退化成本，CU/kWh | 0.01 | 0.01 | 0.01 |
| 发电二次项，CU/(kW²·h) | 0.00015 | 0.00018 | 0.00012 |
| 发电线性项，CU/kWh | 0.17 | 0.15 | 0.12 |
| 发电常数项，CU/h | 0 | 0 | 0 |
| 发电排放成本代理，CU/kWh | 0.015 | 0.015 | 0.015 |
| 不可行电池请求惩罚，CU/kWh | 0.5 | 0.5 | 0.5 |

SOC 范围 `[0.2, 0.8]`，训练初始 SOC 为 0.35，步长固定一小时。
三块电池名义时长分别约为 3.14、4、5 小时，容量与负荷不再同比例。
两侧效率乘积为 86.49%。这些值属于场景假设，而非设备标定结果。
不引入最小运行时长、爬坡、启停成本或机组最小出力的额外假设。

## 动作与能量守恒

动作 `[a_dg, a_batt]` 范围分别为 `[0,1]` 和 `[-1,1]`。
正电池功率放电，负电池功率充电。

```text
pd = a_dg × dg_max
pb_requested = a_batt × (discharge_rating if a_batt >= 0 else charge_rating)
max_discharge = min(discharge_rating, (SOC - SOC_min) × E × eta_dis / dt)
max_charge = min(charge_rating, (SOC_max - SOC) × E / (eta_ch × dt))
pb = clip(pb_requested, -max_charge, max_discharge)

pb >= 0: SOC_next = SOC - pb × dt / (eta_dis × E)
pb <  0: SOC_next = SOC - eta_ch × pb × dt / E
```

先将功率限制到真实可行范围，再更新 SOC，不再通过事后裁剪 SOC 制造或丢失电量。
请求修正量计入惩罚，NaN、无穷和越动作边界的输入报错。
没有同时充放电的动作路径。

```text
net_i = load_i - pv_i - pd_i - pb_i
supply_i = min(max(-net_i, 0), link_i)
demand_i = min(max(net_i, 0), link_i)
sent_total = min(sum(supply_i), sum(demand_i)/eta_link)
export_i = supply_i/sum(supply) × sent_total
import_i = demand_i/sum(demand) × sent_total × eta_link
pg_i = max(net_i - import_i, 0)
spill_i = max(-net_i - export_i, 0)
pv_i + pd_i + pb_i + pg_i + import_i = load_i + spill_i + export_i
```

总供给或总需求为零时，对应交换量为零，不进行除零。
`eta_link=0.98` 是整段源端到受端的统一交付效率，不是每端各损失 2%。
因此 `sum(import) = 0.98 × sum(export)`。发送量和接收量都分别受本站联络上限约束。
互济按可交换余量/缺额等比例撮合，无未来数据，无每小时优化器，
不会替智能体修改发电或充放电动作。任何合法动作都可以通过大电网补足缺额。
本版本不加入共享变压器上限、负荷削减或交流电压/无功潮流。

`pg` 仅表示购电，无售电收入。`spill` 是未利用的总剩余功率，
一般策略也可能过量发电，不能总把它叫作光伏弃电。
在优化结果没有过量发电的前提下，它可以用于统计未利用光伏。

## 成本与奖励

```text
generation = (costa × pd² + costb × pd + costc) × dt
battery = cycle_cost × abs(pb) × dt
purchase = pg × price × dt
emission = env_param × pd × dt
request_penalty = battery_punishment_param × abs(pb_requested - pb) × dt
exchange = 0.003 × export_i × dt

economic_cost = generation + battery + purchase + exchange
full_cost = economic_cost + emission + request_penalty
agent_reward = -reward_scale × full_cost
team_reward = mean(float32(agent_reward))
```

删除了正放电即乘 0.3 的电池成本折扣，以及与高 SOC 绑定的电池成本多项式。
没有额外的“设备活跃奖励”、吞吐奖励、强制最低放电功率或 SOC 高位奖励。
静置电池没有吞吐成本，循环必须承担效率损失和正的退化成本。
发电成本也不再额外乘 0.5。

## 日期与评估

训练使用行 0–730，验证使用 731–912，测试使用 913–1094，均为零基索引。
正常训练每 24 小时存为一段时间截断轨迹，但日期和 SOC 连续；
`reset_all()` 才随机选训练起点并恢复初始 SOC。
评估前保存训练日期和 SOC，结束后恢复，不消耗训练 RNG。

联合 MILP 使用 `tools/run_physical_microgrid_milp.py`：

- 一起优化三个智能体，成本包括购电、发电、退化、排放和输送费。
- 互济同样受节点功率限制、统一损耗和禁止购电再转供约束。
- 当前统一电价/输送费/损耗下，最优总交换量与环境等比例撮合一致；
  个别节点可能有多个等成本分配，回放核对全队每小时购电、弃电、发送与接收总量。
- 同时逐智能体核对电池功率与 SOC，检查分段线性误差证书。
- 显式充放电互斥，使用 SciPy/HiGHS，不增加依赖。
- 发电二次成本采用 32 段凸弦插值，保留误差界、下界和求解间隙。
- 每个解在当前 `MGEnv` 中逐小时回放，核对能量、成本和约束。
- 每天初始和末尾 SOC 都为 0.35，防止靠免费消耗初始储能节费。
- 日循环约束属于 MILP 的保守参考条件，训练环境仍跨天连续。
  因此不宣称它是任意非循环 RL 策略的严格奖励上界。
- 同日、同初始和末尾 SOC 分别重新优化关闭互济与关闭电池方案，用完整成本比较。
- 互济没有内部买卖电价，源端承担输送费和本地成本；优化的是全队成本，
  不声称实现了对每个独立业主公平的结算机制。

## 参数迭代与验收

物理 v1 的历史校准见 `docs/physical_microgrid_validation.md`。
本次先在六个验证代表日比较九组容量、联络功率、发电成本组合；
首个通过的候选在完整验证集出现两个发电机活跃度边缘案例，
所以转到同一预声明序列中的下一候选。全部 182 个验证日通过后才锁定参数测试。
每个智能体每天需满足：

- SOC 跨度至少覆盖可用窗口的 90%，放电量至少为可用储能电网侧电量的 90%。
- 充、放电各至少两小时，小时功率超过额定值 1% 才计为有效。
- 发电机有效运行至少四小时而非全天，每小时超过额定值 5% 才计数。
- 本地发电和电网分别至少对应本地负荷的 5%；
  全队发电供给负荷 10%–75%，全队电网至少 10%。
  供给型微网可以向外发送，不能再要求其发电量小于本地负荷的 75%。
- 利用至少 90% 的可用光伏。测试日光照变化不受固定光伏发电占比约束。
- 每日完整成本比无电池优化方案至少低 1%，比无互济优化方案至少低 0.5%；
  平均互济成本收益至少 1%，每天送达互济量至少为总负荷的 2%。
- 所有联合/对照 MILP 求解成功，通过日末 SOC、回放和误差证书检查。

详细结果、运行命令与限制见 `docs/heterogeneous_microgrid_validation.md`。
