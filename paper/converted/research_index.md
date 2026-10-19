# 强化学习与微电网研究索引

范围：本地 10 篇论文。中文内容为研读归纳，不是原文翻译或复现结论。不重新认定期刊分区、Top 标记或发表日期；预印本/接收稿结论以本地版本为限。

[可搜索网页](research_index.html) | [原全文目录](index.html)

## 主题导航

- 多智能体与协作：[04](#paper-04)、[05](#paper-05)、[06](#paper-06)
- 连续控制与经济调度：[01](#paper-01)、[02](#paper-02)、[03](#paper-03)、[07](#paper-07)、[08](#paper-08)
- 安全、可行性与动作修正：[01](#paper-01)、[03](#paper-03)、[06](#paper-06)、[09](#paper-09)、[10](#paper-10)
- 泛化、扰动与快速适应：[01](#paper-01)、[02](#paper-02)、[03](#paper-03)、[08](#paper-08)
- 消融与收益归因：[02](#paper-02)、[03](#paper-03)、[04](#paper-04)、[09](#paper-09)、[10](#paper-10)
- SOC 与长期库存公平性：[04](#paper-04)、[06](#paper-06)、[07](#paper-07)、[09](#paper-09)
- 可解释性与物理轨迹：[01](#paper-01)、[07](#paper-07)、[09](#paper-09)、[10](#paper-10)

## FACMAC + EA 写作核对

- 本集合不能单独证明 FACMAC + EA 的首创性，尚需更广泛检索。
- 同任务对比需统一观测信息、奖励、设备约束、数据划分与终端库存。
- EA 的种群评估交互步数和墙钟时间必须计入训练预算。
- 分开报告系统配置、市场机制、算法及安全层的贡献。
- 独立训练种子、测试场景和超参数配置数量分别统计。
- 优先组织证据链，不据此直接更改现有算法或实验配置。

<a id="paper-01"></a>
## 01 可解释与电池寿命 PPO

Explainable deep reinforcement learning for resilient and battery-aware microgrid control

Energy Conversion and Management | DOI: 10.1016/j.enconman.2026.121215 | publisher_version | 26 页

[全文](01_Explainable_RL_battery_aware/paper.html) | [原始PDF](../01_ECM_2026_Explainable_RL_battery_aware.pdf) | [Markdown](01_Explainable_RL_battery_aware/paper.md) | [纯文本](01_Explainable_RL_battery_aware/paper.txt) | [逐页截图](01_Explainable_RL_battery_aware/screenshots.html)

标签：单智能体 / 连续动作 / 可解释性 / 电池寿命 / 鲁棒性

- **算法**：集中式单智能体 PPO；课程学习；SHAP 与 LIME 解释策略。
- **任务**：兼顾供电韧性、负荷优先级和电池循环损伤的微电网控制。
- **约束处理**：环境限制 SOC、功率及充放电互斥；允许按优先级削减负荷。
- **实验**：与规则控制和 MPC 比较，并使用大量扰动场景评估韧性与寿命指标。
- **可借鉴点**：参考物理轨迹、韧性与经济性权衡、策略解释的呈现方式。
- **引用边界**：不是 MARL；寿命是模型估计而非实测。扰动场景数不等于训练种子数，分阶段奖励也不能直接横向比较。
- 原文定位（paper.txt 行号，非 PDF 页码）：`SHAP` 第 9 行；`LIME` 第 9 行；`Perlin` 第 69 行

### 章节入口

- [论文首页](01_Explainable_RL_battery_aware/paper.html#_1)
- [全文](01_Explainable_RL_battery_aware/paper.html#_2)
- [Abstract](01_Explainable_RL_battery_aware/paper.html#abstract)
- [1. Introduction](01_Explainable_RL_battery_aware/paper.html#1-introduction)
- [1.1. Deep reinforcement learning for optimal energy management](01_Explainable_RL_battery_aware/paper.html#11-deep-reinforcement-learning-for-optimal-energy-management)
- [1.2. Deep reinforcement learning for enhancing microgrid resilience](01_Explainable_RL_battery_aware/paper.html#12-deep-reinforcement-learning-for-enhancing-microgrid-resilience)
- [1.3. Integrating battery degradation and lifetime](01_Explainable_RL_battery_aware/paper.html#13-integrating-battery-degradation-and-lifetime)
- [1.4. Research gaps and novelties](01_Explainable_RL_battery_aware/paper.html#14-research-gaps-and-novelties)
- [1.5. Paper organisation](01_Explainable_RL_battery_aware/paper.html#15-paper-organisation)
- [2. Modelling of microgrid, control strategy, and explainability techniques](01_Explainable_RL_battery_aware/paper.html#2-modelling-of-microgrid-control-strategy-and-explainability-techniques)
- [2.1. Microgrid description](01_Explainable_RL_battery_aware/paper.html#21-microgrid-description)
- [2.2. Proximal Policy Optimisation](01_Explainable_RL_battery_aware/paper.html#22-proximal-policy-optimisation)
- [2.3. Deep reinforcement learning environment](01_Explainable_RL_battery_aware/paper.html#23-deep-reinforcement-learning-environment)
- [2.4. Curriculum-based training stages](01_Explainable_RL_battery_aware/paper.html#24-curriculum-based-training-stages)
- [2.5. Computational-cost protocol](01_Explainable_RL_battery_aware/paper.html#25-computational-cost-protocol)
- [2.6. Explainability model](01_Explainable_RL_battery_aware/paper.html#26-explainability-model)
- [3. Simulation results and discussions](01_Explainable_RL_battery_aware/paper.html#3-simulation-results-and-discussions)
- [3.3. Deep reinforcement learning -based policy evaluation](01_Explainable_RL_battery_aware/paper.html#33-deep-reinforcement-learning-based-policy-evaluation)
- [3.5. Explainability analysis](01_Explainable_RL_battery_aware/paper.html#35-explainability-analysis)
- [Shapley Additive Explanations Results:](01_Explainable_RL_battery_aware/paper.html#shapley-additive-explanations-results)
- [Local Interpretable Model-agnostic Explanations Results:](01_Explainable_RL_battery_aware/paper.html#local-interpretable-model-agnostic-explanations-results)
- [4. Conclusion](01_Explainable_RL_battery_aware/paper.html#4-conclusion)
- [CRediT authorship contribution statement](01_Explainable_RL_battery_aware/paper.html#credit-authorship-contribution-statement)
- [Declaration of competing interest](01_Explainable_RL_battery_aware/paper.html#declaration-of-competing-interest)
- [Data availability](01_Explainable_RL_battery_aware/paper.html#data-availability)
- [References](01_Explainable_RL_battery_aware/paper.html#references)

<a id="paper-02"></a>
## 02 MoE 多评论家 SAC

Mixture-of-experts based multi-critic deep reinforcement learning for sustainable management of data center microgrids

Applied Energy | DOI: 10.1016/j.apenergy.2026.127561 | publisher_version | 15 页

[全文](02_Mixture_of_experts_multi_critic/paper.html) | [原始PDF](../02_AE_2026_Mixture_of_experts_multi_critic.pdf) | [Markdown](02_Mixture_of_experts_multi_critic/paper.md) | [纯文本](02_Mixture_of_experts_multi_critic/paper.txt) | [逐页截图](02_Mixture_of_experts_multi_critic/screenshots.html)

标签：单智能体 / 连续动作 / 多目标 / MoE / 预测误差

- **算法**：集中式 SAC；MoE 评论家共享专家，并对不同目标设置门控和输出头。
- **任务**：联合调节数据中心工作负载、制冷与电池，兼顾购电、碳排放和任务积压。
- **约束处理**：设备与任务约束结合环境模型和奖励；基础设置使用未来信息，另测观测噪声。
- **实验**：SAC、DDPG 及 MoE 结构变体；按月选择测试日并测试预测误差。
- **可借鉴点**：参考目标分解、结构消融以及预测误差敏感性实验。
- **引用边界**：Multi-critic 不等于 multi-agent；不能当作合作 MARL 证据。比较时需统一预测信息、目标权重和检查点选择规则。
- 原文定位（paper.txt 行号，非 PDF 页码）：`perfect` 第 333 行；`experts` 第 1 行；`Gaussian` 第 491 行

### 章节入口

- [论文首页](02_Mixture_of_experts_multi_critic/paper.html#_1)
- [全文](02_Mixture_of_experts_multi_critic/paper.html#_2)
- [H I G H L I G H T S](02_Mixture_of_experts_multi_critic/paper.html#h-i-g-h-l-i-g-h-t-s)
- [A R T I C L E I N F O](02_Mixture_of_experts_multi_critic/paper.html#a-r-t-i-c-l-e-i-n-f-o)
- [Abstract](02_Mixture_of_experts_multi_critic/paper.html#abstract)
- [1 . Introduction](02_Mixture_of_experts_multi_critic/paper.html#1-introduction)
- [1.1 . Background and motivation](02_Mixture_of_experts_multi_critic/paper.html#11-background-and-motivation)
- [1.2 . Literature review](02_Mixture_of_experts_multi_critic/paper.html#12-literature-review)
- [1.3 . Research gaps and contributions](02_Mixture_of_experts_multi_critic/paper.html#13-research-gaps-and-contributions)
- [1.4 . Paper organization](02_Mixture_of_experts_multi_critic/paper.html#14-paper-organization)
- [2 . Preliminarily of the data center microgrid](02_Mixture_of_experts_multi_critic/paper.html#2-preliminarily-of-the-data-center-microgrid)
- [2.1 . Workload model](02_Mixture_of_experts_multi_critic/paper.html#21-workload-model)
- [2.2 . IT equipment model](02_Mixture_of_experts_multi_critic/paper.html#22-it-equipment-model)
- [2.3 . Cooling model](02_Mixture_of_experts_multi_critic/paper.html#23-cooling-model)
- [2.4 . Energy model](02_Mixture_of_experts_multi_critic/paper.html#24-energy-model)
- [2.5 . Sustainable management](02_Mixture_of_experts_multi_critic/paper.html#25-sustainable-management)
- [3 . Problem formulation as a multi-reward Markov decision process](02_Mixture_of_experts_multi_critic/paper.html#3-problem-formulation-as-a-multi-reward-markov-decision-process)
- [3.1 . Dynamic decision-making process](02_Mixture_of_experts_multi_critic/paper.html#31-dynamic-decision-making-process)
- [3.2 . State](02_Mixture_of_experts_multi_critic/paper.html#32-state)
- [3.3 . Action](02_Mixture_of_experts_multi_critic/paper.html#33-action)
- [3.4 . State transition](02_Mixture_of_experts_multi_critic/paper.html#34-state-transition)
- [3.5 . Reward function](02_Mixture_of_experts_multi_critic/paper.html#35-reward-function)
- [4 . Multi-critic reinforcement learning method](02_Mixture_of_experts_multi_critic/paper.html#4-multi-critic-reinforcement-learning-method)
- [4.1 . Mixture-of-experts based multi-critic soft actor-critic](02_Mixture_of_experts_multi_critic/paper.html#41-mixture-of-experts-based-multi-critic-soft-actor-critic)
- [4.2 . Mixture-of-experts with multi-head architecture for multi-critic networks](02_Mixture_of_experts_multi_critic/paper.html#42-mixture-of-experts-with-multi-head-architecture-for-multi-critic-networks)
- [4.3 . Parameter update for soft actor multi-critic](02_Mixture_of_experts_multi_critic/paper.html#43-parameter-update-for-soft-actor-multi-critic)
- [4.4 . Multi-critic SAC with MoE algorithm pseudocode](02_Mixture_of_experts_multi_critic/paper.html#44-multi-critic-sac-with-moe-algorithm-pseudocode)
- [5 . Case studies](02_Mixture_of_experts_multi_critic/paper.html#5-case-studies)
- [5.1 . Data source](02_Mixture_of_experts_multi_critic/paper.html#51-data-source)
- [5.2 . Analysis of the learning performance of the proposed SAC-MoE algorithm](02_Mixture_of_experts_multi_critic/paper.html#52-analysis-of-the-learning-performance-of-the-proposed-sac-moe-algorithm)
- [5.3 . Analysis of the control behavior of DC microgrid](02_Mixture_of_experts_multi_critic/paper.html#53-analysis-of-the-control-behavior-of-dc-microgrid)
- [5.4 . Analysis of the coupling effects of workload, cooling, and energy systems](02_Mixture_of_experts_multi_critic/paper.html#54-analysis-of-the-coupling-effects-of-workload-cooling-and-energy-systems)
- [5.5 . Sensitivity analysis of the number of experts](02_Mixture_of_experts_multi_critic/paper.html#55-sensitivity-analysis-of-the-number-of-experts)
- [5.6 . Robustness analysis under prediction errors](02_Mixture_of_experts_multi_critic/paper.html#56-robustness-analysis-under-prediction-errors)
- [5.7 . Extension to multi-objective reinforcement learning with dynamic weights](02_Mixture_of_experts_multi_critic/paper.html#57-extension-to-multi-objective-reinforcement-learning-with-dynamic-weights)
- [6 . Conclusions and future work](02_Mixture_of_experts_multi_critic/paper.html#6-conclusions-and-future-work)
- [CRediT authorship contribution statement](02_Mixture_of_experts_multi_critic/paper.html#credit-authorship-contribution-statement)
- [Declaration of competing interest](02_Mixture_of_experts_multi_critic/paper.html#declaration-of-competing-interest)
- [Acknowledgement](02_Mixture_of_experts_multi_critic/paper.html#acknowledgement)
- [Data availability](02_Mixture_of_experts_multi_critic/paper.html#data-availability)
- [References](02_Mixture_of_experts_multi_critic/paper.html#references)

<a id="paper-03"></a>
## 03 铁路双时间尺度 HRL

Hierarchical reinforcement learning for cost-effective railway microgrid management

Applied Energy | DOI: 10.1016/j.apenergy.2026.128698 | publisher_version | 18 页

[全文](03_Hierarchical_RL_railway/paper.html) | [原始PDF](../03_AE_2026_Hierarchical_RL_railway_microgrid.pdf) | [Markdown](03_Hierarchical_RL_railway/paper.md) | [纯文本](03_Hierarchical_RL_railway/paper.txt) | [逐页截图](03_Hierarchical_RL_railway/screenshots.html)

标签：分层强化学习 / 连续动作 / 电压约束 / 消融 / 预测误差

- **算法**：上层 15 分钟选择参考电压范围，下层 30 秒调度储能；SM-DSAC-T。
- **任务**：弱节点铁路微电网的电压调节与运行成本优化。
- **约束处理**：SOC 与功率边界及电压相关惩罚；实验中平均参考电压违约并非严格为零。
- **实验**：无微电网、固定参考、规则、不同预测误差 MPC，以及网强和储能容量敏感性。
- **可借鉴点**：区分设备配置收益、分层框架增益和算法增益；参考适用范围与失效边界分析。
- **引用边界**：相对无微电网约节省 34.7%，相对固定参考方案增量约 3.6%；不能把前者全归因于 HRL。
- 原文定位（paper.txt 行号，非 PDF 页码）：`3.6%` 第 583 行；`Table 6` 第 582 行；`0.18%` 第 582 行

### 章节入口

- [论文首页](03_Hierarchical_RL_railway/paper.html#_1)
- [全文](03_Hierarchical_RL_railway/paper.html#_2)
- [H I G H L I G H T S](03_Hierarchical_RL_railway/paper.html#h-i-g-h-l-i-g-h-t-s)
- [A R T I C L E I N F O](03_Hierarchical_RL_railway/paper.html#a-r-t-i-c-l-e-i-n-f-o)
- [Abstract](03_Hierarchical_RL_railway/paper.html#abstract)
- [1 . Introduction](03_Hierarchical_RL_railway/paper.html#1-introduction)
- [2 . Model formulation](03_Hierarchical_RL_railway/paper.html#2-model-formulation)
- [2.1 . Railway microgrid configuration and energy management system modeling](03_Hierarchical_RL_railway/paper.html#21-railway-microgrid-configuration-and-energy-management-system-modeling)
- [2.1.1 . Objective function](03_Hierarchical_RL_railway/paper.html#211-objective-function)
- [2.1.2 . System constraints](03_Hierarchical_RL_railway/paper.html#212-system-constraints)
- [2.2 . Traction network power flow model](03_Hierarchical_RL_railway/paper.html#22-traction-network-power-flow-model)
- [2.2.1 . Admittance matrix formulation](03_Hierarchical_RL_railway/paper.html#221-admittance-matrix-formulation)
- [2.2.2 . Power flow solution](03_Hierarchical_RL_railway/paper.html#222-power-flow-solution)
- [3 . Proposed hierarchical reinforcement learning framework](03_Hierarchical_RL_railway/paper.html#3-proposed-hierarchical-reinforcement-learning-framework)
- [3.1 . Rationale for reinforcement learning](03_Hierarchical_RL_railway/paper.html#31-rationale-for-reinforcement-learning)
- [3.2 . Markov decision process formulation](03_Hierarchical_RL_railway/paper.html#32-markov-decision-process-formulation)
- [3.3 . Introduction to SM-DSAC-T](03_Hierarchical_RL_railway/paper.html#33-introduction-to-sm-dsac-t)
- [3.3.1 . Standard SAC and DSAC-T](03_Hierarchical_RL_railway/paper.html#331-standard-sac-and-dsac-t)
- [3.3.2 . Introduction to DSAC-T](03_Hierarchical_RL_railway/paper.html#332-introduction-to-dsac-t)
- [3.3.3 . The proposed SM-DSAC-T](03_Hierarchical_RL_railway/paper.html#333-the-proposed-sm-dsac-t)
- [3.3.4 . Theoretical properties of the soft-min operator](03_Hierarchical_RL_railway/paper.html#334-theoretical-properties-of-the-soft-min-operator)
- [4 . Numerical results and analysis](03_Hierarchical_RL_railway/paper.html#4-numerical-results-and-analysis)
- [4.1 . Test system and experimental settings](03_Hierarchical_RL_railway/paper.html#41-test-system-and-experimental-settings)
- [4.1.1 . Test system](03_Hierarchical_RL_railway/paper.html#411-test-system)
- [4.1.2 . Experimental settings](03_Hierarchical_RL_railway/paper.html#412-experimental-settings)
- [4.1.3 . Case study setup](03_Hierarchical_RL_railway/paper.html#413-case-study-setup)
- [4.1.4 . Numerical settings and convergence of the power flow solver](03_Hierarchical_RL_railway/paper.html#414-numerical-settings-and-convergence-of-the-power-flow-solver)
- [4.2 . Learning performance evaluation and learning component ablation](03_Hierarchical_RL_railway/paper.html#42-learning-performance-evaluation-and-learning-component-ablation)
- [4.2.1 . Learning performance evaluation](03_Hierarchical_RL_railway/paper.html#421-learning-performance-evaluation)
- [4.2.2 . Algorithmic and state-representation ablation analysis](03_Hierarchical_RL_railway/paper.html#422-algorithmic-and-state-representation-ablation-analysis)
- [4.3 . Framework-level performance evaluation and ablation analysis](03_Hierarchical_RL_railway/paper.html#43-framework-level-performance-evaluation-and-ablation-analysis)
- [4.4 . Comparison of the proposed framework with model-based and rule-based baselines](03_Hierarchical_RL_railway/paper.html#44-comparison-of-the-proposed-framework-with-model-based-and-rule-based-baselines)
- [4.5 . Voltage regulation performance analysis](03_Hierarchical_RL_railway/paper.html#45-voltage-regulation-performance-analysis)
- [4.6 . Energy scheduling performance analysis](03_Hierarchical_RL_railway/paper.html#46-energy-scheduling-performance-analysis)
- [4.7 . Sensitivity analysis](03_Hierarchical_RL_railway/paper.html#47-sensitivity-analysis)
- [4.7.1 . Sensitivity analysis to load and renewable variability](03_Hierarchical_RL_railway/paper.html#471-sensitivity-analysis-to-load-and-renewable-variability)
- [4.7.2 . Sensitivity analysis to objective function weight settings](03_Hierarchical_RL_railway/paper.html#472-sensitivity-analysis-to-objective-function-weight-settings)
- [4.7.3 . Sensitivity analysis of the smoothing parameter 𝛽](03_Hierarchical_RL_railway/paper.html#473-sensitivity-analysis-of-the-smoothing-parameter)
- [4.7.4 . Sensitivity analysis under different BESS capacities and feeder impedance levels](03_Hierarchical_RL_railway/paper.html#474-sensitivity-analysis-under-different-bess-capacities-and-feeder-impedance-levels)
- [4.8 . Scalability and applicability analysis](03_Hierarchical_RL_railway/paper.html#48-scalability-and-applicability-analysis)
- [5 . Conclusion and discussion](03_Hierarchical_RL_railway/paper.html#5-conclusion-and-discussion)
- [CRediT authorship contribution statement](03_Hierarchical_RL_railway/paper.html#credit-authorship-contribution-statement)
- [Declaration of competing interest](03_Hierarchical_RL_railway/paper.html#declaration-of-competing-interest)
- [Acknowledgements](03_Hierarchical_RL_railway/paper.html#acknowledgements)
- [Data availability](03_Hierarchical_RL_railway/paper.html#data-availability)
- [References](03_Hierarchical_RL_railway/paper.html#references)

<a id="paper-04"></a>
## 04 P2P 市场 LSTM-MAPPO

Multi-agent reinforcement learning-based joint design of low-carbon P2P market and bidding strategy in microgrids

Applied Energy | DOI: 10.1016/j.apenergy.2026.128693 | author_preprint | 10 页

[全文](04_MARL_P2P_market/paper.html) | [原始PDF](../04_AE_2026_MARL_P2P_market_author_preprint.pdf) | [Markdown](04_MARL_P2P_market/paper.md) | [纯文本](04_MARL_P2P_market/paper.txt) | [逐页截图](04_MARL_P2P_market/screenshots.html)

标签：多智能体 / CTDE / P2P市场 / 循环网络 / SOC核算

- **算法**：LSTM-MAPPO；集中训练、分散决策，联合报价、交易量与储能预留。
- **任务**：四个自利微电网参与低碳 P2P 双向拍卖。
- **约束处理**：储能与市场边界；未建模完整 AC 潮流和线路拥塞约束。
- **实验**：分别固定算法比较市场机制、固定市场机制比较学习算法；包含扰动测试。
- **可借鉴点**：参考局部历史建模、CTDE 叙述，以及机制与算法的分开对照。
- **引用边界**：本地文件为预印本；个体利润目标不同于团队共享奖励。终端储能库存不一致时，经济收益需要补充核算。
- 原文定位（paper.txt 行号，非 PDF 页码）：`LSTM` 第 222 行；`MAPPO` 第 25 行；`efficien` 第 9 行

### 章节入口

- [论文首页](04_MARL_P2P_market/paper.html#_1)
- [全文](04_MARL_P2P_market/paper.html#_2)
- [I. INTRODUCTION](04_MARL_P2P_market/paper.html#i-introduction)
- [II. RELATED WORKS](04_MARL_P2P_market/paper.html#ii-related-works)
- [III. PROBLEM STATEMENT](04_MARL_P2P_market/paper.html#iii-problem-statement)
- [A. Microgrid System](04_MARL_P2P_market/paper.html#a-microgrid-system)
- [B. Electricity Market Framework](04_MARL_P2P_market/paper.html#b-electricity-market-framework)
- [C. Intraday P2P Market Mechanism](04_MARL_P2P_market/paper.html#c-intraday-p2p-market-mechanism)
- [IV. MARL-BASED BIDDING STRATEGY IN INTRADAY P2P MARKET](04_MARL_P2P_market/paper.html#iv-marl-based-bidding-strategy-in-intraday-p2p-market)
- [A. Bidding and Energy Storage Strategies for Microgrid i](04_MARL_P2P_market/paper.html#a-bidding-and-energy-storage-strategies-for-microgrid-i)
- [B. DEC-POMDP Formulation](04_MARL_P2P_market/paper.html#b-dec-pomdp-formulation)
- [C. LSTM-MAPPO with CTDE Training Method](04_MARL_P2P_market/paper.html#c-lstm-mappo-with-ctde-training-method)
- [V. NUMERICAL SIMULATION](04_MARL_P2P_market/paper.html#v-numerical-simulation)
- [A. Environment Setup](04_MARL_P2P_market/paper.html#a-environment-setup)
- [B. Baselines](04_MARL_P2P_market/paper.html#b-baselines)
- [C. Parameter and Details of Training](04_MARL_P2P_market/paper.html#c-parameter-and-details-of-training)
- [D. Performance Comparison of Market Clearing Mechanism](04_MARL_P2P_market/paper.html#d-performance-comparison-of-market-clearing-mechanism)
- [E. Performance Comparison of LSTM-MAPPO](04_MARL_P2P_market/paper.html#e-performance-comparison-of-lstm-mappo)
- [VI. CONCLUSION](04_MARL_P2P_market/paper.html#vi-conclusion)
- [ACKNOWLEDGMENT](04_MARL_P2P_market/paper.html#acknowledgment)
- [REFERENCES](04_MARL_P2P_market/paper.html#references)

<a id="paper-05"></a>
## 05 MARLEM 市场仿真框架

MARLEM: A multi-agent reinforcement learning simulation framework for implicit cooperation in decentralized local energy markets

Applied Energy | DOI: 10.1016/j.apenergy.2026.127546 | author_preprint | 32 页

[全文](05_MARLEM/paper.html) | [原始PDF](../05_AE_2026_MARLEM_author_preprint.pdf) | [Markdown](05_MARLEM/paper.md) | [纯文本](05_MARLEM/paper.txt) | [逐页截图](05_MARLEM/screenshots.html)

标签：仿真框架 / P2P市场 / 非学习演示 / 多智能体

- **算法**：模块化多智能体市场平台，设计支持多种训练范式；本版结果使用随机非学习智能体。
- **任务**：研究本地能源市场规则、资产配置及隐式合作的仿真基础设施。
- **约束处理**：市场和设备边界、图网络传输及简化损耗；不是完整 AC 潮流。
- **实验**：七智能体、无电池与配置电池场景的框架演示；完整 MARL 训练留待后续工作。
- **可借鉴点**：用于平台模块、观测与市场指标设计的相关工作。
- **引用边界**：本地文件为预印本。不能引用其结果证明训练后的 MARL 合作优势，也不能据此证明 EA + RL 有效。
- 原文定位（paper.txt 行号，非 PDF 页码）：`zero-intelligence agent baseline` 第 641 行；`full MARL training` 第 624 行；`forthcoming` 第 562 行

### 章节入口

- [论文首页](05_MARLEM/paper.html#_1)
- [全文](05_MARLEM/paper.html#_2)
- [Abstract](05_MARLEM/paper.html#abstract)
- [1 Introduction](05_MARLEM/paper.html#1-introduction)
- [1.1 Motivation](05_MARLEM/paper.html#11-motivation)
- [1.2 Problem statement](05_MARLEM/paper.html#12-problem-statement)
- [1.3 Research gap](05_MARLEM/paper.html#13-research-gap)
- [1.4 Novelty and contribution](05_MARLEM/paper.html#14-novelty-and-contribution)
- [1.5 Paper structure](05_MARLEM/paper.html#15-paper-structure)
- [2 Literature review](05_MARLEM/paper.html#2-literature-review)
- [2.1 Centralized and decentralized energy markets](05_MARLEM/paper.html#21-centralized-and-decentralized-energy-markets)
- [2.2 Multi-agent systems in energy grid management](05_MARLEM/paper.html#22-multi-agent-systems-in-energy-grid-management)
- [2.3 Application and limitations of MARL for DER coordination](05_MARLEM/paper.html#23-application-and-limitations-of-marl-for-der-coordination)
- [2.4 Simulation frameworks for LEMs and MARL](05_MARLEM/paper.html#24-simulation-frameworks-for-lems-and-marl)
- [• Specialized RL environments for energy:](05_MARLEM/paper.html#specialized-rl-environments-for-energy)
- [2.5 Research gap and contribution opportunity](05_MARLEM/paper.html#25-research-gap-and-contribution-opportunity)
- [3 Methodology](05_MARLEM/paper.html#3-methodology)
- [3.1 Simulation framework architecture](05_MARLEM/paper.html#31-simulation-framework-architecture)
- [3.2 Formulation of the decentralized partially observable Markov decision process](05_MARLEM/paper.html#32-formulation-of-the-decentralized-partially-observable-markov-decision-process)
- [3.2.1 Environment state space (S) and transition function (𝑇)](05_MARLEM/paper.html#321-environment-state-space-s-and-transition-function-t)
- [3.2.2 Joint observation space (Ω)](05_MARLEM/paper.html#322-joint-observation-space)
- [3.2.3 Action space (𝐴)](05_MARLEM/paper.html#323-action-space-a)
- [3.2.4 Reward function (𝑅)](05_MARLEM/paper.html#324-reward-function-r)
- [3.3 LEM design and trading mechanisms](05_MARLEM/paper.html#33-lem-design-and-trading-mechanisms)
- [3.3.1 Role of the distribution system operator](05_MARLEM/paper.html#331-role-of-the-distribution-system-operator)
- [3.3.2 Multi-stage order matching process](05_MARLEM/paper.html#332-multi-stage-order-matching-process)
- [3.3.3 Clearing mechanisms](05_MARLEM/paper.html#333-clearing-mechanisms)
- [3.3.4 Reputation and decentralized validation](05_MARLEM/paper.html#334-reputation-and-decentralized-validation)
- [3.4 Agent model and decision-making process](05_MARLEM/paper.html#34-agent-model-and-decision-making-process)
- [3.4.1 Agent architecture and state management](05_MARLEM/paper.html#341-agent-architecture-and-state-management)
- [3.4.2 Physically constrained decision-making](05_MARLEM/paper.html#342-physically-constrained-decision-making)
- [3.4.3 Battery storage dynamics](05_MARLEM/paper.html#343-battery-storage-dynamics)
- [3.4.4 Time series profiles](05_MARLEM/paper.html#344-time-series-profiles)
- [3.5 Physical grid model and constraints](05_MARLEM/paper.html#35-physical-grid-model-and-constraints)
- [3.5.1 Grid topology and agent allocation](05_MARLEM/paper.html#351-grid-topology-and-agent-allocation)
- [3.5.2 Modeling physical constraints](05_MARLEM/paper.html#352-modeling-physical-constraints)
- [3.5.3 Dynamic grid state updates and management](05_MARLEM/paper.html#353-dynamic-grid-state-updates-and-management)
- [3.5.4 Zonal grid structure](05_MARLEM/paper.html#354-zonal-grid-structure)
- [3.6 Implicit cooperation and performance measurement](05_MARLEM/paper.html#36-implicit-cooperation-and-performance-measurement)
- [3.6.1 The coordination feedback loop](05_MARLEM/paper.html#361-the-coordination-feedback-loop)
- [3.6.2 Key performance indicators](05_MARLEM/paper.html#362-key-performance-indicators)
- [3. Resource coordination KPIs: These metrics evaluate how effectively DERs are utilized and coordinated.](05_MARLEM/paper.html#3-resource-coordination-kpis-these-metrics-evaluate-how-effectively-ders-are-utilized-and-coordinated)
- [3.7 Multi-agent reinforcement learning framework](05_MARLEM/paper.html#37-multi-agent-reinforcement-learning-framework)
- [3.7.1 Training and execution paradigms](05_MARLEM/paper.html#371-training-and-execution-paradigms)
- [3.7.2 Reinforcement learning algorithms and infrastructure](05_MARLEM/paper.html#372-reinforcement-learning-algorithms-and-infrastructure)
- [3.8 Analytics and visualizations](05_MARLEM/paper.html#38-analytics-and-visualizations)
- [3.9 Complete simulation workflow](05_MARLEM/paper.html#39-complete-simulation-workflow)
- [4 Experimental setup](05_MARLEM/paper.html#4-experimental-setup)
- [4.1 Case study: battery storage coordination](05_MARLEM/paper.html#41-case-study-battery-storage-coordination)
- [4.2 Baseline models](05_MARLEM/paper.html#42-baseline-models)
- [4.3 Other implemented case studies](05_MARLEM/paper.html#43-other-implemented-case-studies)
- [5 Results and discussion](05_MARLEM/paper.html#5-results-and-discussion)
- [5.1 Framework capability demonstration](05_MARLEM/paper.html#51-framework-capability-demonstration)
- [5.1.1 Price volatility and market stability](05_MARLEM/paper.html#511-price-volatility-and-market-stability)
- [5.1.2 Grid stability and system coordination](05_MARLEM/paper.html#512-grid-stability-and-system-coordination)
- [5.1.3 Market structure and economic efficiency](05_MARLEM/paper.html#513-market-structure-and-economic-efficiency)
- [5.2 Discussion](05_MARLEM/paper.html#52-discussion)
- [6 Conclusion and future work](05_MARLEM/paper.html#6-conclusion-and-future-work)
- [References](05_MARLEM/paper.html#references)

<a id="paper-06"></a>
## 06 离散 DQN 修正预测 + MPEC

Lean multi-agent deep reinforcement learning for uncertainty handling in the energy management of networked microgrids

Applied Energy | DOI: 10.1016/j.apenergy.2026.127354 | publisher_version | 31 页

[全文](06_Lean_MARL_networked_microgrids/paper.html) | [原始PDF](../1-s2.0-S0306261926000061-main.pdf) | [Markdown](06_Lean_MARL_networked_microgrids/paper.md) | [纯文本](06_Lean_MARL_networked_microgrids/paper.txt) | [逐页截图](06_Lean_MARL_networked_microgrids/screenshots.html)

标签：多智能体 / 离散动作 / 优化求解器 / 网络约束 / 不确定性

- **算法**：多智能体 DQN 学习离散预测误差修正系数，再由双层 MPEC / MILP 求解调度。
- **任务**：联网微电网的不确定性处理、零售商收益与微电网运行优化。
- **约束处理**：优化模型处理设备、终端 SOC 和网络约束；不可行修正受到惩罚。
- **实验**：5 节点与 14 节点系统，与确定性及随机优化对照；测试日覆盖较窄。
- **可借鉴点**：参考不确定性处理、模型与学习分工、终端 SOC 约束及网络指标。
- **引用边界**：不是连续动作端到端调度，测试时仍需优化求解器；不能直接作为当前 FACMAC 的同任务基线。
- 原文定位（paper.txt 行号，非 PDF 页码）：`MPEC` 第 25 行；`July` 第 421 行

### 章节入口

- [论文首页](06_Lean_MARL_networked_microgrids/paper.html#_1)
- [全文](06_Lean_MARL_networked_microgrids/paper.html#_2)
- [H I G H L I G H T S](06_Lean_MARL_networked_microgrids/paper.html#h-i-g-h-l-i-g-h-t-s)
- [G R A P H I C A L A B S T R A C T](06_Lean_MARL_networked_microgrids/paper.html#g-r-a-p-h-i-c-a-l-a-b-s-t-r-a-c-t)
- [A R T I C L E I N F O](06_Lean_MARL_networked_microgrids/paper.html#a-r-t-i-c-l-e-i-n-f-o)
- [Abstract](06_Lean_MARL_networked_microgrids/paper.html#abstract)
- [1 . Introduction](06_Lean_MARL_networked_microgrids/paper.html#1-introduction)
- [2 . Deep reinforcement learning (DRL)](06_Lean_MARL_networked_microgrids/paper.html#2-deep-reinforcement-learning-drl)
- [2.1 . Deep Q-networks (DQN)](06_Lean_MARL_networked_microgrids/paper.html#21-deep-q-networks-dqn)
- [3 . Problem formulation](06_Lean_MARL_networked_microgrids/paper.html#3-problem-formulation)
- [3.1 . Upper-level problem](06_Lean_MARL_networked_microgrids/paper.html#31-upper-level-problem)
- [3.2 . Lower-level problem](06_Lean_MARL_networked_microgrids/paper.html#32-lower-level-problem)
- [3.2.1 . KKT formulations of lower-level problem](06_Lean_MARL_networked_microgrids/paper.html#321-kkt-formulations-of-lower-level-problem)
- [4 . Lean multi-agent DRL (L-MADRL framework)](06_Lean_MARL_networked_microgrids/paper.html#4-lean-multi-agent-drl-l-madrl-framework)
- [4.1 . Training process](06_Lean_MARL_networked_microgrids/paper.html#41-training-process)
- [4.1.1 . States](06_Lean_MARL_networked_microgrids/paper.html#411-states)
- [4.1.2 . Actions](06_Lean_MARL_networked_microgrids/paper.html#412-actions)
- [4.1.3 . Rewards](06_Lean_MARL_networked_microgrids/paper.html#413-rewards)
- [4.2 . Testing process](06_Lean_MARL_networked_microgrids/paper.html#42-testing-process)
- [5 . Two-stage stochastic optimization](06_Lean_MARL_networked_microgrids/paper.html#5-two-stage-stochastic-optimization)
- [5.1 . Uncertainty modelling](06_Lean_MARL_networked_microgrids/paper.html#51-uncertainty-modelling)
- [5.2 . Risk-neutral stochastic optimization (RNSO)](06_Lean_MARL_networked_microgrids/paper.html#52-risk-neutral-stochastic-optimization-rnso)
- [5.3 . Risk-aware stochastic optimization (RASO)](06_Lean_MARL_networked_microgrids/paper.html#53-risk-aware-stochastic-optimization-raso)
- [6 . Case study](06_Lean_MARL_networked_microgrids/paper.html#6-case-study)
- [6.1 . Case 1: PJM 5 bus network](06_Lean_MARL_networked_microgrids/paper.html#61-case-1-pjm-5-bus-network)
- [6.2 . Case 2: IEEE 14 bus network](06_Lean_MARL_networked_microgrids/paper.html#62-case-2-ieee-14-bus-network)
- [7 . Conclusion and future work](06_Lean_MARL_networked_microgrids/paper.html#7-conclusion-and-future-work)
- [CRediT authorship contribution statement](06_Lean_MARL_networked_microgrids/paper.html#credit-authorship-contribution-statement)
- [Funding](06_Lean_MARL_networked_microgrids/paper.html#funding)
- [Declaration of competing interest](06_Lean_MARL_networked_microgrids/paper.html#declaration-of-competing-interest)
- [Appendix A](06_Lean_MARL_networked_microgrids/paper.html#appendix-a)
- [Data availability](06_Lean_MARL_networked_microgrids/paper.html#data-availability)
- [References](06_Lean_MARL_networked_microgrids/paper.html#references)

<a id="paper-07"></a>
## 07 DDPG 经济状态特征 EMS

Deep reinforcement learning-based multi-objective energy management system for microgrids under flexible energy market

Applied Energy | DOI: 10.1016/j.apenergy.2025.127223 | publisher_version | 10 页

[全文](07_Multi_objective_EMS/paper.html) | [原始PDF](../1-s2.0-S0306261925019531-main.pdf) | [Markdown](07_Multi_objective_EMS/paper.md) | [纯文本](07_Multi_objective_EMS/paper.txt) | [逐页截图](07_Multi_objective_EMS/screenshots.html)

标签：单智能体 / 连续动作 / 经济特征 / 连续多日 / SOC核算

- **算法**：集中式 DDPG，加入 EPEM 与储能购电成本状态；配合超级电容快速缓冲。
- **任务**：灵活电价下的微电网多目标运行和能量管理。
- **约束处理**：设备动作边界与底层控制；需要区分储能系统配置和策略本身。
- **实验**：冬夏分别训练，连续十天评估；比较 PPO、PSO、LP-MPC 与无储能方案。
- **可借鉴点**：参考连续多日轨迹、日成本表和领域经济状态特征。
- **引用边界**：10.2% 与 33% 节省相对无储能，并非相对 MPC。冬夏分别训练不等于单策略跨季节泛化。
- 原文定位（paper.txt 行号，非 PDF 页码）：`Table 4` 第 309 行；`Non-ES` 第 313 行；`10.2` 第 310 行

### 章节入口

- [论文首页](07_Multi_objective_EMS/paper.html#_1)
- [全文](07_Multi_objective_EMS/paper.html#_2)
- [H I G H L I G H T S](07_Multi_objective_EMS/paper.html#h-i-g-h-l-i-g-h-t-s)
- [A R T I C L E I N F O](07_Multi_objective_EMS/paper.html#a-r-t-i-c-l-e-i-n-f-o)
- [Abstract](07_Multi_objective_EMS/paper.html#abstract)
- [1 . Introduction](07_Multi_objective_EMS/paper.html#1-introduction)
- [2 . The topology of the proposed RLEMS and MG](07_Multi_objective_EMS/paper.html#2-the-topology-of-the-proposed-rlems-and-mg)
- [3 . The proposed energy price evaluation model](07_Multi_objective_EMS/paper.html#3-the-proposed-energy-price-evaluation-model)
- [3.1 . Dynamic energy price evaluation](07_Multi_objective_EMS/paper.html#31-dynamic-energy-price-evaluation)
- [3.2 . MG environment status evaluation](07_Multi_objective_EMS/paper.html#32-mg-environment-status-evaluation)
- [4 . The framework of the reinforcement learning-based intelligent EMS agent](07_Multi_objective_EMS/paper.html#4-the-framework-of-the-reinforcement-learning-based-intelligent-ems-agent)
- [4.1 . Environment reformulation](07_Multi_objective_EMS/paper.html#41-environment-reformulation)
- [4.2 . Reward function](07_Multi_objective_EMS/paper.html#42-reward-function)
- [4.3 . Reinforcement learning algorithm](07_Multi_objective_EMS/paper.html#43-reinforcement-learning-algorithm)
- [5 . Numerical experiments](07_Multi_objective_EMS/paper.html#5-numerical-experiments)
- [5.1 . Performance of the proposed RLEMS in different scenarios](07_Multi_objective_EMS/paper.html#51-performance-of-the-proposed-rlems-in-different-scenarios)
- [5.2 . Ablation experimental validation](07_Multi_objective_EMS/paper.html#52-ablation-experimental-validation)
- [5.3 . Comparative experiments](07_Multi_objective_EMS/paper.html#53-comparative-experiments)
- [6 . Conclusions](07_Multi_objective_EMS/paper.html#6-conclusions)
- [CRediT authorship contribution statement](07_Multi_objective_EMS/paper.html#credit-authorship-contribution-statement)
- [Declaration of competing interest](07_Multi_objective_EMS/paper.html#declaration-of-competing-interest)
- [Appendix](07_Multi_objective_EMS/paper.html#appendix)
- [Data availability](07_Multi_objective_EMS/paper.html#data-availability)
- [References](07_Multi_objective_EMS/paper.html#references)
- [Author biography](07_Multi_objective_EMS/paper.html#author-biography)

<a id="paper-08"></a>
## 08 Reptile 元学习 + PPO

Meta-reinforcement learning for multi-energy building microgrids: Curriculum design, rapid adaptation, and scalable deployment

Energy Conversion and Management | DOI: 10.1016/j.enconman.2026.121652 | publisher_version | 22 页

[全文](08_Meta_RL_building_microgrids/paper.html) | [原始PDF](../1-s2.0-S0196890426006217-main.pdf) | [Markdown](08_Meta_RL_building_microgrids/paper.md) | [纯文本](08_Meta_RL_building_microgrids/paper.txt) | [逐页截图](08_Meta_RL_building_microgrids/screenshots.html)

标签：单智能体 / 连续动作 / 元强化学习 / 课程学习 / 泛化

- **算法**：增强一阶 Reptile 外循环，PPO 内循环；课程逐步增加建筑与季节多样性。
- **任务**：多能源建筑微电网的快速适应与跨建筑部署。
- **约束处理**：设备硬边界与能量平衡软惩罚；偏好 SOC 区间不同于硬边界。
- **实验**：五栋未见建筑、四季以及少量适应回合；比较适应前后和不同课程。
- **可借鉴点**：参考未见场景划分、适应预算、性能与约束/购电依赖之间的权衡。
- **引用边界**：不是社区 MARL；12.2% 成本下降是适应后相对适应前，非相对 MPC。多个模型配置不等于独立训练种子。
- 原文定位（paper.txt 行号，非 PDF 页码）：`12.2%` 第 11 行；`five unseen` 第 11 行；`Wilcoxon` 第 11 行

### 章节入口

- [论文首页](08_Meta_RL_building_microgrids/paper.html#_1)
- [全文](08_Meta_RL_building_microgrids/paper.html#_2)
- [A R T I C L E I N F O](08_Meta_RL_building_microgrids/paper.html#a-r-t-i-c-l-e-i-n-f-o)
- [Abstract](08_Meta_RL_building_microgrids/paper.html#abstract)
- [1. Introduction](08_Meta_RL_building_microgrids/paper.html#1-introduction)
- [1.1. Literature review](08_Meta_RL_building_microgrids/paper.html#11-literature-review)
- [1.2. Contributions](08_Meta_RL_building_microgrids/paper.html#12-contributions)
- [2. Methodology](08_Meta_RL_building_microgrids/paper.html#2-methodology)
- [2.1. Meta-reinforcement learning](08_Meta_RL_building_microgrids/paper.html#21-meta-reinforcement-learning)
- [3. Multi-energy building microgrid](08_Meta_RL_building_microgrids/paper.html#3-multi-energy-building-microgrid)
- [3.1. Multi-energy system modeling](08_Meta_RL_building_microgrids/paper.html#31-multi-energy-system-modeling)
- [3.1.1. Photovoltaic generation modeling](08_Meta_RL_building_microgrids/paper.html#311-photovoltaic-generation-modeling)
- [3.1.2. Battery energy storage systems](08_Meta_RL_building_microgrids/paper.html#312-battery-energy-storage-systems)
- [3.1.3. Heatpump](08_Meta_RL_building_microgrids/paper.html#313-heatpump)
- [3.2. State space design and observability](08_Meta_RL_building_microgrids/paper.html#32-state-space-design-and-observability)
- [3.2.1. Core system state variables](08_Meta_RL_building_microgrids/paper.html#321-core-system-state-variables)
- [3.2.2. Temporal and contextual features](08_Meta_RL_building_microgrids/paper.html#322-temporal-and-contextual-features)
- [3.2.3. Historical action feedback](08_Meta_RL_building_microgrids/paper.html#323-historical-action-feedback)
- [3.3. Action space formulation](08_Meta_RL_building_microgrids/paper.html#33-action-space-formulation)
- [3.4. Multi-objective reward function design](08_Meta_RL_building_microgrids/paper.html#34-multi-objective-reward-function-design)
- [3.4.1. Economic reward component](08_Meta_RL_building_microgrids/paper.html#341-economic-reward-component)
- [3.4.2. Energy balance and system stability](08_Meta_RL_building_microgrids/paper.html#342-energy-balance-and-system-stability)
- [3.4.3. State-of-charge maintenance reward function](08_Meta_RL_building_microgrids/paper.html#343-state-of-charge-maintenance-reward-function)
- [3.4.4. Renewable energy maximization reward](08_Meta_RL_building_microgrids/paper.html#344-renewable-energy-maximization-reward)
- [3.4.5. Action correction and constraint guidance](08_Meta_RL_building_microgrids/paper.html#345-action-correction-and-constraint-guidance)
- [3.4.6. Composite reward function](08_Meta_RL_building_microgrids/paper.html#346-composite-reward-function)
- [4. Meta-learning framework](08_Meta_RL_building_microgrids/paper.html#4-meta-learning-framework)
- [4.1. Data engineering for meta-learning task](08_Meta_RL_building_microgrids/paper.html#41-data-engineering-for-meta-learning-task)
- [4.1.1. Curriculum learning](08_Meta_RL_building_microgrids/paper.html#411-curriculum-learning)
- [4.2. Meta-learning training](08_Meta_RL_building_microgrids/paper.html#42-meta-learning-training)
- [5. Results & discussions](08_Meta_RL_building_microgrids/paper.html#5-results-discussions)
- [5.1. Hyperparameter tuning](08_Meta_RL_building_microgrids/paper.html#51-hyperparameter-tuning)
- [5.1.1. Inner-loop hyperparameters](08_Meta_RL_building_microgrids/paper.html#511-inner-loop-hyperparameters)
- [5.1.2. Outer-loop hyperparameters](08_Meta_RL_building_microgrids/paper.html#512-outer-loop-hyperparameters)
- [5.2. Model performance](08_Meta_RL_building_microgrids/paper.html#52-model-performance)
- [5.2.1. Curriculum performance](08_Meta_RL_building_microgrids/paper.html#521-curriculum-performance)
- [5.3. Model adaptation](08_Meta_RL_building_microgrids/paper.html#53-model-adaptation)
- [5.3.1. Deployment considerations](08_Meta_RL_building_microgrids/paper.html#531-deployment-considerations)
- [5.4. Limitations & future directions](08_Meta_RL_building_microgrids/paper.html#54-limitations-future-directions)
- [5.4.1. Reward design](08_Meta_RL_building_microgrids/paper.html#541-reward-design)
- [5.4.2. Curriculum engineering](08_Meta_RL_building_microgrids/paper.html#542-curriculum-engineering)
- [5.4.3. Framework applicability](08_Meta_RL_building_microgrids/paper.html#543-framework-applicability)
- [5.4.4. Peer-to-peer energy trading](08_Meta_RL_building_microgrids/paper.html#544-peer-to-peer-energy-trading)
- [6. Conclusion](08_Meta_RL_building_microgrids/paper.html#6-conclusion)
- [CRediT authorship contribution statement](08_Meta_RL_building_microgrids/paper.html#credit-authorship-contribution-statement)
- [Declaration of competing interest](08_Meta_RL_building_microgrids/paper.html#declaration-of-competing-interest)
- [Acknowledgments](08_Meta_RL_building_microgrids/paper.html#acknowledgments)
- [Appendix. Reinforcement learning](08_Meta_RL_building_microgrids/paper.html#appendix-reinforcement-learning)
- [Data availability](08_Meta_RL_building_microgrids/paper.html#data-availability)
- [References](08_Meta_RL_building_microgrids/paper.html#references)

<a id="paper-09"></a>
## 09 规则语义掩码 DQN

Deep reinforcement learning strategy integrating environmental perception and semantic masking for real-time operational optimization of building cluster microgrids

Energy | DOI: 10.1016/j.energy.2026.140676 | accepted_manuscript | 40 页

[全文](09_Perception_semantic_masking/paper.html) | [原始PDF](../Deep%20Reinforcement%20Learning%20Strategy%20Integrating%20Environmental%20Perception%20and%20Semantic%20Masking%20for%20Real-Time%20Operational%20Optimization%20of%20Building%20Cluster%20Microgrids.pdf) | [Markdown](09_Perception_semantic_masking/paper.md) | [纯文本](09_Perception_semantic_masking/paper.txt) | [逐页截图](09_Perception_semantic_masking/screenshots.html)

标签：单智能体 / 离散动作 / 动作掩码 / 规则知识 / SOC核算

- **算法**：集中式离散 DQN；物理可行性筛选叠加电价、SOC 和削峰规则形成候选动作集。
- **任务**：虚拟建筑群、光伏、电池与 EV 充电负荷的三十天运行优化。
- **约束处理**：动作掩码与执行前修正；通过奖励约束日终 SOC，兼顾应急储备。
- **实验**：不同动作步长下与原始 DQN、PSO、WOA、DE 比较；观测消融和边界触发统计。
- **可借鉴点**：参考原始/实际执行动作差异、边界触发、日终库存和经济启发式消融。
- **引用边界**：本地文件为接收稿，第 1 页是机构封面。semantic 指人工规则而非大语言模型；收益混合了物理筛选与经济规则贡献。
- 原文定位（paper.txt 行号，非 PDF 页码）：`Table 6` 第 364 行；`Table 7` 第 396 行；`semantic rules` 第 60 行

### 章节入口

- [论文首页](09_Perception_semantic_masking/paper.html#_1)
- [全文](09_Perception_semantic_masking/paper.html#_2)
- [Please note:](09_Perception_semantic_masking/paper.html#please-note)
- [Abstract](09_Perception_semantic_masking/paper.html#abstract)
- [Keywords:](09_Perception_semantic_masking/paper.html#keywords)
- [1.Introduction](09_Perception_semantic_masking/paper.html#1introduction)
- [1.1 Background](09_Perception_semantic_masking/paper.html#11-background)
- [1.2 Literature Review](09_Perception_semantic_masking/paper.html#12-literature-review)
- [1.3 Research Objectives](09_Perception_semantic_masking/paper.html#13-research-objectives)
- [2. Data Analysis and Modeling Process](09_Perception_semantic_masking/paper.html#2-data-analysis-and-modeling-process)
- [2.1 Energy Consumption Data Sources and Analysis for Building Cluster Microgrids](09_Perception_semantic_masking/paper.html#21-energy-consumption-data-sources-and-analysis-for-building-cluster-microgrids)
- [Fig.3 Number of charging piles and average cumulative energy consumption](09_Perception_semantic_masking/paper.html#fig3-number-of-charging-piles-and-average-cumulative-energy-consumption)
- [2.2 Real-Time Data Generation Process for Charging Piles](09_Perception_semantic_masking/paper.html#22-real-time-data-generation-process-for-charging-piles)
- [2.3 Modeling of PV Energy Storage Microgrids](09_Perception_semantic_masking/paper.html#23-modeling-of-pv-energy-storage-microgrids)
- [3. Operation Optimization Methods](09_Perception_semantic_masking/paper.html#3-operation-optimization-methods)
- [3.1 DRL Settings](09_Perception_semantic_masking/paper.html#31-drl-settings)
- [(1) State observation](09_Perception_semantic_masking/paper.html#1-state-observation)
- [(2) Action selection](09_Perception_semantic_masking/paper.html#2-action-selection)
- [(3) Reward setting](09_Perception_semantic_masking/paper.html#3-reward-setting)
- [3.2 Environmental Perception and Semantic Masking DQN](09_Perception_semantic_masking/paper.html#32-environmental-perception-and-semantic-masking-dqn)
- [(1) Deep Q-networks](09_Perception_semantic_masking/paper.html#1-deep-q-networks)
- [(2) Environmental perception and semantic masking](09_Perception_semantic_masking/paper.html#2-environmental-perception-and-semantic-masking)
- [(3) Semantic mask DQN](09_Perception_semantic_masking/paper.html#3-semantic-mask-dqn)
- [3.3 Experimental Setup](09_Perception_semantic_masking/paper.html#33-experimental-setup)
- [4. Results](09_Perception_semantic_masking/paper.html#4-results)
- [4.1 EV Electricity Consumption Data Generation Results](09_Perception_semantic_masking/paper.html#41-ev-electricity-consumption-data-generation-results)
- [4.2 Analysis of Optimization Results](09_Perception_semantic_masking/paper.html#42-analysis-of-optimization-results)
- [4.3 Operation Process Analysis](09_Perception_semantic_masking/paper.html#43-operation-process-analysis)
- [5. Discussion](09_Perception_semantic_masking/paper.html#5-discussion)
- [6. Conclusion](09_Perception_semantic_masking/paper.html#6-conclusion)
- [Acknowledgment](09_Perception_semantic_masking/paper.html#acknowledgment)
- [References](09_Perception_semantic_masking/paper.html#references)

<a id="paper-10"></a>
## 10 安全分层 DDQN + TD3

Toward resilient distribution system via microgrid formation: A safe hierarchical hybrid reinforcement learning approach

Applied Energy | DOI: 10.1016/j.apenergy.2026.128001 | publisher_version | 16 页

[全文](10_Safe_hierarchical_hybrid_RL/paper.html) | [原始PDF](../1-s2.0-S0306261926006537-main.pdf) | [Markdown](10_Safe_hierarchical_hybrid_RL/paper.md) | [纯文本](10_Safe_hierarchical_hybrid_RL/paper.txt) | [逐页截图](10_Safe_hierarchical_hybrid_RL/screenshots.html)

标签：分层强化学习 / 混合动作 / 动作掩码 / 安全投影 / 消融

- **算法**：DDQN 选择开关拓扑，TD3 调节 DER；离散掩码、QP 安全投影和多优先经验回放。
- **任务**：停电后形成孤岛微电网、恢复配电系统负荷。
- **约束处理**：规则过滤拓扑动作，模型驱动 QP 修正连续动作；保证范围取决于模型及可行性假设。
- **实验**：123/356 节点；混合动作基线、MILP、MPER 组件消融、安全层触发与修正幅度。
- **可借鉴点**：参考安全层介入频率、修正范数、模块消融和包含求解器的端到端时延。
- **引用边界**：不是纯连续动作合作调度；测试投影触发约 11.14%，不代表裸策略始终安全。正文与表格部分时延数字不一致，引用需核对。
- 原文定位（paper.txt 行号，非 PDF 页码）：`11.14%` 第 477 行；`Table 1` 第 479 行；`Table 3` 第 499 行

### 章节入口

- [论文首页](10_Safe_hierarchical_hybrid_RL/paper.html#_1)
- [全文](10_Safe_hierarchical_hybrid_RL/paper.html#_2)
- [H I G H L I G H T S](10_Safe_hierarchical_hybrid_RL/paper.html#h-i-g-h-l-i-g-h-t-s)
- [A R T I C L E I N F O](10_Safe_hierarchical_hybrid_RL/paper.html#a-r-t-i-c-l-e-i-n-f-o)
- [Abstract](10_Safe_hierarchical_hybrid_RL/paper.html#abstract)
- [1. Introduction](10_Safe_hierarchical_hybrid_RL/paper.html#1-introduction)
- [1.1. Background](10_Safe_hierarchical_hybrid_RL/paper.html#11-background)
- [1.2. Literature review](10_Safe_hierarchical_hybrid_RL/paper.html#12-literature-review)
- [1.3. Contributions and paper organization](10_Safe_hierarchical_hybrid_RL/paper.html#13-contributions-and-paper-organization)
- [2. Problem formulation of MGF](10_Safe_hierarchical_hybrid_RL/paper.html#2-problem-formulation-of-mgf)
- [2.1. Objective](10_Safe_hierarchical_hybrid_RL/paper.html#21-objective)
- [2.2. Constraints](10_Safe_hierarchical_hybrid_RL/paper.html#22-constraints)
- [3. Safe hierarchical hybrid reinforcement learning](10_Safe_hierarchical_hybrid_RL/paper.html#3-safe-hierarchical-hybrid-reinforcement-learning)
- [3.1. Formulate MGF as a hierarchical partially observable MDP](10_Safe_hierarchical_hybrid_RL/paper.html#31-formulate-mgf-as-a-hierarchical-partially-observable-mdp)
- [3.2. SHHRL-based MGF framework](10_Safe_hierarchical_hybrid_RL/paper.html#32-shhrl-based-mgf-framework)
- [3.3. Neural networks parameters optimization](10_Safe_hierarchical_hybrid_RL/paper.html#33-neural-networks-parameters-optimization)
- [3.4. Safety layer embedded with expert knowledge](10_Safe_hierarchical_hybrid_RL/paper.html#34-safety-layer-embedded-with-expert-knowledge)
- [3.5. MPER for stabilizing learning](10_Safe_hierarchical_hybrid_RL/paper.html#35-mper-for-stabilizing-learning)
- [4. Case studies](10_Safe_hierarchical_hybrid_RL/paper.html#4-case-studies)
- [4.1. Case settings](10_Safe_hierarchical_hybrid_RL/paper.html#41-case-settings)
- [4.2. The modified 123-node system](10_Safe_hierarchical_hybrid_RL/paper.html#42-the-modified-123-node-system)
- [4.3. The modified 356-node regional system in China](10_Safe_hierarchical_hybrid_RL/paper.html#43-the-modified-356-node-regional-system-in-china)
- [5. Conclusion](10_Safe_hierarchical_hybrid_RL/paper.html#5-conclusion)
- [CRediT authorship contribution statement](10_Safe_hierarchical_hybrid_RL/paper.html#credit-authorship-contribution-statement)
- [Declaration of competing interest](10_Safe_hierarchical_hybrid_RL/paper.html#declaration-of-competing-interest)
- [Acknowledgment](10_Safe_hierarchical_hybrid_RL/paper.html#acknowledgment)
- [Data availability](10_Safe_hierarchical_hybrid_RL/paper.html#data-availability)
- [References](10_Safe_hierarchical_hybrid_RL/paper.html#references)
