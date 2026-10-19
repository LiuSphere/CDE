# GAVDAC Microgrid

GAVDAC: Graph-Attentive Value Decomposition Actor-Critic.

三智能体微电网连续动作调度项目。当前主方法为无交叉 GAVDAC + EA，
环境为 `gavdac_dispatch_v3`；Python 3.12，依赖由 uv 管理，训练在 CPU 上执行。

## 安装与运行

在项目根目录执行：

```powershell
uv sync
uv lock --check
uv run python src/direct_main.py
```

默认加载 `gavdac_ea_main` 与 `microgrid`，训练约 600 万环境步。
主方法沿用 v03 的网络和训练参数，但移除交叉，保留种群选择、精英继承、
变异、种群经验回放和 RL 策略注入。
W&B 默认在线，请先配置自己的账户与项目权限；未配置在线记录时不要直接启动正式训练。
命令会启动新训练，不会续训已有 checkpoint。

显式配置、种子及独立短结果路径：

```powershell
uv run python src/direct_main.py --config gavdac_ea_main --env-config microgrid --override seed=0 --override unique_token=mains0a --override local_results_path=results/mains0a --override env_args.result_dir=results/mains0a --override wandb_dir=results/mains0a/wandb
```

使用尚不存在的新 token 和目录，不覆盖旧运行。Windows 下保持路径简短。
正式训练仅上传必要聚合指标，调度图在正常训练结束后生成。
最佳 RL 与 EA Actor 分开保存，不能逐日挑选两者中较优的结果来报告主性能。

## 配置

入口依次合并 `src/config/default.yaml`、环境 YAML、算法 YAML 和命令行覆盖项。

| 配置 | 用途 |
| --- | --- |
| `gavdac_ea_main` | 默认主方法：无交叉，batch128、hidden48、mixer32、buffer50000、600万步 |
| `gavdac_ea_main_no_ea` | 本轮匹配 C1：主方法同尺寸、同预算，仅关闭 EA |
| `gavdac_ea_tune_v03_long` | 保留的历史 v03，启用交叉，不再是默认主方法 |
| `gavdac_ea` | 历史通用大尺寸 GAVDAC + EA，沿用含交叉的共享默认值，不等同于主方法 |
| `gavdac_no_ea` | 通用去 EA 控制组 |
| `gavdac_ea_no_mutation` | 通用去变异控制组 |
| `gavdac_ea_no_crossover` | 通用去交叉控制组 |
| `gavdac_ea_no_variation` | 通用去变异和交叉控制组 |

通用消融目前继承共享默认尺寸；正式论文对比前需建立与 v03 对齐的独立配置，
不能把通用配置与 v03 的差异全部解释为 EA 收益。详见 `docs/paper_experiment_plan.md`。
现存 `gavdac_v03_no_*` 配置属于旧 v03 实验矩阵，不能直接作为新主方法的匹配消融；
其中 `gavdac_v03_no_crossover` 与新主方法的算法参数相同。
旧 v03 和0.25交叉配置及结果保持不变，不把历史结果重新标记成新主方法验证。
历史 VMAS、学习率/噪声扫描及 v04-v06 配置已从当前目录移除。

## 目录

| 目录 | 内容 |
| --- | --- |
| `src/` | 环境、采样、回放、共享 Actor、critic、graph mixer、learner 和 EA |
| `src/envs/data/` | 版本化负荷、电价、光伏 CSV |
| `tools/` | MILP、PPO 辅助实验、评估、绘图、性能分析及报告恢复 |
| `tests/` | 物理模型、训练、张量、EA、经济性和最终图回归测试 |
| `docs/` | [文档索引](docs/README.md)：当前说明；`archive/` 为必要历史审计 |
| `results/` | 本地运行结果，不纳入 Git；保留 v03 完成及中止运行 |

历史建模文档和旧模型 MILP 检查仍可用于解释模型演变，但不是当前正式评估协议。
当前物理模型 MILP 工具为 `tools/run_physical_microgrid_milp.py`，
使用前通过 `--help` 核对选项；完美预测与学习策略的信息及库存边界差异必须披露。
不要依据脚本名称中的 “oracle” 或 “current” 推断其适用于最新模型。

## 验证

```powershell
uv run python -m unittest discover -s tests
```

短程主方法冒烟测试：

```powershell
uv run python src/direct_main.py --override wandb_mode=disabled --override save_model=False --override showcase_enabled=False --override t_max=12500 --override unique_token=smoke_main --override local_results_path=results/smoke_main --override env_args.result_dir=results/smoke_main
```

模型选择只用验证集，独立测试集不得用于调参。SOC 可跨天传递；
每日 SOC 不闭合不等于物理违规，库存差异需独立披露。
现有入口未完整固定全部随机源，不宣称种子标签保证完全复现。

## 实验与历史版本

本轮按 [正式实验计划](docs/paper_experiment_plan.md) 重新训练 C1、C2、MAPPO，
各3个种子，不复用历史结果。连续 SOC 为主评价，每日重置版另列。
规则基线只用当前观测和训练数据派生的固定阈值。

验证时段的评价底座检查（输出目录必须尚不存在）：

```powershell
uv run python tools/evaluate_microgrid_comparison.py --rule --split validation --rows 731 732 733 734 735 736 737 --mode continuous --output results/rulecheck01
uv run python tools/freeze_paper_protocol.py --output results/protocol01/frozen.json
```

评价保存轨迹、独立成本汇总与物理检查。已有 Actor 用 `--checkpoint` 替代 `--rule`，
自动从其保存源码生成独立运行目录，并保留源码、输入与权重哈希校验。
仅在最终评价阶段使用 `--split test --all-days --retrospective-holdout`；
工具默认不允许无明确标记的保留评价，不在开发时查看保留结果。

本轮调度使用本机1槽、`ssh 707` 的 `/home/arch707/data/CDE` 2槽。
707先按相同提交及锁文件准备，再通过SSH标准输入同步本机W&B密钥：

```powershell
uv run python tools/sync_remote_wandb.py --host 707 --project /home/arch707/data/CDE
uv run python tools/experiment_slot.py --slots 1 --status
```

`tools/experiment_slot.py --slots 1 --slot 0 --config gavdac_ea_main_no_ea --seed 2 --token <new-token> --detach`
会启动本机正式训练；仅在计划前置验收后执行。
707对应 `--slots 2 --slot 0` 或 `--slot 1`，并加
`--credential-file /home/arch707/.config/gavdac/wandb.json`，不在命令里填写密钥。
监督器保存PID创建标识、日志、源码哈希及退出码；`exit.json`仍需验收，不能直接当成成功。
远端冷克隆没有旧 SB3 桌面外部夹具时，使用
`uv run python tools/validate_experiment_host.py --without-desktop-sb3-reference --output results/<new-dir>/qualification.json`。
它逐项列出未执行的旧参考测试，不把排除项算作通过；所有其余测试仍会执行并暴露错误。

主方法移除交叉是配置选择，不是一次新的完整多种子验证。
历史 v03、消融和0.25交叉运行保留原配置、源码快照和指标；中止运行不能计作完成。
实际记录见 `results/facmac_tuning/`、`results/paper20261018/` 和 `results/cx25check/`。
旧交叉探索不自动续跑；正式训练需单独授权，不因切换默认配置而启动。

当前代码统一使用 GAVDAC 名称，配置前缀为 `gavdac_`，环境标识为
`gavdac_dispatch_v3`。旧名称不提供兼容别名；历史结果、源码快照和检查点
配置保留原始标识，应配合其对应源码使用，不代表可直接续训。
