# FACMAC + EA Microgrid

三智能体微电网连续动作调度项目。当前主方法为 v03 FACMAC + EA，
环境为 `facmac_dispatch_v3`；Python 3.12，依赖由 uv 管理，训练在 CPU 上执行。

## 安装与运行

在项目根目录执行：

```powershell
uv sync
uv lock --check
uv run python src/direct_main.py
```

默认加载 `facmac_ea_tune_v03_long` 与 `microgrid`，训练约 600 万环境步。
W&B 默认在线，请先配置自己的账户与项目权限；未配置在线记录时不要直接启动正式训练。
命令会启动新训练，不会续训已有 checkpoint。

显式配置、种子及独立短结果路径：

```powershell
uv run python src/direct_main.py --config facmac_ea_tune_v03_long --env-config microgrid --override seed=0 --override unique_token=v03s0b --override local_results_path=results/v03s0b --override env_args.result_dir=results/v03s0b --override wandb_dir=results/v03s0b/wandb
```

使用尚不存在的新 token 和目录，不覆盖旧运行。Windows 下保持路径简短。
正式训练仅上传必要聚合指标，调度图在正常训练结束后生成。
最佳 RL 与 EA Actor 分开保存，不能逐日挑选两者中较优的结果来报告主性能。

## 配置

入口依次合并 `src/config/default.yaml`、环境 YAML、算法 YAML 和命令行覆盖项。

| 配置 | 用途 |
| --- | --- |
| `facmac_ea_tune_v03_long` | 默认主方法：batch128、hidden48、mixer32、buffer50000、600万步 |
| `facmac_ea` | 通用大尺寸 FACMAC + EA，沿用共享默认值，不等同于 v03 |
| `facmac_no_ea` | 通用去 EA 控制组 |
| `facmac_ea_no_mutation` | 通用去变异控制组 |
| `facmac_ea_no_crossover` | 通用去交叉控制组 |
| `facmac_ea_no_variation` | 通用去变异和交叉控制组 |

通用消融目前继承共享默认尺寸；正式论文对比前需建立与 v03 对齐的独立配置，
不能把通用配置与 v03 的差异全部解释为 EA 收益。详见 `docs/paper_experiment_plan.md`。
历史 VMAS、学习率/噪声扫描及 v04-v06 配置已从当前目录移除。

## 目录

| 目录 | 内容 |
| --- | --- |
| `src/` | 环境、采样、回放、共享 Actor、critic、graph mixer、learner 和 EA |
| `src/envs/data/` | 版本化负荷、电价、光伏 CSV |
| `tools/` | MILP、PPO 辅助实验、评估、绘图、性能分析及报告恢复 |
| `tests/` | 物理模型、训练、张量、EA、经济性和最终图回归测试 |
| `docs/` | 当前模型、论文计划及工具说明；`archive/` 为历史资料 |
| `results/` | 本地运行结果，不纳入 Git；保留 v03 完成及中止运行 |

历史建模文档和旧模型 MILP 检查仍可用于解释模型演变，但不是当前正式评估协议。
当前物理模型 MILP 工具为 `tools/run_physical_microgrid_milp.py`，
使用前通过 `--help` 核对选项；完美预测与学习策略的信息及库存边界差异必须披露。
不要依据脚本名称中的 “oracle” 或 “current” 推断其适用于最新模型。

## 验证

```powershell
uv run python -m unittest discover -s tests
```

短程 v03 冒烟测试：

```powershell
uv run python src/direct_main.py --override wandb_mode=disabled --override save_model=False --override showcase_enabled=False --override t_max=12500 --override unique_token=smoke_v03 --override local_results_path=results/smoke_v03 --override env_args.result_dir=results/smoke_v03
```

模型选择只用验证集，独立测试集不得用于调参。SOC 可跨天传递；
每日 SOC 不闭合不等于物理违规，库存差异需独立披露。
现有入口未完整固定全部随机源，不宣称种子标签保证完全复现。

## 当前实验状态

用户已选定 v03，跨版本自动调参及自动化均已停止。
种子0训练达预算但原始报告失败，已显式恢复；种子1正常完成；
种子2由用户中止，不能计作完整第三种子。
具体产物状态见 `results/facmac_tuning/campaign_20261017_a/state.json`。
代码清理不改变已有模型或数据，也不自动提交、推送或启动正式实验。
