# CPU 轻量配置第二版

配置名：`facmac_ea_compact_v2`。
保留原来的 `facmac_ea_compact`，新增独立实验，不修改共享默认参数。

## 相对上一版的差异

| 参数 | 上一版 | 第二版 |
| --- | ---: | ---: |
| `buffer_size` | 100000 回合 | 50000 回合 |
| `save_model_interval` | 10000 环境步 | 50000 环境步 |

其余合并后的参数与上一版一致：

- CPU 执行，`batch_size: 128`。
- Actor/Critic 隐藏宽度 48、混合器宽度 32、超网络宽度 64。
- 总预算 `t_max: 3000000`，预热 512 回合。
- 普通更新间隔 2、每次普通更新数 1、EA 学习更新数 3。
- 种群大小 5，每周期演化 1 代。
- 学习率、奖励、环境约束和输入数据不变。
- 验证间隔 10000 步、固定验证日数 8。

算法配置文件不会互相继承，因此第二版显式重复上一版相对
`default.yaml` 的轻量化差异，其他参数继续继承共享默认值。

## 影响

回放容量减半会减少预分配回放张量的内存，不能据此保证每次更新更快。
缓冲区写满后保留的历史经验范围会缩小，长期效果仍需验证。

`save_model_interval` 只控制普通周期检查点；按验证结果保存最佳策略的
`save_best_policy` 仍然开启，频率不受这个参数直接控制。
普通恢复点更稀疏，不能把这项调整视为完整断点续训能力的保证。

本版没有减少总环境步数或更新次数，也没有声称收益与上一版等价。

## 验证

`tests/test_facmac_compact.py` 的 4 项测试通过。两版均跨过 512 回合预热，
各完成 8 次学习器更新和 11 次 EA 演化。配置比较确认除名称、回放容量和
普通保存间隔外，合并后的参数一致。短跑关闭模型保存，因此没有实际跑到
50000 步验证周期检查点写盘。
日志：`results/facmac_compact_v2_tests_20261017_a.log`。

## 运行

```powershell
uv run python src/direct_main.py --config facmac_ea_compact_v2 --env-config microgrid
```

本地短跑，不上传 W&B、不保存模型：

```powershell
uv run python src/direct_main.py --config facmac_ea_compact_v2 --env-config microgrid --override wandb_mode=disabled --override save_model=False --override t_max=12500
```

验证配置差异以及两版跨预热的实际训练：

```powershell
uv run python tests/test_facmac_compact.py
```
