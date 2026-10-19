# 强化学习与微电网文献筛选

检索日期：2026-10-18，北京时间。

**当前状态：10 篇有效全文已齐全，并已使用 MinerU 4.0.12 本地 Basic 档完成全部 220 页全文转换。`converted/` 提供每篇的 Markdown、纯文本、离线 HTML、逐页截图和原始解析结果。**

全文阅读入口：[`converted/index.html`](converted/index.html)。每篇分别提供 Markdown、纯文本、离线 HTML、原始 MinerU 输出和逐页截图。本文后续重试记录属于历史过程，不代表当前下载状态。

## 筛选口径

- 未发现文章正文，依据本项目 `docs/paper_experiment_plan.md` 的研究主线筛选：FACMAC + EA、连续动作、多智能体微电网经济调度、储能与设备约束、样本效率和稳定性。
- 时间窗口为 2025-10-18 至 2026-10-18。优先依据首次公开/上线时间；刊期、预印本日期和 DOI 登记时间分别记录，不将 DOI 登记时间冒充上线时间。
- “一区 Top”暂按**2025 年中科院升级版大类一区 + Top**理解，不等同于能源与燃料小类一区，也不等同于 JCR Q1。
- 分区来自公开转载记录，未通过付费分区平台的机构登录核验。2026 年“新锐分区”与 2025 年中科院分区分开，不混称为同一版本。投稿或毕业认定仍应核对学校采用的版本。
- “优秀”在这里表示与选题相关、方法和实验设计值得精读的候选，不代表已逐项复现或完成学术质量审计。近年论文不能仅凭引用量排名。
- 十份有效全文中，七份为出版社正式版，两份为已找到相应期刊 DOI 的作者预印本，一份为作者接受稿。预印本及接受稿不是出版社最终版。

## 期刊建议

以下优先级是针对本项目的选刊判断，不是录用概率预测。

| 优先级 | 期刊 | 公开转载的 2025 年分区 | 与本文方向的关系 |
| --- | --- | --- | --- |
| 1 | Applied Energy | 工程技术大类一区，Top；能源与燃料小类二区 | 首选方向。已有多智能体微电网调度、局部观测、市场协同和强化学习研究，可对照本文方法及工程价值。 |
| 2 | Energy Conversion and Management | 工程技术大类一区，Top；能源与燃料、力学、热力学小类一区 | 适合将运行经济性、储能寿命、可再生能源利用和物理可行性作为主线，而不只展示算法回报。 |
| 3 | Energy | 工程技术大类一区，Top；能源与燃料小类二区 | 与微电网调度、运行约束和系统经济性相关；应以实际能源系统收益支撑算法改进。 |
| 4 | Renewable Energy | 工程技术大类一区，Top；能源与燃料小类二区 | 当论文更突出光伏/风电消纳、储能协同和不确定性时，可作为备选。 |

**如果要求严格的“能源与燃料小类一区”，不能把以上四本全部视为符合；本轮公开记录中 ECM 符合该小类口径。**

分区核对入口：

- Applied Energy：https://journalcompass.com/journals/applied-energy-03062619.html
- Applied Energy 另一公开记录：https://hy.iikx.com/sci/technology/10388.html
- ECM：https://www.zazhi.com.cn/sci/213317.html
- ECM 另一公开记录：https://letpub.com.cn/index.php?journalid=2500&page=journalapp&view=detail
- Energy：https://www.ablesci.com/journal/detail?id=DGLvAr
- Renewable Energy：https://www.ablesci.com/journal/detail?id=pnan6r

上述公开网页不是中科院分区表官方授权查询的替代品。最终认定前应在学校入口核实。

## 10 篇论文清单

编号与转换结果文件夹前缀一致。06 至 10 保留用户下载时的源文件名，映射见 `catalog.json`。

| 编号 | 论文 | 期刊及刊期 | DOI | 状态 |
| --- | --- | --- | --- | --- |
| 01 | Explainable deep reinforcement learning for resilient and battery-aware microgrid control | Energy Conversion and Management，2026-04-01 | 10.1016/j.enconman.2026.121215 | 已下载，出版社版，26 页；PDF 载明 2026-02-16 上线 |
| 02 | Mixture-of-experts based multi-critic deep reinforcement learning for sustainable management of data center microgrids | Applied Energy，2026-05-15 | 10.1016/j.apenergy.2026.127561 | 已下载，出版社版，15 页；PDF 载明 2026-02-27 上线 |
| 03 | Hierarchical reinforcement learning for cost-effective railway microgrid management | Applied Energy，2026-12，未来刊期 | 10.1016/j.apenergy.2026.128698 | 已下载，出版社版，18 页；PDF 和高校记录均载明 2026-08-19 已上线 |
| 04 | Multi-agent reinforcement learning-based joint design of low-carbon P2P market and bidding strategy in microgrids | Applied Energy，2026-12，未来刊期 | 10.1016/j.apenergy.2026.128693 | 已下载，作者预印本，10 页；arXiv 首次公开 2026-04-03；期刊 DOI 已核对题名和作者 |
| 05 | MARLEM: A multi-agent reinforcement learning simulation framework for implicit cooperation in decentralized local energy markets | Applied Energy，2026-06-01 | 10.1016/j.apenergy.2026.127546 | 已下载，作者预印本，32 页；arXiv 首次公开 2026-02-17；期刊 DOI 已核对题名和作者 |
| 06 | Lean multi-agent deep reinforcement learning for uncertainty handling in the energy management of networked microgrids | Applied Energy，2026-04-01 | 10.1016/j.apenergy.2026.127354 | 用户补齐，出版社版，31 页 |
| 07 | Deep reinforcement learning-based multi-objective energy management system for microgrids under flexible energy market | Applied Energy，2026-03-01 | 10.1016/j.apenergy.2025.127223 | 用户补齐，出版社版，10 页；DOI 登记不等同上线日期 |
| 08 | Meta-reinforcement learning for multi-energy building microgrids: Curriculum design, rapid adaptation, and scalable deployment | Energy Conversion and Management，2026-09-01 | 10.1016/j.enconman.2026.121652 | 用户补齐，出版社版，22 页 |
| 09 | Deep reinforcement learning strategy integrating environmental perception and semantic masking for real-time operational optimization of building cluster microgrids | Energy，2026-04，349 卷 | 10.1016/j.energy.2026.140676 | 用户补齐，作者接受稿，40 页（含机构封面） |
| 10 | Toward resilient distribution system via microgrid formation: A safe hierarchical hybrid reinforcement learning approach | Applied Energy，2026-08-15 | 10.1016/j.apenergy.2026.128001 | 用户补齐，出版社版，16 页 |

03、04 的未来刊期不代表未来才公开；03 已由全文确认上线日期，04 的本地版本按预印本日期计入窗口。未把仅有未来刊期且无当前公开证据的论文计入已下载数量。

## 阅读重点

### 04：多智能体协同与局部观测

优先阅读。研究对象直接是多微电网，以 DEC-POMDP 描述不确定条件下的局部决策和市场协同。可用于梳理局部观测、跨微电网交互和共享收益之间的关系。其自利竞价和市场设计不同于本项目合作式经济调度，不能直接当成同任务的 FACMAC 基线。

- 期刊页：https://www.sciencedirect.com/science/article/pii/S0306261926013498
- 作者版本：https://arxiv.org/abs/2604.02728
- PDF：https://arxiv.org/pdf/2604.02728

### 05：可复现环境与 CTDE 比较

优先阅读。重点看物理约束、储能建模、隐式协作奖励，以及不同集中/分散训练执行范式如何公平比较。该文侧重本地能源市场框架，与纯调度方法论文的贡献形式不同。不要把同名但不同团队的 GitHub 项目混为该文代码。

- 期刊 DOI：https://doi.org/10.1016/j.apenergy.2026.127546
- 作者版本：https://arxiv.org/abs/2602.16063
- PDF：https://arxiv.org/pdf/2602.16063
- 论文给出的代码仓库：https://github.com/salazarna/marlem

### 01：储能约束、鲁棒性与 MPC 对照

优先阅读。PPO 调度结合电池寿命与韧性目标，并给出不确定性测试和解释性分析。适合参考成本以外的约束诊断、SOC 轨迹、稳健性和在线计算成本的报告方式。其主要目标不是本项目完全相同的经济成本，不能移用收益数字作横向排名。

- 期刊 DOI：https://doi.org/10.1016/j.enconman.2026.121215
- 高校仓储：https://www.open-access.bcu.ac.uk/16898/
- PDF：https://www.open-access.bcu.ac.uk/16898/1/1-s2.0-S0196890426001846-main.pdf

### 02：多目标冲突与消融设计

以多奖励 MDP 和多 critic 学习数据中心微电网的不同目标，适合参考如何解释网络组件及多目标收益。多 critic 不等于多智能体；不能把该文当作 FACMAC 或价值分解的直接替代算法。可借鉴其消融思路，不据此擅自给当前项目新增 MoE 组件。

- 期刊 DOI：https://doi.org/10.1016/j.apenergy.2026.127561
- 高校仓储：https://ira.lib.polyu.edu.hk/handle/10397/120687
- PDF：https://ira.lib.polyu.edu.hk/bitstream/10397/120687/1/1-s2.0-S0306261926002138-main.pdf

### 03：连续调度、约束与分层控制

将电压目标和经济调度分层处理，可参考连续控制的约束表达、强基线和工程案例展示。铁路牵引负荷具有专用动态特性，因此与本项目环境不是同一测试任务。

- 期刊 DOI：https://doi.org/10.1016/j.apenergy.2026.128698
- 高校仓储：https://eprints.whiterose.ac.uk/id/eprint/245329/
- PDF：https://eprints.whiterose.ac.uk/id/eprint/245329/7/1-s2.0-S0306261926013541-main.pdf

### 待下载论文的用途

以下判断仅依据出版社摘要/亮点或高校仓储摘要，**未声称读过全文**。

- 06：多微电网协调、网络约束和不确定性。DQN + KKT/MPEC 混合求解，不是连续动作 FACMAC 的同构基线。https://doi.org/10.1016/j.apenergy.2026.127354
- 07：少量训练数据、在线经济运行和多目标评价，可辅助讨论样本效率及连续多日运行。https://doi.org/10.1016/j.apenergy.2025.127223
- 08：PPO + 元学习 + 课程学习，重点参考未见建筑与不同季节的泛化验证，而非直接将其迁移收益等同于 EA 收益。https://doi.org/10.1016/j.enconman.2026.121652
- 09：语义掩码与储能边界触发，适合对照训练稳定性和约束处理。https://orca.cardiff.ac.uk/id/eprint/185781/
- 10：安全分层混合动作 RL、无效动作屏蔽和连续动作安全投影，适合作为物理可行性相关工作；微电网重构/故障恢复不是日常经济调度。https://doi.org/10.1016/j.apenergy.2026.128001

## 对当前文章的建议

建议先读 04、05、01，再读 02、03。06、07 的主题与本文也很接近，但本轮尚未下载成功。

本轮没有确认到足够贴近“FACMAC + 神经进化 + 连续动作微电网调度”的同构近期论文。因此不要写成“上述论文已经证明 EA 优于 FACMAC”，也不要仅凭加入 EA 就预设创新充分。本文仍需要以 EA/变异/交叉消融、总交互预算、多种子、独立测试和物理可行性结果支撑结论。

## 文件与校验

- `bibliography.bib`：10 篇候选的引用条目，来自 Crossref 元数据。
- `catalog.csv` / `catalog.json`：题名、作者、DOI、下载状态、文件哈希及页数；只有已核验的 5 篇具有 PDF 路径。
- `qa/first_pages.png`：五篇有效全文首页的可视核验图。
- `EXCLUDED_wrong_repository_link_wind_turbine.pdf`：错误开放索引导致下载的风机论文，保留用于追溯，**不是第六篇合格论文**，不应引用为微电网 RL 文献。
- `build_catalog.py`：本次清单与校验信息生成脚本；不修改训练项目。

下载只使用出版社公开入口、作者 arXiv 和高校仓储。没有购买论文、登录机构账号、绕过付费墙或完成人机验证。

## 浏览器直接下载重试

2026-10-18 北京时间，按用户要求，用 Codex 内置浏览器重试 06 至 10：

- 06、07、08、10：逐篇打开对应 ScienceDirect 论文页，均显示 “Are you a robot?”，没有到达论文下载按钮。
- 09：Cardiff 仓储介绍页可正常打开；浏览器下载链接请求超时，随后直接点击 “Download (3MB)” 后进入 “Verification Required” 页面，未获得有效 PDF。
- 本轮没有新增已核验的全文，数量仍为 5。未擅自操作验证码。
- 验证页截图：`qa/sciencedirect_browser_verification.png`。
- 本轮测试的是 Codex 内置浏览器，不代表用户本机 Chrome/Edge 中的登录状态和访问结果。

### 验证恢复后的再次重试

2026-10-18，用户再次要求下载后，重新检查了当前浏览器会话：

- ScienceDirect 的论文介绍页现已可正常访问，不再一概描述为人机验证阻断。
- 06 的出版社 PDF 曾在浏览器阅读器中实际显示，页数为 31；自动下载与阅读器保存操作未返回有效本地文件，仍不计入已下载数量。
- 07、08 的开放论文页和 PDF 入口可访问，但本轮未成功取得并校验本地 PDF。
- 10 的当前页面明确显示 “Access through your organization” 和 “Purchase PDF”，需要合法的机构访问或其他授权全文来源；未购买或绕过访问限制。
- 本轮未新增已核验本地全文，仍为 5 篇。浏览器打开全文不等于已经保存到本地。
