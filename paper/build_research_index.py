"""Build a local research index without modifying source papers or conversions."""

import html
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "converted"
NOTES = [
    {
        "number": "01",
        "alias": "可解释与电池寿命 PPO",
        "tags": ["单智能体", "连续动作", "可解释性", "电池寿命", "鲁棒性"],
        "method": "集中式单智能体 PPO；课程学习；SHAP 与 LIME 解释策略。",
        "task": "兼顾供电韧性、负荷优先级和电池循环损伤的微电网控制。",
        "constraints": "环境限制 SOC、功率及充放电互斥；允许按优先级削减负荷。",
        "evaluation": "与规则控制和 MPC 比较，并使用大量扰动场景评估韧性与寿命指标。",
        "use": "参考物理轨迹、韧性与经济性权衡、策略解释的呈现方式。",
        "caution": "不是 MARL；寿命是模型估计而非实测。扰动场景数不等于训练种子数，分阶段奖励也不能直接横向比较。",
        "evidence": ["SHAP", "LIME", "Perlin"],
    },
    {
        "number": "02",
        "alias": "MoE 多评论家 SAC",
        "tags": ["单智能体", "连续动作", "多目标", "MoE", "预测误差"],
        "method": "集中式 SAC；MoE 评论家共享专家，并对不同目标设置门控和输出头。",
        "task": "联合调节数据中心工作负载、制冷与电池，兼顾购电、碳排放和任务积压。",
        "constraints": "设备与任务约束结合环境模型和奖励；基础设置使用未来信息，另测观测噪声。",
        "evaluation": "SAC、DDPG 及 MoE 结构变体；按月选择测试日并测试预测误差。",
        "use": "参考目标分解、结构消融以及预测误差敏感性实验。",
        "caution": "Multi-critic 不等于 multi-agent；不能当作合作 MARL 证据。比较时需统一预测信息、目标权重和检查点选择规则。",
        "evidence": ["perfect", "experts", "Gaussian"],
    },
    {
        "number": "03",
        "alias": "铁路双时间尺度 HRL",
        "tags": ["分层强化学习", "连续动作", "电压约束", "消融", "预测误差"],
        "method": "上层 15 分钟选择参考电压范围，下层 30 秒调度储能；SM-DSAC-T。",
        "task": "弱节点铁路微电网的电压调节与运行成本优化。",
        "constraints": "SOC 与功率边界及电压相关惩罚；实验中平均参考电压违约并非严格为零。",
        "evaluation": "无微电网、固定参考、规则、不同预测误差 MPC，以及网强和储能容量敏感性。",
        "use": "区分设备配置收益、分层框架增益和算法增益；参考适用范围与失效边界分析。",
        "caution": "相对无微电网约节省 34.7%，相对固定参考方案增量约 3.6%；不能把前者全归因于 HRL。",
        "evidence": ["3.6%", "Table 6", "0.18%"],
    },
    {
        "number": "04",
        "alias": "P2P 市场 LSTM-MAPPO",
        "tags": ["多智能体", "CTDE", "P2P市场", "循环网络", "SOC核算"],
        "method": "LSTM-MAPPO；集中训练、分散决策，联合报价、交易量与储能预留。",
        "task": "四个自利微电网参与低碳 P2P 双向拍卖。",
        "constraints": "储能与市场边界；未建模完整 AC 潮流和线路拥塞约束。",
        "evaluation": "分别固定算法比较市场机制、固定市场机制比较学习算法；包含扰动测试。",
        "use": "参考局部历史建模、CTDE 叙述，以及机制与算法的分开对照。",
        "caution": "本地文件为预印本；个体利润目标不同于团队共享奖励。终端储能库存不一致时，经济收益需要补充核算。",
        "evidence": ["LSTM", "MAPPO", "efficien"],
    },
    {
        "number": "05",
        "alias": "MARLEM 市场仿真框架",
        "tags": ["仿真框架", "P2P市场", "非学习演示", "多智能体"],
        "method": "模块化多智能体市场平台，设计支持多种训练范式；本版结果使用随机非学习智能体。",
        "task": "研究本地能源市场规则、资产配置及隐式合作的仿真基础设施。",
        "constraints": "市场和设备边界、图网络传输及简化损耗；不是完整 AC 潮流。",
        "evaluation": "七智能体、无电池与配置电池场景的框架演示；完整 MARL 训练留待后续工作。",
        "use": "用于平台模块、观测与市场指标设计的相关工作。",
        "caution": "本地文件为预印本。不能引用其结果证明训练后的 MARL 合作优势，也不能据此证明 EA + RL 有效。",
        "evidence": ["zero-intelligence agent baseline", "full MARL training", "forthcoming"],
    },
    {
        "number": "06",
        "alias": "离散 DQN 修正预测 + MPEC",
        "tags": ["多智能体", "离散动作", "优化求解器", "网络约束", "不确定性"],
        "method": "多智能体 DQN 学习离散预测误差修正系数，再由双层 MPEC / MILP 求解调度。",
        "task": "联网微电网的不确定性处理、零售商收益与微电网运行优化。",
        "constraints": "优化模型处理设备、终端 SOC 和网络约束；不可行修正受到惩罚。",
        "evaluation": "5 节点与 14 节点系统，与确定性及随机优化对照；测试日覆盖较窄。",
        "use": "参考不确定性处理、模型与学习分工、终端 SOC 约束及网络指标。",
        "caution": "不是连续动作端到端调度，测试时仍需优化求解器；不能直接作为当前 FACMAC 的同任务基线。",
        "evidence": ["MPEC", "July", "terminal"],
    },
    {
        "number": "07",
        "alias": "DDPG 经济状态特征 EMS",
        "tags": ["单智能体", "连续动作", "经济特征", "连续多日", "SOC核算"],
        "method": "集中式 DDPG，加入 EPEM 与储能购电成本状态；配合超级电容快速缓冲。",
        "task": "灵活电价下的微电网多目标运行和能量管理。",
        "constraints": "设备动作边界与底层控制；需要区分储能系统配置和策略本身。",
        "evaluation": "冬夏分别训练，连续十天评估；比较 PPO、PSO、LP-MPC 与无储能方案。",
        "use": "参考连续多日轨迹、日成本表和领域经济状态特征。",
        "caution": "10.2% 与 33% 节省相对无储能，并非相对 MPC。冬夏分别训练不等于单策略跨季节泛化。",
        "evidence": ["Table 4", "Non-ES", "10.2"],
    },
    {
        "number": "08",
        "alias": "Reptile 元学习 + PPO",
        "tags": ["单智能体", "连续动作", "元强化学习", "课程学习", "泛化"],
        "method": "增强一阶 Reptile 外循环，PPO 内循环；课程逐步增加建筑与季节多样性。",
        "task": "多能源建筑微电网的快速适应与跨建筑部署。",
        "constraints": "设备硬边界与能量平衡软惩罚；偏好 SOC 区间不同于硬边界。",
        "evaluation": "五栋未见建筑、四季以及少量适应回合；比较适应前后和不同课程。",
        "use": "参考未见场景划分、适应预算、性能与约束/购电依赖之间的权衡。",
        "caution": "不是社区 MARL；12.2% 成本下降是适应后相对适应前，非相对 MPC。多个模型配置不等于独立训练种子。",
        "evidence": ["12.2%", "five unseen", "Wilcoxon"],
    },
    {
        "number": "09",
        "alias": "规则语义掩码 DQN",
        "tags": ["单智能体", "离散动作", "动作掩码", "规则知识", "SOC核算"],
        "method": "集中式离散 DQN；物理可行性筛选叠加电价、SOC 和削峰规则形成候选动作集。",
        "task": "虚拟建筑群、光伏、电池与 EV 充电负荷的三十天运行优化。",
        "constraints": "动作掩码与执行前修正；通过奖励约束日终 SOC，兼顾应急储备。",
        "evaluation": "不同动作步长下与原始 DQN、PSO、WOA、DE 比较；观测消融和边界触发统计。",
        "use": "参考原始/实际执行动作差异、边界触发、日终库存和经济启发式消融。",
        "caution": "本地文件为接收稿，第 1 页是机构封面。semantic 指人工规则而非大语言模型；收益混合了物理筛选与经济规则贡献。",
        "evidence": ["Table 6", "Table 7", "semantic rules"],
    },
    {
        "number": "10",
        "alias": "安全分层 DDQN + TD3",
        "tags": ["分层强化学习", "混合动作", "动作掩码", "安全投影", "消融"],
        "method": "DDQN 选择开关拓扑，TD3 调节 DER；离散掩码、QP 安全投影和多优先经验回放。",
        "task": "停电后形成孤岛微电网、恢复配电系统负荷。",
        "constraints": "规则过滤拓扑动作，模型驱动 QP 修正连续动作；保证范围取决于模型及可行性假设。",
        "evaluation": "123/356 节点；混合动作基线、MILP、MPER 组件消融、安全层触发与修正幅度。",
        "use": "参考安全层介入频率、修正范数、模块消融和包含求解器的端到端时延。",
        "caution": "不是纯连续动作合作调度；测试投影触发约 11.14%，不代表裸策略始终安全。正文与表格部分时延数字不一致，引用需核对。",
        "evidence": ["11.14%", "Table 1", "Table 3"],
    },
]

TOPICS = {
    "多智能体与协作": ["04", "05", "06"],
    "连续控制与经济调度": ["01", "02", "03", "07", "08"],
    "安全、可行性与动作修正": ["01", "03", "06", "09", "10"],
    "泛化、扰动与快速适应": ["01", "02", "03", "08"],
    "消融与收益归因": ["02", "03", "04", "09", "10"],
    "SOC 与长期库存公平性": ["04", "06", "07", "09"],
    "可解释性与物理轨迹": ["01", "07", "09", "10"],
}


class HeadingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.headings = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag in {"h1", "h2", "h3", "h4"}:
            self.current = {"id": dict(attrs).get("id", ""), "text": "", "tag": tag}

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"] += data

    def handle_endtag(self, tag):
        if self.current is not None and tag == self.current["tag"]:
            if self.current["id"]:
                self.headings.append(self.current)
            self.current = None


def link(path):
    return quote(path, safe="/#")


def build():
    catalog = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))
    assert len(catalog) == len(NOTES) == 10
    records = []
    for meta, note in zip(catalog, NOTES, strict=True):
        assert meta["number"] == note["number"]
        folders = list(OUT.glob(f"{note['number']}_*"))
        assert len(folders) == 1, folders
        folder = folders[0]
        parser = HeadingParser()
        parser.feed((folder / "paper.html").read_text(encoding="utf-8"))
        text = (folder / "paper.txt").read_text(encoding="utf-8")
        evidence = []
        for term in note["evidence"]:
            pos = text.casefold().find(term.casefold())
            if pos >= 0:
                line = text[:pos].count("\n") + 1
                evidence.append({"term": term, "line": line})
        pdf = Path(meta["pdf_path"])
        assert pdf.exists()
        record = {
            **meta, **note, "folder": folder.name,
            "links": {
                "全文": f"{folder.name}/paper.html",
                "原始PDF": f"../{pdf.name}",
                "Markdown": f"{folder.name}/paper.md",
                "纯文本": f"{folder.name}/paper.txt",
                "逐页截图": f"{folder.name}/screenshots.html",
            },
            "headings": parser.headings,
            "evidence_locations": evidence,
            "fulltext": text,
        }
        for target in record["links"].values():
            assert (OUT / target).exists(), target
        records.append(record)

    data = {"scope": "10篇本地论文；中文研读注释，不是作者原文或复现结论",
            "topics": TOPICS, "papers": records}
    (OUT / "research_index.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# 强化学习与微电网研究索引", "",
        "范围：本地 10 篇论文。中文内容为研读归纳，不是原文翻译或复现结论。"
        "不重新认定期刊分区、Top 标记或发表日期；预印本/接收稿结论以本地版本为限。",
        "", "[可搜索网页](research_index.html) | [原全文目录](index.html)", "",
        "## 主题导航", "",
    ]
    for topic, numbers in TOPICS.items():
        md.append(f"- {topic}：" + "、".join(f"[{n}](#paper-{n})" for n in numbers))
    md += ["", "## FACMAC + EA 写作核对", "",
           "- 本集合不能单独证明 FACMAC + EA 的首创性，尚需更广泛检索。",
           "- 同任务对比需统一观测信息、奖励、设备约束、数据划分与终端库存。",
           "- EA 的种群评估交互步数和墙钟时间必须计入训练预算。",
           "- 分开报告系统配置、市场机制、算法及安全层的贡献。",
           "- 独立训练种子、测试场景和超参数配置数量分别统计。",
           "- 优先组织证据链，不据此直接更改现有算法或实验配置。", ""]
    articles = []
    e = html.escape
    for p in records:
        n = p["number"]
        md += [f'<a id="paper-{n}"></a>', f"## {n} {p['alias']}", "",
               p["title"], "", f"{p['journal']} | DOI: {p['doi']} | {p['version']} | {p['pages']} 页", "",
               " | ".join(f"[{label}]({link(target)})" for label, target in p["links"].items()),
               "", "标签：" + " / ".join(p["tags"]), ""]
        fields = [("method", "算法"), ("task", "任务"), ("constraints", "约束处理"),
                  ("evaluation", "实验"), ("use", "可借鉴点"), ("caution", "引用边界")]
        for key, label in fields:
            md.append(f"- **{label}**：{p[key]}")
        if p["evidence_locations"]:
            md.append("- 原文定位（paper.txt 行号，非 PDF 页码）：" + "；".join(
                f"`{x['term']}` 第 {x['line']} 行" for x in p["evidence_locations"]))
        md += ["", "### 章节入口", ""]
        heading_links = []
        for h in p["headings"]:
            if h["tag"] == "h1":
                continue
            target = link(f"{p['folder']}/paper.html#{h['id']}")
            md.append(f"- [{h['text']}]({target})")
            heading_links.append(f'<li><a href="{target}">{e(h["text"])}</a></li>')
        md.append("")
        resources = " ".join(f'<a href="{link(target)}">{e(label)}</a>'
                             for label, target in p["links"].items())
        articles.append(
            f'<article id="paper-{n}" data-number="{n}"><h2>{n} {e(p["alias"])}</h2>'
            f'<p class="title">{e(p["title"])}</p>'
            f'<p class="meta">{e(p["journal"])} · {e(p["version"])} · {p["pages"]} 页'
            f'<br>DOI: {e(p["doi"])}</p><nav class="resources">{resources}</nav>'
            f'<p class="tags">{e(" / ".join(p["tags"]))}</p><dl>'
            + "".join(f'<dt>{label}</dt><dd>{e(p[key])}</dd>' for key, label in fields)
            + '</dl><p class="match" hidden></p>'
            + '<details><summary>章节入口</summary><ul>'
            + "".join(heading_links) + '</ul></details></article>'
        )
    (OUT / "research_index.md").write_text("\n".join(md), encoding="utf-8")
    topics_html = "".join(
        f'<option value="{e(t)}">{e(t)}</option>' for t in TOPICS)
    # Inline data keeps search functional under file:// without fetch or a server.
    inline_data = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>强化学习与微电网研究索引</title><style>
*{box-sizing:border-box;letter-spacing:0}body{margin:0;background:#fff;color:#222;
font:15px/1.75 "Segoe UI","Microsoft YaHei",sans-serif}
main{max-width:1100px;margin:auto;padding:28px 32px 70px}
h1{font-size:26px;line-height:1.4}h2{font-size:20px;line-height:1.5;margin-bottom:6px}
a{color:#006b78;overflow-wrap:anywhere}p,dd,li{overflow-wrap:anywhere}
.meta,.tags{font-size:13px;color:#53616a}.scope{border-left:3px solid #b38626;padding-left:14px}
.toolbar{display:flex;gap:14px;align-items:end;flex-wrap:wrap;background:#f3f6f7;
padding:16px;margin:24px 0 10px}
label{display:grid;gap:4px}.search{flex:1;min-width:180px}
input,select{font:inherit;max-width:100%;min-height:42px;border:1px solid #a7b3b9;
border-radius:4px;background:white;padding:7px 10px}
:focus-visible{outline:2px solid #006b78;outline-offset:3px}
article{border-top:1px solid #ccd6da;padding:20px 0;scroll-margin-top:18px}
.title{margin:0 0 8px;font-weight:600}.resources{display:flex;flex-wrap:wrap;gap:8px 20px}
dl{display:grid;grid-template-columns:88px minmax(0,1fr);gap:8px 12px}
dt{font-weight:600}dd{margin:0}details{margin-top:16px}summary{cursor:pointer;color:#006b78}
.match{border-left:3px solid #6d8f51;background:#f5f8f1;padding:10px 14px;white-space:pre-wrap}
[hidden]{display:none!important}.protocol{border-top:2px solid #53616a;margin-top:32px}
@media(max-width:600px){main{padding:18px 16px 50px}h1{font-size:23px}
dl{grid-template-columns:1fr;gap:3px}dd{margin-bottom:10px}.toolbar{gap:12px}
label{width:100%}.toolbar input,.toolbar select{width:100%}}
</style></head><body><main>
<header><h1>强化学习与微电网研究索引</h1>
<p>10 篇论文 · 方法、实验与引用边界</p>
<nav class="resources"><a href="index.html">原全文目录</a>
<a href="research_index.md">Markdown 索引</a><a href="research_index.json">结构化索引</a></nav>
<p class="scope">中文注释为研读归纳，不是作者原文或复现结论。预印本与接收稿以本地版本为限；
不重新认定期刊分区、Top 标记或发表日期。原文定位与截图均链接到已有本地文件。</p></header>
<form class="toolbar" role="search" onsubmit="return false">
<label class="search">检索<input id="query" type="search" placeholder="关键词、算法、DOI 或正文短语"></label>
<label>主题<select id="topic"><option value="">全部主题</option>__TOPICS__</select></label>
</form><p id="count" role="status" aria-live="polite">10 / 10 篇</p>
<div id="papers">__ARTICLES__</div>
<p id="empty" hidden>没有匹配的论文。</p>
<section class="protocol"><h2>FACMAC + EA 写作核对</h2><ul>
<li>本集合不能单独证明创新性或首创性，尚需更广泛检索。</li>
<li>同任务对比统一观测、奖励、约束、数据划分和终端 SOC 库存。</li>
<li>EA 种群评估步数与墙钟时间计入总预算。</li>
<li>分别报告系统配置、市场机制、算法与安全层贡献。</li>
<li>区分独立训练种子、测试场景与超参数配置。</li></ul></section>
</main><script id="index-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('index-data').textContent);
const query=document.getElementById('query'),topic=document.getElementById('topic');
const searchable=data.papers.map(p=>({...p,haystack:JSON.stringify(p).toLocaleLowerCase()}));
function filter(){
 const terms=query.value.trim().toLocaleLowerCase().split(/\\s+/).filter(Boolean);
 const selected=topic.value?new Set(data.topics[topic.value]):null;
 let count=0;
 for(const p of searchable){
  const article=document.getElementById('paper-'+p.number);
  const visible=(!selected||selected.has(p.number))&&terms.every(t=>p.haystack.includes(t));
  article.hidden=!visible;if(visible)count++;
  const match=article.querySelector('.match');match.hidden=true;match.textContent='';
  if(visible&&terms.length){
   const text=p.fulltext,lower=text.toLocaleLowerCase();
   const term=terms.find(t=>lower.includes(t));
   if(term){const pos=lower.indexOf(term),start=Math.max(0,pos-90),end=Math.min(text.length,pos+term.length+180);
    const line=text.slice(0,pos).split('\\n').length;
    match.textContent='正文命中 · 纯文本第 '+line+' 行\\n'+(start?'…':'')+text.slice(start,end)+(end<text.length?'…':'');
    match.hidden=false;}
  }
 }
 document.getElementById('count').textContent=count+' / '+data.papers.length+' 篇';
 document.getElementById('empty').hidden=count!==0;
}
query.addEventListener('input',filter);topic.addEventListener('change',filter);
</script></body></html>"""
    page = page.replace("__TOPICS__", topics_html).replace(
        "__ARTICLES__", "\n".join(articles)).replace("__DATA__", inline_data)
    (OUT / "research_index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"papers": len(records),
                      "pages": sum(p["pages"] for p in records),
                      "headings": sum(len(p["headings"]) for p in records),
                      "output": str(OUT / "research_index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    build()
