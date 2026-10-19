"""Convert the ten verified papers with MinerU, preserving sources and raw output."""

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import markdown
import pymupdf
from bs4 import BeautifulSoup
from latex2mathml import exceptions as math_errors
from latex2mathml.converter import convert as mathml
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "converted"
NEW_FILES = {
    "06": "1-s2.0-S0306261926000061-main.pdf",
    "07": "1-s2.0-S0306261925019531-main.pdf",
    "08": "1-s2.0-S0196890426006217-main.pdf",
    "09": (
        "Deep Reinforcement Learning Strategy Integrating Environmental Perception and "
        "Semantic Masking for Real-Time Operational Optimization of Building Cluster Microgrids.pdf"
    ),
    "10": "1-s2.0-S0306261926006537-main.pdf",
}
PDF_ONLINE_DATES = {
    "06": "2026-01-13",
    "07": "2025-12-19",
    "08": "2026-05-22",
    "10": "2026-05-09",
}
LABELS = {
    "01": "Explainable_RL_battery_aware",
    "02": "Mixture_of_experts_multi_critic",
    "03": "Hierarchical_RL_railway",
    "04": "MARL_P2P_market",
    "05": "MARLEM",
    "06": "Lean_MARL_networked_microgrids",
    "07": "Multi_objective_EMS",
    "08": "Meta_RL_building_microgrids",
    "09": "Perception_semantic_masking",
    "10": "Safe_hierarchical_hybrid_RL",
}
VERSIONS = {
    "publisher_version": "出版社正式版",
    "author_preprint": "作者预印本",
    "accepted_manuscript": "作者接受稿（含机构封面）",
}
CSS = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;color:#20252b;
background:#f4f6f8;font:16px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}
main{max-width:1080px;margin:0 auto;padding:36px 44px;background:white;min-height:100vh}
h1{font-size:28px;line-height:1.4}h2{font-size:22px;border-bottom:1px solid #ddd;
padding-bottom:8px;margin-top:36px}h3{font-size:18px}a{color:#006b78;overflow-wrap:anywhere}
img{max-width:100%;height:auto;display:block;margin:18px auto}
table{border-collapse:collapse;display:block;overflow:auto;max-width:100%;font-size:14px}
th,td{border:1px solid #d7dde2;padding:8px 10px;text-align:left}
.table-scroll{overflow-x:auto;max-width:100%}
table.index-table{display:table;min-width:940px;width:100%}
.index-table th{white-space:nowrap}
.index-table td:nth-child(2){min-width:380px;max-width:450px}
.index-table td:nth-child(3){min-width:125px}
.index-table td:not(:nth-child(2)):not(:nth-child(3)){white-space:nowrap}
blockquote{margin:20px 0;padding:8px 18px;border-left:4px solid #658a92;background:#f4f8f9}
pre{overflow:auto;background:#f4f6f8;padding:16px}code{font-family:Consolas,monospace}
math{max-width:100%;overflow-x:auto}
math:not([display="block"]){display:inline-block;vertical-align:middle}
math[display="block"]{display:block;margin:20px 0;padding:8px 0}
.math-source{white-space:pre-wrap;border-left:3px solid #ad6720;padding-left:12px}
p,li{overflow-wrap:anywhere}details{border-top:1px solid #ddd;padding:16px 0}
.notice{color:#52616a;font-size:14px}.gallery{display:grid;grid-template-columns:repeat(3,1fr);
gap:16px}.gallery img{border:1px solid #ddd;max-height:330px;object-fit:contain}
@media(max-width:650px){main{padding:20px 16px}h1{font-size:23px}
.gallery{grid-template-columns:repeat(2,1fr)}}
@media print{body{background:white}main{max-width:none;padding:0}.gallery{display:none}}
"""


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def folder_for(row):
    return OUT / f"{row['number']}_{LABELS[row['number']]}"


def read_catalog():
    return json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))


def prepare():
    """Validate titles against actual content, then update only literature metadata."""
    rows = read_catalog()
    OUT.mkdir(exist_ok=True)
    thumbnails = []
    for row in rows:
        number = row["number"]
        path = ROOT / NEW_FILES[number] if number in NEW_FILES else Path(row["pdf_path"])
        data = path.read_bytes()
        assert data.startswith(b"%PDF-"), path
        with pymupdf.open(path) as doc:
            first_text = "".join(page.get_text() for page in list(doc)[:3])
            assert normalize(row["title"]) in normalize(first_text), f"Title mismatch: {path.name}"
            if row["version"] != "author_preprint":
                assert normalize(row["doi"]) in normalize("".join(p.get_text() for p in doc)), path
            row.update(
                pdf_path=str(path),
                status="downloaded_verified",
                version="accepted_manuscript" if number == "09" else (
                    "publisher_version" if number in NEW_FILES else row["version"]
                ),
                pages=len(doc),
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
            if number in PDF_ONLINE_DATES:
                row.update(verified_public_date=PDF_ONLINE_DATES[number],
                           public_date_type="pdf_available_online")
            dest = folder_for(row)
            shots = dest / "screenshots"
            shots.mkdir(parents=True, exist_ok=True)
            for i, page in enumerate(doc):
                target = shots / f"page_{i + 1:03}.png"
                if not target.exists():
                    page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(target)
            front = Image.open(shots / ("page_002.png" if number == "09" else "page_001.png")).convert("RGB")
            front.thumbnail((370, 510))
            thumbnails.append((number, front))
        print(f"Verified {number}: {row['pages']} pages; {path.name}", flush=True)
    assert len(rows) == 10
    (ROOT / "catalog.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (ROOT / "catalog.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sheet = Image.new("RGB", (1560, 1680), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (number, front) in enumerate(thumbnails):
        x, y = 15 + (i % 4) * 390, 10 + (i // 4) * 560
        draw.text((x, y), f"Paper {number}", fill="black")
        sheet.paste(front, (x, y + 25))
    sheet.save(OUT / "all_first_pages.png")
    return rows


def write_html(path, title, body):
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style>"
        f"<main>{body}</main></html>",
        encoding="utf-8",
    )


def render_reading_html(content):
    fragment = markdown.markdown(
        content, extensions=["tables", "fenced_code", "sane_lists", "mdx_math", "toc"],
        extension_configs={"mdx_math": {"enable_dollar_delimiter": True}},
    )
    soup = BeautifulSoup(fragment, "html.parser")
    details = soup.new_tag("details")
    summary = soup.new_tag("summary")
    summary.string = "章节目录"
    details.append(summary)
    listing = soup.new_tag("ul")
    for heading in soup.find_all(["h2", "h3"]):
        if heading.get_text() in {"论文首页", "全文"}:
            continue
        item = soup.new_tag("li")
        link = soup.new_tag("a", href="#" + heading["id"])
        link.string = heading.get_text()
        item.append(link)
        listing.append(item)
    details.append(listing)
    cover_heading = soup.find("h2", string="论文首页")
    if cover_heading is not None:
        cover_heading.insert_before(details)
    # Markdown intentionally skips raw HTML tables, so parse their math-bearing cells separately.
    for cell in soup.find_all(["td", "th"]):
        inner = cell.decode_contents()
        if "$" not in inner and r"\(" not in inner:
            continue
        parsed = BeautifulSoup(
            markdown.markdown(inner, extensions=["mdx_math"],
                              extension_configs={"mdx_math": {"enable_dollar_delimiter": True}}),
            "html.parser",
        )
        for paragraph in parsed.find_all("p"):
            paragraph.unwrap()
        cell.clear()
        cell.extend(list(parsed.contents))
    failures = []
    known_errors = tuple(
        value for name, value in vars(math_errors).items()
        if name.endswith("Error") and isinstance(value, type) and issubclass(value, Exception)
    )
    for equation in soup.find_all("script", type=re.compile(r"^math/tex")):
        source = equation.get_text()
        display = "block" if "display" in equation.get("type", "") else "inline"
        try:
            rendered = mathml(source, display=display)
        except known_errors as error:
            failures.append({"source": source, "error": type(error).__name__})
            replacement = soup.new_tag("code", attrs={"class": "math-source"})
            replacement.string = source
            equation.replace_with(replacement)
        else:
            equation.replace_with(BeautifulSoup(rendered, "html.parser"))
    return str(soup), failures


def clean_formatting(body, title):
    # Preserve formula/code content while removing noisy PDF font-weight inference.
    segments = re.split(r"(```.*?```|\$\$.*?\$\$|\$(?:\\.|[^$])+\$)", body, flags=re.DOTALL)
    body = "".join(part if i % 2 else part.replace("**", "") for i, part in enumerate(segments))
    lines = []
    for line in body.splitlines():
        heading = re.match(r"^#{1,6}\s+(.+)$", line)
        if heading:
            label = heading[1].strip()
            if normalize(label) == normalize(title):
                continue
            if normalize(label) == "abstract":
                label = "Abstract"
            numbering = re.match(r"^(\d+(?:\.\d+)*)(?:\.\s|\s)", label)
            level = 2 + numbering[1].count(".") if numbering else 2
            line = "#" * min(level, 6) + " " + label
        lines.append(line)
    return "\n".join(lines).strip()


def format_paper(row, tier):
    dest = folder_for(row)
    raw_path = dest / "mineru" / "markdown.md"
    raw = raw_path.read_text(encoding="utf-8")
    assert len(raw.strip()) > 3000, f"Suspiciously short extraction: {row['number']}"
    # Work on a separate copy; do not rewrite equations, references, or raw MinerU output.
    body = raw.replace("\r\n", "\n")
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{4,}", "\n\n\n", body).strip()
    body = re.sub(r"!\[([^\]]*)\]\((?!https?://|data:)([^)]+)\)",
                  lambda m: f"![{m[1]}](mineru/{m[2]})", body)
    for original_tag in re.findall(r"<img\b[^>]*>", body, flags=re.IGNORECASE):
        image = BeautifulSoup(original_tag, "html.parser").find("img")
        source = image.get("src", "")
        if source and not source.startswith(("http:", "https:", "data:", "/")):
            image["src"] = "mineru/" + source
            body = body.replace(original_tag, str(image))
    body = clean_formatting(body, row["title"])
    headers = [
        f"# {row['title']}",
        "",
        f"- 期刊：{row['journal']}",
        f"- DOI：{row['doi']}",
        f"- 版本：{VERSIONS[row['version']]}",
        f"- 原文页数：{row['pages']}",
        f"- 文本解析：MinerU 4.0.12 / {tier} / 全文",
        f"- [原始 PDF](../../{Path(row['pdf_path']).name})",
        "- [逐页截图](screenshots.html)",
        "",
        "> 以下为机器解析的原文，未翻译、未改写。公式、表格和双栏阅读顺序可能存在识别误差，请以原始 PDF 和逐页截图为准。",
        "",
        "## 论文首页",
        "",
        f"![论文首页](screenshots/page_{2 if row['number'] == '09' else 1:03}.png)",
        "",
        "## 全文",
        "",
    ]
    content = "\n".join(headers) + body + "\n"
    (dest / "paper.md").write_text(content, encoding="utf-8")
    fragment, failures = render_reading_html(content)
    write_html(dest / "paper.html", row["title"], fragment)
    # Derive plain text through the parser, keeping original LaTeX instead of MathML tokens.
    plain_html = markdown.markdown(
        body, extensions=["tables", "fenced_code", "mdx_math"],
        extension_configs={"mdx_math": {"enable_dollar_delimiter": True}},
    )
    plain = BeautifulSoup(plain_html, "html.parser")
    for equation in plain.find_all("script", type=re.compile(r"^math/tex")):
        equation.replace_with(equation.get_text())
    for table in plain.find_all("table"):
        tabular = "\n".join(
            "\t".join(cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"]))
            for tr in table.find_all("tr")
        )
        table.replace_with("\n" + tabular + "\n")
    text = plain.get_text("\n", strip=True)
    (dest / "paper.txt").write_text(
        row["title"] + f"\nDOI: {row['doi']}\n\n" + text + "\n", encoding="utf-8"
    )
    gallery = [
        f"<h1>{html.escape(row['title'])}</h1>",
        '<p><a href="paper.html">返回全文</a></p>',
        "<p>点击缩略图查看完整分辨率截图。页码按源 PDF 的物理页顺序排列。</p>",
        '<div class="gallery">',
    ]
    for i in range(1, row["pages"] + 1):
        name = f"screenshots/page_{i:03}.png"
        gallery.append(f'<a href="{name}"><img src="{name}" loading="lazy" alt="Page {i}">第 {i} 页</a>')
    gallery.append("</div>")
    write_html(dest / "screenshots.html", row["title"] + " - screenshots", "".join(gallery))
    images = [image["src"] for image in BeautifulSoup(fragment, "html.parser").find_all("img")]
    for image in images:
        if not image.startswith(("https:", "http:", "data:")):
            assert (dest / unquote(image)).is_file(), f"Missing image: {image}"
    raw_json = json.loads((dest / "mineru" / "middle_json.json").read_text(encoding="utf-8"))
    actual_pages = len(raw_json.get("pages", []))
    assert actual_pages == row["pages"], (row["number"], actual_pages, row["pages"])
    return {
        "number": row["number"],
        "title": row["title"],
        "tier": tier,
        "source_sha256": row["sha256"],
        "pages": row["pages"],
        "markdown_characters": len(body),
        "embedded_images": len(images) - 1,
        "mathml_conversion_warnings": failures,
        "screenshots": row["pages"],
        "folder": dest.name,
    }


def make_index(rows, reports):
    lines = [
        "# 强化学习与微电网论文全文集",
        "",
        f"共 {len(reports)} 篇；源 PDF 保留在上级文件夹。",
        "",
        "每篇包含格式化 Markdown、纯文本、离线 HTML 阅读版、逐页截图，以及原始 MinerU 解析结果。",
        "",
        "- MinerU 4.0.12 本地 Basic 档；未上传远程服务。",
        "- 正文不翻译、不改写；统一字重和标题层级，HTML 包含章节目录和离线 MathML 公式。",
        "- 截图按源 PDF 物理页码排列；第 09 篇第 1 页为机构封面，第 2 页为论文首页。",
        "",
        "> 机器解析可能存在公式识别、表格结构或阅读顺序误差；引用时请核对原始 PDF。",
        "",
        "| 编号 | 论文 | 版本 | 页数 | 阅读版 | Markdown | 纯文本 | 截图 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        dest = folder_for(row).name
        if not (OUT / dest / "paper.md").is_file():
            continue
        lines.append(
            f"| {row['number']} | {row['title']} | {VERSIONS[row['version']]} | {row['pages']} | "
            f"[打开]({dest}/paper.html) | [打开]({dest}/paper.md) | [打开]({dest}/paper.txt) | "
            f"[逐页]({dest}/screenshots.html) |"
        )
    lines.extend(["", "## 首页总览", "", "![论文首页总览](all_first_pages.png)", ""])
    text = "\n".join(lines)
    (OUT / "README.md").write_text(text, encoding="utf-8")
    index = BeautifulSoup(markdown.markdown(text, extensions=["tables"]), "html.parser")
    table = index.find("table")
    table["class"] = "index-table"
    wrapper = index.new_tag("div", attrs={"class": "table-scroll"})
    table.wrap(wrapper)
    write_html(OUT / "index.html", "强化学习与微电网论文全文集", str(index))
    (OUT / "conversion_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def convert(rows, tier, selected):
    from mineru import MinerUParser
    from mineru.parser.writer import FileBasedDataWriter

    parser = MinerUParser(tier=tier)
    reports_path = OUT / "conversion_report.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8")) if reports_path.exists() else []
    for row in rows:
        if selected and row["number"] not in selected:
            continue
        started = time.monotonic()
        dest = folder_for(row)
        raw_dir = dest / "mineru"
        if not (raw_dir / "middle_json.json").exists():
            print(f"START {row['number']}: {row['title']}", flush=True)
            result = parser.parse(row["pdf_path"], page_range="all")
            assert len(result.pages) == row["pages"]
            result.save(FileBasedDataWriter(str(raw_dir)))
        report = format_paper(row, tier)
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        reports = [r for r in reports if r["number"] != row["number"]] + [report]
        reports.sort(key=lambda r: r["number"])
        make_index(rows, reports)
        print(f"DONE {row['number']}: {report}", flush=True)
    parser.close()


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--prepare-only", action="store_true")
    arg.add_argument("--format-only", action="store_true")
    arg.add_argument("--tier", default="basic", choices=["flash", "basic", "standard", "advanced"])
    arg.add_argument("--only", nargs="*")
    args = arg.parse_args()
    rows = prepare()
    if args.format_only:
        reports = [format_paper(row, args.tier) for row in rows
                   if (folder_for(row) / "mineru" / "markdown.md").exists()]
        make_index(rows, reports)
        return
    if not args.prepare_only:
        # The installed console shim has a stale interpreter; reuse its packages
        # with the available Python 3.12 without changing the project's environment.
        packages = Path.home() / "scoop/persist/uv/tools/versions/mineru/Lib/site-packages"
        sys.path.insert(0, str(packages))
        convert(rows, args.tier, args.only)


if __name__ == "__main__":
    main()
