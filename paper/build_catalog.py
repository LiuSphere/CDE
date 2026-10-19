import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pymupdf
from PIL import Image, ImageDraw
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
PAPERS = [
    ("01", "10.1016/j.enconman.2026.121215", "01_ECM_2026_Explainable_RL_battery_aware.pdf", "publisher_version", "2026-02-16"),
    ("02", "10.1016/j.apenergy.2026.127561", "02_AE_2026_Mixture_of_experts_multi_critic.pdf", "publisher_version", "2026-02-27"),
    ("03", "10.1016/j.apenergy.2026.128698", "03_AE_2026_Hierarchical_RL_railway_microgrid.pdf", "publisher_version", "2026-08-19"),
    ("04", "10.1016/j.apenergy.2026.128693", "04_AE_2026_MARL_P2P_market_author_preprint.pdf", "author_preprint", "2026-04-03"),
    ("05", "10.1016/j.apenergy.2026.127546", "05_AE_2026_MARLEM_author_preprint.pdf", "author_preprint", "2026-02-17"),
    ("06", "10.1016/j.apenergy.2026.127354", "1-s2.0-S0306261926000061-main.pdf", "publisher_version", "2026-01-13"),
    ("07", "10.1016/j.apenergy.2025.127223", "1-s2.0-S0306261925019531-main.pdf", "publisher_version", "2025-12-19"),
    ("08", "10.1016/j.enconman.2026.121652", "1-s2.0-S0196890426006217-main.pdf", "publisher_version", "2026-05-22"),
    ("09", "10.1016/j.energy.2026.140676", "Deep Reinforcement Learning Strategy Integrating Environmental Perception and Semantic Masking for Real-Time Operational Optimization of Building Cluster Microgrids.pdf", "accepted_manuscript", ""),
    ("10", "10.1016/j.apenergy.2026.128001", "1-s2.0-S0306261926006537-main.pdf", "publisher_version", "2026-05-09"),
]


def normalized(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def main():
    catalog = []
    bibtex = []
    thumbnails = []
    for number, doi, filename, version, public_date in PAPERS:
        request = Request(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "MicrogridLiteratureCatalog/1.0"},
        )
        with urlopen(request, timeout=40) as response:
            metadata = json.load(response)["message"]
        title = metadata["title"][0]
        authors = [
            f"{author.get('given', '')} {author['family']}".strip()
            for author in metadata["author"]
        ]
        journal = metadata["container-title"][0]
        row = {
            "number": number,
            "title": title,
            "authors": "; ".join(authors),
            "journal": journal,
            "doi": doi,
            "issue_date_parts": metadata.get("published-print", metadata["published"])["date-parts"][0],
            "doi_registered_at": metadata["created"]["date-time"],
            "verified_public_date": public_date,
            "public_date_type": "preprint_first_posted" if version == "author_preprint" else "pdf_available_online" if public_date else "not_verified",
            "version": version,
            "status": "downloaded_verified" if filename else "pending_download",
            "pdf_path": str(ROOT / filename) if filename else "",
            "pages": "",
            "bytes": "",
            "sha256": "",
        }
        if filename:
            path = ROOT / filename
            data = path.read_bytes()
            assert data.startswith(b"%PDF-"), filename
            reader = PdfReader(path)
            first_text = reader.pages[0].extract_text() or ""
            assert normalized(title) in normalized(first_text), f"Title mismatch: {filename}"
            row.update(pages=len(reader.pages), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            document = pymupdf.open(path)
            assert len(document) == len(reader.pages)
            pixmap = document[0].get_pixmap(matrix=pymupdf.Matrix(0.8, 0.8), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            image.thumbnail((490, 680))
            thumbnails.append((number, image))
            document.close()
        catalog.append(row)
        date = row["issue_date_parts"]
        author_field = " and ".join(
            f"{author['family']}, {author.get('given', '')}".rstrip(", ")
            for author in metadata["author"]
        )
        fields = {
            "title": "{" + title + "}",
            "author": author_field,
            "journal": journal,
            "year": str(date[0]),
            "volume": metadata.get("volume", ""),
            "pages": metadata.get("page", metadata.get("article-number", "")),
            "doi": doi,
            "note": f"Local full-text status: {version}",
        }
        bibtex.append(
            f"@article{{microgrid_{number},\n"
            + ",\n".join(f"  {key} = {{{value}}}" for key, value in fields.items() if value)
            + "\n}"
        )
        print(f"{number}: {row['status']}; {title}", flush=True)

    assert sum(row["status"] == "downloaded_verified" for row in catalog) == 10
    (ROOT / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    with (ROOT / "catalog.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(catalog[0]))
        writer.writeheader()
        writer.writerows(catalog)
    (ROOT / "bibliography.bib").write_text("\n\n".join(bibtex) + "\n", encoding="utf-8")
    qa = ROOT / "qa"
    qa.mkdir(exist_ok=True)
    sheet = Image.new("RGB", (1560, 740 * ((len(thumbnails) + 2) // 3)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (number, image) in enumerate(thumbnails):
        x = 15 + (index % 3) * 520
        y = 15 + (index // 3) * 740
        draw.text((x, y), f"Paper {number}", fill="black")
        sheet.paste(image, (x, y + 25))
    sheet.save(qa / "first_pages.png")
    print("Verified: 10 PDFs, 10 records, 0 pending. Excluded PDF not counted.")


if __name__ == "__main__":
    main()
