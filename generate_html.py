"""Build research.html and index.html from papers.xlsx.

Each sheet in papers.xlsx becomes a section of the research page, in workbook
order (sheets whose name starts with "_" are skipped). Each row is one paper.
Columns are matched by header name, so their order doesn't matter and any of
them except Title can be left out or blank:

    Title     Paper title (plain text).
    Authors   Coauthors, e.g. "with Jane Doe and John Smith". HTML allowed.
    Note      Status or venue, e.g. "Job Market Paper" or
              "Revise and Resubmit, Journal of Political Economy". HTML allowed.
    Coverage  Press coverage links, e.g. '<a href="...">New York Times</a>, ...'.
    Abstract  Abstract text. Blank lines start new paragraphs.
    Link      URL of the paper. Becomes the "Paper" button and the title link.
    Links     Extra buttons, one per line as "Label | URL" (e.g. "Slides | https://...").
    Latex     1 if the abstract is written in LaTeX (\\%, \\$, etc.) and should be
              converted to plain text; otherwise the abstract is treated as HTML.
    Home      1 to also list the paper under "Recent papers" on the home page
              (in spreadsheet order).

Usage: python generate_html.py
"""

import html
import re
from pathlib import Path

from openpyxl import load_workbook
from pylatexenc.latex2text import LatexNodes2Text

ROOT = Path(__file__).resolve().parent
EXCEL_PATH = ROOT / "papers.xlsx"
TEMPLATES = ROOT / "templates"

latex = LatexNodes2Text()


def slugify(text, max_words=6):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words[:max_words]) or "item"


def clean(value):
    """Return a stripped string, or "" for empty cells."""
    if value is None:
        return ""
    return str(value).strip()


def is_set(value):
    return value in ("1", "1.0", "yes", "Yes", "TRUE", "True", "x", "X")


def read_sheet(ws):
    rows = ws.iter_rows(values_only=True)
    header = [clean(h).lower() for h in next(rows, [])]
    papers = []
    for row in rows:
        paper = {name: clean(value) for name, value in zip(header, row) if name}
        if paper.get("title"):
            papers.append(paper)
    return papers


def render_abstract(paper):
    text = paper.get("abstract", "")
    if not text:
        return ""
    if is_set(paper.get("latex")):
        text = html.escape(latex.latex_to_text(text), quote=False)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    return "\n".join(f"<p>{p}</p>" for p in paragraphs)


def extra_links(paper):
    links = []
    for line in paper.get("links", "").splitlines():
        if "|" in line:
            label, url = (part.strip() for part in line.split("|", 1))
            if label and url:
                links.append((label, url))
    return links


def abstract_block(abstract, abstract_id):
    lines = [f'    <div class="abstract" id="{abstract_id}" hidden>']
    lines.extend(f"        {p}" for p in abstract.splitlines())
    lines.append("    </div>")
    return lines


def render_paper(paper, paper_id, indent="                "):
    """Full entry for the research page."""
    title = html.escape(paper["title"], quote=False)
    link = paper.get("link", "")
    abstract = render_abstract(paper)

    lines = [f'<article class="paper" id="{paper_id}">']
    if link:
        lines.append(f'    <h3 class="paper-title"><a href="{html.escape(link)}">{title}</a></h3>')
    else:
        lines.append(f'    <h3 class="paper-title">{title}</h3>')
    if paper.get("authors"):
        lines.append(f'    <p class="paper-authors">{paper["authors"]}</p>')
    if paper.get("note"):
        lines.append(f'    <span class="paper-note">{paper["note"]}</span>')
    if paper.get("coverage"):
        lines.append(f'    <p class="paper-coverage">Coverage: {paper["coverage"]}</p>')

    buttons = []
    if abstract:
        buttons.append(
            f'<button class="btn abstract-toggle" type="button" aria-expanded="false" '
            f'aria-controls="{paper_id}-abstract">Abstract</button>'
        )
    if link:
        buttons.append(f'<a class="btn" href="{html.escape(link)}">Paper</a>')
    for label, url in extra_links(paper):
        buttons.append(f'<a class="btn" href="{html.escape(url)}">{html.escape(label)}</a>')
    if buttons:
        lines.append('    <div class="paper-actions">')
        lines.extend(f"        {b}" for b in buttons)
        lines.append("    </div>")
    if abstract:
        lines.extend(abstract_block(abstract, f"{paper_id}-abstract"))
    lines.append("</article>")
    return "\n".join(indent + line for line in lines)


def render_recent(paper, paper_id, indent="                    "):
    """Compact entry for the home page's "Recent papers" list."""
    title = html.escape(paper["title"], quote=False)
    link = paper.get("link", "")
    abstract = render_abstract(paper)

    lines = ["<li>"]
    if link:
        lines.append(f'    <a class="recent-title" href="{html.escape(link)}">{title}</a>')
    else:
        lines.append(f'    <span class="recent-title">{title}</span>')
    if paper.get("authors"):
        lines.append(f'    <div class="recent-authors">({paper["authors"]})</div>')
    if paper.get("note"):
        lines.append(f'    <div class="recent-venue">{paper["note"]}</div>')

    actions = []
    if abstract:
        actions.append(
            f'<button class="recent-link abstract-toggle" type="button" aria-expanded="false" '
            f'aria-controls="recent-{paper_id}">Abstract</button>'
        )
    if paper.get("coverage"):
        actions.append(f'<span class="recent-coverage">Coverage: {paper["coverage"]}</span>')
    if actions:
        lines.append('    <div class="recent-actions">')
        lines.extend(f"        {a}" for a in actions)
        lines.append("    </div>")
    if abstract:
        lines.extend(abstract_block(abstract, f"recent-{paper_id}"))
    lines.append("</li>")
    return "\n".join(indent + line for line in lines)


def fill(template_name, **values):
    page = (TEMPLATES / template_name).read_text(encoding="utf-8")
    for key, value in values.items():
        page = page.replace("{{ " + key + " }}", value)
    (ROOT / template_name).write_text(page, encoding="utf-8", newline="\n")


def generate_html():
    workbook = load_workbook(EXCEL_PATH, read_only=True, data_only=True)
    used_ids = set()
    nav, sections, recent = [], [], []

    for ws in workbook.worksheets:
        if ws.title.startswith("_"):
            continue
        papers = read_sheet(ws)
        if not papers:
            continue
        section_id = slugify(ws.title)
        name = html.escape(ws.title, quote=False)
        nav.append(f'                <a class="btn" href="#{section_id}">{name}</a>')

        rendered = []
        for paper in papers:
            paper_id = base = slugify(paper["title"])
            n = 2
            while paper_id in used_ids or paper_id == section_id:
                paper_id, n = f"{base}-{n}", n + 1
            used_ids.add(paper_id)
            rendered.append(render_paper(paper, paper_id))
            if is_set(paper.get("home")):
                recent.append(render_recent(paper, paper_id))

        sections.append(
            f'            <section class="paper-section" id="{section_id}">\n'
            f"                <h2>{name}</h2>\n"
            + "\n".join(rendered)
            + "\n            </section>"
        )

    fill("research.html", SECTION_NAV="\n".join(nav), SECTIONS="\n\n".join(sections))
    fill("index.html", RECENT_PAPERS="\n".join(recent))
    print(f"Wrote research.html ({len(used_ids)} papers) and index.html ({len(recent)} recent papers)")


if __name__ == "__main__":
    generate_html()
