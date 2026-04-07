#!/usr/bin/env python3
"""
Auto-Citetion: find and evaluate papers for your research.

Usage:
  python auto_citetion.py config.json
  python auto_citetion.py config.json -o results/
  python auto_citetion.py config.json --skip-llm
  python auto_citetion.py config.json --skip-search --skip-llm
  python auto_citetion.py config.json --fast
"""

import argparse
import json
import re
import sys
from pathlib import Path

from search import (
    Paper, PaperPool, score_and_categorize,
    ss_keyword, ss_citations, ss_authors,
    si_semantic, si_similar, si_detail, si_collect_ids,
    arxiv_search,
)
from evaluate import LLMEvaluator

# ── Category labels for the report ───────────────────────────────────────

CAT = {
    "similar_method":    ("MOST SIMILAR (must differentiate)", "Related Work + Introduction"),
    "counterfactual_xai":("COUNTERFACTUAL EXPLANATIONS",       "Counterfactual section + Method"),
    "shortcut_spurious": ("SHORTCUT LEARNING & SPURIOUS",      "Shortcut Learning section + Intro"),
    "vlm_multimodal":    ("VLM / MULTIMODAL",                  "Explainability section + Method"),
    "diffusion_editing": ("DIFFUSION-BASED EDITING",           "Counterfactual section + Method"),
    "explainability":    ("EXPLAINABILITY & CONCEPTS",          "Explainability section"),
    "augmentation":      ("GENERATIVE AUGMENTATION",            "Augmentation section + Experiments"),
    "other":             ("OTHER RELEVANT",                     "Various"),
}


def load_known(path: Path) -> set[str]:
    titles = set()
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split("|")
            if len(parts) >= 4:
                t = parts[2].strip().lower()
                if t and t != "title":
                    titles.add(t)
    return titles


def is_known(title: str, known: set[str]) -> bool:
    tl = title.lower().strip()
    for k in known:
        if tl in k or k in tl:
            return True
        w1, w2 = set(tl.split()), set(k.split())
        if len(w1 & w2) / max(len(w1 | w2), 1) > 0.5:
            return True
    return False


# ── Search stage ──────────────────────────────────────────────────────────

def run_search(cfg: dict, out: Path) -> list[Paper]:
    pool = PaperPool()
    cookie_path = out / ".scholar_inbox_cookie"
    cookie = cookie_path.read_text().strip() if cookie_path.exists() else ""

    if cookie:
        si_semantic(pool, cfg.get("scholar_inbox_queries", []), cookie)
        abstract = cfg.get("paper_abstract", "")
        if abstract:
            ids = si_collect_ids(abstract, cookie)
            if ids:
                si_similar(pool, ids[:25], cookie)
                si_detail(pool, ids[:15], cookie)
    else:
        print("No .scholar_inbox_cookie found, skipping Scholar Inbox.", file=sys.stderr)

    ss_keyword(pool, cfg.get("semantic_scholar_queries", []))
    ss_citations(pool, cfg.get("seed_arxiv_ids", []))
    ss_authors(pool, cfg.get("key_authors", []))
    arxiv_search(pool, cfg.get("arxiv_queries", []))

    print(f"\nTotal pool: {pool.size}", file=sys.stderr)
    return pool.all()


# ── Report generation ─────────────────────────────────────────────────────

def generate_report(papers: list[Paper]) -> str:
    has_llm = any(p.llm_verdict and p.llm_verdict != "error" for p in papers)

    lines = ["# Final Reading List", ""]

    if has_llm:
        must = [p for p in papers if p.llm_verdict == "must_cite"]
        should = [p for p in papers if p.llm_verdict == "should_cite"]
        maybe = [p for p in papers if p.llm_verdict == "maybe_cite"]
        skip = [p for p in papers if p.llm_verdict in ("skip", "error", "")]

        lines += [
            "## Summary", "",
            "| Verdict | Count |", "|---------|-------|",
            f"| **Must Cite** | {len(must)} |",
            f"| **Should Cite** | {len(should)} |",
            f"| **Maybe Cite** | {len(maybe)} |",
            f"| Skip | {len(skip)} |", "", "---", "",
        ]

        for label, group in [("MUST CITE", must), ("SHOULD CITE", should), ("MAYBE CITE", maybe)]:
            if not group:
                continue
            lines += [f"## {label}", ""]
            for i, p in enumerate(group, 1):
                lines += _fmt(i, p)
            lines += ["---", ""]

        # By section
        sec_map: dict[str, list[Paper]] = {}
        for p in must + should:
            for s in p.llm_sections:
                sec_map.setdefault(s, []).append(p)
        if sec_map:
            lines += ["## Papers by Section", ""]
            for s in sorted(sec_map):
                lines.append(f"### {s}")
                for p in sec_map[s]:
                    tag = "**MUST**" if p.llm_verdict == "must_cite" else "SHOULD"
                    lines.append(f"- [{tag}] {p.title} ({p.year})")
                lines.append("")
    else:
        lines.append(f"**{len(papers)} papers found** (no LLM evaluation)\n")

    # By topic
    grouped: dict[str, list[Paper]] = {}
    for p in papers:
        grouped.setdefault(p.category or "other", []).append(p)

    lines += ["---", "", "## Papers by Topic", ""]
    for cat in list(CAT) + ["other"]:
        if cat not in grouped:
            continue
        label, where = CAT.get(cat, (cat, ""))
        lines += [f"### {label}", f"*Cite in: {where}*", ""]
        for i, p in enumerate(grouped[cat][:15], 1):
            lines += _fmt(i, p, brief=True)
    return "\n".join(lines)


def _fmt(i: int, p: Paper, brief: bool = False) -> list[str]:
    lines = [f"**{i}. {p.title}**", ""]
    lines.append(f"- Authors: {p.authors} | Year: {p.year} | Venue: {p.venue or 'preprint'}")
    if p.arxiv_id:
        lines.append(f"- arXiv: [{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id})")
    if p.llm_verdict and p.llm_verdict != "error":
        lines.append(f"- **LLM: {p.llm_verdict}** ({p.llm_score}/10) — {p.llm_relationship}")
        lines.append(f"- Cite in: {', '.join(p.llm_sections)}")
        lines.append(f"- Why: {p.llm_reasoning}")
        if p.llm_differentiation:
            lines.append(f"- Differs: {p.llm_differentiation}")
    else:
        lines.append(f"- Score: {p.score} | Found by: {p.source_count} sources")
    if not brief and p.abstract:
        lines.append(f"- Abstract: {p.abstract[:300]}…")
    lines.append("")
    return lines


def write_paper_files(papers: list[Paper], d: Path) -> int:
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in papers:
        tag = (p.llm_verdict or p.category or "other").upper().replace("_", "-")
        slug = re.sub(r"[^a-z0-9\s]", "", p.title.lower())[:60].replace(" ", "_")
        path = d / f"{tag}_{slug}.md"
        if path.exists():
            continue
        arxiv = f"**arXiv:** [{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id})\n" if p.arxiv_id else ""
        llm = ""
        if p.llm_verdict:
            llm = f"\n## LLM Evaluation\n\n- Verdict: {p.llm_verdict} ({p.llm_score}/10)\n- {p.llm_relationship}\n- Cite in: {', '.join(p.llm_sections)}\n- {p.llm_reasoning}\n- {p.llm_differentiation}\n"
        path.write_text(f"# {p.title}\n\n**Authors:** {p.authors}\n**Year:** {p.year}\n**Venue:** {p.venue}\n{arxiv}{llm}\n## Abstract\n\n{p.abstract or 'N/A'}\n")
        n += 1
    return n


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(prog="auto-citetion",
        description="Find and evaluate papers for your research.")
    ap.add_argument("config", help="Search config JSON file")
    ap.add_argument("-o", "--output", default=".", help="Output directory")
    ap.add_argument("--refs", help="references.md path for dedup")
    ap.add_argument("--skip-search", action="store_true")
    ap.add_argument("--skip-si", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--top", type=int, default=80, help="Papers to evaluate with LLM")
    ap.add_argument("--min-score", type=float, default=4.0)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())
    known = load_known(Path(args.refs) if args.refs else out / "references.md")
    raw_path = out / "all_candidates.json"

    # Stage 1: Search
    if not args.skip_search:
        print("=" * 50, "\nSTAGE 1: Search\n" + "=" * 50, file=sys.stderr)
        papers = run_search(cfg, out)
        score_and_categorize(papers)
        papers = [p for p in papers if not is_known(p.title, known) and p.score >= args.min_score]
        papers.sort(key=lambda p: p.score, reverse=True)
        raw_path.write_text(json.dumps([p.to_dict() for p in papers], indent=2))
        print(f"Saved {len(papers)} papers", file=sys.stderr)
    else:
        papers = [Paper.from_dict(d) for d in json.loads(raw_path.read_text())]
        papers = [p for p in papers if not is_known(p.title, known) and p.score >= args.min_score]

    # Stage 2: LLM
    if not args.skip_llm:
        print("\n" + "=" * 50, "\nSTAGE 2: LLM Evaluation\n" + "=" * 50, file=sys.stderr)
        top_n = 30 if args.fast else args.top
        subset = papers[:top_n]
        ctx = cfg.get("paper_context", cfg.get("paper_abstract", ""))
        ev = LLMEvaluator(args.model)
        ev.evaluate_batch(subset, ctx)
        ev.unload()
        papers = subset
        (out / "llm_results.json").write_text(json.dumps([p.to_dict() for p in papers], indent=2))

    # Stage 3: Report
    print("\n" + "=" * 50, "\nSTAGE 3: Report\n" + "=" * 50, file=sys.stderr)
    report_path = out / "final_reading_list.md"
    report_path.write_text(generate_report(papers))
    n = write_paper_files(papers, out / "papers")

    print(f"\nDone! Report: {report_path} | {n} paper files created", file=sys.stderr)


if __name__ == "__main__":
    main()
