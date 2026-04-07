"""Paper search across Semantic Scholar, Scholar Inbox, and arXiv."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from urllib.parse import quote


# ── Data model ────────────────────────────────────────────────────────────

@dataclass
class Paper:
    title: str = ""
    authors: str = ""
    year: str = ""
    venue: str = ""
    arxiv_id: str = ""
    citation_count: int = 0
    abstract: str = ""
    score: float = 0.0
    category: str = ""
    sources: list[str] = field(default_factory=list)
    source_count: int = 0
    llm_verdict: str = ""
    llm_score: int = 0
    llm_relationship: str = ""
    llm_sections: list[str] = field(default_factory=list)
    llm_reasoning: str = ""
    llm_differentiation: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> Paper:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PaperPool:
    def __init__(self):
        self._papers: dict[str, Paper] = {}

    def add(self, p: Paper, source: str) -> bool:
        key = p.title.lower().strip()
        if not key:
            return False
        if key in self._papers:
            self._papers[key].sources.append(source)
            self._papers[key].source_count += 1
            return False
        p.sources = [source]
        p.source_count = 1
        self._papers[key] = p
        return True

    def add_many(self, papers: list[Paper], source: str) -> int:
        return sum(1 for p in papers if self.add(p, source))

    @property
    def size(self) -> int:
        return len(self._papers)

    def all(self) -> list[Paper]:
        return list(self._papers.values())


# ── HTTP helpers ──────────────────────────────────────────────────────────

def _get(url: str, headers: dict | None = None) -> bytes | None:
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "auto-citetion/1.0")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("    rate limited, waiting 10s…", file=sys.stderr)
            time.sleep(10)
            return _get(url, headers)
        return None
    except Exception:
        return None


def _post(url: str, data: dict, headers: dict | None = None) -> dict | None:
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "auto-citetion/1.0")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(10)
            return _post(url, data, headers)
        return None
    except Exception:
        return None


# ── Parsers ───────────────────────────────────────────────────────────────

def _parse_ss(d: dict) -> Paper | None:
    if not d or not d.get("title"):
        return None
    ext = d.get("externalIds") or {}
    au = d.get("authors") or []
    names = ", ".join(a.get("name", "") for a in au[:4])
    if len(au) > 4:
        names += " et al."
    return Paper(title=d["title"], authors=names, year=str(d.get("year") or ""),
                 venue=d.get("venue") or "", arxiv_id=ext.get("ArXiv", ""),
                 citation_count=d.get("citationCount") or 0,
                 abstract=d.get("abstract") or "")


def _parse_si(d: dict) -> Paper | None:
    if not d or not d.get("title"):
        return None
    y = d.get("publication_date", d.get("year", ""))
    if isinstance(y, str) and len(y) >= 4:
        y = y[:4]
    return Paper(title=d["title"], authors=d.get("authors", ""),
                 year=str(y), venue=d.get("venue", ""),
                 arxiv_id=d.get("arxiv_id", ""), abstract=d.get("abstract", ""))


# ── Semantic Scholar ──────────────────────────────────────────────────────

SS = "https://api.semanticscholar.org/graph/v1"
SS_F = "paperId,externalIds,title,year,venue,citationCount,abstract,authors"


def ss_keyword(pool: PaperPool, queries: list[str]) -> None:
    print("\n[Semantic Scholar] keyword search", file=sys.stderr)
    for i, q in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {q[:60]}", file=sys.stderr)
        raw = _get(f"{SS}/paper/search?query={quote(q)}&limit=20&fields={SS_F}")
        if raw:
            data = json.loads(raw.decode())
            papers = [p for d in data.get("data", []) if (p := _parse_ss(d))]
            pool.add_many(papers, f"ss_kw:{i}")
        time.sleep(0.4)


def ss_citations(pool: PaperPool, arxiv_ids: list[str]) -> None:
    print("\n[Semantic Scholar] citation chains", file=sys.stderr)
    for i, aid in enumerate(arxiv_ids):
        print(f"  [{i+1}/{len(arxiv_ids)}] {aid}", file=sys.stderr)
        for d in ["citations", "references"]:
            raw = _get(f"{SS}/paper/ArXiv:{aid}/{d}?limit=200&fields={SS_F}")
            if raw:
                data = json.loads(raw.decode())
                fld = "citingPaper" if d == "citations" else "citedPaper"
                papers = [p for it in data.get("data", []) if (x := it.get(fld)) and (p := _parse_ss(x))]
                pool.add_many(papers, f"ss_{d}:{aid}")
            time.sleep(0.4)


def ss_authors(pool: PaperPool, authors: list[str]) -> None:
    print("\n[Semantic Scholar] author tracking", file=sys.stderr)
    for i, name in enumerate(authors):
        print(f"  [{i+1}/{len(authors)}] {name}", file=sys.stderr)
        raw = _get(f"{SS}/author/search?query={quote(name)}&limit=1")
        if not raw:
            time.sleep(0.4)
            continue
        data = json.loads(raw.decode())
        if not data.get("data"):
            time.sleep(0.4)
            continue
        aid = data["data"][0].get("authorId")
        if aid:
            raw2 = _get(f"{SS}/author/{aid}/papers?limit=50&fields={SS_F}")
            if raw2:
                data2 = json.loads(raw2.decode())
                papers = [p for d in data2.get("data", []) if (p := _parse_ss(d))]
                pool.add_many(papers, f"author:{name}")
        time.sleep(0.4)


# ── Scholar Inbox ─────────────────────────────────────────────────────────

SI = "https://api.scholar-inbox.com/api"


def _si_h(cookie: str) -> dict:
    return {"Cookie": f"session={cookie}", "Origin": "https://www.scholar-inbox.com",
            "Referer": "https://www.scholar-inbox.com/"}


def si_semantic(pool: PaperPool, queries: list[str], cookie: str) -> None:
    print("\n[Scholar Inbox] semantic search", file=sys.stderr)
    h = _si_h(cookie)
    for i, q in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {q[:60]}…", file=sys.stderr)
        for page in range(3):
            r = _post(f"{SI}/semantic-search", {"text_input": q, "embedding": None, "p": page}, h)
            if not r or not r.get("papers"):
                break
            papers = [p for d in r["papers"] if (p := _parse_si(d))]
            pool.add_many(papers, f"si_sem:{i}")
            time.sleep(1.5)


def si_similar(pool: PaperPool, paper_ids: list[int], cookie: str) -> None:
    print("\n[Scholar Inbox] similar papers", file=sys.stderr)
    h = _si_h(cookie)
    for i, pid in enumerate(paper_ids):
        print(f"  [{i+1}/{len(paper_ids)}] id={pid}", file=sys.stderr)
        raw = _get(f"{SI}/get_similar_papers?paper_id={pid}", h)
        if raw:
            r = json.loads(raw.decode())
            papers = [p for d in r.get("similar_papers", r.get("papers", [])) if (p := _parse_si(d))]
            pool.add_many(papers, f"si_sim:{pid}")
        time.sleep(1.5)


def si_detail(pool: PaperPool, paper_ids: list[int], cookie: str) -> None:
    print("\n[Scholar Inbox] paper refs+cited_by", file=sys.stderr)
    h = _si_h(cookie)
    for i, pid in enumerate(paper_ids):
        print(f"  [{i+1}/{len(paper_ids)}] id={pid}", file=sys.stderr)
        raw = _get(f"{SI}/paper/{pid}", h)
        if raw:
            r = json.loads(raw.decode())
            for key in ["references", "cited_by", "similar_papers"]:
                papers = [p for d in r.get(key, []) if (p := _parse_si(d))]
                pool.add_many(papers, f"si_{key}:{pid}")
        time.sleep(1.5)


def si_collect_ids(query: str, cookie: str, limit: int = 30) -> list[int]:
    r = _post(f"{SI}/semantic-search", {"text_input": query, "embedding": None, "p": 0}, _si_h(cookie))
    if not r or "papers" not in r:
        return []
    return [pid for d in r["papers"][:limit] if (pid := d.get("id", d.get("paper_id")))]


# ── arXiv ─────────────────────────────────────────────────────────────────

def arxiv_search(pool: PaperPool, queries: list[str]) -> None:
    print("\n[arXiv] search", file=sys.stderr)
    ns = "{http://www.w3.org/2005/Atom}"
    for i, q in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {q[:60]}", file=sys.stderr)
        raw = _get(f"http://export.arxiv.org/api/query?search_query={quote(q)}&max_results=30&sortBy=relevance")
        if raw:
            try:
                root = ET.fromstring(raw.decode())
                for e in root.findall(f"{ns}entry"):
                    title = e.findtext(f"{ns}title", "").replace("\n", " ").strip()
                    if not title:
                        continue
                    au = [a.findtext(f"{ns}name", "") for a in e.findall(f"{ns}author")]
                    arxid = ""
                    for lnk in e.findall(f"{ns}link"):
                        href = lnk.get("href", "")
                        if "arxiv.org/abs/" in href:
                            arxid = href.split("/abs/")[-1].split("v")[0]
                    p = Paper(title=title, authors=", ".join(au[:4]) + (" et al." if len(au) > 4 else ""),
                              year=e.findtext(f"{ns}published", "")[:4], venue="arXiv",
                              arxiv_id=arxid, abstract=e.findtext(f"{ns}summary", "").replace("\n", " ").strip())
                    pool.add(p, f"arxiv:{i}")
            except Exception:
                pass
        time.sleep(3)


# ── Scoring ───────────────────────────────────────────────────────────────

HIGH = ["spurious correlation", "shortcut learning", "counterfactual explanation",
        "counterfactual image", "bias discovery", "feature discovery", "model diagnosis",
        "vision-language model", "attention map", "grad-cam", "score-cam",
        "counterfactual generation", "semantic feature", "concept discovery"]
MED = ["explainability", "interpretability", "debiasing", "group robustness",
       "diffusion", "image editing", "saliency", "concept-based", "attribution",
       "vlm", "multimodal"]
LOW = ["robustness", "distribution shift", "imagenet", "augmentation", "causal", "classifier"]

CATEGORIES = {
    "similar_method": ["counterfactual", "feature discovery", "bias discovery", "model diagnosis", "attention map"],
    "counterfactual_xai": ["counterfactual explanation", "counterfactual image", "counterfactual generation"],
    "shortcut_spurious": ["spurious correlation", "shortcut learning", "group robustness", "debiasing"],
    "vlm_multimodal": ["vision-language", "vlm", "multimodal", "clip", "large language model"],
    "diffusion_editing": ["diffusion", "image editing", "text-guided", "generative model"],
    "explainability": ["explainability", "interpretability", "attribution", "grad-cam", "score-cam", "saliency"],
    "augmentation": ["augmentation", "data augmentation", "synthetic data"],
}


def score_and_categorize(papers: list[Paper]) -> None:
    for p in papers:
        text = f"{p.title} {p.abstract}".lower()
        s = sum(3.0 for kw in HIGH if kw in text)
        s += sum(2.0 for kw in MED if kw in text)
        s += sum(1.0 for kw in LOW if kw in text)
        s += min(p.citation_count / 40, 8.0)
        if p.year.isdigit():
            y = int(p.year)
            if y >= 2023: s += 3
            if y >= 2024: s += 2
            if y >= 2025: s += 2
        if p.venue and any(v in p.venue.lower() for v in ["neurips", "icml", "iclr", "cvpr", "eccv", "iccv"]):
            s += 4
        s += p.source_count * 2
        s += len(set(src.split(":")[0] for src in p.sources)) * 3
        p.score = round(max(s, 0), 1)

        best, best_n = "other", 0
        for cat, kws in CATEGORIES.items():
            n = sum(1 for kw in kws if kw in text)
            if n > best_n:
                best, best_n = cat, n
        p.category = best
