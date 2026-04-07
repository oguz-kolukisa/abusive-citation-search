# Auto-Citetion

Find papers you should cite. Searches 3 academic APIs, then scores each paper with a local LLM.

## How it works

```
config.json → Search (7 strategies) → Score & Filter → LLM Evaluate → final_reading_list.md
```

**Search strategies:**
- Scholar Inbox semantic search + similar papers + refs/cited_by
- Semantic Scholar keyword search + citation chains + author tracking
- arXiv API search

**LLM evaluation** (gemma-4-E4B-it by default):
- Verdict: must_cite / should_cite / maybe_cite / skip
- Which section to cite in
- Why to cite + how your work differs

## Setup

```bash
git clone https://github.com/oguz-kolukisa/auto-citetion.git
cd auto-citetion
uv sync
```

**Scholar Inbox** (optional, best semantic search):
1. Log into [scholar-inbox.com](https://www.scholar-inbox.com)
2. DevTools (F12) → Application → Cookies → copy `session` value
3. `echo "your_cookie" > .scholar_inbox_cookie`

**LLM** requires GPU with ~16GB VRAM and HuggingFace login (`huggingface-cli login`).

## Usage

### 1. Create config

```bash
cp example_config.json my_paper.json
# Edit with your paper's abstract, queries, seed papers, key authors
```

### 2. Run

```bash
# Full pipeline
uv run python auto_citetion.py my_paper.json -o results/

# Without LLM (just search + keyword scoring)
uv run python auto_citetion.py my_paper.json --skip-llm

# Re-evaluate existing results with LLM
uv run python auto_citetion.py my_paper.json --skip-search

# Quick test
uv run python auto_citetion.py my_paper.json --fast

# Different model
uv run python auto_citetion.py my_paper.json --model google/gemma-4-E2B-it
```

### Options

```
config              Config JSON file (required)
-o, --output DIR    Output directory (default: .)
--refs PATH         references.md for dedup
--skip-search       Skip search, use existing results
--skip-si           Skip Scholar Inbox
--skip-llm          Skip LLM evaluation
--model ID          HuggingFace model (default: google/gemma-4-E4B-it)
--top N             Papers to evaluate with LLM (default: 80)
--min-score N       Minimum keyword score (default: 4.0)
--fast              Fewer queries + top 30 only
```

## Output

```
results/
├── final_reading_list.md   # Grouped by verdict + topic with reasoning
├── papers/                 # One markdown file per paper
├── all_candidates.json     # Raw search results
└── llm_results.json        # LLM evaluations
```

## Config format

```json
{
  "paper_context": "Full paper description for LLM (title, abstract, contributions, sections)",
  "paper_abstract": "Short abstract for Scholar Inbox search",
  "scholar_inbox_queries": ["1-2 sentence descriptions of your paper"],
  "semantic_scholar_queries": ["keyword combinations"],
  "seed_arxiv_ids": ["2004.07780"],
  "key_authors": ["Author Name"],
  "arxiv_queries": ["all:\"keyword\" AND cat:cs.CV"]
}
```

## Files

```
auto_citetion.py  — CLI entry point (search → evaluate → report)
search.py         — All search strategies + scoring
evaluate.py       — LLM evaluation with HuggingFace transformers
example_config.json
```

## License

MIT
