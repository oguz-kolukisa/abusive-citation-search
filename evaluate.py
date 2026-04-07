"""LLM-based paper relevance evaluation using a local HuggingFace model."""

from __future__ import annotations

import json
import sys

from search import Paper

PROMPT = """You are evaluating whether a candidate paper should be cited in a research paper.

RESEARCH PAPER:
{context}

CANDIDATE PAPER:
Title: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}

Respond in EXACTLY this JSON format (no other text):
{{
  "verdict": "must_cite" | "should_cite" | "maybe_cite" | "skip",
  "relevance_score": <1-10>,
  "relationship": "<similar_method|builds_upon|we_build_upon|same_problem|competing_approach|foundational|supports_claim|provides_benchmark|tangentially_related>",
  "cite_in_sections": ["<section1>", "<section2>"],
  "reasoning": "<2-3 sentences>",
  "differentiation": "<1 sentence if similar, else empty>"
}}

Be strict: "must_cite" only for very similar or foundational work. "skip" for tangential papers."""


class LLMEvaluator:

    def __init__(self, model_id: str = "google/gemma-4-E4B-it"):
        self.model_id = model_id
        self._model = None
        self._proc = None

    def load(self) -> None:
        if self._model:
            return
        print(f"Loading {self.model_id}…", file=sys.stderr)
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch

        self._proc = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="auto",
        )
        print(f"Loaded on {self._model.device}", file=sys.stderr)

    def evaluate(self, paper: Paper, context: str) -> None:
        self.load()
        prompt = PROMPT.format(
            context=context, title=paper.title, authors=paper.authors,
            year=paper.year, abstract=paper.abstract[:1500],
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self._proc(text=text, return_tensors="pt").to(self._model.device)
        n = inputs["input_ids"].shape[-1]
        out = self._model.generate(**inputs, max_new_tokens=512, temperature=0.3, top_p=0.9, do_sample=True)
        resp = self._proc.decode(out[0][n:], skip_special_tokens=True).strip()

        start, end = resp.find("{"), resp.rfind("}") + 1
        if start >= 0 and end > 0:
            try:
                d = json.loads(resp[start:end])
                paper.llm_verdict = d.get("verdict", "skip")
                paper.llm_score = int(d.get("relevance_score", 0))
                paper.llm_relationship = d.get("relationship", "")
                paper.llm_sections = d.get("cite_in_sections", [])
                paper.llm_reasoning = d.get("reasoning", "")
                paper.llm_differentiation = d.get("differentiation", "")
                return
            except (json.JSONDecodeError, ValueError):
                pass
        paper.llm_verdict = "error"
        paper.llm_reasoning = f"Parse failed: {resp[:200]}"

    def evaluate_batch(self, papers: list[Paper], context: str) -> None:
        for i, p in enumerate(papers):
            print(f"  [{i+1}/{len(papers)}] {p.title[:60]}…", file=sys.stderr)
            try:
                self.evaluate(p, context)
            except Exception as e:
                print(f"    error: {e}", file=sys.stderr)
                p.llm_verdict = "error"
                p.llm_reasoning = str(e)

    def unload(self) -> None:
        if self._model:
            del self._model, self._proc
            self._model = self._proc = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
