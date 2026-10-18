"""
Retrieval over the curated playbook.

Deliberately BM25 over hand-written chunks rather than embeddings over scraped
API docs. Two reasons:

1. A strong model already knows sklearn better than any retrieved doc chunk, so
   dumping upstream documentation adds noise. What it does *not* know is which
   choices this server rewards -- so the corpus is decision rules, not API prose.
2. Lexical search has no model to download, no GPU, no index build. The whole
   retriever is a few hundred lines of stdlib, which matters when the target
   deployment is a small local model on modest hardware.

Swap in embeddings by implementing the same search() signature.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "it", "for", "on", "with",
    "that", "this", "be", "as", "at", "by", "if", "you", "your", "not", "but", "from", "can",
    "will", "do", "does", "when", "what", "how", "why", "which", "use", "using", "used",
}

# Light normalisation so "overfitting" matches "overfit" without a stemmer dep.
_SUFFIXES = ("ization", "isation", "ingly", "ing", "edly", "ed", "es", "s", "ly")


def _norm(tok: str) -> str:
    for suf in _SUFFIXES:
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    """Emit snake_case identifiers both whole and split.

    Without this, the corpus token `roc_auc` never matches a user typing
    "roc auc", which is how people actually write it.
    """
    out: list[str] = []
    for t in re.findall(r"[a-z0-9_]{2,}", text.lower()):
        if t in _STOP:
            continue
        out.append(_norm(t))
        if "_" in t:
            out.extend(_norm(p) for p in t.split("_") if len(p) >= 2 and p not in _STOP)
    return out


class Chunk:
    __slots__ = ("id", "title", "source", "text", "tokens", "tf", "length")

    def __init__(self, cid: str, title: str, source: str, text: str):
        self.id, self.title, self.source, self.text = cid, title, source, text
        # The heading is repeated so a chunk's own topic carries extra weight.
        self.tokens = tokenize(title) * 3 + tokenize(text)
        self.tf = Counter(self.tokens)
        self.length = len(self.tokens)


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.72):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.avg_len = (sum(c.length for c in chunks) / len(chunks)) if chunks else 0.0
        self.df: Counter = Counter()
        for c in chunks:
            self.df.update(set(c.tf))
        self.n = len(chunks)

    def _idf(self, term: str) -> float:
        d = self.df.get(term, 0)
        return math.log(1 + (self.n - d + 0.5) / (d + 0.5))

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        if not q:
            return []
        scored: list[tuple[Chunk, float]] = []
        for c in self.chunks:
            s = 0.0
            for term in q:
                f = c.tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * c.length / (self.avg_len or 1))
                s += self._idf(term) * f * (self.k1 + 1) / denom
            if s > 0:
                scored.append((c, s))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]


def _split_markdown(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    parts = re.split(r"^## ", raw, flags=re.MULTILINE)
    out: list[Chunk] = []
    preamble = parts[0].strip()
    doc_title = preamble.splitlines()[0].lstrip("# ").strip() if preamble else path.stem
    for i, part in enumerate(parts[1:], start=1):
        lines = part.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue
        out.append(Chunk(f"{path.stem}#{i}", f"{doc_title} / {title}", path.name, body))
    return out


@lru_cache(maxsize=1)
def get_index() -> BM25Index:
    chunks: list[Chunk] = []
    for f in sorted(KNOWLEDGE_DIR.glob("*.md")):
        chunks.extend(_split_markdown(f))
    return BM25Index(chunks)


def search(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    hits = get_index().search(query, top_k=top_k)
    if not hits:
        return []
    top = hits[0][1]
    return [{"title": c.title, "source": c.source, "relevance": round(s / top, 3), "content": c.text}
            for c, s in hits]


def corpus_stats() -> dict[str, Any]:
    idx = get_index()
    return {"chunks": idx.n, "files": sorted({c.source for c in idx.chunks}),
            "topics": [c.title for c in idx.chunks]}
