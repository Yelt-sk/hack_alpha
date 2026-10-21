"""Seam for the knowledge retriever.

The retriever itself is built separately. This module fixes the contract so both
sides can be developed independently: implement :func:`search` in
``agent/rag_impl.py`` with the same signature and it is picked up automatically.

If no implementation is present, or it raises, the tool reports that cleanly and
the run continues -- retrieval is an advantage, not a dependency.
"""

from __future__ import annotations

from typing import Any

from .tools.registry import tool


def _implementation():
    try:
        from . import rag_impl  # type: ignore[attr-defined]
    except ImportError:
        return None
    return getattr(rag_impl, "search", None)


@tool
def search_knowledge(query: str, top_k: int = 3) -> dict[str, Any]:
    """Look up guidance: which encoder suits which model, how to read a diagnosis,
    what to try for a given data problem, and when a recipe is enough versus when
    the task needs hand-written code.

    Consult it before guessing. One lookup is cheaper than an experiment that was
    never going to work.

    Args:
        query: What you want to know, in plain words.
        top_k: How many passages to return.
    """
    search = _implementation()
    if search is None:
        return {
            "status": "unavailable",
            "note": "No knowledge base is installed in this build.",
            "hint": "Proceed on your own judgement; the other tools are unaffected.",
        }
    try:
        results = search(query, top_k=top_k)
    except Exception as err:  # noqa: BLE001
        return {"status": "error", "problem": f"{type(err).__name__}: {err}",
                "hint": "Retrieval failed; continue without it."}
    return {"status": "ok", "query": query, "results": results}
