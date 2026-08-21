from __future__ import annotations

from typing import Any

from encode import encode_tokens

_index: dict[str, dict[str, list[list[float]]]] = {}
_texts: dict[str, dict[str, str]] = {}


def reset(brain_id: str | None = None) -> None:
    if brain_id is None:
        _index.clear()
        _texts.clear()
        return
    _index.pop(brain_id, None)
    _texts.pop(brain_id, None)


def stats(brain_id: str) -> dict[str, Any]:
    docs = _index.get(brain_id) or {}
    return {"brain_id": brain_id, "n_docs": len(docs)}


def encodings(brain_id: str) -> dict[str, list[list[float]]]:
    return dict(_index.get(brain_id) or {})


def maxsim(query: list[list[float]], doc: list[list[float]]) -> float:
    if not query or not doc:
        return 0.0
    try:
        import numpy as np
    except ImportError:
        return _maxsim_loop(query, doc)
    q = np.asarray(query, dtype=np.float32)
    d = np.asarray(doc, dtype=np.float32)
    if q.ndim != 2 or d.ndim != 2 or q.size == 0 or d.size == 0:
        return 0.0
    qn = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    dn = d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-12, None)
    sims = qn @ dn.T
    return float(sims.max(axis=1).sum())


def _maxsim_loop(query: list[list[float]], doc: list[list[float]]) -> float:
    total = 0.0
    for q_vec in query:
        best = 0.0
        q_norm = _norm(q_vec)
        if q_norm <= 0:
            continue
        for d_vec in doc:
            d_norm = _norm(d_vec)
            if d_norm <= 0:
                continue
            score = _dot(q_vec, d_vec) / (q_norm * d_norm)
            if score > best:
                best = score
        total += best
    return total


def _dot(left: list[float], right: list[float]) -> float:
    n = min(len(left), len(right))
    return sum(left[i] * right[i] for i in range(n))


def _norm(vector: list[float]) -> float:
    return sum(x * x for x in vector) ** 0.5


def index_chunks(
    brain_id: str,
    chunks: list[dict[str, str]],
    *,
    replace: bool = True,
) -> dict[str, Any]:
    if replace:
        reset(brain_id)
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        text = str(chunk.get("text") or "")
        if not chunk_id:
            continue
        _index.setdefault(brain_id, {})[chunk_id] = encode_tokens(text)
        _texts.setdefault(brain_id, {})[chunk_id] = text
    return stats(brain_id)


def retrieve(
    query: str,
    brain_id: str,
    k: int,
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    docs = _index.get(brain_id) or {}
    if not docs:
        return [], {}, {}
    q_toks = encode_tokens(query)
    scored = [
        (chunk_id, maxsim(q_toks, tokens))
        for chunk_id, tokens in docs.items()
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    ids = [chunk_id for chunk_id, _ in scored[: max(1, int(k))]]
    texts = _texts.get(brain_id) or {}
    return (
        ids,
        {chunk_id: score for chunk_id, score in scored if chunk_id in set(ids)},
        {chunk_id: texts.get(chunk_id, "") for chunk_id in ids},
    )
