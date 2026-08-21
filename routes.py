from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from src.core.plugins.context import PluginContext


class IndexBody(BaseModel):
    brain_id: str = "searchbenchsmoke"
    limit: int = Field(1000, ge=1, le=20000)


def _list_chunks(data_adapter: Any, brain_id: str, limit: int) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    skip = 0
    page = min(100, limit)
    while len(chunks) < limit:
        batch, total = data_adapter.get_text_chunks(
            brain_id, page, skip, None, None, "desc"
        )
        if not batch:
            break
        for chunk in batch:
            chunks.append({"id": str(chunk.id), "text": chunk.text or ""})
            if len(chunks) >= limit:
                break
        skip += len(batch)
        if total is not None and skip >= int(total):
            break
    return chunks


def create_router(context: PluginContext) -> APIRouter:
    router = APIRouter(prefix="/search-colbert", tags=["search-colbert-plugin"])
    data = context.adapters.data

    @router.get("/health")
    def health(brain_id: Optional[str] = Query(default=None)) -> dict[str, Any]:
        from encode import status
        from index import stats

        payload = status()
        if brain_id:
            payload["index"] = stats(brain_id)
        return payload

    @router.post("/index")
    def build_index(body: IndexBody) -> dict[str, Any]:
        from index import index_chunks

        chunks = _list_chunks(data, body.brain_id, body.limit)
        return index_chunks(body.brain_id, chunks)

    return router
