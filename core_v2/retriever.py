from __future__ import annotations

from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

from core.config import get_settings

_VECTORSTORE: Milvus | None = None


def _get_vectorstore() -> Milvus:
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    settings = get_settings()
    embedding = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
    _VECTORSTORE = Milvus(
        embedding_function=embedding,
        collection_name=settings.v2_milvus_collection,
        connection_args={
            "uri": uri,
            "user": settings.milvus_user or None,
            "password": settings.milvus_password or None,
            "db_name": settings.milvus_db_name,
        },
        text_field="child_text",
        vector_field="embedding",
        auto_id=False,
    )
    return _VECTORSTORE


def _format_hit(doc: Any, score: float, idx: int) -> str:
    metadata = doc.metadata or {}
    parent_text = metadata.get("parent_text", "")
    return (
        f"[RAG#{idx}] score={score:.4f}\n"
        f"source={metadata.get('source', 'unknown')} title={metadata.get('title', 'unknown')}\n"
        f"child_text={doc.page_content}\n"
        f"parent_text={parent_text}"
    )


def retrieve_with_parent(query: str) -> tuple[list[dict[str, Any]], str]:
    settings = get_settings()
    store = _get_vectorstore()
    docs = store.similarity_search_with_score(query, k=settings.v2_rag_top_k)
    hits: list[dict[str, Any]] = []
    chunks: list[str] = []
    for idx, (doc, score) in enumerate(docs, start=1):
        hit = {"score": float(score), "content": doc.page_content, "metadata": doc.metadata}
        hits.append(hit)
        chunks.append(_format_hit(doc, float(score), idx))
    return hits, ("\n\n".join(chunks) if chunks else "未命中 v2 Milvus 结果。")
