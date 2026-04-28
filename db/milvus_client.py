from __future__ import annotations

import asyncio
from typing import Any

from pymilvus import Collection, connections, utility

from core.config import get_settings


class MilvusClient:
    """Milvus singleton with asyncio-friendly wrappers."""

    _connected: bool = False
    _alias: str = "default"

    @classmethod
    async def init(cls) -> None:
        """Initialize Milvus connection singleton."""
        if cls._connected:
            return

        settings = get_settings()
        kwargs: dict[str, Any] = {
            "alias": cls._alias,
            "host": settings.milvus_host,
            "port": settings.milvus_port,
            "db_name": settings.milvus_db_name,
            "timeout": settings.milvus_timeout,
            "secure": settings.milvus_secure,
        }
        if settings.milvus_user:
            kwargs["user"] = settings.milvus_user
        if settings.milvus_password:
            kwargs["password"] = settings.milvus_password

        await asyncio.to_thread(connections.connect, **kwargs)
        cls._connected = True

    @classmethod
    def get_alias(cls) -> str:
        """Get current Milvus connection alias."""
        if not cls._connected:
            raise RuntimeError("Milvus client has not been initialized. Call MilvusClient.init() first.")
        return cls._alias

    @classmethod
    async def search(
        cls,
        *,
        collection_name: str,
        query_vector: list[float],
        vector_field: str,
        output_fields: list[str],
        limit: int,
        metric_type: str = "IP",
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run vector similarity search and return normalized hits."""
        if not cls._connected:
            raise RuntimeError("Milvus client has not been initialized. Call MilvusClient.init() first.")

        search_params: dict[str, Any] = {"metric_type": metric_type, "params": params or {"nprobe": 10}}

        def _search_sync() -> list[dict[str, Any]]:
            collection = Collection(name=collection_name, using=cls._alias)
            collection.load()
            results = collection.search(
                data=[query_vector],
                anns_field=vector_field,
                param=search_params,
                limit=limit,
                output_fields=output_fields,
            )
            normalized: list[dict[str, Any]] = []
            for batch in results:
                for hit in batch:
                    entity = hit.entity
                    payload: dict[str, Any] = {}
                    for field in output_fields:
                        payload[field] = entity.get(field) if entity is not None else None
                    payload["score"] = float(hit.score)
                    normalized.append(payload)
            return normalized

        return await asyncio.to_thread(_search_sync)

    @classmethod
    async def has_collection(cls, collection_name: str) -> bool:
        """Check whether a Milvus collection exists."""
        if not cls._connected:
            return False
        return await asyncio.to_thread(
            utility.has_collection,
            collection_name=collection_name,
            using=cls._alias,
        )

    @classmethod
    async def health_check(cls) -> bool:
        """Check Milvus connectivity."""
        if not cls._connected:
            return False
        try:
            await asyncio.to_thread(connections.get_connection_addr, cls._alias)
            return True
        except Exception:
            return False

    @classmethod
    async def close(cls) -> None:
        """Disconnect Milvus singleton connection."""
        if not cls._connected:
            return
        await asyncio.to_thread(connections.disconnect, cls._alias)
        cls._connected = False
