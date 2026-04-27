from __future__ import annotations

import asyncio
from typing import Any

from pymilvus import connections, utility

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
