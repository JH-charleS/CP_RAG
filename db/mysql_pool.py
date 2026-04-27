from __future__ import annotations

from typing import Any

import aiomysql
from aiomysql import Pool

from core.config import get_settings


class MySQLPool:
    """Async MySQL connection pool singleton manager."""

    _pool: Pool | None = None

    @classmethod
    async def init(cls) -> Pool:
        """Initialize and return MySQL connection pool singleton."""
        if cls._pool is not None:
            return cls._pool

        settings = get_settings()
        cls._pool = await aiomysql.create_pool(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            db=settings.mysql_db,
            charset=settings.mysql_charset,
            minsize=settings.mysql_min_size,
            maxsize=settings.mysql_max_size,
            connect_timeout=settings.mysql_connect_timeout,
            autocommit=settings.mysql_autocommit,
        )
        return cls._pool

    @classmethod
    def get_pool(cls) -> Pool:
        """Get initialized pool, raise if not initialized."""
        if cls._pool is None:
            raise RuntimeError("MySQL pool has not been initialized. Call MySQLPool.init() first.")
        return cls._pool

    @classmethod
    async def health_check(cls) -> bool:
        """Check MySQL connectivity using lightweight query."""
        if cls._pool is None:
            return False
        try:
            async with cls._pool.acquire() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    row: tuple[Any, ...] | None = await cursor.fetchone()
                    return bool(row and row[0] == 1)
        except Exception:
            return False

    @classmethod
    async def close(cls) -> None:
        """Gracefully close MySQL pool."""
        if cls._pool is not None:
            cls._pool.close()
            await cls._pool.wait_closed()
            cls._pool = None
