from __future__ import annotations

from redis.asyncio import ConnectionPool, Redis

from core.config import get_settings


class RedisClient:
    """Async Redis singleton client manager."""

    _pool: ConnectionPool | None = None
    _client: Redis | None = None

    @classmethod
    async def init(cls) -> Redis:
        """Initialize and return Redis client singleton."""
        if cls._client is not None:
            return cls._client

        settings = get_settings()
        cls._pool = ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
        cls._client = Redis(connection_pool=cls._pool)
        await cls._client.ping()
        return cls._client

    @classmethod
    def get_client(cls) -> Redis:
        """Get initialized Redis client, raise if not initialized."""
        if cls._client is None:
            raise RuntimeError("Redis client has not been initialized. Call RedisClient.init() first.")
        return cls._client

    @classmethod
    async def health_check(cls) -> bool:
        """Check Redis connectivity."""
        if cls._client is None:
            return False
        try:
            await cls._client.ping()
            return True
        except Exception:
            return False

    @classmethod
    async def close(cls) -> None:
        """Close Redis client and underlying pool."""
        if cls._client is not None:
            await cls._client.aclose()
            cls._client = None

        if cls._pool is not None:
            await cls._pool.disconnect()
            cls._pool = None
