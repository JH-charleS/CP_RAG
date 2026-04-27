from fastapi import APIRouter

from db.milvus_client import MilvusClient
from db.mysql_pool import MySQLPool
from db.redis_client import RedisClient

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", summary="Service health check")
async def health_check() -> dict[str, object]:
    redis_ok, mysql_ok, milvus_ok = await RedisClient.health_check(), await MySQLPool.health_check(), await MilvusClient.health_check()
    overall = redis_ok and mysql_ok and milvus_ok
    return {
        "status": "ok" if overall else "degraded",
        "services": {
            "redis": redis_ok,
            "mysql": mysql_ok,
            "milvus": milvus_ok,
        },
    }
