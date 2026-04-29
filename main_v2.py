from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api_v2.router import api_router_v2
from core.config import get_settings
from db.milvus_client import MilvusClient
from db.mysql_pool import MySQLPool
from db.redis_client import RedisClient


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await RedisClient.init()
    await MySQLPool.init()
    await MilvusClient.init()
    try:
        yield
    finally:
        await MilvusClient.close()
        await MySQLPool.close()
        await RedisClient.close()


settings = get_settings()
app = FastAPI(
    title=f"{settings.app_name}_v2",
    debug=settings.app_debug,
    lifespan=lifespan,
)
app.include_router(api_router_v2, prefix=settings.v2_api_prefix)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.get("/", include_in_schema=False)
async def index_v2() -> FileResponse:
    return FileResponse(WEB_DIR / "v2_test.html")
