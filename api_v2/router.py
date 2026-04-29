from fastapi import APIRouter

from api.routes.health import router as health_router
from api_v2.routes.query import router as query_router

api_router_v2 = APIRouter()
api_router_v2.include_router(health_router)
api_router_v2.include_router(query_router)
