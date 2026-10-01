from fastapi import APIRouter

from app.api.v1.routes.history import router as history_router
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.predict import router as prediction_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(prediction_router, tags=["prediction"])
api_router.include_router(history_router, tags=["prediction history"])
