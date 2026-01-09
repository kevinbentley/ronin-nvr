"""API routes for RoninNVR."""

from fastapi import APIRouter

from app.api import auth, cameras, health, ml, onvif, playback, storage

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router)
api_router.include_router(cameras.router)
api_router.include_router(storage.router)
api_router.include_router(playback.router)
api_router.include_router(ml.router)
api_router.include_router(onvif.router)
