"""API routes for RoninNVR."""

from fastapi import APIRouter

from app.api import cameras, health, recordings, storage

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(cameras.router)
api_router.include_router(recordings.router)
api_router.include_router(storage.router)
