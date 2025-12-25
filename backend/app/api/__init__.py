"""API routes for RoninNVR."""

from fastapi import APIRouter

from app.api import health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
