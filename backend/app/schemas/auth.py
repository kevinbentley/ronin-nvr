"""Authentication schemas for request/response validation."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.base import UTCBaseModel


class LoginRequest(BaseModel):
    """Request body for user login."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """Response containing JWT access token."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(UTCBaseModel):
    """Response containing user information."""

    id: int
    username: str
    is_admin: bool
    is_active: bool
    created_at: datetime


class UserCreate(BaseModel):
    """Request body for creating a new user."""

    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8)
    is_admin: bool = False
