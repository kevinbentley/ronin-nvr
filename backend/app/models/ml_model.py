"""ML model registry for tracking available models."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Float,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class MLModel(Base):
    """Registered ML model for inference."""

    __tablename__ = "ml_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Model identification
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(50), default="1.0.0")

    # Model file location
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "onnx", "pytorch"

    # Model capabilities
    class_names: Mapped[list] = mapped_column(JSON, nullable=False)
    input_size: Mapped[list] = mapped_column(JSON, nullable=False)  # [width, height]

    # Configuration defaults
    default_confidence_threshold: Mapped[float] = mapped_column(Float, default=0.5)
    default_nms_threshold: Mapped[float] = mapped_column(Float, default=0.45)

    # Status
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)

    # Additional info
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Timestamps (using TIMESTAMP WITH TIME ZONE for proper UTC handling)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<MLModel(name={self.name}, type={self.model_type}, "
            f"enabled={self.is_enabled}, default={self.is_default})>"
        )
