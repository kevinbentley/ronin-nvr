#!/usr/bin/env python3
"""Register the YOLOv8n model in the database."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from app.database import async_session_maker
from app.models.ml_model import MLModel
from app.config import get_settings

# YOLOv8n COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


async def register_model():
    """Register YOLOv8n model in database."""
    settings = get_settings()
    model_path = settings.ml_models_directory / "yolov8n.onnx"

    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        return False

    async with async_session_maker() as session:
        # Check if already registered
        result = await session.execute(
            select(MLModel).where(MLModel.name == "yolov8n")
        )
        existing = result.scalar_one_or_none()

        if existing:
            print(f"Model 'yolov8n' already registered (id={existing.id})")
            return True

        # Create new model entry
        model = MLModel(
            name="yolov8n",
            display_name="YOLOv8 Nano",
            version="8.0",
            file_path=str(model_path),
            model_type="onnx",
            class_names=COCO_CLASSES,
            input_size=[640, 640],
            default_confidence_threshold=0.5,
            default_nms_threshold=0.45,
            is_enabled=True,
            is_default=True,
            description="YOLOv8 Nano - Fast object detection model trained on COCO dataset",
        )

        session.add(model)
        await session.commit()
        await session.refresh(model)

        print(f"Registered model 'yolov8n' (id={model.id})")
        print(f"  File: {model.file_path}")
        print(f"  Classes: {len(model.class_names)}")
        print(f"  Input size: {model.input_size}")
        print(f"  Confidence threshold: {model.default_confidence_threshold}")

    return True


if __name__ == "__main__":
    success = asyncio.run(register_model())
    sys.exit(0 if success else 1)
