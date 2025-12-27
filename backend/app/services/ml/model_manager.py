"""ML Model Manager for loading and managing inference models."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """A loaded ONNX model ready for inference."""

    name: str
    session: "ort.InferenceSession"
    input_name: str
    input_shape: tuple[int, ...]
    class_names: list[str]
    confidence_threshold: float
    nms_threshold: float

    @property
    def input_size(self) -> tuple[int, int]:
        """Return expected input (width, height)."""
        # ONNX models typically use NCHW format: (batch, channels, height, width)
        if len(self.input_shape) == 4:
            return (self.input_shape[3], self.input_shape[2])
        return (640, 640)


# COCO class names for YOLOv8
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
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class ModelManager:
    """Manage ML model loading and caching.

    Handles loading ONNX models, caching them in memory, and providing
    access to loaded models for inference.
    """

    def __init__(self, models_directory: Optional[Path] = None):
        """Initialize model manager.

        Args:
            models_directory: Directory containing model files
        """
        settings = get_settings()
        self.models_directory = models_directory or settings.ml_models_directory
        self._loaded_models: dict[str, LoadedModel] = {}
        self._default_confidence = settings.ml_confidence_threshold
        self._default_nms = settings.ml_nms_threshold

        # Ensure models directory exists
        self.models_directory.mkdir(parents=True, exist_ok=True)

        if ort is None:
            logger.warning("onnxruntime not installed - ML inference unavailable")

    def _get_model_path(self, model_name: str) -> Path:
        """Get the file path for a model by name.

        Args:
            model_name: Model name (without extension)

        Returns:
            Path to model file
        """
        # First check for exact match with .onnx extension
        model_path = self.models_directory / f"{model_name}.onnx"
        if model_path.exists():
            return model_path

        # Check if the name already includes extension
        if model_name.endswith(".onnx"):
            model_path = self.models_directory / model_name
            if model_path.exists():
                return model_path

        return self.models_directory / f"{model_name}.onnx"

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model file exists.

        Args:
            model_name: Model name to check

        Returns:
            True if model file exists
        """
        return self._get_model_path(model_name).exists()

    def list_available_models(self) -> list[str]:
        """List all available model files.

        Returns:
            List of model names (without .onnx extension)
        """
        models = []
        for path in self.models_directory.glob("*.onnx"):
            models.append(path.stem)
        return sorted(models)

    def load_model(
        self,
        model_name: str,
        class_names: Optional[list[str]] = None,
        confidence_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
    ) -> Optional[LoadedModel]:
        """Load an ONNX model for inference.

        Args:
            model_name: Name of model to load
            class_names: Optional custom class names (defaults to COCO)
            confidence_threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold

        Returns:
            LoadedModel instance or None if loading fails
        """
        if ort is None:
            logger.error("Cannot load model: onnxruntime not installed")
            return None

        # Check if already loaded
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        model_path = self._get_model_path(model_name)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None

        try:
            logger.info(f"Loading model: {model_path}")

            # Create inference session
            # Use CPU provider by default, can add CUDA later
            providers = ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(model_path), providers=providers)

            # Get input details
            input_info = session.get_inputs()[0]
            input_name = input_info.name
            input_shape = tuple(input_info.shape)

            # Handle dynamic dimensions (often batch size is dynamic)
            resolved_shape = []
            for dim in input_shape:
                if isinstance(dim, str) or dim is None:
                    resolved_shape.append(1)  # Default batch size
                else:
                    resolved_shape.append(dim)
            input_shape = tuple(resolved_shape)

            loaded_model = LoadedModel(
                name=model_name,
                session=session,
                input_name=input_name,
                input_shape=input_shape,
                class_names=class_names or COCO_CLASSES,
                confidence_threshold=confidence_threshold or self._default_confidence,
                nms_threshold=nms_threshold or self._default_nms,
            )

            self._loaded_models[model_name] = loaded_model
            logger.info(
                f"Model loaded: {model_name}, input shape: {input_shape}, "
                f"classes: {len(loaded_model.class_names)}"
            )
            return loaded_model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory.

        Args:
            model_name: Name of model to unload

        Returns:
            True if model was unloaded, False if not loaded
        """
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            logger.info(f"Model unloaded: {model_name}")
            return True
        return False

    def get_model(self, model_name: str) -> Optional[LoadedModel]:
        """Get a loaded model by name.

        Args:
            model_name: Model name

        Returns:
            LoadedModel or None if not loaded
        """
        return self._loaded_models.get(model_name)

    def unload_all(self) -> None:
        """Unload all models from memory."""
        model_names = list(self._loaded_models.keys())
        for name in model_names:
            self.unload_model(name)
        logger.info("All models unloaded")


# Global model manager instance
model_manager = ModelManager()
