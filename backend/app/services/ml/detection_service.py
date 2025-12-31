"""Detection service for running ML inference on frames."""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.services.ml.model_manager import LoadedModel, ModelManager, model_manager

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """A single detection result from inference."""

    class_name: str
    class_id: int
    confidence: float
    # Bounding box in normalized coordinates (0-1)
    x: float
    y: float
    width: float
    height: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> tuple[float, float]:
        """Return center point of bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)


class DetectionService:
    """Service for running object detection on frames.

    Uses YOLO-style models via ONNX Runtime to detect objects in video frames.
    """

    def __init__(self, model_mgr: Optional[ModelManager] = None):
        """Initialize detection service.

        Args:
            model_mgr: Optional ModelManager instance (uses global if not provided)
        """
        self.model_manager = model_mgr or model_manager

    def preprocess_frame(
        self,
        frame: np.ndarray,
        target_size: tuple[int, int] = (640, 640),
    ) -> tuple[np.ndarray, float, float]:
        """Preprocess a frame for YOLO inference.

        Args:
            frame: Input frame in BGR format (H, W, 3)
            target_size: Target size (width, height)

        Returns:
            Tuple of (preprocessed_frame, scale_x, scale_y)
            - preprocessed_frame: (1, 3, H, W) float32 array normalized to 0-1
            - scale_x, scale_y: Scale factors to convert predictions back to original size
        """
        original_height, original_width = frame.shape[:2]
        target_width, target_height = target_size

        # Calculate scale factors
        scale_x = original_width / target_width
        scale_y = original_height / target_height

        # Resize frame
        resized = cv2.resize(frame, target_size)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to 0-1 and convert to float32
        normalized = rgb.astype(np.float32) / 255.0

        # Transpose from HWC to CHW format
        transposed = normalized.transpose(2, 0, 1)

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched, scale_x, scale_y

    def postprocess_yolo(
        self,
        outputs: np.ndarray,
        scale_x: float,
        scale_y: float,
        model: LoadedModel,
        original_width: int,
        original_height: int,
    ) -> list[DetectionResult]:
        """Post-process YOLO model outputs.

        Args:
            outputs: Raw model output
            scale_x, scale_y: Scale factors for coordinate conversion
            model: Loaded model with class names and thresholds
            original_width, original_height: Original frame dimensions

        Returns:
            List of DetectionResult objects
        """
        # YOLOv8 output format: (1, num_classes + 4, num_predictions)
        # Transpose to (num_predictions, num_classes + 4)
        predictions = outputs[0].T

        # Get confidence threshold
        conf_threshold = model.confidence_threshold
        nms_threshold = model.nms_threshold

        boxes = []
        scores = []
        class_ids = []

        for prediction in predictions:
            # First 4 values are x_center, y_center, width, height
            x_center, y_center, width, height = prediction[:4]

            # Remaining values are class scores
            class_scores = prediction[4:]

            # Get best class
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence < conf_threshold:
                continue

            # Convert from center format to corner format
            x1 = (x_center - width / 2) * scale_x
            y1 = (y_center - height / 2) * scale_y
            box_width = width * scale_x
            box_height = height * scale_y

            # Clip to frame bounds
            x1 = max(0, min(x1, original_width))
            y1 = max(0, min(y1, original_height))
            box_width = min(box_width, original_width - x1)
            box_height = min(box_height, original_height - y1)

            boxes.append([x1, y1, box_width, box_height])
            scores.append(float(confidence))
            class_ids.append(int(class_id))

        if not boxes:
            return []

        # Apply Non-Maximum Suppression
        boxes_array = np.array(boxes)
        scores_array = np.array(scores)

        # OpenCV NMS expects boxes as (x, y, w, h)
        indices = cv2.dnn.NMSBoxes(
            boxes_array.tolist(),
            scores_array.tolist(),
            conf_threshold,
            nms_threshold,
        )

        results = []
        for idx in indices:
            # Handle both old and new OpenCV API
            if isinstance(idx, (list, np.ndarray)):
                idx = idx[0]

            box = boxes_array[idx]
            class_id = class_ids[idx]

            # Normalize coordinates to 0-1
            norm_x = box[0] / original_width
            norm_y = box[1] / original_height
            norm_width = box[2] / original_width
            norm_height = box[3] / original_height

            # Get class name
            if class_id < len(model.class_names):
                class_name = model.class_names[class_id]
            else:
                class_name = f"class_{class_id}"

            results.append(DetectionResult(
                class_name=class_name,
                class_id=class_id,
                confidence=scores[idx],
                x=float(norm_x),
                y=float(norm_y),
                width=float(norm_width),
                height=float(norm_height),
            ))

        return results

    def detect(
        self,
        frame: np.ndarray,
        model_name: str,
        confidence_threshold: Optional[float] = None,
    ) -> list[DetectionResult]:
        """Run object detection on a frame.

        Args:
            frame: Input frame in BGR format (H, W, 3)
            model_name: Name of model to use
            confidence_threshold: Optional override for confidence threshold

        Returns:
            List of DetectionResult objects
        """
        # Load model if needed
        model = self.model_manager.load_model(model_name)
        if model is None:
            logger.error(f"Could not load model: {model_name}")
            return []

        # Override confidence if specified
        if confidence_threshold is not None:
            original_conf = model.confidence_threshold
            model.confidence_threshold = confidence_threshold

        try:
            original_height, original_width = frame.shape[:2]

            # Preprocess
            input_size = model.input_size
            preprocessed, scale_x, scale_y = self.preprocess_frame(frame, input_size)

            # Run inference
            outputs = model.session.run(None, {model.input_name: preprocessed})

            # Post-process
            results = self.postprocess_yolo(
                outputs[0],
                scale_x,
                scale_y,
                model,
                original_width,
                original_height,
            )

            return results

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

        finally:
            # Restore original confidence threshold
            if confidence_threshold is not None:
                model.confidence_threshold = original_conf

    def detect_batch(
        self,
        frames: list[np.ndarray],
        model_name: str,
    ) -> list[list[DetectionResult]]:
        """Run detection on multiple frames.

        Note: Currently processes frames sequentially. Batch inference
        could be added for GPU acceleration.

        Args:
            frames: List of frames in BGR format
            model_name: Name of model to use

        Returns:
            List of detection results, one list per frame
        """
        return [self.detect(frame, model_name) for frame in frames]


# Global detection service instance
detection_service = DetectionService()
