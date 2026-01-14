"""Detection methods for the benchmark framework.

Implements multiple detection approaches:
- YOLO (v8n and 11l) for object detection
- MOG2 for background subtraction
- Frame differencing for simple motion
- Edge detection for scene changes
- Corruption detection for image quality issues
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import BenchmarkConfig
from .models import Detection, DetectionMethod, EventType

logger = logging.getLogger(__name__)

# COCO class to EventType mapping
COCO_TO_EVENT_TYPE = {
    "person": EventType.PERSON,
    "bicycle": EventType.VEHICLE,
    "car": EventType.VEHICLE,
    "motorcycle": EventType.VEHICLE,
    "bus": EventType.VEHICLE,
    "train": EventType.VEHICLE,
    "truck": EventType.VEHICLE,
    "boat": EventType.VEHICLE,
    "bird": EventType.ANIMAL,
    "cat": EventType.ANIMAL,
    "dog": EventType.ANIMAL,
    "horse": EventType.ANIMAL,
    "sheep": EventType.ANIMAL,
    "cow": EventType.ANIMAL,
    "elephant": EventType.ANIMAL,
    "bear": EventType.ANIMAL,
    "zebra": EventType.ANIMAL,
    "giraffe": EventType.ANIMAL,
}


class BaseDetector(ABC):
    """Abstract base class for detectors."""

    method: DetectionMethod
    processing_time: float = 0.0
    frames_processed: int = 0

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_seconds: float,
        previous_frame: np.ndarray | None = None,
    ) -> list[Detection]:
        """Run detection on a frame.

        Args:
            frame: BGR frame to analyze
            frame_number: Frame number in video
            timestamp_seconds: Timestamp in video
            previous_frame: Previous frame (for motion-based detectors)

        Returns:
            List of Detection objects
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state (for video transitions)."""
        pass


class YOLODetector(BaseDetector):
    """YOLO object detector using ONNX Runtime."""

    def __init__(
        self,
        model_path: Path,
        method: DetectionMethod,
        confidence_threshold: float = 0.25,
        input_size: tuple[int, int] = (640, 640),
    ):
        """Initialize YOLO detector.

        Args:
            model_path: Path to ONNX model
            method: Detection method identifier
            confidence_threshold: Minimum confidence for detections
            input_size: Model input size (width, height)
        """
        self.model_path = model_path
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size

        self._session = None
        self._input_name = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of ONNX Runtime session."""
        if self._initialized:
            return

        try:
            import onnxruntime as ort

            providers = []
            available = ort.get_available_providers()

            if "CUDAExecutionProvider" in available:
                providers.append(
                    (
                        "CUDAExecutionProvider",
                        {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"},
                    )
                )
            providers.append("CPUExecutionProvider")

            self._session = ort.InferenceSession(str(self.model_path), providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._initialized = True

            active = self._session.get_providers()
            logger.info(f"YOLO {self.method.value} initialized with providers: {active}")

        except ImportError:
            raise RuntimeError("ONNX Runtime not available")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize YOLO: {e}")

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Preprocess frame for YOLO inference."""
        original_h, original_w = frame.shape[:2]
        target_w, target_h = self.input_size

        # Resize with letterboxing
        scale = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # Convert to model input format
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, 0)  # Add batch dimension

        return blob, scale, (pad_x, pad_y)

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        padding: tuple[int, int],
        frame_shape: tuple[int, int],
    ) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        """Postprocess YOLO output."""
        # YOLO output shape: (1, 84, 8400) for YOLOv8/11
        # 84 = 4 (bbox) + 80 (classes)
        predictions = output[0]  # Remove batch dimension

        # Transpose to (8400, 84)
        if predictions.shape[0] == 84:
            predictions = predictions.T

        # Extract boxes and class scores
        boxes = predictions[:, :4]  # x_center, y_center, width, height
        class_scores = predictions[:, 4:]

        # Get best class for each prediction
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Filter by confidence
        mask = confidences >= self.confidence_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return []

        # Convert from center format to corner format
        x_center, y_center, width, height = boxes.T
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Remove padding and scale back to original size
        pad_x, pad_y = padding
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Clip to frame bounds
        original_h, original_w = frame_shape
        x1 = np.clip(x1, 0, original_w)
        y1 = np.clip(y1, 0, original_h)
        x2 = np.clip(x2, 0, original_w)
        y2 = np.clip(y2, 0, original_h)

        # NMS
        boxes_for_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            confidences.tolist(),
            self.confidence_threshold,
            0.45,
        )

        # Handle different return types from NMSBoxes
        if indices is None or len(indices) == 0:
            return []

        # Flatten indices if needed (NMSBoxes can return nested arrays)
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        elif isinstance(indices, tuple):
            indices = list(indices)

        results = []
        coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush",
        ]

        for idx in indices:
            idx = int(idx)  # Ensure integer index
            class_name = coco_classes[class_ids[idx]] if class_ids[idx] < len(coco_classes) else "unknown"
            bbox = (int(x1[idx]), int(y1[idx]), int(x2[idx] - x1[idx]), int(y2[idx] - y1[idx]))
            results.append((class_name, float(confidences[idx]), bbox))

        return results

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_seconds: float,
        previous_frame: np.ndarray | None = None,
    ) -> list[Detection]:
        """Run YOLO detection on frame."""
        self._ensure_initialized()

        start_time = time.perf_counter()

        # Preprocess
        blob, scale, padding = self._preprocess(frame)

        # Inference
        outputs = self._session.run(None, {self._input_name: blob})

        # Postprocess
        results = self._postprocess(outputs[0], scale, padding, frame.shape[:2])

        elapsed = time.perf_counter() - start_time
        self.processing_time += elapsed
        self.frames_processed += 1

        # Convert to Detection objects
        detections = []
        for class_name, confidence, bbox in results:
            event_type = COCO_TO_EVENT_TYPE.get(class_name, EventType.UNKNOWN)
            detections.append(
                Detection(
                    method=self.method,
                    event_type=event_type,
                    frame_number=frame_number,
                    timestamp_seconds=timestamp_seconds,
                    confidence=confidence,
                    bbox=bbox,
                    metadata={"class_name": class_name},
                )
            )

        return detections

    def reset(self) -> None:
        """Reset detector state."""
        pass  # YOLO is stateless


class MOG2Detector(BaseDetector):
    """MOG2 background subtraction detector."""

    method = DetectionMethod.MOG2

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = False,
        min_area_percent: float = 0.05,
    ):
        """Initialize MOG2 detector.

        Args:
            history: Length of history for background model
            var_threshold: Threshold for variance to classify as foreground
            detect_shadows: Whether to detect shadows
            min_area_percent: Minimum motion area as percentage of frame
        """
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.min_area_percent = min_area_percent

        self._bg_subtractor = None
        self._use_cuda = False
        self._cuda_stream = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._initialize()

    def _initialize(self) -> None:
        """Initialize background subtractor."""
        # Try CUDA MOG2 first
        try:
            self._bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows,
            )
            self._cuda_stream = cv2.cuda.Stream()
            self._use_cuda = True
            logger.info("MOG2 using CUDA acceleration")
        except (cv2.error, AttributeError):
            # Fall back to CPU
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows,
            )
            self._use_cuda = False
            logger.info("MOG2 using CPU (CUDA not available)")

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_seconds: float,
        previous_frame: np.ndarray | None = None,
    ) -> list[Detection]:
        """Run MOG2 background subtraction."""
        start_time = time.perf_counter()

        # Apply background subtraction
        if self._use_cuda:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            # OpenCV 4.10 CUDA MOG2: apply(input, learningRate, stream) -> mask
            gpu_fg = self._bg_subtractor.apply(gpu_frame, -1, self._cuda_stream)
            self._cuda_stream.waitForCompletion()
            fg_mask = gpu_fg.download()
        else:
            fg_mask = self._bg_subtractor.apply(frame)

        # Morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
        fg_mask = cv2.dilate(fg_mask, self._kernel, iterations=2)

        # Calculate motion percentage
        height, width = frame.shape[:2]
        total_pixels = height * width
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_percent = (motion_pixels / total_pixels) * 100

        elapsed = time.perf_counter() - start_time
        self.processing_time += elapsed
        self.frames_processed += 1

        # Check if motion exceeds threshold
        if motion_percent < self.min_area_percent:
            return []

        # Find contours for bounding boxes
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (total_pixels * self.min_area_percent / 100):
                continue

            x, y, w, h = cv2.boundingRect(contour)
            detections.append(
                Detection(
                    method=self.method,
                    event_type=EventType.MOTION,
                    frame_number=frame_number,
                    timestamp_seconds=timestamp_seconds,
                    confidence=min(1.0, motion_percent / 10),  # Scale to 0-1
                    bbox=(x, y, w, h),
                    metadata={"motion_percent": motion_percent, "area": area},
                )
            )

        return detections

    def reset(self) -> None:
        """Reset background model."""
        self._initialize()


class FrameDiffDetector(BaseDetector):
    """Simple frame differencing motion detector."""

    method = DetectionMethod.FRAME_DIFF

    def __init__(
        self,
        threshold: int = 30,
        min_area_percent: float = 0.05,
        blur_size: int = 21,
    ):
        """Initialize frame diff detector.

        Args:
            threshold: Pixel difference threshold (0-255)
            min_area_percent: Minimum motion area as percentage of frame
            blur_size: Gaussian blur kernel size
        """
        self.threshold = threshold
        self.min_area_percent = min_area_percent
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_seconds: float,
        previous_frame: np.ndarray | None = None,
    ) -> list[Detection]:
        """Run frame differencing."""
        start_time = time.perf_counter()

        if previous_frame is None:
            self.frames_processed += 1
            return []

        # Convert to grayscale
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        gray_current = cv2.GaussianBlur(gray_current, (self.blur_size, self.blur_size), 0)
        gray_previous = cv2.GaussianBlur(gray_previous, (self.blur_size, self.blur_size), 0)

        # Compute difference
        diff = cv2.absdiff(gray_current, gray_previous)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Calculate motion percentage
        height, width = frame.shape[:2]
        total_pixels = height * width
        motion_pixels = cv2.countNonZero(thresh)
        motion_percent = (motion_pixels / total_pixels) * 100

        # Morphological operations
        thresh = cv2.dilate(thresh, self._kernel, iterations=2)

        elapsed = time.perf_counter() - start_time
        self.processing_time += elapsed
        self.frames_processed += 1

        if motion_percent < self.min_area_percent:
            return []

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        min_area = total_pixels * self.min_area_percent / 100

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            detections.append(
                Detection(
                    method=self.method,
                    event_type=EventType.MOTION,
                    frame_number=frame_number,
                    timestamp_seconds=timestamp_seconds,
                    confidence=min(1.0, motion_percent / 10),
                    bbox=(x, y, w, h),
                    metadata={"motion_percent": motion_percent},
                )
            )

        return detections

    def reset(self) -> None:
        """Reset detector state."""
        pass  # Stateless


class EdgeDetector(BaseDetector):
    """Edge-based scene change detector."""

    method = DetectionMethod.EDGE_DETECTION

    def __init__(
        self,
        change_threshold: float = 0.1,
        canny_low: int = 50,
        canny_high: int = 150,
    ):
        """Initialize edge detector.

        Args:
            change_threshold: Minimum edge change percentage to trigger
            canny_low: Canny edge detection low threshold
            canny_high: Canny edge detection high threshold
        """
        self.change_threshold = change_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
        self._previous_edge_count: int | None = None

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_seconds: float,
        previous_frame: np.ndarray | None = None,
    ) -> list[Detection]:
        """Run edge-based change detection."""
        start_time = time.perf_counter()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # Count edge pixels
        edge_count = cv2.countNonZero(edges)

        elapsed = time.perf_counter() - start_time
        self.processing_time += elapsed
        self.frames_processed += 1

        # Compare with previous
        if self._previous_edge_count is None:
            self._previous_edge_count = edge_count
            return []

        # Calculate change
        if self._previous_edge_count > 0:
            change = abs(edge_count - self._previous_edge_count) / self._previous_edge_count
        else:
            change = 1.0 if edge_count > 0 else 0.0

        self._previous_edge_count = edge_count

        if change < self.change_threshold:
            return []

        return [
            Detection(
                method=self.method,
                event_type=EventType.MOTION,
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                confidence=min(1.0, change),
                bbox=None,  # Full-frame detection
                metadata={"edge_change": change, "edge_count": edge_count},
            )
        ]

    def reset(self) -> None:
        """Reset detector state."""
        self._previous_edge_count = None


class CorruptionDetector(BaseDetector):
    """Frame corruption detector for video decode errors."""

    method = DetectionMethod.CORRUPTION

    def __init__(
        self,
        streak_threshold: int = 50,
        repeated_col_threshold: float = 0.5,
        mean_diff_threshold: float = 0.5,
    ):
        """Initialize corruption detector.

        Args:
            streak_threshold: Horizontal streak detection threshold
            repeated_col_threshold: Fraction of columns with repeated pixels
            mean_diff_threshold: Maximum mean row diff for "repeated" column
        """
        self.streak_threshold = streak_threshold
        self.repeated_col_threshold = repeated_col_threshold
        self.mean_diff_threshold = mean_diff_threshold

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_seconds: float,
        previous_frame: np.ndarray | None = None,
    ) -> list[Detection]:
        """Detect frame corruption."""
        start_time = time.perf_counter()

        is_corrupt = self._check_corruption(frame)

        elapsed = time.perf_counter() - start_time
        self.processing_time += elapsed
        self.frames_processed += 1

        if not is_corrupt:
            return []

        return [
            Detection(
                method=self.method,
                event_type=EventType.CORRUPT_IMAGE,
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                confidence=1.0,
                bbox=None,
                metadata={"corruption_type": "vertical_banding"},
            )
        ]

    def _check_corruption(self, frame: np.ndarray) -> bool:
        """Check for vertical banding corruption."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        height, width = gray.shape[:2]

        # Focus on bottom 40% where corruption typically appears
        bottom_start = int(height * 0.6)
        bottom = gray[bottom_start:, :]

        # Compute row-to-row differences
        row_diffs = np.abs(np.diff(bottom.astype(float), axis=0))
        col_mean_diff = np.mean(row_diffs, axis=0)

        # Count columns with very small differences
        repeated_cols = np.sum(col_mean_diff < self.mean_diff_threshold)
        repeated_pct = repeated_cols / width

        overall_mean = np.mean(col_mean_diff)

        return repeated_pct > self.repeated_col_threshold or overall_mean < self.mean_diff_threshold

    def reset(self) -> None:
        """Reset detector state."""
        pass  # Stateless


class DetectorFactory:
    """Factory for creating detector instances."""

    @staticmethod
    def create_all(config: BenchmarkConfig) -> dict[DetectionMethod, BaseDetector]:
        """Create all enabled detectors from config.

        Args:
            config: Benchmark configuration

        Returns:
            Dictionary mapping DetectionMethod to detector instances
        """
        detectors: dict[DetectionMethod, BaseDetector] = {}

        for method in config.enabled_methods:
            detector = DetectorFactory.create(method, config)
            if detector is not None:
                detectors[method] = detector

        return detectors

    @staticmethod
    def create(method: DetectionMethod, config: BenchmarkConfig) -> BaseDetector | None:
        """Create a single detector.

        Args:
            method: Detection method to create
            config: Benchmark configuration

        Returns:
            Detector instance or None if creation fails
        """
        try:
            if method == DetectionMethod.YOLOV8N:
                return YOLODetector(
                    model_path=config.yolov8n_path,
                    method=method,
                    confidence_threshold=config.yolo_confidence,
                )

            elif method == DetectionMethod.YOLO11L:
                return YOLODetector(
                    model_path=config.yolo11l_path,
                    method=method,
                    confidence_threshold=config.yolo_confidence,
                )

            elif method == DetectionMethod.MOG2:
                return MOG2Detector(
                    history=config.mog2_history,
                    var_threshold=config.mog2_var_threshold,
                    detect_shadows=config.mog2_detect_shadows,
                    min_area_percent=config.mog2_min_area_percent,
                )

            elif method == DetectionMethod.FRAME_DIFF:
                return FrameDiffDetector(
                    threshold=config.frame_diff_threshold,
                    min_area_percent=config.frame_diff_min_area_percent,
                )

            elif method == DetectionMethod.EDGE_DETECTION:
                return EdgeDetector(
                    change_threshold=config.edge_change_threshold,
                )

            elif method == DetectionMethod.CORRUPTION:
                return CorruptionDetector(
                    streak_threshold=config.corruption_streak_threshold,
                )

            else:
                logger.warning(f"Unknown detection method: {method}")
                return None

        except Exception as e:
            logger.error(f"Failed to create detector {method}: {e}")
            return None
