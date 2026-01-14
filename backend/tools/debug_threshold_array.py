#!/usr/bin/env python3
"""Debug per-class threshold array."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.tensorrt_inference import TensorRTDetector, COCO_CLASSES

# Test threshold array creation
print("COCO class 'person' index:", COCO_CLASSES.index("person"))
print("COCO class 'truck' index:", COCO_CLASSES.index("truck"))
print()

# Create detector with per-class thresholds
detector = TensorRTDetector(
    model_path="/opt3/ronin/ml_models/yolov8n_dynamic.onnx",
    confidence_threshold=0.65,
    class_thresholds={"person": 0.45, "dog": 0.45, "cat": 0.45},
    warmup_iterations=0,
)

print("Threshold array shape:", detector._class_threshold_array.shape)
print("Threshold for person (idx 0):", detector._class_threshold_array[0])
print("Threshold for truck (idx 7):", detector._class_threshold_array[7])
print("Threshold for dog (idx 16):", detector._class_threshold_array[16])
print("Threshold for cat (idx 15):", detector._class_threshold_array[15])
print()

# Show all unique thresholds
import numpy as np
unique_thresholds = np.unique(detector._class_threshold_array)
print("Unique thresholds:", unique_thresholds)

# Show which classes have each threshold
for thresh in unique_thresholds:
    indices = np.where(detector._class_threshold_array == thresh)[0]
    classes = [COCO_CLASSES[i] for i in indices[:5]]
    if len(indices) > 5:
        print(f"  {thresh}: {classes} ... ({len(indices)} total)")
    else:
        print(f"  {thresh}: {classes}")
