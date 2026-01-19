"""Activity characterization service using Vision LLM."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from app.services.vllm.client import VLLMClient
from app.services.vllm.mosaic import add_frame_numbers, create_mosaic

logger = logging.getLogger(__name__)


class ConcernLevel(str, Enum):
    """Concern level classification from VLLM analysis."""

    NONE = "none"  # No activity / false positive
    LOW = "low"  # Normal expected activity
    MEDIUM = "medium"  # Unusual but not threatening
    HIGH = "high"  # Suspicious behavior
    EMERGENCY = "emergency"  # Immediate threat

    @classmethod
    def from_string(cls, value: str) -> "ConcernLevel":
        """Parse concern level from string, case-insensitive."""
        value_lower = value.lower().strip()
        for level in cls:
            if level.value == value_lower:
                return level
        # Default to MEDIUM if unknown
        logger.warning(f"Unknown concern level '{value}', defaulting to MEDIUM")
        return cls.MEDIUM


@dataclass
class ActivityAnalysis:
    """Result of activity characterization analysis."""

    description: str
    concern_level: ConcernLevel
    activity_type: Optional[str]
    raw_response: str
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "description": self.description,
            "concern_level": self.concern_level.value,
            "activity_type": self.activity_type,
            "confidence": self.confidence,
        }


# System prompt for the VLLM
SYSTEM_PROMPT = """You are a security camera analysis system. Your job is to analyze
security camera footage and characterize any detected activity.

You will be shown a 2x2 grid of frames captured 1 second apart (labeled T+0.0s, T+1.0s,
T+2.0s, T+3.0s from top-left to bottom-right). Analyze the temporal sequence to
understand what is happening.

For each analysis, provide:
1. A brief description of what you observe
2. The type of activity (e.g., "delivery", "pedestrian", "vehicle", "animal", "none")
3. A concern level: NONE, LOW, MEDIUM, HIGH, or EMERGENCY

Concern Level Guidelines:
- NONE: No person/activity detected, likely a false positive detection
- LOW: Normal expected activity (delivery person, mail carrier, resident, neighbor)
- MEDIUM: Unusual but not threatening (unfamiliar person walking by, late night activity)
- HIGH: Suspicious behavior (peering into windows, checking door handles, loitering)
- EMERGENCY: Immediate threat (break-in attempt, assault, vandalism in progress)

Respond in this exact format:
DESCRIPTION: <your description>
ACTIVITY_TYPE: <type>
CONCERN_LEVEL: <NONE|LOW|MEDIUM|HIGH|EMERGENCY>
"""


class ActivityCharacterizer:
    """Characterizes detected activities using Vision LLM analysis."""

    def __init__(
        self,
        client: VLLMClient,
        frame_count: int = 4,
        frame_interval_ms: int = 1000,
    ):
        """Initialize activity characterizer.

        Args:
            client: VLLMClient instance for inference
            frame_count: Number of frames to capture (default 4 for 2x2 grid)
            frame_interval_ms: Milliseconds between frames
        """
        self.client = client
        self.frame_count = frame_count
        self.frame_interval_ms = frame_interval_ms

    def _build_prompt(
        self,
        scene_description: Optional[str] = None,
        detected_class: Optional[str] = None,
    ) -> str:
        """Build the analysis prompt with context.

        Args:
            scene_description: Baseline description of the camera's view
            detected_class: Class that triggered the detection (e.g., "person")

        Returns:
            Formatted prompt string
        """
        parts = []

        if scene_description:
            parts.append(f"Scene Context: {scene_description}")

        if detected_class:
            parts.append(
                f"A '{detected_class}' was detected by the object detection system."
            )

        parts.append(
            "Analyze these 4 consecutive frames and describe what activity is occurring. "
            "Determine the concern level based on the behavior observed."
        )

        return "\n\n".join(parts)

    def _parse_response(self, response: str) -> ActivityAnalysis:
        """Parse VLLM response into structured ActivityAnalysis.

        Args:
            response: Raw text response from VLLM

        Returns:
            Parsed ActivityAnalysis object
        """
        description = ""
        activity_type = None
        concern_level = ConcernLevel.MEDIUM

        # Parse DESCRIPTION
        desc_match = re.search(
            r"DESCRIPTION:\s*(.+?)(?=ACTIVITY_TYPE:|CONCERN_LEVEL:|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if desc_match:
            description = desc_match.group(1).strip()

        # Parse ACTIVITY_TYPE
        type_match = re.search(
            r"ACTIVITY_TYPE:\s*(\S+)",
            response,
            re.IGNORECASE,
        )
        if type_match:
            activity_type = type_match.group(1).strip()

        # Parse CONCERN_LEVEL
        level_match = re.search(
            r"CONCERN_LEVEL:\s*(\S+)",
            response,
            re.IGNORECASE,
        )
        if level_match:
            concern_level = ConcernLevel.from_string(level_match.group(1))

        # Fallback: use full response as description if parsing failed
        if not description:
            description = response.strip()

        return ActivityAnalysis(
            description=description,
            concern_level=concern_level,
            activity_type=activity_type,
            raw_response=response,
        )

    async def analyze_frames(
        self,
        frames: list[np.ndarray],
        timestamps_ms: Optional[list[float]] = None,
        scene_description: Optional[str] = None,
        detected_class: Optional[str] = None,
    ) -> ActivityAnalysis:
        """Analyze a sequence of frames for activity characterization.

        Args:
            frames: List of BGR numpy arrays (should be 4 frames for 2x2 grid)
            timestamps_ms: Optional timestamps for each frame
            scene_description: Baseline description of the camera's view
            detected_class: Class that triggered the detection

        Returns:
            ActivityAnalysis with description and concern level
        """
        if not frames:
            raise ValueError("No frames provided for analysis")

        # Generate labels for frames
        labels = add_frame_numbers(frames, timestamps_ms)

        # Create 2x2 mosaic
        mosaic = create_mosaic(
            frames,
            grid_size=(2, 2),
            labels=labels,
            target_size=(1280, 720),  # Reasonable size for VLLM input
        )

        # Build prompt
        prompt = self._build_prompt(scene_description, detected_class)

        # Send to VLLM
        logger.info(f"Sending mosaic to VLLM for analysis (class={detected_class})")
        response = await self.client.analyze_image(
            image=mosaic,
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        # Parse response
        analysis = self._parse_response(response.content)

        logger.info(
            f"Activity analysis complete: "
            f"level={analysis.concern_level.value}, "
            f"type={analysis.activity_type}, "
            f"desc={analysis.description[:100]}..."
        )

        return analysis

    async def analyze_from_mosaic(
        self,
        mosaic: np.ndarray,
        scene_description: Optional[str] = None,
        detected_class: Optional[str] = None,
    ) -> ActivityAnalysis:
        """Analyze a pre-built mosaic image.

        Args:
            mosaic: Pre-built mosaic image as BGR numpy array
            scene_description: Baseline description of the camera's view
            detected_class: Class that triggered the detection

        Returns:
            ActivityAnalysis with description and concern level
        """
        # Build prompt
        prompt = self._build_prompt(scene_description, detected_class)

        # Send to VLLM
        logger.info(f"Sending pre-built mosaic to VLLM for analysis")
        response = await self.client.analyze_image(
            image=mosaic,
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        # Parse response
        return self._parse_response(response.content)
