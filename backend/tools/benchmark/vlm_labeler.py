"""VLM-based ground truth labeling using OpenAI-compatible API."""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Callable

import httpx

from .config import BenchmarkConfig
from .models import CandidateEvent, EventType, VLMLabel

logger = logging.getLogger(__name__)

# System prompt for VLM labeling
VLM_SYSTEM_PROMPT = """You are an expert image analyst for a security camera system. Your task is to accurately identify objects of interest in security camera footage.

Analyze the image and report what you see. Focus on:
1. People - any humans visible in the image
2. Vehicles - cars, trucks, motorcycles, bicycles, etc.
3. Animals - pets, wildlife, any animals
4. Image quality issues - corruption, streaking, banding, or other artifacts

Be conservative in your analysis:
- Only report objects you can clearly identify
- If an object is ambiguous or partially visible, note the uncertainty
- Consider the context of a typical outdoor security camera view

Respond in JSON format with the following structure:
{
    "objects_detected": ["list", "of", "objects"],
    "has_person": true/false,
    "has_vehicle": true/false,
    "has_animal": true/false,
    "is_corrupt": true/false,
    "confidence": "high" | "medium" | "low",
    "notes": "any additional observations"
}"""


class VLMLabeler:
    """Labels candidate events using a Vision Language Model."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize VLM labeler.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.endpoint = config.vlm_endpoint
        self.model = config.vlm_model
        self.timeout = config.vlm_timeout
        self.max_retries = config.vlm_max_retries
        self.retry_delay = config.vlm_retry_delay

    def _encode_image(self, image_path: Path) -> str:
        """Encode image as base64 string.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_request(self, image_path: Path) -> dict:
        """Build OpenAI-compatible API request.

        Args:
            image_path: Path to image to analyze

        Returns:
            Request payload dictionary
        """
        image_data = self._encode_image(image_path)

        # Determine image media type
        suffix = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/jpeg")

        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": VLM_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "What objects of interest do you see in this security camera image?",
                        },
                    ],
                },
            ],
            "max_tokens": 500,
            "temperature": 0.1,  # Low temperature for consistent analysis
        }

    def _parse_response(self, response_text: str) -> dict:
        """Parse VLM response text to extract structured data.

        Args:
            response_text: Raw response from VLM

        Returns:
            Parsed response dictionary
        """
        # Try to extract JSON from response
        try:
            # Look for JSON block
            if "```json" in response_text:
                json_start = response_text.index("```json") + 7
                json_end = response_text.index("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.index("{")
                json_end = response_text.rindex("}") + 1
                json_str = response_text[json_start:json_end]
            else:
                # Fall back to text analysis
                return self._parse_text_response(response_text)

            return json.loads(json_str)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON from VLM response: {e}")
            return self._parse_text_response(response_text)

    def _parse_text_response(self, text: str) -> dict:
        """Parse text response when JSON parsing fails.

        Args:
            text: Raw text response

        Returns:
            Inferred response dictionary
        """
        text_lower = text.lower()

        return {
            "objects_detected": [],
            "has_person": any(word in text_lower for word in ["person", "people", "human", "man", "woman", "child"]),
            "has_vehicle": any(word in text_lower for word in ["car", "truck", "vehicle", "bike", "motorcycle"]),
            "has_animal": any(word in text_lower for word in ["dog", "cat", "animal", "bird", "deer"]),
            "is_corrupt": any(word in text_lower for word in ["corrupt", "artifact", "banding", "streak", "error"]),
            "confidence": "low",
            "notes": "Parsed from text response",
        }

    def label_event(self, event: CandidateEvent) -> VLMLabel:
        """Label a single candidate event using VLM.

        Args:
            event: Candidate event with frame_path set

        Returns:
            VLM label for the event
        """
        if event.frame_path is None or not event.frame_path.exists():
            logger.error(f"No frame image for event at frame {event.frame_number}")
            event.vlm_label = VLMLabel.ERROR
            event.vlm_response = "No frame image available"
            return VLMLabel.ERROR

        request_payload = self._build_request(event.frame_path)

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self.endpoint,
                        json=request_payload,
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()

                data = response.json()

                # Extract response content
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                elif "response" in data:
                    content = data["response"]
                else:
                    content = str(data)

                event.vlm_response = content

                # Parse the response
                parsed = self._parse_response(content)

                # Extract detected objects
                event.vlm_detected_objects = parsed.get("objects_detected", [])

                # Determine label based on detection results
                label = self._determine_label(event, parsed)
                event.vlm_label = label

                # Log detailed VLM response
                logger.info(f"VLM frame {event.frame_number}: {label.value}")
                logger.info(f"  VLM says: person={parsed.get('has_person')}, "
                          f"vehicle={parsed.get('has_vehicle')}, "
                          f"animal={parsed.get('has_animal')}")
                logger.info(f"  Detectors: {[d.method.value for d in event.detections]} "
                          f"-> {[d.event_type.value for d in event.detections]}")
                if parsed.get('objects_detected'):
                    logger.info(f"  Objects: {parsed.get('objects_detected')}")
                # Full response at debug level
                logger.debug(f"  Raw response: {content[:300]}..."
                           if len(content) > 300 else f"  Raw response: {content}")

                return label

            except httpx.TimeoutException:
                logger.warning(f"VLM request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            except httpx.HTTPStatusError as e:
                logger.error(f"VLM HTTP error: {e.response.status_code}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"VLM request failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        # All retries exhausted
        event.vlm_label = VLMLabel.ERROR
        event.vlm_response = "Failed after all retries"
        return VLMLabel.ERROR

    def _determine_label(
        self,
        event: CandidateEvent,
        vlm_result: dict,
    ) -> VLMLabel:
        """Determine the ground truth label based on VLM result and detections.

        Args:
            event: Candidate event
            vlm_result: Parsed VLM response

        Returns:
            Ground truth label
        """
        # Get what the detectors claimed to find
        detected_types = event.event_types_detected

        # Get what VLM actually found
        vlm_has_person = vlm_result.get("has_person", False)
        vlm_has_vehicle = vlm_result.get("has_vehicle", False)
        vlm_has_animal = vlm_result.get("has_animal", False)
        vlm_is_corrupt = vlm_result.get("is_corrupt", False)
        confidence = vlm_result.get("confidence", "medium")

        # Handle corruption detection
        if EventType.CORRUPT_IMAGE in detected_types:
            if vlm_is_corrupt:
                return VLMLabel.TRUE_POSITIVE
            else:
                return VLMLabel.FALSE_POSITIVE

        # Check if any claimed detection type was validated
        validated = False

        if EventType.PERSON in detected_types and vlm_has_person:
            validated = True
        if EventType.VEHICLE in detected_types and vlm_has_vehicle:
            validated = True
        if EventType.ANIMAL in detected_types and vlm_has_animal:
            validated = True

        # Motion detections are validated by any interesting object
        if EventType.MOTION in detected_types:
            if vlm_has_person or vlm_has_vehicle or vlm_has_animal:
                validated = True

        if confidence == "low":
            return VLMLabel.UNCERTAIN

        return VLMLabel.TRUE_POSITIVE if validated else VLMLabel.FALSE_POSITIVE

    def label_events(
        self,
        events: list[CandidateEvent],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Label multiple candidate events.

        Args:
            events: List of candidate events
            progress_callback: Optional callback(current, total) for progress updates
        """
        total = len(events)
        logger.info(f"Labeling {total} events with VLM...")

        for i, event in enumerate(events):
            if event.frame_path is None:
                logger.warning(f"Skipping event {i+1}/{total}: no frame image")
                continue

            logger.info(f"--- VLM Event {i+1}/{total} (video: {event.video.path.name}) ---")
            label = self.label_event(event)

            if progress_callback:
                progress_callback(i + 1, total)

        # Summary
        labels = [e.vlm_label for e in events if e.vlm_label is not None]
        tp = sum(1 for l in labels if l == VLMLabel.TRUE_POSITIVE)
        fp = sum(1 for l in labels if l == VLMLabel.FALSE_POSITIVE)
        uncertain = sum(1 for l in labels if l == VLMLabel.UNCERTAIN)
        errors = sum(1 for l in labels if l == VLMLabel.ERROR)

        logger.info(f"VLM labeling complete: {tp} TP, {fp} FP, {uncertain} uncertain, {errors} errors")


def test_vlm_connection(config: BenchmarkConfig) -> bool:
    """Test VLM endpoint connectivity.

    Args:
        config: Benchmark configuration

    Returns:
        True if connection successful
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            # Try a simple request
            response = client.get(
                config.vlm_endpoint.replace("/chat/completions", "/models"),
            )
            if response.status_code in (200, 404):
                # 404 is OK - means server is running but endpoint differs
                logger.info(f"VLM endpoint reachable: {config.vlm_endpoint}")
                return True
    except Exception as e:
        logger.error(f"VLM connection test failed: {e}")

    return False
