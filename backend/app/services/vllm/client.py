"""VLLM client for vision language model inference."""

import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VLLMResponse:
    """Response from VLLM inference."""

    content: str
    raw_response: dict


class VLLMClient:
    """Client for communicating with Vision LLM servers.

    Supports OpenAI-compatible vision API format used by vLLM, Ollama,
    and other vision LLM servers.
    """

    def __init__(
        self,
        endpoint: str,
        timeout: int = 60,
        model: Optional[str] = None,
    ):
        """Initialize VLLM client.

        Args:
            endpoint: Base URL of the VLLM server (e.g., http://192.168.1.125:9001)
            timeout: Request timeout in seconds
            model: Model name to use (if server supports multiple models)
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string.

        Args:
            image: BGR numpy array (OpenCV format)

        Returns:
            Base64 encoded JPEG string
        """
        # Convert BGR to RGB
        rgb_image = image[:, :, ::-1]

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)

        # Encode as JPEG
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        # Base64 encode
        return base64.b64encode(buffer.read()).decode("utf-8")

    async def analyze_image(
        self,
        image: np.ndarray,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> VLLMResponse:
        """Send image to VLLM for analysis.

        Args:
            image: BGR numpy array (OpenCV format)
            prompt: User prompt describing what to analyze
            system_prompt: Optional system prompt for context

        Returns:
            VLLMResponse with model's analysis
        """
        client = await self._get_client()

        # Convert image to base64
        image_b64 = self._image_to_base64(image)

        # Build messages in OpenAI vision format
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        # User message with image
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        })

        # Build request payload
        payload = {
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.1,  # Low temperature for consistent analysis
        }

        if self.model:
            payload["model"] = self.model

        # Send request to chat completions endpoint
        url = f"{self.endpoint}/v1/chat/completions"

        try:
            logger.debug(f"Sending request to VLLM: {url}")
            response = await client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            logger.debug(f"VLLM response: {content[:200]}...")

            return VLLMResponse(content=content, raw_response=data)

        except httpx.TimeoutException:
            logger.error(f"VLLM request timed out after {self.timeout}s")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"VLLM request failed: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"VLLM request error: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if VLLM server is available.

        Returns:
            True if server responds, False otherwise
        """
        client = await self._get_client()

        try:
            # Try models endpoint first (standard OpenAI API)
            response = await client.get(
                f"{self.endpoint}/v1/models",
                timeout=5.0,
            )
            if response.status_code == 200:
                return True
        except Exception:
            pass

        try:
            # Fallback to health endpoint
            response = await client.get(
                f"{self.endpoint}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False
