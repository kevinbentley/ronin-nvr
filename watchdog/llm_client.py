"""LLM client for system analysis."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import httpx

from collectors import SystemState
from config import WatchdogConfig

logger = logging.getLogger(__name__)


@dataclass
class WatchdogAction:
    """Action recommended by the LLM."""

    action_type: str  # "stop_container", "kill_process", "restart_container"
    target: str | int  # Container name or PID
    reason: str


@dataclass
class AnalysisResult:
    """Result from LLM analysis."""

    analysis: str
    severity: str  # "normal", "warning", "critical"
    actions: list[WatchdogAction]
    raw_response: str
    tokens_used: int


SYSTEM_PROMPT = """You are a system watchdog analyzing a Linux server running RoninNVR (a video surveillance system).

Your job is to detect anomalies that could crash the server and recommend corrective actions.

## Known Issues
- The transcode-worker can spawn too many ffmpeg processes, exhausting memory
- Memory exhaustion has crashed the server before
- Normal operation: 3-6 ffmpeg processes for transcoding, 6-10 total with live detection

## Guidelines
- Memory > 90%: Consider stopping transcode workers
- More than 10-12 ffmpeg processes: Likely runaway, stop transcode container
- Single ffmpeg using > 3GB: May be stuck, consider killing it
- Be conservative - only recommend actions when clearly necessary
- Stopping containers is safer than killing processes

## Response Format
You MUST respond with valid JSON only (no markdown, no explanation outside JSON):
{
  "analysis": "Brief explanation of what you observe",
  "severity": "normal|warning|critical",
  "actions": [
    {"type": "stop_container", "target": "container-name", "reason": "why"},
    {"type": "kill_process", "target": 12345, "reason": "why"}
  ]
}

Valid action types: stop_container, kill_process, restart_container
For stop_container: target should include the service pattern (e.g., "transcode-worker")
For kill_process: target should be the PID as an integer

If everything looks normal, return:
{"analysis": "System operating normally", "severity": "normal", "actions": []}
"""


class LLMClient:
    """Client for LLM-based system analysis."""

    def __init__(self, config: WatchdogConfig) -> None:
        """Initialize the LLM client."""
        self.config = config
        self.endpoint = config.llm_endpoint.rstrip("/")
        self.model = config.llm_model
        self.timeout = config.llm_timeout
        self.max_tokens = config.llm_max_tokens

    async def analyze(self, state: SystemState) -> Optional[AnalysisResult]:
        """Analyze system state and return recommended actions."""
        prompt = self._build_prompt(state)

        # Log prompt in debug mode
        logger.debug("=" * 60)
        logger.debug("LLM PROMPT (System):")
        logger.debug("=" * 60)
        for line in SYSTEM_PROMPT.split("\n"):
            logger.debug(f"  {line}")
        logger.debug("=" * 60)
        logger.debug("LLM PROMPT (User):")
        logger.debug("=" * 60)
        for line in prompt.split("\n"):
            logger.debug(f"  {line}")
        logger.debug("=" * 60)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.endpoint}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": self.max_tokens,
                        "temperature": 0.1,  # Low temperature for consistent analysis
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Extract response content
                content = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)

                # Log completion in debug mode
                logger.debug("=" * 60)
                logger.debug(f"LLM COMPLETION ({tokens_used} tokens):")
                logger.debug("=" * 60)
                for line in content.split("\n"):
                    logger.debug(f"  {line}")
                logger.debug("=" * 60)

                # Parse the response
                return self._parse_response(content, tokens_used)

        except httpx.TimeoutException:
            logger.error(f"LLM request timed out after {self.timeout}s")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM request failed: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"LLM request error: {e}")
            return None

    def _build_prompt(self, state: SystemState) -> str:
        """Build the analysis prompt from system state."""
        return f"""Analyze this system state and determine if any action is needed.

{state.format_for_llm()}

Remember to respond with JSON only. What is your analysis?"""

    def _parse_response(self, content: str, tokens_used: int) -> Optional[AnalysisResult]:
        """Parse LLM response into structured result."""
        try:
            # Try to extract JSON from the response
            # Handle case where LLM wraps JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            # Also try to find raw JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)

            data = json.loads(content)

            # Parse actions
            actions = []
            for action_data in data.get("actions", []):
                action = WatchdogAction(
                    action_type=action_data.get("type", "unknown"),
                    target=action_data.get("target", ""),
                    reason=action_data.get("reason", "No reason provided"),
                )
                actions.append(action)

            return AnalysisResult(
                analysis=data.get("analysis", "No analysis provided"),
                severity=data.get("severity", "normal"),
                actions=actions,
                raw_response=content,
                tokens_used=tokens_used,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {content}")
            # Return a fallback result
            return AnalysisResult(
                analysis=f"Failed to parse response: {content[:200]}",
                severity="warning",
                actions=[],
                raw_response=content,
                tokens_used=tokens_used,
            )

    async def test_connection(self) -> bool:
        """Test if the LLM endpoint is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.endpoint}/models")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"LLM endpoint test failed: {e}")
            return False
