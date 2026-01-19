#!/usr/bin/env python3
"""Quick test script for VLLM integration."""

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.services.vllm.client import VLLMClient
from app.services.vllm.mosaic import create_mosaic, add_frame_numbers
from app.services.vllm.characterization import ActivityCharacterizer


async def test_health_check():
    """Test VLLM health check."""
    print("Testing VLLM health check...")
    client = VLLMClient(endpoint="http://192.168.1.125:9001", timeout=30)

    try:
        healthy = await client.health_check()
        print(f"  Health check: {'PASS' if healthy else 'FAIL'}")
        return healthy
    finally:
        await client.close()


async def test_mosaic_creation():
    """Test mosaic creation with synthetic frames."""
    print("\nTesting mosaic creation...")

    # Create 4 synthetic frames with different colors
    frames = []
    colors = [
        (255, 0, 0),    # Blue (BGR)
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
    ]

    for i, color in enumerate(colors):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = color
        frames.append(frame)

    # Create labels
    timestamps = [0, 1000, 2000, 3000]
    labels = add_frame_numbers(frames, timestamps)

    # Create mosaic
    mosaic = create_mosaic(frames, grid_size=(2, 2), labels=labels)

    print(f"  Mosaic shape: {mosaic.shape}")
    print(f"  Labels: {labels}")

    # Save test mosaic
    import cv2
    cv2.imwrite("/tmp/test_mosaic.jpg", mosaic)
    print("  Saved test mosaic to /tmp/test_mosaic.jpg")

    return True


async def test_simple_analysis():
    """Test a simple image analysis with the VLLM."""
    print("\nTesting simple image analysis...")

    client = VLLMClient(endpoint="http://192.168.1.125:9001", timeout=60)

    try:
        # Create a simple test image
        import cv2
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (128, 128, 128)  # Gray background

        # Add some text
        cv2.putText(
            test_image,
            "TEST IMAGE",
            (200, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2,
        )

        # Send to VLLM
        response = await client.analyze_image(
            image=test_image,
            prompt="What do you see in this image? Describe it briefly.",
        )

        print(f"  Response: {response.content[:200]}...")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False
    finally:
        await client.close()


async def test_activity_characterization():
    """Test full activity characterization pipeline."""
    print("\nTesting activity characterization...")

    client = VLLMClient(endpoint="http://192.168.1.125:9001", timeout=60)
    characterizer = ActivityCharacterizer(client)

    try:
        # Create 4 synthetic frames simulating a sequence
        frames = []
        for i in range(4):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (64, 64, 64)  # Dark gray background

            # Add frame number
            import cv2
            cv2.putText(
                frame,
                f"Frame {i+1}",
                (250, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                2,
            )
            frames.append(frame)

        timestamps = [0, 1000, 2000, 3000]

        analysis = await characterizer.analyze_frames(
            frames=frames,
            timestamps_ms=timestamps,
            scene_description="Test scene with gray background",
            detected_class="person",
        )

        print(f"  Concern Level: {analysis.concern_level.value}")
        print(f"  Activity Type: {analysis.activity_type}")
        print(f"  Description: {analysis.description[:150]}...")

        return True

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await client.close()


async def main():
    """Run all tests."""
    print("=" * 60)
    print("VLLM Integration Tests")
    print("=" * 60)

    results = []

    results.append(("Health Check", await test_health_check()))
    results.append(("Mosaic Creation", await test_mosaic_creation()))
    results.append(("Simple Analysis", await test_simple_analysis()))
    results.append(("Activity Characterization", await test_activity_characterization()))

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
