#!/usr/bin/env python3
"""Test FSM behavior with various detection patterns.

Validates that the state machine correctly handles:
1. Flickering detections (intermittent false positives)
2. Stationary objects (light fixtures misclassified as people)
3. Real arrivals and departures
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.object_fsm import (
    ObjectStateMachine,
    ObjectState,
    EventType,
)


@dataclass
class MockTrack:
    """Mock track for testing."""
    track_id: int
    class_id: int = 0
    class_name: str = "car"
    x: float = 0.5
    y: float = 0.5
    width: float = 0.1
    height: float = 0.1
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    confidence: float = 0.8


def test_flickering_pattern():
    """Test: 0-0-0-1-1-0-1-0-0-1-0-0-0-0-0

    Intermittent detections should NOT generate arrival/departure.
    """
    print("=" * 60)
    print("TEST: Flickering Pattern")
    print("Pattern: 0-0-0-1-1-0-1-0-0-1-0-0-0-0-0")
    print("=" * 60)

    # Use shorter timeouts for testing
    fsm = ObjectStateMachine(
        validation_frames=5,  # Need 5 detections to confirm
        lost_seconds=3.0,     # 3 seconds without detection = departed
        fps=1.0,              # 1 FPS detection
    )

    pattern = [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    arrivals = []
    departures = []

    for i, detected in enumerate(pattern):
        tracks = [MockTrack(track_id=1)] if detected else []
        events = fsm.update(tracks)

        for event in events:
            if event.event_type == EventType.ARRIVAL:
                arrivals.append(i)
                print(f"  Frame {i}: ARRIVAL")
            elif event.event_type == EventType.DEPARTURE:
                departures.append(i)
                print(f"  Frame {i}: DEPARTURE")

        # Simulate 1 second between frames
        time.sleep(0.01)  # Shortened for test speed

    # Wait for lost_seconds to trigger departure check
    time.sleep(0.1)
    events = fsm.update([])
    for event in events:
        if event.event_type == EventType.DEPARTURE:
            departures.append(len(pattern))

    print(f"\nResults:")
    print(f"  Arrivals: {len(arrivals)} (expected: 0)")
    print(f"  Departures: {len(departures)} (expected: 0)")

    success = len(arrivals) == 0 and len(departures) == 0
    print(f"  Status: {'PASS' if success else 'FAIL'}")
    return success


def test_real_arrival_departure():
    """Test: Real object that arrives, stays, and leaves.

    Pattern: 0-0-1-1-1-1-1-1-1-1-1-1-0-0-0-0
    Should generate exactly 1 arrival and 1 departure.
    """
    print("\n" + "=" * 60)
    print("TEST: Real Arrival/Departure")
    print("Pattern: 0-0-1-1-1-1-1-1-1-1-1-1-0-0-0-0")
    print("=" * 60)

    fsm = ObjectStateMachine(
        validation_frames=5,
        lost_seconds=2.0,
        fps=1.0,
    )

    # Moving object (has velocity)
    track = MockTrack(track_id=1, velocity_x=0.01, velocity_y=0.01)

    pattern = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    arrivals = []
    departures = []

    for i, detected in enumerate(pattern):
        tracks = [track] if detected else []
        events = fsm.update(tracks)

        for event in events:
            if event.event_type == EventType.ARRIVAL:
                arrivals.append(i)
                print(f"  Frame {i}: ARRIVAL")
            elif event.event_type == EventType.DEPARTURE:
                departures.append(i)
                print(f"  Frame {i}: DEPARTURE")

        time.sleep(0.01)

    # Wait for departure detection
    for _ in range(5):
        time.sleep(0.5)
        events = fsm.update([])
        for event in events:
            if event.event_type == EventType.DEPARTURE:
                departures.append(len(pattern))
                print(f"  After pattern: DEPARTURE")

    print(f"\nResults:")
    print(f"  Arrivals: {len(arrivals)} (expected: 1)")
    print(f"  Departures: {len(departures)} (expected: 1)")

    success = len(arrivals) == 1 and len(departures) == 1
    print(f"  Status: {'PASS' if success else 'FAIL'}")
    return success


def test_stationary_false_positive():
    """Test: Stationary object (light fixture as person).

    Continuously detected but never moves. Should eventually become
    PARKED and not generate departure when detection stops.
    """
    print("\n" + "=" * 60)
    print("TEST: Stationary False Positive (Light Fixture)")
    print("Continuously detected, no movement, then stops")
    print("=" * 60)

    fsm = ObjectStateMachine(
        validation_frames=5,
        lost_seconds=2.0,
        stationary_seconds=2.0,  # Become stationary after 2s
        parked_seconds=3.0,      # Become parked after 3s more
        fps=1.0,
    )

    # Stationary object (no velocity)
    track = MockTrack(track_id=1, class_name="person", velocity_x=0.0, velocity_y=0.0)

    arrivals = []
    departures = []
    state_changes = []

    # Phase 1: Continuous detection for validation (10 frames)
    print("\n  Phase 1: Validation period")
    for i in range(10):
        events = fsm.update([track])
        for event in events:
            if event.event_type == EventType.ARRIVAL:
                arrivals.append(i)
                print(f"    Frame {i}: ARRIVAL")
            elif event.event_type == EventType.STATE_CHANGE:
                state_changes.append((i, event.old_state, event.new_state))
                print(f"    Frame {i}: {event.old_state.value} -> {event.new_state.value}")
        time.sleep(0.3)  # 0.3s per frame to speed up test

    # Phase 2: Continue detection, should become STATIONARY then PARKED
    print("\n  Phase 2: Becoming stationary/parked")
    for i in range(10, 30):
        events = fsm.update([track])
        for event in events:
            if event.event_type == EventType.STATE_CHANGE:
                state_changes.append((i, event.old_state, event.new_state))
                print(f"    Frame {i}: {event.old_state.value} -> {event.new_state.value}")
        time.sleep(0.3)

    # Check current state
    lifecycle = fsm.get_lifecycle(1)
    print(f"\n  Current state: {lifecycle.state.value if lifecycle else 'None'}")

    # Phase 3: Stop detection - should NOT generate departure if parked
    print("\n  Phase 3: Detection stops")
    for i in range(30, 40):
        events = fsm.update([])
        for event in events:
            if event.event_type == EventType.DEPARTURE:
                departures.append(i)
                print(f"    Frame {i}: DEPARTURE (unexpected!)")
        time.sleep(0.3)

    print(f"\nResults:")
    print(f"  Arrivals: {len(arrivals)} (expected: 1)")
    print(f"  Departures: {len(departures)} (expected: 0 - parked objects don't depart)")

    # Success if we got arrival but no departure for parked object
    success = len(arrivals) == 1 and len(departures) == 0
    print(f"  Status: {'PASS' if success else 'FAIL'}")

    # Note: After lost_seconds, object moves to _departed dict with state=DEPARTED
    # but crucially, no DEPARTURE *event* was generated - which is correct!
    if lifecycle:
        print(f"  Note: Object is now in departed dict (state={lifecycle.state.value})")
        print(f"        But no DEPARTURE event was generated - this is correct!")

    return success


def test_arrival_then_park():
    """Test: Object arrives, parks, then leaves.

    Should get arrival when it arrives, but not departure when
    it's been parked and detection stops.
    """
    print("\n" + "=" * 60)
    print("TEST: Arrival -> Park -> Detection Stops")
    print("=" * 60)

    fsm = ObjectStateMachine(
        validation_frames=3,
        lost_seconds=1.0,
        stationary_seconds=1.0,
        parked_seconds=2.0,
        fps=1.0,
    )

    arrivals = []
    departures = []

    # Phase 1: Moving object arrives
    print("\n  Phase 1: Object arrives (moving)")
    track = MockTrack(track_id=1, class_name="car", velocity_x=0.01, velocity_y=0.01)
    for i in range(5):
        events = fsm.update([track])
        for event in events:
            if event.event_type == EventType.ARRIVAL:
                arrivals.append(i)
                print(f"    Frame {i}: ARRIVAL")
        time.sleep(0.3)

    # Phase 2: Object stops (stationary)
    print("\n  Phase 2: Object stops moving")
    track.velocity_x = 0.0
    track.velocity_y = 0.0
    for i in range(5, 20):
        events = fsm.update([track])
        for event in events:
            if event.event_type == EventType.STATE_CHANGE:
                print(f"    Frame {i}: State -> {event.new_state.value}")
        time.sleep(0.3)

    # Phase 3: Detection stops
    print("\n  Phase 3: Detection stops (parked car, no departure expected)")
    for i in range(20, 30):
        events = fsm.update([])
        for event in events:
            if event.event_type == EventType.DEPARTURE:
                departures.append(i)
                print(f"    Frame {i}: DEPARTURE")
        time.sleep(0.3)

    print(f"\nResults:")
    print(f"  Arrivals: {len(arrivals)} (expected: 1)")
    print(f"  Departures: {len(departures)} (expected: 0 for parked object)")

    success = len(arrivals) == 1 and len(departures) == 0
    print(f"  Status: {'PASS' if success else 'FAIL'}")
    return success


def main():
    """Run all FSM tests."""
    print("\n" + "=" * 60)
    print("OBJECT FSM VALIDATION TESTS")
    print("=" * 60)

    results = []

    results.append(("Flickering Pattern", test_flickering_pattern()))
    results.append(("Real Arrival/Departure", test_real_arrival_departure()))
    results.append(("Stationary False Positive", test_stationary_false_positive()))
    results.append(("Arrival then Park", test_arrival_then_park()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
