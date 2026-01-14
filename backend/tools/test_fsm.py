#!/usr/bin/env python3
"""Test Object Finite State Machine.

Usage:
    source /opt/venv/bin/activate
    cd /workspace/ronin-nvr/backend
    python tools/test_fsm.py
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml.object_fsm import (
    ObjectStateMachine, ObjectState, EventType, ObjectEvent
)


@dataclass
class MockTrack:
    """Mock tracked object for testing."""
    track_id: int
    class_id: int
    class_name: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0


def test_arrival_event():
    """Test that arrival events are generated."""
    print("=" * 60)
    print("Test: Arrival Event Generation")
    print("=" * 60)

    fsm = ObjectStateMachine(
        validation_frames=3,
        velocity_threshold=0.002,
        fps=30.0,
    )

    # Simulate new object appearing
    track = MockTrack(
        track_id=1, class_id=2, class_name="car",
        x=0.3, y=0.4, width=0.1, height=0.08,
        confidence=0.9, velocity_x=0.01, velocity_y=0.0
    )

    # Frame 1-2: tentative
    for i in range(2):
        events = fsm.update([track])
        print(f"Frame {i+1}: {len(events)} events, state: {fsm.get_lifecycle(1).state.value}")

    # Frame 3: should become ACTIVE and generate ARRIVAL
    events = fsm.update([track])
    print(f"Frame 3: {len(events)} events")

    arrival_events = [e for e in events if e.event_type == EventType.ARRIVAL]
    assert len(arrival_events) == 1, "Should have one arrival event"
    print(f"  ARRIVAL event: {arrival_events[0]}")

    lifecycle = fsm.get_lifecycle(1)
    assert lifecycle.state == ObjectState.ACTIVE
    print(f"  Final state: {lifecycle.state.value}")

    print("\n[PASS] Arrival event test")


def test_stationary_transition():
    """Test transition to stationary state."""
    print("=" * 60)
    print("Test: Stationary Transition")
    print("=" * 60)

    fsm = ObjectStateMachine(
        validation_frames=2,
        velocity_threshold=0.002,
        stationary_seconds=0.5,  # Short for testing
        fps=30.0,
    )

    # Moving track
    track = MockTrack(
        track_id=1, class_id=2, class_name="car",
        x=0.3, y=0.4, width=0.1, height=0.08,
        confidence=0.9, velocity_x=0.01, velocity_y=0.0
    )

    # Become active
    for _ in range(3):
        fsm.update([track])

    print(f"After activation: {fsm.get_lifecycle(1).state.value}")

    # Now stop moving
    track.velocity_x = 0.0
    track.velocity_y = 0.0

    # Simulate stationary for a bit (30 frames = 1 second at 30fps)
    for i in range(30):
        events = fsm.update([track])

        state_changes = [e for e in events if e.event_type == EventType.STATE_CHANGE]
        if state_changes:
            print(f"Frame {i+1}: State change to {state_changes[0].new_state.value}")

    lifecycle = fsm.get_lifecycle(1)
    print(f"Final state: {lifecycle.state.value}")
    print(f"Stationary frames: {lifecycle.stationary_frames}")

    assert lifecycle.state == ObjectState.STATIONARY
    print("\n[PASS] Stationary transition test")


def test_parked_transition():
    """Test transition to parked state."""
    print("=" * 60)
    print("Test: Parked Transition")
    print("=" * 60)

    fsm = ObjectStateMachine(
        validation_frames=2,
        velocity_threshold=0.002,
        stationary_seconds=0.1,
        parked_seconds=0.3,  # Short for testing
        fps=30.0,
    )

    track = MockTrack(
        track_id=1, class_id=2, class_name="car",
        x=0.3, y=0.4, width=0.1, height=0.08,
        confidence=0.9, velocity_x=0.0, velocity_y=0.0
    )

    # Become active then immediately stationary
    for _ in range(3):
        fsm.update([track])

    lifecycle = fsm.get_lifecycle(1)
    print(f"Initial state: {lifecycle.state.value}")

    # Wait to become stationary first
    time.sleep(0.15)
    fsm.update([track])

    lifecycle = fsm.get_lifecycle(1)
    print(f"After stationary wait: {lifecycle.state.value}")

    # Now wait for parked transition
    time.sleep(0.35)
    fsm.update([track])

    lifecycle = fsm.get_lifecycle(1)
    print(f"After parked wait: {lifecycle.state.value}")
    print(f"Time in state: {lifecycle.time_in_state:.2f}s")

    assert lifecycle.state == ObjectState.PARKED
    print("\n[PASS] Parked transition test")


def test_departure_event():
    """Test departure event generation."""
    print("=" * 60)
    print("Test: Departure Event")
    print("=" * 60)

    fsm = ObjectStateMachine(
        validation_frames=2,
        lost_seconds=0.1,  # Short for testing
        fps=30.0,
    )

    track = MockTrack(
        track_id=1, class_id=0, class_name="person",
        x=0.3, y=0.4, width=0.1, height=0.2,
        confidence=0.9, velocity_x=0.01, velocity_y=0.0
    )

    # Become active
    for _ in range(3):
        fsm.update([track])

    print(f"Active state: {fsm.get_lifecycle(1).state.value}")

    # Object disappears
    time.sleep(0.2)
    events = fsm.update([])  # No tracks

    print(f"After disappearance: {len(events)} events")

    departure_events = [e for e in events if e.event_type == EventType.DEPARTURE]
    assert len(departure_events) == 1
    print(f"  DEPARTURE event: track_id={departure_events[0].track_id}")

    print("\n[PASS] Departure event test")


def test_parked_no_departure():
    """Test that parked objects don't generate false departures."""
    print("=" * 60)
    print("Test: Parked Objects No False Departure")
    print("=" * 60)

    fsm = ObjectStateMachine(
        validation_frames=2,
        stationary_seconds=0.05,
        parked_seconds=0.1,
        lost_seconds=0.1,
        fps=30.0,
    )

    track = MockTrack(
        track_id=1, class_id=2, class_name="car",
        x=0.3, y=0.4, width=0.1, height=0.08,
        confidence=0.9, velocity_x=0.0, velocity_y=0.0
    )

    # Become active then parked
    for _ in range(3):
        fsm.update([track])

    time.sleep(0.15)
    fsm.update([track])

    lifecycle = fsm.get_lifecycle(1)
    print(f"State: {lifecycle.state.value}")
    assert lifecycle.state == ObjectState.PARKED

    # Now object disappears (but it's parked, so shouldn't be "departure")
    time.sleep(0.15)
    events = fsm.update([])

    departure_events = [e for e in events if e.event_type == EventType.DEPARTURE]
    print(f"Departure events: {len(departure_events)}")

    assert len(departure_events) == 0, "Parked objects shouldn't generate departure"
    print("\n[PASS] Parked no false departure test")


def test_resume_from_stationary():
    """Test object resuming movement from stationary."""
    print("=" * 60)
    print("Test: Resume From Stationary")
    print("=" * 60)

    fsm = ObjectStateMachine(
        validation_frames=2,
        velocity_threshold=0.002,
        stationary_seconds=0.1,
        fps=30.0,
    )

    track = MockTrack(
        track_id=1, class_id=0, class_name="person",
        x=0.3, y=0.4, width=0.05, height=0.15,
        confidence=0.9, velocity_x=0.0, velocity_y=0.0
    )

    # Become stationary
    for _ in range(3):
        fsm.update([track])

    time.sleep(0.15)
    fsm.update([track])

    lifecycle = fsm.get_lifecycle(1)
    print(f"After stopping: {lifecycle.state.value}")

    # Start moving again
    track.velocity_x = 0.01
    events = fsm.update([track])

    state_changes = [e for e in events if e.event_type == EventType.STATE_CHANGE]
    for sc in state_changes:
        print(f"State change: {sc.old_state.value} -> {sc.new_state.value}")

    lifecycle = fsm.get_lifecycle(1)
    print(f"After moving: {lifecycle.state.value}")

    assert lifecycle.state == ObjectState.ACTIVE
    print("\n[PASS] Resume from stationary test")


def test_notification_decisions():
    """Test notification decision logic."""
    print("=" * 60)
    print("Test: Notification Decisions")
    print("=" * 60)

    fsm = ObjectStateMachine()

    # Test arrival notification
    arrival = ObjectEvent(
        event_type=EventType.ARRIVAL,
        track_id=1, class_name="car", class_id=2,
        timestamp=time.time()
    )
    assert fsm.should_notify(arrival) == True
    print(f"ARRIVAL: should_notify = True")

    # Test departure notification
    departure = ObjectEvent(
        event_type=EventType.DEPARTURE,
        track_id=1, class_name="car", class_id=2,
        timestamp=time.time()
    )
    assert fsm.should_notify(departure) == True
    print(f"DEPARTURE: should_notify = True")

    # Test state change (no notification)
    state_change = ObjectEvent(
        event_type=EventType.STATE_CHANGE,
        track_id=1, class_name="car", class_id=2,
        timestamp=time.time(),
        old_state=ObjectState.ACTIVE,
        new_state=ObjectState.STATIONARY
    )
    assert fsm.should_notify(state_change) == False
    print(f"STATE_CHANGE: should_notify = False")

    print("\n[PASS] Notification decisions test")


def main():
    print("\nObject State Machine Tests\n")

    test_arrival_event()
    print()

    test_stationary_transition()
    print()

    test_parked_transition()
    print()

    test_departure_event()
    print()

    test_parked_no_departure()
    print()

    test_resume_from_stationary()
    print()

    test_notification_decisions()
    print()

    print("=" * 60)
    print("All FSM tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
