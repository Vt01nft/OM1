"""Tests for SafetyMonitorProvider."""

import time
from unittest.mock import MagicMock

import pytest

from providers.safety_monitor_provider import (
    SafetyEvent,
    SafetyEventType,
    SafetyLevel,
    SafetyMonitorProvider,
    SafetyStatus,
    SafetyThresholds,
    SafetyZone,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    SafetyMonitorProvider.reset_instance()
    yield
    SafetyMonitorProvider.reset_instance()


@pytest.fixture
def provider():
    """Create a SafetyMonitorProvider instance."""
    p = SafetyMonitorProvider()
    yield p
    p.stop()


@pytest.fixture
def thresholds():
    """Create default SafetyThresholds."""
    return SafetyThresholds()


class TestSafetyThresholds:
    """Tests for SafetyThresholds dataclass."""

    def test_default_values(self, thresholds):
        """Test default threshold values."""
        assert thresholds.battery_critical == 10.0
        assert thresholds.battery_low == 20.0
        assert thresholds.battery_warning == 30.0
        assert thresholds.temp_critical == 85.0
        assert thresholds.proximity_danger == 0.3
        assert thresholds.watchdog_timeout == 5.0

    def test_to_dict(self, thresholds):
        """Test conversion to dictionary."""
        result = thresholds.to_dict()
        assert isinstance(result, dict)
        assert result["battery_critical"] == 10.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"battery_critical": 5.0, "temp_critical": 90.0}
        t = SafetyThresholds.from_dict(data)
        assert t.battery_critical == 5.0
        assert t.temp_critical == 90.0
        assert t.battery_low == 20.0


class TestSafetyEvent:
    """Tests for SafetyEvent dataclass."""

    def test_create_event(self):
        """Test creating a safety event."""
        event = SafetyEvent(
            event_type=SafetyEventType.BATTERY_LOW,
            timestamp="2026-01-27T10:00:00",
            level=SafetyLevel.WARNING,
            message="Battery low",
            data={"battery_level": 15.0},
        )
        assert event.event_type == SafetyEventType.BATTERY_LOW
        assert event.level == SafetyLevel.WARNING

    def test_to_dict(self):
        """Test event conversion to dictionary."""
        event = SafetyEvent(
            event_type=SafetyEventType.EMERGENCY_STOP,
            timestamp="2026-01-27T10:00:00",
            level=SafetyLevel.EMERGENCY,
            message="Test",
        )
        result = event.to_dict()
        assert result["event_type"] == "emergency_stop"
        assert result["level"] == "emergency"

    def test_from_dict(self):
        """Test event creation from dictionary."""
        data = {
            "event_type": "battery_critical",
            "timestamp": "2026-01-27T10:00:00",
            "level": "critical",
            "message": "Critical",
        }
        event = SafetyEvent.from_dict(data)
        assert event.event_type == SafetyEventType.BATTERY_CRITICAL


class TestSafetyZone:
    """Tests for SafetyZone dataclass."""

    def test_create_zone(self):
        """Test creating a safety zone."""
        zone = SafetyZone(
            name="test",
            x_min=0.0,
            x_max=10.0,
            y_min=0.0,
            y_max=10.0,
        )
        assert zone.name == "test"
        assert zone.zone_type == "restricted"

    def test_contains_point_inside(self):
        """Test point inside zone."""
        zone = SafetyZone("test", 0.0, 10.0, 0.0, 10.0)
        assert zone.contains_point(5.0, 5.0) is True
        assert zone.contains_point(0.0, 0.0) is True

    def test_contains_point_outside(self):
        """Test point outside zone."""
        zone = SafetyZone("test", 0.0, 10.0, 0.0, 10.0)
        assert zone.contains_point(-1.0, 5.0) is False
        assert zone.contains_point(11.0, 5.0) is False

    def test_to_dict(self):
        """Test zone conversion to dictionary."""
        zone = SafetyZone("test", 1.0, 2.0, 3.0, 4.0, "warning")
        result = zone.to_dict()
        assert result["name"] == "test"
        assert result["zone_type"] == "warning"


class TestSafetyStatus:
    """Tests for SafetyStatus dataclass."""

    def test_create_status(self):
        """Test creating a safety status."""
        status = SafetyStatus(
            level=SafetyLevel.NORMAL,
            is_emergency_stopped=False,
            battery_level=80.0,
            temperature=35.0,
            closest_obstacle=2.0,
            active_warnings=[],
            timestamp="2026-01-27T10:00:00",
        )
        assert status.level == SafetyLevel.NORMAL

    def test_to_dict(self):
        """Test status conversion to dictionary."""
        status = SafetyStatus(
            level=SafetyLevel.WARNING,
            is_emergency_stopped=False,
            battery_level=25.0,
            temperature=45.0,
            closest_obstacle=0.8,
            active_warnings=["Low battery"],
            timestamp="2026-01-27T10:00:00",
        )
        result = status.to_dict()
        assert result["level"] == "warning"


class TestSafetyMonitorProviderInit:
    """Tests for SafetyMonitorProvider initialization."""

    def test_default_initialization(self, provider):
        """Test default initialization."""
        assert provider.thresholds is not None
        assert provider.is_safe_to_operate() is True
        assert provider.is_emergency_stopped() is False

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        p1 = SafetyMonitorProvider()
        p2 = SafetyMonitorProvider()
        assert p1 is p2
        p1.stop()


class TestSafetyMonitorProviderLifecycle:
    """Tests for lifecycle methods."""

    def test_start_stop(self, provider):
        """Test start and stop methods."""
        provider.start()
        assert provider._running is True
        provider.stop()
        assert provider._running is False

    def test_double_start(self, provider):
        """Test double start."""
        provider.start()
        provider.start()
        assert provider._running is True


class TestSafetyMonitorProviderEmergencyStop:
    """Tests for emergency stop functionality."""

    def test_trigger_emergency_stop(self, provider):
        """Test emergency stop activation."""
        provider.trigger_emergency_stop("Test")
        assert provider.is_emergency_stopped() is True
        assert provider.get_safety_level() == SafetyLevel.EMERGENCY
        assert provider.is_safe_to_operate() is False

    def test_emergency_callback(self, provider):
        """Test emergency callback."""
        callback = MagicMock()
        provider.register_emergency_callback(callback)
        provider.trigger_emergency_stop("Test")
        time.sleep(0.1)
        callback.assert_called_once()

    def test_clear_emergency_stop(self, provider):
        """Test clearing emergency stop."""
        provider.trigger_emergency_stop("Test")
        result = provider.clear_emergency_stop()
        assert result is True
        assert provider.is_emergency_stopped() is False

    def test_recovery_callback(self, provider):
        """Test recovery callback."""
        callback = MagicMock()
        provider.register_recovery_callback(callback)
        provider.trigger_emergency_stop("Test")
        provider.clear_emergency_stop()
        time.sleep(0.1)
        callback.assert_called_once()


class TestSafetyMonitorProviderBattery:
    """Tests for battery monitoring."""

    def test_update_battery_level(self, provider):
        """Test updating battery level."""
        provider.update_battery_level(75.0)
        status = provider.get_status()
        assert status.battery_level == 75.0

    def test_battery_clamp(self, provider):
        """Test battery clamping."""
        provider.update_battery_level(150.0)
        status = provider.get_status()
        assert status.battery_level == 100.0

        provider.update_battery_level(-10.0)
        status = provider.get_status()
        assert status.battery_level == 0.0


class TestSafetyMonitorProviderTemperature:
    """Tests for temperature monitoring."""

    def test_update_temperature(self, provider):
        """Test updating temperature."""
        provider.update_temperature(45.0)
        status = provider.get_status()
        assert status.temperature == 45.0


class TestSafetyMonitorProviderProximity:
    """Tests for proximity monitoring."""

    def test_update_closest_obstacle(self, provider):
        """Test updating obstacle distance."""
        provider.update_closest_obstacle(1.5)
        status = provider.get_status()
        assert status.closest_obstacle == 1.5

    def test_proximity_clamp(self, provider):
        """Test proximity clamping."""
        provider.update_closest_obstacle(-1.0)
        status = provider.get_status()
        assert status.closest_obstacle == 0.0


class TestSafetyMonitorProviderZones:
    """Tests for safety zones."""

    def test_add_zone(self, provider):
        """Test adding a zone."""
        zone = SafetyZone("test", 0.0, 10.0, 0.0, 10.0)
        provider.add_safety_zone(zone)
        zones = provider.get_safety_zones()
        assert "test" in zones

    def test_remove_zone(self, provider):
        """Test removing a zone."""
        zone = SafetyZone("test", 0.0, 10.0, 0.0, 10.0)
        provider.add_safety_zone(zone)
        result = provider.remove_safety_zone("test")
        assert result is True
        assert "test" not in provider.get_safety_zones()

    def test_remove_nonexistent(self, provider):
        """Test removing nonexistent zone."""
        result = provider.remove_safety_zone("nonexistent")
        assert result is False


class TestSafetyMonitorProviderWatchdog:
    """Tests for watchdog functionality."""

    def test_heartbeat(self, provider):
        """Test heartbeat."""
        provider.enable_watchdog(True)
        provider.heartbeat()
        assert provider._last_heartbeat > 0


class TestSafetyMonitorProviderStatus:
    """Tests for status queries."""

    def test_get_status(self, provider):
        """Test get_status."""
        provider.update_battery_level(75.0)
        provider.update_temperature(45.0)
        provider.update_closest_obstacle(1.5)

        status = provider.get_status()
        assert isinstance(status, SafetyStatus)
        assert status.battery_level == 75.0
        assert status.temperature == 45.0
        assert status.closest_obstacle == 1.5

    def test_get_safety_level(self, provider):
        """Test get_safety_level."""
        level = provider.get_safety_level()
        assert level == SafetyLevel.NORMAL

    def test_is_safe_to_operate(self, provider):
        """Test is_safe_to_operate."""
        assert provider.is_safe_to_operate() is True
        provider.trigger_emergency_stop("Test")
        assert provider.is_safe_to_operate() is False


class TestSafetyMonitorProviderEventHistory:
    """Tests for event history."""

    def test_get_event_history(self, provider):
        """Test event history retrieval."""
        provider.trigger_emergency_stop("Test")
        events = provider.get_event_history()
        assert len(events) > 0
        assert isinstance(events[0], SafetyEvent)


class TestSafetyMonitorProviderCallbacks:
    """Tests for callback functionality."""

    def test_warning_callback(self, provider):
        """Test warning callback."""
        callback = MagicMock()
        provider.register_warning_callback(callback)
        provider.start()
        provider.update_battery_level(15.0)
        time.sleep(0.3)
        callback.assert_called()

    def test_callback_exception(self, provider):
        """Test callback exception handling."""

        def bad_callback():
            raise ValueError("Test")

        provider.register_emergency_callback(bad_callback)
        provider.trigger_emergency_stop("Test")
        time.sleep(0.1)
        assert provider.is_emergency_stopped() is True


class TestSafetyMonitorProviderThresholds:
    """Tests for threshold configuration."""

    def test_set_thresholds(self, provider):
        """Test setting thresholds."""
        new_thresholds = SafetyThresholds(
            battery_critical=5.0,
            temp_critical=90.0,
        )
        provider.set_thresholds(new_thresholds)
        assert provider.thresholds.battery_critical == 5.0
        assert provider.thresholds.temp_critical == 90.0


class TestSafetyMonitorProviderEdgeCases:
    """Tests for edge cases."""

    def test_rapid_updates(self, provider):
        """Test rapid updates."""
        for i in range(100):
            provider.update_battery_level(50.0 + i % 10)
            provider.update_temperature(40.0 + i % 5)
            provider.update_closest_obstacle(1.0 + (i % 10) * 0.1)
        assert provider.is_safe_to_operate() is True

    def test_thread_safety(self, provider):
        """Test thread safety."""
        import threading

        def update_loop():
            for _ in range(50):
                provider.update_battery_level(50.0)
                provider.heartbeat()
                time.sleep(0.01)

        threads = [threading.Thread(target=update_loop) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert True
