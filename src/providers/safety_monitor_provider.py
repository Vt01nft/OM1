"""Safety Monitor Provider for OM1.

This module provides comprehensive safety monitoring for OM1 robots including
emergency stop capability, battery monitoring, temperature monitoring,
collision prevention, and safety event logging.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

from .singleton import singleton


class SafetyLevel(Enum):
    """Safety level classifications for the robot system."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafetyEventType(Enum):
    """Types of safety events that can occur."""

    EMERGENCY_STOP = "emergency_stop"
    BATTERY_LOW = "battery_low"
    BATTERY_CRITICAL = "battery_critical"
    TEMPERATURE_HIGH = "temperature_high"
    TEMPERATURE_CRITICAL = "temperature_critical"
    COLLISION_WARNING = "collision_warning"
    COLLISION_IMMINENT = "collision_imminent"
    ZONE_VIOLATION = "zone_violation"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    SYSTEM_RECOVERED = "system_recovered"
    SAFETY_CLEARED = "safety_cleared"


@dataclass
class SafetyThresholds:
    """Configurable thresholds for safety monitoring.

    Parameters
    ----------
    battery_critical : float
        Battery percentage below which is critical (default 10.0).
    battery_low : float
        Battery percentage below which is low (default 20.0).
    battery_warning : float
        Battery percentage below which triggers warning (default 30.0).
    temp_critical : float
        Temperature in Celsius above which is critical (default 85.0).
    temp_high : float
        Temperature in Celsius above which is high (default 75.0).
    temp_warning : float
        Temperature in Celsius above which triggers warning (default 65.0).
    proximity_danger : float
        Distance in meters below which collision is imminent (default 0.3).
    proximity_warning : float
        Distance in meters below which collision warning (default 0.8).
    watchdog_timeout : float
        Seconds without heartbeat before watchdog triggers (default 5.0).
    """

    battery_critical: float = 10.0
    battery_low: float = 20.0
    battery_warning: float = 30.0
    temp_critical: float = 85.0
    temp_high: float = 75.0
    temp_warning: float = 65.0
    proximity_danger: float = 0.3
    proximity_warning: float = 0.8
    watchdog_timeout: float = 5.0

    def to_dict(self) -> dict:
        """Convert thresholds to dictionary.

        Returns
        -------
        dict
            Dictionary representation of thresholds.
        """
        return {
            "battery_critical": self.battery_critical,
            "battery_low": self.battery_low,
            "battery_warning": self.battery_warning,
            "temp_critical": self.temp_critical,
            "temp_high": self.temp_high,
            "temp_warning": self.temp_warning,
            "proximity_danger": self.proximity_danger,
            "proximity_warning": self.proximity_warning,
            "watchdog_timeout": self.watchdog_timeout,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SafetyThresholds":
        """Create SafetyThresholds from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing threshold values.

        Returns
        -------
        SafetyThresholds
            New SafetyThresholds instance.
        """
        return cls(
            battery_critical=data.get("battery_critical", 10.0),
            battery_low=data.get("battery_low", 20.0),
            battery_warning=data.get("battery_warning", 30.0),
            temp_critical=data.get("temp_critical", 85.0),
            temp_high=data.get("temp_high", 75.0),
            temp_warning=data.get("temp_warning", 65.0),
            proximity_danger=data.get("proximity_danger", 0.3),
            proximity_warning=data.get("proximity_warning", 0.8),
            watchdog_timeout=data.get("watchdog_timeout", 5.0),
        )


@dataclass
class SafetyEvent:
    """Record of a safety event occurrence.

    Parameters
    ----------
    event_type : SafetyEventType
        The type of safety event.
    timestamp : str
        ISO format timestamp of when event occurred.
    level : SafetyLevel
        The severity level of the event.
    message : str
        Human-readable description of the event.
    data : dict
        Additional data associated with the event.
    """

    event_type: SafetyEventType
    timestamp: str
    level: SafetyLevel
    message: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert SafetyEvent to dictionary.

        Returns
        -------
        dict
            Dictionary representation of the event.
        """
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "message": self.message,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SafetyEvent":
        """Create SafetyEvent from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing event data.

        Returns
        -------
        SafetyEvent
            New SafetyEvent instance.
        """
        return cls(
            event_type=SafetyEventType(data.get("event_type", "safety_cleared")),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            level=SafetyLevel(data.get("level", "normal")),
            message=data.get("message", ""),
            data=data.get("data", {}),
        )


@dataclass
class SafetyZone:
    """Definition of a restricted safety zone.

    Parameters
    ----------
    name : str
        Identifier for the zone.
    x_min : float
        Minimum X coordinate of zone boundary.
    x_max : float
        Maximum X coordinate of zone boundary.
    y_min : float
        Minimum Y coordinate of zone boundary.
    y_max : float
        Maximum Y coordinate of zone boundary.
    zone_type : str
        Type: "restricted" (cannot enter) or "warning" (alert only).
    """

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    zone_type: str = "restricted"

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within this zone.

        Parameters
        ----------
        x : float
            X coordinate to check.
        y : float
            Y coordinate to check.

        Returns
        -------
        bool
            True if point is within zone boundaries.
        """
        in_x = self.x_min <= x <= self.x_max
        in_y = self.y_min <= y <= self.y_max
        return in_x and in_y

    def to_dict(self) -> dict:
        """Convert SafetyZone to dictionary.

        Returns
        -------
        dict
            Dictionary representation of the zone.
        """
        return {
            "name": self.name,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "zone_type": self.zone_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SafetyZone":
        """Create SafetyZone from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing zone data.

        Returns
        -------
        SafetyZone
            New SafetyZone instance.
        """
        return cls(
            name=data.get("name", "unknown"),
            x_min=data.get("x_min", 0.0),
            x_max=data.get("x_max", 0.0),
            y_min=data.get("y_min", 0.0),
            y_max=data.get("y_max", 0.0),
            zone_type=data.get("zone_type", "restricted"),
        )


@dataclass
class SafetyStatus:
    """Current safety status of the robot system.

    Parameters
    ----------
    level : SafetyLevel
        Current overall safety level.
    is_emergency_stopped : bool
        Whether emergency stop is active.
    battery_level : float
        Current battery percentage.
    temperature : float
        Current system temperature in Celsius.
    closest_obstacle : float
        Distance to closest obstacle in meters.
    active_warnings : List[str]
        List of active warning messages.
    timestamp : str
        ISO format timestamp of status update.
    """

    level: SafetyLevel
    is_emergency_stopped: bool
    battery_level: float
    temperature: float
    closest_obstacle: float
    active_warnings: List[str]
    timestamp: str

    def to_dict(self) -> dict:
        """Convert SafetyStatus to dictionary.

        Returns
        -------
        dict
            Dictionary representation of status.
        """
        return {
            "level": self.level.value,
            "is_emergency_stopped": self.is_emergency_stopped,
            "battery_level": self.battery_level,
            "temperature": self.temperature,
            "closest_obstacle": self.closest_obstacle,
            "active_warnings": self.active_warnings,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SafetyStatus":
        """Create SafetyStatus from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing status data.

        Returns
        -------
        SafetyStatus
            New SafetyStatus instance.
        """
        return cls(
            level=SafetyLevel(data.get("level", "normal")),
            is_emergency_stopped=data.get("is_emergency_stopped", False),
            battery_level=data.get("battery_level", 100.0),
            temperature=data.get("temperature", 25.0),
            closest_obstacle=data.get("closest_obstacle", float("inf")),
            active_warnings=data.get("active_warnings", []),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


@singleton
class SafetyMonitorProvider:
    """Centralized safety monitoring provider for OM1 robots.

    Provides comprehensive safety monitoring including emergency stop,
    battery monitoring, temperature monitoring, collision prevention,
    safety zones, and watchdog functionality.
    """

    def __init__(
        self,
        thresholds: Optional[SafetyThresholds] = None,
        monitor_interval: float = 0.1,
        max_event_history: int = 1000,
    ):
        """Initialize the SafetyMonitorProvider.

        Parameters
        ----------
        thresholds : Optional[SafetyThresholds]
            Safety thresholds configuration. Uses defaults if None.
        monitor_interval : float
            Interval in seconds between safety checks (default 0.1).
        max_event_history : int
            Maximum number of events to keep in history (default 1000).
        """
        self.thresholds = thresholds or SafetyThresholds()
        self.monitor_interval = monitor_interval
        self.max_event_history = max_event_history

        self._is_emergency_stopped = False
        self._battery_level = 100.0
        self._temperature = 25.0
        self._closest_obstacle = float("inf")
        self._current_position = (0.0, 0.0)
        self._current_level = SafetyLevel.NORMAL
        self._active_warnings: List[str] = []

        self._emergency_callbacks: List[Callable[[], None]] = []
        self._warning_callbacks: List[Callable[[SafetyEvent], None]] = []
        self._recovery_callbacks: List[Callable[[], None]] = []

        self._safety_zones: Dict[str, SafetyZone] = {}

        self._event_history: List[SafetyEvent] = []
        self._history_lock = threading.Lock()

        self._last_heartbeat = time.time()
        self._watchdog_enabled = False

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=2)

        logging.info("SafetyMonitorProvider initialized")

    def start(self) -> None:
        """Start the safety monitoring system."""
        if self._running:
            logging.warning("SafetyMonitorProvider already running")
            return

        self._running = True
        self._stop_event.clear()
        self._last_heartbeat = time.time()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="safety-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

        logging.info("SafetyMonitorProvider started")

    def stop(self) -> None:
        """Stop the safety monitoring system."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        self._executor.shutdown(wait=False)

        logging.info("SafetyMonitorProvider stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop that runs in background thread."""
        while self._running and not self._stop_event.is_set():
            try:
                self._check_safety_conditions()
                self._check_watchdog()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logging.error(f"SafetyMonitor error: {e}")

    def _check_safety_conditions(self) -> None:
        """Check all safety conditions and update status."""
        warnings = []
        new_level = SafetyLevel.NORMAL

        battery_level = self._check_battery()
        if battery_level == SafetyLevel.CRITICAL:
            new_level = SafetyLevel.CRITICAL
            warnings.append("Battery critically low")
        elif battery_level == SafetyLevel.WARNING:
            if new_level == SafetyLevel.NORMAL:
                new_level = SafetyLevel.WARNING
            warnings.append("Battery low")

        temp_level = self._check_temperature()
        if temp_level == SafetyLevel.CRITICAL:
            new_level = SafetyLevel.CRITICAL
            warnings.append("Temperature critical")
        elif temp_level == SafetyLevel.WARNING:
            if new_level == SafetyLevel.NORMAL:
                new_level = SafetyLevel.WARNING
            warnings.append("Temperature high")

        proximity_level = self._check_proximity()
        if proximity_level == SafetyLevel.EMERGENCY:
            new_level = SafetyLevel.EMERGENCY
            warnings.append("Collision imminent")
        elif proximity_level == SafetyLevel.WARNING:
            if new_level == SafetyLevel.NORMAL:
                new_level = SafetyLevel.WARNING
            warnings.append("Obstacle nearby")

        zone_violation = self._check_safety_zones()
        if zone_violation:
            new_level = SafetyLevel.CRITICAL
            warnings.append(f"Zone violation: {zone_violation}")

        self._active_warnings = warnings
        self._current_level = new_level

        if new_level == SafetyLevel.EMERGENCY:
            if not self._is_emergency_stopped:
                self.trigger_emergency_stop("Automatic emergency stop")

    def _check_battery(self) -> SafetyLevel:
        """Check battery level against thresholds."""
        if self._battery_level <= self.thresholds.battery_critical:
            if not self._has_recent_event(SafetyEventType.BATTERY_CRITICAL):
                self._log_event(
                    SafetyEventType.BATTERY_CRITICAL,
                    SafetyLevel.CRITICAL,
                    f"Battery critical: {self._battery_level:.1f}%",
                    {"battery_level": self._battery_level},
                )
            return SafetyLevel.CRITICAL
        elif self._battery_level <= self.thresholds.battery_low:
            if not self._has_recent_event(SafetyEventType.BATTERY_LOW):
                self._log_event(
                    SafetyEventType.BATTERY_LOW,
                    SafetyLevel.WARNING,
                    f"Battery low: {self._battery_level:.1f}%",
                    {"battery_level": self._battery_level},
                )
            return SafetyLevel.WARNING
        return SafetyLevel.NORMAL

    def _check_temperature(self) -> SafetyLevel:
        """Check temperature against thresholds."""
        if self._temperature >= self.thresholds.temp_critical:
            event_type = SafetyEventType.TEMPERATURE_CRITICAL
            if not self._has_recent_event(event_type):
                self._log_event(
                    SafetyEventType.TEMPERATURE_CRITICAL,
                    SafetyLevel.CRITICAL,
                    f"Temperature critical: {self._temperature:.1f}C",
                    {"temperature": self._temperature},
                )
            return SafetyLevel.CRITICAL
        elif self._temperature >= self.thresholds.temp_high:
            if not self._has_recent_event(SafetyEventType.TEMPERATURE_HIGH):
                self._log_event(
                    SafetyEventType.TEMPERATURE_HIGH,
                    SafetyLevel.WARNING,
                    f"Temperature high: {self._temperature:.1f}C",
                    {"temperature": self._temperature},
                )
            return SafetyLevel.WARNING
        return SafetyLevel.NORMAL

    def _check_proximity(self) -> SafetyLevel:
        """Check proximity to obstacles."""
        if self._closest_obstacle <= self.thresholds.proximity_danger:
            if not self._has_recent_event(SafetyEventType.COLLISION_IMMINENT):
                self._log_event(
                    SafetyEventType.COLLISION_IMMINENT,
                    SafetyLevel.EMERGENCY,
                    f"Collision imminent: {self._closest_obstacle:.2f}m",
                    {"distance": self._closest_obstacle},
                )
            return SafetyLevel.EMERGENCY
        elif self._closest_obstacle <= self.thresholds.proximity_warning:
            if not self._has_recent_event(SafetyEventType.COLLISION_WARNING):
                self._log_event(
                    SafetyEventType.COLLISION_WARNING,
                    SafetyLevel.WARNING,
                    f"Obstacle nearby: {self._closest_obstacle:.2f}m",
                    {"distance": self._closest_obstacle},
                )
            return SafetyLevel.WARNING
        return SafetyLevel.NORMAL

    def _check_safety_zones(self) -> Optional[str]:
        """Check if current position violates any safety zones."""
        x, y = self._current_position
        for name, zone in self._safety_zones.items():
            if zone.contains_point(x, y):
                if zone.zone_type == "restricted":
                    evt = SafetyEventType.ZONE_VIOLATION
                    if not self._has_recent_event(evt):
                        self._log_event(
                            SafetyEventType.ZONE_VIOLATION,
                            SafetyLevel.CRITICAL,
                            f"Entered restricted zone: {name}",
                            {"zone": name, "position": {"x": x, "y": y}},
                        )
                    return name
        return None

    def _check_watchdog(self) -> None:
        """Check watchdog timer for system hangs."""
        if not self._watchdog_enabled:
            return

        elapsed = time.time() - self._last_heartbeat
        if elapsed > self.thresholds.watchdog_timeout:
            if not self._has_recent_event(SafetyEventType.WATCHDOG_TIMEOUT):
                self._log_event(
                    SafetyEventType.WATCHDOG_TIMEOUT,
                    SafetyLevel.CRITICAL,
                    f"Watchdog timeout: {elapsed:.1f}s",
                    {"elapsed": elapsed},
                )
                self.trigger_emergency_stop("Watchdog timeout")

    def _has_recent_event(
        self, event_type: SafetyEventType, within_seconds: float = 5.0
    ) -> bool:
        """Check if an event type occurred recently."""
        with self._history_lock:
            now = datetime.now()
            for event in reversed(self._event_history):
                event_time = datetime.fromisoformat(event.timestamp)
                if (now - event_time).total_seconds() > within_seconds:
                    break
                if event.event_type == event_type:
                    return True
        return False

    def _log_event(
        self,
        event_type: SafetyEventType,
        level: SafetyLevel,
        message: str,
        data: Optional[dict] = None,
    ) -> SafetyEvent:
        """Log a safety event."""
        event = SafetyEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            data=data or {},
        )

        with self._history_lock:
            self._event_history.append(event)
            if len(self._event_history) > self.max_event_history:
                self._event_history.pop(0)

        log_level = {
            SafetyLevel.NORMAL: logging.INFO,
            SafetyLevel.WARNING: logging.WARNING,
            SafetyLevel.CRITICAL: logging.ERROR,
            SafetyLevel.EMERGENCY: logging.CRITICAL,
        }.get(level, logging.INFO)

        logging.log(log_level, f"SafetyEvent: {message}")

        if level in [SafetyLevel.WARNING, SafetyLevel.CRITICAL]:
            self._executor.submit(self._notify_warning_callbacks, event)

        return event

    def _notify_warning_callbacks(self, event: SafetyEvent) -> None:
        """Notify all warning callbacks of an event."""
        for callback in self._warning_callbacks:
            try:
                callback(event)
            except Exception as e:
                logging.error(f"Warning callback error: {e}")

    def _notify_emergency_callbacks(self) -> None:
        """Notify all emergency callbacks."""
        for callback in self._emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logging.error(f"Emergency callback error: {e}")

    def _notify_recovery_callbacks(self) -> None:
        """Notify all recovery callbacks."""
        for callback in self._recovery_callbacks:
            try:
                callback()
            except Exception as e:
                logging.error(f"Recovery callback error: {e}")

    def trigger_emergency_stop(self, reason: str = "Manual trigger") -> None:
        """Trigger an emergency stop.

        Parameters
        ----------
        reason : str
            Reason for the emergency stop.
        """
        if self._is_emergency_stopped:
            logging.warning("Emergency stop already active")
            return

        self._is_emergency_stopped = True
        self._current_level = SafetyLevel.EMERGENCY

        self._log_event(
            SafetyEventType.EMERGENCY_STOP,
            SafetyLevel.EMERGENCY,
            f"Emergency stop: {reason}",
            {"reason": reason},
        )

        logging.critical(f"EMERGENCY STOP: {reason}")

        self._executor.submit(self._notify_emergency_callbacks)

    def clear_emergency_stop(self) -> bool:
        """Clear the emergency stop if conditions are safe.

        Returns
        -------
        bool
            True if emergency stop was cleared, False if unsafe.
        """
        if not self._is_emergency_stopped:
            return True

        if self._battery_level <= self.thresholds.battery_critical:
            logging.warning("Cannot clear emergency: battery critical")
            return False

        if self._temperature >= self.thresholds.temp_critical:
            logging.warning("Cannot clear emergency: temperature critical")
            return False

        if self._closest_obstacle <= self.thresholds.proximity_danger:
            logging.warning("Cannot clear emergency: obstacle too close")
            return False

        self._is_emergency_stopped = False
        self._current_level = SafetyLevel.NORMAL

        self._log_event(
            SafetyEventType.SAFETY_CLEARED,
            SafetyLevel.NORMAL,
            "Emergency stop cleared",
            {},
        )

        logging.info("Emergency stop cleared")

        self._executor.submit(self._notify_recovery_callbacks)

        return True

    def register_emergency_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called on emergency stop.

        Parameters
        ----------
        callback : Callable[[], None]
            Function to call when emergency stop is triggered.
        """
        self._emergency_callbacks.append(callback)

    def register_warning_callback(
        self, callback: Callable[[SafetyEvent], None]
    ) -> None:
        """Register a callback for safety warnings.

        Parameters
        ----------
        callback : Callable[[SafetyEvent], None]
            Function to call when a warning event occurs.
        """
        self._warning_callbacks.append(callback)

    def register_recovery_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback for system recovery.

        Parameters
        ----------
        callback : Callable[[], None]
            Function to call when system recovers from emergency.
        """
        self._recovery_callbacks.append(callback)

    def update_battery_level(self, level: float) -> None:
        """Update the current battery level.

        Parameters
        ----------
        level : float
            Battery level as percentage (0-100).
        """
        self._battery_level = max(0.0, min(100.0, level))

    def update_temperature(self, temp: float) -> None:
        """Update the current system temperature.

        Parameters
        ----------
        temp : float
            Temperature in Celsius.
        """
        self._temperature = temp

    def update_closest_obstacle(self, distance: float) -> None:
        """Update the distance to closest obstacle.

        Parameters
        ----------
        distance : float
            Distance in meters. Use float('inf') for no obstacle.
        """
        self._closest_obstacle = max(0.0, distance)

    def update_position(self, x: float, y: float) -> None:
        """Update the current robot position.

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.
        """
        self._current_position = (x, y)

    def heartbeat(self) -> None:
        """Send a heartbeat to reset the watchdog timer."""
        self._last_heartbeat = time.time()

    def enable_watchdog(self, enabled: bool = True) -> None:
        """Enable or disable the watchdog timer.

        Parameters
        ----------
        enabled : bool
            True to enable watchdog, False to disable.
        """
        self._watchdog_enabled = enabled
        if enabled:
            self._last_heartbeat = time.time()

    def add_safety_zone(self, zone: SafetyZone) -> None:
        """Add a safety zone.

        Parameters
        ----------
        zone : SafetyZone
            The safety zone to add.
        """
        self._safety_zones[zone.name] = zone
        logging.info(f"Added safety zone: {zone.name}")

    def remove_safety_zone(self, name: str) -> bool:
        """Remove a safety zone.

        Parameters
        ----------
        name : str
            Name of the zone to remove.

        Returns
        -------
        bool
            True if zone was removed, False if not found.
        """
        if name in self._safety_zones:
            del self._safety_zones[name]
            logging.info(f"Removed safety zone: {name}")
            return True
        return False

    def get_safety_zones(self) -> Dict[str, SafetyZone]:
        """Get all configured safety zones.

        Returns
        -------
        Dict[str, SafetyZone]
            Dictionary of zone name to SafetyZone.
        """
        return self._safety_zones.copy()

    def is_safe_to_operate(self) -> bool:
        """Check if it is safe for the robot to operate.

        Returns
        -------
        bool
            True if safe to operate, False otherwise.
        """
        if self._is_emergency_stopped:
            return False
        critical_levels = [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]
        if self._current_level in critical_levels:
            return False
        return True

    def is_emergency_stopped(self) -> bool:
        """Check if emergency stop is active.

        Returns
        -------
        bool
            True if emergency stop is active.
        """
        return self._is_emergency_stopped

    def get_safety_level(self) -> SafetyLevel:
        """Get the current safety level.

        Returns
        -------
        SafetyLevel
            Current safety level.
        """
        return self._current_level

    def get_status(self) -> SafetyStatus:
        """Get comprehensive safety status.

        Returns
        -------
        SafetyStatus
            Current safety status.
        """
        return SafetyStatus(
            level=self._current_level,
            is_emergency_stopped=self._is_emergency_stopped,
            battery_level=self._battery_level,
            temperature=self._temperature,
            closest_obstacle=self._closest_obstacle,
            active_warnings=self._active_warnings.copy(),
            timestamp=datetime.now().isoformat(),
        )

    def get_event_history(self, limit: int = 100) -> List[SafetyEvent]:
        """Get recent safety event history.

        Parameters
        ----------
        limit : int
            Maximum number of events to return (default 100).

        Returns
        -------
        List[SafetyEvent]
            List of recent safety events.
        """
        with self._history_lock:
            return self._event_history[-limit:]

    def set_thresholds(self, thresholds: SafetyThresholds) -> None:
        """Update safety thresholds.

        Parameters
        ----------
        thresholds : SafetyThresholds
            New threshold values.
        """
        self.thresholds = thresholds
        logging.info("Safety thresholds updated")
