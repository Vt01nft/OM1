"""Tests for teleops_status_provider."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock ALL external dependencies BEFORE any provider imports
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["rclpy"] = MagicMock()
sys.modules["rclpy.node"] = MagicMock()
sys.modules["rclpy.qos"] = MagicMock()
sys.modules["sensor_msgs"] = MagicMock()
sys.modules["sensor_msgs.msg"] = MagicMock()
sys.modules["geometry_msgs"] = MagicMock()
sys.modules["geometry_msgs.msg"] = MagicMock()
sys.modules["nav_msgs"] = MagicMock()
sys.modules["nav_msgs.msg"] = MagicMock()
sys.modules["std_msgs"] = MagicMock()
sys.modules["std_msgs.msg"] = MagicMock()
sys.modules["elevenlabs"] = MagicMock()
sys.modules["riva"] = MagicMock()
sys.modules["riva.client"] = MagicMock()
sys.modules["pyaudio"] = MagicMock()
sys.modules["sounddevice"] = MagicMock()
sys.modules["websocket"] = MagicMock()
sys.modules["websockets"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()
sys.modules["pyrealsense2"] = MagicMock()
sys.modules["mjpeg"] = MagicMock()
sys.modules["mjpeg.client"] = MagicMock()
sys.modules["unitree"] = MagicMock()
sys.modules["unitree_sdk2py"] = MagicMock()
sys.modules["unitree_sdk2py.core"] = MagicMock()
sys.modules["unitree_sdk2py.core.channel"] = MagicMock()


class TestBatteryStatus:
    """Tests for BatteryStatus class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

    def test_initialization(self):
        """Test BatteryStatus initializes correctly."""
        from providers.teleops_status_provider import BatteryStatus

        battery = BatteryStatus(
            battery_level=85.5,
            temperature=25.0,
            voltage=12.5,
            timestamp="2023-01-01T12:00:00Z",
        )

        assert battery.battery_level == 85.5
        assert battery.temperature == 25.0
        assert battery.voltage == 12.5
        assert battery.timestamp == "2023-01-01T12:00:00Z"
        assert battery.charging_status is False

    def test_initialization_with_charging_status(self):
        """Test BatteryStatus initializes with charging status."""
        from providers.teleops_status_provider import BatteryStatus

        battery = BatteryStatus(
            battery_level=85.5,
            temperature=25.0,
            voltage=12.5,
            timestamp="2023-01-01T12:00:00Z",
            charging_status=True,
        )

        assert battery.charging_status is True

    def test_to_dict(self):
        """Test BatteryStatus converts to dictionary correctly."""
        from providers.teleops_status_provider import BatteryStatus

        battery = BatteryStatus(
            battery_level=85.5,
            temperature=25.0,
            voltage=12.5,
            timestamp="2023-01-01T12:00:00Z",
            charging_status=True,
        )

        result = battery.to_dict()
        expected = {
            "battery_level": 85.5,
            "charging_status": True,
            "temperature": 25.0,
            "voltage": 12.5,
            "timestamp": "2023-01-01T12:00:00Z",
        }

        assert result == expected

    def test_from_dict_complete_data(self):
        """Test BatteryStatus creates from complete dictionary."""
        from providers.teleops_status_provider import BatteryStatus

        data = {
            "battery_level": 75.0,
            "charging_status": True,
            "temperature": 30.0,
            "voltage": 11.8,
            "timestamp": "2023-01-01T12:00:00Z",
        }

        battery = BatteryStatus.from_dict(data)

        assert battery.battery_level == 75.0
        assert battery.charging_status is True
        assert battery.temperature == 30.0
        assert battery.voltage == 11.8
        assert battery.timestamp == "2023-01-01T12:00:00Z"

    def test_from_dict_missing_data(self):
        """Test BatteryStatus handles missing data with defaults."""
        from providers.teleops_status_provider import BatteryStatus

        with patch("time.time", return_value=1234567890.0):
            battery = BatteryStatus.from_dict({})

        assert battery.battery_level == 0.0
        assert battery.charging_status is False
        assert battery.temperature == 0.0
        assert battery.voltage == 0.0
        assert battery.timestamp == "1234567890.0"

    def test_from_dict_partial_data(self):
        """Test BatteryStatus handles partial data."""
        from providers.teleops_status_provider import BatteryStatus

        data = {"battery_level": 50.0, "temperature": 20.0}

        with patch("time.time", return_value=1234567890.0):
            battery = BatteryStatus.from_dict(data)

        assert battery.battery_level == 50.0
        assert battery.temperature == 20.0
        assert battery.voltage == 0.0
        assert battery.charging_status is False
        assert battery.timestamp == "1234567890.0"


class TestCommandStatus:
    """Tests for CommandStatus class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

    def test_initialization(self):
        """Test CommandStatus initializes correctly."""
        from providers.teleops_status_provider import CommandStatus

        command = CommandStatus(
            vx=1.0, vy=0.5, vyaw=0.2, timestamp="2023-01-01T12:00:00Z"
        )

        assert command.vx == 1.0
        assert command.vy == 0.5
        assert command.vyaw == 0.2
        assert command.timestamp == "2023-01-01T12:00:00Z"

    def test_to_dict(self):
        """Test CommandStatus converts to dictionary correctly."""
        from providers.teleops_status_provider import CommandStatus

        command = CommandStatus(
            vx=1.5, vy=-0.5, vyaw=0.8, timestamp="2023-01-01T12:00:00Z"
        )

        result = command.to_dict()
        expected = {
            "vx": 1.5,
            "vy": -0.5,
            "vyaw": 0.8,
            "timestamp": "2023-01-01T12:00:00Z",
        }

        assert result == expected

    def test_from_dict_complete_data(self):
        """Test CommandStatus creates from complete dictionary."""
        from providers.teleops_status_provider import CommandStatus

        data = {"vx": 2.0, "vy": -1.0, "vyaw": 1.5, "timestamp": "2023-01-01T12:00:00Z"}

        command = CommandStatus.from_dict(data)

        assert command.vx == 2.0
        assert command.vy == -1.0
        assert command.vyaw == 1.5
        assert command.timestamp == "2023-01-01T12:00:00Z"

    def test_from_dict_missing_data(self):
        """Test CommandStatus handles missing data with defaults."""
        from providers.teleops_status_provider import CommandStatus

        with patch("time.time", return_value=1234567890.0):
            command = CommandStatus.from_dict({})

        assert command.vx == 0.0
        assert command.vy == 0.0
        assert command.vyaw == 0.0
        assert command.timestamp == 1234567890.0

    def test_from_dict_partial_data(self):
        """Test CommandStatus handles partial data."""
        from providers.teleops_status_provider import CommandStatus

        data = {"vx": 3.0, "vyaw": 2.0}

        with patch("time.time", return_value=1234567890.0):
            command = CommandStatus.from_dict(data)

        assert command.vx == 3.0
        assert command.vy == 0.0
        assert command.vyaw == 2.0
        assert command.timestamp == 1234567890.0


class TestActionType:
    """Tests for ActionType enum."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

    def test_enum_values(self):
        """Test ActionType enum has correct values."""
        from providers.teleops_status_provider import ActionType

        assert ActionType.AI.value == "AI"
        assert ActionType.TELEOPS.value == "TELEOPS"
        assert ActionType.CONTROLLER.value == "CONTROLLER"

    def test_enum_equality(self):
        """Test ActionType enum equality."""
        from providers.teleops_status_provider import ActionType

        assert ActionType.AI == ActionType.AI
        assert ActionType.AI != ActionType.TELEOPS


class TestActionStatus:
    """Tests for ActionStatus class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

    def test_initialization(self):
        """Test ActionStatus initializes correctly."""
        from providers.teleops_status_provider import ActionStatus, ActionType

        action = ActionStatus(action=ActionType.TELEOPS, timestamp=1234567890.0)

        assert action.action == ActionType.TELEOPS
        assert action.timestamp == 1234567890.0

    def test_to_dict(self):
        """Test ActionStatus converts to dictionary correctly."""
        from providers.teleops_status_provider import ActionStatus, ActionType

        action = ActionStatus(action=ActionType.CONTROLLER, timestamp=1234567890.0)

        result = action.to_dict()
        expected = {"action": "CONTROLLER", "timestamp": 1234567890.0}

        assert result == expected

    def test_from_dict_complete_data(self):
        """Test ActionStatus creates from complete dictionary."""
        from providers.teleops_status_provider import ActionStatus, ActionType

        data = {"action": "TELEOPS", "timestamp": 1234567890.0}

        action = ActionStatus.from_dict(data)

        assert action.action == ActionType.TELEOPS
        assert action.timestamp == 1234567890.0

    def test_from_dict_missing_data(self):
        """Test ActionStatus handles missing data with defaults."""
        from providers.teleops_status_provider import ActionStatus, ActionType

        with patch("time.time", return_value=1234567890.0):
            action = ActionStatus.from_dict({})

        assert action.action == ActionType.AI
        assert action.timestamp == 1234567890.0

    def test_from_dict_partial_data(self):
        """Test ActionStatus handles partial data."""
        from providers.teleops_status_provider import ActionStatus, ActionType

        data = {"action": "CONTROLLER"}

        with patch("time.time", return_value=1234567890.0):
            action = ActionStatus.from_dict(data)

        assert action.action == ActionType.CONTROLLER
        assert action.timestamp == 1234567890.0


class TestTeleopsStatus:
    """Tests for TeleopsStatus class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

    def test_initialization(self):
        """Test TeleopsStatus initializes correctly."""
        from providers.teleops_status_provider import BatteryStatus, TeleopsStatus

        battery = BatteryStatus(
            battery_level=80.0,
            temperature=25.0,
            voltage=12.0,
            timestamp="2023-01-01T12:00:00Z",
        )

        teleops = TeleopsStatus(
            update_time="2023-01-01T12:00:00Z",
            battery_status=battery,
            machine_name="test_machine",
            video_connected=True,
        )

        assert teleops.update_time == "2023-01-01T12:00:00Z"
        assert teleops.battery_status == battery
        assert teleops.machine_name == "test_machine"
        assert teleops.video_connected is True

    def test_initialization_with_defaults(self):
        """Test TeleopsStatus initializes with default values."""
        from providers.teleops_status_provider import (
            ActionType,
            BatteryStatus,
            TeleopsStatus,
        )

        battery = BatteryStatus(
            battery_level=80.0,
            temperature=25.0,
            voltage=12.0,
            timestamp="2023-01-01T12:00:00Z",
        )

        teleops = TeleopsStatus(
            update_time="2023-01-01T12:00:00Z", battery_status=battery
        )

        assert teleops.machine_name == "unknown"
        assert teleops.video_connected is False
        assert teleops.action_status.action == ActionType.AI

    def test_to_dict(self):
        """Test TeleopsStatus converts to dictionary correctly."""
        from providers.teleops_status_provider import (
            ActionStatus,
            ActionType,
            BatteryStatus,
            TeleopsStatus,
        )

        battery = BatteryStatus(
            battery_level=80.0,
            temperature=25.0,
            voltage=12.0,
            timestamp="2023-01-01T12:00:00Z",
        )

        action = ActionStatus(action=ActionType.TELEOPS, timestamp=1234567890.0)

        teleops = TeleopsStatus(
            update_time="2023-01-01T12:00:00Z",
            battery_status=battery,
            action_status=action,
            machine_name="test_machine",
            video_connected=True,
        )

        result = teleops.to_dict()

        assert result["machine_name"] == "test_machine"
        assert result["update_time"] == "2023-01-01T12:00:00Z"
        assert result["video_connected"] is True
        assert "battery_status" in result
        assert "action_status" in result

    def test_from_dict_complete_data(self):
        """Test TeleopsStatus creates from complete dictionary."""
        from providers.teleops_status_provider import ActionType, TeleopsStatus

        data = {
            "update_time": "2023-01-01T12:00:00Z",
            "machine_name": "test_machine",
            "video_connected": True,
            "battery_status": {
                "battery_level": 75.0,
                "temperature": 30.0,
                "voltage": 11.5,
                "timestamp": "2023-01-01T12:00:00Z",
                "charging_status": True,
            },
            "action_status": {"action": "TELEOPS", "timestamp": 1234567890.0},
        }

        teleops = TeleopsStatus.from_dict(data)

        assert teleops.update_time == "2023-01-01T12:00:00Z"
        assert teleops.machine_name == "test_machine"
        assert teleops.video_connected is True
        assert teleops.battery_status.battery_level == 75.0
        assert teleops.action_status.action == ActionType.TELEOPS

    def test_from_dict_missing_data(self):
        """Test TeleopsStatus handles missing data with defaults."""
        from providers.teleops_status_provider import ActionType, TeleopsStatus

        with patch("time.time", return_value=1234567890.0):
            teleops = TeleopsStatus.from_dict({})

        assert teleops.update_time == 1234567890.0
        assert teleops.machine_name == "unknown"
        assert teleops.video_connected is False
        assert teleops.battery_status.battery_level == 0.0
        assert teleops.action_status.action == ActionType.AI


class TestTeleopsStatusProvider:
    """Tests for TeleopsStatusProvider class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

    def test_initialization_default(self):
        """Test TeleopsStatusProvider initializes with default values."""
        from providers.teleops_status_provider import TeleopsStatusProvider

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        provider = TeleopsStatusProvider()

        assert provider is not None
        assert provider.api_key is None
        assert provider.base_url == "https://api.openmind.org/api/core/teleops/status"
        assert provider.executor is not None

    def test_initialization_with_params(self):
        """Test TeleopsStatusProvider initializes with custom parameters."""
        from providers.teleops_status_provider import TeleopsStatusProvider

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        provider = TeleopsStatusProvider(
            api_key="test_key", base_url="https://custom.api.com/status"
        )

        assert provider.api_key == "test_key"
        assert provider.base_url == "https://custom.api.com/status"

    def test_singleton_pattern(self):
        """Test TeleopsStatusProvider follows singleton pattern."""
        from providers.teleops_status_provider import TeleopsStatusProvider

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        provider1 = TeleopsStatusProvider()
        provider2 = TeleopsStatusProvider()

        assert provider1 is provider2

    def test_get_status_method_exists(self):
        """Test get_status method exists and is callable."""
        from providers.teleops_status_provider import TeleopsStatusProvider

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        provider = TeleopsStatusProvider()

        assert hasattr(provider, "get_status")
        assert callable(getattr(provider, "get_status"))

    def test_get_status_returns_dict(self):
        """Test get_status method returns a dictionary."""
        from providers.teleops_status_provider import TeleopsStatusProvider

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        provider = TeleopsStatusProvider()
        result = provider.get_status()

        assert isinstance(result, dict)
