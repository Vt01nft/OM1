"""Tests for unitree_go2_amcl_provider."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock ALL external dependencies BEFORE any provider imports
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()
sys.modules["zenoh_msgs.nav_msgs"] = MagicMock()
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


class TestUnitreeGo2AMCLProvider:
    """Tests for UnitreeGo2AMCLProvider class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

    def test_initialization_default_parameters(self):
        """Test provider initializes correctly with default parameters."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        provider = UnitreeGo2AMCLProvider()
        assert provider is not None
        assert provider.pose_tolerance == 0.4
        assert provider.yaw_tolerance == 0.2
        assert provider.localization_pose is None
        assert provider.localization_status is False
        assert not provider.is_localized

    def test_initialization_custom_parameters(self):
        """Test provider initializes correctly with custom parameters."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        provider = UnitreeGo2AMCLProvider(
            topic="custom_amcl", pose_tolerance=0.5, yaw_tolerance=0.3
        )
        assert provider.pose_tolerance == 0.5
        assert provider.yaw_tolerance == 0.3

    def test_singleton_pattern(self):
        """Test that provider follows singleton pattern."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        provider1 = UnitreeGo2AMCLProvider()
        provider2 = UnitreeGo2AMCLProvider()
        assert provider1 is provider2

    def test_amcl_message_callback_valid_message_localized(self):
        """Test amcl_message_callback processes valid localized message."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        # Mock numpy array
        mock_numpy = sys.modules["numpy"]
        mock_numpy.array.return_value = (
            [0.1, 0, 0, 0, 0, 0, 0, 0.1] + [0] * 28 + [0.03] + [0] * 0
        )
        mock_numpy.sqrt.side_effect = lambda x: 0.2 if x == 0.2 else 0.17

        # Mock zenoh message
        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"test_data"
        mock_sample.payload = mock_payload

        # Mock AMCL pose message
        mock_pose = MagicMock()
        mock_amcl_msg = MagicMock()
        mock_amcl_msg.covariance = (
            [0.1, 0, 0, 0, 0, 0, 0, 0.1] + [0] * 28 + [0.03] + [0] * 0
        )
        mock_amcl_msg.pose = mock_pose

        sys.modules["zenoh_msgs"].nav_msgs.AMCLPose.deserialize.return_value = (
            mock_amcl_msg
        )

        provider = UnitreeGo2AMCLProvider()
        provider.amcl_message_callback(mock_sample)

        assert provider.localization_pose is mock_pose
        assert provider.localization_status is True
        assert provider.is_localized is True

    def test_amcl_message_callback_valid_message_not_localized(self):
        """Test amcl_message_callback processes valid non-localized message."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        # Mock numpy array with high uncertainty
        mock_numpy = sys.modules["numpy"]
        mock_numpy.array.return_value = (
            [1.0, 0, 0, 0, 0, 0, 0, 1.0] + [0] * 28 + [0.5] + [0] * 0
        )
        mock_numpy.sqrt.side_effect = lambda x: 1.41 if x == 2.0 else 0.71

        # Mock zenoh message
        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"test_data"
        mock_sample.payload = mock_payload

        # Mock AMCL pose message
        mock_pose = MagicMock()
        mock_amcl_msg = MagicMock()
        mock_amcl_msg.covariance = (
            [1.0, 0, 0, 0, 0, 0, 0, 1.0] + [0] * 28 + [0.5] + [0] * 0
        )
        mock_amcl_msg.pose = mock_pose

        sys.modules["zenoh_msgs"].nav_msgs.AMCLPose.deserialize.return_value = (
            mock_amcl_msg
        )

        provider = UnitreeGo2AMCLProvider()
        provider.amcl_message_callback(mock_sample)

        assert provider.localization_pose is mock_pose
        assert provider.localization_status is False
        assert provider.is_localized is False

    def test_amcl_message_callback_empty_payload(self):
        """Test amcl_message_callback handles empty payload."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        # Mock zenoh message with empty payload
        mock_sample = MagicMock()
        mock_sample.payload = None

        provider = UnitreeGo2AMCLProvider()
        provider.amcl_message_callback(mock_sample)

        assert provider.localization_pose is None
        assert provider.localization_status is False

    def test_start_when_not_running(self):
        """Test start method when provider is not running."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        provider = UnitreeGo2AMCLProvider()
        provider.running = False

        with patch.object(provider, "register_message_callback") as mock_register:
            provider.start()
            mock_register.assert_called_once_with(provider.amcl_message_callback)
            assert provider.running is True

    def test_start_when_already_running(self):
        """Test start method when provider is already running."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        provider = UnitreeGo2AMCLProvider()
        provider.running = True

        with patch.object(provider, "register_message_callback") as mock_register:
            provider.start()
            mock_register.assert_not_called()

    def test_start_with_custom_callback(self):
        """Test start method with custom callback parameter."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        provider = UnitreeGo2AMCLProvider()
        provider.running = False

        custom_callback = MagicMock()
        with patch.object(provider, "register_message_callback") as mock_register:
            provider.start(custom_callback)
            mock_register.assert_called_once_with(provider.amcl_message_callback)

    def test_is_localized_property_true(self):
        """Test is_localized property returns True when localized."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        provider = UnitreeGo2AMCLProvider()
        provider.localization_status = True

        assert provider.is_localized is True

    def test_is_localized_property_false(self):
        """Test is_localized property returns False when not localized."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        provider = UnitreeGo2AMCLProvider()
        provider.localization_status = False

        assert provider.is_localized is False

    def test_pose_property_returns_current_pose(self):
        """Test pose property returns current localization pose."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        provider = UnitreeGo2AMCLProvider()
        mock_pose = MagicMock()
        provider.localization_pose = mock_pose

        assert provider.pose is mock_pose

    def test_pose_property_returns_none_initially(self):
        """Test pose property returns None when no pose available."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        provider = UnitreeGo2AMCLProvider()

        assert provider.pose is None

    def test_pose_tolerance_edge_case(self):
        """Test localization with pose tolerance at boundary."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        # Mock numpy to return exactly the tolerance value
        mock_numpy = sys.modules["numpy"]
        mock_numpy.array.return_value = (
            [0.08, 0, 0, 0, 0, 0, 0, 0.08] + [0] * 28 + [0.02] + [0] * 0
        )
        mock_numpy.sqrt.side_effect = lambda x: 0.4 if x == 0.16 else 0.14

        # Mock zenoh message
        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"test_data"
        mock_sample.payload = mock_payload

        # Mock AMCL pose message
        mock_pose = MagicMock()
        mock_amcl_msg = MagicMock()
        mock_amcl_msg.covariance = (
            [0.08, 0, 0, 0, 0, 0, 0, 0.08] + [0] * 28 + [0.02] + [0] * 0
        )
        mock_amcl_msg.pose = mock_pose

        sys.modules["zenoh_msgs"].nav_msgs.AMCLPose.deserialize.return_value = (
            mock_amcl_msg
        )

        provider = UnitreeGo2AMCLProvider()
        provider.amcl_message_callback(mock_sample)

        assert provider.localization_status is False

    def test_yaw_tolerance_edge_case(self):
        """Test localization with yaw tolerance at boundary."""
        from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider

        if hasattr(UnitreeGo2AMCLProvider, "reset"):
            UnitreeGo2AMCLProvider.reset()

        # Mock numpy to return exactly the tolerance value
        mock_numpy = sys.modules["numpy"]
        mock_numpy.array.return_value = (
            [0.02, 0, 0, 0, 0, 0, 0, 0.02] + [0] * 28 + [0.04] + [0] * 0
        )
        mock_numpy.sqrt.side_effect = lambda x: 0.2 if x == 0.04 else 0.2

        # Mock zenoh message
        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"test_data"
        mock_sample.payload = mock_payload

        # Mock AMCL pose message
        mock_pose = MagicMock()
        mock_amcl_msg = MagicMock()
        mock_amcl_msg.covariance = (
            [0.02, 0, 0, 0, 0, 0, 0, 0.02] + [0] * 28 + [0.04] + [0] * 0
        )
        mock_amcl_msg.pose = mock_pose

        sys.modules["zenoh_msgs"].nav_msgs.AMCLPose.deserialize.return_value = (
            mock_amcl_msg
        )

        provider = UnitreeGo2AMCLProvider()
        provider.amcl_message_callback(mock_sample)

        assert provider.localization_status is False
