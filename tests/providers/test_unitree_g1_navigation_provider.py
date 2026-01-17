"""Tests for unitree_g1_navigation_provider."""

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
sys.modules["om1_speech"] = MagicMock()


class TestUnitreeG1NavigationProvider:
    """Tests for UnitreeG1NavigationProvider class."""

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
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
        ):

            mock_zenoh_session = MagicMock()
            mock_session.return_value = mock_zenoh_session

            provider = UnitreeG1NavigationProvider()

            assert provider is not None
            assert provider.navigation_status_topic == "navigate_to_pose/_action/status"
            assert provider.goal_pose_topic == "goal_pose"
            assert provider.cancel_goal_topic == "navigate_to_pose/_action/cancel_goal"
            assert provider.running is False
            assert provider.navigation_status == "UNKNOWN"
            assert provider._nav_in_progress is False
            assert provider._current_destination is None

    def test_initialization_custom_parameters(self):
        """Test provider initializes correctly with custom parameters."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
        ):

            mock_zenoh_session = MagicMock()
            mock_session.return_value = mock_zenoh_session

            provider = UnitreeG1NavigationProvider(
                navigation_status_topic="custom_status",
                goal_pose_topic="custom_goal",
                cancel_goal_topic="custom_cancel",
            )

            assert provider.navigation_status_topic == "custom_status"
            assert provider.goal_pose_topic == "custom_goal"
            assert provider.cancel_goal_topic == "custom_cancel"

    def test_initialization_zenoh_session_error(self):
        """Test provider handles zenoh session creation error."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
        ):

            mock_session.side_effect = Exception("Connection failed")

            provider = UnitreeG1NavigationProvider()

            assert provider.session is None

    def test_singleton_pattern(self):
        """Test provider follows singleton pattern."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
        ):

            mock_zenoh_session = MagicMock()
            mock_session.return_value = mock_zenoh_session

            provider1 = UnitreeG1NavigationProvider()
            provider2 = UnitreeG1NavigationProvider()

            assert provider1 is provider2

    def test_start_success(self):
        """Test start method succeeds when session is available."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
        ):

            mock_zenoh_session = MagicMock()
            mock_session.return_value = mock_zenoh_session

            provider = UnitreeG1NavigationProvider()
            provider.start()

            assert provider.running is True
            mock_zenoh_session.declare_subscriber.assert_called_once()

    def test_start_no_session(self):
        """Test start method handles missing session gracefully."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
        ):

            mock_session.side_effect = Exception("Connection failed")

            provider = UnitreeG1NavigationProvider()
            provider.start()

            assert provider.running is False

    def test_start_already_running(self):
        """Test start method when provider is already running."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
        ):

            mock_zenoh_session = MagicMock()
            mock_session.return_value = mock_zenoh_session

            provider = UnitreeG1NavigationProvider()
            provider.running = True

            provider.start()

            # Should not call declare_subscriber again
            mock_zenoh_session.declare_subscriber.assert_not_called()

    def test_navigation_status_message_callback_accepted_status(self):
        """Test navigation status callback with ACCEPTED status."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
            patch("providers.unitree_g1_navigation_provider.nav_msgs") as mock_nav_msgs,
        ):

            mock_zenoh_session = MagicMock()
            mock_session.return_value = mock_zenoh_session

            # Mock the status message
            mock_status = MagicMock()
            mock_status.status = 1  # ACCEPTED
            mock_nav2_status = MagicMock()
            mock_nav2_status.status_list = [mock_status]
            mock_nav_msgs.Nav2Status.deserialize.return_value = mock_nav2_status

            # Mock zenoh sample
            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test_data"

            provider = UnitreeG1NavigationProvider()
            provider._nav_in_progress = False

            provider.navigation_status_message_callback(mock_sample)

            assert provider.navigation_status == "ACCEPTED"
            assert provider._nav_in_progress is True

    def test_navigation_status_message_callback_executing_status(self):
        """Test navigation status callback with EXECUTING status."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
            patch("providers.unitree_g1_navigation_provider.nav_msgs") as mock_nav_msgs,
        ):

            mock_zenoh_session = MagicMock()
            mock_session.return_value = mock_zenoh_session

            # Mock the status message
            mock_status = MagicMock()
            mock_status.status = 2  # EXECUTING
            mock_nav2_status = MagicMock()
            mock_nav2_status.status_list = [mock_status]
            mock_nav_msgs.Nav2Status.deserialize.return_value = mock_nav2_status

            # Mock zenoh sample
            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test_data"

            provider = UnitreeG1NavigationProvider()
            provider._nav_in_progress = False

            provider.navigation_status_message_callback(mock_sample)

            assert provider.navigation_status == "EXECUTING"
            assert provider._nav_in_progress is True

    def test_navigation_status_message_callback_succeeded_status(self):
        """Test navigation status callback with SUCCEEDED status."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
            patch("providers.unitree_g1_navigation_provider.nav_msgs") as mock_nav_msgs,
        ):

            mock_zenoh_session = MagicMock()
            mock_session.return_value = mock_zenoh_session

            # Mock the status message
            mock_status = MagicMock()
            mock_status.status = 4  # SUCCEEDED
            mock_nav2_status = MagicMock()
            mock_nav2_status.status_list = [mock_status]
            mock_nav_msgs.Nav2Status.deserialize.return_value = mock_nav2_status

            # Mock zenoh sample
            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test_data"

            provider = UnitreeG1NavigationProvider()
            provider._nav_in_progress = True

            provider.navigation_status_message_callback(mock_sample)

            assert provider.navigation_status == "SUCCEEDED"
            assert provider._nav_in_progress is False

    def test_navigation_status_message_callback_unknown_status(self):
        """Test navigation status callback with unknown status code."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
            patch("providers.unitree_g1_navigation_provider.nav_msgs") as mock_nav_msgs,
        ):

            mock_zenoh_session = MagicMock()
            mock_session.return_value = mock_zenoh_session

            # Mock the status message
            mock_status = MagicMock()
            mock_status.status = 999  # Unknown status
            mock_nav2_status = MagicMock()
            mock_nav2_status.status_list = [mock_status]
            mock_nav_msgs.Nav2Status.deserialize.return_value = mock_nav2_status

            # Mock zenoh sample
            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test_data"

            provider = UnitreeG1NavigationProvider()

            provider.navigation_status_message_callback(mock_sample)

            assert provider.navigation_status == "UNKNOWN"

    def test_navigation_status_message_callback_empty_status_list(self):
        """Test navigation status callback with empty status list."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
            patch("providers.unitree_g1_navigation_provider.nav_msgs") as mock_nav_msgs,
        ):

            mock_zenoh_session = MagicMock()
            mock_session.return_value = mock_zenoh_session

            # Mock the status message with empty list
            mock_nav2_status = MagicMock()
            mock_nav2_status.status_list = []
            mock_nav_msgs.Nav2Status.deserialize.return_value = mock_nav2_status

            # Mock zenoh sample
            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test_data"

            provider = UnitreeG1NavigationProvider()
            original_status = provider.navigation_status

            provider.navigation_status_message_callback(mock_sample)

            # Status should remain unchanged
            assert provider.navigation_status == original_status

    def test_navigation_status_message_callback_no_payload(self):
        """Test navigation status callback with no payload."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
        ):

            mock_zenoh_session = MagicMock()
            mock_session.return_value = mock_zenoh_session

            # Mock zenoh sample with no payload
            mock_sample = MagicMock()
            mock_sample.payload = None

            provider = UnitreeG1NavigationProvider()
            original_status = provider.navigation_status

            provider.navigation_status_message_callback(mock_sample)

            # Status should remain unchanged
            assert provider.navigation_status == original_status

    def test_ai_status_publisher_creation_error(self):
        """Test provider handles AI status publisher creation error."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
        ):

            mock_zenoh_session = MagicMock()
            mock_zenoh_session.declare_publisher.side_effect = Exception(
                "Publisher error"
            )
            mock_session.return_value = mock_zenoh_session

            provider = UnitreeG1NavigationProvider()

            # Should handle the error gracefully
            assert provider.ai_status_pub is None

    def test_ai_status_topic_configuration(self):
        """Test AI status topic is configured correctly."""
        from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider

        if hasattr(UnitreeG1NavigationProvider, "reset"):
            UnitreeG1NavigationProvider.reset()

        with (
            patch(
                "providers.unitree_g1_navigation_provider.open_zenoh_session"
            ) as mock_session,
            patch(
                "providers.unitree_g1_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
        ):

            mock_zenoh_session = MagicMock()
            mock_session.return_value = mock_zenoh_session

            provider = UnitreeG1NavigationProvider()

            assert provider.ai_status_topic == "om/ai/request"
            mock_zenoh_session.declare_publisher.assert_called_with("om/ai/request")
