"""Tests for unitree_go2_navigation_provider."""

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


class TestUnitreeGo2NavigationProvider:
    """Tests for UnitreeGo2NavigationProvider class."""

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

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton instances between tests."""
        try:
            from providers.unitree_go2_navigation_provider import (
                UnitreeGo2NavigationProvider,
            )

            if hasattr(UnitreeGo2NavigationProvider, "reset"):
                UnitreeGo2NavigationProvider.reset()
        except ImportError:
            pass
        yield
        try:
            from providers.unitree_go2_navigation_provider import (
                UnitreeGo2NavigationProvider,
            )

            if hasattr(UnitreeGo2NavigationProvider, "reset"):
                UnitreeGo2NavigationProvider.reset()
        except ImportError:
            pass

    @pytest.fixture
    def mock_zenoh_session(self):
        """Mock zenoh session."""
        mock_session = MagicMock()
        mock_session.declare_publisher.return_value = MagicMock()
        mock_session.declare_subscriber.return_value = MagicMock()
        return mock_session

    @pytest.fixture
    def mock_dependencies(self, mock_zenoh_session):
        """Mock all dependencies."""
        with (
            patch(
                "providers.unitree_go2_navigation_provider.open_zenoh_session",
                return_value=mock_zenoh_session,
            ),
            patch(
                "providers.unitree_go2_navigation_provider.ElevenLabsTTSProvider"
            ) as mock_tts,
            patch("providers.unitree_go2_navigation_provider.logging") as mock_logging,
        ):
            mock_tts_instance = MagicMock()
            mock_tts.return_value = mock_tts_instance
            yield {
                "session": mock_zenoh_session,
                "tts": mock_tts_instance,
                "logging": mock_logging,
            }

    def test_initialization_with_defaults(self, mock_dependencies):
        """Test provider initializes correctly with default parameters."""
        from providers.unitree_go2_navigation_provider import (
            UnitreeGo2NavigationProvider,
        )

        provider = UnitreeGo2NavigationProvider()

        assert provider is not None
        assert provider.navigation_status_topic == "navigate_to_pose/_action/status"
        assert provider.goal_pose_topic == "goal_pose"
        assert provider.cancel_goal_topic == "navigate_to_pose/_action/cancel_goal"
        assert provider.navigation_status == "UNKNOWN"
        assert provider.running is False
        assert provider._nav_in_progress is False
        assert provider._current_destination is None
        assert provider.ai_status_topic == "om/ai/request"

    def test_initialization_with_custom_topics(self, mock_dependencies):
        """Test provider initializes correctly with custom topics."""
        from providers.unitree_go2_navigation_provider import (
            UnitreeGo2NavigationProvider,
        )

        provider = UnitreeGo2NavigationProvider(
            navigation_status_topic="custom_status",
            goal_pose_topic="custom_goal",
            cancel_goal_topic="custom_cancel",
        )

        assert provider.navigation_status_topic == "custom_status"
        assert provider.goal_pose_topic == "custom_goal"
        assert provider.cancel_goal_topic == "custom_cancel"

    def test_initialization_zenoh_session_failure(self):
        """Test provider handles zenoh session initialization failure gracefully."""
        with patch(
            "providers.unitree_go2_navigation_provider.open_zenoh_session",
            side_effect=Exception("Connection failed"),
        ):
            from providers.unitree_go2_navigation_provider import (
                UnitreeGo2NavigationProvider,
            )

            provider = UnitreeGo2NavigationProvider()

            assert provider.session is None
            assert provider.ai_status_pub is None

    def test_singleton_pattern(self, mock_dependencies):
        """Test provider implements singleton pattern correctly."""
        from providers.unitree_go2_navigation_provider import (
            UnitreeGo2NavigationProvider,
        )

        provider1 = UnitreeGo2NavigationProvider()
        provider2 = UnitreeGo2NavigationProvider()

        assert provider1 is provider2

    def test_navigation_status_message_callback_accepted_status(
        self, mock_dependencies
    ):
        """Test navigation status callback handles ACCEPTED status correctly."""
        from providers.unitree_go2_navigation_provider import (
            UnitreeGo2NavigationProvider,
        )

        provider = UnitreeGo2NavigationProvider()

        # Mock the zenoh sample and nav status
        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"test_data"
        mock_sample.payload = mock_payload

        mock_nav_status = MagicMock()
        mock_status_item = MagicMock()
        mock_status_item.status = 1  # ACCEPTED
        mock_nav_status.status_list = [mock_status_item]

        with patch(
            "providers.unitree_go2_navigation_provider.nav_msgs.Nav2Status.deserialize",
            return_value=mock_nav_status,
        ):
            with patch.object(provider, "_publish_ai_status") as mock_publish_ai:
                provider.navigation_status_message_callback(mock_sample)

                assert provider.navigation_status == "ACCEPTED"
                assert provider._nav_in_progress is True
                mock_publish_ai.assert_called_once_with(enabled=False)

    def test_navigation_status_message_callback_executing_status(
        self, mock_dependencies
    ):
        """Test navigation status callback handles EXECUTING status correctly."""
        from providers.unitree_go2_navigation_provider import (
            UnitreeGo2NavigationProvider,
        )

        provider = UnitreeGo2NavigationProvider()

        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"test_data"
        mock_sample.payload = mock_payload

        mock_nav_status = MagicMock()
        mock_status_item = MagicMock()
        mock_status_item.status = 2  # EXECUTING
        mock_nav_status.status_list = [mock_status_item]

        with patch(
            "providers.unitree_go2_navigation_provider.nav_msgs.Nav2Status.deserialize",
            return_value=mock_nav_status,
        ):
            with patch.object(provider, "_publish_ai_status") as mock_publish_ai:
                provider.navigation_status_message_callback(mock_sample)

                assert provider.navigation_status == "EXECUTING"
                assert provider._nav_in_progress is True
                mock_publish_ai.assert_called_once_with(enabled=False)

    def test_navigation_status_message_callback_succeeded_status(
        self, mock_dependencies
    ):
        """Test navigation status callback handles SUCCEEDED status correctly."""
        from providers.unitree_go2_navigation_provider import (
            UnitreeGo2NavigationProvider,
        )

        provider = UnitreeGo2NavigationProvider()
        provider._nav_in_progress = True  # Set navigation in progress

        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"test_data"
        mock_sample.payload = mock_payload

        mock_nav_status = MagicMock()
        mock_status_item = MagicMock()
        mock_status_item.status = 4  # SUCCEEDED
        mock_nav_status.status_list = [mock_status_item]

        with patch(
            "providers.unitree_go2_navigation_provider.nav_msgs.Nav2Status.deserialize",
            return_value=mock_nav_status,
        ):
            with patch.object(provider, "_publish_ai_status") as mock_publish_ai:
                provider.navigation_status_message_callback(mock_sample)

                assert provider.navigation_status == "SUCCEEDED"
                assert provider._nav_in_progress is False
                mock_publish_ai.assert_called_once_with(enabled=True)

    def test_navigation_status_message_callback_unknown_status(self, mock_dependencies):
        """Test navigation status callback handles unknown status codes."""
        from providers.unitree_go2_navigation_provider import (
            UnitreeGo2NavigationProvider,
        )

        provider = UnitreeGo2NavigationProvider()

        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"test_data"
        mock_sample.payload = mock_payload

        mock_nav_status = MagicMock()
        mock_status_item = MagicMock()
        mock_status_item.status = 999  # Unknown status
        mock_nav_status.status_list = [mock_status_item]

        with patch(
            "providers.unitree_go2_navigation_provider.nav_msgs.Nav2Status.deserialize",
            return_value=mock_nav_status,
        ):
            provider.navigation_status_message_callback(mock_sample)

            assert provider.navigation_status == "UNKNOWN"

    def test_navigation_status_message_callback_no_payload(self, mock_dependencies):
        """Test navigation status callback handles sample with no payload."""
        from providers.unitree_go2_navigation_provider import (
            UnitreeGo2NavigationProvider,
        )

        provider = UnitreeGo2NavigationProvider()
        original_status = provider.navigation_status

        mock_sample = MagicMock()
        mock_sample.payload = None

        provider.navigation_status_message_callback(mock_sample)

        # Status should remain unchanged
        assert provider.navigation_status == original_status

    def test_navigation_status_message_callback_empty_status_list(
        self, mock_dependencies
    ):
        """Test navigation status callback handles empty status list."""
        from providers.unitree_go2_navigation_provider import (
            UnitreeGo2NavigationProvider,
        )

        provider = UnitreeGo2NavigationProvider()
        original_status = provider.navigation_status

        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"test_data"
        mock_sample.payload = mock_payload

        mock_nav_status = MagicMock()
        mock_nav_status.status_list = []

        with patch(
            "providers.unitree_go2_navigation_provider.nav_msgs.Nav2Status.deserialize",
            return_value=mock_nav_status,
        ):
            provider.navigation_status_message_callback(mock_sample)

            # Status should remain unchanged
            assert provider.navigation_status == original_status

    def test_navigation_status_message_callback_no_duplicate_ai_disable(
        self, mock_dependencies
    ):
        """Test navigation status callback doesn't disable AI multiple times."""
        from providers.unitree_go2_navigation_provider import (
            UnitreeGo2NavigationProvider,
        )

        provider = UnitreeGo2NavigationProvider()
        provider._nav_in_progress = True  # Already in progress

        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"test_data"
        mock_sample.payload = mock_payload

        mock_nav_status = MagicMock()
        mock_status_item = MagicMock()
        mock_status_item.status = 1  # ACCEPTED
        mock_nav_status.status_list = [mock_status_item]

        with patch(
            "providers.unitree_go2_navigation_provider.nav_msgs.Nav2Status.deserialize",
            return_value=mock_nav_status,
        ):
            with patch.object(provider, "_publish_ai_status") as mock_publish_ai:
                provider.navigation_status_message_callback(mock_sample)

                # Should not call publish_ai_status since already in progress
                mock_publish_ai.assert_not_called()

    def test_navigation_status_message_callback_no_ai_enable_when_not_in_progress(
        self, mock_dependencies
    ):
        """Test navigation status callback doesn't enable AI when not in progress."""
        from providers.unitree_go2_navigation_provider import (
            UnitreeGo2NavigationProvider,
        )

        provider = UnitreeGo2NavigationProvider()
        provider._nav_in_progress = False  # Not in progress

        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"test_data"
        mock_sample.payload = mock_payload

        mock_nav_status = MagicMock()
        mock_status_item = MagicMock()
        mock_status_item.status = 4  # SUCCEEDED
        mock_nav_status.status_list = [mock_status_item]

        with patch(
            "providers.unitree_go2_navigation_provider.nav_msgs.Nav2Status.deserialize",
            return_value=mock_nav_status,
        ):
            with patch.object(provider, "_publish_ai_status") as mock_publish_ai:
                provider.navigation_status_message_callback(mock_sample)

                # Should not call publish_ai_status since not in progress
                mock_publish_ai.assert_not_called()

    def test_ai_status_publisher_creation_failure(self):
        """Test provider handles AI status publisher creation failure gracefully."""
        mock_session = MagicMock()
        mock_session.declare_publisher.side_effect = Exception(
            "Publisher creation failed"
        )

        with patch(
            "providers.unitree_go2_navigation_provider.open_zenoh_session",
            return_value=mock_session,
        ):
            from providers.unitree_go2_navigation_provider import (
                UnitreeGo2NavigationProvider,
            )

            provider = UnitreeGo2NavigationProvider()

            assert provider.ai_status_pub is None

    def test_status_map_coverage(self, mock_dependencies):
        """Test all status codes in status_map are handled correctly."""
        from providers.unitree_go2_navigation_provider import (
            UnitreeGo2NavigationProvider,
            status_map,
        )

        provider = UnitreeGo2NavigationProvider()

        for status_code, expected_status in status_map.items():
            mock_sample = MagicMock()
            mock_payload = MagicMock()
            mock_payload.to_bytes.return_value = b"test_data"
            mock_sample.payload = mock_payload

            mock_nav_status = MagicMock()
            mock_status_item = MagicMock()
            mock_status_item.status = status_code
            mock_nav_status.status_list = [mock_status_item]

            with patch(
                "providers.unitree_go2_navigation_provider.nav_msgs.Nav2Status.deserialize",
                return_value=mock_nav_status,
            ):
                provider.navigation_status_message_callback(mock_sample)

                assert provider.navigation_status == expected_status
