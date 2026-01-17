"""Tests for avatar_provider."""

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


class TestAvatarProvider:
    """Tests for AvatarProvider class."""

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

    @pytest.fixture
    def mock_zenoh_dependencies(self):
        """Mock zenoh and related dependencies."""
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_subscriber = MagicMock()

        mock_session.declare_publisher.return_value = mock_publisher
        mock_session.declare_subscriber.return_value = mock_subscriber

        mock_avatar_request = MagicMock()
        mock_avatar_response = MagicMock()
        mock_string = MagicMock()

        with (
            patch(
                "providers.avatar_provider.open_zenoh_session",
                return_value=mock_session,
            ),
            patch("providers.avatar_provider.prepare_header", return_value={}),
            patch("providers.avatar_provider.AvatarFaceRequest", mock_avatar_request),
            patch("providers.avatar_provider.AvatarFaceResponse", mock_avatar_response),
            patch("providers.avatar_provider.String", mock_string),
        ):
            yield {
                "session": mock_session,
                "publisher": mock_publisher,
                "subscriber": mock_subscriber,
                "avatar_request": mock_avatar_request,
                "avatar_response": mock_avatar_response,
                "string": mock_string,
            }

    def test_initialization(self, mock_zenoh_dependencies):
        """Test provider initializes correctly."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider = AvatarProvider()

        assert provider is not None
        assert provider.running is True
        assert provider.session is not None
        assert provider.avatar_publisher is not None
        assert provider.avatar_healthcheck_publisher is not None
        assert provider.avatar_subscriber is not None

    def test_initialization_failure(self):
        """Test provider handles initialization failure gracefully."""
        with patch(
            "providers.avatar_provider.open_zenoh_session",
            side_effect=Exception("Connection failed"),
        ):
            from providers.avatar_provider import AvatarProvider

            if hasattr(AvatarProvider, "reset"):
                AvatarProvider.reset()

            provider = AvatarProvider()
            assert provider is not None

    def test_singleton_pattern(self, mock_zenoh_dependencies):
        """Test provider follows singleton pattern."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider1 = AvatarProvider()
        provider2 = AvatarProvider()
        assert provider1 is provider2

    def test_send_avatar_command_success(self, mock_zenoh_dependencies):
        """Test sending avatar command successfully."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider = AvatarProvider()
        result = provider.send_avatar_command("Happy")

        assert result is True

    def test_send_avatar_command_lowercase_conversion(self, mock_zenoh_dependencies):
        """Test avatar command is converted to lowercase."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider = AvatarProvider()
        result = provider.send_avatar_command("HAPPY")

        assert result is True

    def test_send_avatar_command_not_running(self, mock_zenoh_dependencies):
        """Test sending command when provider is not running."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider = AvatarProvider()
        provider.running = False

        result = provider.send_avatar_command("Happy")
        assert result is False

    def test_send_avatar_command_no_publisher(self, mock_zenoh_dependencies):
        """Test sending command when publisher is None."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider = AvatarProvider()
        provider.avatar_publisher = None

        result = provider.send_avatar_command("Happy")
        assert result is False

    def test_send_avatar_command_exception(self, mock_zenoh_dependencies):
        """Test sending command handles exceptions gracefully."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider = AvatarProvider()
        mock_zenoh_dependencies["publisher"].put.side_effect = Exception("Send failed")

        result = provider.send_avatar_command("Happy")
        assert result is False

    def test_handle_avatar_request_status(self, mock_zenoh_dependencies):
        """Test handling status request."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider = AvatarProvider()

        mock_sample = MagicMock()
        mock_sample.payload.to_bytes.return_value = b"test_payload"

        mock_request = MagicMock()
        mock_request.code = 1  # STATUS value
        mock_request.request_id = "test_id"

        mock_zenoh_dependencies["avatar_request"].deserialize.return_value = (
            mock_request
        )
        mock_zenoh_dependencies["avatar_request"].Code.STATUS.value = 1

        # Should not raise exception
        provider._handle_avatar_request(mock_sample)

    def test_handle_avatar_request_switch_face(self, mock_zenoh_dependencies):
        """Test handling switch face request (ignored)."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider = AvatarProvider()

        mock_sample = MagicMock()
        mock_sample.payload.to_bytes.return_value = b"test_payload"

        mock_request = MagicMock()
        mock_request.code = 2  # SWITCH_FACE value
        mock_request.request_id = "test_id"

        mock_zenoh_dependencies["avatar_request"].deserialize.return_value = (
            mock_request
        )
        mock_zenoh_dependencies["avatar_request"].Code.STATUS.value = 1

        # Should not raise exception
        provider._handle_avatar_request(mock_sample)

    def test_handle_avatar_request_exception(self, mock_zenoh_dependencies):
        """Test handling avatar request with exception."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider = AvatarProvider()

        mock_sample = MagicMock()
        mock_sample.payload.to_bytes.side_effect = Exception("Deserialization failed")

        # Should not raise exception
        provider._handle_avatar_request(mock_sample)

    def test_stop_when_running(self, mock_zenoh_dependencies):
        """Test stopping provider when it's running."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider = AvatarProvider()
        assert provider.running is True

        provider.stop()

        assert provider.running is False
        mock_zenoh_dependencies["session"].close.assert_called_once()

    def test_stop_when_not_running(self, mock_zenoh_dependencies):
        """Test stopping provider when it's not running."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider = AvatarProvider()
        provider.running = False

        # Should not raise exception
        provider.stop()
        assert provider.running is False

    def test_stop_with_no_session(self, mock_zenoh_dependencies):
        """Test stopping provider when session is None."""
        from providers.avatar_provider import AvatarProvider

        if hasattr(AvatarProvider, "reset"):
            AvatarProvider.reset()

        provider = AvatarProvider()
        provider.session = None

        # Should not raise exception
        provider.stop()
        assert provider.running is False
