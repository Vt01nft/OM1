"""Tests for vlm_vila_rtsp_provider."""

import logging
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock ALL external dependencies BEFORE any provider imports
# This must happen at module load time
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
sys.modules["om1_utils"] = MagicMock()
sys.modules["om1_utils.ws"] = MagicMock()
sys.modules["om1_vlm"] = MagicMock()


class TestVLMVilaRTSPProvider:
    """Tests for VLMVilaRTSPProvider class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        # Clear cached provider modules to reset singletons
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        yield
        # Cleanup after test
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

    @pytest.fixture
    def ws_url(self):
        """Provide test websocket URL."""
        return "ws://test.url"

    @pytest.fixture
    def rtsp_url(self):
        """Provide test RTSP URL."""
        return "rtsp://test.url:8554/camera"

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies."""
        with (
            patch("providers.vlm_vila_rtsp_provider.ws.Client") as mock_ws_client,
            patch(
                "providers.vlm_vila_rtsp_provider.VideoRTSPStream"
            ) as mock_video_stream,
        ):
            yield mock_ws_client, mock_video_stream

    def test_initialization_with_defaults(self, ws_url, mock_dependencies):
        """Test provider initializes correctly with default parameters."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        mock_ws_client, mock_video_stream = mock_dependencies

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider = VLMVilaRTSPProvider(ws_url)

        assert provider is not None
        assert provider.running is False
        mock_ws_client.assert_called_once_with(url=ws_url)
        mock_video_stream.assert_called_once()

    def test_initialization_with_custom_parameters(
        self, ws_url, rtsp_url, mock_dependencies
    ):
        """Test provider initializes correctly with custom parameters."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        mock_ws_client, mock_video_stream = mock_dependencies

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider = VLMVilaRTSPProvider(
            ws_url=ws_url, rtsp_url=rtsp_url, decode_format="H265", fps=60
        )

        assert provider is not None
        assert provider.running is False
        mock_ws_client.assert_called_once_with(url=ws_url)

    def test_singleton_behavior(self, ws_url):
        """Test singleton pattern works correctly."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider1 = VLMVilaRTSPProvider(ws_url)
        provider2 = VLMVilaRTSPProvider(ws_url)

        assert provider1 is provider2

    def test_register_frame_callback_with_valid_callback(
        self, ws_url, mock_dependencies
    ):
        """Test registering frame callback with valid callback."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        mock_ws_client, mock_video_stream = mock_dependencies

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider = VLMVilaRTSPProvider(ws_url)
        callback = Mock()

        provider.register_frame_callback(callback)

        mock_video_stream.return_value.register_frame_callback.assert_called_once_with(
            callback
        )

    def test_register_frame_callback_with_none(self, ws_url, mock_dependencies):
        """Test registering frame callback with None callback."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        mock_ws_client, mock_video_stream = mock_dependencies

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider = VLMVilaRTSPProvider(ws_url)

        # This should not raise an exception
        provider.register_frame_callback(None)

    def test_register_message_callback_with_valid_callback(
        self, ws_url, mock_dependencies
    ):
        """Test registering message callback with valid callback."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        mock_ws_client, mock_video_stream = mock_dependencies

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider = VLMVilaRTSPProvider(ws_url)
        callback = Mock()

        provider.register_message_callback(callback)

        mock_ws_client.return_value.register_message_callback.assert_called_once_with(
            callback
        )

    def test_register_message_callback_with_none(self, ws_url, mock_dependencies):
        """Test registering message callback with None callback."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        mock_ws_client, mock_video_stream = mock_dependencies

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider = VLMVilaRTSPProvider(ws_url)

        # This should not raise an exception
        provider.register_message_callback(None)

    def test_start_when_not_running(self, ws_url, mock_dependencies):
        """Test starting provider when not running."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        mock_ws_client, mock_video_stream = mock_dependencies

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider = VLMVilaRTSPProvider(ws_url)

        provider.start()

        assert provider.running is True
        mock_ws_client.return_value.start.assert_called_once()
        mock_video_stream.return_value.start.assert_called_once()

    def test_start_when_already_running(self, ws_url, mock_dependencies, caplog):
        """Test starting provider when already running logs warning."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        mock_ws_client, mock_video_stream = mock_dependencies

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider = VLMVilaRTSPProvider(ws_url)
        provider.running = True

        with caplog.at_level(logging.WARNING):
            provider.start()

        assert "VLM RTSP provider is already running" in caplog.text

    def test_stop_when_running(self, ws_url, mock_dependencies):
        """Test stopping provider when running."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        mock_ws_client, mock_video_stream = mock_dependencies

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider = VLMVilaRTSPProvider(ws_url)
        provider.running = True

        provider.stop()

        assert provider.running is False
        mock_video_stream.return_value.stop.assert_called_once()
        mock_ws_client.return_value.stop.assert_called_once()

    def test_stop_when_not_running(self, ws_url, mock_dependencies):
        """Test stopping provider when not running."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        mock_ws_client, mock_video_stream = mock_dependencies

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider = VLMVilaRTSPProvider(ws_url)

        provider.stop()

        assert provider.running is False
        mock_video_stream.return_value.stop.assert_called_once()
        mock_ws_client.return_value.stop.assert_called_once()

    def test_start_stop_cycle(self, ws_url, mock_dependencies):
        """Test complete start-stop cycle."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        mock_ws_client, mock_video_stream = mock_dependencies

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider = VLMVilaRTSPProvider(ws_url)

        # Start
        provider.start()
        assert provider.running is True

        # Stop
        provider.stop()
        assert provider.running is False

    def test_video_stream_initialization_with_frame_callback(
        self, ws_url, mock_dependencies
    ):
        """Test video stream is initialized with websocket client send_message as frame callback."""
        from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider

        mock_ws_client, mock_video_stream = mock_dependencies

        if hasattr(VLMVilaRTSPProvider, "reset"):
            VLMVilaRTSPProvider.reset()

        provider = VLMVilaRTSPProvider(
            ws_url, rtsp_url="rtsp://test.url", decode_format="H264", fps=30
        )

        # Check that VideoRTSPStream was called with correct parameters
        call_args = mock_video_stream.call_args
        assert call_args[0][0] == "rtsp://test.url"  # rtsp_url
        assert call_args[0][1] == "H264"  # decode_format
        assert (
            call_args[1]["frame_callback"] == mock_ws_client.return_value.send_message
        )
        assert call_args[1]["fps"] == 30
