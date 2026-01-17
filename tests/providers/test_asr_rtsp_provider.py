"""Tests for asr_rtsp_provider."""

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
sys.modules["pyrealsense2"] = MagicMock()
sys.modules["om1_speech"] = MagicMock()
sys.modules["om1_utils"] = MagicMock()
sys.modules["om1_utils.ws"] = MagicMock()


class TestASRRTSPProvider:
    """Tests for ASRRTSPProvider class."""

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
        """Provide websocket URL for testing."""
        return "ws://test.url"

    @pytest.fixture
    def rtsp_url(self):
        """Provide RTSP URL for testing."""
        return "rtsp://test.server:8554/audio"

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies."""
        with (
            patch("providers.asr_rtsp_provider.ws.Client") as mock_ws_client,
            patch(
                "providers.asr_rtsp_provider.AudioRTSPInputStream"
            ) as mock_audio_stream,
        ):
            yield mock_ws_client, mock_audio_stream

    def test_initialization_with_default_parameters(self, ws_url, mock_dependencies):
        """Test provider initializes correctly with default parameters."""
        from providers.asr_rtsp_provider import ASRRTSPProvider

        if hasattr(ASRRTSPProvider, "reset"):
            ASRRTSPProvider.reset()

        mock_ws_client, mock_audio_stream = mock_dependencies
        provider = ASRRTSPProvider(ws_url)

        assert provider is not None
        assert not provider.running
        mock_ws_client.assert_called_once_with(url=ws_url)
        mock_audio_stream.assert_called_once()

    def test_initialization_with_custom_parameters(
        self, ws_url, rtsp_url, mock_dependencies
    ):
        """Test provider initializes correctly with custom parameters."""
        from providers.asr_rtsp_provider import ASRRTSPProvider

        if hasattr(ASRRTSPProvider, "reset"):
            ASRRTSPProvider.reset()

        mock_ws_client, mock_audio_stream = mock_dependencies
        provider = ASRRTSPProvider(
            ws_url=ws_url,
            rtsp_url=rtsp_url,
            rate=44100,
            chunk=1024,
            language_code="en-US",
            enable_tts_interrupt=True,
        )

        assert provider is not None
        assert not provider.running
        mock_ws_client.assert_called_once_with(url=ws_url)

    def test_singleton_behavior(self, ws_url, mock_dependencies):
        """Test singleton pattern works correctly."""
        from providers.asr_rtsp_provider import ASRRTSPProvider

        if hasattr(ASRRTSPProvider, "reset"):
            ASRRTSPProvider.reset()

        mock_ws_client, mock_audio_stream = mock_dependencies
        provider1 = ASRRTSPProvider(ws_url)
        provider2 = ASRRTSPProvider(ws_url)

        assert provider1 is provider2

    def test_register_message_callback_with_valid_callback(
        self, ws_url, mock_dependencies
    ):
        """Test registering a valid message callback."""
        from providers.asr_rtsp_provider import ASRRTSPProvider

        if hasattr(ASRRTSPProvider, "reset"):
            ASRRTSPProvider.reset()

        mock_ws_client, mock_audio_stream = mock_dependencies
        provider = ASRRTSPProvider(ws_url)
        callback = Mock()

        provider.register_message_callback(callback)

        mock_ws_client.return_value.register_message_callback.assert_called_once_with(
            callback
        )

    def test_register_message_callback_with_none(self, ws_url, mock_dependencies):
        """Test registering None as message callback."""
        from providers.asr_rtsp_provider import ASRRTSPProvider

        if hasattr(ASRRTSPProvider, "reset"):
            ASRRTSPProvider.reset()

        mock_ws_client, mock_audio_stream = mock_dependencies
        provider = ASRRTSPProvider(ws_url)

        provider.register_message_callback(None)

        # Should not call register_message_callback when None is passed

    def test_start_when_not_running(self, ws_url, mock_dependencies, caplog):
        """Test starting the provider when not already running."""
        from providers.asr_rtsp_provider import ASRRTSPProvider

        if hasattr(ASRRTSPProvider, "reset"):
            ASRRTSPProvider.reset()

        mock_ws_client, mock_audio_stream = mock_dependencies
        provider = ASRRTSPProvider(ws_url)

        with caplog.at_level(logging.INFO):
            provider.start()

        assert provider.running
        mock_ws_client.return_value.start.assert_called_once()
        mock_audio_stream.return_value.start.assert_called_once()
        assert "ASR RTSP provider started" in caplog.text

    def test_start_when_already_running(self, ws_url, mock_dependencies, caplog):
        """Test starting the provider when already running."""
        from providers.asr_rtsp_provider import ASRRTSPProvider

        if hasattr(ASRRTSPProvider, "reset"):
            ASRRTSPProvider.reset()

        mock_ws_client, mock_audio_stream = mock_dependencies
        provider = ASRRTSPProvider(ws_url)
        provider.running = True

        with caplog.at_level(logging.WARNING):
            provider.start()

        assert provider.running
        assert "ASR RTSP provider is already running" in caplog.text

    def test_stop_when_running(self, ws_url, mock_dependencies, caplog):
        """Test stopping the provider when running."""
        from providers.asr_rtsp_provider import ASRRTSPProvider

        if hasattr(ASRRTSPProvider, "reset"):
            ASRRTSPProvider.reset()

        mock_ws_client, mock_audio_stream = mock_dependencies
        provider = ASRRTSPProvider(ws_url)
        provider.running = True

        provider.stop()

        assert not provider.running
        mock_audio_stream.return_value.stop.assert_called_once()
        mock_ws_client.return_value.stop.assert_called_once()

    def test_stop_when_not_running(self, ws_url, mock_dependencies, caplog):
        """Test stopping the provider when not running."""
        from providers.asr_rtsp_provider import ASRRTSPProvider

        if hasattr(ASRRTSPProvider, "reset"):
            ASRRTSPProvider.reset()

        mock_ws_client, mock_audio_stream = mock_dependencies
        provider = ASRRTSPProvider(ws_url)
        provider.running = False

        with caplog.at_level(logging.WARNING):
            provider.stop()

        assert not provider.running
        assert "ASR RTSP provider is not running" in caplog.text

    def test_start_stop_cycle(self, ws_url, mock_dependencies):
        """Test complete start-stop cycle."""
        from providers.asr_rtsp_provider import ASRRTSPProvider

        if hasattr(ASRRTSPProvider, "reset"):
            ASRRTSPProvider.reset()

        mock_ws_client, mock_audio_stream = mock_dependencies
        provider = ASRRTSPProvider(ws_url)

        # Start
        provider.start()
        assert provider.running

        # Stop
        provider.stop()
        assert not provider.running

    def test_audio_stream_callback_integration(self, ws_url, mock_dependencies):
        """Test that audio stream callback is properly connected to websocket client."""
        from providers.asr_rtsp_provider import ASRRTSPProvider

        if hasattr(ASRRTSPProvider, "reset"):
            ASRRTSPProvider.reset()

        mock_ws_client, mock_audio_stream = mock_dependencies
        provider = ASRRTSPProvider(ws_url)

        # Verify that AudioRTSPInputStream was called with ws_client.send_message as callback
        call_args = mock_audio_stream.call_args
        assert call_args is not None
        assert "audio_data_callback" in call_args.kwargs
        assert (
            call_args.kwargs["audio_data_callback"]
            == mock_ws_client.return_value.send_message
        )
