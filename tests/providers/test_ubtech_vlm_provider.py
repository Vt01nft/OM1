"""
Unit tests for UbtechVLMProvider.

This module contains comprehensive tests for the UbtechVLMProvider class,
testing initialization, WebSocket communication, video streaming,
and singleton behavior.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock external dependencies before any imports
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
sys.modules["sensor_msgs"] = MagicMock()
sys.modules["geometry_msgs"] = MagicMock()
sys.modules["nav_msgs"] = MagicMock()
sys.modules["std_msgs"] = MagicMock()
sys.modules["om1_utils"] = MagicMock()
sys.modules["om1_utils.ws"] = MagicMock()

with patch.dict(
    "sys.modules",
    {"providers.singleton": MagicMock(), "providers.ubtech_video_stream": MagicMock()},
):
    from providers.ubtech_vlm_provider import UbtechVLMProvider


class TestUbtechVLMProvider:
    """Tests for UbtechVLMProvider."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        UbtechVLMProvider._instances = {}
        yield
        UbtechVLMProvider._instances = {}

    @pytest.fixture
    def mock_ws_client(self):
        """Create a mock WebSocket client."""
        mock_client = Mock()
        mock_client.send_message = Mock()
        mock_client.start = Mock()
        mock_client.stop = Mock()
        mock_client.register_message_callback = Mock()
        return mock_client

    @pytest.fixture
    def mock_video_stream(self):
        """Create a mock video stream."""
        mock_stream = Mock()
        mock_stream.start = Mock()
        mock_stream.stop = Mock()
        mock_stream.register_frame_callback = Mock()
        return mock_stream

    @pytest.fixture
    def provider_params(self):
        """Default parameters for provider initialization."""
        return {
            "ws_url": "ws://localhost:8080",
            "robot_ip": "192.168.1.100",
            "fps": 30,
            "resolution": (640, 480),
            "jpeg_quality": 70,
            "stream_url": None,
        }

    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_init_with_default_parameters(
        self, mock_ws_client_cls, mock_video_stream_cls, provider_params
    ):
        """Test initialization with default parameters."""
        mock_ws_client = Mock()
        mock_ws_client_cls.return_value = mock_ws_client
        mock_video_stream = Mock()
        mock_video_stream_cls.return_value = mock_video_stream

        provider = UbtechVLMProvider(**provider_params)

        assert provider.robot_ip == "192.168.1.100"
        assert provider.running is False
        assert provider.ws_client == mock_ws_client
        assert provider.stream_ws_client is None
        assert provider.video_stream == mock_video_stream

        mock_ws_client_cls.assert_called_once_with(url="ws://localhost:8080")
        mock_video_stream_cls.assert_called_once_with(
            frame_callback=mock_ws_client.send_message,
            fps=30,
            resolution=(640, 480),
            jpeg_quality=70,
            robot_ip="192.168.1.100",
        )

    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_init_with_stream_url(
        self, mock_ws_client_cls, mock_video_stream_cls, provider_params
    ):
        """Test initialization with stream URL provided."""
        provider_params["stream_url"] = "ws://localhost:8081"

        mock_ws_client = Mock()
        mock_stream_ws_client = Mock()
        mock_ws_client_cls.side_effect = [mock_ws_client, mock_stream_ws_client]
        mock_video_stream = Mock()
        mock_video_stream_cls.return_value = mock_video_stream

        provider = UbtechVLMProvider(**provider_params)

        assert provider.stream_ws_client == mock_stream_ws_client
        assert mock_ws_client_cls.call_count == 2

    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_init_with_custom_parameters(
        self, mock_ws_client_cls, mock_video_stream_cls
    ):
        """Test initialization with custom parameters."""
        custom_params = {
            "ws_url": "ws://custom:9090",
            "robot_ip": "10.0.0.5",
            "fps": 60,
            "resolution": (1280, 720),
            "jpeg_quality": 90,
            "stream_url": "ws://stream:9091",
        }

        mock_ws_client = Mock()
        mock_stream_ws_client = Mock()
        mock_ws_client_cls.side_effect = [mock_ws_client, mock_stream_ws_client]
        mock_video_stream = Mock()
        mock_video_stream_cls.return_value = mock_video_stream

        provider = UbtechVLMProvider(**custom_params)

        mock_video_stream_cls.assert_called_once_with(
            frame_callback=mock_ws_client.send_message,
            fps=60,
            resolution=(1280, 720),
            jpeg_quality=90,
            robot_ip="10.0.0.5",
        )

    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_register_message_callback_with_valid_callback(
        self, mock_ws_client_cls, mock_video_stream_cls, provider_params
    ):
        """Test registering a valid message callback."""
        mock_ws_client = Mock()
        mock_ws_client_cls.return_value = mock_ws_client
        mock_video_stream_cls.return_value = Mock()

        provider = UbtechVLMProvider(**provider_params)
        callback = Mock()

        provider.register_message_callback(callback)

        mock_ws_client.register_message_callback.assert_called_once_with(callback)

    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_register_message_callback_with_none(
        self, mock_ws_client_cls, mock_video_stream_cls, provider_params
    ):
        """Test registering None as message callback."""
        mock_ws_client = Mock()
        mock_ws_client_cls.return_value = mock_ws_client
        mock_video_stream_cls.return_value = Mock()

        provider = UbtechVLMProvider(**provider_params)

        provider.register_message_callback(None)

        mock_ws_client.register_message_callback.assert_not_called()

    @patch("providers.ubtech_vlm_provider.logging")
    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_start_when_not_running(
        self, mock_ws_client_cls, mock_video_stream_cls, mock_logging, provider_params
    ):
        """Test starting the provider when not already running."""
        mock_ws_client = Mock()
        mock_ws_client_cls.return_value = mock_ws_client
        mock_video_stream = Mock()
        mock_video_stream_cls.return_value = mock_video_stream

        provider = UbtechVLMProvider(**provider_params)
        provider.start()

        assert provider.running is True
        mock_ws_client.start.assert_called_once()
        mock_video_stream.start.assert_called_once()
        mock_logging.info.assert_called_once_with("Ubtech VLM provider started")

    @patch("providers.ubtech_vlm_provider.logging")
    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_start_when_already_running(
        self, mock_ws_client_cls, mock_video_stream_cls, mock_logging, provider_params
    ):
        """Test starting the provider when already running."""
        mock_ws_client = Mock()
        mock_ws_client_cls.return_value = mock_ws_client
        mock_video_stream = Mock()
        mock_video_stream_cls.return_value = mock_video_stream

        provider = UbtechVLMProvider(**provider_params)
        provider.running = True

        provider.start()

        mock_logging.warning.assert_called_once_with(
            "Ubtech VLM provider already running"
        )
        mock_ws_client.start.assert_not_called()
        mock_video_stream.start.assert_not_called()

    @patch("providers.ubtech_vlm_provider.logging")
    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_start_with_stream_client(
        self, mock_ws_client_cls, mock_video_stream_cls, mock_logging, provider_params
    ):
        """Test starting the provider with stream WebSocket client."""
        provider_params["stream_url"] = "ws://stream:8081"

        mock_ws_client = Mock()
        mock_stream_ws_client = Mock()
        mock_ws_client_cls.side_effect = [mock_ws_client, mock_stream_ws_client]
        mock_video_stream = Mock()
        mock_video_stream_cls.return_value = mock_video_stream

        provider = UbtechVLMProvider(**provider_params)
        provider.start()

        assert provider.running is True
        mock_ws_client.start.assert_called_once()
        mock_video_stream.start.assert_called_once()
        mock_stream_ws_client.start.assert_called_once()
        mock_video_stream.register_frame_callback.assert_called_once_with(
            mock_stream_ws_client.send_message
        )

    @patch("providers.ubtech_vlm_provider.logging")
    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_stop_without_stream_client(
        self, mock_ws_client_cls, mock_video_stream_cls, mock_logging, provider_params
    ):
        """Test stopping the provider without stream WebSocket client."""
        mock_ws_client = Mock()
        mock_ws_client_cls.return_value = mock_ws_client
        mock_video_stream = Mock()
        mock_video_stream_cls.return_value = mock_video_stream

        provider = UbtechVLMProvider(**provider_params)
        provider.running = True

        provider.stop()

        assert provider.running is False
        mock_video_stream.stop.assert_called_once()
        mock_ws_client.stop.assert_called_once()
        mock_logging.info.assert_called_once_with("Ubtech VLM provider stopped")

    @patch("providers.ubtech_vlm_provider.logging")
    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_stop_with_stream_client(
        self, mock_ws_client_cls, mock_video_stream_cls, mock_logging, provider_params
    ):
        """Test stopping the provider with stream WebSocket client."""
        provider_params["stream_url"] = "ws://stream:8081"

        mock_ws_client = Mock()
        mock_stream_ws_client = Mock()
        mock_ws_client_cls.side_effect = [mock_ws_client, mock_stream_ws_client]
        mock_video_stream = Mock()
        mock_video_stream_cls.return_value = mock_video_stream

        provider = UbtechVLMProvider(**provider_params)
        provider.running = True

        provider.stop()

        assert provider.running is False
        mock_video_stream.stop.assert_called_once()
        mock_ws_client.stop.assert_called_once()
        mock_stream_ws_client.stop.assert_called_once()

    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_singleton_behavior(
        self, mock_ws_client_cls, mock_video_stream_cls, provider_params
    ):
        """Test that the provider follows singleton pattern."""
        mock_ws_client_cls.return_value = Mock()
        mock_video_stream_cls.return_value = Mock()

        provider1 = UbtechVLMProvider(**provider_params)
        provider2 = UbtechVLMProvider(**provider_params)

        assert provider1 is provider2

    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_init_with_empty_strings(self, mock_ws_client_cls, mock_video_stream_cls):
        """Test initialization with empty string parameters."""
        params = {
            "ws_url": "",
            "robot_ip": "",
            "fps": 30,
            "resolution": (640, 480),
            "jpeg_quality": 70,
            "stream_url": "",
        }

        mock_ws_client = Mock()
        mock_stream_ws_client = Mock()
        mock_ws_client_cls.side_effect = [mock_ws_client, mock_stream_ws_client]
        mock_video_stream_cls.return_value = Mock()

        provider = UbtechVLMProvider(**params)

        assert provider.robot_ip == ""
        assert provider.stream_ws_client == mock_stream_ws_client

    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_init_with_edge_case_values(
        self, mock_ws_client_cls, mock_video_stream_cls
    ):
        """Test initialization with edge case values."""
        params = {
            "ws_url": "ws://test:1",
            "robot_ip": "0.0.0.0",
            "fps": 1,
            "resolution": (1, 1),
            "jpeg_quality": 1,
            "stream_url": None,
        }

        mock_ws_client_cls.return_value = Mock()
        mock_video_stream_cls.return_value = Mock()

        provider = UbtechVLMProvider(**params)

        assert provider.robot_ip == "0.0.0.0"
        assert provider.stream_ws_client is None

    @patch("providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("providers.ubtech_vlm_provider.ws.Client")
    def test_multiple_start_stop_cycles(
        self, mock_ws_client_cls, mock_video_stream_cls, provider_params
    ):
        """Test multiple start/stop cycles."""
        mock_ws_client = Mock()
        mock_ws_client_cls.return_value = mock_ws_client
        mock_video_stream = Mock()
        mock_video_stream_cls.return_value = mock_video_stream

        provider = UbtechVLMProvider(**provider_params)

        # First cycle
        provider.start()
        assert provider.running is True
        provider.stop()
        assert provider.running is False

        # Second cycle
        provider.start()
        assert provider.running is True
        provider.stop()
        assert provider.running is False

        assert mock_ws_client.start.call_count == 2
        assert mock_ws_client.stop.call_count == 2
        assert mock_video_stream.start.call_count == 2
        assert mock_video_stream.stop.call_count == 2
