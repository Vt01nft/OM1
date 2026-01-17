"""Tests for vlm_vila_zenoh_provider."""

import sys
from unittest.mock import MagicMock, patch

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
sys.modules["om1_utils"] = MagicMock()
sys.modules["om1_utils.ws"] = MagicMock()
sys.modules["om1_vlm"] = MagicMock()


class TestVLMVilaZenohProvider:
    """Tests for VLMVilaZenohProvider class."""

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

    def test_initialization(self):
        """Test provider initializes correctly with default parameters."""
        with (
            patch("providers.vlm_vila_zenoh_provider.ws.Client") as mock_ws_client,
            patch(
                "providers.vlm_vila_zenoh_provider.VideoZenohStream"
            ) as mock_video_stream,
        ):

            from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider

            if hasattr(VLMVilaZenohProvider, "reset"):
                VLMVilaZenohProvider.reset()

            ws_url = "ws://test.url"
            provider = VLMVilaZenohProvider(ws_url)

            assert provider is not None
            assert not provider.running
            mock_ws_client.assert_called_once_with(url=ws_url)
            mock_video_stream.assert_called_once()

    def test_initialization_with_custom_parameters(self):
        """Test provider initializes correctly with custom parameters."""
        with (
            patch("providers.vlm_vila_zenoh_provider.ws.Client") as mock_ws_client,
            patch(
                "providers.vlm_vila_zenoh_provider.VideoZenohStream"
            ) as mock_video_stream,
        ):

            from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider

            if hasattr(VLMVilaZenohProvider, "reset"):
                VLMVilaZenohProvider.reset()

            ws_url = "ws://test.url"
            topic = "custom_topic"
            decode_format = "VP9"

            provider = VLMVilaZenohProvider(ws_url, topic, decode_format)

            assert provider is not None
            assert not provider.running
            mock_ws_client.assert_called_once_with(url=ws_url)
            mock_video_stream.assert_called_once_with(
                topic,
                decode_format,
                frame_callback=mock_ws_client.return_value.send_message,
            )

    def test_singleton_behavior(self):
        """Test singleton pattern works correctly."""
        with (
            patch("providers.vlm_vila_zenoh_provider.ws.Client"),
            patch("providers.vlm_vila_zenoh_provider.VideoZenohStream"),
        ):

            from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider

            if hasattr(VLMVilaZenohProvider, "reset"):
                VLMVilaZenohProvider.reset()

            ws_url = "ws://test.url"
            provider1 = VLMVilaZenohProvider(ws_url)
            provider2 = VLMVilaZenohProvider(ws_url)

            assert provider1 is provider2

    def test_register_frame_callback_with_valid_callback(self):
        """Test registering a valid frame callback."""
        with (
            patch("providers.vlm_vila_zenoh_provider.ws.Client"),
            patch(
                "providers.vlm_vila_zenoh_provider.VideoZenohStream"
            ) as mock_video_stream,
        ):

            from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider

            if hasattr(VLMVilaZenohProvider, "reset"):
                VLMVilaZenohProvider.reset()

            ws_url = "ws://test.url"
            provider = VLMVilaZenohProvider(ws_url)
            callback = MagicMock()

            provider.register_frame_callback(callback)

            mock_video_stream.return_value.register_frame_callback.assert_called_once_with(
                callback
            )

    def test_register_frame_callback_with_none(self):
        """Test registering a None frame callback does nothing."""
        with (
            patch("providers.vlm_vila_zenoh_provider.ws.Client"),
            patch(
                "providers.vlm_vila_zenoh_provider.VideoZenohStream"
            ) as mock_video_stream,
        ):

            from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider

            if hasattr(VLMVilaZenohProvider, "reset"):
                VLMVilaZenohProvider.reset()

            ws_url = "ws://test.url"
            provider = VLMVilaZenohProvider(ws_url)

            provider.register_frame_callback(None)

            mock_video_stream.return_value.register_frame_callback.assert_not_called()

    def test_register_message_callback_with_valid_callback(self):
        """Test registering a valid message callback."""
        with (
            patch("providers.vlm_vila_zenoh_provider.ws.Client") as mock_ws_client,
            patch("providers.vlm_vila_zenoh_provider.VideoZenohStream"),
        ):

            from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider

            if hasattr(VLMVilaZenohProvider, "reset"):
                VLMVilaZenohProvider.reset()

            ws_url = "ws://test.url"
            provider = VLMVilaZenohProvider(ws_url)
            callback = MagicMock()

            provider.register_message_callback(callback)

            mock_ws_client.return_value.register_message_callback.assert_called_once_with(
                callback
            )

    def test_register_message_callback_with_none(self):
        """Test registering a None message callback does nothing."""
        with (
            patch("providers.vlm_vila_zenoh_provider.ws.Client") as mock_ws_client,
            patch("providers.vlm_vila_zenoh_provider.VideoZenohStream"),
        ):

            from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider

            if hasattr(VLMVilaZenohProvider, "reset"):
                VLMVilaZenohProvider.reset()

            ws_url = "ws://test.url"
            provider = VLMVilaZenohProvider(ws_url)

            provider.register_message_callback(None)

            mock_ws_client.return_value.register_message_callback.assert_not_called()

    def test_start_when_not_running(self):
        """Test starting the provider when not running."""
        with (
            patch("providers.vlm_vila_zenoh_provider.ws.Client") as mock_ws_client,
            patch(
                "providers.vlm_vila_zenoh_provider.VideoZenohStream"
            ) as mock_video_stream,
        ):

            from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider

            if hasattr(VLMVilaZenohProvider, "reset"):
                VLMVilaZenohProvider.reset()

            ws_url = "ws://test.url"
            provider = VLMVilaZenohProvider(ws_url)

            provider.start()

            assert provider.running
            mock_ws_client.return_value.start.assert_called_once()
            mock_video_stream.return_value.start.assert_called_once()

    def test_start_when_already_running(self):
        """Test starting the provider when already running logs warning."""
        with (
            patch("providers.vlm_vila_zenoh_provider.ws.Client") as mock_ws_client,
            patch(
                "providers.vlm_vila_zenoh_provider.VideoZenohStream"
            ) as mock_video_stream,
            patch("providers.vlm_vila_zenoh_provider.logging") as mock_logging,
        ):

            from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider

            if hasattr(VLMVilaZenohProvider, "reset"):
                VLMVilaZenohProvider.reset()

            ws_url = "ws://test.url"
            provider = VLMVilaZenohProvider(ws_url)
            provider.running = True

            provider.start()

            mock_logging.warning.assert_called_once_with(
                "VLM Zenoh provider is already running"
            )

    def test_stop(self):
        """Test stopping the provider."""
        with (
            patch("providers.vlm_vila_zenoh_provider.ws.Client") as mock_ws_client,
            patch(
                "providers.vlm_vila_zenoh_provider.VideoZenohStream"
            ) as mock_video_stream,
        ):

            from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider

            if hasattr(VLMVilaZenohProvider, "reset"):
                VLMVilaZenohProvider.reset()

            ws_url = "ws://test.url"
            provider = VLMVilaZenohProvider(ws_url)
            provider.running = True

            provider.stop()

            assert not provider.running
            mock_video_stream.return_value.stop.assert_called_once()
            mock_ws_client.return_value.stop.assert_called_once()

    def test_start_stop_cycle(self):
        """Test starting and stopping the provider in sequence."""
        with (
            patch("providers.vlm_vila_zenoh_provider.ws.Client") as mock_ws_client,
            patch(
                "providers.vlm_vila_zenoh_provider.VideoZenohStream"
            ) as mock_video_stream,
        ):

            from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider

            if hasattr(VLMVilaZenohProvider, "reset"):
                VLMVilaZenohProvider.reset()

            ws_url = "ws://test.url"
            provider = VLMVilaZenohProvider(ws_url)

            # Start
            provider.start()
            assert provider.running
            mock_ws_client.return_value.start.assert_called_once()
            mock_video_stream.return_value.start.assert_called_once()

            # Stop
            provider.stop()
            assert not provider.running
            mock_video_stream.return_value.stop.assert_called_once()
            mock_ws_client.return_value.stop.assert_called_once()
