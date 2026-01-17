"""Tests for zenoh_publisher_provider."""

import sys
import time
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


class TestZenohPublisherProvider:
    """Tests for ZenohPublisherProvider class."""

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

    def test_initialization_default_topic(self):
        """Test provider initializes with default topic."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session.return_value = MagicMock()

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()

            assert provider is not None
            assert provider.pub_topic == "speech"
            assert provider.running is False
            assert provider.session is not None
            mock_session.assert_called_once()

    def test_initialization_custom_topic(self):
        """Test provider initializes with custom topic."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session.return_value = MagicMock()

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider(topic="custom_topic")

            assert provider.pub_topic == "custom_topic"

    def test_initialization_session_failure(self):
        """Test provider handles session creation failure."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session.side_effect = Exception("Connection failed")

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()

            assert provider.session is None
            assert provider.pub_topic == "speech"
            assert provider.running is False

    def test_add_pending_message(self):
        """Test adding message to queue."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session.return_value = MagicMock()

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()

            test_text = "Hello, world!"
            provider.add_pending_message(test_text)

            assert not provider._pending_messages.empty()

    def test_start_provider(self):
        """Test starting the provider."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session.return_value = MagicMock()

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()
            provider.start()

            assert provider.running is True
            assert provider._thread is not None

    def test_start_provider_already_running(self):
        """Test starting provider when already running."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session.return_value = MagicMock()

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()
            provider.running = True
            original_thread = provider._thread

            provider.start()

            assert provider._thread == original_thread

    def test_stop_provider(self):
        """Test stopping the provider."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value = mock_session_instance

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()
            provider.start()

            assert provider.running is True

            provider.stop()

            assert provider.running is False
            mock_session_instance.close.assert_called_once()

    def test_stop_provider_no_session(self):
        """Test stopping provider when session is None."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session.side_effect = Exception("No session")

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()
            provider.start()
            provider.stop()

            assert provider.running is False

    def test_publish_message_with_session(self):
        """Test publishing message with valid session."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value = mock_session_instance

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()

            test_msg = {"time_stamp": time.time(), "message": "test"}
            provider._publish_message(test_msg)

            mock_session_instance.put.assert_called_once()

    def test_publish_message_no_session(self):
        """Test publishing message when session is None."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session.side_effect = Exception("No session")

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()

            test_msg = {"time_stamp": time.time(), "message": "test"}
            # Should not raise exception
            provider._publish_message(test_msg)

    def test_run_loop_processes_messages(self):
        """Test that run loop processes queued messages."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value = mock_session_instance

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()

            # Add a message to the queue
            provider.add_pending_message("test message")

            # Start and quickly stop to process the message
            provider.start()
            time.sleep(0.1)  # Give thread time to process
            provider.stop()

            # Verify message was processed
            mock_session_instance.put.assert_called()

    def test_run_loop_handles_empty_queue(self):
        """Test that run loop handles empty queue gracefully."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session.return_value = MagicMock()

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()

            # Start with empty queue
            provider.start()
            time.sleep(0.6)  # Wait longer than timeout
            provider.stop()

            # Should complete without errors
            assert provider.running is False

    def test_run_loop_handles_exception(self):
        """Test that run loop handles exceptions gracefully."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.put.side_effect = Exception("Publish error")
            mock_session.return_value = mock_session_instance

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()
            provider.add_pending_message("test message")

            # Start and stop - should handle exception
            provider.start()
            time.sleep(0.1)
            provider.stop()

            assert provider.running is False

    def test_message_format(self):
        """Test that messages are formatted correctly."""
        with (
            patch(
                "providers.zenoh_publisher_provider.open_zenoh_session"
            ) as mock_session,
            patch("providers.zenoh_publisher_provider.time.time") as mock_time,
        ):

            mock_time.return_value = 1234567890.0
            mock_session.return_value = MagicMock()

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()
            test_text = "Hello, world!"

            provider.add_pending_message(test_text)

            # Get the message from queue
            msg = provider._pending_messages.get()

            assert msg["time_stamp"] == 1234567890.0
            assert msg["message"] == test_text

    def test_thread_cleanup_on_stop(self):
        """Test that thread is properly cleaned up on stop."""
        with patch(
            "providers.zenoh_publisher_provider.open_zenoh_session"
        ) as mock_session:
            mock_session.return_value = MagicMock()

            from providers.zenoh_publisher_provider import ZenohPublisherProvider

            provider = ZenohPublisherProvider()
            provider.start()

            original_thread = provider._thread
            assert original_thread is not None
            assert original_thread.is_alive()

            provider.stop()

            # Thread should be joined and stopped
            assert not original_thread.is_alive()
