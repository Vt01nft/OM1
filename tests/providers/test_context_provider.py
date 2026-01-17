"""Tests for context_provider."""

import json
import logging
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


class TestContextProvider:
    """Tests for ContextProvider class."""

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
    def mock_zenoh_session(self):
        """Mock Zenoh session and related components."""
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher

        with patch(
            "providers.context_provider.open_zenoh_session", return_value=mock_session
        ):
            yield mock_session, mock_publisher

    def test_initialization(self, mock_zenoh_session):
        """Test provider initializes correctly."""
        from providers.context_provider import ContextProvider

        if hasattr(ContextProvider, "reset"):
            ContextProvider.reset()

        provider = ContextProvider()
        assert provider is not None
        assert provider.context_update_topic == "om/mode/context"
        assert provider.session is not None
        assert provider.publisher is not None

    def test_singleton_behavior(self, mock_zenoh_session):
        """Test singleton pattern works correctly."""
        from providers.context_provider import ContextProvider

        if hasattr(ContextProvider, "reset"):
            ContextProvider.reset()

        provider1 = ContextProvider()
        provider2 = ContextProvider()

        assert provider1 is provider2

    def test_initialization_zenoh_failure(self):
        """Test initialization handles Zenoh session failure gracefully."""
        with patch(
            "providers.context_provider.open_zenoh_session",
            side_effect=Exception("Connection failed"),
        ):
            from providers.context_provider import ContextProvider

            if hasattr(ContextProvider, "reset"):
                ContextProvider.reset()

            provider = ContextProvider()
            assert provider.session is None
            assert provider.publisher is None

    def test_update_context_success(self, mock_zenoh_session):
        """Test update_context sends context successfully."""
        mock_session, mock_publisher = mock_zenoh_session

        from providers.context_provider import ContextProvider

        if hasattr(ContextProvider, "reset"):
            ContextProvider.reset()

        provider = ContextProvider()
        context = {"user": "test_user", "location": "office"}

        provider.update_context(context)

        expected_json = json.dumps(context).encode("utf-8")
        mock_publisher.put.assert_called_once_with(expected_json)

    def test_update_context_no_publisher(self):
        """Test update_context handles missing publisher gracefully."""
        with patch(
            "providers.context_provider.open_zenoh_session",
            side_effect=Exception("No connection"),
        ):
            from providers.context_provider import ContextProvider

            if hasattr(ContextProvider, "reset"):
                ContextProvider.reset()

            provider = ContextProvider()
            context = {"test": "data"}

            # Should not raise exception
            provider.update_context(context)

    def test_update_context_publisher_error(self, mock_zenoh_session, caplog):
        """Test update_context handles publisher errors gracefully."""
        mock_session, mock_publisher = mock_zenoh_session
        mock_publisher.put.side_effect = Exception("Publish failed")

        from providers.context_provider import ContextProvider

        if hasattr(ContextProvider, "reset"):
            ContextProvider.reset()

        provider = ContextProvider()
        context = {"test": "data"}

        with caplog.at_level(logging.ERROR):
            provider.update_context(context)

        assert "Error sending context update" in caplog.text

    def test_set_context_field(self, mock_zenoh_session):
        """Test set_context_field creates single-field context update."""
        mock_session, mock_publisher = mock_zenoh_session

        from providers.context_provider import ContextProvider

        if hasattr(ContextProvider, "reset"):
            ContextProvider.reset()

        provider = ContextProvider()

        provider.set_context_field("mood", "happy")

        expected_json = json.dumps({"mood": "happy"}).encode("utf-8")
        mock_publisher.put.assert_called_once_with(expected_json)

    def test_set_context_field_complex_value(self, mock_zenoh_session):
        """Test set_context_field handles complex data types."""
        mock_session, mock_publisher = mock_zenoh_session

        from providers.context_provider import ContextProvider

        if hasattr(ContextProvider, "reset"):
            ContextProvider.reset()

        provider = ContextProvider()
        complex_value = {"nested": {"data": [1, 2, 3]}}

        provider.set_context_field("complex", complex_value)

        expected_json = json.dumps({"complex": complex_value}).encode("utf-8")
        mock_publisher.put.assert_called_once_with(expected_json)

    def test_stop_success(self, mock_zenoh_session):
        """Test stop closes session successfully."""
        mock_session, mock_publisher = mock_zenoh_session

        from providers.context_provider import ContextProvider

        if hasattr(ContextProvider, "reset"):
            ContextProvider.reset()

        provider = ContextProvider()

        provider.stop()

        mock_session.close.assert_called_once()
        assert provider.session is None
        assert provider.publisher is None

    def test_stop_no_session(self):
        """Test stop handles missing session gracefully."""
        with patch(
            "providers.context_provider.open_zenoh_session",
            side_effect=Exception("No connection"),
        ):
            from providers.context_provider import ContextProvider

            if hasattr(ContextProvider, "reset"):
                ContextProvider.reset()

            provider = ContextProvider()

            # Should not raise exception
            provider.stop()

    def test_stop_session_error(self, mock_zenoh_session, caplog):
        """Test stop handles session close errors gracefully."""
        mock_session, mock_publisher = mock_zenoh_session
        mock_session.close.side_effect = Exception("Close failed")

        from providers.context_provider import ContextProvider

        if hasattr(ContextProvider, "reset"):
            ContextProvider.reset()

        provider = ContextProvider()

        with caplog.at_level(logging.ERROR):
            provider.stop()

        assert "Error stopping ContextProvider" in caplog.text
        assert provider.session is None
        assert provider.publisher is None

    def test_update_context_empty_dict(self, mock_zenoh_session):
        """Test update_context handles empty dictionary."""
        mock_session, mock_publisher = mock_zenoh_session

        from providers.context_provider import ContextProvider

        if hasattr(ContextProvider, "reset"):
            ContextProvider.reset()

        provider = ContextProvider()

        provider.update_context({})

        expected_json = json.dumps({}).encode("utf-8")
        mock_publisher.put.assert_called_once_with(expected_json)

    def test_set_context_field_none_value(self, mock_zenoh_session):
        """Test set_context_field handles None value."""
        mock_session, mock_publisher = mock_zenoh_session

        from providers.context_provider import ContextProvider

        if hasattr(ContextProvider, "reset"):
            ContextProvider.reset()

        provider = ContextProvider()

        provider.set_context_field("test_key", None)

        expected_json = json.dumps({"test_key": None}).encode("utf-8")
        mock_publisher.put.assert_called_once_with(expected_json)
