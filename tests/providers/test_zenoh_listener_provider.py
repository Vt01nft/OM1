"""Tests for zenoh_listener_provider."""

import sys
from unittest.mock import MagicMock, Mock, patch

# Mock external dependencies before any imports
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()

import logging

from providers.zenoh_listener_provider import ZenohListenerProvider


class TestZenohListenerProvider:
    """Tests for the ZenohListenerProvider class."""

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_init_successful_session(self, mock_open_session):
        """Test successful initialization with default topic."""
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()

        assert provider.session is mock_session
        assert provider.sub_topic == "speech"
        assert provider.running is False
        mock_open_session.assert_called_once()

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_init_custom_topic(self, mock_open_session):
        """Test initialization with custom topic."""
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider(topic="custom_topic")

        assert provider.session is mock_session
        assert provider.sub_topic == "custom_topic"
        assert provider.running is False

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_init_session_creation_failure(self, mock_open_session, caplog):
        """Test initialization when session creation fails."""
        mock_open_session.side_effect = Exception("Connection failed")

        with caplog.at_level(logging.ERROR):
            provider = ZenohListenerProvider()

        assert provider.session is None
        assert provider.sub_topic == "speech"
        assert provider.running is False
        assert "Error opening Zenoh client: Connection failed" in caplog.text

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_register_message_callback_success(self, mock_open_session):
        """Test successful callback registration."""
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()
        callback = Mock()

        provider.register_message_callback(callback)

        mock_session.declare_subscriber.assert_called_once_with("speech", callback)

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_register_message_callback_no_session(self, mock_open_session, caplog):
        """Test callback registration when session is None."""
        mock_open_session.side_effect = Exception("Connection failed")

        provider = ZenohListenerProvider()
        callback = Mock()

        with caplog.at_level(logging.ERROR):
            provider.register_message_callback(callback)

        assert (
            "Cannot register callback; Zenoh session is not available." in caplog.text
        )

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_register_message_callback_none(self, mock_open_session):
        """Test callback registration with None callback."""
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()

        provider.register_message_callback(None)

        mock_session.declare_subscriber.assert_called_once_with("speech", None)

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_start_without_callback(self, mock_open_session):
        """Test starting provider without callback."""
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()

        provider.start()

        assert provider.running is True

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_start_with_callback(self, mock_open_session):
        """Test starting provider with callback."""
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()
        callback = Mock()

        provider.start(message_callback=callback)

        assert provider.running is True
        mock_session.declare_subscriber.assert_called_once_with("speech", callback)

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_start_already_running(self, mock_open_session, caplog):
        """Test starting provider when already running."""
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()
        provider.running = True

        with caplog.at_level(logging.WARNING):
            provider.start()

        assert "Zenoh Listener Provider is already running" in caplog.text

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_start_none_callback(self, mock_open_session):
        """Test starting provider with explicitly None callback."""
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()

        provider.start(message_callback=None)

        assert provider.running is True
        mock_session.declare_subscriber.assert_not_called()

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_stop_with_session(self, mock_open_session):
        """Test stopping provider with active session."""
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()
        provider.running = True

        provider.stop()

        assert provider.running is False
        mock_session.close.assert_called_once()

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_stop_without_session(self, mock_open_session):
        """Test stopping provider when session is None."""
        mock_open_session.side_effect = Exception("Connection failed")

        provider = ZenohListenerProvider()
        provider.running = True

        provider.stop()

        assert provider.running is False

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_stop_already_stopped(self, mock_open_session):
        """Test stopping provider when already stopped."""
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider()
        provider.running = False

        provider.stop()

        assert provider.running is False
        mock_session.close.assert_called_once()

    @patch("providers.zenoh_listener_provider.open_zenoh_session")
    def test_full_lifecycle(self, mock_open_session):
        """Test complete lifecycle: init, start, stop."""
        mock_session = Mock()
        mock_open_session.return_value = mock_session

        provider = ZenohListenerProvider("test_topic")
        callback = Mock()

        # Start
        provider.start(message_callback=callback)
        assert provider.running is True
        mock_session.declare_subscriber.assert_called_once_with("test_topic", callback)

        # Stop
        provider.stop()
        assert provider.running is False
        mock_session.close.assert_called_once()
