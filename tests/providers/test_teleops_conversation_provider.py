"""Tests for teleops_conversation_provider."""

import sys
from unittest.mock import MagicMock, Mock, patch

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


class TestConversationMessage:
    """Tests for ConversationMessage class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

    def test_initialization(self):
        """Test ConversationMessage initializes correctly."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
        )

        message = ConversationMessage(
            message_type=MessageType.USER, content="Hello", timestamp=123.456
        )

        assert message.message_type == MessageType.USER
        assert message.content == "Hello"
        assert message.timestamp == 123.456

    def test_to_dict(self):
        """Test ConversationMessage converts to dictionary correctly."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
        )

        message = ConversationMessage(
            message_type=MessageType.ROBOT, content="Hi there", timestamp=789.123
        )

        result = message.to_dict()
        expected = {"type": "robot", "content": "Hi there", "timestamp": 789.123}

        assert result == expected

    def test_from_dict_complete_data(self):
        """Test ConversationMessage creates from complete dictionary data."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
        )

        data = {"type": "user", "content": "Test message", "timestamp": 456.789}

        message = ConversationMessage.from_dict(data)

        assert message.message_type == MessageType.USER
        assert message.content == "Test message"
        assert message.timestamp == 456.789

    def test_from_dict_missing_fields(self):
        """Test ConversationMessage creates from partial dictionary data."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
        )

        data = {}
        message = ConversationMessage.from_dict(data)

        assert message.message_type == MessageType.USER
        assert message.content == ""
        assert message.timestamp == 0.0

    def test_from_dict_partial_fields(self):
        """Test ConversationMessage creates with some missing fields."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
        )

        data = {"content": "Partial data"}
        message = ConversationMessage.from_dict(data)

        assert message.message_type == MessageType.USER
        assert message.content == "Partial data"
        assert message.timestamp == 0.0


class TestTeleopsConversationProvider:
    """Tests for TeleopsConversationProvider class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

    def test_initialization_default_values(self):
        """Test provider initializes with default values."""
        from providers.teleops_conversation_provider import TeleopsConversationProvider

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider()

        assert provider.api_key is None
        assert (
            provider.base_url
            == "https://api.openmind.org/api/core/teleops/conversation"
        )
        assert provider.executor is not None

    def test_initialization_with_parameters(self):
        """Test provider initializes with custom parameters."""
        from providers.teleops_conversation_provider import TeleopsConversationProvider

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider(
            api_key="test_key", base_url="https://custom.api.com/conversation"
        )

        assert provider.api_key == "test_key"
        assert provider.base_url == "https://custom.api.com/conversation"

    def test_singleton_pattern(self):
        """Test provider follows singleton pattern."""
        from providers.teleops_conversation_provider import TeleopsConversationProvider

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider1 = TeleopsConversationProvider()
        provider2 = TeleopsConversationProvider()

        assert provider1 is provider2

    def test_is_enabled_with_api_key(self):
        """Test is_enabled returns True when API key is set."""
        from providers.teleops_conversation_provider import TeleopsConversationProvider

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider(api_key="test_key")

        assert provider.is_enabled() is True

    def test_is_enabled_without_api_key(self):
        """Test is_enabled returns False when API key is None."""
        from providers.teleops_conversation_provider import TeleopsConversationProvider

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider()

        assert provider.is_enabled() is False

    def test_is_enabled_with_empty_api_key(self):
        """Test is_enabled returns False when API key is empty string."""
        from providers.teleops_conversation_provider import TeleopsConversationProvider

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider(api_key="")

        assert provider.is_enabled() is False

    @patch("time.time", return_value=123.456)
    def test_store_user_message(self, mock_time):
        """Test storing user message creates correct message type."""
        from providers.teleops_conversation_provider import (
            MessageType,
            TeleopsConversationProvider,
        )

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider()

        with patch.object(provider, "_store_message") as mock_store:
            provider.store_user_message("Hello world")

            mock_store.assert_called_once()
            message = mock_store.call_args[0][0]
            assert message.message_type == MessageType.USER
            assert message.content == "Hello world"
            assert message.timestamp == 123.456

    @patch("time.time", return_value=789.123)
    def test_store_robot_message(self, mock_time):
        """Test storing robot message creates correct message type."""
        from providers.teleops_conversation_provider import (
            MessageType,
            TeleopsConversationProvider,
        )

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider()

        with patch.object(provider, "_store_message") as mock_store:
            provider.store_robot_message("Robot response")

            mock_store.assert_called_once()
            message = mock_store.call_args[0][0]
            assert message.message_type == MessageType.ROBOT
            assert message.content == "Robot response"
            assert message.timestamp == 789.123

    def test_store_user_message_strips_whitespace(self):
        """Test storing user message strips whitespace."""
        from providers.teleops_conversation_provider import TeleopsConversationProvider

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider()

        with patch.object(provider, "_store_message") as mock_store:
            provider.store_user_message("  Hello world  ")

            message = mock_store.call_args[0][0]
            assert message.content == "Hello world"

    def test_store_robot_message_strips_whitespace(self):
        """Test storing robot message strips whitespace."""
        from providers.teleops_conversation_provider import TeleopsConversationProvider

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider()

        with patch.object(provider, "_store_message") as mock_store:
            provider.store_robot_message("  Robot response  ")

            message = mock_store.call_args[0][0]
            assert message.content == "Robot response"

    def test_store_message_submits_to_executor(self):
        """Test _store_message submits task to executor."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
            TeleopsConversationProvider,
        )

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider()
        message = ConversationMessage(MessageType.USER, "test", 123.0)

        with patch.object(provider.executor, "submit") as mock_submit:
            provider._store_message(message)

            mock_submit.assert_called_once_with(provider._store_message_worker, message)

    @patch("providers.teleops_conversation_provider.requests")
    def test_store_message_worker_success(self, mock_requests):
        """Test _store_message_worker handles successful API call."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
            TeleopsConversationProvider,
        )

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.post.return_value = mock_response

        provider = TeleopsConversationProvider(api_key="test_key")
        message = ConversationMessage(MessageType.USER, "test message", 123.0)

        # Should not raise exception
        provider._store_message_worker(message)

        mock_requests.post.assert_called_once_with(
            provider.base_url,
            headers={"Authorization": "Bearer test_key"},
            json=message.to_dict(),
            timeout=2,
        )

    @patch("providers.teleops_conversation_provider.requests")
    def test_store_message_worker_api_error(self, mock_requests):
        """Test _store_message_worker handles API error response."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
            TeleopsConversationProvider,
        )

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_requests.post.return_value = mock_response

        provider = TeleopsConversationProvider(api_key="test_key")
        message = ConversationMessage(MessageType.USER, "test message", 123.0)

        # Should not raise exception
        provider._store_message_worker(message)

    @patch("providers.teleops_conversation_provider.requests")
    def test_store_message_worker_network_exception(self, mock_requests):
        """Test _store_message_worker handles network exceptions."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
            TeleopsConversationProvider,
        )

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        mock_requests.post.side_effect = Exception("Network error")

        provider = TeleopsConversationProvider(api_key="test_key")
        message = ConversationMessage(MessageType.USER, "test message", 123.0)

        # Should not raise exception
        provider._store_message_worker(message)

    def test_store_message_worker_no_api_key(self):
        """Test _store_message_worker skips when no API key."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
            TeleopsConversationProvider,
        )

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider()
        message = ConversationMessage(MessageType.USER, "test message", 123.0)

        with patch("providers.teleops_conversation_provider.requests") as mock_requests:
            provider._store_message_worker(message)
            mock_requests.post.assert_not_called()

    def test_store_message_worker_empty_api_key(self):
        """Test _store_message_worker skips when API key is empty."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
            TeleopsConversationProvider,
        )

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider(api_key="")
        message = ConversationMessage(MessageType.USER, "test message", 123.0)

        with patch("providers.teleops_conversation_provider.requests") as mock_requests:
            provider._store_message_worker(message)
            mock_requests.post.assert_not_called()

    def test_store_message_worker_empty_content(self):
        """Test _store_message_worker skips when message content is empty."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
            TeleopsConversationProvider,
        )

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider(api_key="test_key")
        message = ConversationMessage(MessageType.USER, "", 123.0)

        with patch("providers.teleops_conversation_provider.requests") as mock_requests:
            provider._store_message_worker(message)
            mock_requests.post.assert_not_called()

    def test_store_message_worker_whitespace_only_content(self):
        """Test _store_message_worker skips when message content is whitespace only."""
        from providers.teleops_conversation_provider import (
            ConversationMessage,
            MessageType,
            TeleopsConversationProvider,
        )

        if hasattr(TeleopsConversationProvider, "reset"):
            TeleopsConversationProvider.reset()

        provider = TeleopsConversationProvider(api_key="test_key")
        message = ConversationMessage(MessageType.USER, "   ", 123.0)

        with patch("providers.teleops_conversation_provider.requests") as mock_requests:
            provider._store_message_worker(message)
            mock_requests.post.assert_not_called()


class TestMessageType:
    """Tests for MessageType enum."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

    def test_message_type_values(self):
        """Test MessageType enum has correct values."""
        from providers.teleops_conversation_provider import MessageType

        assert MessageType.USER.value == "user"
        assert MessageType.ROBOT.value == "robot"
