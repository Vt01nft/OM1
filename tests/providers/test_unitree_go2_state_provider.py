"""Tests for unitree_go2_state_provider."""

import sys
from queue import Empty, Full
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

# Mock Unitree SDK
mock_unitree = MagicMock()
mock_channel_factory = MagicMock()
mock_channel_subscriber = MagicMock()
mock_sport_mode_state = MagicMock()

mock_unitree.unitree_sdk2py = MagicMock()
mock_unitree.unitree_sdk2py.core = MagicMock()
mock_unitree.unitree_sdk2py.core.channel = MagicMock()
mock_unitree.unitree_sdk2py.core.channel.ChannelFactoryInitialize = mock_channel_factory
mock_unitree.unitree_sdk2py.core.channel.ChannelSubscriber = mock_channel_subscriber
mock_unitree.unitree_sdk2py.idl = MagicMock()
mock_unitree.unitree_sdk2py.idl.unitree_go = MagicMock()
mock_unitree.unitree_sdk2py.idl.unitree_go.msg = MagicMock()
mock_unitree.unitree_sdk2py.idl.unitree_go.msg.dds_ = MagicMock()
mock_unitree.unitree_sdk2py.idl.unitree_go.msg.dds_.SportModeState_ = (
    mock_sport_mode_state
)

sys.modules["unitree"] = mock_unitree
sys.modules["unitree.unitree_sdk2py"] = mock_unitree.unitree_sdk2py
sys.modules["unitree.unitree_sdk2py.core"] = mock_unitree.unitree_sdk2py.core
sys.modules["unitree.unitree_sdk2py.core.channel"] = (
    mock_unitree.unitree_sdk2py.core.channel
)
sys.modules["unitree.unitree_sdk2py.idl"] = mock_unitree.unitree_sdk2py.idl
sys.modules["unitree.unitree_sdk2py.idl.unitree_go"] = (
    mock_unitree.unitree_sdk2py.idl.unitree_go
)
sys.modules["unitree.unitree_sdk2py.idl.unitree_go.msg"] = (
    mock_unitree.unitree_sdk2py.idl.unitree_go.msg
)
sys.modules["unitree.unitree_sdk2py.idl.unitree_go.msg.dds_"] = (
    mock_unitree.unitree_sdk2py.idl.unitree_go.msg.dds_
)


class TestUnitreeGo2StateProvider:
    """Tests for UnitreeGo2StateProvider class."""

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

    def test_initialization_default_channel(self):
        """Test provider initializes correctly with default channel."""
        from providers.unitree_go2_state_provider import UnitreeGo2StateProvider

        if hasattr(UnitreeGo2StateProvider, "reset"):
            UnitreeGo2StateProvider.reset()

        provider = UnitreeGo2StateProvider()

        assert provider is not None
        assert provider.channel == ""
        assert provider.go2_state is None
        assert provider.go2_state_code is None
        assert provider.go2_action_progress == 0
        assert provider.go2_sport_mode_state_msg is None

    def test_initialization_custom_channel(self):
        """Test provider initializes correctly with custom channel."""
        from providers.unitree_go2_state_provider import UnitreeGo2StateProvider

        if hasattr(UnitreeGo2StateProvider, "reset"):
            UnitreeGo2StateProvider.reset()

        test_channel = "test_channel_name"
        provider = UnitreeGo2StateProvider(channel=test_channel)

        assert provider.channel == test_channel

    def test_singleton_pattern(self):
        """Test singleton pattern works correctly."""
        from providers.unitree_go2_state_provider import UnitreeGo2StateProvider

        if hasattr(UnitreeGo2StateProvider, "reset"):
            UnitreeGo2StateProvider.reset()

        provider1 = UnitreeGo2StateProvider()
        provider2 = UnitreeGo2StateProvider()

        assert provider1 is provider2

    @patch("providers.unitree_go2_state_provider.mp.Process")
    @patch("providers.unitree_go2_state_provider.threading.Thread")
    @patch("providers.unitree_go2_state_provider.get_logging_config")
    def test_start_creates_processes_and_threads(
        self, mock_get_logging, mock_thread, mock_process
    ):
        """Test start method creates necessary processes and threads."""
        from providers.unitree_go2_state_provider import UnitreeGo2StateProvider

        if hasattr(UnitreeGo2StateProvider, "reset"):
            UnitreeGo2StateProvider.reset()

        mock_get_logging.return_value = None
        mock_process_instance = MagicMock()
        mock_thread_instance = MagicMock()
        mock_process.return_value = mock_process_instance
        mock_thread.return_value = mock_thread_instance

        provider = UnitreeGo2StateProvider(channel="test_channel")
        provider.start()

        mock_process.assert_called_once()
        mock_thread.assert_called_once()
        mock_process_instance.start.assert_called_once()
        mock_thread_instance.start.assert_called_once()

    @patch("providers.unitree_go2_state_provider.mp.Process")
    @patch("providers.unitree_go2_state_provider.threading.Thread")
    def test_start_does_not_recreate_running_processes(self, mock_thread, mock_process):
        """Test start method doesn't recreate processes if already running."""
        from providers.unitree_go2_state_provider import UnitreeGo2StateProvider

        if hasattr(UnitreeGo2StateProvider, "reset"):
            UnitreeGo2StateProvider.reset()

        mock_process_instance = MagicMock()
        mock_thread_instance = MagicMock()
        mock_process_instance.is_alive.return_value = True
        mock_thread_instance.is_alive.return_value = True
        mock_process.return_value = mock_process_instance
        mock_thread.return_value = mock_thread_instance

        provider = UnitreeGo2StateProvider()
        provider._go2_state_reader_thread = mock_process_instance
        provider._go2_state_processor_thread = mock_thread_instance

        provider.start()

        # Should not create new instances
        assert mock_process.call_count == 0
        assert mock_thread.call_count == 0

    def test_stop_sets_stop_event(self):
        """Test stop method sets the stop event."""
        from providers.unitree_go2_state_provider import UnitreeGo2StateProvider

        if hasattr(UnitreeGo2StateProvider, "reset"):
            UnitreeGo2StateProvider.reset()

        provider = UnitreeGo2StateProvider()

        assert not provider._stop_event.is_set()
        provider.stop()
        assert provider._stop_event.is_set()


class TestGo2StateProcessor:
    """Tests for go2_state_processor function."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        yield

    @patch("providers.unitree_go2_state_provider.ChannelFactoryInitialize")
    @patch("providers.unitree_go2_state_provider.ChannelSubscriber")
    @patch("providers.unitree_go2_state_provider.setup_logging")
    def test_go2_state_processor_initialization_success(
        self, mock_setup_logging, mock_subscriber, mock_factory
    ):
        """Test successful initialization of go2_state_processor."""
        from providers.unitree_go2_state_provider import go2_state_processor

        data_queue = MagicMock()
        control_queue = MagicMock()
        control_queue.get_nowait.side_effect = [Empty(), "STOP"]

        mock_subscriber_instance = MagicMock()
        mock_subscriber.return_value = mock_subscriber_instance

        go2_state_processor("test_channel", data_queue, control_queue, None)

        mock_setup_logging.assert_called_once()
        mock_factory.assert_called_once_with(0, "test_channel")
        mock_subscriber.assert_called_once()
        mock_subscriber_instance.Init.assert_called_once()
        mock_subscriber_instance.Close.assert_called_once()

    @patch("providers.unitree_go2_state_provider.ChannelFactoryInitialize")
    @patch("providers.unitree_go2_state_provider.setup_logging")
    @patch("providers.unitree_go2_state_provider.logging")
    def test_go2_state_processor_factory_initialization_error(
        self, mock_logging, mock_setup_logging, mock_factory
    ):
        """Test go2_state_processor handles factory initialization errors."""
        from providers.unitree_go2_state_provider import go2_state_processor

        mock_factory.side_effect = Exception("Factory error")
        data_queue = MagicMock()
        control_queue = MagicMock()

        go2_state_processor("test_channel", data_queue, control_queue, None)

        mock_logging.error.assert_called_with(
            "Error initializing Unitree Go2 odom channel: Factory error"
        )

    @patch("providers.unitree_go2_state_provider.ChannelFactoryInitialize")
    @patch("providers.unitree_go2_state_provider.ChannelSubscriber")
    @patch("providers.unitree_go2_state_provider.setup_logging")
    @patch("providers.unitree_go2_state_provider.logging")
    def test_go2_state_processor_subscriber_error(
        self, mock_logging, mock_setup_logging, mock_subscriber, mock_factory
    ):
        """Test go2_state_processor handles subscriber errors."""
        from providers.unitree_go2_state_provider import go2_state_processor

        mock_subscriber.side_effect = Exception("Subscriber error")
        data_queue = MagicMock()
        control_queue = MagicMock()

        go2_state_processor("test_channel", data_queue, control_queue, None)

        mock_logging.error.assert_called_with(
            "Error subscribing to Unitree Go2 state channel: Subscriber error"
        )

    def test_state_codes_mapping(self):
        """Test that state machine codes are correctly mapped."""
        from providers.unitree_go2_state_provider import state_machine_codes

        assert state_machine_codes[100] == "Agile"
        assert state_machine_codes[1001] == "Damping"
        assert state_machine_codes[1002] == "Standing Lock"
        assert state_machine_codes[1004] == "Crouch"
        assert state_machine_codes[2006] == "Crouch"
        assert state_machine_codes[1091] == "Strike a Pose"
        assert state_machine_codes[2012] == "Front Flip"

    @patch("providers.unitree_go2_state_provider.ChannelFactoryInitialize")
    @patch("providers.unitree_go2_state_provider.ChannelSubscriber")
    @patch("providers.unitree_go2_state_provider.setup_logging")
    def test_state_callback_queue_operations(
        self, mock_setup_logging, mock_subscriber, mock_factory
    ):
        """Test state callback properly manages queue operations."""
        from providers.unitree_go2_state_provider import go2_state_processor

        data_queue = MagicMock()
        control_queue = MagicMock()
        control_queue.get_nowait.side_effect = [Empty(), "STOP"]

        mock_subscriber_instance = MagicMock()
        mock_subscriber.return_value = mock_subscriber_instance

        # Capture the callback function
        callback_func = None

        def capture_callback(callback, *args):
            nonlocal callback_func
            callback_func = callback

        mock_subscriber_instance.Init.side_effect = capture_callback

        go2_state_processor("test_channel", data_queue, control_queue, None)

        # Test the callback with a mock message
        mock_msg = MagicMock()
        mock_msg.error_code = 100
        mock_msg.progress = 50

        if callback_func:
            callback_func(mock_msg)
            data_queue.put_nowait.assert_called()

    @patch("providers.unitree_go2_state_provider.ChannelFactoryInitialize")
    @patch("providers.unitree_go2_state_provider.ChannelSubscriber")
    @patch("providers.unitree_go2_state_provider.setup_logging")
    def test_state_callback_queue_full_handling(
        self, mock_setup_logging, mock_subscriber, mock_factory
    ):
        """Test state callback handles full queue properly."""
        from providers.unitree_go2_state_provider import go2_state_processor

        data_queue = MagicMock()
        data_queue.put_nowait.side_effect = [Full(), None]
        data_queue.get_nowait.return_value = "old_data"

        control_queue = MagicMock()
        control_queue.get_nowait.side_effect = [Empty(), "STOP"]

        mock_subscriber_instance = MagicMock()
        mock_subscriber.return_value = mock_subscriber_instance

        # Capture the callback function
        callback_func = None

        def capture_callback(callback, *args):
            nonlocal callback_func
            callback_func = callback

        mock_subscriber_instance.Init.side_effect = capture_callback

        go2_state_processor("test_channel", data_queue, control_queue, None)

        # Test the callback with a mock message
        mock_msg = MagicMock()
        mock_msg.error_code = 100
        mock_msg.progress = 50

        if callback_func:
            callback_func(mock_msg)
            assert data_queue.put_nowait.call_count == 2
            data_queue.get_nowait.assert_called_once()

    @patch("providers.unitree_go2_state_provider.ChannelFactoryInitialize")
    @patch("providers.unitree_go2_state_provider.ChannelSubscriber")
    @patch("providers.unitree_go2_state_provider.setup_logging")
    def test_state_callback_queue_empty_on_get(
        self, mock_setup_logging, mock_subscriber, mock_factory
    ):
        """Test state callback handles empty queue on get."""
        from providers.unitree_go2_state_provider import go2_state_processor

        data_queue = MagicMock()
        data_queue.put_nowait.side_effect = [Full(), None]
        data_queue.get_nowait.side_effect = Empty()

        control_queue = MagicMock()
        control_queue.get_nowait.side_effect = [Empty(), "STOP"]

        mock_subscriber_instance = MagicMock()
        mock_subscriber.return_value = mock_subscriber_instance

        # Capture the callback function
        callback_func = None

        def capture_callback(callback, *args):
            nonlocal callback_func
            callback_func = callback

        mock_subscriber_instance.Init.side_effect = capture_callback

        go2_state_processor("test_channel", data_queue, control_queue, None)

        # Test the callback with a mock message
        mock_msg = MagicMock()
        mock_msg.error_code = 100
        mock_msg.progress = 50

        if callback_func:
            callback_func(mock_msg)
            # Should not raise exception
            assert data_queue.put_nowait.call_count == 1
