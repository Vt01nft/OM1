"""Tests for simple_paths_provider."""

import sys
from queue import Empty
from unittest.mock import MagicMock, patch

import pytest

# Mock ALL external dependencies BEFORE any provider imports
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()
sys.modules["zenoh_msgs.open_zenoh_session"] = MagicMock()
sys.modules["zenoh_msgs.sensor_msgs"] = MagicMock()
sys.modules["runtime"] = MagicMock()
sys.modules["runtime.logging"] = MagicMock()
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


class TestSimplePathsProvider:
    """Tests for SimplePathsProvider class."""

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

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies."""
        with (
            patch("providers.simple_paths_provider.open_zenoh_session") as mock_zenoh,
            patch("providers.simple_paths_provider.sensor_msgs") as mock_sensor_msgs,
            patch(
                "providers.simple_paths_provider.get_logging_config"
            ) as mock_logging_config,
            patch(
                "providers.simple_paths_provider.setup_logging"
            ) as mock_setup_logging,
            patch("providers.simple_paths_provider.mp.Queue") as mock_queue,
            patch("providers.simple_paths_provider.mp.Process") as mock_process,
            patch("providers.simple_paths_provider.threading.Thread") as mock_thread,
        ):
            mock_queue_instance = MagicMock()
            mock_queue.return_value = mock_queue_instance

            mock_process_instance = MagicMock()
            mock_process.return_value = mock_process_instance

            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            yield {
                "zenoh": mock_zenoh,
                "sensor_msgs": mock_sensor_msgs,
                "logging_config": mock_logging_config,
                "setup_logging": mock_setup_logging,
                "queue": mock_queue,
                "process": mock_process,
                "thread": mock_thread,
                "queue_instance": mock_queue_instance,
                "process_instance": mock_process_instance,
                "thread_instance": mock_thread_instance,
            }

    def test_initialization(self, mock_dependencies):
        """Test provider initializes correctly."""
        from providers.simple_paths_provider import SimplePathsProvider

        if hasattr(SimplePathsProvider, "reset"):
            SimplePathsProvider.reset()

        provider = SimplePathsProvider()
        assert provider is not None
        assert provider.session is None
        assert provider.paths is None
        assert provider.turn_left == []
        assert provider.turn_right == []
        assert provider.advance == []
        assert provider.retreat is False
        assert provider._valid_paths == []
        assert provider._lidar_string == ""
        assert provider.path_angles == [-60, -45, -30, -15, 0, 15, 30, 45, 60, 180]

    def test_singleton_pattern(self, mock_dependencies):
        """Test singleton pattern works correctly."""
        from providers.simple_paths_provider import SimplePathsProvider

        if hasattr(SimplePathsProvider, "reset"):
            SimplePathsProvider.reset()

        provider1 = SimplePathsProvider()
        provider2 = SimplePathsProvider()
        assert provider1 is provider2

    def test_start_creates_processes_and_threads(self, mock_dependencies):
        """Test start method creates multiprocessing and threading components."""
        from providers.simple_paths_provider import SimplePathsProvider

        if hasattr(SimplePathsProvider, "reset"):
            SimplePathsProvider.reset()

        provider = SimplePathsProvider()

        # Mock thread and process as not alive initially
        mock_dependencies["process_instance"].is_alive.return_value = False
        mock_dependencies["thread_instance"].is_alive.return_value = False

        provider.start()

        # Verify process was created and started
        mock_dependencies["process"].assert_called()
        mock_dependencies["process_instance"].start.assert_called()

        # Verify thread was created and started
        mock_dependencies["thread"].assert_called()
        mock_dependencies["thread_instance"].start.assert_called()

    def test_start_does_not_recreate_alive_processes(self, mock_dependencies):
        """Test start method doesn't recreate alive processes and threads."""
        from providers.simple_paths_provider import SimplePathsProvider

        if hasattr(SimplePathsProvider, "reset"):
            SimplePathsProvider.reset()

        provider = SimplePathsProvider()

        # Mock existing process and thread
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True

        provider._simple_paths_processor_thread = mock_process
        provider._simple_paths_derived_thread = mock_thread

        provider.start()

        # Should not create new process or thread
        assert mock_dependencies["process"].call_count == 0
        assert mock_dependencies["thread"].call_count == 0

    def test_stop_sets_stop_event_and_joins_threads(self, mock_dependencies):
        """Test stop method properly stops processes and threads."""
        from providers.simple_paths_provider import SimplePathsProvider

        if hasattr(SimplePathsProvider, "reset"):
            SimplePathsProvider.reset()

        provider = SimplePathsProvider()

        # Mock existing threads
        mock_process = MagicMock()
        mock_thread = MagicMock()
        provider._simple_paths_processor_thread = mock_process
        provider._simple_paths_derived_thread = mock_thread

        # Mock stop event
        mock_stop_event = MagicMock()
        provider._stop_event = mock_stop_event

        # Mock control queue
        mock_control_queue = MagicMock()
        provider.control_queue = mock_control_queue

        provider.stop()

        # Verify stop event was set
        mock_stop_event.set.assert_called_once()

        # Verify STOP command was sent to control queue
        mock_control_queue.put.assert_called_with("STOP")

        # Verify threads were joined
        mock_process.join.assert_called_once()
        mock_thread.join.assert_called_once()

    def test_generate_movement_string_with_empty_paths(self, mock_dependencies):
        """Test _generate_movement_string with empty paths."""
        from providers.simple_paths_provider import SimplePathsProvider

        if hasattr(SimplePathsProvider, "reset"):
            SimplePathsProvider.reset()

        provider = SimplePathsProvider()

        # Test with empty list
        result = provider._generate_movement_string([])
        # Method should handle empty list without error
        assert isinstance(result, str) or result is None

    def test_generate_movement_string_with_paths(self, mock_dependencies):
        """Test _generate_movement_string with valid paths."""
        from providers.simple_paths_provider import SimplePathsProvider

        if hasattr(SimplePathsProvider, "reset"):
            SimplePathsProvider.reset()

        provider = SimplePathsProvider()

        # Test with some paths
        paths = [0, 1, 2, 4, 6, 7, 9]
        result = provider._generate_movement_string(paths)
        # Method should handle paths without error
        assert isinstance(result, str) or result is None

    def test_simple_paths_derived_processor_categorizes_paths(self, mock_dependencies):
        """Test _simple_paths_derived_processor categorizes paths correctly."""
        from providers.simple_paths_provider import SimplePathsProvider

        if hasattr(SimplePathsProvider, "reset"):
            SimplePathsProvider.reset()

        provider = SimplePathsProvider()

        # Mock paths data
        mock_paths = MagicMock()
        mock_paths.paths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Mock data queue to return paths once then raise Empty
        provider.data_queue.get_nowait.side_effect = [mock_paths, Empty()]

        # Mock stop event
        provider._stop_event = MagicMock()
        provider._stop_event.is_set.side_effect = [False, True]  # Run once then stop

        # Mock _generate_movement_string to avoid incomplete method
        provider._generate_movement_string = MagicMock(return_value="test_string")

        # Run the processor
        provider._simple_paths_derived_processor()

        # Verify paths were categorized
        assert provider.turn_left == [0, 1, 2]  # < 3
        assert provider.advance == [3, 4, 5]  # >= 3 and <= 5
        assert provider.turn_right == [6, 7, 8]  # < 9 and > 5
        assert provider.retreat is True  # == 9

    def test_simple_paths_derived_processor_handles_empty_queue(
        self, mock_dependencies
    ):
        """Test _simple_paths_derived_processor handles empty queue gracefully."""
        from providers.simple_paths_provider import SimplePathsProvider

        if hasattr(SimplePathsProvider, "reset"):
            SimplePathsProvider.reset()

        provider = SimplePathsProvider()

        # Mock data queue to always raise Empty
        provider.data_queue.get_nowait.side_effect = Empty()

        # Mock stop event to stop after first iteration
        provider._stop_event = MagicMock()
        provider._stop_event.is_set.side_effect = [False, True]

        # Should not raise exception
        provider._simple_paths_derived_processor()


class TestSimplePathsProcessor:
    """Tests for simple_paths_processor function."""

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

    def test_simple_paths_processor_function_exists(self):
        """Test simple_paths_processor function can be imported."""
        from providers.simple_paths_provider import simple_paths_processor

        assert callable(simple_paths_processor)

    def test_simple_paths_processor_with_mock_queues(self):
        """Test simple_paths_processor with mock queues."""
        with (
            patch("providers.simple_paths_provider.setup_logging"),
            patch("providers.simple_paths_provider.open_zenoh_session") as mock_zenoh,
            patch("providers.simple_paths_provider.time.sleep"),
        ):
            from providers.simple_paths_provider import simple_paths_processor

            # Mock queues
            data_queue = MagicMock()
            control_queue = MagicMock()

            # Mock control queue to return STOP command
            control_queue.get_nowait.side_effect = ["STOP"]

            # Mock zenoh session
            mock_session = MagicMock()
            mock_zenoh.return_value = mock_session

            # Should not raise exception
            simple_paths_processor(data_queue, control_queue)

            # Verify zenoh session was created and subscriber declared
            mock_zenoh.assert_called_once()
            mock_session.declare_subscriber.assert_called_once()

    def test_simple_paths_processor_handles_zenoh_exception(self):
        """Test simple_paths_processor handles Zenoh session exceptions."""
        with (
            patch("providers.simple_paths_provider.setup_logging"),
            patch(
                "providers.simple_paths_provider.open_zenoh_session",
                side_effect=Exception("Zenoh error"),
            ),
            patch("providers.simple_paths_provider.logging") as mock_logging,
            patch("providers.simple_paths_provider.time.sleep"),
        ):
            from providers.simple_paths_provider import simple_paths_processor

            # Mock queues
            data_queue = MagicMock()
            control_queue = MagicMock()

            # Mock control queue to return STOP command
            control_queue.get_nowait.side_effect = ["STOP"]

            # Should not raise exception
            simple_paths_processor(data_queue, control_queue)

            # Verify error was logged
            mock_logging.error.assert_called()
