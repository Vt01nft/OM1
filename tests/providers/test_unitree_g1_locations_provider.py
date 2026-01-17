"""Tests for unitree_g1_locations_provider."""

import json
import sys
import threading
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


class TestUnitreeG1LocationsProvider:
    """Tests for UnitreeG1LocationsProvider class."""

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

    def test_initialization_with_defaults(self):
        """Test provider initializes correctly with default parameters."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()
            assert provider.base_url == "http://localhost:5000/maps/locations/list"
            assert provider.timeout == 5
            assert provider.refresh_interval == 30
            assert provider._locations == {}
            assert provider._thread is None

    def test_initialization_with_custom_parameters(self):
        """Test provider initializes correctly with custom parameters."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider(
                base_url="http://custom:8080/locations", timeout=10, refresh_interval=60
            )
            assert provider.base_url == "http://custom:8080/locations"
            assert provider.timeout == 10
            assert provider.refresh_interval == 60

    def test_singleton_pattern(self):
        """Test provider follows singleton pattern."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider1 = UnitreeG1LocationsProvider()
            provider2 = UnitreeG1LocationsProvider()
            assert provider1 is provider2

    def test_start_creates_thread(self):
        """Test start method creates and starts background thread."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            with patch("threading.Thread") as mock_thread:
                mock_thread_instance = Mock()
                mock_thread_instance.is_alive.return_value = False
                mock_thread.return_value = mock_thread_instance

                provider.start()

                mock_thread.assert_called_once()
                mock_thread_instance.start.assert_called_once()
                assert not provider._stop_event.is_set()

    def test_start_already_running_warning(self):
        """Test start method logs warning when already running."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            mock_thread = Mock()
            mock_thread.is_alive.return_value = True
            provider._thread = mock_thread

            with patch(
                "providers.unitree_g1_locations_provider.logging"
            ) as mock_logging:
                provider.start()
                mock_logging.warning.assert_called_once()

    def test_stop_sets_event_and_joins_thread(self):
        """Test stop method sets stop event and joins thread."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            mock_thread = Mock()
            provider._thread = mock_thread

            provider.stop()

            assert provider._stop_event.is_set()
            mock_thread.join.assert_called_once_with(timeout=5)

    def test_stop_no_thread(self):
        """Test stop method works when no thread exists."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()
            provider.stop()  # Should not raise exception

    def test_fetch_success_with_dict_response(self):
        """Test _fetch method successfully processes dict response."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "location1": {"name": "Location 1", "pose": {"x": 1, "y": 2}},
                "location2": {"name": "Location 2", "pose": {"x": 3, "y": 4}},
            }

            with patch(
                "providers.unitree_g1_locations_provider.requests.get",
                return_value=mock_response,
            ):
                provider._fetch()

                locations = provider.get_all_locations()
                assert "location1" in locations
                assert "location2" in locations
                assert locations["location1"]["name"] == "Location 1"

    def test_fetch_success_with_message_json_string(self):
        """Test _fetch method successfully processes response with message as JSON string."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            locations_data = {
                "location1": {"name": "Location 1", "pose": {"x": 1, "y": 2}}
            }

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"message": json.dumps(locations_data)}

            with patch(
                "providers.unitree_g1_locations_provider.requests.get",
                return_value=mock_response,
            ):
                provider._fetch()

                locations = provider.get_all_locations()
                assert "location1" in locations

    def test_fetch_handles_http_error(self):
        """Test _fetch method handles HTTP error responses."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.text = "Not Found"

            with patch(
                "providers.unitree_g1_locations_provider.requests.get",
                return_value=mock_response,
            ):
                with patch(
                    "providers.unitree_g1_locations_provider.logging"
                ) as mock_logging:
                    provider._fetch()
                    mock_logging.error.assert_called()

    def test_fetch_handles_request_exception(self):
        """Test _fetch method handles request exceptions."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            with patch(
                "providers.unitree_g1_locations_provider.requests.get",
                side_effect=Exception("Connection error"),
            ):
                with patch(
                    "providers.unitree_g1_locations_provider.logging"
                ) as mock_logging:
                    provider._fetch()
                    mock_logging.exception.assert_called()

    def test_fetch_empty_base_url(self):
        """Test _fetch method returns early when base_url is empty."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider(base_url="")

            with patch(
                "providers.unitree_g1_locations_provider.requests.get"
            ) as mock_get:
                provider._fetch()
                mock_get.assert_not_called()

    def test_update_locations_with_dict(self):
        """Test _update_locations method processes dict data correctly."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            locations_data = {
                "Kitchen": {"name": "Kitchen", "pose": {"x": 1, "y": 2}},
                "Living Room": {"name": "Living Room", "pose": {"x": 3, "y": 4}},
            }

            provider._update_locations(locations_data)

            locations = provider.get_all_locations()
            assert "kitchen" in locations
            assert "living room" in locations

    def test_update_locations_with_list(self):
        """Test _update_locations method processes list data correctly."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            locations_data = [
                {"name": "Kitchen", "pose": {"x": 1, "y": 2}},
                {"label": "Bedroom", "pose": {"x": 3, "y": 4}},
            ]

            provider._update_locations(locations_data)

            locations = provider.get_all_locations()
            assert "kitchen" in locations
            assert "bedroom" in locations

    def test_update_locations_handles_missing_names(self):
        """Test _update_locations method handles items without names."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            locations_data = [
                {"pose": {"x": 1, "y": 2}},  # No name or label
                {"name": "Kitchen", "pose": {"x": 3, "y": 4}},
            ]

            provider._update_locations(locations_data)

            locations = provider.get_all_locations()
            assert "kitchen" in locations
            assert len(locations) == 1

    def test_get_all_locations_returns_copy(self):
        """Test get_all_locations returns a copy of locations."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            test_data = {"kitchen": {"name": "Kitchen", "pose": {"x": 1, "y": 2}}}
            provider._update_locations(test_data)

            locations1 = provider.get_all_locations()
            locations2 = provider.get_all_locations()

            assert locations1 is not locations2  # Different objects
            assert locations1 == locations2  # Same content

    def test_get_location_found(self):
        """Test get_location returns correct location when found."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            test_data = {"kitchen": {"name": "Kitchen", "pose": {"x": 1, "y": 2}}}
            provider._update_locations(test_data)

            location = provider.get_location("Kitchen")
            assert location is not None
            assert location["name"] == "Kitchen"

    def test_get_location_not_found(self):
        """Test get_location returns None when location not found."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            location = provider.get_location("NonExistent")
            assert location is None

    def test_get_location_empty_label(self):
        """Test get_location returns None for empty label."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            location = provider.get_location("")
            assert location is None

    def test_get_location_case_insensitive(self):
        """Test get_location is case insensitive."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            test_data = {"kitchen": {"name": "Kitchen", "pose": {"x": 1, "y": 2}}}
            provider._update_locations(test_data)

            location1 = provider.get_location("KITCHEN")
            location2 = provider.get_location("kitchen")
            location3 = provider.get_location("Kitchen")

            assert location1 == location2 == location3
            assert location1 is not None

    def test_run_method_periodic_execution(self):
        """Test _run method executes fetch periodically."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider(refresh_interval=0.1)

            fetch_call_count = 0
            original_fetch = provider._fetch

            def mock_fetch():
                nonlocal fetch_call_count
                fetch_call_count += 1
                if fetch_call_count >= 2:
                    provider._stop_event.set()

            provider._fetch = mock_fetch
            provider._run()

            assert fetch_call_count >= 2

    def test_run_method_handles_fetch_exception(self):
        """Test _run method continues after fetch exception."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider(refresh_interval=0.1)

            call_count = 0

            def mock_fetch():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Test exception")
                elif call_count >= 2:
                    provider._stop_event.set()

            provider._fetch = mock_fetch

            with patch("providers.unitree_g1_locations_provider.logging"):
                provider._run()

            assert call_count >= 2

    def test_thread_safety(self):
        """Test thread-safe access to locations."""
        from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider

        if hasattr(UnitreeG1LocationsProvider, "reset"):
            UnitreeG1LocationsProvider.reset()

        with patch("providers.unitree_g1_locations_provider.IOProvider"):
            provider = UnitreeG1LocationsProvider()

            # Test concurrent access doesn't raise exceptions
            def update_locations():
                test_data = {"location1": {"name": "Location1"}}
                provider._update_locations(test_data)

            def read_locations():
                provider.get_all_locations()
                provider.get_location("location1")

            threads = []
            for _ in range(5):
                t1 = threading.Thread(target=update_locations)
                t2 = threading.Thread(target=read_locations)
                threads.extend([t1, t2])

            for t in threads:
                t.start()

            for t in threads:
                t.join()
