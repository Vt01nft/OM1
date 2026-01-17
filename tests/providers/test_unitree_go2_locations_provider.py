"""Tests for unitree_go2_locations_provider."""

import json
import sys
import threading
import time
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


class TestUnitreeGo2LocationsProvider:
    """Tests for UnitreeGo2LocationsProvider class."""

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

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton instances between tests."""
        # Import inside fixture to avoid module issues
        try:
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            if hasattr(UnitreeGo2LocationsProvider, "reset"):
                UnitreeGo2LocationsProvider.reset()
        except:
            pass
        yield
        try:
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            if hasattr(UnitreeGo2LocationsProvider, "reset"):
                UnitreeGo2LocationsProvider.reset()
        except:
            pass

    def test_initialization(self):
        """Test provider initializes correctly with default parameters."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            assert provider.base_url == "http://localhost:5000/maps/locations/list"
            assert provider.timeout == 5
            assert provider.refresh_interval == 30
            assert provider._locations == {}
            assert provider._thread is None
            assert provider._stop_event is not None
            assert provider._lock is not None

    def test_initialization_custom_parameters(self):
        """Test provider initializes correctly with custom parameters."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            custom_url = "http://custom.url:8080/api"
            custom_timeout = 10
            custom_interval = 60

            provider = UnitreeGo2LocationsProvider(
                base_url=custom_url,
                timeout=custom_timeout,
                refresh_interval=custom_interval,
            )

            assert provider.base_url == custom_url
            assert provider.timeout == custom_timeout
            assert provider.refresh_interval == custom_interval

    def test_singleton_pattern(self):
        """Test that provider follows singleton pattern."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider1 = UnitreeGo2LocationsProvider()
            provider2 = UnitreeGo2LocationsProvider()

            assert provider1 is provider2

    def test_start(self):
        """Test starting the background thread."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()
            provider.start()

            assert provider._thread is not None
            assert provider._thread.is_alive()
            assert not provider._stop_event.is_set()

            provider.stop()

    def test_start_already_running(self):
        """Test starting when already running logs warning."""
        with (
            patch("providers.unitree_go2_locations_provider.IOProvider"),
            patch("providers.unitree_go2_locations_provider.logging") as mock_logging,
        ):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()
            provider.start()

            # Start again - should log warning
            provider.start()

            mock_logging.warning.assert_called_with(
                "UnitreeGo2LocationsProvider already running"
            )

            provider.stop()

    def test_stop(self):
        """Test stopping the background thread."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()
            provider.start()

            provider.stop()

            assert provider._stop_event.is_set()

    def test_stop_without_thread(self):
        """Test stopping when no thread exists."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            # Should not raise exception
            provider.stop()

    @patch("providers.unitree_go2_locations_provider.requests")
    def test_fetch_success_dict_format(self, mock_requests):
        """Test successful fetch with dict format response."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "kitchen": {"name": "kitchen", "pose": {"x": 1, "y": 2}},
                "bedroom": {"name": "bedroom", "pose": {"x": 3, "y": 4}},
            }
            mock_requests.get.return_value = mock_response

            provider = UnitreeGo2LocationsProvider()
            provider._fetch()

            locations = provider.get_all_locations()
            assert "kitchen" in locations
            assert "bedroom" in locations
            assert locations["kitchen"]["name"] == "kitchen"

    @patch("providers.unitree_go2_locations_provider.requests")
    def test_fetch_success_nested_message(self, mock_requests):
        """Test successful fetch with nested message format."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            locations_data = {"kitchen": {"name": "kitchen", "pose": {"x": 1, "y": 2}}}
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"message": json.dumps(locations_data)}
            mock_requests.get.return_value = mock_response

            provider = UnitreeGo2LocationsProvider()
            provider._fetch()

            locations = provider.get_all_locations()
            assert "kitchen" in locations

    @patch("providers.unitree_go2_locations_provider.requests")
    def test_fetch_http_error(self, mock_requests):
        """Test fetch handles HTTP error status codes."""
        with (
            patch("providers.unitree_go2_locations_provider.IOProvider"),
            patch("providers.unitree_go2_locations_provider.logging") as mock_logging,
        ):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.text = "Not Found"
            mock_requests.get.return_value = mock_response

            provider = UnitreeGo2LocationsProvider()
            provider._fetch()

            mock_logging.error.assert_called_with(
                "Location list API returned 404: Not Found"
            )

    @patch("providers.unitree_go2_locations_provider.requests")
    def test_fetch_request_exception(self, mock_requests):
        """Test fetch handles request exceptions."""
        with (
            patch("providers.unitree_go2_locations_provider.IOProvider"),
            patch("providers.unitree_go2_locations_provider.logging") as mock_logging,
        ):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            mock_requests.get.side_effect = Exception("Connection error")

            provider = UnitreeGo2LocationsProvider()
            provider._fetch()

            mock_logging.exception.assert_called_with("Error fetching locations")

    def test_fetch_empty_base_url(self):
        """Test fetch returns early when base_url is empty."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider(base_url="")

            # Should not raise exception
            provider._fetch()

    @patch("providers.unitree_go2_locations_provider.requests")
    def test_fetch_invalid_json_message(self, mock_requests):
        """Test fetch handles invalid JSON in nested message."""
        with (
            patch("providers.unitree_go2_locations_provider.IOProvider"),
            patch("providers.unitree_go2_locations_provider.logging") as mock_logging,
        ):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"message": "invalid json"}
            mock_requests.get.return_value = mock_response

            provider = UnitreeGo2LocationsProvider()
            provider._fetch()

            mock_logging.error.assert_called_with(
                "Failed to parse nested message JSON from location list"
            )

    @patch("providers.unitree_go2_locations_provider.requests")
    def test_fetch_unexpected_format(self, mock_requests):
        """Test fetch handles unexpected response format."""
        with (
            patch("providers.unitree_go2_locations_provider.IOProvider"),
            patch("providers.unitree_go2_locations_provider.logging") as mock_logging,
        ):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"message": 123}  # Invalid format
            mock_requests.get.return_value = mock_response

            provider = UnitreeGo2LocationsProvider()
            provider._fetch()

            mock_logging.error.assert_called_with(
                "Unexpected format from location list API"
            )

    def test_update_locations_dict_format(self):
        """Test updating locations with dict format."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            locations_data = {
                "Kitchen": {"name": "Kitchen", "pose": {"x": 1, "y": 2}},
                "Bedroom": {"pose": {"x": 3, "y": 4}},  # Missing name
            }

            provider._update_locations(locations_data)

            locations = provider.get_all_locations()
            assert "kitchen" in locations
            assert "bedroom" in locations
            assert locations["kitchen"]["name"] == "Kitchen"
            assert locations["bedroom"]["name"] == "Bedroom"

    def test_update_locations_list_format(self):
        """Test updating locations with list format."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            locations_data = [
                {"name": "Kitchen", "pose": {"x": 1, "y": 2}},
                {"label": "Bedroom", "pose": {"x": 3, "y": 4}},
                {"pose": {"x": 5, "y": 6}},  # No name or label
                "invalid_item",  # Not a dict
            ]

            provider._update_locations(locations_data)

            locations = provider.get_all_locations()
            assert "kitchen" in locations
            assert "bedroom" in locations
            assert len(locations) == 2

    def test_get_all_locations_empty(self):
        """Test getting all locations when cache is empty."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            locations = provider.get_all_locations()

            assert locations == {}

    def test_get_all_locations_with_data(self):
        """Test getting all locations with cached data."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            test_data = {"kitchen": {"name": "kitchen"}}
            with provider._lock:
                provider._locations = test_data

            locations = provider.get_all_locations()

            assert locations == test_data
            # Ensure it returns a copy
            assert locations is not provider._locations

    def test_get_location_found(self):
        """Test getting a specific location that exists."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            test_data = {"kitchen": {"name": "Kitchen", "pose": {"x": 1, "y": 2}}}
            with provider._lock:
                provider._locations = test_data

            location = provider.get_location("Kitchen")

            assert location is not None
            assert location["name"] == "Kitchen"

    def test_get_location_case_insensitive(self):
        """Test getting location is case insensitive."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            test_data = {"kitchen": {"name": "Kitchen", "pose": {"x": 1, "y": 2}}}
            with provider._lock:
                provider._locations = test_data

            location = provider.get_location("KITCHEN")

            assert location is not None
            assert location["name"] == "Kitchen"

    def test_get_location_with_whitespace(self):
        """Test getting location handles whitespace."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            test_data = {"kitchen": {"name": "Kitchen", "pose": {"x": 1, "y": 2}}}
            with provider._lock:
                provider._locations = test_data

            location = provider.get_location("  Kitchen  ")

            assert location is not None
            assert location["name"] == "Kitchen"

    def test_get_location_not_found(self):
        """Test getting a location that doesn't exist."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            location = provider.get_location("nonexistent")

            assert location is None

    def test_get_location_empty_label(self):
        """Test getting location with empty label."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            location = provider.get_location("")

            assert location is None

    def test_get_location_none_label(self):
        """Test getting location with None label."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            location = provider.get_location(None)

            assert location is None

    @patch("providers.unitree_go2_locations_provider.requests")
    def test_run_background_thread(self, mock_requests):
        """Test the background thread runs and handles exceptions."""
        with (
            patch("providers.unitree_go2_locations_provider.IOProvider"),
            patch("providers.unitree_go2_locations_provider.logging") as mock_logging,
        ):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            # First call succeeds, second call fails
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"test": {"name": "test"}}
            mock_requests.get.side_effect = [mock_response, Exception("Network error")]

            provider = UnitreeGo2LocationsProvider(refresh_interval=0.1)
            provider.start()

            # Wait for at least two fetch attempts
            time.sleep(0.3)

            provider.stop()

            # Should have logged the exception from second call
            mock_logging.exception.assert_called_with("Error fetching locations")

    def test_thread_safety(self):
        """Test thread safety of location access."""
        with patch("providers.unitree_go2_locations_provider.IOProvider"):
            from providers.unitree_go2_locations_provider import (
                UnitreeGo2LocationsProvider,
            )

            provider = UnitreeGo2LocationsProvider()

            # Simulate concurrent access
            def update_locations():
                provider._update_locations({"test": {"name": "test"}})

            def read_locations():
                return provider.get_all_locations()

            # Start multiple threads
            threads = []
            for i in range(10):
                if i % 2 == 0:
                    t = threading.Thread(target=update_locations)
                else:
                    t = threading.Thread(target=read_locations)
                threads.append(t)
                t.start()

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Should not raise any exceptions
            locations = provider.get_all_locations()
            assert isinstance(locations, dict)
