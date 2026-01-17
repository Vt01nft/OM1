"""Tests for d435_provider."""

import logging
import math
import sys
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


class TestD435Provider:
    """Tests for D435Provider class."""

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

    @pytest.fixture
    def mock_zenoh_session(self):
        """Mock zenoh session and related components."""
        mock_session = MagicMock()
        mock_sample = MagicMock()
        mock_payload = MagicMock()
        mock_payload.to_bytes.return_value = b"mock_bytes"
        mock_sample.payload = mock_payload

        mock_point = MagicMock()
        mock_point.x = 1.0
        mock_point.y = 2.0
        mock_point.z = 3.0

        mock_points = MagicMock()
        mock_points.points = [mock_point]

        with (
            patch(
                "providers.d435_provider.open_zenoh_session", return_value=mock_session
            ),
            patch(
                "providers.d435_provider.sensor_msgs.PointCloud.deserialize",
                return_value=mock_points,
            ),
        ):
            yield mock_session, mock_sample, mock_points

    def test_initialization_success(self, mock_zenoh_session):
        """Test provider initializes correctly with successful zenoh connection."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        mock_session, _, _ = mock_zenoh_session
        provider = D435Provider()

        assert provider is not None
        assert provider.obstacle == []
        assert provider.running is True
        assert provider.session == mock_session
        mock_session.declare_subscriber.assert_called_once()

    def test_initialization_zenoh_error(self, caplog):
        """Test provider handles zenoh connection error gracefully."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        with patch(
            "providers.d435_provider.open_zenoh_session",
            side_effect=Exception("Connection failed"),
        ):
            with caplog.at_level(logging.ERROR):
                provider = D435Provider()

                assert provider is not None
                assert provider.running is True
                assert "Error opening Zenoh client: Connection failed" in caplog.text

    def test_singleton_pattern(self, mock_zenoh_session):
        """Test that D435Provider follows singleton pattern."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        provider1 = D435Provider()
        provider2 = D435Provider()
        assert provider1 is provider2

    def test_calculate_angle_and_distance_positive_coordinates(self):
        """Test calculate_angle_and_distance with positive coordinates."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        with patch("providers.d435_provider.open_zenoh_session"):
            provider = D435Provider()

            angle, distance = provider.calculate_angle_and_distance(3.0, 4.0)

            expected_distance = math.sqrt(3.0**2 + 4.0**2)
            expected_angle = math.degrees(math.atan2(4.0, 3.0))

            assert abs(distance - expected_distance) < 1e-10
            assert abs(angle - expected_angle) < 1e-10

    def test_calculate_angle_and_distance_negative_coordinates(self):
        """Test calculate_angle_and_distance with negative coordinates."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        with patch("providers.d435_provider.open_zenoh_session"):
            provider = D435Provider()

            angle, distance = provider.calculate_angle_and_distance(-3.0, -4.0)

            expected_distance = math.sqrt((-3.0) ** 2 + (-4.0) ** 2)
            expected_angle = math.degrees(math.atan2(-4.0, -3.0))

            assert abs(distance - expected_distance) < 1e-10
            assert abs(angle - expected_angle) < 1e-10

    def test_calculate_angle_and_distance_zero_coordinates(self):
        """Test calculate_angle_and_distance with zero coordinates."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        with patch("providers.d435_provider.open_zenoh_session"):
            provider = D435Provider()

            angle, distance = provider.calculate_angle_and_distance(0.0, 0.0)

            assert distance == 0.0
            assert angle == 0.0

    def test_obstacle_callback_success(self, mock_zenoh_session):
        """Test obstacle_callback processes point cloud data successfully."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        mock_session, mock_sample, mock_points = mock_zenoh_session
        provider = D435Provider()

        provider.obstacle_callback(mock_sample)

        assert len(provider.obstacle) == 1
        obstacle = provider.obstacle[0]
        assert obstacle["x"] == 1.0
        assert obstacle["y"] == 2.0
        assert obstacle["z"] == 3.0
        assert "angle" in obstacle
        assert "distance" in obstacle

    def test_obstacle_callback_error_handling(self, mock_zenoh_session, caplog):
        """Test obstacle_callback handles errors gracefully."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        mock_session, mock_sample, _ = mock_zenoh_session
        provider = D435Provider()

        with patch(
            "providers.d435_provider.sensor_msgs.PointCloud.deserialize",
            side_effect=Exception("Deserialization failed"),
        ):
            with caplog.at_level(logging.ERROR):
                provider.obstacle_callback(mock_sample)

                assert (
                    "Error processing obstacle info: Deserialization failed"
                    in caplog.text
                )

    def test_start_when_not_running(self, mock_zenoh_session, caplog):
        """Test start method when provider is not running."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        provider = D435Provider()
        provider.running = False

        with caplog.at_level(logging.INFO):
            provider.start()

            assert provider.running is True

    def test_start_when_already_running(self, mock_zenoh_session, caplog):
        """Test start method when provider is already running."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        provider = D435Provider()
        provider.running = True

        with caplog.at_level(logging.INFO):
            provider.start()

            assert provider.running is True
            assert "D435Provider is already running" in caplog.text

    def test_stop_when_running(self, mock_zenoh_session, caplog):
        """Test stop method when provider is running."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        mock_session, _, _ = mock_zenoh_session
        provider = D435Provider()
        provider.running = True

        with caplog.at_level(logging.INFO):
            provider.stop()

            assert provider.running is False
            mock_session.close.assert_called_once()
            assert "D435Provider stopped and Zenoh session closed" in caplog.text

    def test_stop_when_not_running(self, mock_zenoh_session, caplog):
        """Test stop method when provider is not running."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        provider = D435Provider()
        provider.running = False

        with caplog.at_level(logging.INFO):
            provider.stop()

            assert provider.running is False
            assert "D435Provider is not running" in caplog.text

    def test_stop_with_no_session(self, caplog):
        """Test stop method when session is None."""
        from providers.d435_provider import D435Provider

        if hasattr(D435Provider, "reset"):
            D435Provider.reset()

        with patch("providers.d435_provider.open_zenoh_session", return_value=None):
            provider = D435Provider()
            provider.running = True
            provider.session = None

            with caplog.at_level(logging.INFO):
                provider.stop()

                assert provider.running is False
                assert "D435Provider stopped and Zenoh session closed" in caplog.text
