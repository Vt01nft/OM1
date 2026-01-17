"""Tests for rtk_provider."""

import datetime
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
sys.modules["serial"] = MagicMock()
sys.modules["pynmeagps"] = MagicMock()


class TestRtkProvider:
    """Tests for RtkProvider class."""

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

    def test_initialization_successful_connection(self):
        """Test provider initializes correctly with successful serial connection."""
        with (
            patch("providers.rtk_provider.serial.Serial") as mock_serial,
            patch("threading.Thread") as mock_thread,
        ):
            mock_serial_instance = Mock()
            mock_serial.return_value = mock_serial_instance

            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider("/dev/ttyUSB0")

            assert provider is not None
            assert provider.lat == 0.0
            assert provider.lon == 0.0
            assert provider.alt == 0.0
            assert provider.sat == 0
            assert provider.qua == 0
            assert provider.unix_ts == 0.0
            assert provider.running is True
            mock_serial.assert_called_once_with("/dev/ttyUSB0", 115200, timeout=0.2)
            mock_serial_instance.reset_input_buffer.assert_called_once()

    def test_initialization_failed_connection(self):
        """Test provider handles failed serial connection gracefully."""
        with (
            patch("providers.rtk_provider.serial.Serial") as mock_serial,
            patch("threading.Thread") as mock_thread,
        ):
            from providers.rtk_provider import serial

            mock_serial.side_effect = serial.SerialException("Connection failed")

            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider("/dev/ttyUSB0")

            assert provider is not None
            assert provider.serial_connection is None
            assert provider.running is True

    def test_singleton_pattern(self):
        """Test provider follows singleton pattern."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread"),
        ):
            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider1 = RtkProvider("/dev/ttyUSB0")
            provider2 = RtkProvider("/dev/ttyUSB1")

            assert provider1 is provider2

    def test_utc_time_obj_to_unix_valid_time(self):
        """Test UTC time object conversion to Unix timestamp."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread"),
        ):
            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()

            test_time = datetime.time(12, 30, 45)
            result = provider.utc_time_obj_to_unix(test_time)

            assert isinstance(result, float)
            assert result > 0

    def test_utc_time_obj_to_unix_invalid_input(self):
        """Test UTC time conversion raises TypeError for invalid input."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread"),
        ):
            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()

            with pytest.raises(TypeError, match="Expected a datetime.time object"):
                provider.utc_time_obj_to_unix("not a time object")

    def test_get_latest_gngga_message_with_valid_data(self):
        """Test extracting latest GNGGA message from NMEA data."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread"),
        ):
            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()

            nmea_data = """$GNGGA,123000.00,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47
$GNGGA,123030.00,4807.039,N,01131.001,E,1,08,0.9,545.5,M,46.9,M,,*42"""

            result = provider.get_latest_gngga_message(nmea_data)

            assert result is not None
            assert "123030.00" in result
            assert "4807.039" in result

    def test_get_latest_gngga_message_with_no_valid_data(self):
        """Test extracting GNGGA message returns None when no valid data."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread"),
        ):
            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()

            nmea_data = "$GPGSV,3,1,11,03,03,111,00,04,15,270,00*74"

            result = provider.get_latest_gngga_message(nmea_data)

            assert result is None

    def test_get_latest_gngga_message_with_empty_data(self):
        """Test extracting GNGGA message from empty data."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread"),
        ):
            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()

            result = provider.get_latest_gngga_message("")

            assert result is None

    def test_magRTKProcessor_with_valid_gga_message(self):
        """Test processing valid GGA NMEA message."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread"),
        ):
            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()

            mock_msg = Mock()
            mock_msg.msgID = "GGA"
            mock_msg.lat = 48.1234567
            mock_msg.lon = 11.5678912
            mock_msg.alt = 545.4
            mock_msg.numSV = 8
            mock_msg.quality = 1
            mock_msg.time = datetime.time(12, 30, 45)

            provider.magRTKProcessor(mock_msg)

            assert provider.lat == round(48.1234567, 7)
            assert provider.lon == round(11.5678912, 7)
            assert provider.alt == 545.4
            assert provider.sat == 8
            assert provider.qua == 1
            assert provider._rtk is not None
            assert provider._rtk["rtk_lat"] == provider.lat

    def test_magRTKProcessor_with_non_gga_message(self):
        """Test processing non-GGA NMEA message."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread"),
        ):
            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()
            initial_lat = provider.lat

            mock_msg = Mock()
            mock_msg.msgID = "GSV"

            provider.magRTKProcessor(mock_msg)

            assert provider.lat == initial_lat
            assert provider._rtk is not None

    def test_magRTKProcessor_with_none_message(self):
        """Test processing None message."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread"),
        ):
            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()

            provider.magRTKProcessor(None)

            assert provider._rtk is not None

    def test_magRTKProcessor_with_invalid_gga_data(self):
        """Test processing GGA message with invalid data."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread"),
        ):
            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()

            mock_msg = Mock()
            mock_msg.msgID = "GGA"
            mock_msg.lat = None
            mock_msg.lon = None
            mock_msg.alt = None
            mock_msg.numSV = None
            mock_msg.quality = None
            mock_msg.time = None

            # Should not raise exception
            provider.magRTKProcessor(mock_msg)

            assert provider._rtk is not None

    def test_magRTKProcessor_with_exception_in_parsing(self):
        """Test processing GGA message that raises exception during parsing."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread"),
        ):
            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()

            mock_msg = Mock()
            mock_msg.msgID = "GGA"
            mock_msg.lat = "invalid"
            mock_msg.lon = "invalid"
            mock_msg.alt = "invalid"
            mock_msg.numSV = "invalid"
            mock_msg.quality = "invalid"
            mock_msg.time = "invalid"

            # Should not raise exception
            provider.magRTKProcessor(mock_msg)

            assert provider._rtk is not None

    def test_start_when_not_running(self):
        """Test starting provider when not already running."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread") as mock_thread,
        ):
            mock_thread_instance = Mock()
            mock_thread_instance.is_alive.return_value = False
            mock_thread.return_value = mock_thread_instance

            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()
            provider.running = False
            provider._thread = None

            provider.start()

            assert provider.running is True
            mock_thread.assert_called()
            mock_thread_instance.start.assert_called()

    def test_start_when_already_running(self):
        """Test starting provider when already running."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread") as mock_thread,
        ):
            mock_thread_instance = Mock()
            mock_thread_instance.is_alive.return_value = True

            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()
            provider._thread = mock_thread_instance

            provider.start()

            # Should not create new thread
            assert provider._thread is mock_thread_instance

    def test_run_with_serial_data(self):
        """Test main run loop processes serial data."""
        with (
            patch("providers.rtk_provider.serial.Serial") as mock_serial,
            patch("threading.Thread"),
            patch("providers.rtk_provider.NMEAReader") as mock_nmea,
        ):
            mock_serial_instance = Mock()
            mock_serial_instance.in_waiting = 100
            mock_serial_instance.read.return_value = (
                b"$GNGGA,123000.00,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
            )
            mock_serial.return_value = mock_serial_instance

            mock_parsed = Mock()
            mock_parsed.msgID = "GGA"
            mock_nmea.parse.return_value = mock_parsed

            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()
            provider.running = True

            # Mock the run loop to stop after one iteration
            with patch.object(provider, "running", side_effect=[True, False]):
                provider._run()

            mock_serial_instance.read.assert_called()

    def test_run_with_no_serial_connection(self):
        """Test main run loop when no serial connection."""
        with (
            patch("providers.rtk_provider.serial.Serial"),
            patch("threading.Thread"),
        ):
            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()
            provider.serial_connection = None

            # Mock the run loop to stop after one iteration
            with patch.object(provider, "running", side_effect=[True, False]):
                provider._run()

            # Should complete without error
            assert True

    def test_run_with_decode_error(self):
        """Test main run loop handles decode errors gracefully."""
        with (
            patch("providers.rtk_provider.serial.Serial") as mock_serial,
            patch("threading.Thread"),
        ):
            mock_serial_instance = Mock()
            mock_serial_instance.in_waiting = 100
            mock_serial_instance.read.return_value = b"\xff\xfe\xfd"  # Invalid UTF-8
            mock_serial.return_value = mock_serial_instance

            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()

            # Mock the run loop to stop after one iteration
            with patch.object(provider, "running", side_effect=[True, False]):
                provider._run()

            # Should complete without raising exception
            assert True

    def test_run_with_nmea_parse_error(self):
        """Test main run loop handles NMEA parse errors gracefully."""
        with (
            patch("providers.rtk_provider.serial.Serial") as mock_serial,
            patch("threading.Thread"),
            patch("providers.rtk_provider.NMEAReader") as mock_nmea,
        ):
            mock_serial_instance = Mock()
            mock_serial_instance.in_waiting = 100
            mock_serial_instance.read.return_value = (
                b"$GNGGA,123000.00,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
            )
            mock_serial.return_value = mock_serial_instance

            mock_nmea.parse.side_effect = Exception("Parse error")

            from providers.rtk_provider import RtkProvider

            if hasattr(RtkProvider, "reset"):
                RtkProvider.reset()

            provider = RtkProvider()

            # Mock the run loop to stop after one iteration
            with patch.object(provider, "running", side_effect=[True, False]):
                provider._run()

            # Should complete without raising exception
            assert True
