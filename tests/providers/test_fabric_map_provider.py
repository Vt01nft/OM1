"""Tests for fabric_map_provider."""

import sys
from unittest.mock import MagicMock, mock_open, patch

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


class TestRFData:
    """Tests for RFData class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        yield

    def test_rf_data_initialization(self):
        """Test RFData initializes correctly."""
        from providers.fabric_map_provider import RFData

        rf_data = RFData(
            unix_ts=1234567890.0,
            address="AA:BB:CC:DD:EE:FF",
            name="Test Device",
            rssi=-45,
            tx_power=10,
            service_uuid="uuid-123",
            mfgkey="key123",
            mfgval="val456",
        )

        assert rf_data.unix_ts == 1234567890.0
        assert rf_data.address == "AA:BB:CC:DD:EE:FF"
        assert rf_data.name == "Test Device"
        assert rf_data.rssi == -45
        assert rf_data.tx_power == 10
        assert rf_data.service_uuid == "uuid-123"
        assert rf_data.mfgkey == "key123"
        assert rf_data.mfgval == "val456"

    def test_rf_data_to_dict(self):
        """Test RFData to_dict method."""
        from providers.fabric_map_provider import RFData

        rf_data = RFData(
            unix_ts=1234567890.0,
            address="AA:BB:CC:DD:EE:FF",
            name="Test Device",
            rssi=-45,
            tx_power=10,
            service_uuid="uuid-123",
            mfgkey="key123",
            mfgval="val456",
        )

        result = rf_data.to_dict()
        expected = {
            "unix_ts": 1234567890.0,
            "address": "AA:BB:CC:DD:EE:FF",
            "name": "Test Device",
            "rssi": -45,
            "tx_power": 10,
            "service_uuid": "uuid-123",
            "mfgkey": "key123",
            "mfgval": "val456",
        }

        assert result == expected

    def test_rf_data_with_none_values(self):
        """Test RFData with None values."""
        from providers.fabric_map_provider import RFData

        rf_data = RFData(
            unix_ts=1234567890.0,
            address="AA:BB:CC:DD:EE:FF",
            name=None,
            rssi=-45,
            tx_power=None,
            service_uuid="uuid-123",
            mfgkey="key123",
            mfgval="val456",
        )

        result = rf_data.to_dict()
        assert result["name"] is None
        assert result["tx_power"] is None


class TestRFDataRaw:
    """Tests for RFDataRaw class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        yield

    def test_rf_data_raw_initialization(self):
        """Test RFDataRaw initializes correctly."""
        from providers.fabric_map_provider import RFDataRaw

        rf_data_raw = RFDataRaw(
            unix_ts=1234567890.0,
            address="AA:BB:CC:DD:EE:FF",
            rssi=-45,
            packet="raw_packet_data",
        )

        assert rf_data_raw.unix_ts == 1234567890.0
        assert rf_data_raw.address == "AA:BB:CC:DD:EE:FF"
        assert rf_data_raw.rssi == -45
        assert rf_data_raw.packet == "raw_packet_data"

    def test_rf_data_raw_to_dict(self):
        """Test RFDataRaw to_dict method."""
        from providers.fabric_map_provider import RFDataRaw

        rf_data_raw = RFDataRaw(
            unix_ts=1234567890.0,
            address="AA:BB:CC:DD:EE:FF",
            rssi=-45,
            packet="raw_packet_data",
        )

        result = rf_data_raw.to_dict()
        expected = {
            "unix_ts": 1234567890.0,
            "address": "AA:BB:CC:DD:EE:FF",
            "rssi": -45,
            "packet": "raw_packet_data",
        }

        assert result == expected


class TestFabricData:
    """Tests for FabricData class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        yield

    def test_fabric_data_initialization(self):
        """Test FabricData initializes correctly."""
        from providers.fabric_map_provider import FabricData, RFData, RFDataRaw

        rf_data = [
            RFData(
                unix_ts=1234567890.0,
                address="AA:BB:CC:DD:EE:FF",
                name="Test Device",
                rssi=-45,
                tx_power=10,
                service_uuid="uuid-123",
                mfgkey="key123",
                mfgval="val456",
            )
        ]

        rf_data_raw = [
            RFDataRaw(
                unix_ts=1234567890.0,
                address="AA:BB:CC:DD:EE:FF",
                rssi=-45,
                packet="raw_packet_data",
            )
        ]

        fabric_data = FabricData(
            machine_id="test_machine",
            payload_idx=1,
            gps_unix_ts=1234567890.0,
            gps_lat=40.7128,
            gps_lon=-74.0060,
            gps_alt=10.5,
            gps_qua=1,
            rtk_unix_ts=1234567890.0,
            rtk_lat=40.7128,
            rtk_lon=-74.0060,
            rtk_alt=10.5,
            rtk_qua=1,
            mag=45.0,
            unix_ts=1234567890.0,
            odom_x=1.0,
            odom_y=2.0,
            odom_rockchip_ts=1234567890.0,
            odom_subscriber_ts=1234567890.0,
            odom_yaw_0_360=180.0,
            odom_yaw_m180_p180=0.0,
            rf_data=rf_data,
            rf_data_raw=rf_data_raw,
        )

        assert fabric_data.machine_id == "test_machine"
        assert fabric_data.payload_idx == 1
        assert len(fabric_data.rf_data) == 1
        assert len(fabric_data.rf_data_raw) == 1

    def test_fabric_data_to_dict(self):
        """Test FabricData to_dict method."""
        from providers.fabric_map_provider import FabricData, RFData, RFDataRaw

        rf_data = [
            RFData(
                unix_ts=1234567890.0,
                address="AA:BB:CC:DD:EE:FF",
                name="Test Device",
                rssi=-45,
                tx_power=10,
                service_uuid="uuid-123",
                mfgkey="key123",
                mfgval="val456",
            )
        ]

        rf_data_raw = [
            RFDataRaw(
                unix_ts=1234567890.0,
                address="AA:BB:CC:DD:EE:FF",
                rssi=-45,
                packet="raw_packet_data",
            )
        ]

        fabric_data = FabricData(
            machine_id="test_machine",
            payload_idx=1,
            gps_unix_ts=1234567890.0,
            gps_lat=40.7128,
            gps_lon=-74.0060,
            gps_alt=10.5,
            gps_qua=1,
            rtk_unix_ts=1234567890.0,
            rtk_lat=40.7128,
            rtk_lon=-74.0060,
            rtk_alt=10.5,
            rtk_qua=1,
            mag=45.0,
            unix_ts=1234567890.0,
            odom_x=1.0,
            odom_y=2.0,
            odom_rockchip_ts=1234567890.0,
            odom_subscriber_ts=1234567890.0,
            odom_yaw_0_360=180.0,
            odom_yaw_m180_p180=0.0,
            rf_data=rf_data,
            rf_data_raw=rf_data_raw,
        )

        result = fabric_data.to_dict()

        assert result["machine_id"] == "test_machine"
        assert result["payload_idx"] == 1
        assert len(result["rf_data"]) == 1
        assert len(result["rf_data_raw"]) == 1
        assert result["rf_data"][0]["address"] == "AA:BB:CC:DD:EE:FF"
        assert result["rf_data_raw"][0]["packet"] == "raw_packet_data"

    def test_fabric_data_with_empty_rf_data(self):
        """Test FabricData with empty RF data lists."""
        from providers.fabric_map_provider import FabricData

        fabric_data = FabricData(
            machine_id="test_machine",
            payload_idx=1,
            gps_unix_ts=1234567890.0,
            gps_lat=40.7128,
            gps_lon=-74.0060,
            gps_alt=10.5,
            gps_qua=1,
            rtk_unix_ts=1234567890.0,
            rtk_lat=40.7128,
            rtk_lon=-74.0060,
            rtk_alt=10.5,
            rtk_qua=1,
            mag=45.0,
            unix_ts=1234567890.0,
            odom_x=1.0,
            odom_y=2.0,
            odom_rockchip_ts=1234567890.0,
            odom_subscriber_ts=1234567890.0,
            odom_yaw_0_360=180.0,
            odom_yaw_m180_p180=0.0,
            rf_data=None,
            rf_data_raw=None,
        )

        result = fabric_data.to_dict()
        assert result["rf_data"] == []
        assert result["rf_data_raw"] == []


class TestFabricDataSubmitter:
    """Tests for FabricDataSubmitter class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        yield

    def test_fabric_data_submitter_initialization_default(self):
        """Test FabricDataSubmitter initializes with defaults."""
        from providers.fabric_map_provider import FabricDataSubmitter

        if hasattr(FabricDataSubmitter, "reset"):
            FabricDataSubmitter.reset()

        submitter = FabricDataSubmitter()

        assert submitter.api_key is None
        assert submitter.base_url == "https://api.openmind.org/api/core/fabric/submit"
        assert submitter.write_to_local_file is False
        assert submitter.filename_base == "dump/fabric"
        assert submitter.max_file_size_bytes == 1024 * 1024
        assert submitter.executor is not None

    def test_fabric_data_submitter_initialization_custom(self):
        """Test FabricDataSubmitter initializes with custom values."""
        from providers.fabric_map_provider import FabricDataSubmitter

        if hasattr(FabricDataSubmitter, "reset"):
            FabricDataSubmitter.reset()

        submitter = FabricDataSubmitter(
            api_key="test_key",
            base_url="https://custom.api.com/submit",
            write_to_local_file=True,
        )

        assert submitter.api_key == "test_key"
        assert submitter.base_url == "https://custom.api.com/submit"
        assert submitter.write_to_local_file is True

    def test_singleton_pattern(self):
        """Test FabricDataSubmitter singleton pattern."""
        from providers.fabric_map_provider import FabricDataSubmitter

        if hasattr(FabricDataSubmitter, "reset"):
            FabricDataSubmitter.reset()

        submitter1 = FabricDataSubmitter()
        submitter2 = FabricDataSubmitter()
        assert submitter1 is submitter2

    @patch("time.time")
    def test_update_filename(self, mock_time):
        """Test update_filename method."""
        from providers.fabric_map_provider import FabricDataSubmitter

        if hasattr(FabricDataSubmitter, "reset"):
            FabricDataSubmitter.reset()

        mock_time.return_value = 1234567890.123456

        submitter = FabricDataSubmitter()
        filename = submitter.update_filename()

        expected = "dump/fabric_1234567890_123456Z.jsonl"
        assert filename == expected

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("os.path.getsize")
    def test_write_dict_to_file_new_file(self, mock_getsize, mock_exists, mock_file):
        """Test write_dict_to_file with new file."""
        from providers.fabric_map_provider import FabricDataSubmitter

        if hasattr(FabricDataSubmitter, "reset"):
            FabricDataSubmitter.reset()

        mock_exists.return_value = False

        submitter = FabricDataSubmitter()
        test_data = {"key": "value", "number": 123}

        submitter.write_dict_to_file(test_data)

        mock_file.assert_called_once()
        handle = mock_file.return_value
        written_data = handle.write.call_args[0][0]
        assert "key" in written_data
        assert "value" in written_data

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("os.path.getsize")
    def test_write_dict_to_file_existing_small_file(
        self, mock_getsize, mock_exists, mock_file
    ):
        """Test write_dict_to_file with existing small file."""
        from providers.fabric_map_provider import FabricDataSubmitter

        if hasattr(FabricDataSubmitter, "reset"):
            FabricDataSubmitter.reset()

        mock_exists.return_value = True
        mock_getsize.return_value = 512  # Small file

        submitter = FabricDataSubmitter()
        test_data = {"key": "value"}

        submitter.write_dict_to_file(test_data)

        mock_file.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("time.time")
    def test_write_dict_to_file_large_file_rotation(
        self, mock_time, mock_getsize, mock_exists, mock_file
    ):
        """Test write_dict_to_file with large file that needs rotation."""
        from providers.fabric_map_provider import FabricDataSubmitter

        if hasattr(FabricDataSubmitter, "reset"):
            FabricDataSubmitter.reset()

        mock_exists.return_value = True
        mock_getsize.return_value = 2 * 1024 * 1024  # Large file
        mock_time.return_value = 1234567890.123456

        submitter = FabricDataSubmitter()
        test_data = {"key": "value"}

        submitter.write_dict_to_file(test_data)

        # Should create new filename due to file size
        expected_filename = "dump/fabric_1234567890_123456Z.jsonl"
        assert submitter.filename_current == expected_filename

    def test_write_dict_to_file_invalid_input(self):
        """Test write_dict_to_file with invalid input."""
        from providers.fabric_map_provider import FabricDataSubmitter

        if hasattr(FabricDataSubmitter, "reset"):
            FabricDataSubmitter.reset()

        submitter = FabricDataSubmitter()

        with pytest.raises(ValueError, match="Provided data must be a dictionary"):
            submitter.write_dict_to_file("not a dict")

    def test_write_dict_to_file_list_input(self):
        """Test write_dict_to_file with list input."""
        from providers.fabric_map_provider import FabricDataSubmitter

        if hasattr(FabricDataSubmitter, "reset"):
            FabricDataSubmitter.reset()

        submitter = FabricDataSubmitter()

        with pytest.raises(ValueError, match="Provided data must be a dictionary"):
            submitter.write_dict_to_file([1, 2, 3])

    def test_write_dict_to_file_none_input(self):
        """Test write_dict_to_file with None input."""
        from providers.fabric_map_provider import FabricDataSubmitter

        if hasattr(FabricDataSubmitter, "reset"):
            FabricDataSubmitter.reset()

        submitter = FabricDataSubmitter()

        with pytest.raises(ValueError, match="Provided data must be a dictionary"):
            submitter.write_dict_to_file(None)
