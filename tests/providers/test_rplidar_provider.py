"""
Tests for rplidar_provider.

BOSS LEVEL TEST - 30KB provider with complex import chains.
Following CONTRIBUTING.md guidelines.
"""

import math
import sys
from unittest.mock import MagicMock

import pytest

# =============================================================================
# CRITICAL: Mock ALL external dependencies BEFORE any imports
# =============================================================================

# Mock serial FIRST (rplidar_driver needs this)
sys.modules["serial"] = MagicMock()

# Mock zenoh and related
mock_zenoh = MagicMock()
mock_zenoh.Sample = MagicMock()
sys.modules["zenoh"] = mock_zenoh

# Mock zenoh_msgs
mock_zenoh_msgs = MagicMock()
mock_zenoh_msgs.LaserScan = MagicMock()
mock_zenoh_msgs.open_zenoh_session = MagicMock(return_value=MagicMock())
mock_zenoh_msgs.sensor_msgs = MagicMock()
mock_zenoh_msgs.sensor_msgs.LaserScan = MagicMock()
mock_zenoh_msgs.sensor_msgs.LaserScan.deserialize = MagicMock()
sys.modules["zenoh_msgs"] = mock_zenoh_msgs

# Mock runtime.logging
mock_runtime_logging = MagicMock()
mock_runtime_logging.LoggingConfig = MagicMock()
mock_runtime_logging.get_logging_config = MagicMock(return_value=None)
mock_runtime_logging.setup_logging = MagicMock()
sys.modules["runtime"] = MagicMock()
sys.modules["runtime.logging"] = mock_runtime_logging

# Mock rplidar_driver
mock_rpdriver_class = MagicMock()
mock_rpdriver_module = MagicMock()
mock_rpdriver_module.RPDriver = mock_rpdriver_class
sys.modules["providers.rplidar_driver"] = mock_rpdriver_module

# Mock odom_provider
mock_odom_instance = MagicMock()
mock_odom_instance.position = None
mock_odom_class = MagicMock(return_value=mock_odom_instance)
mock_odom_module = MagicMock()
mock_odom_module.OdomProvider = mock_odom_class
sys.modules["providers.odom_provider"] = mock_odom_module

# Mock d435_provider
mock_d435_instance = MagicMock()
mock_d435_instance.running = False
mock_d435_instance.obstacle = []
mock_d435_class = MagicMock(return_value=mock_d435_instance)
mock_d435_module = MagicMock()
mock_d435_module.D435Provider = mock_d435_class
sys.modules["providers.d435_provider"] = mock_d435_module

# Store singleton instances for reset
_singleton_instances = {}


def mock_singleton(cls):
    """Mock singleton decorator that allows reset."""

    def get_instance(*args, **kwargs):
        if cls not in _singleton_instances:
            _singleton_instances[cls] = object.__new__(cls)
            _singleton_instances[cls].__init__(*args, **kwargs)
        return _singleton_instances[cls]

    get_instance._class = cls
    get_instance.reset = lambda: _singleton_instances.pop(cls, None)
    return get_instance


mock_singleton_module = MagicMock()
mock_singleton_module.singleton = mock_singleton
sys.modules["providers.singleton"] = mock_singleton_module


def reset_all_singletons():
    """Reset all singleton instances."""
    _singleton_instances.clear()
    keys_to_remove = [k for k in sys.modules.keys() if "rplidar_provider" in k]
    for key in keys_to_remove:
        del sys.modules[key]


class TestRPLidarConfig:
    """Tests for RPLidarConfig dataclass."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset before each test."""
        reset_all_singletons()
        yield

    def test_config_default_max_buf_meas(self):
        """Test RPLidarConfig default max_buf_meas is 0."""
        from providers.rplidar_provider import RPLidarConfig

        config = RPLidarConfig()
        assert config.max_buf_meas == 0

    def test_config_default_min_len(self):
        """Test RPLidarConfig default min_len is 5."""
        from providers.rplidar_provider import RPLidarConfig

        config = RPLidarConfig()
        assert config.min_len == 5

    def test_config_default_max_distance_mm(self):
        """Test RPLidarConfig default max_distance_mm is 10000."""
        from providers.rplidar_provider import RPLidarConfig

        config = RPLidarConfig()
        assert config.max_distance_mm == 10000

    def test_config_custom_max_buf_meas(self):
        """Test RPLidarConfig accepts custom max_buf_meas."""
        from providers.rplidar_provider import RPLidarConfig

        config = RPLidarConfig(max_buf_meas=100)
        assert config.max_buf_meas == 100

    def test_config_custom_min_len(self):
        """Test RPLidarConfig accepts custom min_len."""
        from providers.rplidar_provider import RPLidarConfig

        config = RPLidarConfig(min_len=20)
        assert config.min_len == 20

    def test_config_custom_max_distance_mm(self):
        """Test RPLidarConfig accepts custom max_distance_mm."""
        from providers.rplidar_provider import RPLidarConfig

        config = RPLidarConfig(max_distance_mm=5000)
        assert config.max_distance_mm == 5000

    def test_config_all_custom_values(self):
        """Test RPLidarConfig with all custom values."""
        from providers.rplidar_provider import RPLidarConfig

        config = RPLidarConfig(
            max_buf_meas=50, min_len=10, max_distance_mm=8000
        )
        assert config.max_buf_meas == 50
        assert config.min_len == 10
        assert config.max_distance_mm == 8000


class TestRPLidarProviderConstants:
    """Tests for RPLidarProvider class constants via instance."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset before each test."""
        reset_all_singletons()
        yield

    def test_degrees_to_radians_value(self):
        """Test DEGREES_TO_RADIANS is mathematically correct."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        expected = math.pi / 180.0
        assert provider.DEGREES_TO_RADIANS == pytest.approx(expected)

    def test_radians_to_degrees_value(self):
        """Test RADIANS_TO_DEGREES is mathematically correct."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        expected = 180.0 / math.pi
        assert provider.RADIANS_TO_DEGREES == pytest.approx(expected)

    def test_conversion_constants_are_inverses(self):
        """Test DEG_TO_RAD and RAD_TO_DEG are inverses."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        product = provider.DEGREES_TO_RADIANS * provider.RADIANS_TO_DEGREES
        assert product == pytest.approx(1.0)

    def test_default_serial_port_value(self):
        """Test DEFAULT_SERIAL_PORT constant value."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.DEFAULT_SERIAL_PORT == "/dev/cu.usbserial-0001"

    def test_default_half_width_robot_value(self):
        """Test DEFAULT_HALF_WIDTH_ROBOT constant value."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.DEFAULT_HALF_WIDTH_ROBOT == 0.20

    def test_default_relevant_distance_max_value(self):
        """Test DEFAULT_RELEVANT_DISTANCE_MAX constant value."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.DEFAULT_RELEVANT_DISTANCE_MAX == 1.1

    def test_default_relevant_distance_min_value(self):
        """Test DEFAULT_RELEVANT_DISTANCE_MIN constant value."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.DEFAULT_RELEVANT_DISTANCE_MIN == 0.08

    def test_default_sensor_mounting_angle_value(self):
        """Test DEFAULT_SENSOR_MOUNTING_ANGLE constant value."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.DEFAULT_SENSOR_MOUNTING_ANGLE == 180.0

    def test_num_bezier_points_value(self):
        """Test NUM_BEZIER_POINTS constant value."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.NUM_BEZIER_POINTS == 10


class TestRPLidarProviderInitialization:
    """Tests for RPLidarProvider initialization."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset before each test."""
        reset_all_singletons()
        yield

    def test_provider_instantiation(self):
        """Test RPLidarProvider can be instantiated."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider is not None

    def test_default_serial_port_assignment(self):
        """Test default serial port is assigned correctly."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.serial_port == "/dev/cu.usbserial-0001"

    def test_custom_serial_port_assignment(self):
        """Test custom serial port is assigned correctly."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider(serial_port="/dev/ttyUSB0")
        assert provider.serial_port == "/dev/ttyUSB0"

    def test_default_half_width_robot_assignment(self):
        """Test default half_width_robot is assigned correctly."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.half_width_robot == 0.20

    def test_custom_half_width_robot_assignment(self):
        """Test custom half_width_robot is assigned correctly."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider(half_width_robot=0.30)
        assert provider.half_width_robot == 0.30

    def test_running_initially_false(self):
        """Test running flag is initially False."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.running is False

    def test_raw_scan_initially_none(self):
        """Test _raw_scan is initially None."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider._raw_scan is None

    def test_valid_paths_initially_none(self):
        """Test _valid_paths is initially None."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider._valid_paths is None

    def test_lidar_string_initially_none(self):
        """Test _lidar_string is initially None."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider._lidar_string is None

    def test_path_angles_initialization(self):
        """Test path_angles is initialized correctly."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        expected = [-60, -45, -30, -15, 0, 15, 30, 45, 60, 180]
        assert provider.path_angles == expected

    def test_paths_count(self):
        """Test correct number of paths are created."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert len(provider.paths) == 10

    def test_use_zenoh_default_false(self):
        """Test use_zenoh defaults to False."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.use_zenoh is False

    def test_use_zenoh_can_be_true(self):
        """Test use_zenoh can be set to True."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider(use_zenoh=True)
        assert provider.use_zenoh is True

    def test_machine_type_default_go2(self):
        """Test machine_type defaults to go2."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.machine_type == "go2"

    def test_machine_type_can_be_tb4(self):
        """Test machine_type can be set to tb4."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider(machine_type="tb4")
        assert provider.machine_type == "tb4"

    def test_simple_paths_default_false(self):
        """Test simple_paths defaults to False."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.simple_paths is False

    def test_log_file_default_false(self):
        """Test log_file defaults to False."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.log_file is False

    def test_turn_left_initially_empty(self):
        """Test turn_left is initially empty list."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.turn_left == []

    def test_turn_right_initially_empty(self):
        """Test turn_right is initially empty list."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.turn_right == []

    def test_advance_initially_empty(self):
        """Test advance is initially empty list."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.advance == []

    def test_retreat_initially_false(self):
        """Test retreat is initially False."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.retreat is False


class TestRPLidarProviderProperties:
    """Tests for RPLidarProvider properties."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset before each test."""
        reset_all_singletons()
        yield

    def test_valid_paths_property_returns_none(self):
        """Test valid_paths property returns None initially."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.valid_paths is None

    def test_raw_scan_property_returns_none(self):
        """Test raw_scan property returns None initially."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.raw_scan is None

    def test_lidar_string_property_returns_none(self):
        """Test lidar_string property returns None initially."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert provider.lidar_string is None

    def test_movement_options_has_turn_left(self):
        """Test movement_options contains turn_left key."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert "turn_left" in provider.movement_options

    def test_movement_options_has_turn_right(self):
        """Test movement_options contains turn_right key."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert "turn_right" in provider.movement_options

    def test_movement_options_has_advance(self):
        """Test movement_options contains advance key."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert "advance" in provider.movement_options

    def test_movement_options_has_retreat(self):
        """Test movement_options contains retreat key."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert "retreat" in provider.movement_options

    def test_movement_options_turn_left_is_list(self):
        """Test movement_options turn_left is a list."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert isinstance(provider.movement_options["turn_left"], list)

    def test_movement_options_retreat_is_bool(self):
        """Test movement_options retreat is a bool."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert isinstance(provider.movement_options["retreat"], bool)


class TestRPLidarProviderMethods:
    """Tests for RPLidarProvider methods."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset before each test."""
        reset_all_singletons()
        yield

    def test_start_sets_running_true(self):
        """Test start method sets running to True."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider(use_zenoh=True)
        provider.start()
        assert provider.running is True

    def test_stop_sets_running_false(self):
        """Test stop method sets running to False."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider(use_zenoh=True)
        provider.start()
        provider.stop()
        assert provider.running is False

    def test_write_str_to_file_rejects_dict(self):
        """Test write_str_to_file raises ValueError for dict."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        with pytest.raises(ValueError):
            provider.write_str_to_file({"key": "value"})

    def test_write_str_to_file_rejects_list(self):
        """Test write_str_to_file raises ValueError for list."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        with pytest.raises(ValueError):
            provider.write_str_to_file([1, 2, 3])

    def test_write_str_to_file_rejects_int(self):
        """Test write_str_to_file raises ValueError for int."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        with pytest.raises(ValueError):
            provider.write_str_to_file(12345)

    def test_generate_movement_string_empty_returns_warning(self):
        """Test _generate_movement_string with empty paths returns warning."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        result = provider._generate_movement_string([])
        assert "DO NOT MOVE" in result

    def test_generate_movement_string_empty_mentions_surrounded(self):
        """Test _generate_movement_string mentions surrounded."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        result = provider._generate_movement_string([])
        assert "surrounded" in result.lower()

    def test_generate_movement_string_with_paths_mentions_safe(self):
        """Test _generate_movement_string with paths mentions safe."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        provider.advance = [4]
        result = provider._generate_movement_string([4])
        assert "safe" in result.lower()


class TestRPLidarProviderPathUtilities:
    """Tests for path utility methods."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset before each test."""
        reset_all_singletons()
        yield

    def test_distance_point_on_line_is_zero(self):
        """Test distance is 0 when point is on line segment."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        dist = provider.distance_point_to_line_segment(0.5, 0, 0, 0, 1, 0)
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_distance_perpendicular_point(self):
        """Test distance for perpendicular point is correct."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        dist = provider.distance_point_to_line_segment(0.5, 1, 0, 0, 1, 0)
        assert dist == pytest.approx(1.0, abs=1e-10)

    def test_distance_zero_length_segment(self):
        """Test distance when segment has zero length."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        dist = provider.distance_point_to_line_segment(3, 4, 0, 0, 0, 0)
        assert dist == pytest.approx(5.0, abs=1e-10)

    def test_distance_closest_to_start_endpoint(self):
        """Test distance when closest point is start."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        dist = provider.distance_point_to_line_segment(-1, 0, 0, 0, 1, 0)
        assert dist == pytest.approx(1.0, abs=1e-10)

    def test_distance_closest_to_end_endpoint(self):
        """Test distance when closest point is end."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        dist = provider.distance_point_to_line_segment(2, 0, 0, 0, 1, 0)
        assert dist == pytest.approx(1.0, abs=1e-10)

    def test_create_path_zero_degrees_shape(self):
        """Test path at 0 degrees has correct shape."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        path = provider._create_straight_path_from_angle(
            0, length=1.0, num_points=10
        )
        assert path.shape == (2, 10)

    def test_create_path_zero_degrees_x_starts_at_zero(self):
        """Test path at 0 degrees x starts at 0."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        path = provider._create_straight_path_from_angle(
            0, length=1.0, num_points=10
        )
        assert path[0][0] == pytest.approx(0.0, abs=1e-10)

    def test_create_path_zero_degrees_y_ends_at_one(self):
        """Test path at 0 degrees y ends at 1.0."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        path = provider._create_straight_path_from_angle(
            0, length=1.0, num_points=10
        )
        assert path[1][-1] == pytest.approx(1.0, abs=1e-10)

    def test_create_path_90_degrees_x_ends_at_one(self):
        """Test path at 90 degrees x ends at 1.0."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        path = provider._create_straight_path_from_angle(
            90, length=1.0, num_points=10
        )
        assert path[0][-1] == pytest.approx(1.0, abs=1e-10)

    def test_create_path_90_degrees_y_ends_near_zero(self):
        """Test path at 90 degrees y ends near 0."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        path = provider._create_straight_path_from_angle(
            90, length=1.0, num_points=10
        )
        assert path[1][-1] == pytest.approx(0.0, abs=1e-10)

    def test_create_path_180_degrees_y_ends_negative(self):
        """Test path at 180 degrees y ends at -1.0."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        path = provider._create_straight_path_from_angle(
            180, length=1.0, num_points=10
        )
        assert path[1][-1] == pytest.approx(-1.0, abs=1e-10)

    def test_initialize_paths_returns_list(self):
        """Test _initialize_paths returns a list."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert isinstance(provider.paths, list)

    def test_initialize_paths_creates_ten_paths(self):
        """Test _initialize_paths creates exactly 10 paths."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert len(provider.paths) == 10

    def test_paths_match_path_angles_count(self):
        """Test paths count matches path_angles count."""
        from providers.rplidar_provider import RPLidarProvider

        provider = RPLidarProvider()
        assert len(provider.paths) == len(provider.path_angles)
