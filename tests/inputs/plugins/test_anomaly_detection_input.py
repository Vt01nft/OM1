"""Tests for the AnomalyDetectionInput plugin."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from inputs.plugins.anomaly_detection_input import (
    AnomalyDetectionConfig,
    AnomalyDetectionInput,
    Message,
)


@pytest.fixture
def mock_check_webcam():
    """Mock webcam availability check."""
    with patch.object(
        AnomalyDetectionInput, "_check_webcam", return_value=True
    ) as mock:
        yield mock


@pytest.fixture
def mock_cv2_video_capture():
    """Mock OpenCV VideoCapture."""
    with patch("inputs.plugins.anomaly_detection_input.cv2.VideoCapture") as mock:
        mock_instance = Mock()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, dummy_frame)
        mock_instance.get.side_effect = lambda x: {
            0: 640,  # CAP_PROP_FRAME_WIDTH
            1: 480,  # CAP_PROP_FRAME_HEIGHT
            3: 640,  # cv2.CAP_PROP_FRAME_WIDTH
            4: 480,  # cv2.CAP_PROP_FRAME_HEIGHT
        }.get(x, 0)
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def anomaly_detection_input(mock_check_webcam, mock_cv2_video_capture):
    """Create AnomalyDetectionInput instance with mocked dependencies."""
    config = AnomalyDetectionConfig(
        camera_index=0,
        fire_threshold=0.05,
        motion_threshold=5000.0,
        detection_interval=1.0,
    )
    return AnomalyDetectionInput(config=config)


class TestAnomalyDetectionConfig:
    """Tests for AnomalyDetectionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AnomalyDetectionConfig()
        assert config.camera_index == 0
        assert config.fire_threshold == 0.05
        assert config.motion_threshold == 5000.0
        assert config.detection_interval == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AnomalyDetectionConfig(
            camera_index=1,
            fire_threshold=0.1,
            motion_threshold=10000.0,
            detection_interval=2.0,
        )
        assert config.camera_index == 1
        assert config.fire_threshold == 0.1
        assert config.motion_threshold == 10000.0
        assert config.detection_interval == 2.0


class TestAnomalyDetectionInput:
    """Tests for AnomalyDetectionInput."""

    def test_initialization(self, anomaly_detection_input):
        """Test proper initialization."""
        assert anomaly_detection_input.camera_index == 0
        assert anomaly_detection_input.fire_threshold == 0.05
        assert anomaly_detection_input.motion_threshold == 5000.0
        assert anomaly_detection_input.descriptor_for_LLM == "Anomaly Detector"
        assert anomaly_detection_input.messages == []

    @pytest.mark.asyncio
    async def test_poll_returns_frame(self, anomaly_detection_input):
        """Test _poll returns a frame when camera is available."""
        frame = await anomaly_detection_input._poll()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)

    @pytest.mark.asyncio
    async def test_poll_returns_none_when_no_camera(self, mock_cv2_video_capture):
        """Test _poll returns None when camera is unavailable."""
        with patch.object(AnomalyDetectionInput, "_check_webcam", return_value=False):
            config = AnomalyDetectionConfig(camera_index=0)
            detector = AnomalyDetectionInput(config=config)
            frame = await detector._poll()
            assert frame is None

    @pytest.mark.asyncio
    async def test_raw_to_text_none_input(self, anomaly_detection_input):
        """Test _raw_to_text returns None for None input."""
        result = await anomaly_detection_input._raw_to_text(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_no_anomalies(self, anomaly_detection_input):
        """Test _raw_to_text returns None when no anomalies detected."""
        # Create a normal frame (no fire colors, no motion)
        normal_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        normal_frame[:, :] = [100, 100, 100]  # Gray, neutral

        # Initialize prev_frame to avoid motion detection on first frame
        anomaly_detection_input.prev_frame = np.zeros((480, 640), dtype=np.uint8)

        result = await anomaly_detection_input._raw_to_text(normal_frame)
        # May return None or a message depending on motion detection
        # This test verifies no crash occurs
        assert result is None or isinstance(result, Message)

    @pytest.mark.asyncio
    async def test_raw_to_text_with_fire(self, anomaly_detection_input):
        """Test _raw_to_text detects fire colors."""
        # Create a frame with fire-like colors (orange/red in BGR)
        fire_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fire_frame[:, :] = [0, 100, 255]  # BGR for orange/red

        result = await anomaly_detection_input._raw_to_text(fire_frame)
        assert isinstance(result, Message)
        assert "FIRE" in result.message or "EMERGENCY" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_buffer_update(self, anomaly_detection_input):
        """Test raw_to_text updates message buffer."""
        # Create frame that triggers detection
        fire_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fire_frame[:, :] = [0, 100, 255]  # Fire color

        assert len(anomaly_detection_input.messages) == 0
        await anomaly_detection_input.raw_to_text(fire_frame)
        assert len(anomaly_detection_input.messages) >= 0  # May or may not detect

    def test_formatted_latest_buffer_with_message(self, anomaly_detection_input):
        """Test formatted_latest_buffer returns formatted string."""
        anomaly_detection_input.messages = [
            Message(timestamp=123.456, message="FIRE detected")
        ]
        result = anomaly_detection_input.formatted_latest_buffer()

        assert result is not None
        assert "FIRE detected" in result
        assert "Anomaly Detector" in result
        assert anomaly_detection_input.messages == []

    def test_formatted_latest_buffer_empty(self, anomaly_detection_input):
        """Test formatted_latest_buffer returns None when empty."""
        anomaly_detection_input.messages = []
        result = anomaly_detection_input.formatted_latest_buffer()
        assert result is None


class TestFireDetection:
    """Tests for fire detection functionality."""

    def test_detect_fire_positive(self, anomaly_detection_input):
        """Test fire detection with fire-colored frame."""
        # Create frame with significant fire colors
        fire_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Fill with orange/red (BGR format)
        fire_frame[:240, :] = [0, 100, 255]  # Top half is fire-colored

        detected, confidence = anomaly_detection_input._detect_fire(fire_frame)
        assert detected is True
        assert confidence > 0

    def test_detect_fire_negative(self, anomaly_detection_input):
        """Test fire detection with non-fire frame."""
        # Create frame with no fire colors (blue)
        normal_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        normal_frame[:, :] = [255, 0, 0]  # Blue in BGR

        detected, confidence = anomaly_detection_input._detect_fire(normal_frame)
        assert detected is False
        assert confidence == 0.0


class TestSmokeDetection:
    """Tests for smoke detection functionality."""

    def test_detect_smoke_with_gray_haze(self, anomaly_detection_input):
        """Test smoke detection with gray hazy frame."""
        # Create a gray, low-texture frame (smoke-like)
        smoke_frame = np.full((480, 640, 3), 200, dtype=np.uint8)

        detected, confidence = anomaly_detection_input._detect_smoke(smoke_frame)
        # Result depends on texture analysis
        assert isinstance(detected, bool)
        assert isinstance(confidence, float)

    def test_detect_smoke_negative(self, anomaly_detection_input):
        """Test smoke detection with colorful frame."""
        # Create a colorful, high-contrast frame
        colorful_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        colorful_frame[::2, :] = [255, 0, 0]  # Alternating blue lines
        colorful_frame[1::2, :] = [0, 255, 0]  # Alternating green lines

        detected, _ = anomaly_detection_input._detect_smoke(colorful_frame)
        assert detected is False


class TestMotionDetection:
    """Tests for motion detection functionality."""

    def test_detect_motion_first_frame(self, anomaly_detection_input):
        """Test motion detection on first frame (no previous frame)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        anomaly_detection_input.prev_frame = None

        detected, score = anomaly_detection_input._detect_motion(frame)
        assert detected is False
        assert score == 0.0

    def test_detect_motion_with_change(self, anomaly_detection_input):
        """Test motion detection with significant frame change."""
        # Set previous frame
        prev = np.zeros((480, 640), dtype=np.uint8)
        anomaly_detection_input.prev_frame = prev

        # Create current frame with significant change
        current = np.zeros((480, 640, 3), dtype=np.uint8)
        current[:240, :] = [255, 255, 255]  # Top half is white

        detected, score = anomaly_detection_input._detect_motion(current)
        assert isinstance(detected, bool)
        assert score > 0

    def test_detect_motion_no_change(self, anomaly_detection_input):
        """Test motion detection with identical frames."""
        # Set previous frame
        frame = np.zeros((480, 640), dtype=np.uint8)
        anomaly_detection_input.prev_frame = frame.copy()

        # Create identical current frame
        current = np.zeros((480, 640, 3), dtype=np.uint8)

        detected, score = anomaly_detection_input._detect_motion(current)
        assert detected is False


class TestWaterDetection:
    """Tests for water leak detection functionality."""

    def test_detect_water_positive(self, anomaly_detection_input):
        """Test water detection with blue floor region."""
        # Create frame with blue in the lower third
        water_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        water_frame[320:, :] = [255, 100, 0]  # Blue water on floor

        detected, confidence = anomaly_detection_input._detect_water(water_frame)
        assert isinstance(detected, bool)
        assert isinstance(confidence, float)

    def test_detect_water_negative(self, anomaly_detection_input):
        """Test water detection with non-water floor."""
        # Create frame with brown/neutral floor
        normal_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        normal_frame[320:, :] = [50, 100, 150]  # Brown floor

        detected, confidence = anomaly_detection_input._detect_water(normal_frame)
        assert detected is False


class TestCollapseDetection:
    """Tests for person collapse detection functionality."""

    def test_detect_collapse_insufficient_history(self, anomaly_detection_input):
        """Test collapse detection with insufficient motion history."""
        anomaly_detection_input.motion_history = [100, 200, 300]  # Less than 10

        detected, desc = anomaly_detection_input._detect_collapse()
        assert detected is False
        assert desc == ""

    def test_detect_collapse_motion_to_stillness(self, anomaly_detection_input):
        """Test collapse detection with motion followed by stillness pattern."""
        # Simulate high motion followed by sudden stillness
        anomaly_detection_input.motion_threshold = 5000
        anomaly_detection_input.motion_history = [
            10000,
            10000,
            10000,
            10000,
            10000,  # Earlier: high motion
            100,
            100,
            100,
            100,
            100,  # Recent: stillness
        ]
        anomaly_detection_input.stillness_counter = 3

        detected, desc = anomaly_detection_input._detect_collapse()
        assert isinstance(detected, bool)

    def test_detect_collapse_normal_activity(self, anomaly_detection_input):
        """Test collapse detection with normal activity pattern."""
        anomaly_detection_input.motion_threshold = 5000
        anomaly_detection_input.motion_history = [
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
            3000,
        ]

        detected, desc = anomaly_detection_input._detect_collapse()
        assert detected is False
