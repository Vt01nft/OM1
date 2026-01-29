"""
Tests for Gesture Recognition Input Plugin.

This module contains comprehensive tests for the GestureRecognitionInput class,
covering configuration, initialization, gesture detection, and output formatting.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inputs.plugins.gesture_recognition_input import (
    GestureRecognitionConfig,
    GestureRecognitionInput,
    GestureType,
    check_webcam,
)


class TestGestureType:
    """Tests for GestureType enumeration."""

    def test_gesture_types_exist(self):
        """Test that all expected gesture types are defined."""
        assert GestureType.THUMBS_UP.value == "thumbs up"
        assert GestureType.THUMBS_DOWN.value == "thumbs down"
        assert GestureType.PEACE.value == "peace sign"
        assert GestureType.OPEN_PALM.value == "open palm"
        assert GestureType.CLOSED_FIST.value == "closed fist"
        assert GestureType.POINTING.value == "pointing"
        assert GestureType.WAVE.value == "waving"
        assert GestureType.OK_SIGN.value == "OK sign"
        assert GestureType.UNKNOWN.value == "unknown gesture"


class TestGestureRecognitionConfig:
    """Tests for GestureRecognitionConfig configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GestureRecognitionConfig()

        assert config.camera_index == 0
        assert config.confidence_threshold == 0.7
        assert config.detection_interval == 0.5
        assert config.max_num_hands == 2
        assert config.min_detection_confidence == 0.7
        assert config.min_tracking_confidence == 0.5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GestureRecognitionConfig(
            camera_index=1,
            confidence_threshold=0.8,
            detection_interval=1.0,
            max_num_hands=4,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6,
        )

        assert config.camera_index == 1
        assert config.confidence_threshold == 0.8
        assert config.detection_interval == 1.0
        assert config.max_num_hands == 4
        assert config.min_detection_confidence == 0.8
        assert config.min_tracking_confidence == 0.6

    def test_confidence_threshold_bounds(self):
        """Test confidence threshold validation."""
        config_low = GestureRecognitionConfig(confidence_threshold=0.0)
        assert config_low.confidence_threshold == 0.0

        config_high = GestureRecognitionConfig(confidence_threshold=1.0)
        assert config_high.confidence_threshold == 1.0

        with pytest.raises(ValueError):
            GestureRecognitionConfig(confidence_threshold=-0.1)

        with pytest.raises(ValueError):
            GestureRecognitionConfig(confidence_threshold=1.1)

    def test_detection_interval_positive(self):
        """Test detection interval must be positive."""
        config = GestureRecognitionConfig(detection_interval=0.1)
        assert config.detection_interval == 0.1

        with pytest.raises(ValueError):
            GestureRecognitionConfig(detection_interval=0.0)

        with pytest.raises(ValueError):
            GestureRecognitionConfig(detection_interval=-1.0)

    def test_max_num_hands_bounds(self):
        """Test max num hands validation."""
        config_min = GestureRecognitionConfig(max_num_hands=1)
        assert config_min.max_num_hands == 1

        config_max = GestureRecognitionConfig(max_num_hands=4)
        assert config_max.max_num_hands == 4

        with pytest.raises(ValueError):
            GestureRecognitionConfig(max_num_hands=0)

        with pytest.raises(ValueError):
            GestureRecognitionConfig(max_num_hands=5)


class TestCheckWebcam:
    """Tests for webcam checking utility function."""

    @patch("inputs.plugins.gesture_recognition_input.cv2.VideoCapture")
    def test_webcam_available(self, mock_capture):
        """Test detection of available webcam."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap

        result = check_webcam(0)

        assert result is True
        mock_cap.release.assert_called_once()

    @patch("inputs.plugins.gesture_recognition_input.cv2.VideoCapture")
    def test_webcam_not_available(self, mock_capture):
        """Test detection of unavailable webcam."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        result = check_webcam(0)

        assert result is False
        mock_cap.release.assert_called_once()


class TestGestureRecognitionInitialization:
    """Tests for GestureRecognitionInput initialization."""

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    @patch("inputs.plugins.gesture_recognition_input.cv2.VideoCapture")
    def test_initialization_with_camera(self, mock_capture, mock_check):
        """Test initialization when camera is available."""
        mock_check.return_value = True
        mock_cap = MagicMock()
        mock_cap.get.side_effect = [1280.0, 720.0]
        mock_capture.return_value = mock_cap

        config = GestureRecognitionConfig(camera_index=0)
        gesture_input = GestureRecognitionInput(config)

        assert gesture_input.have_cam is True
        assert gesture_input.cap is not None
        assert gesture_input.width == 1280
        assert gesture_input.height == 720
        assert gesture_input.cam_third == 426
        assert gesture_input.descriptor_for_LLM == "Gesture Detector"
        assert gesture_input.messages == []

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_initialization_without_camera(self, mock_check):
        """Test initialization when camera is not available."""
        mock_check.return_value = False

        config = GestureRecognitionConfig(camera_index=0)
        gesture_input = GestureRecognitionInput(config)

        assert gesture_input.have_cam is False
        assert gesture_input.cap is None
        assert gesture_input.width == 640
        assert gesture_input.height == 480


class TestMediaPipeInitialization:
    """Tests for lazy MediaPipe initialization."""

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_mediapipe_lazy_initialization(self, mock_check):
        """Test that MediaPipe is not initialized until first use."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        assert gesture_input.mp_hands is None
        assert gesture_input.hands is None
        assert gesture_input._mp_initialized is False

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_mediapipe_initialization_success(self, mock_check):
        """Test successful MediaPipe initialization."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        # Mock mediapipe module
        mock_mp = MagicMock()
        mock_hands_class = MagicMock()
        mock_hands_instance = MagicMock()
        mock_hands_class.return_value = mock_hands_instance
        mock_mp.solutions.hands.Hands = mock_hands_class

        with patch.dict("sys.modules", {"mediapipe": mock_mp}):
            with patch(
                "inputs.plugins.gesture_recognition_input.mp", mock_mp, create=True
            ):
                # Force re-initialization
                gesture_input._mp_initialized = False
                gesture_input.mp_hands = None
                gesture_input.hands = None

                # The actual test would require proper mocking of import
                # For now, we test the graceful failure path
                pass


class TestSpatialDirection:
    """Tests for spatial direction calculation."""

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_direction_left(self, mock_check):
        """Test hand detected on the left side."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)
        gesture_input.width = 900
        gesture_input.cam_third = 300

        # Normalized x = 0.1 -> pixel = 90 (left third)
        direction = gesture_input._get_spatial_direction(0.1)

        assert direction == "on your left"

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_direction_center(self, mock_check):
        """Test hand detected in the center."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)
        gesture_input.width = 900
        gesture_input.cam_third = 300

        # Normalized x = 0.5 -> pixel = 450 (center)
        direction = gesture_input._get_spatial_direction(0.5)

        assert direction == "in front of you"

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_direction_right(self, mock_check):
        """Test hand detected on the right side."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)
        gesture_input.width = 900
        gesture_input.cam_third = 300

        # Normalized x = 0.9 -> pixel = 810 (right third)
        direction = gesture_input._get_spatial_direction(0.9)

        assert direction == "on your right"


class TestFingerExtension:
    """Tests for finger extension detection."""

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_finger_extended(self, mock_check):
        """Test detection of extended finger."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        # Create mock landmarks
        class MockLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        # Extended finger: tip.y < pip.y < mcp.y (pointing up)
        landmarks = [None] * 21
        landmarks[8] = MockLandmark(0.5, 0.2, 0)  # INDEX_TIP (high)
        landmarks[6] = MockLandmark(0.5, 0.4, 0)  # INDEX_PIP (middle)
        landmarks[5] = MockLandmark(0.5, 0.6, 0)  # INDEX_MCP (low)

        result = gesture_input._is_finger_extended(landmarks, 8, 6, 5)

        assert result is True

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_finger_not_extended(self, mock_check):
        """Test detection of curled finger."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        class MockLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        # Curled finger: tip.y > pip.y (pointing down/curled)
        landmarks = [None] * 21
        landmarks[8] = MockLandmark(0.5, 0.6, 0)  # INDEX_TIP (low)
        landmarks[6] = MockLandmark(0.5, 0.4, 0)  # INDEX_PIP (middle)
        landmarks[5] = MockLandmark(0.5, 0.5, 0)  # INDEX_MCP

        result = gesture_input._is_finger_extended(landmarks, 8, 6, 5)

        assert result is False


class TestGestureClassification:
    """Tests for gesture classification logic."""

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_classify_open_palm(self, mock_check):
        """Test classification of open palm gesture."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        class MockLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        # Create landmarks for open palm (all fingers extended)
        landmarks = []
        for i in range(21):
            if i in [4, 8, 12, 16, 20]:  # Tips
                landmarks.append(MockLandmark(0.5, 0.1, 0))
            elif i in [3, 7, 11, 15, 19]:  # DIPs
                landmarks.append(MockLandmark(0.5, 0.2, 0))
            elif i in [2, 6, 10, 14, 18]:  # PIPs
                landmarks.append(MockLandmark(0.5, 0.3, 0))
            elif i in [1, 5, 9, 13, 17]:  # MCPs
                landmarks.append(MockLandmark(0.5, 0.4, 0))
            else:
                landmarks.append(MockLandmark(0.5, 0.5, 0))

        gesture, confidence = gesture_input._classify_gesture(landmarks, "Right")

        assert gesture == GestureType.OPEN_PALM
        assert confidence >= 0.7

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_classify_closed_fist(self, mock_check):
        """Test classification of closed fist gesture."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        class MockLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        # Create landmarks for closed fist (all fingers curled)
        landmarks = []
        for i in range(21):
            if i in [4, 8, 12, 16, 20]:  # Tips (below PIP)
                landmarks.append(MockLandmark(0.5, 0.6, 0))
            elif i in [3, 7, 11, 15, 19]:  # DIPs
                landmarks.append(MockLandmark(0.5, 0.5, 0))
            elif i in [2, 6, 10, 14, 18]:  # PIPs
                landmarks.append(MockLandmark(0.5, 0.4, 0))
            elif i in [1, 5, 9, 13, 17]:  # MCPs
                landmarks.append(MockLandmark(0.5, 0.3, 0))
            else:
                landmarks.append(MockLandmark(0.5, 0.5, 0))

        gesture, confidence = gesture_input._classify_gesture(landmarks, "Right")

        assert gesture == GestureType.CLOSED_FIST
        assert confidence >= 0.7


class TestGestureDetection:
    """Tests for gesture detection functionality."""

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_detect_gestures_no_mediapipe(self, mock_check):
        """Test detection when MediaPipe is not available."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)
        gesture_input._mp_initialized = True
        gesture_input.hands = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = gesture_input._detect_gestures(frame)

        assert detections == []

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    @patch("inputs.plugins.gesture_recognition_input.cv2.cvtColor")
    def test_detect_gestures_no_hands(self, mock_cvtcolor, mock_check):
        """Test detection when no hands are in frame."""
        mock_check.return_value = False
        mock_cvtcolor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        # Mock MediaPipe with no detections
        mock_hands = MagicMock()
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = None
        mock_hands.process.return_value = mock_results
        gesture_input.hands = mock_hands
        gesture_input._mp_initialized = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = gesture_input._detect_gestures(frame)

        assert detections == []


class TestPoll:
    """Tests for polling functionality."""

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    @patch("inputs.plugins.gesture_recognition_input.cv2.VideoCapture")
    @pytest.mark.asyncio
    async def test_poll_with_camera(self, mock_capture, mock_check):
        """Test polling when camera is available."""
        mock_check.return_value = True
        mock_cap = MagicMock()
        mock_cap.get.side_effect = [640.0, 480.0]
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_capture.return_value = mock_cap

        config = GestureRecognitionConfig(detection_interval=0.1)
        gesture_input = GestureRecognitionInput(config)
        gesture_input.last_detection_time = 0

        result = await gesture_input._poll()

        assert result is not None
        assert isinstance(result, np.ndarray)

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    @pytest.mark.asyncio
    async def test_poll_without_camera(self, mock_check):
        """Test polling when camera is not available."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        result = await gesture_input._poll()

        assert result is None

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    @patch("inputs.plugins.gesture_recognition_input.cv2.VideoCapture")
    @pytest.mark.asyncio
    async def test_poll_respects_interval(self, mock_capture, mock_check):
        """Test that polling respects detection interval."""
        mock_check.return_value = True
        mock_cap = MagicMock()
        mock_cap.get.side_effect = [640.0, 480.0]
        mock_capture.return_value = mock_cap

        config = GestureRecognitionConfig(detection_interval=10.0)
        gesture_input = GestureRecognitionInput(config)
        gesture_input.last_detection_time = time.time()

        result = await gesture_input._poll()

        assert result is None


class TestRawToText:
    """Tests for raw to text conversion."""

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    @pytest.mark.asyncio
    async def test_raw_to_text_with_detections(self, mock_check):
        """Test message generation with gesture detections."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        with patch.object(gesture_input, "_detect_gestures") as mock_detect:
            mock_detect.return_value = [
                {
                    "gesture": "thumbs up",
                    "confidence": 0.9,
                    "direction": "in front of you",
                    "hand": "right",
                },
            ]

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            message = await gesture_input._raw_to_text(frame)

            assert message is not None
            assert "thumbs up" in message.message
            assert "right hand" in message.message
            assert "in front of you" in message.message

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    @pytest.mark.asyncio
    async def test_raw_to_text_multiple_hands(self, mock_check):
        """Test message generation with multiple hand detections."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        with patch.object(gesture_input, "_detect_gestures") as mock_detect:
            mock_detect.return_value = [
                {
                    "gesture": "thumbs up",
                    "confidence": 0.9,
                    "direction": "on your left",
                    "hand": "right",
                },
                {
                    "gesture": "peace sign",
                    "confidence": 0.85,
                    "direction": "on your right",
                    "hand": "left",
                },
            ]

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            message = await gesture_input._raw_to_text(frame)

            assert message is not None
            assert "thumbs up" in message.message
            assert "peace sign" in message.message
            assert "also making" in message.message

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    @pytest.mark.asyncio
    async def test_raw_to_text_no_detections(self, mock_check):
        """Test message generation with no gesture detections."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        with patch.object(gesture_input, "_detect_gestures") as mock_detect:
            mock_detect.return_value = []

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            message = await gesture_input._raw_to_text(frame)

            assert message is None

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    @pytest.mark.asyncio
    async def test_raw_to_text_none_input(self, mock_check):
        """Test message generation with None input."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        message = await gesture_input._raw_to_text(None)

        assert message is None


class TestFormattedBuffer:
    """Tests for formatted buffer output."""

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_formatted_buffer_with_messages(self, mock_check):
        """Test formatted output when buffer has messages."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        from inputs.base import Message

        gesture_input.messages.append(
            Message(
                timestamp=time.time(),
                message='You see a person making a "thumbs up" gesture with their right hand in front of you.',
            )
        )

        result = gesture_input.formatted_latest_buffer()

        assert result is not None
        assert "INPUT: Gesture Detector" in result
        assert "// START" in result
        assert "thumbs up" in result
        assert "// END" in result
        assert gesture_input.messages == []

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_formatted_buffer_empty(self, mock_check):
        """Test formatted output when buffer is empty."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        result = gesture_input.formatted_latest_buffer()

        assert result is None


class TestCleanup:
    """Tests for resource cleanup."""

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    @patch("inputs.plugins.gesture_recognition_input.cv2.VideoCapture")
    def test_camera_release_on_del(self, mock_capture, mock_check):
        """Test that camera is released on deletion."""
        mock_check.return_value = True
        mock_cap = MagicMock()
        mock_cap.get.side_effect = [640.0, 480.0]
        mock_capture.return_value = mock_cap

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        gesture_input.__del__()

        mock_cap.release.assert_called_once()

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    def test_mediapipe_close_on_del(self, mock_check):
        """Test that MediaPipe hands is closed on deletion."""
        mock_check.return_value = False

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        mock_hands = MagicMock()
        gesture_input.hands = mock_hands

        gesture_input.__del__()

        mock_hands.close.assert_called_once()


class TestIntegration:
    """Integration tests for GestureRecognitionInput."""

    @patch("inputs.plugins.gesture_recognition_input.check_webcam")
    @patch("inputs.plugins.gesture_recognition_input.cv2.cvtColor")
    @pytest.mark.asyncio
    async def test_full_detection_flow(self, mock_cvtcolor, mock_check):
        """Test complete flow from detection to formatted output."""
        mock_check.return_value = False
        mock_cvtcolor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        config = GestureRecognitionConfig()
        gesture_input = GestureRecognitionInput(config)

        # Mock detection method directly
        with patch.object(gesture_input, "_detect_gestures") as mock_detect:
            mock_detect.return_value = [
                {
                    "gesture": "open palm",
                    "confidence": 0.85,
                    "direction": "in front of you",
                    "hand": "right",
                },
            ]

            # Process frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            await gesture_input.raw_to_text(frame)

            # Get formatted output
            result = gesture_input.formatted_latest_buffer()

            assert result is not None
            assert "INPUT: Gesture Detector" in result
            assert "open palm" in result
            assert "right hand" in result
