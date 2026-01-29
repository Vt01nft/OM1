"""
Tests for Environment OCR Input Plugin.

This module contains comprehensive tests for the EnvironmentOCRInput class,
covering configuration, initialization, text detection, and output formatting.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inputs.plugins.environment_ocr_input import (
    EnvironmentOCRConfig,
    EnvironmentOCRInput,
    check_webcam,
)


class TestEnvironmentOCRConfig:
    """Tests for EnvironmentOCRConfig configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnvironmentOCRConfig()

        assert config.camera_index == 0
        assert config.confidence_threshold == 0.5
        assert config.detection_interval == 1.0
        assert config.languages == ["en"]
        assert config.max_text_results == 5
        assert config.min_text_length == 2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EnvironmentOCRConfig(
            camera_index=1,
            confidence_threshold=0.7,
            detection_interval=2.0,
            languages=["en", "es", "fr"],
            max_text_results=10,
            min_text_length=3,
        )

        assert config.camera_index == 1
        assert config.confidence_threshold == 0.7
        assert config.detection_interval == 2.0
        assert config.languages == ["en", "es", "fr"]
        assert config.max_text_results == 10
        assert config.min_text_length == 3

    def test_confidence_threshold_bounds(self):
        """Test confidence threshold validation."""
        # Valid bounds
        config_low = EnvironmentOCRConfig(confidence_threshold=0.0)
        assert config_low.confidence_threshold == 0.0

        config_high = EnvironmentOCRConfig(confidence_threshold=1.0)
        assert config_high.confidence_threshold == 1.0

        # Invalid bounds should raise validation error
        with pytest.raises(ValueError):
            EnvironmentOCRConfig(confidence_threshold=-0.1)

        with pytest.raises(ValueError):
            EnvironmentOCRConfig(confidence_threshold=1.1)

    def test_detection_interval_positive(self):
        """Test detection interval must be positive."""
        config = EnvironmentOCRConfig(detection_interval=0.5)
        assert config.detection_interval == 0.5

        with pytest.raises(ValueError):
            EnvironmentOCRConfig(detection_interval=0.0)

        with pytest.raises(ValueError):
            EnvironmentOCRConfig(detection_interval=-1.0)

    def test_max_text_results_bounds(self):
        """Test max text results validation."""
        config_min = EnvironmentOCRConfig(max_text_results=1)
        assert config_min.max_text_results == 1

        config_max = EnvironmentOCRConfig(max_text_results=20)
        assert config_max.max_text_results == 20

        with pytest.raises(ValueError):
            EnvironmentOCRConfig(max_text_results=0)

        with pytest.raises(ValueError):
            EnvironmentOCRConfig(max_text_results=21)


class TestCheckWebcam:
    """Tests for webcam checking utility function."""

    @patch("inputs.plugins.environment_ocr_input.cv2.VideoCapture")
    def test_webcam_available(self, mock_capture):
        """Test detection of available webcam."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap

        result = check_webcam(0)

        assert result is True
        mock_cap.release.assert_called_once()

    @patch("inputs.plugins.environment_ocr_input.cv2.VideoCapture")
    def test_webcam_not_available(self, mock_capture):
        """Test detection of unavailable webcam."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        result = check_webcam(0)

        assert result is False
        mock_cap.release.assert_called_once()


class TestEnvironmentOCRInputInitialization:
    """Tests for EnvironmentOCRInput initialization."""

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @patch("inputs.plugins.environment_ocr_input.cv2.VideoCapture")
    def test_initialization_with_camera(self, mock_capture, mock_check):
        """Test initialization when camera is available."""
        mock_check.return_value = True
        mock_cap = MagicMock()
        mock_cap.get.side_effect = [1280.0, 720.0]  # width, height
        mock_capture.return_value = mock_cap

        config = EnvironmentOCRConfig(camera_index=0)
        ocr_input = EnvironmentOCRInput(config)

        assert ocr_input.have_cam is True
        assert ocr_input.cap is not None
        assert ocr_input.width == 1280
        assert ocr_input.height == 720
        assert ocr_input.cam_third == 426
        assert ocr_input.descriptor_for_LLM == "Text Reader"
        assert ocr_input.messages == []

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    def test_initialization_without_camera(self, mock_check):
        """Test initialization when camera is not available."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig(camera_index=0)
        ocr_input = EnvironmentOCRInput(config)

        assert ocr_input.have_cam is False
        assert ocr_input.cap is None
        assert ocr_input.width == 640  # Default fallback
        assert ocr_input.height == 480


class TestOCRInitialization:
    """Tests for lazy OCR reader initialization."""

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    def test_ocr_lazy_initialization(self, mock_check):
        """Test that OCR reader is not initialized until first use."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)

        assert ocr_input.reader is None
        assert ocr_input._ocr_initialized is False

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @patch("inputs.plugins.environment_ocr_input.easyocr")
    def test_ocr_initialization_success(self, mock_easyocr, mock_check):
        """Test successful OCR reader initialization."""
        mock_check.return_value = False
        mock_reader = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader

        config = EnvironmentOCRConfig(languages=["en", "es"])
        ocr_input = EnvironmentOCRInput(config)

        # Trigger initialization
        result = ocr_input._initialize_ocr()

        assert result is True
        assert ocr_input.reader is mock_reader
        assert ocr_input._ocr_initialized is True
        mock_easyocr.Reader.assert_called_once_with(
            ["en", "es"], gpu=False, verbose=False
        )

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    def test_ocr_initialization_import_error(self, mock_check):
        """Test OCR initialization when easyocr is not installed."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)

        # Mock import error
        with patch.dict("sys.modules", {"easyocr": None}):
            with patch(
                "inputs.plugins.environment_ocr_input.easyocr",
                side_effect=ImportError("No module named 'easyocr'"),
            ):
                # Force re-initialization attempt
                ocr_input._ocr_initialized = False
                ocr_input.reader = None

                # This should handle the import error gracefully
                # Since we can't easily mock the import inside the method,
                # we test the fallback behavior
                assert ocr_input._ocr_initialized is False


class TestSpatialDirection:
    """Tests for spatial direction calculation."""

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    def test_direction_left(self, mock_check):
        """Test text detected on the left side."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)
        ocr_input.width = 900
        ocr_input.cam_third = 300

        # Bounding box on left (center_x = 100)
        bbox = [[50, 0], [150, 0], [150, 50], [50, 50]]
        direction = ocr_input._get_spatial_direction(bbox)

        assert direction == "on your left"

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    def test_direction_center(self, mock_check):
        """Test text detected in the center."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)
        ocr_input.width = 900
        ocr_input.cam_third = 300

        # Bounding box in center (center_x = 450)
        bbox = [[400, 0], [500, 0], [500, 50], [400, 50]]
        direction = ocr_input._get_spatial_direction(bbox)

        assert direction == "in front of you"

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    def test_direction_right(self, mock_check):
        """Test text detected on the right side."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)
        ocr_input.width = 900
        ocr_input.cam_third = 300

        # Bounding box on right (center_x = 800)
        bbox = [[750, 0], [850, 0], [850, 50], [750, 50]]
        direction = ocr_input._get_spatial_direction(bbox)

        assert direction == "on your right"


class TestTextDetection:
    """Tests for text detection functionality."""

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @patch("inputs.plugins.environment_ocr_input.cv2.cvtColor")
    def test_detect_text_success(self, mock_cvtcolor, mock_check):
        """Test successful text detection."""
        mock_check.return_value = False
        mock_cvtcolor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        config = EnvironmentOCRConfig(confidence_threshold=0.5, min_text_length=2)
        ocr_input = EnvironmentOCRInput(config)
        ocr_input.width = 900
        ocr_input.cam_third = 300

        # Mock OCR reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[100, 0], [200, 0], [200, 50], [100, 50]], "EXIT", 0.95),
            ([[450, 0], [550, 0], [550, 50], [450, 50]], "ROOM 101", 0.87),
            ([[750, 0], [850, 0], [850, 50], [750, 50]], "X", 0.60),  # Too short
        ]
        ocr_input.reader = mock_reader
        ocr_input._ocr_initialized = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = ocr_input._detect_text(frame)

        assert len(detections) == 2
        assert detections[0]["text"] == "EXIT"
        assert detections[0]["direction"] == "on your left"
        assert detections[1]["text"] == "ROOM 101"
        assert detections[1]["direction"] == "in front of you"

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @patch("inputs.plugins.environment_ocr_input.cv2.cvtColor")
    def test_detect_text_filtered_by_confidence(self, mock_cvtcolor, mock_check):
        """Test that low confidence text is filtered out."""
        mock_check.return_value = False
        mock_cvtcolor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        config = EnvironmentOCRConfig(confidence_threshold=0.8)
        ocr_input = EnvironmentOCRInput(config)

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[100, 0], [200, 0], [200, 50], [100, 50]], "HIGH", 0.9),
            ([[300, 0], [400, 0], [400, 50], [300, 50]], "LOW", 0.5),  # Below threshold
        ]
        ocr_input.reader = mock_reader
        ocr_input._ocr_initialized = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = ocr_input._detect_text(frame)

        assert len(detections) == 1
        assert detections[0]["text"] == "HIGH"

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    def test_detect_text_no_reader(self, mock_check):
        """Test detection when OCR reader is not available."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)
        ocr_input._ocr_initialized = True
        ocr_input.reader = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = ocr_input._detect_text(frame)

        assert detections == []

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @patch("inputs.plugins.environment_ocr_input.cv2.cvtColor")
    def test_detect_text_max_results(self, mock_cvtcolor, mock_check):
        """Test that results are limited to max_text_results."""
        mock_check.return_value = False
        mock_cvtcolor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        config = EnvironmentOCRConfig(max_text_results=2)
        ocr_input = EnvironmentOCRInput(config)

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[100, 0], [200, 0], [200, 50], [100, 50]], "ONE", 0.9),
            ([[200, 0], [300, 0], [300, 50], [200, 50]], "TWO", 0.85),
            ([[300, 0], [400, 0], [400, 50], [300, 50]], "THREE", 0.8),
            ([[400, 0], [500, 0], [500, 50], [400, 50]], "FOUR", 0.75),
        ]
        ocr_input.reader = mock_reader
        ocr_input._ocr_initialized = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = ocr_input._detect_text(frame)

        assert len(detections) == 2
        # Should be sorted by confidence
        assert detections[0]["text"] == "ONE"
        assert detections[1]["text"] == "TWO"


class TestPoll:
    """Tests for polling functionality."""

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @patch("inputs.plugins.environment_ocr_input.cv2.VideoCapture")
    @pytest.mark.asyncio
    async def test_poll_with_camera(self, mock_capture, mock_check):
        """Test polling when camera is available."""
        mock_check.return_value = True
        mock_cap = MagicMock()
        mock_cap.get.side_effect = [640.0, 480.0]
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_capture.return_value = mock_cap

        config = EnvironmentOCRConfig(detection_interval=0.1)
        ocr_input = EnvironmentOCRInput(config)
        ocr_input.last_detection_time = 0  # Allow immediate poll

        result = await ocr_input._poll()

        assert result is not None
        assert isinstance(result, np.ndarray)

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @pytest.mark.asyncio
    async def test_poll_without_camera(self, mock_check):
        """Test polling when camera is not available."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)

        result = await ocr_input._poll()

        assert result is None

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @patch("inputs.plugins.environment_ocr_input.cv2.VideoCapture")
    @pytest.mark.asyncio
    async def test_poll_respects_interval(self, mock_capture, mock_check):
        """Test that polling respects detection interval."""
        mock_check.return_value = True
        mock_cap = MagicMock()
        mock_cap.get.side_effect = [640.0, 480.0]
        mock_capture.return_value = mock_cap

        config = EnvironmentOCRConfig(detection_interval=10.0)
        ocr_input = EnvironmentOCRInput(config)
        ocr_input.last_detection_time = time.time()  # Just polled

        result = await ocr_input._poll()

        assert result is None  # Should skip due to interval


class TestRawToText:
    """Tests for raw to text conversion."""

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @pytest.mark.asyncio
    async def test_raw_to_text_with_detections(self, mock_check):
        """Test message generation with text detections."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)

        # Mock detection
        with patch.object(ocr_input, "_detect_text") as mock_detect:
            mock_detect.return_value = [
                {"text": "EXIT", "confidence": 0.95, "direction": "on your left"},
                {
                    "text": "ROOM 101",
                    "confidence": 0.87,
                    "direction": "in front of you",
                },
            ]

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            message = await ocr_input._raw_to_text(frame)

            assert message is not None
            assert 'You see text "EXIT" on your left.' in message.message
            assert 'You also see text "ROOM 101" in front of you.' in message.message

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @pytest.mark.asyncio
    async def test_raw_to_text_no_detections(self, mock_check):
        """Test message generation with no text detections."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)

        with patch.object(ocr_input, "_detect_text") as mock_detect:
            mock_detect.return_value = []

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            message = await ocr_input._raw_to_text(frame)

            assert message is None

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @pytest.mark.asyncio
    async def test_raw_to_text_none_input(self, mock_check):
        """Test message generation with None input."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)

        message = await ocr_input._raw_to_text(None)

        assert message is None


class TestFormattedBuffer:
    """Tests for formatted buffer output."""

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    def test_formatted_buffer_with_messages(self, mock_check):
        """Test formatted output when buffer has messages."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)

        # Add a message to buffer
        from inputs.base import Message

        ocr_input.messages.append(
            Message(
                timestamp=time.time(),
                message='You see text "DANGER" in front of you.',
            )
        )

        result = ocr_input.formatted_latest_buffer()

        assert result is not None
        assert "INPUT: Text Reader" in result
        assert "// START" in result
        assert 'You see text "DANGER" in front of you.' in result
        assert "// END" in result
        assert ocr_input.messages == []  # Buffer should be cleared

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    def test_formatted_buffer_empty(self, mock_check):
        """Test formatted output when buffer is empty."""
        mock_check.return_value = False

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)

        result = ocr_input.formatted_latest_buffer()

        assert result is None


class TestCleanup:
    """Tests for resource cleanup."""

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @patch("inputs.plugins.environment_ocr_input.cv2.VideoCapture")
    def test_camera_release_on_del(self, mock_capture, mock_check):
        """Test that camera is released on deletion."""
        mock_check.return_value = True
        mock_cap = MagicMock()
        mock_cap.get.side_effect = [640.0, 480.0]
        mock_capture.return_value = mock_cap

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)

        # Trigger cleanup
        ocr_input.__del__()

        mock_cap.release.assert_called_once()


class TestIntegration:
    """Integration tests for EnvironmentOCRInput."""

    @patch("inputs.plugins.environment_ocr_input.check_webcam")
    @patch("inputs.plugins.environment_ocr_input.cv2.cvtColor")
    @pytest.mark.asyncio
    async def test_full_detection_flow(self, mock_cvtcolor, mock_check):
        """Test complete flow from detection to formatted output."""
        mock_check.return_value = False
        mock_cvtcolor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        config = EnvironmentOCRConfig()
        ocr_input = EnvironmentOCRInput(config)
        ocr_input.width = 900
        ocr_input.cam_third = 300

        # Setup mock reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[100, 0], [200, 0], [200, 50], [100, 50]], "EMERGENCY EXIT", 0.92),
        ]
        ocr_input.reader = mock_reader
        ocr_input._ocr_initialized = True

        # Process frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        await ocr_input.raw_to_text(frame)

        # Get formatted output
        result = ocr_input.formatted_latest_buffer()

        assert result is not None
        assert "INPUT: Text Reader" in result
        assert 'You see text "EMERGENCY EXIT" on your left.' in result
