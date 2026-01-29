"""
Environment OCR Input Plugin for OM1.

This module provides text recognition capabilities for robotic systems,
enabling them to read and understand text in their environment such as
signs, labels, warnings, and displays.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class EnvironmentOCRConfig(SensorConfig):
    """
    Configuration for Environment OCR Input.

    Parameters
    ----------
    camera_index : int
        Index of the camera device to use for capturing images.
    confidence_threshold : float
        Minimum confidence score (0.0-1.0) for text detection.
        Text below this threshold will be ignored.
    detection_interval : float
        Time in seconds between OCR detection cycles.
    languages : List[str]
        List of language codes for OCR detection.
        Common codes: 'en' (English), 'es' (Spanish), 'fr' (French),
        'de' (German), 'zh' (Chinese), 'ja' (Japanese).
    max_text_results : int
        Maximum number of text detections to report per cycle.
    min_text_length : int
        Minimum character length for text to be reported.
        Helps filter out noise and single-character detections.
    """

    camera_index: int = Field(
        default=0,
        description="Index of the camera device",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for text detection",
    )
    detection_interval: float = Field(
        default=1.0,
        gt=0.0,
        description="Seconds between detection cycles",
    )
    languages: List[str] = Field(
        default=["en"],
        description="Language codes for OCR detection",
    )
    max_text_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum text detections to report",
    )
    min_text_length: int = Field(
        default=2,
        ge=1,
        description="Minimum character length for text",
    )


def check_webcam(index: int) -> bool:
    """
    Check if a webcam is available at the specified index.

    Parameters
    ----------
    index : int
        Camera index to check.

    Returns
    -------
    bool
        True if camera is available, False otherwise.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logging.info(f"EnvironmentOCR: Camera not found at index {index}")
        cap.release()
        return False
    logging.info(f"EnvironmentOCR: Camera found at index {index}")
    cap.release()
    return True


class EnvironmentOCRInput(FuserInput[EnvironmentOCRConfig, Optional[np.ndarray]]):
    """
    Environment text recognition input for robotic systems.

    This class captures images from a camera and uses OCR (Optical Character
    Recognition) to detect and read text in the robot's environment. Detected
    text is reported with spatial context (left, center, right) to help the
    robot understand where text is located.

    Use cases include:
    - Reading warning signs and safety labels
    - Identifying room numbers and door signs
    - Reading product labels and barcodes
    - Detecting emergency exit signs
    - Reading digital displays and screens

    The class uses EasyOCR for text detection, which supports multiple
    languages and provides confidence scores for each detection.
    """

    def __init__(self, config: EnvironmentOCRConfig):
        """
        Initialize the Environment OCR input handler.

        Sets up the camera capture, initializes the OCR reader,
        and prepares the message buffer.

        Parameters
        ----------
        config : EnvironmentOCRConfig
            Configuration object with OCR and camera settings.
        """
        super().__init__(config)

        # Track IO
        self.io_provider = IOProvider()

        # Messages buffer
        self.messages: List[Message] = []

        # Descriptor for LLM context
        self.descriptor_for_LLM = "Text Reader"

        # Initialize OCR reader (lazy loading)
        self.reader = None
        self._ocr_initialized = False

        # Camera setup
        self.have_cam = check_webcam(self.config.camera_index)
        self.cap = None
        self.width = 640
        self.height = 480
        self.cam_third = self.width // 3

        if self.have_cam:
            self.cap = cv2.VideoCapture(self.config.camera_index)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.cam_third = self.width // 3
            logging.info(
                f"EnvironmentOCR: Camera resolution {self.width}x{self.height}"
            )

        # Timing control
        self.last_detection_time = 0.0

        logging.info("EnvironmentOCR: Input handler initialized")

    def _initialize_ocr(self) -> bool:
        """
        Lazily initialize the OCR reader.

        EasyOCR can take time to load models, so we defer initialization
        until first use to avoid blocking startup.

        Returns
        -------
        bool
            True if OCR reader initialized successfully, False otherwise.
        """
        if self._ocr_initialized:
            return self.reader is not None

        try:
            import easyocr

            self.reader = easyocr.Reader(
                self.config.languages,
                gpu=False,  # Use CPU for compatibility
                verbose=False,
            )
            self._ocr_initialized = True
            logging.info(
                f"EnvironmentOCR: OCR reader initialized with languages "
                f"{self.config.languages}"
            )
            return True
        except ImportError:
            logging.error(
                "EnvironmentOCR: easyocr not installed. "
                "Install with: pip install easyocr"
            )
            self._ocr_initialized = True
            return False
        except Exception as e:
            logging.error(f"EnvironmentOCR: Failed to initialize OCR: {e}")
            self._ocr_initialized = True
            return False

    def _get_spatial_direction(self, bbox: List[List[int]]) -> str:
        """
        Determine spatial direction based on bounding box position.

        Parameters
        ----------
        bbox : List[List[int]]
            Bounding box coordinates from EasyOCR.
            Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns
        -------
        str
            Spatial direction: "on your left", "in front of you", or "on your right"
        """
        # Calculate center x from bounding box
        x_coords = [point[0] for point in bbox]
        center_x = sum(x_coords) / len(x_coords)

        if center_x < self.cam_third:
            return "on your left"
        elif center_x > 2 * self.cam_third:
            return "on your right"
        else:
            return "in front of you"

    def _detect_text(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect text in the given frame using OCR.

        Parameters
        ----------
        frame : np.ndarray
            Image frame from camera (BGR format).

        Returns
        -------
        List[Dict]
            List of detected text with confidence and position.
            Each dict contains: text, confidence, direction
        """
        if not self._initialize_ocr() or self.reader is None:
            return []

        try:
            # EasyOCR expects RGB, OpenCV gives BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run OCR detection
            results = self.reader.readtext(rgb_frame)

            detections = []
            for bbox, text, confidence in results:
                # Filter by confidence and length
                if confidence < self.config.confidence_threshold:
                    continue
                if len(text.strip()) < self.config.min_text_length:
                    continue

                direction = self._get_spatial_direction(bbox)
                detections.append(
                    {
                        "text": text.strip(),
                        "confidence": confidence,
                        "direction": direction,
                    }
                )

            # Sort by confidence and limit results
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            return detections[: self.config.max_text_results]

        except Exception as e:
            logging.error(f"EnvironmentOCR: Text detection failed: {e}")
            return []

    async def _poll(self) -> Optional[np.ndarray]:
        """
        Poll for new image input from camera.

        Respects the configured detection interval to avoid
        excessive CPU usage from continuous OCR processing.

        Returns
        -------
        Optional[np.ndarray]
            Captured frame if available and interval elapsed, None otherwise.
        """
        await asyncio.sleep(0.1)

        # Check detection interval
        current_time = time.time()
        if current_time - self.last_detection_time < self.config.detection_interval:
            return None

        if not self.have_cam or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logging.debug("EnvironmentOCR: Failed to capture frame")
            return None

        self.last_detection_time = current_time
        return frame

    async def _raw_to_text(self, raw_input: Optional[np.ndarray]) -> Optional[Message]:
        """
        Process raw image input to detect and describe text.

        Parameters
        ----------
        raw_input : Optional[np.ndarray]
            Input image frame to process.

        Returns
        -------
        Optional[Message]
            Timestamped message with detected text description,
            or None if no text detected.
        """
        if raw_input is None:
            return None

        detections = self._detect_text(raw_input)

        if not detections:
            return None

        # Build natural language description
        sentences = []
        for i, detection in enumerate(detections):
            text = detection["text"]
            direction = detection["direction"]

            if i == 0:
                sentences.append(f'You see text "{text}" {direction}.')
            else:
                sentences.append(f'You also see text "{text}" {direction}.')

        message = " ".join(sentences)
        logging.info(f"EnvironmentOCR: {message}")

        return Message(timestamp=time.time(), message=message)

    async def raw_to_text(self, raw_input: Optional[np.ndarray]):
        """
        Convert raw image to text and update message buffer.

        Parameters
        ----------
        raw_input : Optional[np.ndarray]
            Raw image frame to be processed.
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Formats the most recent message with the standard INPUT format,
        adds it to the IO provider, then clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string with detected text or None if buffer is empty.
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.descriptor_for_LLM,
            latest_message.message,
            latest_message.timestamp,
        )
        self.messages = []

        return result

    def __del__(self):
        """Release camera resources on cleanup."""
        if self.cap is not None:
            self.cap.release()
            logging.info("EnvironmentOCR: Camera released")
