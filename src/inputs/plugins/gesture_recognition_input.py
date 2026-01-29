"""
Gesture Recognition Input Plugin for OM1.

This module provides hand gesture recognition capabilities for robotic systems,
enabling them to understand non-verbal communication from humans through
hand gestures detected via camera input.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class GestureType(str, Enum):
    """Enumeration of recognized hand gestures."""

    THUMBS_UP = "thumbs up"
    THUMBS_DOWN = "thumbs down"
    PEACE = "peace sign"
    OPEN_PALM = "open palm"
    CLOSED_FIST = "closed fist"
    POINTING = "pointing"
    WAVE = "waving"
    OK_SIGN = "OK sign"
    UNKNOWN = "unknown gesture"


class GestureRecognitionConfig(SensorConfig):
    """
    Configuration for Gesture Recognition Input.

    Parameters
    ----------
    camera_index : int
        Index of the camera device to use for capturing images.
    confidence_threshold : float
        Minimum confidence score (0.0-1.0) for hand detection.
        Hands below this threshold will be ignored.
    detection_interval : float
        Time in seconds between gesture detection cycles.
    max_num_hands : int
        Maximum number of hands to detect simultaneously.
    min_detection_confidence : float
        Minimum confidence for initial hand detection.
    min_tracking_confidence : float
        Minimum confidence for hand landmark tracking.
    """

    camera_index: int = Field(
        default=0,
        description="Index of the camera device",
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for gesture detection",
    )
    detection_interval: float = Field(
        default=0.5,
        gt=0.0,
        description="Seconds between detection cycles",
    )
    max_num_hands: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Maximum number of hands to detect",
    )
    min_detection_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for hand detection",
    )
    min_tracking_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for hand tracking",
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
        logging.info(f"GestureRecognition: Camera not found at index {index}")
        cap.release()
        return False
    logging.info(f"GestureRecognition: Camera found at index {index}")
    cap.release()
    return True


class GestureRecognitionInput(
    FuserInput[GestureRecognitionConfig, Optional[np.ndarray]]
):
    """
    Hand gesture recognition input for robotic systems.

    This class captures images from a camera and uses MediaPipe to detect
    hand landmarks and classify gestures. Detected gestures are reported
    with spatial context to help the robot understand non-verbal communication.

    Use cases include:
    - Understanding thumbs up/down for approval/disapproval
    - Detecting pointing gestures for direction indication
    - Recognizing wave gestures for greetings
    - Detecting stop/halt gestures (open palm)
    - Understanding OK signs for confirmation

    The class uses MediaPipe Hands for hand landmark detection and implements
    custom gesture classification based on finger positions.
    """

    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self, config: GestureRecognitionConfig):
        """
        Initialize the Gesture Recognition input handler.

        Sets up the camera capture, initializes MediaPipe hands,
        and prepares the message buffer.

        Parameters
        ----------
        config : GestureRecognitionConfig
            Configuration object with gesture recognition settings.
        """
        super().__init__(config)

        # Track IO
        self.io_provider = IOProvider()

        # Messages buffer
        self.messages: List[Message] = []

        # Descriptor for LLM context
        self.descriptor_for_LLM = "Gesture Detector"

        # Initialize MediaPipe (lazy loading)
        self.mp_hands = None
        self.hands = None
        self._mp_initialized = False

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
                f"GestureRecognition: Camera resolution {self.width}x{self.height}"
            )

        # Timing control
        self.last_detection_time = 0.0

        logging.info("GestureRecognition: Input handler initialized")

    def _initialize_mediapipe(self) -> bool:
        """
        Lazily initialize MediaPipe hands.

        MediaPipe can take time to load, so we defer initialization
        until first use to avoid blocking startup.

        Returns
        -------
        bool
            True if MediaPipe initialized successfully, False otherwise.
        """
        if self._mp_initialized:
            return self.hands is not None

        try:
            import mediapipe as mp

            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config.max_num_hands,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self._mp_initialized = True
            logging.info("GestureRecognition: MediaPipe hands initialized")
            return True
        except ImportError:
            logging.error(
                "GestureRecognition: mediapipe not installed. "
                "Install with: pip install mediapipe"
            )
            self._mp_initialized = True
            return False
        except Exception as e:
            logging.error(f"GestureRecognition: Failed to initialize MediaPipe: {e}")
            self._mp_initialized = True
            return False

    def _get_spatial_direction(self, hand_center_x: float) -> str:
        """
        Determine spatial direction based on hand position.

        Parameters
        ----------
        hand_center_x : float
            X coordinate of hand center (0.0 to 1.0, normalized).

        Returns
        -------
        str
            Spatial direction: "on your left", "in front of you", or "on your right"
        """
        # Convert normalized coordinate to pixel
        pixel_x = hand_center_x * self.width

        if pixel_x < self.cam_third:
            return "on your left"
        elif pixel_x > 2 * self.cam_third:
            return "on your right"
        else:
            return "in front of you"

    def _is_finger_extended(
        self, landmarks: List, finger_tip: int, finger_pip: int, finger_mcp: int
    ) -> bool:
        """
        Check if a finger is extended based on landmark positions.

        Parameters
        ----------
        landmarks : List
            List of hand landmarks from MediaPipe.
        finger_tip : int
            Index of finger tip landmark.
        finger_pip : int
            Index of finger PIP joint landmark.
        finger_mcp : int
            Index of finger MCP joint landmark.

        Returns
        -------
        bool
            True if finger is extended, False otherwise.
        """
        tip = landmarks[finger_tip]
        pip = landmarks[finger_pip]
        mcp = landmarks[finger_mcp]

        # Finger is extended if tip is above PIP (lower y value = higher position)
        return tip.y < pip.y and pip.y < mcp.y

    def _is_thumb_extended(self, landmarks: List, handedness: str) -> bool:
        """
        Check if thumb is extended based on landmark positions.

        Parameters
        ----------
        landmarks : List
            List of hand landmarks from MediaPipe.
        handedness : str
            'Left' or 'Right' hand.

        Returns
        -------
        bool
            True if thumb is extended, False otherwise.
        """
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        thumb_mcp = landmarks[self.THUMB_MCP]

        # For thumb, check horizontal extension based on handedness
        if handedness == "Right":
            return thumb_tip.x < thumb_ip.x < thumb_mcp.x
        else:
            return thumb_tip.x > thumb_ip.x > thumb_mcp.x

    def _classify_gesture(
        self, landmarks: List, handedness: str
    ) -> Tuple[GestureType, float]:
        """
        Classify hand gesture based on finger positions.

        Parameters
        ----------
        landmarks : List
            List of hand landmarks from MediaPipe.
        handedness : str
            'Left' or 'Right' hand.

        Returns
        -------
        Tuple[GestureType, float]
            Detected gesture type and confidence score.
        """
        # Check each finger's extension state
        thumb_extended = self._is_thumb_extended(landmarks, handedness)
        index_extended = self._is_finger_extended(
            landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP
        )
        middle_extended = self._is_finger_extended(
            landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP
        )
        ring_extended = self._is_finger_extended(
            landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP
        )
        pinky_extended = self._is_finger_extended(
            landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP
        )

        extended_count = sum(
            [
                thumb_extended,
                index_extended,
                middle_extended,
                ring_extended,
                pinky_extended,
            ]
        )

        # Thumbs up: only thumb extended, pointing up
        thumb_tip = landmarks[self.THUMB_TIP]
        wrist = landmarks[self.WRIST]
        if (
            thumb_extended
            and not index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
            and thumb_tip.y < wrist.y
        ):
            return GestureType.THUMBS_UP, 0.9

        # Thumbs down: only thumb extended, pointing down
        if (
            thumb_extended
            and not index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
            and thumb_tip.y > wrist.y
        ):
            return GestureType.THUMBS_DOWN, 0.9

        # Peace sign: index and middle extended, others closed
        if (
            index_extended
            and middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            return GestureType.PEACE, 0.85

        # Pointing: only index extended
        if (
            index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            return GestureType.POINTING, 0.85

        # Open palm: all fingers extended
        if extended_count >= 4:
            return GestureType.OPEN_PALM, 0.8

        # Closed fist: no fingers extended
        if extended_count == 0:
            return GestureType.CLOSED_FIST, 0.8

        # OK sign: thumb and index forming circle, others extended
        thumb_tip_pos = landmarks[self.THUMB_TIP]
        index_tip_pos = landmarks[self.INDEX_TIP]
        distance = (
            (thumb_tip_pos.x - index_tip_pos.x) ** 2
            + (thumb_tip_pos.y - index_tip_pos.y) ** 2
        ) ** 0.5
        if distance < 0.05 and middle_extended and ring_extended and pinky_extended:
            return GestureType.OK_SIGN, 0.85

        return GestureType.UNKNOWN, 0.5

    def _detect_gestures(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect hand gestures in the given frame.

        Parameters
        ----------
        frame : np.ndarray
            Image frame from camera (BGR format).

        Returns
        -------
        List[Dict]
            List of detected gestures with details.
            Each dict contains: gesture, confidence, direction, hand
        """
        if not self._initialize_mediapipe() or self.hands is None:
            return []

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with MediaPipe
            results = self.hands.process(rgb_frame)

            if not results.multi_hand_landmarks:
                return []

            detections = []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness (left/right)
                handedness = "Right"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label

                # Calculate hand center for spatial direction
                landmarks = hand_landmarks.landmark
                hand_center_x = sum(lm.x for lm in landmarks) / len(landmarks)

                # Classify gesture
                gesture, confidence = self._classify_gesture(landmarks, handedness)

                # Filter by confidence
                if confidence < self.config.confidence_threshold:
                    continue

                # Skip unknown gestures
                if gesture == GestureType.UNKNOWN:
                    continue

                direction = self._get_spatial_direction(hand_center_x)

                detections.append(
                    {
                        "gesture": gesture.value,
                        "confidence": confidence,
                        "direction": direction,
                        "hand": handedness.lower(),
                    }
                )

            return detections

        except Exception as e:
            logging.error(f"GestureRecognition: Detection failed: {e}")
            return []

    async def _poll(self) -> Optional[np.ndarray]:
        """
        Poll for new image input from camera.

        Respects the configured detection interval to avoid
        excessive CPU usage from continuous gesture processing.

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
            logging.debug("GestureRecognition: Failed to capture frame")
            return None

        self.last_detection_time = current_time
        return frame

    async def _raw_to_text(self, raw_input: Optional[np.ndarray]) -> Optional[Message]:
        """
        Process raw image input to detect and describe gestures.

        Parameters
        ----------
        raw_input : Optional[np.ndarray]
            Input image frame to process.

        Returns
        -------
        Optional[Message]
            Timestamped message with detected gesture description,
            or None if no gestures detected.
        """
        if raw_input is None:
            return None

        detections = self._detect_gestures(raw_input)

        if not detections:
            return None

        # Build natural language description
        sentences = []
        for i, detection in enumerate(detections):
            gesture = detection["gesture"]
            direction = detection["direction"]
            hand = detection["hand"]

            if i == 0:
                sentences.append(
                    f'You see a person making a "{gesture}" gesture '
                    f"with their {hand} hand {direction}."
                )
            else:
                sentences.append(
                    f'They are also making a "{gesture}" gesture '
                    f"with their {hand} hand {direction}."
                )

        message = " ".join(sentences)
        logging.info(f"GestureRecognition: {message}")

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
            Formatted string with detected gestures or None if buffer is empty.
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
        """Release resources on cleanup."""
        if self.cap is not None:
            self.cap.release()
            logging.info("GestureRecognition: Camera released")
        if self.hands is not None:
            self.hands.close()
            logging.info("GestureRecognition: MediaPipe hands closed")
