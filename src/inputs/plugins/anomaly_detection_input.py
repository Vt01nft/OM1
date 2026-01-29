"""Anomaly Detection Input Plugin for OM1.

This module provides environmental safety monitoring by detecting
anomalies such as fire, smoke, water leaks, collapsed persons,
and unusual motion in camera frames.
"""

import asyncio
import logging
import time
from typing import Optional

import cv2
import numpy as np
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class AnomalyDetectionConfig(SensorConfig):
    """Configuration for Anomaly Detection Input.

    Parameters
    ----------
    camera_index : int
        Index of the camera device to use for detection.
    fire_threshold : float
        Minimum ratio of fire-colored pixels to trigger fire detection.
    motion_threshold : float
        Minimum motion score to trigger unusual motion detection.
    detection_interval : float
        Time interval in seconds between detection cycles.
    """

    camera_index: int = Field(default=0, description="Index of the camera device")
    fire_threshold: float = Field(
        default=0.05, description="Fire pixel ratio threshold (0.0-1.0)"
    )
    motion_threshold: float = Field(
        default=5000.0, description="Motion detection threshold"
    )
    detection_interval: float = Field(
        default=1.0, description="Detection interval in seconds"
    )


class AnomalyDetectionInput(FuserInput[AnomalyDetectionConfig, Optional[np.ndarray]]):
    """Environmental anomaly detection input for safety monitoring.

    Detects various environmental hazards and anomalies including:
    - Fire and smoke (color-based detection)
    - Unusual motion (frame differencing)
    - Water on floor (reflection patterns)
    - Person collapsed (motion followed by stillness)

    The detection results are converted to natural language descriptions
    for the LLM to process and respond appropriately.
    """

    def __init__(self, config: AnomalyDetectionConfig):
        """Initialize Anomaly Detection input handler.

        Parameters
        ----------
        config : AnomalyDetectionConfig
            Configuration settings for anomaly detection.
        """
        super().__init__(config)

        self.camera_index = self.config.camera_index
        self.fire_threshold = self.config.fire_threshold
        self.motion_threshold = self.config.motion_threshold
        self.detection_interval = self.config.detection_interval

        # Track IO
        self.io_provider = IOProvider()

        # Messages buffer
        self.messages: list[Message] = []

        # Descriptor for LLM context
        self.descriptor_for_LLM = "Anomaly Detector"

        # Previous frame for motion detection
        self.prev_frame: Optional[np.ndarray] = None
        self.motion_history: list[float] = []
        self.stillness_counter = 0

        # Initialize camera
        self.cap: Optional[cv2.VideoCapture] = None
        self.have_cam = self._check_webcam(self.camera_index)

        if self.have_cam:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(
                f"Anomaly Detection initialized with camera {self.camera_index} "
                f"({self.width}x{self.height})"
            )
        else:
            logging.warning(
                f"Anomaly Detection: Camera {self.camera_index} not available"
            )

    def _check_webcam(self, index: int) -> bool:
        """Check if a webcam is available at the given index.

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
            logging.info(f"Anomaly Detection: Camera {index} not found")
            cap.release()
            return False
        logging.info(f"Anomaly Detection: Camera {index} found")
        cap.release()
        return True

    def _detect_fire(self, frame: np.ndarray) -> tuple[bool, float]:
        """Detect fire or flames in the frame using color analysis.

        Parameters
        ----------
        frame : np.ndarray
            BGR image frame to analyze.

        Returns
        -------
        tuple[bool, float]
            Detection result and confidence score.
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Fire color range (orange-red-yellow)
        lower_fire1 = np.array([0, 100, 100])
        upper_fire1 = np.array([25, 255, 255])
        lower_fire2 = np.array([160, 100, 100])
        upper_fire2 = np.array([180, 255, 255])

        # Create masks for fire colors
        mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        fire_mask = cv2.bitwise_or(mask1, mask2)

        # Calculate fire pixel ratio
        fire_pixels = cv2.countNonZero(fire_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        fire_ratio = fire_pixels / total_pixels

        detected = fire_ratio > self.fire_threshold
        confidence = min(fire_ratio / self.fire_threshold, 1.0) if detected else 0.0

        return detected, confidence

    def _detect_smoke(self, frame: np.ndarray) -> tuple[bool, float]:
        """Detect smoke in the frame using gray color and texture analysis.

        Parameters
        ----------
        frame : np.ndarray
            BGR image frame to analyze.

        Returns
        -------
        tuple[bool, float]
            Detection result and confidence score.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Smoke is typically gray with low saturation
        lower_smoke = np.array([0, 0, 150])
        upper_smoke = np.array([180, 50, 255])

        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

        # Check for smoke-like texture (blur and compare)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        texture_diff = cv2.absdiff(gray, blurred)
        texture_score = np.mean(texture_diff)

        smoke_pixels = cv2.countNonZero(smoke_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        smoke_ratio = smoke_pixels / total_pixels

        # Smoke needs both gray color AND hazy texture
        detected = smoke_ratio > 0.3 and texture_score < 20
        confidence = smoke_ratio if detected else 0.0

        return detected, confidence

    def _detect_motion(self, frame: np.ndarray) -> tuple[bool, float]:
        """Detect unusual motion using frame differencing.

        Parameters
        ----------
        frame : np.ndarray
            BGR image frame to analyze.

        Returns
        -------
        tuple[bool, float]
            Detection result and motion score.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0.0

        # Compute frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        motion_score = cv2.countNonZero(thresh)

        self.prev_frame = gray

        # Track motion history for collapse detection
        self.motion_history.append(motion_score)
        if len(self.motion_history) > 30:
            self.motion_history.pop(0)

        detected = motion_score > self.motion_threshold
        return detected, motion_score

    def _detect_collapse(self) -> tuple[bool, str]:
        """Detect potential person collapse based on motion patterns.

        Looks for sudden motion followed by prolonged stillness.

        Returns
        -------
        tuple[bool, str]
            Detection result and description.
        """
        if len(self.motion_history) < 10:
            return False, ""

        recent = self.motion_history[-5:]
        earlier = self.motion_history[-10:-5]

        avg_recent = np.mean(recent)
        avg_earlier = np.mean(earlier)

        # Pattern: significant motion followed by sudden stillness
        if (
            avg_earlier > self.motion_threshold
            and avg_recent < self.motion_threshold / 4
        ):
            self.stillness_counter += 1
            if self.stillness_counter > 3:
                return True, "sudden stillness after movement detected"
        else:
            self.stillness_counter = 0

        return False, ""

    def _detect_water(self, frame: np.ndarray) -> tuple[bool, float]:
        """Detect water on floor using reflection and color patterns.

        Parameters
        ----------
        frame : np.ndarray
            BGR image frame to analyze.

        Returns
        -------
        tuple[bool, float]
            Detection result and confidence score.
        """
        # Focus on lower third of frame (floor area)
        floor_region = frame[int(frame.shape[0] * 0.66) :, :]

        # Convert to HSV
        hsv = cv2.cvtColor(floor_region, cv2.COLOR_BGR2HSV)

        # Water appears as blue-ish or high saturation reflective
        lower_water = np.array([90, 30, 100])
        upper_water = np.array([130, 255, 255])

        water_mask = cv2.inRange(hsv, lower_water, upper_water)
        water_pixels = cv2.countNonZero(water_mask)
        total_pixels = floor_region.shape[0] * floor_region.shape[1]
        water_ratio = water_pixels / total_pixels

        detected = water_ratio > 0.1
        confidence = min(water_ratio / 0.1, 1.0) if detected else 0.0

        return detected, confidence

    async def _poll(self) -> Optional[np.ndarray]:
        """Poll for new camera frame.

        Returns
        -------
        Optional[np.ndarray]
            Captured camera frame or None if unavailable.
        """
        await asyncio.sleep(self.detection_interval)

        if self.have_cam and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                return frame

        return None

    async def _raw_to_text(self, raw_input: Optional[np.ndarray]) -> Optional[Message]:
        """Process frame and generate anomaly detection message.

        Parameters
        ----------
        raw_input : Optional[np.ndarray]
            Camera frame to analyze.

        Returns
        -------
        Optional[Message]
            Timestamped message describing detected anomalies.
        """
        if raw_input is None:
            return None

        anomalies: list[str] = []
        severity = "normal"

        # Run all detections
        fire_detected, fire_conf = self._detect_fire(raw_input)
        smoke_detected, smoke_conf = self._detect_smoke(raw_input)
        motion_detected, motion_score = self._detect_motion(raw_input)
        collapse_detected, collapse_desc = self._detect_collapse()
        water_detected, water_conf = self._detect_water(raw_input)

        # Build anomaly report
        if fire_detected:
            anomalies.append(f"FIRE detected with {fire_conf:.0%} confidence")
            severity = "critical"

        if smoke_detected:
            anomalies.append(f"SMOKE detected with {smoke_conf:.0%} confidence")
            if severity != "critical":
                severity = "warning"

        if water_detected:
            anomalies.append(f"WATER LEAK detected on floor ({water_conf:.0%} conf)")
            if severity == "normal":
                severity = "warning"

        if collapse_detected:
            anomalies.append(f"PERSON MAY HAVE COLLAPSED: {collapse_desc}")
            severity = "critical"

        if motion_detected and severity == "normal":
            anomalies.append("Unusual motion detected")
            severity = "info"

        # Generate message only if anomalies detected
        if anomalies:
            severity_prefix = {
                "critical": "EMERGENCY ALERT",
                "warning": "WARNING",
                "info": "NOTICE",
                "normal": "STATUS",
            }

            message = (
                f"{severity_prefix[severity]}: "
                f"{'; '.join(anomalies)}. "
                f"Recommended action: "
            )

            if severity == "critical":
                message += "Initiate emergency response immediately."
            elif severity == "warning":
                message += "Alert user and monitor situation."
            else:
                message += "Continue monitoring."

            return Message(timestamp=time.time(), message=message)

        return None

    async def raw_to_text(self, raw_input: Optional[np.ndarray]) -> None:
        """Convert raw frame to text and update message buffer.

        Parameters
        ----------
        raw_input : Optional[np.ndarray]
            Camera frame to process.
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format and clear the latest buffer contents.

        Formats the most recent anomaly message for the LLM,
        adds it to the IO provider, then clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if empty.
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        logging.info(f"AnomalyDetection: {latest_message.message}")

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result

    def __del__(self) -> None:
        """Clean up camera resources on deletion."""
        if self.cap is not None:
            self.cap.release()
