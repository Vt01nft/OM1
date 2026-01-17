"""Tests for unitree_go2_frontier_exploration."""

import json
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


class TestUnitreeGo2FrontierExplorationProvider:
    """Tests for UnitreeGo2FrontierExplorationProvider class."""

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

    def test_initialization_with_defaults(self):
        """Test provider initializes correctly with default parameters."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        with patch(
            "providers.unitree_go2_frontier_exploration.ContextProvider"
        ) as mock_context:
            provider = UnitreeGo2FrontierExplorationProvider()

            assert provider is not None
            assert provider.exploration_info is None
            assert provider.exploration_complete is False
            assert hasattr(provider, "context_provider")
            assert provider.context_aware_text == {"exploration_done": True}
            mock_context.assert_called_once()

    def test_initialization_with_custom_parameters(self):
        """Test provider initializes correctly with custom parameters."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        custom_topic = "custom/exploration/status"
        custom_context = {"custom_key": "custom_value"}

        with patch(
            "providers.unitree_go2_frontier_exploration.ContextProvider"
        ) as mock_context:
            provider = UnitreeGo2FrontierExplorationProvider(
                topic=custom_topic, context_aware_text=custom_context
            )

            assert provider.context_aware_text == custom_context
            mock_context.assert_called_once()

    def test_singleton_pattern(self):
        """Test that provider follows singleton pattern."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        with patch("providers.unitree_go2_frontier_exploration.ContextProvider"):
            provider1 = UnitreeGo2FrontierExplorationProvider()
            provider2 = UnitreeGo2FrontierExplorationProvider()
            assert provider1 is provider2

    def test_frontier_exploration_message_callback_valid_json_complete(self):
        """Test message callback with valid JSON indicating exploration complete."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        mock_string = MagicMock()
        mock_string.data = json.dumps(
            {"complete": True, "info": "Exploration finished successfully"}
        )

        with (
            patch(
                "providers.unitree_go2_frontier_exploration.ContextProvider"
            ) as mock_context,
            patch(
                "providers.unitree_go2_frontier_exploration.String.deserialize",
                return_value=mock_string,
            ),
        ):
            provider = UnitreeGo2FrontierExplorationProvider()
            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test"

            provider.frontier_exploration_message_callback(mock_sample)

            assert provider.exploration_complete is True
            assert provider.exploration_info == "Exploration finished successfully"
            provider.context_provider.update_context.assert_called_once_with(
                {"exploration_done": True}
            )

    def test_frontier_exploration_message_callback_valid_json_incomplete(self):
        """Test message callback with valid JSON indicating exploration incomplete."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        mock_string = MagicMock()
        mock_string.data = json.dumps({"complete": False, "info": "Still exploring"})

        with (
            patch(
                "providers.unitree_go2_frontier_exploration.ContextProvider"
            ) as mock_context,
            patch(
                "providers.unitree_go2_frontier_exploration.String.deserialize",
                return_value=mock_string,
            ),
        ):
            provider = UnitreeGo2FrontierExplorationProvider()
            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test"

            provider.frontier_exploration_message_callback(mock_sample)

            assert provider.exploration_complete is False
            assert provider.exploration_info == "Still exploring"
            provider.context_provider.update_context.assert_not_called()

    def test_frontier_exploration_message_callback_invalid_json(self):
        """Test message callback with invalid JSON."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        mock_string = MagicMock()
        mock_string.data = "invalid json"

        with (
            patch(
                "providers.unitree_go2_frontier_exploration.ContextProvider"
            ) as mock_context,
            patch(
                "providers.unitree_go2_frontier_exploration.String.deserialize",
                return_value=mock_string,
            ),
            patch(
                "providers.unitree_go2_frontier_exploration.logging.error"
            ) as mock_log,
        ):
            provider = UnitreeGo2FrontierExplorationProvider()
            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test"

            provider.frontier_exploration_message_callback(mock_sample)

            mock_log.assert_called_once()
            provider.context_provider.update_context.assert_not_called()

    def test_frontier_exploration_message_callback_no_payload(self):
        """Test message callback with no payload."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        with patch("providers.unitree_go2_frontier_exploration.ContextProvider"):
            provider = UnitreeGo2FrontierExplorationProvider()
            mock_sample = MagicMock()
            mock_sample.payload = None

            # Should not raise an exception
            provider.frontier_exploration_message_callback(mock_sample)

    def test_start_when_not_running(self):
        """Test start method when provider is not running."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        with patch("providers.unitree_go2_frontier_exploration.ContextProvider"):
            provider = UnitreeGo2FrontierExplorationProvider()
            provider.running = False

            with patch.object(provider, "register_message_callback") as mock_register:
                provider.start()

                assert provider.running is True
                mock_register.assert_called_once_with(
                    provider.frontier_exploration_message_callback
                )

    def test_start_when_already_running(self):
        """Test start method when provider is already running."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        with (
            patch("providers.unitree_go2_frontier_exploration.ContextProvider"),
            patch(
                "providers.unitree_go2_frontier_exploration.logging.warning"
            ) as mock_warning,
        ):
            provider = UnitreeGo2FrontierExplorationProvider()
            provider.running = True

            with patch.object(provider, "register_message_callback") as mock_register:
                provider.start()

                assert provider.running is True
                mock_register.assert_not_called()
                mock_warning.assert_called_once()

    def test_start_with_custom_callback(self):
        """Test start method with custom callback parameter."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        with patch("providers.unitree_go2_frontier_exploration.ContextProvider"):
            provider = UnitreeGo2FrontierExplorationProvider()
            provider.running = False
            custom_callback = MagicMock()

            with patch.object(provider, "register_message_callback") as mock_register:
                provider.start(message_callback=custom_callback)

                assert provider.running is True
                mock_register.assert_called_once_with(
                    provider.frontier_exploration_message_callback
                )

    def test_status_property(self):
        """Test status property returns exploration complete status."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        with patch("providers.unitree_go2_frontier_exploration.ContextProvider"):
            provider = UnitreeGo2FrontierExplorationProvider()

            # Test initial state
            assert provider.status is False

            # Test after setting exploration_complete
            provider.exploration_complete = True
            assert provider.status is True

    def test_info_property(self):
        """Test info property returns exploration info."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        with patch("providers.unitree_go2_frontier_exploration.ContextProvider"):
            provider = UnitreeGo2FrontierExplorationProvider()

            # Test initial state
            assert provider.info is None

            # Test after setting exploration_info
            test_info = "Test exploration info"
            provider.exploration_info = test_info
            assert provider.info == test_info

    def test_message_callback_missing_fields(self):
        """Test message callback with JSON missing expected fields."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        mock_string = MagicMock()
        mock_string.data = json.dumps({"other_field": "value"})

        with (
            patch(
                "providers.unitree_go2_frontier_exploration.ContextProvider"
            ) as mock_context,
            patch(
                "providers.unitree_go2_frontier_exploration.String.deserialize",
                return_value=mock_string,
            ),
        ):
            provider = UnitreeGo2FrontierExplorationProvider()
            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test"

            provider.frontier_exploration_message_callback(mock_sample)

            # Should handle missing fields gracefully
            assert provider.exploration_complete is False
            assert provider.exploration_info == ""
            provider.context_provider.update_context.assert_not_called()

    def test_message_callback_with_logging(self):
        """Test message callback logs appropriately."""
        from providers.unitree_go2_frontier_exploration import (
            UnitreeGo2FrontierExplorationProvider,
        )

        if hasattr(UnitreeGo2FrontierExplorationProvider, "reset"):
            UnitreeGo2FrontierExplorationProvider.reset()

        mock_string = MagicMock()
        mock_string.data = json.dumps({"complete": True, "info": "Test completed"})

        with (
            patch("providers.unitree_go2_frontier_exploration.ContextProvider"),
            patch(
                "providers.unitree_go2_frontier_exploration.String.deserialize",
                return_value=mock_string,
            ),
            patch(
                "providers.unitree_go2_frontier_exploration.logging.info"
            ) as mock_log,
        ):
            provider = UnitreeGo2FrontierExplorationProvider()
            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test"

            provider.frontier_exploration_message_callback(mock_sample)

            mock_log.assert_called_with(
                "Exploration Status: Completed, Info: %s", "Test completed"
            )
