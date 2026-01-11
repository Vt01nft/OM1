"""
Test suite for GPS Fabric connector timeout bug fix.

This test demonstrates the bug: missing timeout parameter on HTTP request.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from actions.gps.connector.fabric import GPSFabricConfig, GPSFabricConnector
from actions.gps.interface import GPSAction, GPSInput


class TestGPSFabricConnectorTimeout:
    """Test suite demonstrating timeout bug in GPS Fabric connector."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return GPSFabricConfig(fabric_endpoint="http://test-endpoint:8545")

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider to provide test coordinates."""
        with patch("actions.gps.connector.fabric.IOProvider") as mock:
            mock_instance = mock.return_value
            mock_instance.get_dynamic_variable.side_effect = lambda key: {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "yaw_deg": 90.0,
            }.get(key)
            yield mock_instance

    @pytest.mark.asyncio
    async def test_fix_http_request_has_timeout(self, mock_config, mock_io_provider):
        """
        AFTER FIX: Verify HTTP request includes timeout parameter.

        This test verifies that requests.post is called WITH a timeout parameter,
        preventing indefinite hangs.
        """
        with patch("actions.gps.connector.fabric.requests.post") as mock_post:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {"result": True}
            mock_post.return_value = mock_response

            # Create connector and trigger action
            connector = GPSFabricConnector(mock_config)
            test_input = GPSInput(action=GPSAction.SHARE_LOCATION)
            await connector.connect(test_input)

            # Verify timeout is present
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            assert "timeout" in call_kwargs, "Timeout parameter should be present!"
            assert isinstance(
                call_kwargs["timeout"], (int, float)
            ), "Timeout should be a number!"

    @pytest.mark.asyncio
    async def test_network_errors_handled_gracefully(
        self, mock_config, mock_io_provider
    ):
        """
        Verify network errors are caught and logged without crashing.
        """
        with patch("actions.gps.connector.fabric.requests.post") as mock_post:
            with patch("actions.gps.connector.fabric.logging.error") as mock_log:
                # Simulate a network error
                mock_post.side_effect = requests.exceptions.ConnectionError(
                    "Connection refused"
                )

                connector = GPSFabricConnector(mock_config)
                test_input = GPSInput(action=GPSAction.SHARE_LOCATION)

                # Should NOT raise - error should be handled gracefully
                await connector.connect(test_input)

                # Verify error was logged
                mock_log.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
