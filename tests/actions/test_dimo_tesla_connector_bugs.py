"""
Test suite for DIMO Tesla connector bug fixes.

This test demonstrates critical bugs:
1. Missing timeout on HTTP requests (can hang indefinitely)
2. Missing error handling (network errors crash the connector)
3. Return type inconsistency
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

# These imports assume the structure from the codebase
from actions.dimo.connector.tesla import DIMOTeslaConfig, DIMOTeslaConnector
from actions.dimo.interface import TeslaInput


class TestDIMOTeslaConnectorBugs:
    """Test suite demonstrating bugs in DIMO Tesla connector."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return DIMOTeslaConfig(
            client_id="test_client",
            domain="test_domain",
            private_key="test_key",
            token_id=12345,
        )

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider to avoid actual API calls."""
        with patch("actions.dimo.connector.tesla.IOProvider") as mock:
            mock_instance = mock.return_value
            mock_instance.get_dynamic_variable.return_value = None
            yield mock_instance

    @pytest.fixture
    def mock_dimo(self):
        """Mock DIMO SDK to avoid actual API calls."""
        with patch("actions.dimo.connector.tesla.DIMO") as mock:
            mock_instance = mock.return_value
            mock_instance.auth.get_dev_jwt.return_value = {
                "access_token": "test_dev_token"
            }
            mock_instance.token_exchange.exchange.return_value = {
                "token": "test_vehicle_token"
            }
            yield mock_instance

    @pytest.mark.asyncio
    async def test_bug1_missing_timeout_on_http_requests(
        self, mock_config, mock_io_provider, mock_dimo
    ):
        """
        BUG #1: HTTP requests have no timeout parameter.

        This test verifies that requests.post is called WITHOUT a timeout,
        which can cause indefinite hangs.

        Expected after fix: All requests should have timeout parameter.
        """
        with patch("actions.dimo.connector.tesla.requests.post") as mock_post:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            # Create connector and trigger an action
            connector = DIMOTeslaConnector(mock_config)
            test_input = TeslaInput(action="lock doors")
            await connector.connect(test_input)

            # Verify the bug: requests.post was called WITHOUT timeout
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs

            # BUG: This assertion will PASS (proving bug exists)
            # After fix, this should FAIL because timeout should be present
            assert "timeout" not in call_kwargs, (
                "BUG DEMONSTRATED: No timeout parameter in requests.post(). "
                "This can cause indefinite hangs!"
            )

    @pytest.mark.asyncio
    async def test_bug2_network_error_crashes_connector(
        self, mock_config, mock_io_provider, mock_dimo
    ):
        """
        BUG #2: Network errors cause connector to crash.

        This test verifies that if requests.post raises an exception,
        it propagates and crashes the connector instead of being handled gracefully.

        Expected after fix: Network errors should be caught and logged, not crash.
        """
        with patch("actions.dimo.connector.tesla.requests.post") as mock_post:
            # Simulate a network error
            mock_post.side_effect = requests.exceptions.ConnectionError(
                "Connection refused"
            )

            # Create connector
            connector = DIMOTeslaConnector(mock_config)
            test_input = TeslaInput(action="lock doors")

            # BUG: This will raise an exception (proving bug exists)
            # After fix, this should NOT raise - error should be handled gracefully
            with pytest.raises(requests.exceptions.ConnectionError):
                await connector.connect(test_input)

    @pytest.mark.asyncio
    async def test_bug2_timeout_error_crashes_connector(
        self, mock_config, mock_io_provider, mock_dimo
    ):
        """
        BUG #2 (variant): Timeout errors cause connector to crash.

        This test verifies that if requests.post times out,
        it propagates and crashes the connector.

        Expected after fix: Timeout errors should be caught and logged.
        """
        with patch("actions.dimo.connector.tesla.requests.post") as mock_post:
            # Simulate a timeout
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

            # Create connector
            connector = DIMOTeslaConnector(mock_config)
            test_input = TeslaInput(action="unlock doors")

            # BUG: This will raise an exception (proving bug exists)
            with pytest.raises(requests.exceptions.Timeout):
                await connector.connect(test_input)

    @pytest.mark.asyncio
    async def test_all_tesla_actions_lack_timeout(
        self, mock_config, mock_io_provider, mock_dimo
    ):
        """
        BUG #1 (comprehensive): All Tesla actions lack timeout.

        This test verifies that ALL four Tesla actions (lock, unlock, frunk, trunk)
        make HTTP requests without timeout parameters.
        """
        actions_to_test = [
            "lock doors",
            "unlock doors",
            "open frunk",
            "open trunk",
        ]

        with patch("actions.dimo.connector.tesla.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            connector = DIMOTeslaConnector(mock_config)

            for action in actions_to_test:
                mock_post.reset_mock()
                test_input = TeslaInput(action=action)
                await connector.connect(test_input)

                # Verify bug exists for this action
                mock_post.assert_called_once()
                call_kwargs = mock_post.call_args.kwargs
                assert (
                    "timeout" not in call_kwargs
                ), f"BUG in '{action}': No timeout parameter!"


class TestDIMOTeslaConnectorAfterFix:
    """
    Test suite that will PASS after the bugs are fixed.

    These tests demonstrate the CORRECT behavior we want.
    """

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return DIMOTeslaConfig(
            client_id="test_client",
            domain="test_domain",
            private_key="test_key",
            token_id=12345,
        )

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider to avoid actual API calls."""
        with patch("actions.dimo.connector.tesla.IOProvider") as mock:
            mock_instance = mock.return_value
            mock_instance.get_dynamic_variable.return_value = None
            yield mock_instance

    @pytest.fixture
    def mock_dimo(self):
        """Mock DIMO SDK to avoid actual API calls."""
        with patch("actions.dimo.connector.tesla.DIMO") as mock:
            mock_instance = mock.return_value
            mock_instance.auth.get_dev_jwt.return_value = {
                "access_token": "test_dev_token"
            }
            mock_instance.token_exchange.exchange.return_value = {
                "token": "test_vehicle_token"
            }
            yield mock_instance

    @pytest.mark.asyncio
    async def test_fix_http_requests_have_timeout(
        self, mock_config, mock_io_provider, mock_dimo
    ):
        """
        AFTER FIX: Verify all HTTP requests include timeout parameter.

        This test will FAIL until the bug is fixed.
        After fix, all requests.post calls should have timeout parameter.
        """
        with patch("actions.dimo.connector.tesla.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            connector = DIMOTeslaConnector(mock_config)
            test_input = TeslaInput(action="lock doors")
            await connector.connect(test_input)

            # After fix: timeout should be present
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            assert "timeout" in call_kwargs, "Timeout parameter should be present!"
            assert isinstance(
                call_kwargs["timeout"], (int, float)
            ), "Timeout should be a number!"

    @pytest.mark.asyncio
    async def test_fix_network_errors_handled_gracefully(
        self, mock_config, mock_io_provider, mock_dimo
    ):
        """
        AFTER FIX: Verify network errors are caught and don't crash.

        This test will FAIL until the bug is fixed.
        After fix, network errors should be logged but not raise exceptions.
        """
        with patch("actions.dimo.connector.tesla.requests.post") as mock_post:
            with patch("actions.dimo.connector.tesla.logging.error") as mock_log:
                # Simulate a network error
                mock_post.side_effect = requests.exceptions.ConnectionError(
                    "Connection refused"
                )

                connector = DIMOTeslaConnector(mock_config)
                test_input = TeslaInput(action="lock doors")

                # After fix: Should NOT raise - should handle gracefully
                await connector.connect(test_input)

                # Verify error was logged
                mock_log.assert_called()
                error_message = str(mock_log.call_args[0][0])
                assert (
                    "error" in error_message.lower()
                    or "failed" in error_message.lower()
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
