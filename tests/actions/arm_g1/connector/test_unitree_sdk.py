"""Tests for ARMUnitreeSDKConnector."""

from unittest.mock import Mock, patch

import pytest

from actions.arm_g1.connector.unitree_sdk import ARMUnitreeSDKConnector
from actions.arm_g1.interface import ArmInput
from actions.base import ActionConfig


@pytest.fixture
def mock_g1_arm_client():
    """Mock the G1ArmActionClient."""
    with patch("actions.arm_g1.connector.unitree_sdk.G1ArmActionClient") as mock_client:
        mock_instance = Mock()
        mock_instance.SetTimeout = Mock()
        mock_instance.Init = Mock()
        mock_instance.ExecuteAction = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def connector(mock_g1_arm_client):
    """Create an ARMUnitreeSDKConnector instance with mocked dependencies."""
    config = ActionConfig()
    connector = ARMUnitreeSDKConnector(config)
    return connector


@pytest.mark.asyncio
async def test_connect_logs_arm_command(connector, mock_g1_arm_client, caplog):
    """Test that connect logs the arm command action."""
    output_interface = Mock(spec=ArmInput)
    output_interface.action = "clap"

    await connector.connect(output_interface)

    assert "Arm command.action: clap" in caplog.text


@pytest.mark.asyncio
async def test_connect_idle_action_returns_early(connector, mock_g1_arm_client, caplog):
    """Test that idle action returns early without executing."""
    output_interface = Mock(spec=ArmInput)
    output_interface.action = "idle"

    await connector.connect(output_interface)

    assert "No action to perform" in caplog.text
    mock_g1_arm_client.ExecuteAction.assert_not_called()


@pytest.mark.asyncio
async def test_connect_executes_clap_action(connector, mock_g1_arm_client):
    """Test that clap action executes with correct action ID."""
    output_interface = Mock(spec=ArmInput)
    output_interface.action = "clap"

    await connector.connect(output_interface)

    mock_g1_arm_client.ExecuteAction.assert_called_once_with(17)


@pytest.mark.asyncio
async def test_connect_executes_high_five_action(connector, mock_g1_arm_client):
    """Test that high five action executes with correct action ID."""
    output_interface = Mock(spec=ArmInput)
    output_interface.action = "high five"

    await connector.connect(output_interface)

    mock_g1_arm_client.ExecuteAction.assert_called_once_with(18)


@pytest.mark.asyncio
async def test_connect_unknown_action_logs_warning(
    connector, mock_g1_arm_client, caplog
):
    """Test that unknown action logs a warning."""
    output_interface = Mock(spec=ArmInput)
    output_interface.action = "unknown_action"

    await connector.connect(output_interface)

    assert "Unknown action: unknown_action" in caplog.text
    mock_g1_arm_client.ExecuteAction.assert_not_called()
