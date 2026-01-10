"""Tests for DualLLM plugin."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm import LLMConfig
from llm.output_model import Action, CortexOutputModel
from llm.plugins.dual_llm import DualLLM, DualLLMConfig, _extract_voice_input


# Test output model
class DummyOutputModel(BaseModel):
    test_field: str


class TestExtractVoiceInput:
    """Tests for _extract_voice_input helper function."""

    def test_extract_valid_voice_input(self):
        """Test extracting voice input from valid prompt."""
        prompt = "Some text INPUT: Voice // START Hello world // END more text"
        result = _extract_voice_input(prompt)
        assert result == "Hello world"

    def test_extract_voice_input_multiline(self):
        """Test extracting multiline voice input."""
        prompt = "INPUT: Voice // START\nHello\nWorld\n// END"
        result = _extract_voice_input(prompt)
        assert result == "Hello\nWorld"

    def test_extract_voice_input_no_match(self):
        """Test empty string returned when no voice input found."""
        prompt = "Some text without voice input"
        result = _extract_voice_input(prompt)
        assert result == ""

    def test_extract_voice_input_empty_string(self):
        """Test empty string input."""
        result = _extract_voice_input("")
        assert result == ""


class TestDualLLMConfig:
    """Tests for DualLLMConfig class."""

    def test_default_config(self):
        """Test DualLLMConfig default values."""
        config = DualLLMConfig()
        assert config.local_llm_type == "QwenLLM"
        assert config.cloud_llm_type == "OpenAILLM"
        assert "model" in config.local_llm_config
        assert "model" in config.cloud_llm_config

    def test_custom_config(self):
        """Test DualLLMConfig with custom values."""
        config = DualLLMConfig(
            local_llm_type="CustomLocal",
            cloud_llm_type="CustomCloud",
            local_llm_config={"model": "custom-local-model"},
            cloud_llm_config={"model": "custom-cloud-model"},
        )
        assert config.local_llm_type == "CustomLocal"
        assert config.cloud_llm_type == "CustomCloud"


@pytest.fixture
def config():
    """Fixture providing a valid DualLLMConfig for testing."""
    return DualLLMConfig(
        api_key="test_key",
        local_llm_type="QwenLLM",
        local_llm_config={"model": "test-local-model"},
        cloud_llm_type="OpenAILLM",
        cloud_llm_config={"model": "test-cloud-model"},
    )


@pytest.fixture
def mock_response():
    """Fixture providing a valid mock API response."""
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content='{"test_field": "success"}', tool_calls=None)
        )
    ]
    return response


@pytest.fixture
def mock_result_with_actions():
    """Fixture providing a mock result with actions."""
    result = MagicMock(spec=CortexOutputModel)
    result.actions = [Action(type="test_action", value="test_value")]
    return result


@pytest.fixture
def mock_result_no_actions():
    """Fixture providing a mock result without actions."""
    result = MagicMock(spec=CortexOutputModel)
    result.actions = []
    return result


@pytest.fixture(autouse=True)
def mock_avatar_components():
    """Mock all avatar and IO components to prevent Zenoh session creation."""

    def mock_decorator(func=None):
        def decorator(f):
            return f

        if func is not None:
            return decorator(func)
        return decorator

    with (
        patch("llm.plugins.dual_llm.AvatarLLMState.trigger_thinking", mock_decorator),
        patch("llm.plugins.dual_llm.AvatarLLMState") as mock_avatar_state,
        patch("providers.avatar_provider.AvatarProvider") as mock_avatar_provider,
        patch(
            "providers.avatar_llm_state_provider.AvatarProvider"
        ) as mock_avatar_llm_state_provider,
        patch("llm.plugins.dual_llm.get_llm_class") as mock_get_llm_class,
        patch("llm.plugins.qwen_llm.AvatarLLMState.trigger_thinking", mock_decorator),
        patch("llm.plugins.openai_llm.AvatarLLMState.trigger_thinking", mock_decorator),
    ):
        mock_avatar_state._instance = None
        mock_avatar_state._lock = None

        mock_provider_instance = MagicMock()
        mock_provider_instance.running = False
        mock_provider_instance.session = None
        mock_provider_instance.stop = MagicMock()
        mock_avatar_provider.return_value = mock_provider_instance
        mock_avatar_llm_state_provider.return_value = mock_provider_instance

        # Mock LLM classes
        mock_local_llm = MagicMock()
        mock_cloud_llm = MagicMock()
        mock_get_llm_class.side_effect = lambda x: MagicMock(
            return_value=mock_local_llm if "Qwen" in x else mock_cloud_llm
        )

        yield


@pytest.fixture
def llm(config):
    """Fixture providing a DualLLM instance."""
    return DualLLM(config, available_actions=None)


class TestDualLLM:
    """Tests for DualLLM class."""

    @pytest.mark.asyncio
    async def test_init_with_config(self, llm, config):
        """Test DualLLM initialization with provided configuration."""
        assert llm._config.local_llm_type == config.local_llm_type
        assert llm._config.cloud_llm_type == config.cloud_llm_type

    @pytest.mark.asyncio
    async def test_init_creates_both_llms(self, llm):
        """Test DualLLM creates both local and cloud LLM instances."""
        assert llm._local_llm is not None
        assert llm._cloud_llm is not None

    @pytest.mark.asyncio
    async def test_init_sets_skip_state_management(self, llm):
        """Test DualLLM sets skip_state_management on child LLMs."""
        assert llm._local_llm._skip_state_management is True
        assert llm._cloud_llm._skip_state_management is True

    @pytest.mark.asyncio
    async def test_has_function_calls_with_actions(
        self, llm, mock_result_with_actions
    ):
        """Test _has_function_calls returns True when actions exist."""
        entry = {"result": mock_result_with_actions}
        assert llm._has_function_calls(entry) is True

    @pytest.mark.asyncio
    async def test_has_function_calls_without_actions(
        self, llm, mock_result_no_actions
    ):
        """Test _has_function_calls returns False when no actions."""
        entry = {"result": mock_result_no_actions}
        assert llm._has_function_calls(entry) is False

    @pytest.mark.asyncio
    async def test_has_function_calls_none_result(self, llm):
        """Test _has_function_calls returns False when result is None."""
        entry = {"result": None}
        assert llm._has_function_calls(entry) is False

    @pytest.mark.asyncio
    async def test_call_llm_success(self, llm, mock_result_with_actions):
        """Test _call_llm returns result with timing info."""
        mock_llm = MagicMock()
        mock_llm.ask = AsyncMock(return_value=mock_result_with_actions)

        result = await llm._call_llm(mock_llm, "test prompt", [], "local")

        assert result["source"] == "local"
        assert result["result"] == mock_result_with_actions
        assert "time" in result

    @pytest.mark.asyncio
    async def test_call_llm_error(self, llm):
        """Test _call_llm handles exceptions gracefully."""
        mock_llm = MagicMock()
        mock_llm.ask = AsyncMock(side_effect=Exception("Test error"))

        result = await llm._call_llm(mock_llm, "test prompt", [], "local")

        assert result["source"] == "local"
        assert result["result"] is None
        assert "time" in result

    @pytest.mark.asyncio
    async def test_select_best_local_has_functions(
        self, llm, mock_result_with_actions, mock_result_no_actions
    ):
        """Test _select_best returns local when only local has function calls."""
        local_entry = {"result": mock_result_with_actions, "source": "local"}
        cloud_entry = {"result": mock_result_no_actions, "source": "cloud"}

        result = await llm._select_best(local_entry, cloud_entry, "test")
        assert result == local_entry

    @pytest.mark.asyncio
    async def test_select_best_cloud_has_functions(
        self, llm, mock_result_with_actions, mock_result_no_actions
    ):
        """Test _select_best returns cloud when only cloud has function calls."""
        local_entry = {"result": mock_result_no_actions, "source": "local"}
        cloud_entry = {"result": mock_result_with_actions, "source": "cloud"}

        result = await llm._select_best(local_entry, cloud_entry, "test")
        assert result == cloud_entry

    @pytest.mark.asyncio
    async def test_select_best_neither_has_functions(
        self, llm, mock_result_no_actions
    ):
        """Test _select_best returns local when neither has function calls."""
        local_entry = {"result": mock_result_no_actions, "source": "local"}
        cloud_entry = {"result": mock_result_no_actions, "source": "cloud"}

        result = await llm._select_best(local_entry, cloud_entry, "test")
        assert result == local_entry

    @pytest.mark.asyncio
    async def test_ask_returns_none_on_error(self, llm):
        """Test ask returns None when both LLMs fail."""
        llm._local_llm.ask = AsyncMock(side_effect=Exception("Local error"))
        llm._cloud_llm.ask = AsyncMock(side_effect=Exception("Cloud error"))

        result = await llm.ask("test prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_threshold(self, llm):
        """Test TIMEOUT_THRESHOLD is set correctly."""
        assert llm.TIMEOUT_THRESHOLD == 3.2