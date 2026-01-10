"""Tests for QwenLLM plugin."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm import LLMConfig
from llm.output_model import Action, CortexOutputModel
from llm.plugins.qwen_llm import QwenLLM, _parse_qwen_tool_calls


# Test output model
class DummyOutputModel(BaseModel):
    test_field: str


@pytest.fixture
def config():
    """Fixture providing a valid LLMConfig for testing."""
    return LLMConfig(
        base_url="http://127.0.0.1:8000/v1",
        api_key="test_key",
        model="RedHatAI/Qwen3-30B-A3B-quantized.w4a16",
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
def mock_response_with_tool_calls():
    """Fixture providing a mock API response with tool calls."""
    tool_call = MagicMock()
    tool_call.function.name = "test_function"
    tool_call.function.arguments = '{"arg1": "value1"}'

    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content='{"test_field": "success"}', tool_calls=[tool_call])
        )
    ]
    return response


@pytest.fixture
def mock_response_with_xml_tool_calls():
    """Fixture providing a mock API response with XML-style tool calls."""
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content='<tool_call>{"name": "test_function", "arguments": {"arg1": "value1"}}</tool_call>',
                tool_calls=None,
            )
        )
    ]
    return response


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
        patch(
            "llm.plugins.qwen_llm.AvatarLLMState.trigger_thinking", mock_decorator
        ),
        patch("llm.plugins.qwen_llm.AvatarLLMState") as mock_avatar_state,
        patch("providers.avatar_provider.AvatarProvider") as mock_avatar_provider,
        patch(
            "providers.avatar_llm_state_provider.AvatarProvider"
        ) as mock_avatar_llm_state_provider,
    ):
        mock_avatar_state._instance = None
        mock_avatar_state._lock = None

        mock_provider_instance = MagicMock()
        mock_provider_instance.running = False
        mock_provider_instance.session = None
        mock_provider_instance.stop = MagicMock()
        mock_avatar_provider.return_value = mock_provider_instance
        mock_avatar_llm_state_provider.return_value = mock_provider_instance

        yield


@pytest.fixture
def llm(config):
    """Fixture providing a QwenLLM instance."""
    return QwenLLM(config, available_actions=None)


# Tests for _parse_qwen_tool_calls helper function
class TestParseQwenToolCalls:
    """Tests for the _parse_qwen_tool_calls helper function."""

    def test_parse_valid_tool_call(self):
        """Test parsing a valid XML tool call."""
        text = '<tool_call>{"name": "test_func", "arguments": {"key": "value"}}</tool_call>'
        result = _parse_qwen_tool_calls(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "test_func"

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple XML tool calls."""
        text = (
            '<tool_call>{"name": "func1", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "func2", "arguments": {}}</tool_call>'
        )
        result = _parse_qwen_tool_calls(text)
        assert len(result) == 2

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty list."""
        result = _parse_qwen_tool_calls("")
        assert result == []

    def test_parse_no_tool_calls(self):
        """Test parsing text without tool calls returns empty list."""
        result = _parse_qwen_tool_calls("Just some regular text")
        assert result == []

    def test_parse_non_string_input(self):
        """Test parsing non-string input returns empty list."""
        result = _parse_qwen_tool_calls(None)  # type: ignore[arg-type]
        assert result == []

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON in tool call continues gracefully."""
        text = "<tool_call>not valid json</tool_call>"
        result = _parse_qwen_tool_calls(text)
        assert result == []


# Tests for QwenLLM class
class TestQwenLLM:
    """Tests for the QwenLLM class."""

    @pytest.mark.asyncio
    async def test_init_with_config(self, llm, config):
        """Test QwenLLM initialization with config."""
        assert llm._config.model == config.model

    @pytest.mark.asyncio
    async def test_init_default_model(self):
        """Test QwenLLM uses default model when not specified."""
        config = LLMConfig(base_url="http://127.0.0.1:8000/v1", api_key="test_key")
        llm = QwenLLM(config, available_actions=None)
        assert llm._config.model == "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"

    @pytest.mark.asyncio
    async def test_init_extra_body(self, llm):
        """Test QwenLLM sets extra_body with thinking disabled."""
        assert llm._extra_body == {"chat_template_kwargs": {"enable_thinking": False}}

    @pytest.mark.asyncio
    async def test_ask_success(self, llm, mock_response):
        """Test successful ask with valid response."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.chat.completions,
                "create",
                AsyncMock(return_value=mock_response),
            )

            result = await llm.ask("test prompt")
            assert result is None or isinstance(result, CortexOutputModel)

    @pytest.mark.asyncio
    async def test_ask_with_tool_calls(self, llm, mock_response_with_tool_calls):
        """Test ask with tool calls in response."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.chat.completions,
                "create",
                AsyncMock(return_value=mock_response_with_tool_calls),
            )

            result = await llm.ask("test prompt")
            assert isinstance(result, CortexOutputModel)
            assert result.actions == [Action(type="test_function", value="value1")]

    @pytest.mark.asyncio
    async def test_ask_with_xml_tool_calls(self, llm, mock_response_with_xml_tool_calls):
        """Test ask with XML-style tool calls in response content."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.chat.completions,
                "create",
                AsyncMock(return_value=mock_response_with_xml_tool_calls),
            )

            result = await llm.ask("test prompt")
            assert isinstance(result, CortexOutputModel)
            assert len(result.actions) == 1
            assert result.actions[0].type == "test_function"

    @pytest.mark.asyncio
    async def test_ask_api_error(self, llm):
        """Test ask handles API errors gracefully."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.chat.completions,
                "create",
                AsyncMock(side_effect=Exception("API error")),
            )

            result = await llm.ask("test prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_ask_invalid_json(self, llm):
        """Test ask handles invalid JSON response gracefully."""
        invalid_response = MagicMock()
        invalid_response.choices = [MagicMock(message=MagicMock(content="invalid"))]

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                llm._client.chat.completions,
                "create",
                AsyncMock(return_value=invalid_response),
            )

            result = await llm.ask("test prompt")
            assert result is None or isinstance(result, CortexOutputModel)