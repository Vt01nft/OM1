"""Tests for NearAILLM plugin."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm import LLMConfig
from llm.output_model import Action, CortexOutputModel
from llm.plugins.near_ai_llm import NearAILLM


# Test output model
class DummyOutputModel(BaseModel):
    test_field: str


@pytest.fixture
def config():
    """Fixture providing a valid LLMConfig for testing."""
    return LLMConfig(
        base_url="https://api.openmind.org/api/core/nearai",
        api_key="test_key",
        model="qwen3-30b-a3b-instruct-2507",
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
            message=MagicMock(
                content='{"test_field": "success"}', tool_calls=[tool_call]
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
            "llm.plugins.near_ai_llm.AvatarLLMState.trigger_thinking", mock_decorator
        ),
        patch("llm.plugins.near_ai_llm.AvatarLLMState") as mock_avatar_state,
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
    """Fixture providing a NearAILLM instance."""
    return NearAILLM(config, available_actions=None)


@pytest.mark.asyncio
async def test_init_with_config(llm, config):
    """Test NearAILLM initialization with provided configuration."""
    assert llm._client.base_url == config.base_url
    assert llm._config.model == config.model


@pytest.mark.asyncio
async def test_init_default_model():
    """Test NearAILLM uses default model when not specified."""
    config = LLMConfig(
        base_url="https://api.openmind.org/api/core/nearai",
        api_key="test_key",
    )
    llm = NearAILLM(config, available_actions=None)
    assert llm._config.model == "qwen3-30b-a3b-instruct-2507"


@pytest.mark.asyncio
async def test_init_default_base_url():
    """Test NearAILLM uses default base URL when not specified."""
    config = LLMConfig(api_key="test_key", model="test_model")
    llm = NearAILLM(config, available_actions=None)
    assert str(llm._client.base_url) == "https://api.openmind.org/api/core/nearai/"


@pytest.mark.asyncio
async def test_init_empty_key():
    """Test NearAILLM raises error when API key is missing."""
    config = LLMConfig(base_url="https://api.openmind.org/api/core/nearai")
    with pytest.raises(ValueError, match="config file missing api_key"):
        NearAILLM(config, available_actions=None)


@pytest.mark.asyncio
async def test_ask_success(llm, mock_response):
    """Test successful API request and response parsing."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.beta.chat.completions,
            "parse",
            AsyncMock(return_value=mock_response),
        )

        result = await llm.ask("test prompt")
        assert result is None  # No tool_calls means None returned


@pytest.mark.asyncio
async def test_ask_with_tool_calls(llm, mock_response_with_tool_calls):
    """Test successful API request with tool calls."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.beta.chat.completions,
            "parse",
            AsyncMock(return_value=mock_response_with_tool_calls),
        )

        result = await llm.ask("test prompt")
        assert isinstance(result, CortexOutputModel)
        assert result.actions == [Action(type="test_function", value="value1")]


@pytest.mark.asyncio
async def test_ask_api_error(llm):
    """Test error handling for API exceptions."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.beta.chat.completions,
            "parse",
            AsyncMock(side_effect=Exception("API error")),
        )

        result = await llm.ask("test prompt")
        assert result is None


@pytest.mark.asyncio
async def test_ask_with_messages(llm, mock_response_with_tool_calls):
    """Test ask with conversation history messages."""
    with pytest.MonkeyPatch.context() as m:
        mock_parse = AsyncMock(return_value=mock_response_with_tool_calls)
        m.setattr(
            llm._client.beta.chat.completions,
            "parse",
            mock_parse,
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = await llm.ask("test prompt", messages=messages)

        assert isinstance(result, CortexOutputModel)
        # Verify messages were passed correctly
        call_kwargs = mock_parse.call_args[1]
        assert len(call_kwargs["messages"]) == 3  # 2 history + 1 new


@pytest.mark.asyncio
async def test_ask_io_provider_timing(llm, mock_response):
    """Test timing metrics collection."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            llm._client.beta.chat.completions,
            "parse",
            AsyncMock(return_value=mock_response),
        )

        await llm.ask("test prompt")
        assert llm.io_provider.llm_start_time is not None
        assert llm.io_provider.llm_end_time is not None
        assert llm.io_provider.llm_end_time >= llm.io_provider.llm_start_time