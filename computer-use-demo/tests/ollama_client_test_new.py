"""Tests for the Ollama client module."""
import os
import pytest
import httpx
import asyncio
import contextlib
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock, patch, MagicMock

@contextlib.asynccontextmanager
async def mock_http_client(mock_responses: Dict[str, httpx.Response]) -> AsyncGenerator[None, None]:
    """Context manager to mock httpx client responses."""
    async def mock_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        key = f"{method} {url}"
        if key not in mock_responses:
            raise RuntimeError(f"No mock response for {key}")
        return mock_responses[key]
    
    with patch("httpx.AsyncClient.request", new=mock_request):
        yield

from computer_use_demo.ollama_client import OllamaClient, OllamaResponse
from anthropic.types.beta import BetaMessage

# Sample test data
SAMPLE_MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

SAMPLE_SYSTEM = [{"text": "You are a helpful assistant"}]

SAMPLE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "test_function",
        "description": "A test function",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}]

@pytest.fixture
async def ollama_client():
    """Create an Ollama client for testing."""
    client = OllamaClient(base_url="http://test:11434")
    client._initialized = True  # Skip initialization for tests
    return client

@pytest.mark.asyncio
async def test_initialize_success(ollama_client):
    """Test successful client initialization."""
    ollama_client._initialized = False  # Reset initialization state
    
    mock_response = httpx.Response(
        200,
        json={"models": []},
        request=httpx.Request("GET", f"{ollama_client.base_url}/api/tags")
    )
    
    mock_responses = {
        "GET http://test:11434/api/tags": mock_response
    }
    
    async with mock_http_client(mock_responses):
        client = await ollama_client.initialize()
        assert client == ollama_client

@pytest.mark.asyncio
async def test_initialize_failure(ollama_client):
    """Test failed client initialization."""
    ollama_client._initialized = False  # Reset initialization state
    
    with patch("httpx.AsyncClient.request", side_effect=httpx.RequestError("Connection failed")):
        with pytest.raises(RuntimeError, match="Failed to connect to Ollama"):
            await ollama_client.initialize()

@pytest.mark.asyncio
async def test_ensure_model_exists_already_exists(ollama_client):
    """Test ensuring a model exists when it's already available."""
    mock_response = httpx.Response(
        200,
        json={"model": "llama2"},
        request=httpx.Request("GET", f"{ollama_client.base_url}/api/show")
    )
    
    mock_responses = {
        "GET http://test:11434/api/show": mock_response
    }
    
    async with mock_http_client(mock_responses):
        await ollama_client.ensure_model_exists("llama2")

@pytest.mark.asyncio
async def test_ensure_model_exists_needs_download(ollama_client):
    """Test ensuring a model exists when it needs to be downloaded."""
    mock_show_response = httpx.Response(
        404,
        json={"error": "model not found"},
        request=httpx.Request("GET", f"{ollama_client.base_url}/api/show")
    )
    
    mock_pull_response = httpx.Response(
        200,
        json={"status": "success"},
        request=httpx.Request("POST", f"{ollama_client.base_url}/api/pull")
    )
    
    mock_responses = {
        "GET http://test:11434/api/show": mock_show_response,
        "POST http://test:11434/api/pull": mock_pull_response
    }
    
    async with mock_http_client(mock_responses):
        await ollama_client.ensure_model_exists("llama2")

@pytest.mark.asyncio
async def test_ensure_model_exists_unsupported_model(ollama_client):
    """Test ensuring an unsupported model exists."""
    with pytest.raises(ValueError, match="Model unsupported_model is not supported"):
        await ollama_client.ensure_model_exists("unsupported_model")

@pytest.mark.asyncio
async def test_beta_messages_create(ollama_client):
    """Test creating a chat completion."""
    mock_show_response = httpx.Response(
        200,
        json={"model": "llama2"},
        request=httpx.Request("GET", f"{ollama_client.base_url}/api/show")
    )
    
    mock_response_data = {
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        }
    }
    
    mock_chat_response = httpx.Response(
        200,
        json=mock_response_data,
        headers={"X-Request-ID": "test123"},
        request=httpx.Request("POST", f"{ollama_client.base_url}/api/chat")
    )
    
    mock_responses = {
        "GET http://test:11434/api/show": mock_show_response,
        "POST http://test:11434/api/chat": mock_chat_response
    }
    
    async with mock_http_client(mock_responses):
        response = await ollama_client.beta.messages.create(
            max_tokens=100,
            messages=SAMPLE_MESSAGES,
            model="llama2",
            system=SAMPLE_SYSTEM,
            tools=SAMPLE_TOOLS,
            betas=[]
        )
        
        assert isinstance(response, OllamaResponse)
        beta_message = response.parse()
        assert isinstance(beta_message, BetaMessage)
        assert beta_message.role == "assistant"
        for block in beta_message.content:
            assert block.type == "text"
            assert block.text == "Hello! How can I help you today?"

@pytest.mark.asyncio
async def test_beta_messages_create_empty_response(ollama_client):
    """Test creating a chat completion with empty response."""
    mock_show_response = httpx.Response(
        200,
        json={"model": "llama2"},
        request=httpx.Request("GET", f"{ollama_client.base_url}/api/show")
    )
    
    mock_chat_response = httpx.Response(
        200,
        json={"message": {"role": "assistant", "content": ""}},
        request=httpx.Request("POST", f"{ollama_client.base_url}/api/chat")
    )
    
    mock_responses = {
        "GET http://test:11434/api/show": mock_show_response,
        "POST http://test:11434/api/chat": mock_chat_response
    }
    
    async with mock_http_client(mock_responses):
        with pytest.raises(ValueError, match="Empty response content from Ollama"):
            await ollama_client.beta.messages.create(
                max_tokens=100,
                messages=SAMPLE_MESSAGES,
                model="llama2",
                system=SAMPLE_SYSTEM,
                tools=SAMPLE_TOOLS,
                betas=[]
            )

@pytest.mark.asyncio
async def test_beta_messages_create_invalid_response(ollama_client):
    """Test creating a chat completion with invalid response format."""
    mock_show_response = httpx.Response(
        200,
        json={"model": "llama2"},
        request=httpx.Request("GET", f"{ollama_client.base_url}/api/show")
    )
    
    mock_chat_response = httpx.Response(
        200,
        json={"invalid": "format"},
        request=httpx.Request("POST", f"{ollama_client.base_url}/api/chat")
    )
    
    mock_responses = {
        "GET http://test:11434/api/show": mock_show_response,
        "POST http://test:11434/api/chat": mock_chat_response
    }
    
    async with mock_http_client(mock_responses):
        with pytest.raises(ValueError, match="Response missing 'message' field"):
            await ollama_client.beta.messages.create(
                max_tokens=100,
                messages=SAMPLE_MESSAGES,
                model="llama2",
                system=SAMPLE_SYSTEM,
                tools=SAMPLE_TOOLS,
                betas=[]
            )

@pytest.mark.asyncio
async def test_beta_messages_with_raw_response(ollama_client):
    """Test the with_raw_response method chaining."""
    messages = ollama_client.beta.messages.with_raw_response()
    assert messages == ollama_client.beta.messages