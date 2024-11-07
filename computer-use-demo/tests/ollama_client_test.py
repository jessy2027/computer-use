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
    ollama_client._initialized = False  # Reset initialization state
    """Test successful client initialization."""
    mock_response = httpx.Response(
        200,
        json={"models": []},
        request=httpx.Request("GET", "http://test:11434/api/tags")
    )
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        client = await ollama_client.initialize()
        assert client == ollama_client
        mock_get.assert_called_once_with("http://test:11434/api/tags")

@pytest.mark.asyncio
async def test_initialize_failure(ollama_client):
    ollama_client._initialized = False  # Reset initialization state
    """Test failed client initialization."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.RequestError("Connection failed")
        with pytest.raises(RuntimeError, match="Failed to connect to Ollama"):
            await ollama_client.initialize()

@pytest.mark.asyncio
async def test_ensure_model_exists_already_exists(ollama_client):
    """Test ensuring a model exists when it's already available."""
    ollama_client._initialized = True  # Ensure client is initialized
    mock_show_response = httpx.Response(
        200,
        json={"model": "llama2"},
        request=httpx.Request("GET", "http://test:11434/api/show")
    )
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_show_response
        await ollama_client.ensure_model_exists("llama2")
        mock_get.assert_called_once_with(
            "http://test:11434/api/show",
            params={"name": "llama2"}
        )

@pytest.mark.asyncio
async def test_ensure_model_exists_needs_download(ollama_client):
    ollama_client._initialized = True  # Ensure client is initialized
    """Test ensuring a model exists when it needs to be downloaded."""
    mock_show_response = httpx.Response(
        404,
        json={"error": "model not found"},
        request=httpx.Request("GET", "http://test:11434/api/show")
    )
    mock_pull_response = httpx.Response(
        200,
        json={"status": "success"},
        request=httpx.Request("POST", "http://test:11434/api/pull")
    )
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_get.return_value = mock_show_response
            mock_post.return_value = mock_pull_response
            
            await ollama_client.ensure_model_exists("llama2")
            
            mock_get.assert_called_once()
            mock_post.assert_called_once_with(
                "http://test:11434/api/pull",
                json={"name": "llama2"},
                timeout=600.0
            )

@pytest.mark.asyncio
async def test_ensure_model_exists_unsupported_model(ollama_client):
    ollama_client._initialized = True  # Ensure client is initialized
    """Test ensuring an unsupported model exists."""
    with pytest.raises(ValueError, match="Model unsupported_model is not supported"):
        await ollama_client.ensure_model_exists("unsupported_model")

@pytest.mark.asyncio
async def test_beta_messages_create(ollama_client):
    mock_show_response = httpx.Response(
        200,
        json={"model": "llama2"},
        request=httpx.Request("GET", "http://test:11434/api/show")
    )
    ollama_client._initialized = True  # Ensure client is initialized
    """Test creating a chat completion."""
    mock_response_data = {
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        }
    }
    mock_response = httpx.Response(
        200,
        json=mock_response_data,
        headers={"X-Request-ID": "test123"},
        request=httpx.Request("POST", "http://test:11434/api/chat")
    )
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
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
        assert beta_message.content[0]["text"] == "Hello! How can I help you today?"
        
        mock_post.assert_called_once_with(
            "http://test:11434/api/chat",
            json={
                "model": "llama2",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ],
                "system": "You are a helpful assistant",
                "format": "json",
                "stream": False
            }
        )

@pytest.mark.asyncio
async def test_beta_messages_create_empty_response(ollama_client):
    ollama_client._initialized = True  # Ensure client is initialized
    """Test creating a chat completion with empty response."""
    mock_response = httpx.Response(
        200,
        json={"message": {"content": ""}},
        request=httpx.Request("POST", "http://test:11434/api/chat")
    )
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        with pytest.raises(ValueError, match="Empty response from Ollama"):
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
    ollama_client._initialized = True  # Ensure client is initialized
    """Test creating a chat completion with invalid response format."""
    mock_response = httpx.Response(
        200,
        json={"invalid": "format"},
        request=httpx.Request("POST", "http://test:11434/api/chat")
    )
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        with pytest.raises(ValueError, match="Invalid response format from Ollama"):
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