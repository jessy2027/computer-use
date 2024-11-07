"""
Client for interacting with Ollama API.
"""
import os
import httpx
from typing import Any, Dict, List, Optional, Union
from anthropic.types.beta import BetaMessage

class OllamaResponse:
    """Wrapper for Ollama API response to match Anthropic's interface."""
    def __init__(self, beta_message: BetaMessage, http_response: httpx.Response):
        self.beta_message = beta_message
        self.http_response = http_response

    def parse(self) -> BetaMessage:
        return self.beta_message

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    SUPPORTED_MODELS = ["llama2", "mistral", "neural-chat"]  # List of supported models
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        # Initialize beta property immediately
        self.beta = self.Beta(self)
        self._initialized = False
        
    async def initialize(self):
        """Initialize the client by testing the connection to Ollama"""
        try:
            async with httpx.AsyncClient() as async_client:
                response = await async_client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
            self._initialized = True
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")
            
    async def ensure_model_exists(self, model_name: str) -> None:
        """Ensure a model exists, downloading it if necessary.
        
        Args:
            model_name: The name of the model to ensure exists
            
        Raises:
            ValueError: If the model is not supported
            RuntimeError: If the model cannot be loaded or downloaded
        """
        if not self._initialized:
            await self.initialize()
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} is not supported. Supported models: {self.SUPPORTED_MODELS}")
            
        try:
            async with httpx.AsyncClient() as async_client:
                # Check if model exists
                response = await async_client.get(f"{self.base_url}/api/show", params={"name": model_name})
                if response.status_code == 404:
                    # Model doesn't exist, try to pull it
                    pull_response = await async_client.post(
                        f"{self.base_url}/api/pull",
                        json={"name": model_name},
                        timeout=600.0  # 10 minutes timeout for model download
                    )
                    pull_response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to ensure model {model_name} exists: {e}")
        
    class Beta:
        def __init__(self, client):
            self.client = client
            self.messages = self.Messages(client)
            
        class Messages:
            def __init__(self, client):
                self.client = client
                
            def with_raw_response(self):
                """Method chaining to match Anthropic's interface"""
                return self
                
            async def create(
                self,
                max_tokens: int,
                messages: list[dict],
                model: str,
                system: list[dict],
                tools: list[dict],
                betas: list[str],
            ) -> OllamaResponse:
                """
                Create a chat completion with Ollama API.
                
                Args:
                    max_tokens: Maximum tokens to generate
                    messages: Chat history in the format [{"role": str, "content": str}]
                    model: Model name to use (must be one of the supported models)
                    system: System messages in the format [{"text": str}]
                    tools: List of available tools (currently not used by Ollama)
                    betas: List of beta features to enable (currently not used by Ollama)
                    
                Returns:
                    OllamaResponse: Wrapper containing both the HTTP response and parsed BetaMessage
                    
                Raises:
                    ValueError: If the model is not supported or if the input format is invalid
                    RuntimeError: If connection or model loading fails
                """
                # Validate input parameters
                if not messages or not isinstance(messages, list):
                    raise ValueError("Messages must be a non-empty list")
                
                for msg in messages:
                    if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                        raise ValueError("Each message must be a dict with 'role' and 'content' keys")
                    if msg["role"] not in ["user", "assistant", "system"]:
                        raise ValueError(f"Invalid message role: {msg['role']}")
                
                if not model:
                    raise ValueError("Model name is required")
                
                # Ensure model exists and is ready
                await self.client.ensure_model_exists(model)
                
                system_prompt = system[0].get("text", "") if system else ""
                
                # Convert messages to Ollama format
                ollama_messages = []
                for msg in messages:
                    if isinstance(msg["content"], str):
                        content = msg["content"]
                    else:
                        # If content is a list of blocks, combine text blocks
                        content = "\n".join(
                            block["text"] 
                            for block in msg["content"] 
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    ollama_messages.append({
                        "role": msg["role"],
                        "content": content
                    })
                    
                # Prepare Ollama request
                request_data = {
                    "model": model,
                    "messages": ollama_messages,
                    "system": system_prompt,
                    "format": "json",  # Request JSON output for better parsing
                    "stream": False
                }
                
                try:
                    # Make request to Ollama API asynchronously
                    async with httpx.AsyncClient(timeout=60.0) as async_client:
                        try:
                            http_response = await async_client.post(
                                f"{self.client.base_url}/api/chat",
                                json=request_data
                            )
                            http_response.raise_for_status()
                        except httpx.TimeoutException as e:
                            raise RuntimeError(f"Request to Ollama timed out: {e}")
                        except httpx.RequestError as e:
                            raise RuntimeError(f"Failed to connect to Ollama: {e}")
                        except httpx.HTTPStatusError as e:
                            raise RuntimeError(f"Ollama API returned error {e.response.status_code}: {e.response.text}")
                    
                        try:
                            # Convert Ollama response to Anthropic format
                            ollama_response = http_response.json()
                        except ValueError as e:
                            raise ValueError(f"Invalid JSON response from Ollama: {e}")
                    
                        if not isinstance(ollama_response, dict):
                            raise ValueError(f"Expected dict response, got {type(ollama_response)}")
                        
                        if "message" not in ollama_response:
                            raise ValueError(f"Response missing 'message' field: {ollama_response}")
                        
                        content = ollama_response.get("message", {}).get("content", "")
                        if not content:
                            raise ValueError("Empty response content from Ollama")
                        
                except Exception as e:
                    if isinstance(e, (ValueError, RuntimeError)):
                        raise
                    raise RuntimeError(f"Unexpected error while getting response from Ollama: {e}")
                
                # Create BetaMessage response
                beta_message = BetaMessage(
                    id="msg_" + http_response.headers.get("X-Request-ID", "unknown"),
                    type="message",
                    role="assistant",
                    content=[
                        {"type": "text", "text": ollama_response["message"]["content"]}
                    ],
                    model=model,
                    stop_reason="stop_sequence",
                    stop_sequence=None,
                    usage={
                        "input_tokens": 0,  # Ollama doesn't provide token counts
                        "output_tokens": 0
                    }
                )
                
                return OllamaResponse(beta_message, http_response)