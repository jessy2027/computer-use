"""
Client for interacting with Ollama API.
"""
import httpx
from typing import Any, Optional
from anthropic.types.beta import BetaMessage

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    SUPPORTED_MODELS = ["llama2", "mistral", "neural-chat"]  # List of supported models
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=60.0)  # Increased timeout for model loading
        
        # Test connection and list available models
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")
            
    def ensure_model_exists(self, model_name: str) -> None:
        """Ensure a model exists, downloading it if necessary."""
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} is not supported. Supported models: {self.SUPPORTED_MODELS}")
            
        try:
            # Check if model exists
            response = self.client.get(f"{self.base_url}/api/show", params={"name": model_name})
            if response.status_code == 404:
                # Model doesn't exist, try to pull it
                pull_response = self.client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name},
                    timeout=600.0  # 10 minutes timeout for model download
                )
                pull_response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to ensure model {model_name} exists: {e}")
        
    def beta(self):
        return self
        
    def messages(self):
        return self
        
    def with_raw_response(self):
        return self
        
    async def create(
        self,
        max_tokens: int,
        messages: list[dict],
        model: str,
        system: list[dict],
        tools: list[dict],
        betas: list[str],
    ) -> BetaMessage:
        """
        Create a chat completion with Ollama API.
        
        Args:
            max_tokens: Maximum tokens to generate
            messages: Chat history
            model: Model name to use
            system: System messages
            tools: Available tools
            betas: Beta features to enable
            
        Returns:
            BetaMessage: Response from Ollama API adapted to Anthropic format
            
        Raises:
            ValueError: If the model is not supported
            RuntimeError: If connection or model loading fails
        """
        # Ensure model exists and is ready
        self.ensure_model_exists(model)
        
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
            # Make request to Ollama API
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json=request_data
            )
            response.raise_for_status()
            
            # Convert Ollama response to Anthropic format
            ollama_response = response.json()
            
            if not isinstance(ollama_response, dict) or "message" not in ollama_response:
                raise ValueError(f"Invalid response format from Ollama: {ollama_response}")
                
            content = ollama_response.get("message", {}).get("content", "")
            if not content:
                raise ValueError("Empty response from Ollama")
                
        except Exception as e:
            raise RuntimeError(f"Failed to get response from Ollama: {e}")
        
        # Create BetaMessage response
        return BetaMessage(
            id="msg_" + response.headers.get("X-Request-ID", "unknown"),
            type="message",
            role="assistant",
            content=[{
                "type": "text",
                "text": ollama_response.get("message", {}).get("content", "")
            }],
            model=model,
            stop_reason="stop_sequence",
            stop_sequence=None,
            usage={
                "input_tokens": 0,  # Ollama doesn't provide token counts
                "output_tokens": 0
            }
        )

    def parse(self) -> BetaMessage:
        """
        Parse the raw response into a BetaMessage.
        
        Returns:
            BetaMessage: Parsed response
        """
        return self.response  # Response already parsed in create()