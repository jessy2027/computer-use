"""
Client for interacting with Ollama API.
"""
import httpx
from typing import Any, Optional
from anthropic.types.beta import BetaMessage

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://192.168.1.143"):
        self.base_url = base_url
        self.client = httpx.Client()
        
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
        """
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
        
        # Make request to Ollama API
        response = self.client.post(
            f"{self.base_url}/api/chat",
            json=request_data
        )
        
        # Convert Ollama response to Anthropic format
        ollama_response = response.json()
        
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