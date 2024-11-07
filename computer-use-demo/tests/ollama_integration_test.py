"""Integration tests for the Ollama client."""
import os
import pytest
import httpx
import asyncio
from computer_use_demo.ollama_client import OllamaClient

@pytest.mark.asyncio
async def test_send_test_message():
    """Test sending a 'test' message to the Ollama API."""
    # Create client
    client = OllamaClient(base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    
    try:
        # Initialize client
        await client.initialize()
        
        # Create a message
        response = await client.beta.messages.create(
            max_tokens=100,
            messages=[{"role": "user", "content": "test"}],
            model="llama2",  # Use llama2 as default
            system=[{"text": "You are a helpful assistant"}],
            tools=[],
            betas=[]
        )
        
        # Get and parse the response
        message = response.parse()
        
        # Basic validation of the response
        assert message.role == "assistant"
        assert len(message.content) > 0
        assert message.content[0].type == "text"
        assert message.content[0].text
        
        # Print the response for manual inspection
        print(f"\nOllama response: {message.content[0].text}")
        
    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")
        
    finally:
        # Cleanup if needed
        pass