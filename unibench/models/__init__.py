from .base import ModelClient, ModelResponse
from .mock_client import MockClient
from .groq_client import GroqClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient

PROVIDER_REGISTRY = {
    "mock": MockClient,
    "groq": GroqClient,
    "gemini": GeminiClient,
    "anthropic": AnthropicClient,
}


def build_client(name: str, provider: str, model_id: str) -> ModelClient:
    """Factory: instantiate the right ModelClient subclass for a config entry."""
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {list(PROVIDER_REGISTRY)}"
        )
    return PROVIDER_REGISTRY[provider](name=name, model_id=model_id)
