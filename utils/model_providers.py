"""
Unified interface for multiple LLM providers.
Provides consistent API for OpenAI, Anthropic, and Ollama.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Generator, Union
import logging
import json

from .experiment_tracker import ModelConfig, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    latency_seconds: float = 0.0
    raw_response: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding raw_response)."""
        return {
            "content": self.content,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "latency_seconds": self.latency_seconds
        }


@dataclass
class Message:
    """Chat message structure."""
    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate a response from chat messages."""
        pass

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from the LLM. Default implementation uses non-streaming."""
        response = self.generate(prompt, system_prompt, **kwargs)
        yield response.content

    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        return True


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (GPT models)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            import openai
            api_key = self.config.additional_params.get('api_key')
            if not api_key:
                import os
                api_key = os.environ.get('OPENAI_API_KEY')

            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
            else:
                logger.warning("OpenAI API key not found")
        except ImportError:
            logger.warning("openai package not installed")

    def is_available(self) -> bool:
        return self.client is not None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using OpenAI chat completions."""
        messages = []
        if system_prompt:
            messages.append(Message("system", system_prompt))
        messages.append(Message("user", prompt))
        return self.chat(messages, **kwargs)

    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate from chat messages."""
        import time

        if not self.client:
            return LLMResponse(
                content="Error: OpenAI client not initialized",
                finish_reason="error"
            )

        start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=kwargs.get('model', self.config.model_name),
                messages=[m.to_dict() for m in messages],
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                top_p=kwargs.get('top_p', self.config.top_p)
            )

            latency = time.perf_counter() - start_time

            return LLMResponse(
                content=response.choices[0].message.content or "",
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
                model=response.model,
                finish_reason=response.choices[0].finish_reason or "",
                latency_seconds=latency,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                finish_reason="error",
                latency_seconds=time.perf_counter() - start_time
            )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from OpenAI."""
        if not self.client:
            yield "Error: OpenAI client not initialized"
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=kwargs.get('model', self.config.model_name),
                messages=messages,
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield f"Error: {str(e)}"


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider (Claude models)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            api_key = self.config.additional_params.get('api_key')
            if not api_key:
                import os
                api_key = os.environ.get('ANTHROPIC_API_KEY')

            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                logger.warning("Anthropic API key not found")
        except ImportError:
            logger.warning("anthropic package not installed")

    def is_available(self) -> bool:
        return self.client is not None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using Anthropic messages API."""
        import time

        if not self.client:
            return LLMResponse(
                content="Error: Anthropic client not initialized",
                finish_reason="error"
            )

        start_time = time.perf_counter()

        try:
            response = self.client.messages.create(
                model=kwargs.get('model', self.config.model_name),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.config.temperature)
            )

            latency = time.perf_counter() - start_time

            content = ""
            if response.content and len(response.content) > 0:
                content = response.content[0].text

            return LLMResponse(
                content=content,
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=(response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
                model=response.model,
                finish_reason=response.stop_reason or "",
                latency_seconds=latency,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                finish_reason="error",
                latency_seconds=time.perf_counter() - start_time
            )

    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate from chat messages."""
        import time

        if not self.client:
            return LLMResponse(
                content="Error: Anthropic client not initialized",
                finish_reason="error"
            )

        start_time = time.perf_counter()

        # Extract system message and convert format
        system_prompt = ""
        anthropic_messages = []
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        try:
            response = self.client.messages.create(
                model=kwargs.get('model', self.config.model_name),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                system=system_prompt,
                messages=anthropic_messages,
                temperature=kwargs.get('temperature', self.config.temperature)
            )

            latency = time.perf_counter() - start_time

            content = ""
            if response.content and len(response.content) > 0:
                content = response.content[0].text

            return LLMResponse(
                content=content,
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=(response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
                model=response.model,
                finish_reason=response.stop_reason or "",
                latency_seconds=latency,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                finish_reason="error",
                latency_seconds=time.perf_counter() - start_time
            )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from Anthropic."""
        if not self.client:
            yield "Error: Anthropic client not initialized"
            return

        try:
            with self.client.messages.stream(
                model=kwargs.get('model', self.config.model_name),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            yield f"Error: {str(e)}"


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.additional_params.get(
            'base_url', 'http://localhost:11434'
        )
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def is_available(self) -> bool:
        return self._available

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using Ollama API."""
        messages = []
        if system_prompt:
            messages.append(Message("system", system_prompt))
        messages.append(Message("user", prompt))
        return self.chat(messages, **kwargs)

    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate from chat messages via Ollama."""
        import time
        import requests

        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": kwargs.get('model', self.config.model_name),
                    "messages": [m.to_dict() for m in messages],
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get('temperature', self.config.temperature),
                        "top_p": kwargs.get('top_p', self.config.top_p),
                        "num_predict": kwargs.get('max_tokens', self.config.max_tokens)
                    }
                },
                timeout=kwargs.get('timeout', 300)
            )

            latency = time.perf_counter() - start_time

            if response.status_code != 200:
                return LLMResponse(
                    content=f"Error: HTTP {response.status_code}",
                    finish_reason="error",
                    latency_seconds=latency
                )

            data = response.json()

            return LLMResponse(
                content=data.get('message', {}).get('content', ''),
                prompt_tokens=data.get('prompt_eval_count', 0),
                completion_tokens=data.get('eval_count', 0),
                total_tokens=data.get('prompt_eval_count', 0) + data.get('eval_count', 0),
                model=self.config.model_name,
                finish_reason=data.get('done_reason', 'stop'),
                latency_seconds=latency,
                raw_response=data
            )
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                finish_reason="error",
                latency_seconds=time.perf_counter() - start_time
            )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream response from Ollama."""
        import requests

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": kwargs.get('model', self.config.model_name),
                    "messages": messages,
                    "stream": True
                },
                stream=True,
                timeout=kwargs.get('timeout', 300)
            )

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        yield data['message']['content']
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            yield f"Error: {str(e)}"

    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
        return []


def get_provider(config: ModelConfig) -> BaseLLMProvider:
    """
    Factory function to get appropriate provider.

    Args:
        config: Model configuration with provider info

    Returns:
        Appropriate LLM provider instance
    """
    provider_type = config.provider
    if isinstance(provider_type, str):
        try:
            provider_type = LLMProvider(provider_type.lower())
        except ValueError:
            raise ValueError(f"Unknown provider: {config.provider}")

    providers = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.OLLAMA: OllamaProvider
    }

    provider_class = providers.get(provider_type)
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_type}")

    return provider_class(config)


def create_provider_from_config() -> BaseLLMProvider:
    """Create provider from application config."""
    from .config import get_config_manager

    config = get_config_manager().get_config().llm

    model_config = ModelConfig(
        provider=config.provider,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        additional_params={
            'api_key': config.api_key,
            'base_url': config.base_url
        }
    )

    return get_provider(model_config)
