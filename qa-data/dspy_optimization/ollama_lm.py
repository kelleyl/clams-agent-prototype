"""DSPy LM configuration helpers for Ollama and vLLM backends.

Uses DSPy's native LM support via LiteLLM, which correctly handles
callbacks, history, adapters, and structured output parsing.

See: https://dspy.ai/learn/programming/language_models/
"""
import dspy
import logging

logger = logging.getLogger(__name__)


def make_ollama_lm(model: str,
                   api_base: str = "http://localhost:11434",
                   temperature: float = 0.7,
                   max_tokens: int = 4000,
                   **kwargs) -> dspy.LM:
    """Create a DSPy LM via Ollama (ollama_chat/ prefix)."""
    model_id = f"ollama_chat/{model}" if not model.startswith("ollama") else model
    lm = dspy.LM(
        model=model_id,
        api_base=api_base,
        api_key="",
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    logger.info(f"Created DSPy LM: {model_id} @ {api_base}")
    return lm


def make_vllm_lm(model: str,
                 api_base: str = "http://localhost:8100/v1",
                 temperature: float = 0.7,
                 max_tokens: int = 4000,
                 **kwargs) -> dspy.LM:
    """Create a DSPy LM via vLLM's OpenAI-compatible API.

    Args:
        model: Model name as served by vLLM (e.g. "Qwen/Qwen3.5-9B").
        api_base: vLLM server URL with /v1 suffix.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.

    Returns:
        A dspy.LM instance ready for dspy.configure(lm=...).
    """
    # vLLM serves an OpenAI-compatible API, so use the openai/ prefix
    model_id = f"openai/{model}" if not model.startswith("openai/") else model
    lm = dspy.LM(
        model=model_id,
        api_base=api_base,
        api_key="unused",  # vLLM doesn't require a key by default
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    logger.info(f"Created DSPy LM: {model_id} @ {api_base}")
    return lm
