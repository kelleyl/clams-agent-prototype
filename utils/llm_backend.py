from typing import Dict, Any, Optional, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline
)
import logging
from dataclasses import dataclass
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a language model."""
    name: str
    model_id: str
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    load_in_8bit: bool = False  # For quantization
    context_window: int = 4096  # Maximum context window size

# Define available models
AVAILABLE_MODELS = {
    "mistral-7b": ModelConfig(
        name="mistral-7b",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        max_length=4096,
        context_window=8192
    ),
    "llama2-7b": ModelConfig(
        name="llama2-7b",
        model_id="meta-llama/Llama-2-7b-chat-hf",
        max_length=4096,
        context_window=4096
    ),
    "phi2": ModelConfig(
        name="phi2",
        model_id="microsoft/phi-2",
        max_length=2048,
        context_window=2048
    )
}

class LLMBackend:
    """
    A backend for managing and interacting with language models.
    Supports multiple models from Hugging Face and handles model loading/unloading.
    """
    
    def __init__(self, model_name: str = "llama2-7b", **kwargs):
        """
        Initialize the LLM backend.
        
        Args:
            model_name: Name of the model to use (must be in AVAILABLE_MODELS)
            **kwargs: Additional arguments to override model config
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not found. Available models: {list(AVAILABLE_MODELS.keys())}")
            
        self.config = AVAILABLE_MODELS[model_name]
        # Override config with kwargs
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
                
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model {self.config.model_id}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            
            # Load model with appropriate settings
            model_kwargs = {
                "device_map": self.config.device,
                "torch_dtype": torch.float16 if self.config.device == "cuda" else torch.float32,
            }
            
            if self.config.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs
            )
            
            logger.info(f"Successfully loaded {self.config.name}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The user's input prompt
            system_prompt: Optional system prompt to prepend
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            The generated response text
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded first")
            
        # Construct full prompt with system prompt if provided
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_length,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            **kwargs
        }
        
        try:
            # Tokenize input
            inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
                
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from the response
            response = response[len(full_prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        return {
            "name": self.config.name,
            "model_id": self.config.model_id,
            "device": self.config.device,
            "max_length": self.config.max_length,
            "context_window": self.config.context_window
        }
        
    def __del__(self):
        """Cleanup when the object is deleted."""
        if self.model:
            try:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error cleaning up model: {e}") 