from typing import Dict, Any
import os
import json
from dataclasses import dataclass, asdict, field

@dataclass
class LLMConfig:
    """Configuration for the LLM component."""
    model_name: str = "deepseek-r1:8b"  # Default Ollama model
    base_url: str = "http://localhost:11434"  # Ollama default URL
    temperature: float = 0.7
    top_p: float = 0.9
    max_length: int = 2048
    provider: str = "ollama"  # "ollama" or "openai"
    system_prompt: str = """You are an AI assistant helping to analyze video content using CLAMS tools. 
Your goal is to understand user requests about video content and create appropriate pipelines of CLAMS tools to process the videos.
You have access to various CLAMS tools that can analyze different aspects of videos, such as:
- Text detection and recognition
- Speech transcription
- Scene detection
- Object detection
- And more

When responding to user queries:
1. Understand what information they want to extract from the video
2. Determine which CLAMS tools would be most appropriate
3. Consider the optimal order of tool execution
4. Suggest appropriate parameters for each tool
5. Explain your reasoning clearly"""

@dataclass
class AppConfig:
    """Configuration for the CLAMS Agent application."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    cache_dir: str = "data/cache"
    max_video_size: int = 500 * 1024 * 1024  # 500MB
    supported_video_formats: list = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
    

class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default config.
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config.json"
        )
        self.config = self._load_config()
        
    def _load_config(self) -> AppConfig:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)
                    llm_config = LLMConfig(**config_dict.get('llm', {}))
                    return AppConfig(llm=llm_config)
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
                
        return AppConfig()
        
    def save_config(self):
        """Save current configuration to file."""
        config_dict = asdict(self.config)
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        if 'llm' in updates:
            self.config.llm = LLMConfig(**updates['llm'])
        
        # Update other top-level configs
        for key, value in updates.items():
            if key != 'llm' and hasattr(self.config, key):
                setattr(self.config, key, value)
                
        self.save_config()
        
    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        return self.config 