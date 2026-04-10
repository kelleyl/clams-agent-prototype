"""
Configuration management for CLAMS Agent.
Supports LLM, execution backend, and experiment tracking configuration.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import os
import json


class ExecutionBackend(str, Enum):
    """Available execution backends for CLAMS apps."""
    CLI = "cli"
    DOCKER = "docker"
    HTTP = "http"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    """Configuration for the LLM component."""
    provider: str = "ollama"  # "ollama", "openai", "anthropic"
    model_name: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"  # Ollama default
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    api_key: Optional[str] = None
    system_prompt: str = """You are an AI assistant helping to analyze video content using CLAMS tools.
Your goal is to understand user requests about video content and create appropriate pipelines of CLAMS tools to process the videos."""

    def __post_init__(self):
        # Load API key from environment if not set
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")


@dataclass
class ExecutionConfig:
    """Configuration for CLAMS app execution."""
    # Default backend to use
    default_backend: str = "cli"

    # CLI backend settings
    cli_apps_dir: str = field(default_factory=lambda: os.environ.get(
        "CLAMS_APPS_DIR",
        str(Path.home() / "clams")
    ))

    # GPU availability (controls which execution tools are enabled)
    gpu_available: bool = False

    # Docker backend settings
    docker_registry: str = "ghcr.io/clamsproject"
    docker_cache_dir: str = "/cache"
    docker_gpu_enabled: bool = True
    docker_gpu_device: str = "0"
    docker_network: str = "host"

    # HTTP backend settings - map app names to base URLs
    http_endpoints: Dict[str, str] = field(default_factory=dict)

    # App-specific backend overrides
    # e.g., {"smolvlm2-captioner": "docker", "swt-detection": "http"}
    app_backends: Dict[str, str] = field(default_factory=dict)

    # Timeout settings
    default_timeout: int = 300  # 5 minutes
    app_timeouts: Dict[str, int] = field(default_factory=dict)

    def get_backend_for_app(self, app_name: str) -> ExecutionBackend:
        """Get the execution backend for a specific app."""
        backend_str = self.app_backends.get(app_name, self.default_backend)
        return ExecutionBackend(backend_str)

    def get_timeout_for_app(self, app_name: str) -> int:
        """Get the timeout for a specific app."""
        return self.app_timeouts.get(app_name, self.default_timeout)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    experiments_dir: str = field(default_factory=lambda: os.environ.get(
        "CLAMS_EXPERIMENTS_DIR",
        "experiments"
    ))
    storage_backend: str = "json"  # "json" or "sqlite"
    auto_save: bool = True
    track_system_info: bool = True
    track_tokens: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    cache_dir: str = "data/cache"
    max_video_size: int = 500 * 1024 * 1024  # 500MB
    supported_video_formats: List[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"]
    )


class ConfigManager:
    """
    Manages application configuration with environment variable support.

    Environment variables:
        CLAMS_APPS_DIR - Directory containing CLAMS apps
        CLAMS_DEFAULT_BACKEND - Default execution backend (cli, docker, http)
        CLAMS_EXPERIMENTS_DIR - Directory for experiment results
        LLM_PROVIDER - LLM provider (ollama, openai, anthropic)
        LLM_MODEL - Model name
        OLLAMA_BASE_URL - Ollama server URL
        OPENAI_API_KEY - OpenAI API key
        ANTHROPIC_API_KEY - Anthropic API key
    """

    def __init__(self, config_path: Optional[str] = None):
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
        """Load configuration from file and environment variables."""
        config_dict = {}

        # Load from file if exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)
            except Exception as e:
                print(f"Error loading config from {self.config_path}: {e}, using defaults")

        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)

        # Build config objects
        llm_dict = config_dict.get('llm', {})
        execution_dict = config_dict.get('execution', {})
        experiment_dict = config_dict.get('experiment', {})

        llm_config = LLMConfig(**{k: v for k, v in llm_dict.items()
                                  if k in LLMConfig.__dataclass_fields__})
        execution_config = ExecutionConfig(**{k: v for k, v in execution_dict.items()
                                              if k in ExecutionConfig.__dataclass_fields__})
        experiment_config = ExperimentConfig(**{k: v for k, v in experiment_dict.items()
                                                if k in ExperimentConfig.__dataclass_fields__})

        # Build top-level config
        top_level_fields = {k: v for k, v in config_dict.items()
                          if k not in ['llm', 'execution', 'experiment']
                          and k in AppConfig.__dataclass_fields__}

        return AppConfig(
            llm=llm_config,
            execution=execution_config,
            experiment=experiment_config,
            **top_level_fields
        )

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Ensure nested dicts exist
        if 'llm' not in config:
            config['llm'] = {}
        if 'execution' not in config:
            config['execution'] = {}
        if 'experiment' not in config:
            config['experiment'] = {}

        # LLM environment variables
        env_mappings = [
            ("LLM_PROVIDER", "llm", "provider"),
            ("LLM_MODEL", "llm", "model_name"),
            ("OLLAMA_BASE_URL", "llm", "base_url"),
            ("LLM_TEMPERATURE", "llm", "temperature"),
            ("LLM_MAX_TOKENS", "llm", "max_tokens"),

            # Execution environment variables
            ("CLAMS_GPU_AVAILABLE", "execution", "gpu_available"),
            ("CLAMS_APPS_DIR", "execution", "cli_apps_dir"),
            ("CLAMS_DEFAULT_BACKEND", "execution", "default_backend"),
            ("DOCKER_REGISTRY", "execution", "docker_registry"),
            ("DOCKER_CACHE_DIR", "execution", "docker_cache_dir"),

            # Experiment environment variables
            ("CLAMS_EXPERIMENTS_DIR", "experiment", "experiments_dir"),
            ("EXPERIMENT_STORAGE", "experiment", "storage_backend"),
        ]

        for env_var, section, key in env_mappings:
            value = os.environ.get(env_var)
            if value is not None:
                # Convert types as needed
                if key in ['temperature']:
                    value = float(value)
                elif key in ['max_tokens', 'default_timeout']:
                    value = int(value)
                elif key in ['gpu_available', 'docker_gpu_enabled', 'auto_save', 'track_system_info']:
                    value = value.lower() in ('true', '1', 'yes')

                config[section][key] = value

        return config

    def save_config(self):
        """Save current configuration to file."""
        config_dict = self._config_to_dict(self.config)

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert AppConfig to serializable dictionary."""
        result = {
            'llm': asdict(config.llm),
            'execution': asdict(config.execution),
            'experiment': asdict(config.experiment),
            'cache_dir': config.cache_dir,
            'max_video_size': config.max_video_size,
            'supported_video_formats': config.supported_video_formats
        }

        # Remove None values and convert Path objects
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items() if v is not None}
            elif isinstance(d, list):
                return [clean_dict(v) for v in d]
            elif isinstance(d, Path):
                return str(d)
            return d

        return clean_dict(result)

    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates
        """
        if 'llm' in updates:
            for key, value in updates['llm'].items():
                if hasattr(self.config.llm, key):
                    setattr(self.config.llm, key, value)

        if 'execution' in updates:
            for key, value in updates['execution'].items():
                if hasattr(self.config.execution, key):
                    setattr(self.config.execution, key, value)

        if 'experiment' in updates:
            for key, value in updates['experiment'].items():
                if hasattr(self.config.experiment, key):
                    setattr(self.config.experiment, key, value)

        # Update top-level configs
        for key, value in updates.items():
            if key not in ['llm', 'execution', 'experiment'] and hasattr(self.config, key):
                setattr(self.config, key, value)

        self.save_config()

    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        return self.config

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.config.llm

    def get_execution_config(self) -> ExecutionConfig:
        """Get execution configuration."""
        return self.config.execution

    def get_experiment_config(self) -> ExperimentConfig:
        """Get experiment configuration."""
        return self.config.experiment


# Convenience function for quick config access
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get or create the global ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
