"""
Experiment tracking framework for CLAMS model comparisons.
Supports multiple LLM providers and structured result storage.
"""
import json
import sqlite3
import hashlib
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class ModelConfig:
    """Configuration for an LLM model experiment."""
    provider: Union[LLMProvider, str]
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize provider to enum
        if isinstance(self.provider, str):
            try:
                self.provider = LLMProvider(self.provider.lower())
            except ValueError:
                pass  # Keep as string if not a known provider

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            'provider': self.provider.value if isinstance(self.provider, LLMProvider) else self.provider,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'additional_params': self.additional_params
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**d)

    def get_config_hash(self) -> str:
        """Generate unique hash for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def __str__(self) -> str:
        provider_str = self.provider.value if isinstance(self.provider, LLMProvider) else self.provider
        return f"{provider_str}:{self.model_name}"


@dataclass
class ExperimentMetrics:
    """Metrics from an experiment run."""
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Error metrics (lower is better)
    character_error_rate: Optional[float] = None  # CER
    word_error_rate: Optional[float] = None  # WER

    # Performance metrics
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    execution_time_seconds: float = 0.0

    # Cost tracking
    estimated_cost_usd: float = 0.0

    # Custom metrics
    custom: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentMetrics':
        """Create from dictionary."""
        # Handle custom field separately
        custom = d.pop('custom', {})
        metrics = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        metrics.custom = custom
        return metrics


@dataclass
class ExperimentResult:
    """Complete result from an experiment run."""
    experiment_id: str
    experiment_name: str
    model_config: ModelConfig
    metrics: ExperimentMetrics

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_name: str = ""
    dataset_size: int = 0
    task_type: str = ""

    # System info
    system_info: Dict[str, str] = field(default_factory=dict)

    # Detailed results
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Configuration used
    experiment_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'model_config': self.model_config.to_dict(),
            'metrics': self.metrics.to_dict(),
            'timestamp': self.timestamp,
            'dataset_name': self.dataset_name,
            'dataset_size': self.dataset_size,
            'task_type': self.task_type,
            'system_info': self.system_info,
            'predictions': self.predictions,
            'errors': self.errors,
            'experiment_config': self.experiment_config
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentResult':
        """Create from dictionary."""
        d = d.copy()
        d['model_config'] = ModelConfig.from_dict(d['model_config'])
        d['metrics'] = ExperimentMetrics.from_dict(d['metrics'])
        return cls(**d)


class ExperimentStorage(ABC):
    """Abstract storage interface for experiments."""

    @abstractmethod
    def save(self, result: ExperimentResult) -> str:
        """Save experiment result. Returns experiment_id."""
        pass

    @abstractmethod
    def load(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Load experiment by ID."""
        pass

    @abstractmethod
    def list_experiments(
        self,
        filter_by: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[ExperimentResult]:
        """List experiments with optional filters."""
        pass

    @abstractmethod
    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        pass


class JSONExperimentStorage(ExperimentStorage):
    """JSON file-based experiment storage."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        self._load_index()

    def _load_index(self):
        """Load experiment index."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    self.index = json.load(f)
            except Exception:
                self.index = {"experiments": {}}
        else:
            self.index = {"experiments": {}}

    def _save_index(self):
        """Save experiment index."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def save(self, result: ExperimentResult) -> str:
        """Save experiment result to JSON file."""
        # Create experiment directory
        exp_dir = self.base_dir / result.experiment_name.replace(' ', '_')
        exp_dir.mkdir(exist_ok=True)

        # Save result file
        result_path = exp_dir / f"{result.experiment_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        # Update index
        self.index["experiments"][result.experiment_id] = {
            "name": result.experiment_name,
            "path": str(result_path),
            "timestamp": result.timestamp,
            "model": result.model_config.model_name,
            "provider": str(result.model_config.provider.value if isinstance(result.model_config.provider, LLMProvider) else result.model_config.provider),
            "task": result.task_type
        }
        self._save_index()

        logger.info(f"Saved experiment {result.experiment_id} to {result_path}")
        return result.experiment_id

    def load(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Load experiment by ID."""
        if experiment_id not in self.index["experiments"]:
            return None

        path = Path(self.index["experiments"][experiment_id]["path"])
        if not path.exists():
            return None

        with open(path, 'r') as f:
            data = json.load(f)
        return ExperimentResult.from_dict(data)

    def list_experiments(
        self,
        filter_by: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[ExperimentResult]:
        """List all experiments, optionally filtered."""
        results = []

        for exp_id, meta in self.index["experiments"].items():
            # Apply filters
            if filter_by:
                skip = False
                for key, value in filter_by.items():
                    if meta.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            result = self.load(exp_id)
            if result:
                results.append(result)

            if len(results) >= limit:
                break

        return sorted(results, key=lambda r: r.timestamp, reverse=True)

    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        if experiment_id not in self.index["experiments"]:
            return False

        path = Path(self.index["experiments"][experiment_id]["path"])
        if path.exists():
            path.unlink()

        del self.index["experiments"][experiment_id]
        self._save_index()
        return True


class SQLiteExperimentStorage(ExperimentStorage):
    """SQLite-based experiment storage for larger datasets."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                experiment_name TEXT,
                timestamp TEXT,
                model_provider TEXT,
                model_name TEXT,
                task_type TEXT,
                dataset_name TEXT,
                accuracy REAL,
                f1_score REAL,
                execution_time REAL,
                data TEXT
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON experiments(timestamp)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_model
            ON experiments(model_name)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_task
            ON experiments(task_type)
        ''')

        conn.commit()
        conn.close()

    def save(self, result: ExperimentResult) -> str:
        """Save experiment to SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        provider_str = result.model_config.provider.value if isinstance(
            result.model_config.provider, LLMProvider
        ) else str(result.model_config.provider)

        cursor.execute('''
            INSERT OR REPLACE INTO experiments
            (experiment_id, experiment_name, timestamp, model_provider,
             model_name, task_type, dataset_name, accuracy, f1_score,
             execution_time, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.experiment_id,
            result.experiment_name,
            result.timestamp,
            provider_str,
            result.model_config.model_name,
            result.task_type,
            result.dataset_name,
            result.metrics.accuracy,
            result.metrics.f1_score,
            result.metrics.execution_time_seconds,
            json.dumps(result.to_dict())
        ))

        conn.commit()
        conn.close()

        logger.info(f"Saved experiment {result.experiment_id} to SQLite")
        return result.experiment_id

    def load(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Load experiment by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT data FROM experiments WHERE experiment_id = ?',
            (experiment_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return ExperimentResult.from_dict(json.loads(row[0]))
        return None

    def list_experiments(
        self,
        filter_by: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[ExperimentResult]:
        """List experiments with optional filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = 'SELECT data FROM experiments'
        params = []

        if filter_by:
            conditions = []
            field_mapping = {
                'model_name': 'model_name',
                'model': 'model_name',
                'task_type': 'task_type',
                'task': 'task_type',
                'dataset_name': 'dataset_name',
                'provider': 'model_provider'
            }
            for key, value in filter_by.items():
                db_field = field_mapping.get(key, key)
                if db_field in ['model_name', 'task_type', 'dataset_name', 'model_provider']:
                    conditions.append(f'{db_field} = ?')
                    params.append(value)
            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)

        query += f' ORDER BY timestamp DESC LIMIT {limit}'

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [ExperimentResult.from_dict(json.loads(row[0])) for row in rows]

    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'DELETE FROM experiments WHERE experiment_id = ?',
            (experiment_id,)
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted


class ExperimentRun:
    """Context manager for a single experiment run."""

    def __init__(
        self,
        tracker: 'ExperimentTracker',
        experiment_id: str,
        experiment_name: str,
        model_config: ModelConfig,
        task_type: str,
        dataset_name: str,
        config: Dict[str, Any]
    ):
        self.tracker = tracker
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.model_config = model_config
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.config = config

        self.predictions: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.metrics = ExperimentMetrics()
        self._start_time = None

        # Collect system info
        self.system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor()
        }

    def __enter__(self) -> 'ExperimentRun':
        import time
        self._start_time = time.perf_counter()
        logger.info(f"Starting experiment: {self.experiment_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        if self._start_time:
            self.metrics.execution_time_seconds = time.perf_counter() - self._start_time

        if exc_type:
            self.errors.append(f"{exc_type.__name__}: {exc_val}")

        self._save()
        logger.info(f"Experiment complete: {self.experiment_id}")
        return False  # Don't suppress exceptions

    def log_prediction(
        self,
        input_data: Any,
        prediction: Any,
        ground_truth: Any = None,
        metadata: Optional[Dict] = None
    ):
        """Log a single prediction."""
        self.predictions.append({
            "input": str(input_data)[:500] if input_data else None,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "metadata": metadata or {}
        })

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics."""
        for key, value in metrics.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
            else:
                self.metrics.custom[key] = value

    def log_error(self, error: str):
        """Log an error."""
        self.errors.append(error)

    def log_tokens(self, prompt_tokens: int, completion_tokens: int):
        """Log token usage."""
        self.metrics.prompt_tokens += prompt_tokens
        self.metrics.completion_tokens += completion_tokens
        self.metrics.total_tokens = (
            self.metrics.prompt_tokens + self.metrics.completion_tokens
        )

    def _save(self):
        """Save experiment result."""
        result = ExperimentResult(
            experiment_id=self.experiment_id,
            experiment_name=self.experiment_name,
            model_config=self.model_config,
            metrics=self.metrics,
            dataset_name=self.dataset_name,
            dataset_size=len(self.predictions),
            task_type=self.task_type,
            system_info=self.system_info,
            predictions=self.predictions,
            errors=self.errors,
            experiment_config=self.config
        )
        self.tracker.storage.save(result)


class ExperimentTracker:
    """
    Main experiment tracking interface.

    Usage:
        tracker = ExperimentTracker("experiments/")

        with tracker.start_experiment("ocr-comparison", model_config) as exp:
            for sample in dataset:
                result = model.predict(sample)
                exp.log_prediction(sample, result)
            exp.log_metrics({"accuracy": 0.95})

        # Compare experiments
        comparison = tracker.compare_experiments(["exp1", "exp2"])
    """

    def __init__(
        self,
        experiments_dir: Union[str, Path],
        storage_backend: str = "json"
    ):
        """
        Initialize experiment tracker.

        Args:
            experiments_dir: Directory for storing experiments
            storage_backend: "json" or "sqlite"
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        if storage_backend == "sqlite":
            db_path = self.experiments_dir / "experiments.db"
            self.storage = SQLiteExperimentStorage(db_path)
        else:
            self.storage = JSONExperimentStorage(self.experiments_dir)

        self.current_experiment: Optional[ExperimentRun] = None
        logger.info(f"ExperimentTracker initialized with {storage_backend} storage at {experiments_dir}")

    def start_experiment(
        self,
        name: str,
        model_config: ModelConfig,
        task_type: str = "",
        dataset_name: str = "",
        config: Optional[Dict[str, Any]] = None
    ) -> ExperimentRun:
        """Start a new experiment run."""
        experiment_id = self._generate_id(name, model_config)

        run = ExperimentRun(
            tracker=self,
            experiment_id=experiment_id,
            experiment_name=name,
            model_config=model_config,
            task_type=task_type,
            dataset_name=dataset_name,
            config=config or {}
        )

        self.current_experiment = run
        return run

    def _generate_id(self, name: str, config: ModelConfig) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = config.get_config_hash()
        clean_name = name.replace(' ', '_').replace('/', '-')[:20]
        return f"{clean_name}_{config_hash}_{timestamp}"

    def load_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Load a specific experiment by ID."""
        return self.storage.load(experiment_id)

    def list_experiments(
        self,
        name: Optional[str] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        limit: int = 100
    ) -> List[ExperimentResult]:
        """List experiments with optional filters."""
        filters = {}
        if model:
            filters['model_name'] = model
        if task:
            filters['task_type'] = task

        results = self.storage.list_experiments(filters if filters else None, limit)

        if name:
            results = [r for r in results if name.lower() in r.experiment_name.lower()]

        return results

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Specific metrics to compare (default: common metrics)

        Returns:
            Comparison dictionary with experiments and summary statistics
        """
        results = []
        for exp_id in experiment_ids:
            result = self.storage.load(exp_id)
            if result:
                results.append(result)

        if not results:
            return {"error": "No experiments found", "experiments": []}

        # Default metrics to compare
        if metrics is None:
            metrics = [
                "accuracy", "f1_score", "precision", "recall",
                "character_error_rate", "word_error_rate",
                "execution_time_seconds", "total_tokens"
            ]

        comparison = {
            "experiments": [],
            "metrics_summary": {}
        }

        for result in results:
            exp_summary = {
                "id": result.experiment_id,
                "name": result.experiment_name,
                "model": result.model_config.model_name,
                "provider": str(result.model_config.provider.value if isinstance(result.model_config.provider, LLMProvider) else result.model_config.provider),
                "timestamp": result.timestamp,
                "dataset_size": result.dataset_size,
                "metrics": {}
            }

            for metric in metrics:
                value = getattr(result.metrics, metric, None)
                if value is None:
                    value = result.metrics.custom.get(metric)
                exp_summary["metrics"][metric] = value

            comparison["experiments"].append(exp_summary)

        # Calculate summary statistics for each metric
        for metric in metrics:
            values = [
                e["metrics"][metric]
                for e in comparison["experiments"]
                if e["metrics"][metric] is not None
            ]
            if values:
                # Determine if lower or higher is better
                lower_is_better = metric in [
                    'character_error_rate', 'word_error_rate',
                    'execution_time_seconds', 'total_tokens', 'estimated_cost_usd'
                ]

                best_value = min(values) if lower_is_better else max(values)
                best_idx = values.index(best_value)

                comparison["metrics_summary"][metric] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "best_experiment": comparison["experiments"][best_idx]["id"],
                    "best_value": best_value,
                    "lower_is_better": lower_is_better
                }

        return comparison

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        return self.storage.delete(experiment_id)


# Convenience function
def get_experiment_tracker(
    experiments_dir: Optional[Union[str, Path]] = None
) -> ExperimentTracker:
    """Get experiment tracker with default or custom directory."""
    if experiments_dir is None:
        from .config import get_config_manager
        try:
            config = get_config_manager().get_config()
            experiments_dir = config.experiment.experiments_dir
        except Exception:
            experiments_dir = "experiments"

    return ExperimentTracker(experiments_dir)
