"""
Unified CLAMS execution system with pluggable backends.
Routes to CLI, Docker, or HTTP execution based on configuration.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .config import ConfigManager, ExecutionConfig, ExecutionBackend
from .execution_backend import (
    BaseExecutor, CLIExecutor, DockerExecutor, HTTPExecutor, ExecutionResult
)

logger = logging.getLogger(__name__)


class UnifiedCLAMSExecutor:
    """
    Unified executor that routes to appropriate backend based on configuration.

    Supports:
    - CLI execution (local Python scripts via subprocess)
    - Docker execution (containerized apps)
    - HTTP execution (running CLAMS services)

    Usage:
        executor = UnifiedCLAMSExecutor()

        # Execute single app
        result = executor.execute_app("swt-detection", input_mmif)

        # Execute pipeline
        results = executor.execute_pipeline(
            ["swt-detection", "doctr-wrapper"],
            "/path/to/video.mp4"
        )
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """
        Initialize unified executor.

        Args:
            config: Execution configuration. If None, loads from ConfigManager.
        """
        if config is None:
            config = ConfigManager().get_config().execution

        self.config = config
        self.backends: Dict[str, BaseExecutor] = {}

        # Initialize available backends
        self._init_backends()

        logger.info(f"UnifiedCLAMSExecutor initialized with backends: "
                   f"{list(self.backends.keys())}")

    def _init_backends(self):
        """Initialize execution backends based on configuration."""
        # CLI backend
        apps_dir = Path(self.config.cli_apps_dir)
        if apps_dir.exists():
            try:
                self.backends["cli"] = CLIExecutor(str(apps_dir))
                logger.info(f"CLI backend initialized with {len(self.backends['cli'].get_available_apps())} apps")
            except Exception as e:
                logger.warning(f"Failed to initialize CLI backend: {e}")

        # Docker backend
        try:
            docker_executor = DockerExecutor(
                registry=self.config.docker_registry,
                cache_dir=self.config.docker_cache_dir,
                gpu_enabled=self.config.docker_gpu_enabled,
                gpu_device=getattr(self.config, 'docker_gpu_device', '0'),
                network=self.config.docker_network
            )
            if docker_executor._docker_available:
                self.backends["docker"] = docker_executor
                logger.info(f"Docker backend initialized with {len(docker_executor.get_available_apps())} images")
        except Exception as e:
            logger.debug(f"Docker backend not available: {e}")

        # HTTP backend
        if self.config.http_endpoints:
            try:
                self.backends["http"] = HTTPExecutor(self.config.http_endpoints)
                logger.info(f"HTTP backend initialized with {len(self.config.http_endpoints)} endpoints")
            except Exception as e:
                logger.warning(f"Failed to initialize HTTP backend: {e}")

    def get_backend_for_app(self, app_name: str) -> Optional[BaseExecutor]:
        """
        Determine which backend to use for an app.

        Priority:
        1. App-specific override from config
        2. Default backend from config
        3. Any available backend that has the app
        """
        # Check for app-specific override
        app_backend = self.config.app_backends.get(app_name)
        if app_backend and app_backend in self.backends:
            backend = self.backends[app_backend]
            if backend.is_available(app_name):
                return backend

        # Try default backend
        default = self.config.default_backend
        if default in self.backends:
            backend = self.backends[default]
            if backend.is_available(app_name):
                return backend

        # Fall back to any available backend
        for backend in self.backends.values():
            if backend.is_available(app_name):
                return backend

        return None

    def execute_app(
        self,
        app_name: str,
        input_mmif: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        backend: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a CLAMS app using the appropriate backend.

        Args:
            app_name: Name of the app to execute (without 'app-' prefix)
            input_mmif: Input MMIF document as dictionary
            parameters: App-specific parameters
            timeout: Execution timeout (uses config default if not specified)
            backend: Force specific backend ("cli", "docker", "http")

        Returns:
            ExecutionResult with output MMIF or error
        """
        # Determine timeout
        if timeout is None:
            timeout = self.config.get_timeout_for_app(app_name)

        # Get executor
        if backend and backend in self.backends:
            executor = self.backends[backend]
            if not executor.is_available(app_name):
                return ExecutionResult(
                    success=False,
                    app_name=app_name,
                    backend=backend,
                    error=f"App {app_name} not available in {backend} backend"
                )
        else:
            executor = self.get_backend_for_app(app_name)
            if not executor:
                available = self.get_available_apps()
                return ExecutionResult(
                    success=False,
                    app_name=app_name,
                    backend="none",
                    error=f"No backend available for app: {app_name}. Available apps: {list(available.keys())}"
                )

        logger.info(f"Executing {app_name} via {executor.__class__.__name__}")
        return executor.execute(app_name, input_mmif, parameters, timeout)

    def execute_pipeline(
        self,
        apps: List[str],
        video_path: str,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[ExecutionResult]:
        """
        Execute a pipeline of CLAMS apps sequentially.

        Args:
            apps: List of app names to execute in order
            video_path: Path to the input video
            parameters: Optional dict of app_name -> parameters

        Returns:
            List of execution results for each app
        """
        results = []
        current_mmif = self.create_initial_mmif(video_path)

        for app_name in apps:
            app_params = (parameters or {}).get(app_name, {})

            result = self.execute_app(
                app_name=app_name,
                input_mmif=current_mmif,
                parameters=app_params
            )
            results.append(result)

            if result.success and result.output_mmif:
                current_mmif = result.output_mmif
            else:
                logger.warning(f"Pipeline stopped at {app_name}: {result.error}")
                break

        return results

    @staticmethod
    def create_initial_mmif(video_path: str) -> Dict[str, Any]:
        """Create initial MMIF document for a video file."""
        video_path = Path(video_path).absolute()
        return {
            "metadata": {"mmif": "http://mmif.clams.ai/1.0"},
            "documents": [{
                "@type": "http://mmif.clams.ai/vocabulary/VideoDocument/v1",
                "properties": {
                    "id": "d1",
                    "mime": "video/mp4",
                    "location": f"file://{video_path}"
                }
            }],
            "views": []
        }

    @staticmethod
    def create_initial_mmif_for_images(image_paths: List[str]) -> Dict[str, Any]:
        """Create initial MMIF document with ImageDocument entries.

        Used when processing standalone images or FFmpeg-extracted frames.
        """
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".tiff": "image/tiff",
            ".tif": "image/tiff", ".bmp": "image/bmp",
        }
        documents = []
        for i, path in enumerate(image_paths):
            abs_path = Path(path).absolute()
            ext = abs_path.suffix.lower()
            documents.append({
                "@type": "http://mmif.clams.ai/vocabulary/ImageDocument/v1",
                "properties": {
                    "id": f"d{i + 1}",
                    "mime": mime_map.get(ext, "image/jpeg"),
                    "location": f"file://{abs_path}"
                }
            })
        return {
            "metadata": {"mmif": "http://mmif.clams.ai/1.0"},
            "documents": documents,
            "views": []
        }

    @staticmethod
    def create_initial_mmif_combined(
        video_path: str = None,
        image_paths: List[str] = None
    ) -> Dict[str, Any]:
        """Create MMIF with both VideoDocument and ImageDocument entries.

        Used when FFmpeg extracts frames from a video — downstream apps can
        consume either the original video or the extracted images.
        """
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".tiff": "image/tiff",
            ".tif": "image/tiff", ".bmp": "image/bmp",
        }
        documents = []
        doc_id = 1

        if video_path:
            abs_path = Path(video_path).absolute()
            documents.append({
                "@type": "http://mmif.clams.ai/vocabulary/VideoDocument/v1",
                "properties": {
                    "id": f"d{doc_id}",
                    "mime": "video/mp4",
                    "location": f"file://{abs_path}"
                }
            })
            doc_id += 1

        for path in (image_paths or []):
            abs_path = Path(path).absolute()
            ext = abs_path.suffix.lower()
            documents.append({
                "@type": "http://mmif.clams.ai/vocabulary/ImageDocument/v1",
                "properties": {
                    "id": f"d{doc_id}",
                    "mime": mime_map.get(ext, "image/jpeg"),
                    "location": f"file://{abs_path}"
                }
            })
            doc_id += 1

        return {
            "metadata": {"mmif": "http://mmif.clams.ai/1.0"},
            "documents": documents,
            "views": []
        }

    def get_available_apps(self) -> Dict[str, List[str]]:
        """
        Get all available apps and their backends.

        Returns:
            Dict mapping app_name to list of available backends
        """
        apps: Dict[str, List[str]] = {}

        for backend_name, backend in self.backends.items():
            for app_name in backend.get_available_apps():
                if app_name not in apps:
                    apps[app_name] = []
                apps[app_name].append(backend_name)

        return apps

    def get_app_info(self, app_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an app from any available backend."""
        for backend in self.backends.values():
            if backend.is_available(app_name):
                info = backend.get_app_info(app_name)
                if info:
                    return info
        return None


# Backward compatibility aliases
CLAMSExecutor = UnifiedCLAMSExecutor
CLAMSExecutionResult = ExecutionResult


# Singleton instance
_clams_executor: Optional[UnifiedCLAMSExecutor] = None


def get_clams_executor() -> UnifiedCLAMSExecutor:
    """Get or create the unified executor singleton."""
    global _clams_executor
    if _clams_executor is None:
        _clams_executor = UnifiedCLAMSExecutor()
    return _clams_executor


def reset_clams_executor():
    """Reset the executor singleton (useful for testing)."""
    global _clams_executor
    _clams_executor = None


def create_clams_execution_tools(executor: UnifiedCLAMSExecutor, gpu_available: bool = True):
    """Create LangChain tools for CLAMS app execution.

    Args:
        executor: The unified executor instance.
        gpu_available: If False, exclude GPU-requiring execution tools
            (swt-detection, doctr, smolvlm2). Generic tools like
            run_clams_pipeline and get_available_clams_apps are always included.
    """
    from langchain_core.tools import tool

    @tool("execute_swt_detection")
    def execute_swt_detection(video_path: str, use_classifier: bool = True) -> str:
        """Execute Scenes With Text (SWT) detection on a video.

        Detects frames containing text in the video, producing TimeFrame annotations
        for scenes with text like slates, chyrons, and credits.

        Args:
            video_path: Path to the video file
            use_classifier: Whether to use the CNN classifier (default: True)

        Returns:
            JSON string with detection results including timeframes
        """
        mmif = executor.create_initial_mmif(video_path)
        result = executor.execute_app(
            "swt-detection",
            mmif,
            {"useClassifier": use_classifier}
        )

        if result.success:
            # Extract key info from MMIF
            timeframes = []
            for view in result.output_mmif.get('views', []):
                for ann in view.get('annotations', []):
                    if 'TimeFrame' in ann.get('@type', ''):
                        props = ann.get('properties', {})
                        timeframes.append({
                            "start_ms": props.get('start'),
                            "end_ms": props.get('end'),
                            "label": props.get('label'),
                            "classification": props.get('classification', {})
                        })

            return json.dumps({
                "success": True,
                "app": "swt-detection",
                "backend": result.backend,
                "timeframes_detected": len(timeframes),
                "timeframes": timeframes[:20],  # First 20
                "execution_time": result.execution_time_seconds
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": result.error
            })

    @tool("execute_doctr_ocr")
    def execute_doctr_ocr(video_path: str) -> str:
        """Execute DocTR OCR on a video to extract text from frames.

        Performs optical character recognition on video frames,
        producing TextDocument annotations with the extracted text.

        Args:
            video_path: Path to the video file

        Returns:
            JSON string with OCR results including extracted text
        """
        mmif = executor.create_initial_mmif(video_path)
        result = executor.execute_app("doctr-wrapper", mmif)

        if result.success:
            # Extract text from annotations
            texts = []
            for view in result.output_mmif.get('views', []):
                for ann in view.get('annotations', []):
                    if 'TextDocument' in ann.get('@type', '') or 'Alignment' in ann.get('@type', ''):
                        props = ann.get('properties', {})
                        if props.get('text'):
                            texts.append({
                                "text": props.get('text'),
                                "confidence": props.get('confidence'),
                                "start_ms": props.get('start'),
                                "end_ms": props.get('end')
                            })

            return json.dumps({
                "success": True,
                "app": "doctr-wrapper",
                "backend": result.backend,
                "texts_extracted": len(texts),
                "texts": texts[:20],
                "execution_time": result.execution_time_seconds
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": result.error
            })

    @tool("execute_smolvlm2_captioner")
    def execute_smolvlm2_captioner(
        video_path: str,
        config: str = "default"
    ) -> str:
        """Execute SmolVLM2 captioner to generate descriptions of video frames.

        Uses a vision-language model to describe visual content in video frames,
        producing TextDocument annotations with frame descriptions.

        Args:
            video_path: Path to the video file
            config: Configuration preset to use (default, slate-direct-transcription, etc.)

        Returns:
            JSON string with captioning results
        """
        mmif = executor.create_initial_mmif(video_path)
        result = executor.execute_app(
            "smolvlm2-captioner",
            mmif,
            {"config": config}
        )

        if result.success:
            # Extract captions
            captions = []
            for view in result.output_mmif.get('views', []):
                for ann in view.get('annotations', []):
                    props = ann.get('properties', {})
                    if props.get('text') or props.get('caption'):
                        captions.append({
                            "text": props.get('text') or props.get('caption'),
                            "timestamp_ms": props.get('start') or props.get('timestamp'),
                        })

            return json.dumps({
                "success": True,
                "app": "smolvlm2-captioner",
                "backend": result.backend,
                "captions_generated": len(captions),
                "captions": captions[:20],
                "execution_time": result.execution_time_seconds
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": result.error
            })

    @tool("run_clams_pipeline")
    def run_clams_pipeline(video_path: str, apps: str) -> str:
        """Run a pipeline of CLAMS apps on a video.

        Executes multiple CLAMS apps in sequence, passing output from one
        to the next. Use get_available_clams_apps to see what's available.

        Args:
            video_path: Path to the video file
            apps: Comma-separated list of app names (e.g., "swt-detection,doctr-wrapper")

        Returns:
            JSON string with combined pipeline results
        """
        app_list = [a.strip() for a in apps.split(',')]
        results = executor.execute_pipeline(app_list, video_path)

        summary = []
        for r in results:
            summary.append({
                "app": r.app_name,
                "backend": r.backend,
                "success": r.success,
                "annotations_created": r.annotations_created,
                "execution_time": r.execution_time_seconds,
                "error": r.error if not r.success else None
            })

        return json.dumps({
            "pipeline": app_list,
            "results": summary,
            "total_annotations": sum(r.annotations_created for r in results if r.success)
        }, indent=2)

    @tool("get_available_clams_apps")
    def get_available_clams_apps() -> str:
        """Get list of available CLAMS apps and their execution backends.

        Returns:
            JSON string with available apps and their backends (cli, docker, http)
        """
        apps = executor.get_available_apps()
        return json.dumps({
            "apps": apps,
            "total": len(apps)
        }, indent=2)

    # Always include generic tools
    tools = [run_clams_pipeline, get_available_clams_apps]

    # GPU-requiring execution tools only when GPU is available
    if gpu_available:
        tools.extend([execute_swt_detection, execute_doctr_ocr, execute_smolvlm2_captioner])
    else:
        logger.info("GPU not available - GPU-requiring execution tools (swt, doctr, smolvlm2) disabled")

    return tools
