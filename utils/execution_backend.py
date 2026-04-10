"""
Execution backends for CLAMS apps.
Supports CLI (local subprocess), Docker (containerized), and HTTP (API) execution.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import json
import tempfile
import time
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Unified result from any execution backend."""
    success: bool
    app_name: str
    backend: str  # "cli", "docker", or "http"
    output_mmif: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_seconds: float = 0.0
    annotations_created: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "app_name": self.app_name,
            "backend": self.backend,
            "output_mmif": self.output_mmif,
            "error": self.error,
            "execution_time_seconds": self.execution_time_seconds,
            "annotations_created": self.annotations_created,
            "metadata": self.metadata
        }


class BaseExecutor(ABC):
    """Abstract base class for execution backends."""

    @abstractmethod
    def execute(
        self,
        app_name: str,
        input_mmif: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        timeout: int = 300
    ) -> ExecutionResult:
        """
        Execute a CLAMS app and return results.

        Args:
            app_name: Name of the CLAMS app to execute
            input_mmif: Input MMIF document as dictionary
            parameters: Optional app-specific parameters
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with output MMIF or error details
        """
        pass

    @abstractmethod
    def is_available(self, app_name: str) -> bool:
        """Check if app is available in this backend."""
        pass

    @abstractmethod
    def get_available_apps(self) -> List[str]:
        """Get list of available app names."""
        pass

    def get_app_info(self, app_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific app. Optional to implement."""
        return None

    @staticmethod
    def count_annotations(mmif: Dict[str, Any]) -> int:
        """Count annotations in MMIF output."""
        count = 0
        for view in mmif.get('views', []):
            count += len(view.get('annotations', []))
        return count


class CLIExecutor(BaseExecutor):
    """Execute CLAMS apps via local CLI interface (subprocess)."""

    def __init__(self, apps_dir: str):
        """
        Initialize CLI executor.

        Args:
            apps_dir: Directory containing CLAMS app directories
        """
        self.apps_dir = Path(apps_dir)
        self.available_apps: Dict[str, Path] = {}
        self._discover_apps()

    def _discover_apps(self):
        """Discover available CLAMS apps in the apps directory."""
        if not self.apps_dir.exists():
            logger.warning(f"Apps directory does not exist: {self.apps_dir}")
            return

        for item in self.apps_dir.iterdir():
            if item.is_dir():
                # Check for app-* naming pattern
                if item.name.startswith('app-'):
                    cli_path = item / 'cli.py'
                    app_path = item / 'app.py'

                    if cli_path.exists() or app_path.exists():
                        app_name = item.name.replace('app-', '')
                        self.available_apps[app_name] = item
                        logger.debug(f"Discovered CLI app: {app_name} at {item}")

        logger.info(f"CLIExecutor discovered {len(self.available_apps)} apps")

    def _find_python(self, app_path: Path) -> str:
        """Find Python interpreter for an app (venv or system)."""
        for venv_dir in ['.venv', 'venv', '.venv-py311', '.venv-test']:
            venv_python = app_path / venv_dir / 'bin' / 'python3'
            if venv_python.exists():
                return str(venv_python)
            venv_python = app_path / venv_dir / 'bin' / 'python'
            if venv_python.exists():
                return str(venv_python)
        return 'python3'

    def execute(
        self,
        app_name: str,
        input_mmif: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        timeout: int = 300
    ) -> ExecutionResult:
        """Execute app via CLI subprocess."""
        start_time = time.perf_counter()

        if app_name not in self.available_apps:
            return ExecutionResult(
                success=False,
                app_name=app_name,
                backend="cli",
                error=f"App not found: {app_name}. Available: {list(self.available_apps.keys())}"
            )

        app_path = self.available_apps[app_name]
        input_file = None
        output_file = None

        try:
            # Write input MMIF to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(input_mmif, f)
                input_file = f.name

            # Prepare output file
            output_file = tempfile.mktemp(suffix='.json')

            # Determine which script to use
            cli_script = app_path / 'cli.py'
            app_script = app_path / 'app.py'

            if cli_script.exists():
                script_path = cli_script
            elif app_script.exists():
                script_path = app_script
            else:
                return ExecutionResult(
                    success=False,
                    app_name=app_name,
                    backend="cli",
                    error=f"No cli.py or app.py found in {app_path}"
                )

            # Find Python interpreter
            python_path = self._find_python(app_path)

            # Build command
            cmd = [python_path, str(script_path)]

            # Add parameters
            if parameters:
                for key, value in parameters.items():
                    if isinstance(value, bool):
                        if value:
                            cmd.append(f"--{key}")
                    elif isinstance(value, list):
                        for v in value:
                            cmd.extend([f"--{key}", str(v)])
                    else:
                        cmd.extend([f"--{key}", str(value)])

            # Add input/output files
            cmd.extend([input_file, output_file])

            logger.info(f"Executing CLI: {' '.join(cmd[:5])}...")  # Truncate for logging

            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(app_path),
                timeout=timeout
            )

            execution_time = time.perf_counter() - start_time

            if result.returncode != 0:
                error_msg = result.stderr[:1000] if result.stderr else "Unknown error"
                logger.error(f"CLI execution failed for {app_name}: {error_msg}")
                return ExecutionResult(
                    success=False,
                    app_name=app_name,
                    backend="cli",
                    error=f"Exit code {result.returncode}: {error_msg}",
                    execution_time_seconds=execution_time
                )

            # Read output MMIF
            output_mmif = None
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                with open(output_file, 'r') as f:
                    output_mmif = json.load(f)
            elif result.stdout:
                # Try parsing stdout as MMIF
                try:
                    output_mmif = json.loads(result.stdout)
                except json.JSONDecodeError:
                    pass

            if output_mmif is None:
                return ExecutionResult(
                    success=False,
                    app_name=app_name,
                    backend="cli",
                    error="No valid MMIF output produced",
                    execution_time_seconds=execution_time
                )

            return ExecutionResult(
                success=True,
                app_name=app_name,
                backend="cli",
                output_mmif=output_mmif,
                execution_time_seconds=execution_time,
                annotations_created=self.count_annotations(output_mmif)
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                app_name=app_name,
                backend="cli",
                error=f"Execution timed out after {timeout} seconds",
                execution_time_seconds=timeout
            )
        except Exception as e:
            logger.error(f"CLI execution error for {app_name}: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                app_name=app_name,
                backend="cli",
                error=str(e),
                execution_time_seconds=time.perf_counter() - start_time
            )
        finally:
            # Cleanup temp files
            for f in [input_file, output_file]:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

    def is_available(self, app_name: str) -> bool:
        return app_name in self.available_apps

    def get_available_apps(self) -> List[str]:
        return list(self.available_apps.keys())

    def get_app_info(self, app_name: str) -> Optional[Dict[str, Any]]:
        """Get app metadata by running metadata.py."""
        if app_name not in self.available_apps:
            return None

        app_path = self.available_apps[app_name]
        metadata_script = app_path / 'metadata.py'

        if not metadata_script.exists():
            return None

        try:
            python_path = self._find_python(app_path)
            result = subprocess.run(
                [python_path, str(metadata_script)],
                capture_output=True,
                text=True,
                cwd=str(app_path),
                timeout=30
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            logger.warning(f"Failed to get metadata for {app_name}: {e}")

        return None


class DockerExecutor(BaseExecutor):
    """Execute CLAMS apps via Docker containers."""

    def __init__(
        self,
        registry: str = "ghcr.io/clamsproject",
        cache_dir: str = "/cache",
        gpu_enabled: bool = True,
        gpu_device: str = "0",
        network: str = "host"
    ):
        """
        Initialize Docker executor.

        Args:
            registry: Docker registry prefix for images
            cache_dir: Host directory to mount as /cache
            gpu_enabled: Whether to enable GPU support
            gpu_device: GPU device ID to use (default: "0" for single GPU)
            network: Docker network mode
        """
        self.registry = registry
        self.cache_dir = Path(cache_dir)
        self.gpu_enabled = gpu_enabled
        self.gpu_device = gpu_device
        self.network = network
        self.available_images: Dict[str, str] = {}
        self._docker_available = self._check_docker()

        if self._docker_available:
            self._discover_images()

    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            logger.warning("Docker is not available")
            return False

    def _is_podman(self) -> bool:
        """Check if 'docker' is actually podman."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True, text=True, timeout=5
            )
            return "podman" in result.stdout.lower()
        except Exception:
            return False

    def _discover_images(self):
        """Discover available CLAMS Docker images."""
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    # Parse image name to app name
                    # e.g., ghcr.io/clamsproject/app-swt-detection:1.0.0
                    if self.registry in line or 'app-' in line:
                        parts = line.split('/')
                        image_name = parts[-1] if parts else line
                        app_part = image_name.split(':')[0]
                        if app_part.startswith('app-'):
                            app_name = app_part.replace('app-', '')
                            self.available_images[app_name] = line
                            logger.debug(f"Discovered Docker image: {app_name} -> {line}")

            logger.info(f"DockerExecutor discovered {len(self.available_images)} images")
        except Exception as e:
            logger.warning(f"Could not discover Docker images: {e}")

    def execute(
        self,
        app_name: str,
        input_mmif: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        timeout: int = 300
    ) -> ExecutionResult:
        """Execute app in Docker container."""
        start_time = time.perf_counter()

        if not self._docker_available:
            return ExecutionResult(
                success=False,
                app_name=app_name,
                backend="docker",
                error="Docker is not available"
            )

        image = self.available_images.get(app_name)
        if not image:
            return ExecutionResult(
                success=False,
                app_name=app_name,
                backend="docker",
                error=f"Docker image not found for {app_name}. Available: {list(self.available_images.keys())}"
            )

        input_file = None
        output_dir = None

        try:
            # Create temp directory for I/O
            output_dir = tempfile.mkdtemp()
            input_file = os.path.join(output_dir, 'input.mmif')
            output_file = os.path.join(output_dir, 'output.mmif')

            # Write input MMIF
            with open(input_file, 'w') as f:
                json.dump(input_mmif, f)

            # Build docker command (-i for stdin piping)
            cmd = ["docker", "run", "--rm", "-i"]

            # Volume mounts
            cmd.extend(["-v", f"{output_dir}:/data"])

            # Mount cache directory if it exists
            if self.cache_dir.exists():
                cmd.extend(["-v", f"{self.cache_dir}:/cache"])

            # Mount directories containing media files referenced in MMIF
            mounted_dirs = set()
            for doc in input_mmif.get("documents", []):
                loc = doc.get("properties", {}).get("location", "")
                if loc.startswith("file://"):
                    media_path = loc[7:]
                    media_dir = str(Path(media_path).parent)
                    if media_dir not in mounted_dirs:
                        cmd.extend(["-v", f"{media_dir}:{media_dir}:ro"])
                        mounted_dirs.add(media_dir)

            # GPU support (single device to avoid hogging shared resources)
            # Podman uses --device nvidia.com/gpu=N, Docker uses --gpus device=N
            if self.gpu_enabled:
                if self._is_podman():
                    cmd.extend(["--device", f"nvidia.com/gpu={self.gpu_device}"])
                else:
                    cmd.extend(["--gpus", f"device={self.gpu_device}"])

            # Network mode
            if self.network:
                cmd.extend(["--network", self.network])

            # Image
            cmd.append(image)

            # Command inside container: CLAMS apps read MMIF from stdin, write to stdout
            cmd.extend(["python3", "cli.py"])

            # Add parameters
            if parameters:
                for key, value in parameters.items():
                    if isinstance(value, bool):
                        if value:
                            cmd.append(f"--{key}")
                    else:
                        cmd.extend([f"--{key}", str(value)])

            logger.info(f"Executing Docker: {' '.join(cmd[:8])}...")

            # Read input MMIF to feed via stdin
            with open(input_file, 'r') as f:
                input_mmif_str = f.read()

            # Execute with MMIF piped via stdin/stdout
            result = subprocess.run(
                cmd,
                input=input_mmif_str,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            execution_time = time.perf_counter() - start_time

            if result.returncode != 0:
                error_msg = result.stderr[:1000] if result.stderr else "Unknown error"
                return ExecutionResult(
                    success=False,
                    app_name=app_name,
                    backend="docker",
                    error=f"Docker exit code {result.returncode}: {error_msg}",
                    execution_time_seconds=execution_time
                )

            # Parse output MMIF from stdout
            if result.stdout.strip():
                try:
                    output_mmif = json.loads(result.stdout)
                    return ExecutionResult(
                        success=True,
                        app_name=app_name,
                        backend="docker",
                        output_mmif=output_mmif,
                        execution_time_seconds=execution_time,
                        annotations_created=self.count_annotations(output_mmif)
                    )
                except json.JSONDecodeError as e:
                    return ExecutionResult(
                        success=False,
                        app_name=app_name,
                        backend="docker",
                        error=f"Invalid JSON output: {e}",
                        execution_time_seconds=execution_time
                    )
            else:
                return ExecutionResult(
                    success=False,
                    app_name=app_name,
                    backend="docker",
                    error="No output MMIF produced (empty stdout)",
                    execution_time_seconds=execution_time
                )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                app_name=app_name,
                backend="docker",
                error=f"Docker execution timed out after {timeout} seconds",
                execution_time_seconds=timeout
            )
        except Exception as e:
            logger.error(f"Docker execution error for {app_name}: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                app_name=app_name,
                backend="docker",
                error=str(e),
                execution_time_seconds=time.perf_counter() - start_time
            )
        finally:
            # Cleanup
            if output_dir and os.path.exists(output_dir):
                import shutil
                try:
                    shutil.rmtree(output_dir)
                except:
                    pass

    def is_available(self, app_name: str) -> bool:
        return self._docker_available and app_name in self.available_images

    def get_available_apps(self) -> List[str]:
        return list(self.available_images.keys())

    def pull_image(self, app_name: str, tag: str = "latest") -> bool:
        """Pull a Docker image for an app."""
        image = f"{self.registry}/app-{app_name}:{tag}"
        try:
            result = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                timeout=600  # 10 minutes for large images
            )
            if result.returncode == 0:
                self.available_images[app_name] = image
                return True
        except Exception as e:
            logger.error(f"Failed to pull image {image}: {e}")
        return False


class HTTPExecutor(BaseExecutor):
    """Execute CLAMS apps via HTTP API endpoints."""

    def __init__(self, endpoints: Optional[Dict[str, str]] = None):
        """
        Initialize HTTP executor.

        Args:
            endpoints: Dict mapping app_name to base URL
                       e.g., {"swt-detection": "http://localhost:5000"}
        """
        self.endpoints: Dict[str, str] = endpoints or {}
        self._health_cache: Dict[str, bool] = {}

    def add_endpoint(self, app_name: str, base_url: str):
        """Register an app endpoint."""
        self.endpoints[app_name] = base_url.rstrip('/')
        self._health_cache.pop(app_name, None)  # Clear health cache

    def remove_endpoint(self, app_name: str):
        """Remove an app endpoint."""
        self.endpoints.pop(app_name, None)
        self._health_cache.pop(app_name, None)

    def _check_health(self, app_name: str) -> bool:
        """Check if an endpoint is healthy."""
        if app_name not in self.endpoints:
            return False

        try:
            import requests
            response = requests.get(
                self.endpoints[app_name],
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def execute(
        self,
        app_name: str,
        input_mmif: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        timeout: int = 300
    ) -> ExecutionResult:
        """Execute app via HTTP POST to /annotate endpoint."""
        start_time = time.perf_counter()

        if app_name not in self.endpoints:
            return ExecutionResult(
                success=False,
                app_name=app_name,
                backend="http",
                error=f"No HTTP endpoint configured for {app_name}"
            )

        base_url = self.endpoints[app_name]

        try:
            import requests

            # Build URL with query parameters for CLAMS apps
            # CLAMS Restifier expects parameters as query strings
            url = base_url
            if parameters:
                query_parts = []
                for key, value in parameters.items():
                    if isinstance(value, list):
                        for v in value:
                            query_parts.append(f"{key}={v}")
                    elif isinstance(value, bool):
                        query_parts.append(f"{key}={'true' if value else 'false'}")
                    else:
                        query_parts.append(f"{key}={value}")
                if query_parts:
                    url += "?" + "&".join(query_parts)

            logger.info(f"Executing HTTP POST to {url[:50]}...")

            response = requests.post(
                url,
                json=input_mmif,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )

            execution_time = time.perf_counter() - start_time

            if response.status_code == 200:
                try:
                    output_mmif = response.json()
                    return ExecutionResult(
                        success=True,
                        app_name=app_name,
                        backend="http",
                        output_mmif=output_mmif,
                        execution_time_seconds=execution_time,
                        annotations_created=self.count_annotations(output_mmif)
                    )
                except json.JSONDecodeError:
                    return ExecutionResult(
                        success=False,
                        app_name=app_name,
                        backend="http",
                        error="Invalid JSON response",
                        execution_time_seconds=execution_time
                    )
            else:
                error_text = response.text[:500] if response.text else "No error message"
                return ExecutionResult(
                    success=False,
                    app_name=app_name,
                    backend="http",
                    error=f"HTTP {response.status_code}: {error_text}",
                    execution_time_seconds=execution_time
                )

        except ImportError:
            return ExecutionResult(
                success=False,
                app_name=app_name,
                backend="http",
                error="requests library not installed"
            )
        except Exception as e:
            logger.error(f"HTTP execution error for {app_name}: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                app_name=app_name,
                backend="http",
                error=str(e),
                execution_time_seconds=time.perf_counter() - start_time
            )

    def is_available(self, app_name: str) -> bool:
        """Check if endpoint is configured and healthy."""
        if app_name not in self.endpoints:
            return False

        # Use cached health status if available
        if app_name in self._health_cache:
            return self._health_cache[app_name]

        # Check health
        healthy = self._check_health(app_name)
        self._health_cache[app_name] = healthy
        return healthy

    def get_available_apps(self) -> List[str]:
        return list(self.endpoints.keys())

    def get_app_info(self, app_name: str) -> Optional[Dict[str, Any]]:
        """Get app metadata via HTTP GET."""
        if app_name not in self.endpoints:
            return None

        try:
            import requests
            response = requests.get(self.endpoints[app_name], timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass

        return None
