"""HTTP service registry for CLAMS tool endpoints."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = "data/tool_services.json"


@dataclass
class ToolService:
    tool_id: str
    app_name: str
    endpoint_url: str
    image: str = ""
    container_name: str = ""
    method: str = "POST"
    request_path: str = "/"
    status_path: str = "/"
    enabled: bool = True
    default_params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def request_url(self) -> str:
        return _join_url(self.endpoint_url, self.request_path)

    @property
    def status_url(self) -> str:
        return _join_url(self.endpoint_url, self.status_path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ServiceCallResult:
    ok: bool
    tool_id: str
    url: str
    status_code: int | None = None
    response_json: Any = None
    response_text: str = ""
    error: str = ""


class ServiceRegistry:
    def __init__(self, services: list[ToolService]):
        self.services = {service.tool_id: service for service in services}

    @classmethod
    def from_env(cls) -> "ServiceRegistry":
        path = service_config_path()
        if path.exists():
            return cls.from_path(path)
        return cls(default_services())

    @classmethod
    def from_path(cls, path: Path | str) -> "ServiceRegistry":
        with Path(path).open() as f:
            raw = json.load(f)
        services = raw.get("services", raw if isinstance(raw, list) else [])
        return cls([ToolService(**item) for item in services])

    def for_tool(self, tool_id: str) -> ToolService | None:
        return self.services.get(tool_id)

    def list_services(self) -> list[ToolService]:
        return sorted(self.services.values(), key=lambda s: s.tool_id)

    def health(self, service: ToolService, timeout: float = 2.0) -> ServiceCallResult:
        try:
            req = Request(service.status_url, method="GET")
            with urlopen(req, timeout=timeout) as response:
                body = response.read(1000).decode("utf-8", errors="replace")
                return ServiceCallResult(
                    ok=200 <= response.status < 300,
                    tool_id=service.tool_id,
                    url=service.status_url,
                    status_code=response.status,
                    response_text=body,
                )
        except (OSError, URLError) as exc:
            return ServiceCallResult(
                ok=False,
                tool_id=service.tool_id,
                url=service.status_url,
                error=str(exc),
            )

    def invoke_mmif(
        self,
        tool_id: str,
        input_mmif: dict,
        params: dict[str, Any] | None = None,
        timeout: float = 900.0,
    ) -> ServiceCallResult:
        service = self.for_tool(tool_id)
        if service is None:
            return ServiceCallResult(
                ok=False,
                tool_id=tool_id,
                url="",
                error=f"No service configured for {tool_id}",
            )
        if not service.enabled:
            return ServiceCallResult(
                ok=False,
                tool_id=tool_id,
                url=service.request_url,
                error=f"Service is disabled for {tool_id}",
            )

        call_params = dict(service.default_params)
        if params:
            call_params.update(params)
        url = service.request_url
        if call_params:
            url = f"{url}?{urlencode(call_params, doseq=True)}"

        try:
            payload = json.dumps(input_mmif).encode("utf-8")
            req = Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method=service.method.upper(),
            )
            with urlopen(req, timeout=timeout) as response:
                status = response.status
                body = response.read().decode("utf-8", errors="replace")
            parsed = None
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                pass
            return ServiceCallResult(
                ok=200 <= status < 300,
                tool_id=tool_id,
                url=url,
                status_code=status,
                response_json=parsed,
                response_text=body[:2000],
            )
        except (OSError, URLError) as exc:
            return ServiceCallResult(
                ok=False,
                tool_id=tool_id,
                url=url,
                error=str(exc),
            )


def service_config_path() -> Path:
    raw = os.environ.get("CLAMS_HAYSTACK_TOOL_SERVICES", DEFAULT_CONFIG_PATH)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def default_services() -> list[ToolService]:
    return [
        ToolService(
            tool_id="swt_detection",
            app_name="swt-detection",
            endpoint_url="http://127.0.0.1:5561/",
            image="ghcr.io/clamsproject/app-swt-detection:v8.3",
            container_name="clams_swt_detection",
            notes="Historical Aristotle pipeline port for SWT.",
        ),
        ToolService(
            tool_id="asr_whisper",
            app_name="whisper-wrapper",
            endpoint_url="http://127.0.0.1:5562/",
            image="ghcr.io/clamsproject/app-whisper-wrapper:v15",
            container_name="clams_whisper_wrapper",
        ),
        ToolService(
            tool_id="smolvlm2_captioner",
            app_name="smolvlm2-captioner",
            endpoint_url="http://127.0.0.1:5563/",
            image="ghcr.io/clamsproject/app-smolvlm2-captioner:latest",
            container_name="clams_smolvlm2_captioner",
        ),
        ToolService(
            tool_id="vlm_ocr_kie",
            app_name="qwen-ocr",
            endpoint_url="http://127.0.0.1:5564/",
            image="localhost/app-qwen-ocr:test",
            container_name="clams_qwen_ocr",
            notes="Qwen OCR image exists on Aristotle; use config for KIE prompts.",
        ),
        ToolService(
            tool_id="credits_ocr",
            app_name="qwen-ocr",
            endpoint_url="http://127.0.0.1:5564/",
            image="localhost/app-qwen-ocr:test",
            container_name="clams_qwen_ocr",
            notes="Shares the Qwen OCR endpoint with a credits-specific prompt config.",
        ),
        ToolService(
            tool_id="scene_summary",
            app_name="smolvlm2-captioner",
            endpoint_url="http://127.0.0.1:5563/",
            image="ghcr.io/clamsproject/app-smolvlm2-captioner:latest",
            container_name="clams_smolvlm2_captioner",
            notes="Initial scene-summary service uses the captioner endpoint.",
        ),
    ]


def _join_url(base: str, path: str) -> str:
    if not path or path == "/":
        return base.rstrip("/") + "/"
    return base.rstrip("/") + "/" + path.lstrip("/")
