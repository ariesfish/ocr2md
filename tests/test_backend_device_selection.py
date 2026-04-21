from __future__ import annotations

from types import SimpleNamespace

from ocr2md.backend.base import BaseBackend
from ocr2md import load_config
from ocr2md.config import GLMOCRBackendConfig


class _DummyBackend(BaseBackend):
    def _process_inputs(self, messages, task_type=None):
        return {}


def _build_backend() -> _DummyBackend:
    backend = _DummyBackend()
    backend._torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: True)
        ),
    )
    return backend


def test_glm_backend_defaults_to_auto_device() -> None:
    config = GLMOCRBackendConfig(required_model_files=["config.json"])
    assert config.device == "auto"


def test_default_yaml_uses_auto_device() -> None:
    cfg = load_config()
    assert cfg.pipeline.glm_ocr_backend is not None
    assert cfg.pipeline.glm_ocr_backend.device == "auto"


def test_auto_device_prefers_mps_when_cuda_is_unavailable() -> None:
    backend = _build_backend()
    assert backend._resolve_device_strategy() == "mps"
