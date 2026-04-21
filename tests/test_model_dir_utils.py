from __future__ import annotations

from pathlib import Path

from ocr2md.backend.base import BaseBackend
from ocr2md.config import LayoutConfig
from ocr2md.layout.layout_detector import PPDocLayoutDetector


class _DummyBackend(BaseBackend):
    name = "GLM-OCR"
    model_dir_hints = ("glm-ocr", "glm", "ocr")
    default_model_subdir = "glm-ocr"

    def _process_inputs(self, messages, task_type=None):
        return {}


def test_backend_resolves_shared_weights_root(tmp_path: Path) -> None:
    weights_root = tmp_path / "weights"
    glm_dir = weights_root / "glm-ocr"
    glm_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tokenizer.json"):
        (glm_dir / filename).write_text("ok", encoding="utf-8")

    backend = _DummyBackend()
    backend.model_dir = str(weights_root)
    backend.required_model_files = frozenset({"config.json", "tokenizer.json"})

    assert backend._resolve_model_dir() == glm_dir.resolve()


def test_layout_detector_resolves_shared_weights_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    weights_root = tmp_path / "weights"
    layout_dir = weights_root / "pp-doclayout-v3"
    layout_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (layout_dir / filename).write_text("ok", encoding="utf-8")

    called: dict[str, str] = {}

    class _FakeProcessor:
        @staticmethod
        def from_pretrained(model_dir: str):
            called["processor"] = model_dir
            return object()

    class _FakeModel:
        @staticmethod
        def from_pretrained(model_dir: str):
            called["model"] = model_dir
            return _FakeLoadedModel()

    class _FakeLoadedModel:
        def eval(self):
            return self

        def to(self, _device):
            return self

    monkeypatch.setattr(
        "ocr2md.layout.layout_detector.PPDocLayoutV3ImageProcessorFast",
        _FakeProcessor,
    )
    monkeypatch.setattr(
        "ocr2md.layout.layout_detector.PPDocLayoutV3ForObjectDetection",
        _FakeModel,
    )

    detector = PPDocLayoutDetector(LayoutConfig(model_dir=str(weights_root)))
    detector.start()

    assert called["processor"] == str(layout_dir.resolve())
    assert called["model"] == str(layout_dir.resolve())
