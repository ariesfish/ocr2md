from __future__ import annotations

from ocr2md import load_config
from ocr2md.web.services.model_manager import ModelManager
from ocr2md.web.stores.task_store import TaskStore


def test_build_download_command_modelscope() -> None:
    cfg = load_config()
    store = TaskStore()
    manager = ModelManager(cfg, store)
    command = manager.build_download_command("glm", "modelscope")
    assert command[0] == "modelscope"
    assert "--model" in command
    assert "--local_dir" in command


def test_build_download_command_huggingface() -> None:
    cfg = load_config()
    store = TaskStore()
    manager = ModelManager(cfg, store)
    command = manager.build_download_command("glm", "huggingface")
    assert command[0] == "hf"
    assert "--local-dir" in command


def test_create_download_task_can_switch_to_ready_model_dir(tmp_path) -> None:
    cfg = load_config()
    store = TaskStore()
    manager = ModelManager(cfg, store)
    backend_config = cfg.pipeline.glm_ocr_backend
    assert backend_config is not None
    weights_root = tmp_path / "weights"
    ready_dir = weights_root / "glm-ocr"
    layout_dir = weights_root / "pp-doclayout-v3"
    ready_dir.mkdir(parents=True, exist_ok=True)
    layout_dir.mkdir(parents=True, exist_ok=True)

    for filename in backend_config.required_model_files:
        (ready_dir / filename).write_text("ok", encoding="utf-8")
    for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (layout_dir / filename).write_text("ok", encoding="utf-8")

    task_id = manager.create_download_task(
        "glm",
        request_id="req_ready_dir",
        model_dir=str(weights_root),
    )

    snapshot = store.get_task_snapshot(task_id)
    assert snapshot["status"] == "succeeded"
    assert snapshot["result"]["model_dir"] == str(ready_dir.resolve())
    assert backend_config.model_dir == str(ready_dir.resolve())
    assert cfg.pipeline.layout.model_dir == str(layout_dir.resolve())


def test_set_model_dir_reports_ready_for_complete_root(tmp_path) -> None:
    cfg = load_config()
    store = TaskStore()
    manager = ModelManager(cfg, store)
    backend_config = cfg.pipeline.glm_ocr_backend
    assert backend_config is not None

    weights_root = tmp_path / "weights"
    ready_dir = weights_root / "glm-ocr"
    layout_dir = weights_root / "pp-doclayout-v3"
    ready_dir.mkdir(parents=True, exist_ok=True)
    layout_dir.mkdir(parents=True, exist_ok=True)

    for filename in backend_config.required_model_files:
        (ready_dir / filename).write_text("ok", encoding="utf-8")
    for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (layout_dir / filename).write_text("ok", encoding="utf-8")

    status = manager.set_model_dir("glm", str(weights_root))

    assert status["status"] == "ready"
    assert status["progress"] == 100
    assert status["model_root_dir"] == str(weights_root.resolve())
    assert backend_config.model_dir == str(ready_dir.resolve())
    assert cfg.pipeline.layout.model_dir == str(layout_dir.resolve())


def test_set_model_dir_reports_not_ready_when_layout_files_missing(tmp_path) -> None:
    cfg = load_config()
    store = TaskStore()
    manager = ModelManager(cfg, store)
    backend_config = cfg.pipeline.glm_ocr_backend
    assert backend_config is not None

    weights_root = tmp_path / "weights"
    ready_dir = weights_root / "glm-ocr"
    layout_dir = weights_root / "pp-doclayout-v3"
    ready_dir.mkdir(parents=True, exist_ok=True)
    layout_dir.mkdir(parents=True, exist_ok=True)

    for filename in backend_config.required_model_files:
        (ready_dir / filename).write_text("ok", encoding="utf-8")

    status = manager.set_model_dir("glm", str(weights_root))

    assert status["status"] == "not_ready"
    assert "layout model directory is incomplete" in str(status["message"])
    assert str(layout_dir.resolve()) in str(status["message"])


def test_get_models_status_reports_missing_layout_directory_with_required_files(
    tmp_path,
) -> None:
    cfg = load_config()
    store = TaskStore()
    manager = ModelManager(cfg, store)
    backend_config = cfg.pipeline.glm_ocr_backend
    assert backend_config is not None

    ready_dir = tmp_path / "glm-ocr"
    ready_dir.mkdir(parents=True, exist_ok=True)
    missing_layout_dir = tmp_path / "pp-doclayout-v3"

    for filename in backend_config.required_model_files:
        (ready_dir / filename).write_text("ok", encoding="utf-8")

    backend_config.model_dir = str(ready_dir.resolve())
    cfg.pipeline.layout.model_dir = str(missing_layout_dir.resolve())

    status = manager.get_models_status()["glm"]

    assert status["status"] == "not_ready"
    assert "layout model directory is missing" in str(status["message"])
    assert str(missing_layout_dir.resolve()) in str(status["message"])
    assert status["model_root_dir"] == str(tmp_path.resolve())


def test_layout_download_target_uses_modelscope_by_default(tmp_path) -> None:
    cfg = load_config()
    store = TaskStore()
    manager = ModelManager(cfg, store)

    weights_root = tmp_path / "weights"
    cfg.pipeline.layout.model_dir = str((weights_root / "pp-doclayout-v3").resolve())

    target = manager._layout_download_target()

    assert target.name == "layout"
    assert target.download_source == "modelscope"
    assert target.modelscope_repo_id == "PaddlePaddle/PP-DocLayoutV3_safetensors"
    assert target.huggingface_repo_id == "PaddlePaddle/PP-DocLayoutV3_safetensors"
    assert target.model_dir == (weights_root / "pp-doclayout-v3").resolve()
    assert target.required_files == [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    ]


def test_resolve_download_source_honors_requested_source() -> None:
    assert ModelManager._resolve_download_source("modelscope", "huggingface") == "modelscope"
    assert ModelManager._resolve_download_source("huggingface", "modelscope") == "huggingface"
    assert ModelManager._resolve_download_source("", "modelscope") == "modelscope"
