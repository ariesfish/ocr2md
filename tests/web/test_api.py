from __future__ import annotations

import base64
import importlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from ocr2md.config import Config
from ocr2md.web.errors import OCRAPIError
from ocr2md.web.stores.task_store import TaskStore


def _fake_result() -> dict:
    return {
        "request_id": "ocr2md_test_req",
        "input": {"filename": "demo.png", "width": 1024, "height": 768},
        "glm": {
            "name": "GLM-OCR",
            "latency_ms": 120,
            "confidence_avg": 0.85,
            "text": "hello glm",
            "boxes": [
                {
                    "index": 0,
                    "label": "text",
                    "score": 0.85,
                    "bbox_2d": [0, 0, 100, 100],
                    "polygon": None,
                }
            ],
            "error": None,
        },
        "manifest": None,
    }


class FakeRunner:
    def __init__(self):
        self.result = _fake_result()
        self.stop_calls = 0
        self.last_selected_pages: list[int] | None = None

    def run_ocr(self, **kwargs: dict) -> dict:
        self.last_selected_pages = kwargs.get("selected_pages")
        return self.result

    def stop(self) -> None:
        self.stop_calls += 1
        return None


class FakeQueue:
    def __init__(self, task_store: TaskStore, runner: FakeRunner):
        self.task_store = task_store
        self.runner = runner

    def submit(
        self,
        *,
        task_id: str,
        request_id: str,
        filename: str,
        file_bytes: bytes,
        selected_pages: list[int] | None = None,
    ) -> None:
        self.task_store.update_task(
            task_id,
            status="running",
            progress=40,
            stage="queued",
            message=f"received {filename}:{len(file_bytes)}",
        )
        self.task_store.mark_succeeded(
            task_id,
            result=self.runner.run_ocr(selected_pages=selected_pages),
            message="done",
        )

    def stop(self) -> None:
        return None


class FakeModelManager:
    def __init__(self, task_store: TaskStore):
        self.task_store = task_store
        self.ready = True
        self.last_download_model_dir: str | None = None
        self.last_set_model_dir: str | None = None
        self.status_payload = {
            "glm": {"name": "GLM-OCR", "status": "ready", "progress": 100}
        }

    def get_models_status(self) -> dict:
        return self.status_payload

    def assert_models_ready(self) -> None:
        if not self.ready:
            raise OCRAPIError(
                status_code=409,
                code="MODEL_NOT_READY",
                message="glm: not_ready",
            )

    def create_download_task(
        self,
        model_key: str,
        request_id: str,
        source: str | None = None,
        model_dir: str | None = None,
    ) -> str:
        self.last_download_model_dir = model_dir
        task = self.task_store.create_task(
            task_type="download",
            request_id=request_id,
            model_key=model_key,
        )
        self.task_store.update_task(
            task.task_id,
            status="running",
            progress=50,
            stage="downloading",
            message=source or "modelscope",
        )
        return task.task_id

    def set_model_dir(self, model_key: str, model_dir: str) -> dict:
        self.last_set_model_dir = model_dir
        self.status_payload[model_key] = {
            "name": "GLM-OCR",
            "status": "ready",
            "progress": 100,
            "message": None,
        }
        return self.status_payload[model_key]


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[TestClient, SimpleNamespace], None, None]:
    web_app_module = importlib.import_module("ocr2md.web.app")
    cfg = Config()
    task_store = TaskStore(ttl_seconds=1800)
    runner = FakeRunner()
    model_manager = FakeModelManager(task_store)
    queue = FakeQueue(task_store, runner)
    services = SimpleNamespace(
        config=cfg,
        task_store=task_store,
        ocr_runner=runner,
        model_manager=model_manager,
        ocr_queue=queue,
    )

    monkeypatch.setattr(web_app_module, "build_services", lambda config: services)
    app = web_app_module.create_app(cfg)
    with TestClient(app) as test_client:
        yield test_client, services


def test_models_status_ready(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.get("/api/models/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["models"]["glm"]["status"] == "ready"


def test_create_ocr_job_success(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/ocr/jobs",
        files={"file": ("demo.png", b"png-bytes", "image/png")},
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    task_response = test_client.get(f"/api/tasks/{task_id}")
    assert task_response.status_code == 200
    task_payload = task_response.json()
    assert task_payload["status"] == "succeeded"
    assert task_payload["result"]["glm"]["name"] == "GLM-OCR"


def test_create_pdf_ocr_job_success(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/ocr/jobs",
        files={"file": ("demo.pdf", b"%PDF-1.4\n", "application/pdf")},
    )
    assert response.status_code == 200


def test_create_pdf_ocr_job_accepts_page_selection(
    client: tuple[TestClient, SimpleNamespace],
) -> None:
    test_client, services = client
    response = test_client.post(
        "/api/ocr/jobs",
        data={"page_selection": "1,3-4"},
        files={"file": ("demo.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    assert response.status_code == 200
    assert services.ocr_runner.last_selected_pages == [0, 2, 3]


def test_page_selection_rejected_for_non_pdf(
    client: tuple[TestClient, SimpleNamespace],
) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/ocr/jobs",
        data={"page_selection": "1-2"},
        files={"file": ("demo.png", b"png-bytes", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_PAGE_SELECTION"


def test_invalid_page_selection_rejected(
    client: tuple[TestClient, SimpleNamespace],
) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/ocr/jobs",
        data={"page_selection": "3-1"},
        files={"file": ("demo.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_PAGE_SELECTION"


def test_create_ocr_job_model_not_ready(
    client: tuple[TestClient, SimpleNamespace],
) -> None:
    test_client, services = client
    services.model_manager.ready = False
    response = test_client.post(
        "/api/ocr/jobs",
        files={"file": ("demo.png", b"png-bytes", "image/png")},
    )
    assert response.status_code == 409
    payload = response.json()
    assert payload["code"] == "MODEL_NOT_READY"
    assert "request_id" in payload


def test_invalid_upload_rejected(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/ocr/jobs",
        files={"file": ("demo.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "UNSUPPORTED_FILE_TYPE"


def test_missing_task_returns_410(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.get("/api/tasks/not_exists")
    assert response.status_code == 410
    payload = response.json()
    assert payload["code"] in {"TASK_NOT_FOUND", "TASK_EXPIRED"}


def test_unknown_api_route_uses_unified_error(
    client: tuple[TestClient, SimpleNamespace],
) -> None:
    test_client, _ = client
    response = test_client.get("/api/not-found")
    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "NOT_FOUND"
    assert payload["message"]
    assert "request_id" in payload


def test_validation_error_uses_unified_error(
    client: tuple[TestClient, SimpleNamespace],
) -> None:
    test_client, _ = client
    response = test_client.post("/api/ocr/jobs")
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "INVALID_REQUEST"
    assert payload["message"]
    assert "request_id" in payload


def test_sync_ocr_schema(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/ocr/run",
        files={"file": ("demo.png", b"png-bytes", "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["glm"]["boxes"][0]["score"] == pytest.approx(0.85)
    assert payload["glm"]["text"] == "hello glm"


def test_sync_pdf_ocr_schema(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/ocr/run",
        files={"file": ("demo.pdf", b"%PDF-1.4\n", "application/pdf")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["glm"]["name"] == "GLM-OCR"


def test_download_task_created(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/models/glm/download", json={"source": "modelscope"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_type"] == "download"
    task_id = payload["task_id"]

    task_response = test_client.get(f"/api/tasks/{task_id}")
    assert task_response.status_code == 200
    assert task_response.json()["status"] == "running"


def test_download_task_accepts_model_dir(
    client: tuple[TestClient, SimpleNamespace],
) -> None:
    test_client, services = client
    response = test_client.post(
        "/api/models/glm/download",
        json={"source": "modelscope", "model_dir": "/tmp/custom-glm"},
    )
    assert response.status_code == 200
    assert services.model_manager.last_download_model_dir == "/tmp/custom-glm"


def test_set_model_dir_returns_status_and_stops_runner(
    client: tuple[TestClient, SimpleNamespace],
) -> None:
    test_client, services = client
    response = test_client.post(
        "/api/models/glm/set-dir",
        json={"model_dir": "/tmp/custom-glm"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert services.model_manager.last_set_model_dir == "/tmp/custom-glm"
    assert services.ocr_runner.stop_calls == 1


def test_get_task_layout_preview(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, services = client
    task = services.task_store.create_task(task_type="ocr", request_id="req_layout")

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z8ioAAAAASUVORK5CYII="
    )

    with tempfile.TemporaryDirectory(prefix="ocr2md_layout_test_") as tmp_dir:
        root = Path(tmp_dir)
        layout_file = root / "layout_page0.png"
        layout_file.write_bytes(png_bytes)

        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps({"layout": {"files": [str(layout_file)]}}),
            encoding="utf-8",
        )

        services.task_store.mark_succeeded(
            task.task_id,
            result={"manifest": str(manifest)},
            message="done",
        )

        response = test_client.get(f"/api/tasks/{task.task_id}/layout?page=0")

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("image/")


def test_get_task_layout_preview_from_live_workspace(
    client: tuple[TestClient, SimpleNamespace],
) -> None:
    test_client, services = client
    task = services.task_store.create_task(task_type="ocr", request_id="req_layout_live")

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z8ioAAAAASUVORK5CYII="
    )

    with tempfile.TemporaryDirectory(prefix="ocr2md_layout_live_") as tmp_dir:
        services.config.pipeline.output.base_output_dir = tmp_dir
        task_dir = Path(tmp_dir) / task.task_id / "layout"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "layout_page0.png").write_bytes(png_bytes)

        response = test_client.get(f"/api/tasks/{task.task_id}/layout?page=0")

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("image/")


def test_get_task_crop_preview(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, services = client
    task = services.task_store.create_task(task_type="ocr", request_id="req_crop")

    with tempfile.TemporaryDirectory(prefix="ocr2md_crop_test_") as tmp_dir:
        services.config.pipeline.output.base_output_dir = tmp_dir
        input_dir = Path(tmp_dir) / task.task_id / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        image_path = input_dir / "demo.png"
        Image.new("RGB", (10, 10), color="white").save(image_path)

        response = test_client.get(
            f"/api/tasks/{task.task_id}/crop?page=0&bbox=0,0,1000,1000"
        )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("image/")
