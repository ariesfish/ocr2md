from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ocr2md.web.services.ocr_runner import PipelineOCRRunner
from ocr2md.web.services.ocr_queue import OCRTaskQueue
from ocr2md.web.stores.task_store import TaskStore


class _FakeOCRRunner(PipelineOCRRunner):
    def __init__(self) -> None:
        return None

    def run_ocr(
        self,
        source_path: Path,
        request_id: str,
        task_id: Optional[str] = None,
        persist_artifacts: bool = True,
        progress_callback: Optional[Callable[[int, str, Optional[str]], None]] = None,
        model_result_callback: Optional[
            Callable[[str, Dict[str, Any], Dict[str, Any]], None]
        ] = None,
    ) -> Dict[str, Any]:
        del source_path, task_id, persist_artifacts

        input_meta = {"filename": "demo.png", "width": 1000, "height": 500}
        glm_result = {
            "name": "GLM-OCR",
            "latency_ms": 1200,
            "confidence_avg": 0.8,
            "text": "glm text",
            "boxes": [],
            "error": None,
        }

        if progress_callback is not None:
            progress_callback(80, "glm_inference", None)

        time.sleep(0.2)

        if model_result_callback is not None:
            model_result_callback("glm", glm_result, input_meta)

        time.sleep(0.2)

        return {
            "request_id": request_id,
            "input": input_meta,
            "glm": glm_result,
            "manifest": None,
        }


def test_ocr_queue_publishes_partial_model_result_before_completion() -> None:
    task_store = TaskStore(ttl_seconds=1800)
    queue = OCRTaskQueue(task_store=task_store, ocr_runner=_FakeOCRRunner())

    task = task_store.create_task(task_type="ocr", request_id="req_partial")
    queue.submit(
        task_id=task.task_id,
        request_id="req_partial",
        filename="demo.png",
        file_bytes=b"png-bytes",
    )

    try:
        seen_partial = False
        partial_deadline = time.time() + 2.0
        while time.time() < partial_deadline:
            snapshot = task_store.get_task_snapshot(task.task_id)
            result = snapshot.get("result") if isinstance(snapshot, dict) else None
            if (
                snapshot.get("status") == "running"
                and isinstance(result, dict)
                and "glm" in result
            ):
                seen_partial = True
                break
            time.sleep(0.02)

        assert seen_partial, "should expose glm result before task completes"

        finished_deadline = time.time() + 2.0
        while time.time() < finished_deadline:
            snapshot = task_store.get_task_snapshot(task.task_id)
            if snapshot.get("status") == "succeeded":
                break
            time.sleep(0.02)

        final_snapshot = task_store.get_task_snapshot(task.task_id)
        assert final_snapshot["status"] == "succeeded"
        assert final_snapshot["result"]["glm"]["text"] == "glm text"
    finally:
        queue.stop()
