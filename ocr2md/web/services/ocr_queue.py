from __future__ import annotations

import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import List, Optional

from ...utils.logging import get_logger
from ..stores.task_store import TaskStore
from .ocr_runner import PipelineOCRRunner

logger = get_logger(__name__)


@dataclass
class OCRQueueItem:
    task_id: str
    request_id: str
    filename: str
    file_path: Path
    selected_pages: Optional[List[int]] = None


class OCRTaskQueue:
    def __init__(self, task_store: TaskStore, ocr_runner: PipelineOCRRunner):
        self._task_store = task_store
        self._ocr_runner = ocr_runner
        self._stop_event = Event()
        self._queue: Queue[OCRQueueItem] = Queue()
        self._thread = Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def submit(
        self,
        *,
        task_id: str,
        request_id: str,
        filename: str,
        file_bytes: bytes,
        selected_pages: Optional[List[int]] = None,
    ) -> None:
        safe_filename = Path(filename).name or "upload.png"
        temp_dir = Path(tempfile.mkdtemp(prefix="ocr2md_upload_"))
        temp_path = (temp_dir / safe_filename).resolve()
        temp_path.write_bytes(file_bytes)

        self._queue.put(
            OCRQueueItem(
                task_id=task_id,
                request_id=request_id,
                filename=filename,
                file_path=temp_path,
                selected_pages=selected_pages,
            )
        )

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                partial_result: dict = {"request_id": item.request_id}

                def _on_model_result(
                    model_key: str,
                    model_result: dict,
                    input_meta: dict,
                ) -> None:
                    partial_result["input"] = input_meta
                    partial_result[model_key] = model_result
                    self._task_store.update_task(
                        item.task_id,
                        status="running",
                        result=deepcopy(partial_result),
                    )

                def _on_progress(
                    progress: int,
                    stage: str,
                    message: str | None,
                ) -> None:
                    self._task_store.update_task(
                        item.task_id,
                        status="running",
                        progress=progress,
                        stage=stage,
                        message=message,
                    )

                self._task_store.update_task(
                    item.task_id,
                    status="running",
                    progress=5,
                    stage="accepted",
                    message="task accepted by worker",
                )

                result = self._ocr_runner.run_ocr(
                    source_path=item.file_path,
                    request_id=item.request_id,
                    task_id=item.task_id,
                    selected_pages=item.selected_pages,
                    persist_artifacts=True,
                    progress_callback=_on_progress,
                    model_result_callback=_on_model_result,
                )
                self._task_store.mark_succeeded(
                    item.task_id,
                    result=result,
                    message="ocr completed",
                )
            except Exception as exc:
                self._task_store.mark_failed(
                    item.task_id,
                    code="OCR_FAILED",
                    message=str(exc),
                    request_id=item.request_id,
                )
            finally:
                try:
                    item.file_path.unlink(missing_ok=True)
                    item.file_path.parent.rmdir()
                except Exception:
                    logger.warning("Failed to delete temp input: %s", item.file_path)
                self._queue.task_done()
