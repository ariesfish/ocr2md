from __future__ import annotations

import gc
import json
import re
import shutil
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...config import Config
from ...ocr_pipeline import OCRPipeline
from ...parser_result import PipelineResult
from ...utils.image_utils import PYPDFIUM2_AVAILABLE
from ...utils.logging import get_logger
from .result_mapper import map_backend_result

logger = get_logger(__name__)

_MARKDOWN_IMAGE_PATH_PATTERN = re.compile(r"(!\[[^\]]*\]\()((?:\./)?imgs/[^)\s]+)(\))")


class PipelineOCRRunner:
    def __init__(self, config: Config):
        self._config = config
        self._pipeline: Optional[OCRPipeline] = None
        self._pipeline_lock = Lock()
        self._timeout_seconds = int(config.web.model_timeout_seconds)
        self._model_key = "glm"

    def stop(self) -> None:
        with self._pipeline_lock:
            if self._pipeline is not None:
                self._pipeline.stop()
                self._pipeline = None

    def _ensure_pipeline(self) -> OCRPipeline:
        with self._pipeline_lock:
            if self._pipeline is None:
                self._pipeline = OCRPipeline(self._config.pipeline)
            if not self._pipeline._started:
                self._pipeline.start()
            return self._pipeline

    @staticmethod
    def _safe_model_payload(
        regions: List[Tuple[Any, Dict[str, Any], str, int]], page_count: int
    ) -> List[List[Dict[str, Any]]]:
        payload: List[List[Dict[str, Any]]] = [[] for _ in range(page_count)]
        per_page_index: List[int] = [0 for _ in range(page_count)]
        for _img, info, _task, page_idx in regions:
            page_slot = payload[page_idx]
            page_item = {
                "index": per_page_index[page_idx],
                "label": str(info.get("label", "text")),
                "content": "",
                "bbox_2d": info.get("bbox_2d"),
                "score": info.get("score"),
                "polygon": info.get("polygon"),
            }
            per_page_index[page_idx] += 1
            page_slot.append(page_item)
        return payload

    @staticmethod
    def _run_with_timeout(
        fn: Callable[[], Any],
        timeout_seconds: int,
    ) -> Tuple[bool, Optional[Any], Optional[BaseException], float]:
        done = Event()
        holder: Dict[str, Any] = {}

        def _target() -> None:
            try:
                holder["result"] = fn()
            except BaseException as exc:
                holder["error"] = exc
            finally:
                done.set()

        started_at = time.perf_counter()
        thread = Thread(target=_target, daemon=True)
        thread.start()
        finished = done.wait(timeout=timeout_seconds)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        if not finished:
            return False, None, TimeoutError("model execution timeout"), elapsed_ms

        if "error" in holder:
            return True, None, holder["error"], elapsed_ms
        return True, holder.get("result"), None, elapsed_ms

    @staticmethod
    def _resolve_source_page_indices(
        pipeline: OCRPipeline,
        source_path: Path,
        selected_pages: Optional[List[int]] = None,
    ) -> List[int]:
        if source_path.suffix.lower() != ".pdf":
            return [0]

        if not PYPDFIUM2_AVAILABLE:
            raise RuntimeError(
                "PDF support requires pypdfium2. Install: pip install pypdfium2"
            )

        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(source_path))
        try:
            page_count = len(pdf)
        finally:
            pdf.close()

        if selected_pages:
            out_of_range = [page for page in selected_pages if page < 0 or page >= page_count]
            if out_of_range:
                first_page = out_of_range[0] + 1
                raise ValueError(
                    f"Selected PDF page out of range: {first_page}, total pages={page_count}"
                )
            return list(selected_pages)

        max_pages = getattr(pipeline.page_loader, "pdf_max_pages", None)
        if max_pages is None:
            return list(range(page_count))

        try:
            max_pages_int = int(max_pages)
        except (TypeError, ValueError):
            return list(range(page_count))

        if max_pages_int <= 0:
            return list(range(page_count))
        return list(range(min(page_count, max_pages_int)))

    @staticmethod
    def _close_image(image: Any) -> None:
        close = getattr(image, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    @staticmethod
    def _rewrite_markdown_asset_urls(
        markdown_text: str,
        *,
        task_id: str,
        model_key: str,
    ) -> str:
        if not markdown_text.strip():
            return markdown_text

        def _replace(match: re.Match[str]) -> str:
            relative_path = match.group(2)
            normalized_path = relative_path[2:] if relative_path.startswith("./") else relative_path
            return (
                f"{match.group(1)}/api/tasks/{task_id}/models/{model_key}/assets/{normalized_path}"
                f"{match.group(3)}"
            )

        return _MARKDOWN_IMAGE_PATH_PATTERN.sub(_replace, markdown_text)

    def _release_page_resources(
        self,
        page: Any,
        page_regions: List[Tuple[Any, Dict[str, Any], str, int]],
    ) -> None:
        for region_image, _info, _task, _page_idx in page_regions:
            self._close_image(region_image)
        self._close_image(page)
        gc.collect()

    def run_ocr(
        self,
        source_path: Path,
        request_id: str,
        task_id: Optional[str] = None,
        selected_pages: Optional[List[int]] = None,
        persist_artifacts: bool = True,
        progress_callback: Optional[Callable[[int, str, Optional[str]], None]] = None,
        model_result_callback: Optional[
            Callable[[str, Dict[str, Any], Dict[str, Any]], None]
        ] = None,
    ) -> Dict[str, Any]:
        pipeline = self._ensure_pipeline()
        source_path = source_path.expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {source_path}")

        workspace: Optional[Dict[str, Path]] = None
        copied_source = source_path
        resolved_task_id = task_id or f"sync_{request_id[-8:]}"

        if persist_artifacts:
            resolved_task_id, workspace = pipeline._prepare_task_workspace(
                source_path=source_path,
                task_id=task_id,
                output_root_dir=self._config.pipeline.output.base_output_dir,
            )
            copied_source = workspace["input"] / source_path.name
            shutil.copy2(source_path, copied_source)

        if progress_callback is not None:
            progress_callback(10, "input_ready", None)

        with tempfile.TemporaryDirectory(
            prefix="ocr2md_sync_layout_"
        ) as temp_layout_root:
            layout_dir = (
                str(workspace["layout"])
                if workspace is not None
                else str(Path(temp_layout_root).resolve())
            )

            if progress_callback is not None:
                progress_callback(20, "layout_detection", None)

            source_page_indices = self._resolve_source_page_indices(
                pipeline,
                copied_source,
                selected_pages=selected_pages,
            )
            total_pages = len(source_page_indices)
            if total_pages <= 0:
                raise RuntimeError(f"No pages loaded from source: {copied_source}")

            width: Optional[int] = None
            height: Optional[int] = None
            input_meta = {
                "filename": source_path.name,
                "width": None,
                "height": None,
                "page_count": total_pages,
                "source_pages": [page + 1 for page in source_page_indices],
            }
            grouped_results: List[List[Dict[str, Any]]] = [
                [] for _ in range(total_pages)
            ]
            latency_ms: Optional[int] = None
            backend_error: Optional[str] = None
            started_at = time.perf_counter()

            for page_idx, (page, _unit_idx) in enumerate(
                pipeline.page_loader.iter_pages_with_unit_indices(
                    [str(copied_source)],
                    page_indices=source_page_indices,
                )
            ):
                if page_idx >= total_pages:
                    break

                if width is None or height is None:
                    width = int(page.width)
                    height = int(page.height)
                    input_meta["width"] = width
                    input_meta["height"] = height

                page_layout = pipeline._run_layout_detection(
                    pages=[page],
                    save_layout_visualization=True,
                    layout_vis_output_dir=layout_dir,
                    global_start_idx=page_idx,
                )
                page_regions_local = pipeline._prepare_regions([page], page_layout)
                page_regions = [
                    (img, info, task, page_idx)
                    for img, info, task, _ in page_regions_local
                ]
                grouped_results[page_idx] = self._safe_model_payload(
                    page_regions, total_pages
                )[page_idx]

                if progress_callback is not None:
                    layout_progress = 20 + int(15 * (page_idx + 1) / total_pages)
                    progress_callback(
                        layout_progress,
                        "layout_detection",
                        f"page={page_idx + 1}/{total_pages}",
                    )
                    progress_callback(
                        min(layout_progress + 1, 35),
                        "ocr_prepare",
                        f"page={page_idx + 1}/{total_pages}, regions={len(page_regions)}",
                    )

                finished, output, error, _elapsed_ms = self._run_with_timeout(
                    lambda page_regions=page_regions: pipeline._run_backend_regions(
                        self._model_key, page_regions
                    ),
                    timeout_seconds=self._timeout_seconds,
                )
                latency_ms = int((time.perf_counter() - started_at) * 1000)

                if not finished and isinstance(error, TimeoutError):
                    backend_error = f"{self._model_key} inference timeout after {self._timeout_seconds}s"
                    self._release_page_resources(page, page_regions)
                    break
                if error is not None:
                    backend_error = str(error)
                    self._release_page_resources(page, page_regions)
                    break

                page_recognition_results = output or []
                grouped_results[page_idx] = [
                    deepcopy(item)
                    for result_page_idx, item in page_recognition_results
                    if result_page_idx == page_idx
                ]

                if model_result_callback is not None:
                    partial_json_str, partial_markdown = pipeline.result_formatter.process(
                        grouped_results
                    )
                    model_result_callback(
                        self._model_key,
                        map_backend_result(
                            model_key=self._model_key,
                            json_result=json.loads(partial_json_str),
                            markdown_result=partial_markdown,
                            latency_ms=latency_ms,
                            error=None,
                        ),
                        deepcopy(input_meta),
                    )

                if progress_callback is not None:
                    ocr_progress = 35 + int(45 * (page_idx + 1) / total_pages)
                    progress_callback(
                        ocr_progress,
                        f"{self._model_key}_inference",
                        f"page={page_idx + 1}/{total_pages}",
                    )

                self._release_page_resources(page, page_regions)

            if width is None or height is None:
                raise RuntimeError(f"No pages loaded from source: {copied_source}")

            if latency_ms is None:
                latency_ms = int((time.perf_counter() - started_at) * 1000)

            json_str, markdown_payload = pipeline.result_formatter.process(
                grouped_results
            )
            json_payload = json.loads(json_str)

            mapped_result = map_backend_result(
                model_key=self._model_key,
                json_result=json_payload,
                markdown_result=markdown_payload,
                latency_ms=latency_ms,
                error=backend_error,
            )

            if progress_callback is not None:
                progress_callback(80, f"{self._model_key}_inference", backend_error)
                progress_callback(92, "formatting", None)

            model_artifacts: Dict[str, Dict[str, Any]] = {}
            if workspace is not None:
                model_output_root = workspace["results"] / self._model_key
                model_output_root.mkdir(parents=True, exist_ok=True)
                PipelineResult(
                    json_result=json_payload,
                    markdown_result=markdown_payload,
                    original_images=[str(copied_source)],
                    layout_vis_dir=str(workspace["layout"]),
                    layout_image_indices=list(range(total_pages)),
                ).save(
                    output_dir=str(model_output_root),
                    save_layout_visualization=False,
                )
                result_dir = model_output_root / copied_source.stem
                markdown_path = result_dir / f"{copied_source.stem}.md"
                if markdown_path.is_file():
                    mapped_result["text"] = self._rewrite_markdown_asset_urls(
                        markdown_path.read_text(encoding="utf-8"),
                        task_id=resolved_task_id,
                        model_key=self._model_key,
                    )
                model_artifacts[self._model_key] = {
                    "result_dir": str(result_dir.resolve()),
                    "json": str((result_dir / f"{copied_source.stem}.json").resolve()),
                    "markdown": str(
                        (result_dir / f"{copied_source.stem}.md").resolve()
                    ),
                    "imgs_dir": str((result_dir / "imgs").resolve()),
                }

            manifest_path: Optional[str] = None
            if workspace is not None:
                layout_files = pipeline._collect_layout_files(workspace["layout"])
                manifest = {
                    "task_id": resolved_task_id,
                    "task_dir": str(workspace["task"].resolve()),
                    "input": {
                        "original_path": str(source_path),
                        "copied_path": str(copied_source.resolve()),
                        "source_pages": [page + 1 for page in source_page_indices],
                    },
                    "layout": {
                        "dir": str(workspace["layout"].resolve()),
                        "files": layout_files,
                    },
                    "models": model_artifacts,
                }
                manifest_file = workspace["meta"] / "manifest.json"
                manifest_file.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                manifest_path = str(manifest_file.resolve())

            if model_result_callback is not None:
                model_result_callback(
                    self._model_key,
                    deepcopy(mapped_result),
                    deepcopy(input_meta),
                )

            return {
                "request_id": request_id,
                "input": {
                    "filename": source_path.name,
                    "width": width,
                    "height": height,
                    "page_count": total_pages,
                    "source_pages": [page + 1 for page in source_page_indices],
                },
                "glm": mapped_result,
                "manifest": manifest_path,
            }
