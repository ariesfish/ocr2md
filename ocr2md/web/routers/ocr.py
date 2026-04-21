from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, Request, UploadFile

from ..errors import OCRAPIError
from ..schemas import OCRResult, TaskCreatedResponse

router = APIRouter(prefix="/ocr", tags=["ocr"])


def _parse_page_selection(page_selection: Optional[str]) -> Optional[List[int]]:
    cleaned = str(page_selection or "").strip()
    if not cleaned:
        return None

    selected_pages: List[int] = []
    seen: set[int] = set()
    for chunk in cleaned.split(","):
        token = chunk.strip()
        if not token:
            continue

        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            try:
                start = int(start_raw.strip())
                end = int(end_raw.strip())
            except ValueError as exc:
                raise OCRAPIError(
                    status_code=400,
                    code="INVALID_PAGE_SELECTION",
                    message=f"Invalid page range: {token}",
                ) from exc
            if start <= 0 or end <= 0 or end < start:
                raise OCRAPIError(
                    status_code=400,
                    code="INVALID_PAGE_SELECTION",
                    message=f"Invalid page range: {token}",
                )
            for page in range(start, end + 1):
                page_index = page - 1
                if page_index not in seen:
                    selected_pages.append(page_index)
                    seen.add(page_index)
            continue

        try:
            page = int(token)
        except ValueError as exc:
            raise OCRAPIError(
                status_code=400,
                code="INVALID_PAGE_SELECTION",
                message=f"Invalid page number: {token}",
            ) from exc
        if page <= 0:
            raise OCRAPIError(
                status_code=400,
                code="INVALID_PAGE_SELECTION",
                message=f"Invalid page number: {token}",
            )

        page_index = page - 1
        if page_index not in seen:
            selected_pages.append(page_index)
            seen.add(page_index)

    if not selected_pages:
        raise OCRAPIError(
            status_code=400,
            code="INVALID_PAGE_SELECTION",
            message="Page selection is empty",
        )
    return selected_pages


def _validate_upload(
    *,
    filename: str,
    file_bytes: bytes,
    max_size_bytes: int,
    allowed_suffixes: set[str],
) -> None:
    if not filename:
        raise OCRAPIError(
            status_code=400, code="INVALID_FILE", message="Missing filename"
        )
    if not file_bytes:
        raise OCRAPIError(
            status_code=400, code="EMPTY_FILE", message="Uploaded file is empty"
        )
    if len(file_bytes) > max_size_bytes:
        raise OCRAPIError(
            status_code=400,
            code="FILE_TOO_LARGE",
            message=f"File too large, max bytes={max_size_bytes}",
        )
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in allowed_suffixes:
        raise OCRAPIError(
            status_code=400,
            code="UNSUPPORTED_FILE_TYPE",
            message=f"Unsupported file type: .{suffix}",
        )


@router.post("/jobs", response_model=TaskCreatedResponse)
async def create_ocr_job(
    request: Request,
    file: UploadFile = File(...),
    page_selection: str | None = Form(default=None),
) -> TaskCreatedResponse:
    services = request.app.state.services
    request_id = request.state.request_id

    file_bytes = await file.read()
    allowed_suffixes = {
        item.lower() for item in services.config.web.allowed_image_suffixes
    }
    max_size_bytes = int(services.config.web.upload_max_mb) * 1024 * 1024
    _validate_upload(
        filename=file.filename or "",
        file_bytes=file_bytes,
        max_size_bytes=max_size_bytes,
        allowed_suffixes=allowed_suffixes,
    )
    selected_pages = _parse_page_selection(page_selection)
    if selected_pages is not None and Path(file.filename or "").suffix.lower() != ".pdf":
        raise OCRAPIError(
            status_code=400,
            code="INVALID_PAGE_SELECTION",
            message="Page selection is only supported for PDF files",
        )

    services.model_manager.assert_models_ready()

    task = services.task_store.create_task(task_type="ocr", request_id=request_id)
    services.ocr_queue.submit(
        task_id=task.task_id,
        request_id=request_id,
        filename=file.filename or "upload.png",
        file_bytes=file_bytes,
        selected_pages=selected_pages,
    )
    return TaskCreatedResponse(task_id=task.task_id, task_type="ocr", status="queued")


@router.post("/run", response_model=OCRResult)
async def run_sync_ocr(
    request: Request,
    file: UploadFile = File(...),
    page_selection: str | None = Form(default=None),
) -> OCRResult:
    services = request.app.state.services
    request_id = request.state.request_id
    services.model_manager.assert_models_ready()

    file_bytes = await file.read()
    allowed_suffixes = {
        item.lower() for item in services.config.web.allowed_image_suffixes
    }
    max_size_bytes = int(services.config.web.upload_max_mb) * 1024 * 1024
    _validate_upload(
        filename=file.filename or "",
        file_bytes=file_bytes,
        max_size_bytes=max_size_bytes,
        allowed_suffixes=allowed_suffixes,
    )
    selected_pages = _parse_page_selection(page_selection)
    if selected_pages is not None and Path(file.filename or "").suffix.lower() != ".pdf":
        raise OCRAPIError(
            status_code=400,
            code="INVALID_PAGE_SELECTION",
            message="Page selection is only supported for PDF files",
        )

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        prefix="ocr2md_sync_",
        delete=False,
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_path = Path(temp_file.name).resolve()

    try:
        result = services.ocr_runner.run_ocr(
            source_path=temp_path,
            request_id=request_id,
            selected_pages=selected_pages,
            persist_artifacts=False,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return OCRResult.model_validate(result)
