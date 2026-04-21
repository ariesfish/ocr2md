from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, StreamingResponse

from ...utils.image_utils import crop_image_region, pdf_to_images_pil_iter
from ..errors import OCRAPIError, TaskLookupError
from ..schemas import TaskSnapshot

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _resolve_task_dir(request: Request, task_id: str) -> Path:
    services = request.app.state.services
    output_root = Path(services.config.pipeline.output.base_output_dir)
    return output_root.expanduser().resolve() / task_id


def _resolve_live_layout_file(task_dir: Path, page: int) -> Path | None:
    layout_dir = task_dir / "layout"
    for ext in (".jpg", ".png"):
        candidate = layout_dir / f"layout_page{page}{ext}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_task_source_file(task_dir: Path) -> Path:
    input_dir = task_dir / "input"
    if not input_dir.is_dir():
        raise OCRAPIError(
            status_code=404,
            code="TASK_SOURCE_NOT_FOUND",
            message=f"Task input directory not found: {input_dir}",
        )
    candidates = sorted(path for path in input_dir.iterdir() if path.is_file())
    if not candidates:
        raise OCRAPIError(
            status_code=404,
            code="TASK_SOURCE_NOT_FOUND",
            message=f"Task input file not found under: {input_dir}",
        )
    return candidates[0].resolve()


def _resolve_task_model_result_dir(task_dir: Path, model_key: str) -> Path:
    result_root = task_dir / "results" / model_key
    if not result_root.is_dir():
        raise OCRAPIError(
            status_code=404,
            code="MODEL_RESULT_NOT_FOUND",
            message=f"Model result directory not found: {result_root}",
        )

    candidates = sorted(path.resolve() for path in result_root.iterdir() if path.is_dir())
    if not candidates:
        raise OCRAPIError(
            status_code=404,
            code="MODEL_RESULT_NOT_FOUND",
            message=f"No model result found under: {result_root}",
        )

    return candidates[0]


@router.get("/{task_id}", response_model=TaskSnapshot)
def get_task(task_id: str, request: Request) -> TaskSnapshot:
    services = request.app.state.services
    try:
        payload = services.task_store.get_task_snapshot(task_id)
    except TaskLookupError:
        raise
    return TaskSnapshot.model_validate(payload)


@router.get("/{task_id}/layout")
def get_task_layout_preview(
    task_id: str, request: Request, page: int = 0
) -> FileResponse:
    if page < 0:
        raise OCRAPIError(
            status_code=400,
            code="INVALID_PAGE_INDEX",
            message=f"page must be >= 0, got {page}",
        )

    services = request.app.state.services
    payload = services.task_store.get_task_snapshot(task_id)
    result = payload.get("result") if isinstance(payload, dict) else None
    task_dir = _resolve_task_dir(request, task_id)

    live_layout_path = _resolve_live_layout_file(task_dir, page)
    if live_layout_path is not None:
        return FileResponse(path=live_layout_path)

    if not isinstance(result, dict):
        raise OCRAPIError(
            status_code=404,
            code="LAYOUT_PREVIEW_NOT_FOUND",
            message=f"Task has no result payload: {task_id}",
        )

    manifest_raw = result.get("manifest")
    if not isinstance(manifest_raw, str) or not manifest_raw.strip():
        raise OCRAPIError(
            status_code=404,
            code="LAYOUT_PREVIEW_NOT_FOUND",
            message=f"Task result has no manifest: {task_id}",
        )

    manifest_path = Path(manifest_raw).expanduser().resolve()
    if not manifest_path.is_file():
        raise OCRAPIError(
            status_code=404,
            code="MANIFEST_NOT_FOUND",
            message=f"Manifest file not found: {manifest_path}",
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OCRAPIError(
            status_code=422,
            code="MANIFEST_INVALID",
            message=f"Manifest json parse failed: {exc}",
        ) from exc

    layout = manifest.get("layout") if isinstance(manifest, dict) else None
    files = layout.get("files") if isinstance(layout, dict) else None
    if not isinstance(files, list) or not files:
        raise OCRAPIError(
            status_code=404,
            code="LAYOUT_PREVIEW_NOT_FOUND",
            message=f"No layout preview files found for task: {task_id}",
        )

    if page >= len(files):
        raise OCRAPIError(
            status_code=404,
            code="LAYOUT_PAGE_NOT_FOUND",
            message=f"Layout page {page} not found, total={len(files)}",
        )

    layout_path = Path(str(files[page])).expanduser().resolve()
    if not layout_path.is_file():
        raise OCRAPIError(
            status_code=404,
            code="LAYOUT_FILE_NOT_FOUND",
            message=f"Layout file missing: {layout_path}",
        )

    return FileResponse(path=layout_path)


@router.get("/{task_id}/crop")
def get_task_crop_preview(
    task_id: str,
    request: Request,
    page: int = 0,
    bbox: str = "",
) -> StreamingResponse:
    if page < 0:
        raise OCRAPIError(
            status_code=400,
            code="INVALID_PAGE_INDEX",
            message=f"page must be >= 0, got {page}",
        )

    try:
        coords = [int(float(item.strip())) for item in bbox.split(",") if item.strip()]
    except ValueError as exc:
        raise OCRAPIError(
            status_code=400,
            code="INVALID_BBOX",
            message=f"bbox must contain 4 numeric values, got: {bbox}",
        ) from exc

    if len(coords) != 4:
        raise OCRAPIError(
            status_code=400,
            code="INVALID_BBOX",
            message=f"bbox must contain 4 numeric values, got: {bbox}",
        )

    task_dir = _resolve_task_dir(request, task_id)
    source_path = _resolve_task_source_file(task_dir)
    suffix = source_path.suffix.lower()
    payload = request.app.state.services.task_store.get_task_snapshot(task_id)
    result = payload.get("result") if isinstance(payload, dict) else None
    input_meta = result.get("input") if isinstance(result, dict) else None
    source_pages = input_meta.get("source_pages") if isinstance(input_meta, dict) else None
    source_page = page
    if isinstance(source_pages, list) and 0 <= page < len(source_pages):
        try:
            source_page = int(source_pages[page]) - 1
        except (TypeError, ValueError):
            source_page = page

    if suffix == ".pdf":
        try:
            image = next(
                pdf_to_images_pil_iter(
                    str(source_path),
                    dpi=200,
                    max_width_or_height=3500,
                    start_page_id=source_page,
                    end_page_id=source_page,
                )
            )
        except StopIteration as exc:
            raise OCRAPIError(
                status_code=404,
                code="CROP_PAGE_NOT_FOUND",
                message=f"PDF page not found: {page}",
            ) from exc
    else:
        if page != 0:
            raise OCRAPIError(
                status_code=404,
                code="CROP_PAGE_NOT_FOUND",
                message=f"Image input has only one page, got: {page}",
            )
        from PIL import Image

        image = Image.open(source_path)

    try:
        cropped = crop_image_region(image, coords)
        buffer = BytesIO()
        cropped.save(buffer, format="PNG")
        buffer.seek(0)
    finally:
        close = getattr(image, "close", None)
        if callable(close):
            close()
        close = getattr(locals().get("cropped"), "close", None)
        if callable(close):
            close()

    return StreamingResponse(buffer, media_type="image/png")


@router.get("/{task_id}/models/{model_key}/assets/{asset_path:path}")
def get_task_model_asset(
    task_id: str,
    model_key: str,
    asset_path: str,
    request: Request,
) -> FileResponse:
    if model_key != "glm":
        raise OCRAPIError(
            status_code=404,
            code="MODEL_NOT_FOUND",
            message=f"Unsupported model key: {model_key}",
        )

    if not asset_path.strip():
        raise OCRAPIError(
            status_code=400,
            code="INVALID_ASSET_PATH",
            message="asset_path cannot be empty",
        )

    task_dir = _resolve_task_dir(request, task_id)
    result_dir = _resolve_task_model_result_dir(task_dir, model_key)
    asset_file = (result_dir / asset_path).resolve()
    try:
        asset_file.relative_to(result_dir)
    except ValueError as exc:
        raise OCRAPIError(
            status_code=400,
            code="INVALID_ASSET_PATH",
            message=f"Asset path escapes result directory: {asset_path}",
        ) from exc

    if not asset_file.is_file():
        raise OCRAPIError(
            status_code=404,
            code="ASSET_NOT_FOUND",
            message=f"Asset file not found: {asset_path}",
        )

    return FileResponse(path=asset_file)
