from __future__ import annotations

from typing import Optional, cast

from fastapi import APIRouter, Request

from ..schemas import (
    DownloadRequest,
    ModelDirRequest,
    ModelStatusItem,
    ModelsStatusResponse,
    TaskCreatedResponse,
    TaskStatus,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/status", response_model=ModelsStatusResponse)
def models_status(request: Request) -> ModelsStatusResponse:
    services = request.app.state.services
    payload = services.model_manager.get_models_status()
    return ModelsStatusResponse.model_validate({"models": payload})


@router.post("/{model_key}/set-dir", response_model=ModelStatusItem)
def set_model_dir(
    model_key: str,
    request: Request,
    payload: ModelDirRequest,
) -> ModelStatusItem:
    services = request.app.state.services
    status = services.model_manager.set_model_dir(
        model_key=model_key,
        model_dir=payload.model_dir,
    )
    services.ocr_runner.stop()
    return ModelStatusItem.model_validate(status)


@router.post("/{model_key}/download", response_model=TaskCreatedResponse)
def create_download_task(
    model_key: str,
    request: Request,
    payload: Optional[DownloadRequest] = None,
) -> TaskCreatedResponse:
    services = request.app.state.services
    request_id = request.state.request_id
    task_id = services.model_manager.create_download_task(
        model_key=model_key,
        request_id=request_id,
        source=(payload.source if payload else None),
        model_dir=(payload.model_dir if payload else None),
    )
    if payload is not None and payload.model_dir is not None:
        services.ocr_runner.stop()
    snapshot = services.task_store.get_task_snapshot(task_id)
    return TaskCreatedResponse(
        task_id=task_id,
        task_type="download",
        status=cast(TaskStatus, str(snapshot.get("status", "queued"))),
    )
