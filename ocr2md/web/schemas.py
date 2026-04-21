from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

TaskStatus = Literal["queued", "running", "succeeded", "failed", "expired"]
TaskType = Literal["ocr", "download"]
ModelStatusType = Literal["not_ready", "downloading", "ready", "error"]
ModelKey = Literal["glm"]
DownloadSource = Literal["modelscope", "huggingface"]


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ModelStatusItem(BaseModel):
    name: str
    status: ModelStatusType
    progress: int = Field(default=0, ge=0, le=100)
    message: Optional[str] = None
    model_root_dir: Optional[str] = None


class ModelsStatusResponse(BaseModel):
    models: Dict[str, ModelStatusItem]


class DownloadRequest(BaseModel):
    source: Optional[DownloadSource] = None
    model_dir: Optional[str] = None


class ModelDirRequest(BaseModel):
    model_dir: str


class TaskCreatedResponse(BaseModel):
    task_id: str
    task_type: TaskType
    status: TaskStatus


class OCRBox(BaseModel):
    index: int
    label: str
    score: Optional[float] = None
    bbox_2d: Tuple[float, float, float, float]
    polygon: Optional[List[List[float]]] = None


class OCRModelResult(BaseModel):
    name: str
    latency_ms: Optional[int] = None
    confidence_avg: Optional[float] = None
    text: str = ""
    boxes: List[OCRBox] = Field(default_factory=list)
    error: Optional[str] = None


class OCRInputMeta(BaseModel):
    filename: str
    width: Optional[int] = None
    height: Optional[int] = None
    page_count: Optional[int] = None
    source_pages: Optional[List[int]] = None


class OCRResult(BaseModel):
    request_id: str
    input: OCRInputMeta
    glm: OCRModelResult
    manifest: Optional[str] = None


class TaskSnapshot(BaseModel):
    task_id: str
    task_type: TaskType
    status: TaskStatus
    progress: int = Field(default=0, ge=0, le=100)
    stage: Optional[str] = None
    message: Optional[str] = None
    model_key: Optional[ModelKey] = None
    created_at: str
    updated_at: str
    result: Optional[dict] = None
    error: Optional[dict] = None
