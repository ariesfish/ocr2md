from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
import subprocess
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Tuple

from ...config import Config
from ...utils.logging import get_logger
from ...utils.model_dir_utils import (
    GLM_DEFAULT_SUBDIR,
    GLM_MODEL_DIR_HINTS,
    LAYOUT_DEFAULT_SUBDIR,
    LAYOUT_MODEL_DIR_HINTS,
    LAYOUT_REQUIRED_FILES,
    resolve_model_dir_path,
)
from ..errors import OCRAPIError
from ..stores.task_store import TaskStore

logger = get_logger(__name__)

MODEL_KEY = "glm"
MODEL_NAME = "GLM-OCR"


@dataclass(frozen=True)
class DownloadTarget:
    name: str
    model_dir: Path
    required_files: List[str]
    download_source: str
    modelscope_repo_id: str
    huggingface_repo_id: str


class ModelManager:
    def __init__(self, config: Config, task_store: TaskStore):
        self._config = config
        self._task_store = task_store

    def _model_config(self, model_key: str):
        if model_key == MODEL_KEY:
            return self._config.pipeline.glm_ocr_backend
        raise OCRAPIError(
            status_code=400,
            code="INVALID_MODEL_KEY",
            message=f"Unsupported model_key: {model_key}",
        )

    def _resolve_model_dir(self, model_dir: str) -> Path:
        return resolve_model_dir_path(
            model_dir,
            required_files=(
                getattr(
                    self._config.pipeline.glm_ocr_backend, "required_model_files", []
                )
                if self._config.pipeline.glm_ocr_backend is not None
                else []
            ),
            preferred_dir_hints=GLM_MODEL_DIR_HINTS,
            default_subdir_name=GLM_DEFAULT_SUBDIR,
            search_roots=[Path.cwd()],
            must_exist=False,
        )

    @staticmethod
    def _matches_dir_hints(path: Path, hints: Tuple[str, ...]) -> bool:
        path_name = path.name.strip().lower().replace("_", "-")
        return any(hint.strip().lower().replace("_", "-") in path_name for hint in hints)

    def _resolve_layout_dir(self, model_dir: str) -> Path:
        raw = Path(str(model_dir).strip()).expanduser()
        base = raw.resolve() if raw.is_absolute() else (Path.cwd() / raw).resolve()

        if base.is_dir() and not self._matches_dir_hints(base, LAYOUT_MODEL_DIR_HINTS):
            hinted_subdirs = sorted(
                [
                    item.resolve()
                    for item in base.iterdir()
                    if item.is_dir() and self._matches_dir_hints(item, LAYOUT_MODEL_DIR_HINTS)
                ],
                key=lambda item: item.name.lower(),
            )
            if hinted_subdirs:
                return hinted_subdirs[0]

        return resolve_model_dir_path(
            model_dir,
            required_files=LAYOUT_REQUIRED_FILES,
            preferred_dir_hints=LAYOUT_MODEL_DIR_HINTS,
            default_subdir_name=LAYOUT_DEFAULT_SUBDIR,
            search_roots=[Path.cwd()],
            must_exist=False,
        )

    def _apply_model_dir_override(
        self, model_key: str, model_dir: Optional[str]
    ) -> Optional[Path]:
        if model_dir is None:
            return None

        cleaned = str(model_dir).strip()
        if not cleaned:
            raise OCRAPIError(
                status_code=400,
                code="INVALID_MODEL_DIR",
                message="model_dir cannot be empty",
            )

        model_config = self._model_config(model_key)
        if model_config is None:
            raise OCRAPIError(
                status_code=404,
                code="MODEL_NOT_FOUND",
                message=f"Unknown model config: {model_key}",
            )

        resolved_glm_dir = self._resolve_model_dir(cleaned)
        model_config.model_dir = str(resolved_glm_dir)

        resolved_layout_dir = self._resolve_layout_dir(cleaned)
        self._config.pipeline.layout.model_dir = str(resolved_layout_dir)
        return resolved_glm_dir

    def _required_files_status(
        self, model_key: str
    ) -> Tuple[Path, List[str], List[str]]:
        model_config = self._model_config(model_key)
        if model_config is None or not bool(getattr(model_config, "enabled", False)):
            raise OCRAPIError(
                status_code=404,
                code="MODEL_DISABLED",
                message=f"Model is disabled in config: {model_key}",
            )

        required_files = [
            str(item).strip()
            for item in list(getattr(model_config, "required_model_files", []) or [])
            if str(item).strip()
        ]
        if not required_files:
            raise OCRAPIError(
                status_code=500,
                code="MODEL_CONFIG_ERROR",
                message=f"required_model_files is empty for model: {model_key}",
            )

        model_dir = self._resolve_model_dir(str(getattr(model_config, "model_dir", "")))
        existing = []
        for filename in required_files:
            if (model_dir / filename).is_file():
                existing.append(filename)
        return model_dir, required_files, existing

    def _layout_required_files_status(self) -> Tuple[Path, List[str], List[str]]:
        required_files = sorted(LAYOUT_REQUIRED_FILES)
        layout_dir = self._resolve_layout_dir(str(self._config.pipeline.layout.model_dir))
        existing = []
        for filename in required_files:
            if (layout_dir / filename).is_file():
                existing.append(filename)
        return layout_dir, required_files, existing

    @staticmethod
    def _format_missing_dir_message(
        prefix: str,
        model_dir: Path,
    ) -> str:
        return f"{prefix} model directory is missing: {model_dir}"

    @staticmethod
    def _format_missing_message(prefix: str, model_dir: Path, missing: List[str]) -> str:
        del missing
        return f"{prefix} model directory is incomplete: {model_dir}"

    @staticmethod
    def _model_root_dir(glm_dir: Path, layout_dir: Path) -> str:
        common_path = os.path.commonpath([str(glm_dir), str(layout_dir)])
        return str(Path(common_path).resolve())

    def _current_model_status(self) -> Dict:
        glm_dir, glm_required_files, glm_existing_files = self._required_files_status(
            MODEL_KEY
        )
        layout_dir, layout_required_files, layout_existing_files = (
            self._layout_required_files_status()
        )
        model_root_dir = self._model_root_dir(glm_dir, layout_dir)

        if not glm_dir.exists():
            return {
                "name": MODEL_NAME,
                "status": "not_ready",
                "progress": 0,
                "message": self._format_missing_dir_message("glm", glm_dir),
                "model_root_dir": model_root_dir,
            }

        if not layout_dir.exists():
            return {
                "name": MODEL_NAME,
                "status": "not_ready",
                "progress": 0,
                "message": self._format_missing_dir_message("layout", layout_dir),
                "model_root_dir": model_root_dir,
            }

        glm_missing = [
            item for item in glm_required_files if item not in glm_existing_files
        ]
        layout_missing = [
            item for item in layout_required_files if item not in layout_existing_files
        ]
        total_required = len(glm_required_files) + len(layout_required_files)
        total_existing = len(glm_existing_files) + len(layout_existing_files)

        if not glm_missing and not layout_missing:
            return {
                "name": MODEL_NAME,
                "status": "ready",
                "progress": 100,
                "message": None,
                "model_root_dir": model_root_dir,
            }

        messages: List[str] = []
        if glm_missing:
            messages.append(self._format_missing_message("glm", glm_dir, glm_missing))
        if layout_missing:
            messages.append(
                self._format_missing_message("layout", layout_dir, layout_missing)
            )
        progress = int((total_existing / total_required) * 100) if total_required else 0
        return {
            "name": MODEL_NAME,
            "status": "not_ready",
            "progress": progress,
            "message": "; ".join(messages),
            "model_root_dir": model_root_dir,
        }

    def get_models_status(self) -> Dict[str, Dict]:
        try:
            active_download = self._task_store.latest_download_task(MODEL_KEY)
            if active_download is not None and active_download.status in {
                "queued",
                "running",
            }:
                return {
                    MODEL_KEY: {
                        "name": MODEL_NAME,
                        "status": "downloading",
                        "progress": int(active_download.progress),
                        "message": active_download.message,
                    }
                }
            return {MODEL_KEY: self._current_model_status()}
        except OCRAPIError as exc:
            return {
                MODEL_KEY: {
                    "name": MODEL_NAME,
                    "status": "error",
                    "progress": 0,
                    "message": exc.message,
                }
            }
        except Exception as exc:
            return {
                MODEL_KEY: {
                    "name": MODEL_NAME,
                    "status": "error",
                    "progress": 0,
                    "message": str(exc),
                }
            }

    def assert_models_ready(self) -> None:
        status = self.get_models_status().get(MODEL_KEY, {})
        if status.get("status") != "ready":
            detail = status.get("message") or status.get("status") or "unknown"
            raise OCRAPIError(
                status_code=409,
                code="MODEL_NOT_READY",
                message=f"{MODEL_KEY}: {detail}",
            )

    def set_model_dir(self, model_key: str, model_dir: str) -> Dict:
        self._apply_model_dir_override(model_key, model_dir)
        return self._current_model_status()

    def build_download_command(
        self,
        model_key: str,
        source: str,
        files: Optional[List[str]] = None,
    ) -> List[str]:
        target = self._glm_download_target(model_key)
        return self._build_download_command_for_target(target, source, files=files)

    def _glm_download_target(self, model_key: str) -> DownloadTarget:
        model_config = self._model_config(model_key)
        if model_config is None:
            raise OCRAPIError(
                status_code=404,
                code="MODEL_NOT_FOUND",
                message=f"Unknown model config: {model_key}",
            )

        required_files = [
            str(item).strip()
            for item in list(getattr(model_config, "required_model_files", []) or [])
            if str(item).strip()
        ]
        model_dir = self._resolve_model_dir(str(getattr(model_config, "model_dir", "")))
        return DownloadTarget(
            name="glm",
            model_dir=model_dir,
            required_files=required_files,
            download_source=str(getattr(model_config, "download_source", "modelscope")),
            modelscope_repo_id=str(getattr(model_config, "modelscope_repo_id", "")).strip(),
            huggingface_repo_id=str(getattr(model_config, "huggingface_repo_id", "")).strip(),
        )

    def _layout_download_target(self) -> DownloadTarget:
        layout_config = self._config.pipeline.layout
        model_dir = self._resolve_layout_dir(str(layout_config.model_dir))
        return DownloadTarget(
            name="layout",
            model_dir=model_dir,
            required_files=sorted(LAYOUT_REQUIRED_FILES),
            download_source=str(getattr(layout_config, "download_source", "modelscope")),
            modelscope_repo_id=str(getattr(layout_config, "modelscope_repo_id", "")).strip(),
            huggingface_repo_id=str(getattr(layout_config, "huggingface_repo_id", "")).strip(),
        )

    def _download_targets(self, model_key: str) -> List[DownloadTarget]:
        if model_key != MODEL_KEY:
            self._model_config(model_key)
        return [self._glm_download_target(model_key), self._layout_download_target()]

    @staticmethod
    def _resolve_download_source(
        requested_source: str,
        default_source: str,
    ) -> str:
        return (requested_source or default_source or "modelscope").strip()

    @staticmethod
    def _build_download_command_for_target(
        target: DownloadTarget,
        source: str,
        files: Optional[List[str]] = None,
    ) -> List[str]:
        required_files = files or list(target.required_files)
        target.model_dir.mkdir(parents=True, exist_ok=True)

        if source == "modelscope":
            repo_id = target.modelscope_repo_id
            return [
                "modelscope",
                "download",
                "--model",
                repo_id,
                *required_files,
                "--local_dir",
                str(target.model_dir),
            ]
        
        if source == "huggingface":
            repo_id = target.huggingface_repo_id
            return [
                "hf",
                "download",
                repo_id,
                *required_files,
                "--local-dir",
                str(target.model_dir),
            ]

        raise OCRAPIError(
            status_code=400,
            code="INVALID_DOWNLOAD_SOURCE",
            message=f"Unsupported download source: {source}",
        )

    def _run_download_task(
        self,
        task_id: str,
        model_key: str,
        source: str,
        request_id: str,
    ) -> None:
        try:
            targets = self._download_targets(model_key)
            total = sum(len(target.required_files) for target in targets)
            done = sum(
                1
                for target in targets
                for file_name in target.required_files
                if (target.model_dir / file_name).is_file()
            )
            self._task_store.update_task(
                task_id,
                status="running",
                stage="prepare",
                progress=int((done / total) * 100) if total else 0,
                message=(
                    f"glm dir: {targets[0].model_dir}; "
                    f"layout dir: {targets[1].model_dir}"
                ),
            )

            for target in targets:
                download_source = self._resolve_download_source(source, target.download_source)
                for file_name in target.required_files:
                    target_file = target.model_dir / file_name
                    if target_file.is_file():
                        done += 1
                        self._task_store.update_task(
                            task_id,
                            stage="downloading",
                            progress=int((done / total) * 100),
                            message=f"[{target.name}] already exists: {file_name}",
                        )
                        continue

                    command = self._build_download_command_for_target(
                        target,
                        download_source,
                        files=[file_name],
                    )
                    self._task_store.update_task(
                        task_id,
                        stage="downloading",
                        progress=int((done / total) * 100),
                        message=(
                            f"[{target.name}] downloading {file_name} via "
                            f"{download_source}"
                        ),
                    )
                    logger.info("Running download command: %s", shlex.join(command))

                    proc = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                    )
                    if proc.returncode != 0:
                        last_error = (proc.stderr or proc.stdout or "").strip()[-400:]
                        self._task_store.mark_failed(
                            task_id,
                            code="DOWNLOAD_FAILED",
                            message=(
                                f"[{target.name}] download failed for {file_name}. "
                                f"source={download_source}. detail={last_error}"
                            ),
                            request_id=request_id,
                        )
                        return

                    done += 1
                    self._task_store.update_task(
                        task_id,
                        stage="downloading",
                        progress=int((done / total) * 100),
                        message=(
                            f"[{target.name}] downloaded {file_name} via "
                            f"{download_source}"
                        ),
                    )

            missing_parts: List[str] = []
            for target in targets:
                missing = [
                    file_name
                    for file_name in target.required_files
                    if not (target.model_dir / file_name).is_file()
                ]
                if missing:
                    missing_parts.append(f"{target.name}: {', '.join(missing)}")

            if missing_parts:
                self._task_store.mark_failed(
                    task_id,
                    code="MODEL_VALIDATION_FAILED",
                    message=(
                        "missing required files after download: "
                        + "; ".join(missing_parts)
                    ),
                    request_id=request_id,
                )
                return

            glm_target = targets[0]
            layout_target = targets[1]
            self._task_store.mark_succeeded(
                task_id,
                result={
                    "model_key": model_key,
                    "source": source,
                    "model_dir": str(glm_target.model_dir),
                    "layout_model_dir": str(layout_target.model_dir),
                },
                message="model download completed",
            )
        except OCRAPIError as exc:
            self._task_store.mark_failed(
                task_id,
                code=exc.code,
                message=exc.message,
                request_id=request_id,
            )
        except Exception as exc:
            self._task_store.mark_failed(
                task_id,
                code="DOWNLOAD_FAILED",
                message=str(exc),
                request_id=request_id,
            )

    def create_download_task(
        self,
        model_key: str,
        request_id: str,
        source: Optional[str] = None,
        model_dir: Optional[str] = None,
    ) -> str:
        if self._task_store.has_active_download(model_key):
            raise OCRAPIError(
                status_code=409,
                code="DOWNLOAD_IN_PROGRESS",
                message=f"Download already running for model: {model_key}",
            )

        model_config = self._model_config(model_key)
        if model_config is None or not bool(getattr(model_config, "enabled", False)):
            raise OCRAPIError(
                status_code=404,
                code="MODEL_DISABLED",
                message=f"Model is disabled in config: {model_key}",
            )

        resolved_model_dir = self._apply_model_dir_override(model_key, model_dir)

        statuses = self.get_models_status()
        if statuses.get(model_key, {}).get("status") == "ready":
            if resolved_model_dir is None:
                raise OCRAPIError(
                    status_code=409,
                    code="MODEL_ALREADY_READY",
                    message=f"Model already ready: {model_key}",
                )
            task = self._task_store.create_task(
                task_type="download",
                request_id=request_id,
                model_key=model_key,
                task_id=f"dl_{model_key}_{self._task_store._gen_task_id('task')}",
            )
            self._task_store.mark_succeeded(
                task.task_id,
                result={
                    "model_key": model_key,
                    "source": source
                    or str(getattr(model_config, "download_source", "modelscope")),
                    "model_dir": str(
                        resolved_model_dir
                        or self._resolve_model_dir(
                            str(getattr(model_config, "model_dir", ""))
                        )
                    ),
                    "layout_model_dir": str(
                        self._resolve_layout_dir(str(self._config.pipeline.layout.model_dir))
                    ),
                },
                message="model ready",
            )
            return task.task_id

        resolved_source = source or str(
            getattr(model_config, "download_source", "modelscope")
        )
        task = self._task_store.create_task(
            task_type="download",
            request_id=request_id,
            model_key=model_key,
            task_id=f"dl_{model_key}_{self._task_store._gen_task_id('task')}",
        )
        thread = Thread(
            target=self._run_download_task,
            args=(task.task_id, model_key, resolved_source, request_id),
            daemon=True,
        )
        thread.start()
        return task.task_id
