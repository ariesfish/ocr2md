"""Local OCR pipeline.

This module provides a standalone pipeline implementation dedicated to:
- local GLM-OCR backend only
- layout enabled path only
- local file input only (single image or one PDF)
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
from uuid import uuid4

from .backend import GLMOCRBackend
from .dataloader import PageLoader
from .parser_result import PipelineResult
from .postprocess import ResultFormatter
from .utils.image_utils import crop_image_region
from .utils.logging import get_logger, get_profiler

if TYPE_CHECKING:
    from PIL import Image

    from .backend.base import BaseBackend
    from .config import Config, PipelineConfig
    from .layout.base import BaseLayoutDetector

logger = get_logger(__name__)
profiler = get_profiler(__name__)


class OCRPipeline:
    """Local OCR pipeline with layout detection."""

    def __init__(
        self,
        config: Optional[Union["PipelineConfig", "Config", Dict, str, Path]] = None,
        layout_detector: Optional["BaseLayoutDetector"] = None,
        result_formatter: Optional[ResultFormatter] = None,
    ):
        config = self._resolve_pipeline_config(config)
        self.config = config
        self.enable_layout = config.enable_layout

        if not self.enable_layout:
            raise ValueError("OCRPipeline only supports enable_layout=true.")

        glm_backend_config = config.glm_ocr_backend
        if glm_backend_config is None:
            raise ValueError("OCRPipeline requires glm_ocr_backend config.")
        if not glm_backend_config.enabled:
            raise ValueError("OCRPipeline requires glm_ocr_backend.enabled=true.")

        self.base_output_dir = str(config.output.base_output_dir)
        self.page_loader = PageLoader(config.page_loader)
        self._started = False

        self._backend_key = "glm"
        self._ocr_backends: Dict[str, "BaseBackend"] = {
            self._backend_key: GLMOCRBackend(
                glm_backend_config,
                config.page_loader,
            )
        }
        self.ocr_backend = self._ocr_backends[self._backend_key]
        logger.info("OCRPipeline enabled backend: GLM-OCR")

        if result_formatter is not None:
            self.result_formatter = result_formatter
        else:
            self.result_formatter = ResultFormatter(config.result_formatter)

        if layout_detector is not None:
            self.layout_detector = layout_detector
        else:
            from .layout import PPDocLayoutDetector

            if PPDocLayoutDetector is None:
                from .layout import _raise_layout_import_error

                _raise_layout_import_error()
            self.layout_detector = PPDocLayoutDetector(config.layout)

    @staticmethod
    def _resolve_pipeline_config(
        config: Optional[Union["PipelineConfig", "Config", Dict, str, Path]],
    ) -> "PipelineConfig":
        """Normalize constructor config input to PipelineConfig."""
        from .config import Config, PipelineConfig, load_config

        if config is None:
            return load_config().pipeline

        if isinstance(config, PipelineConfig):
            return config

        if isinstance(config, Config):
            return config.pipeline

        if isinstance(config, (str, Path)):
            return load_config(config).pipeline

        if isinstance(config, dict):
            return PipelineConfig.model_validate(config)

        raise TypeError(
            "config must be PipelineConfig, Config, dict, str/Path, or None."
        )

    def process(
        self,
        source: Union[str, Path],
        save_layout_visualization: bool = False,
        layout_vis_output_dir: Optional[str] = None,
    ) -> Generator[PipelineResult, None, None]:
        """Process one local image/PDF with the GLM OCR backend."""
        if not self._started:
            self.start()

        resolved_layout_vis_output_dir = self._resolve_layout_vis_output_dir(
            save_layout_visualization=save_layout_visualization,
            layout_vis_output_dir=layout_vis_output_dir,
        )

        source_path, pages, regions = self._load_source_and_regions(
            source=source,
            save_layout_visualization=save_layout_visualization,
            layout_vis_output_dir=resolved_layout_vis_output_dir,
        )

        recognition_results = self._run_backend_regions(self._backend_key, regions)
        grouped_results = self._group_recognition_results(
            recognition_results=recognition_results,
            page_count=len(pages),
        )

        json_result, markdown_result = self.result_formatter.process(grouped_results)

        yield PipelineResult(
            json_result=json_result,
            markdown_result=markdown_result,
            original_images=[str(source_path)],
            layout_vis_dir=resolved_layout_vis_output_dir,
            layout_image_indices=list(range(len(pages))),
        )

    def run_task(
        self,
        source: Union[str, Path],
        task_id: Optional[str] = None,
        output_root_dir: Optional[Union[str, Path]] = None,
        save_layout_visualization: bool = True,
    ) -> Dict[str, Any]:
        """Run one OCR task and persist all artifacts under one task directory.

        Directory structure:
        - {output_root_dir}/{task_id}/input
        - {output_root_dir}/{task_id}/layout
        - {output_root_dir}/{task_id}/results
        - {output_root_dir}/{task_id}/meta
        """
        source_path = self._resolve_local_source(source)
        task_id_resolved, workspace = self._prepare_task_workspace(
            source_path=source_path,
            task_id=task_id,
            output_root_dir=output_root_dir,
        )

        copied_source = workspace["input"] / source_path.name
        shutil.copy2(source_path, copied_source)

        single_result = next(
            self.process(
                str(copied_source),
                save_layout_visualization=save_layout_visualization,
                layout_vis_output_dir=str(workspace["layout"]),
            )
        )

        model_output_root = workspace["results"] / self._backend_key
        model_output_root.mkdir(parents=True, exist_ok=True)
        single_result.save(
            output_dir=str(model_output_root),
            save_layout_visualization=False,
        )
        result_dir = model_output_root / copied_source.stem
        model_artifacts = {
            self._backend_key: {
                "result_dir": str(result_dir.resolve()),
                "json": str((result_dir / f"{copied_source.stem}.json").resolve()),
                "markdown": str((result_dir / f"{copied_source.stem}.md").resolve()),
                "imgs_dir": str((result_dir / "imgs").resolve()),
            }
        }

        layout_files = self._collect_layout_files(workspace["layout"])
        manifest: Dict[str, Any] = {
            "task_id": task_id_resolved,
            "task_dir": str(workspace["task"].resolve()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {
                "original_path": str(source_path),
                "copied_path": str(copied_source.resolve()),
            },
            "layout": {
                "dir": str(workspace["layout"].resolve()),
                "files": layout_files,
            },
            "models": model_artifacts,
        }

        manifest_path = workspace["meta"] / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["manifest"] = str(manifest_path.resolve())
        return manifest

    def _resolve_output_root_dir(
        self,
        output_root_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Resolve output root directory from arg or config."""
        raw = output_root_dir if output_root_dir is not None else self.base_output_dir
        resolved = Path(raw).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def _prepare_task_workspace(
        self,
        source_path: Path,
        task_id: Optional[str],
        output_root_dir: Optional[Union[str, Path]],
    ) -> Tuple[str, Dict[str, Path]]:
        """Create one task workspace with categorized subdirectories."""
        resolved_root = self._resolve_output_root_dir(output_root_dir)

        normalized_task_id = task_id.strip() if isinstance(task_id, str) else ""
        if not normalized_task_id:
            normalized_task_id = uuid4().hex[:12]

        task_dir = resolved_root / normalized_task_id
        if task_dir.exists() and any(task_dir.iterdir()):
            normalized_task_id = f"{normalized_task_id}_{uuid4().hex[:6]}"
            task_dir = resolved_root / normalized_task_id

        input_dir = task_dir / "input"
        layout_dir = task_dir / "layout"
        results_dir = task_dir / "results"
        meta_dir = task_dir / "meta"

        for directory in (input_dir, layout_dir, results_dir, meta_dir):
            directory.mkdir(parents=True, exist_ok=True)

        source_meta = {
            "source_name": source_path.name,
            "source_suffix": source_path.suffix.lower(),
        }
        (meta_dir / "source.json").write_text(
            json.dumps(source_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return normalized_task_id, {
            "task": task_dir,
            "input": input_dir,
            "layout": layout_dir,
            "results": results_dir,
            "meta": meta_dir,
        }

    @staticmethod
    def _collect_layout_files(layout_dir: Path) -> List[str]:
        """Collect layout overlay image file paths from layout directory."""
        files: List[Path] = []
        files.extend(sorted(layout_dir.glob("layout_page*.jpg")))
        files.extend(sorted(layout_dir.glob("layout_page*.png")))
        return [str(path.resolve()) for path in files]

    def _load_source_and_regions(
        self,
        source: Union[str, Path],
        save_layout_visualization: bool,
        layout_vis_output_dir: Optional[str],
    ) -> Tuple[Path, List["Image.Image"], List[Tuple["Image.Image", Dict, str, int]]]:
        """Load source, run shared layout detection once, and crop regions once."""
        source_path = self._resolve_local_source(source)
        pages, _ = self.page_loader.load_pages_with_unit_indices([str(source_path)])

        if not pages:
            raise RuntimeError(f"No pages loaded from source: {source_path}")

        suffix = source_path.suffix.lower()
        if suffix != ".pdf" and len(pages) != 1:
            raise ValueError("Only a single local image file is supported.")

        layout_results = self._run_layout_detection(
            pages=pages,
            save_layout_visualization=save_layout_visualization,
            layout_vis_output_dir=layout_vis_output_dir,
        )
        regions = self._prepare_regions(pages, layout_results)
        return source_path, pages, regions

    def _resolve_layout_vis_output_dir(
        self,
        save_layout_visualization: bool,
        layout_vis_output_dir: Optional[str],
    ) -> Optional[str]:
        """Resolve output directory for layout overlay visualization images."""
        if not save_layout_visualization:
            return None

        if layout_vis_output_dir:
            resolved = Path(layout_vis_output_dir).expanduser().resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            return str(resolved)

        temp_dir = tempfile.mkdtemp(prefix="ocr2md_layout_vis_")
        logger.debug(
            "layout visualization output dir is not provided; using temp dir: %s",
            temp_dir,
        )
        return temp_dir

    def _run_layout_detection(
        self,
        pages: List["Image.Image"],
        save_layout_visualization: bool,
        layout_vis_output_dir: Optional[str],
        global_start_idx: int = 0,
    ) -> List[List[Dict]]:
        """Run layout detection by detector batch size."""
        layout_results: List[List[Dict]] = []
        batch_size = int(getattr(self.layout_detector, "batch_size", 1) or 1)

        for start_idx in range(0, len(pages), batch_size):
            batch_pages = pages[start_idx : start_idx + batch_size]
            batch_layout = cast(Any, self.layout_detector).process(
                batch_pages,
                save_visualization=save_layout_visualization
                and layout_vis_output_dir is not None,
                visualization_output_dir=layout_vis_output_dir,
                global_start_idx=global_start_idx + start_idx,
            )
            layout_results.extend(batch_layout)

        return layout_results

    def _resolve_local_source(self, source: Union[str, Path]) -> Path:
        """Resolve one local image/PDF source path."""
        if not isinstance(source, (str, Path)):
            raise TypeError("source must be a str or Path.")

        raw_path = str(source)

        if raw_path.startswith("file://"):
            raw_path = raw_path[7:]

        if raw_path.startswith(("http://", "https://", "data:")):
            raise ValueError("Only local file input is supported in this pipeline.")

        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {path}")

        allowed_suffix = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".gif",
            ".webp",
            ".pdf",
        }
        if path.suffix.lower() not in allowed_suffix:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        return path

    def _prepare_regions(
        self,
        pages: List["Image.Image"],
        layout_results: List[List[Dict]],
    ) -> List[Tuple["Image.Image", Dict, str, int]]:
        """Prepare regions that need recognition."""
        regions: List[Tuple["Image.Image", Dict, str, int]] = []
        with profiler.measure("crop_regions"):
            for page_idx, (page, layouts) in enumerate(zip(pages, layout_results)):
                for region in layouts:
                    cropped = crop_image_region(
                        page,
                        region["bbox_2d"],
                        region.get("polygon"),
                    )
                    regions.append((cropped, region, region["task_type"], page_idx))
        return regions

    def _run_backend_regions(
        self,
        backend_key: str,
        regions: List[Tuple["Image.Image", Dict, str, int]],
    ) -> List[Tuple[int, Dict]]:
        """Run OCR over regions for one backend in serial order."""
        backend = self._ocr_backends[backend_key]
        results: List[Tuple[int, Dict]] = []

        for img, info, task, page_idx in regions:
            region_info = deepcopy(info)

            if task == "skip":
                region_info["content"] = None
                results.append((page_idx, region_info))
                continue

            try:
                response, status_code = backend.process(img, task)
                if status_code == 200:
                    region_info["content"] = self._extract_content_from_response(
                        response
                    )
                else:
                    region_info["content"] = ""
            except Exception as e:
                logger.warning(
                    "Recognition failed (backend=%s): %s",
                    backend_key,
                    e,
                )
                if not backend.is_started:
                    raise RuntimeError(
                        f"{backend_key} backend startup failed: {e}"
                    ) from e
                region_info["content"] = ""

            results.append((page_idx, region_info))

        return results

    def _group_recognition_results(
        self,
        recognition_results: List[Tuple[int, Dict]],
        page_count: int,
    ) -> List[List[Dict]]:
        grouped_results: List[List[Dict]] = [[] for _ in range(page_count)]
        for page_idx, region_info in recognition_results:
            grouped_results[page_idx].append(region_info)
        return grouped_results

    @staticmethod
    def _extract_content_from_response(response: Dict) -> str:
        """Extract OCR content from OpenAI-style response payload."""
        try:
            return str(response["choices"][0]["message"]["content"]).strip()
        except Exception:
            return ""

    def start(self):
        """Start the pipeline."""
        if self._started:
            return

        logger.info("Starting OCRPipeline...")
        self.layout_detector.start()

        self._started = True
        logger.info("OCRPipeline started! backends=%s", self._backend_key)

    def stop(self):
        """Stop the pipeline."""
        if not self._started:
            return

        logger.info("Stopping OCRPipeline...")

        for backend in self._ocr_backends.values():
            backend.stop()

        self.layout_detector.stop()
        self._started = False
        logger.info("OCRPipeline stopped!")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
