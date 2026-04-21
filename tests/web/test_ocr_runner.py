from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from ocr2md.config import Config
from ocr2md.web.services.ocr_runner import PipelineOCRRunner


def test_run_ocr_streams_pdf_pages(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    callback_payloads: list[dict] = []
    progress_events: list[tuple[int, str, str | None]] = []
    seen_page_indices: list[int] = []

    class _FakePageLoader:
        def iter_pages_with_unit_indices(
            self,
            sources: list[str],
            page_indices: list[int] | None = None,
        ):
            assert sources == [str(pdf_path)]
            seen_page_indices.extend(page_indices or [])
            yield Image.new("RGB", (100, 200), color="white"), 0
            yield Image.new("RGB", (120, 240), color="white"), 0

    class _FakePipeline:
        def __init__(self) -> None:
            self.page_loader = _FakePageLoader()
            self.result_formatter = SimpleNamespace(process=self._format_result)

        @staticmethod
        def _format_result(grouped_results):
            text_parts: list[str] = []
            for page in grouped_results:
                for item in page:
                    content = item.get("content")
                    if isinstance(content, str) and content.strip():
                        text_parts.append(content.strip())
            return json.dumps(grouped_results), "\n".join(text_parts)

        def _load_source_and_regions(self, **kwargs):
            raise AssertionError("streaming path should not load all PDF pages at once")

        def _run_layout_detection(
            self,
            pages,
            save_layout_visualization,
            layout_vis_output_dir,
            global_start_idx=0,
        ):
            del pages, save_layout_visualization, layout_vis_output_dir
            return [
                [
                    {
                        "label": "text",
                        "bbox_2d": [0, 0, 100, 100],
                        "score": 0.9,
                        "polygon": None,
                        "task_type": "text",
                        "index": global_start_idx,
                    }
                ]
            ]

        def _prepare_regions(self, pages, layout_results):
            return [(pages[0], layout_results[0][0], "text", 0)]

        def _run_backend_regions(self, backend_key: str, regions):
            assert backend_key == "glm"
            page_idx = regions[0][3]
            item = dict(regions[0][1])
            item["content"] = f"page {page_idx + 1}"
            return [(page_idx, item)]

    runner = PipelineOCRRunner(Config())
    fake_pipeline = _FakePipeline()
    monkeypatch.setattr(runner, "_ensure_pipeline", lambda: fake_pipeline)
    monkeypatch.setattr(
        runner,
        "_resolve_source_page_indices",
        lambda pipeline, source_path, selected_pages=None: [0, 1],
    )

    result = runner.run_ocr(
        source_path=pdf_path,
        request_id="req_pdf",
        persist_artifacts=False,
        progress_callback=lambda progress, stage, message: progress_events.append(
            (progress, stage, message)
        ),
        model_result_callback=lambda model_key,
        model_result,
        input_meta: callback_payloads.append(
            {
                "model_key": model_key,
                "model_result": model_result,
                "input_meta": input_meta,
            }
        ),
    )

    assert result["input"] == {
        "filename": "demo.pdf",
        "width": 100,
        "height": 200,
        "page_count": 2,
        "source_pages": [1, 2],
    }
    assert result["glm"]["text"] == "page 1\npage 2"
    assert len(callback_payloads) >= 2
    assert callback_payloads[0]["model_result"]["text"] == "page 1"
    assert callback_payloads[-1]["model_result"]["text"] == "page 1\npage 2"
    assert callback_payloads[0]["input_meta"]["page_count"] == 2
    assert callback_payloads[0]["input_meta"]["source_pages"] == [1, 2]
    assert seen_page_indices == [0, 1]
    assert any(stage == "layout_detection" for _, stage, _ in progress_events)
    assert any(stage == "glm_inference" for _, stage, _ in progress_events)


def test_run_ocr_uses_selected_pdf_pages(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    seen_page_indices: list[int] = []

    class _FakePageLoader:
        def iter_pages_with_unit_indices(
            self,
            sources: list[str],
            page_indices: list[int] | None = None,
        ):
            assert sources == [str(pdf_path)]
            seen_page_indices.extend(page_indices or [])
            for _ in page_indices or []:
                yield Image.new("RGB", (100, 200), color="white"), 0

    class _FakePipeline:
        def __init__(self) -> None:
            self.page_loader = _FakePageLoader()
            self.result_formatter = SimpleNamespace(process=lambda grouped_results: (json.dumps(grouped_results), ""))

        def _run_layout_detection(self, pages, save_layout_visualization, layout_vis_output_dir, global_start_idx=0):
            del pages, save_layout_visualization, layout_vis_output_dir
            return [[{"label": "text", "bbox_2d": [0, 0, 100, 100], "score": 0.9, "polygon": None, "task_type": "text", "index": global_start_idx}]]

        def _prepare_regions(self, pages, layout_results):
            return [(pages[0], layout_results[0][0], "text", 0)]

        def _run_backend_regions(self, backend_key: str, regions):
            del backend_key
            return [(regions[0][3], dict(regions[0][1], content="ok"))]

    runner = PipelineOCRRunner(Config())
    fake_pipeline = _FakePipeline()
    monkeypatch.setattr(runner, "_ensure_pipeline", lambda: fake_pipeline)
    monkeypatch.setattr(
        runner,
        "_resolve_source_page_indices",
        lambda pipeline, source_path, selected_pages=None: selected_pages or [0, 1, 2],
    )

    result = runner.run_ocr(
        source_path=pdf_path,
        request_id="req_pdf_selected",
        persist_artifacts=False,
        selected_pages=[1, 3],
    )

    assert seen_page_indices == [1, 3]
    assert result["input"]["source_pages"] == [2, 4]
    assert result["input"]["page_count"] == 2
