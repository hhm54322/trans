#!/usr/bin/env python3
"""Profile the current legacy dense-CAD PDF pipeline for one source page.

This diagnostic intentionally calls the production PDF pipeline unchanged. It
only wraps selected in-process functions to emit elapsed time and counts. No
translated document is exported, and neither source text nor model output is
written to the report.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import os
import resource
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import fitz


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app import main as application
from api.app.services.exports import create_document_export
from api.app.services import documents, visual_pdf


DEFAULT_SOURCE = ROOT / "Attach_TOR_1_260805_224430.pdf"
OUTPUT_DIRECTORY = ROOT / "output" / "performance-profiles"
_CURRENT_PHASE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "legacy_cad_phase", default="unattributed"
)


class Timeline:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(
        self,
        stage: str,
        started_at: float,
        *,
        status: str = "ok",
        **metadata: Any,
    ) -> None:
        finished_at = time.perf_counter()
        event = {
            "stage": stage,
            "started_ms": round((started_at - self.started_at) * 1000),
            "elapsed_ms": round((finished_at - started_at) * 1000),
            "status": status,
        }
        event.update({key: value for key, value in metadata.items() if value is not None})
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return sorted(self._events, key=lambda item: (item["started_ms"], item["stage"]))

    def aggregates(self) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in self.events:
            groups[event["stage"]].append(event)
        output = []
        for stage, events in sorted(groups.items()):
            elapsed = [int(event["elapsed_ms"]) for event in events]
            output.append(
                {
                    "stage": stage,
                    "call_count": len(events),
                    "sum_elapsed_ms": sum(elapsed),
                    "max_elapsed_ms": max(elapsed),
                    "min_elapsed_ms": min(elapsed),
                }
            )
        return output


class Patches:
    def __init__(self) -> None:
        self._entries: List[Tuple[object, str, Any]] = []

    def set(self, target: object, name: str, replacement: Any) -> None:
        self._entries.append((target, name, getattr(target, name)))
        setattr(target, name, replacement)

    def restore(self) -> None:
        for target, name, original in reversed(self._entries):
            setattr(target, name, original)


def _result_counts(value: Any) -> Dict[str, int]:
    if isinstance(value, list):
        counts = {"result_count": len(value)}
        if value and isinstance(value[0], dict) and "entries" in value[0]:
            counts["sheet_count"] = len(value)
            counts["candidate_count"] = sum(
                len(item.get("entries") or {}) for item in value if isinstance(item, dict)
            )
        return counts
    if isinstance(value, tuple):
        return {"result_count": len(value)}
    if isinstance(value, dict):
        return {"result_count": len(value)}
    return {}


def _sync_wrapper(timeline: Timeline, stage: str, original: Callable[..., Any]):
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            timeline.record(stage, started_at, status="error", error_type=type(exc).__name__)
            raise
        timeline.record(stage, started_at, **_result_counts(result))
        return result

    return wrapped


def _async_wrapper(timeline: Timeline, stage: str, original: Callable[..., Any], details=None):
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        token = _CURRENT_PHASE.set(stage)
        try:
            result = await original(*args, **kwargs)
        except BaseException as exc:
            timeline.record(stage, started_at, status="error", error_type=type(exc).__name__)
            raise
        finally:
            _CURRENT_PHASE.reset(token)
        metadata = details(args, kwargs, result) if details else _result_counts(result)
        timeline.record(stage, started_at, **metadata)
        return result

    return wrapped


def _iter_wrapper(timeline: Timeline, original: Callable[..., Iterable[Any]]):
    def wrapped(*args: Any, **kwargs: Any):
        started_at = time.perf_counter()
        count = 0
        try:
            for item in original(*args, **kwargs):
                count += 1
                yield item
        except BaseException as exc:
            timeline.record("native_pdf_extract_and_route", started_at, status="error", error_type=type(exc).__name__)
            raise
        timeline.record("native_pdf_extract_and_route", started_at, page_count=count)

    return wrapped


def _indexed_details(args: Tuple[Any, ...], _kwargs: Dict[str, Any], result: Any) -> Dict[str, int]:
    expected_ids = args[2] if len(args) > 2 else []
    items = result[0] if isinstance(result, tuple) and result else []
    return {"expected_ids": len(expected_ids), "returned_items": len(items)}


def _segment_details(args: Tuple[Any, ...], _kwargs: Dict[str, Any], _result: Any) -> Dict[str, int]:
    segments = args[0] if args else []
    return {
        "segment_count": len(segments),
        "source_characters": sum(len(str(getattr(item, "text", ""))) for item in segments),
    }


def _payload_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    input_value = payload.get("input")
    input_kind = "image" if isinstance(input_value, list) else "text"
    try:
        request_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        request_bytes = None
    return {
        "phase": _CURRENT_PHASE.get(),
        "endpoint_model": payload.get("model"),
        "input_kind": input_kind,
        "request_bytes": request_bytes,
    }


def _model_http_summary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    requests = [event for event in events if event["stage"] == "model_http_request"]
    if not requests:
        return {"request_count": 0, "actual_max_concurrency": 0, "by_phase": []}

    boundaries: List[Tuple[int, int]] = []
    for event in requests:
        started = int(event["started_ms"])
        boundaries.append((started, 1))
        boundaries.append((started + int(event["elapsed_ms"]), -1))
    # Starts must be applied before finishes at an identical millisecond so a
    # back-to-back connection is not reported as zero concurrent requests.
    current = 0
    maximum = 0
    for _at_ms, delta in sorted(boundaries, key=lambda item: (item[0], -item[1])):
        current += delta
        maximum = max(maximum, current)

    by_phase = []
    for phase in sorted({str(event.get("phase") or "unattributed") for event in requests}):
        phase_requests = [event for event in requests if str(event.get("phase") or "unattributed") == phase]
        started = min(int(event["started_ms"]) for event in phase_requests)
        finished = max(int(event["started_ms"]) + int(event["elapsed_ms"]) for event in phase_requests)
        by_phase.append(
            {
                "phase": phase,
                "request_count": len(phase_requests),
                "wall_elapsed_ms": finished - started,
                "sum_elapsed_ms": sum(int(event["elapsed_ms"]) for event in phase_requests),
                "error_count": sum(event.get("status") == "error" for event in phase_requests),
                "request_bytes": sum(int(event.get("request_bytes") or 0) for event in phase_requests),
            }
        )
    return {
        "request_count": len(requests),
        "actual_max_concurrency": maximum,
        "by_phase": by_phase,
    }


def _profile_page(
    source: Path,
    source_page: int,
    *,
    rows_per_sheet: int,
    vision_concurrency: int,
    label: str,
) -> Tuple[Dict[str, Any], Timeline]:
    if not source.exists():
        raise FileNotFoundError(f"未找到测试 PDF：{source}")
    with fitz.open(source) as document:
        if source_page < 1 or source_page > document.page_count:
            raise ValueError(f"页码超出范围：{source_page}，文件共 {document.page_count} 页")
        isolated = fitz.open()
        try:
            isolated.insert_pdf(document, from_page=source_page - 1, to_page=source_page - 1)
            content = isolated.tobytes(garbage=0, deflate=False)
        finally:
            isolated.close()

    timeline = Timeline()
    patches = Patches()
    progress: List[Dict[str, Any]] = []

    def report_progress(completed_pages: int, message: str) -> None:
        progress.append(
            {
                "at_ms": round((time.perf_counter() - timeline.started_at) * 1000),
                "completed_pages": completed_pages,
                "message": message,
            }
        )

    original_iter_pages = application.iter_pdf_pages
    original_tesseract = documents.ocr_image_text_blocks
    original_provider_post = application.translator.provider._post
    # These values are patched only inside this isolated diagnostic process.
    # They are restored before the report is written, so app defaults remain
    # untouched for local or deployed requests.
    patches.set(application, "CAD_PADDLE_DETECTOR_ROWS_PER_SHEET", rows_per_sheet)
    patches.set(application, "CAD_INDEXED_MODEL_CONCURRENCY", vision_concurrency)
    patches.set(application, "iter_pdf_pages", _iter_wrapper(timeline, original_iter_pages))
    patches.set(
        visual_pdf,
        "_dense_tesseract_seed_candidates",
        _sync_wrapper(timeline, "local_tesseract_candidate_detection", visual_pdf._dense_tesseract_seed_candidates),
    )
    patches.set(
        documents,
        "ocr_image_text_blocks",
        _sync_wrapper(timeline, "tesseract_single_pass", original_tesseract),
    )
    for name, stage in [
        ("prepare_dense_cad_translation_sheets", "indexed_sheet_render_and_build"),
        ("subset_indexed_translation_sheet", "indexed_sheet_subset_build"),
        ("prepare_dense_cad_review_sheets", "high_resolution_review_sheet_build"),
        ("recover_dense_cad_review_text_lines", "local_paddle_targeted_recovery"),
        ("detect_dense_cad_paddle_candidates", "full_page_paddle_supplement_detection"),
        ("prepare_native_pdf_source", "native_pdf_prepare_for_export"),
    ]:
        patches.set(application, name, _sync_wrapper(timeline, stage, getattr(application, name)))
    patches.set(
        application,
        "_translate_document_segment_batch",
        _async_wrapper(
            timeline,
            "native_text_model_translation",
            application._translate_document_segment_batch,
            _segment_details,
        ),
    )
    patches.set(
        application,
        "_translate_document_segments",
        _async_wrapper(
            timeline,
            "local_recovery_text_model_translation",
            application._translate_document_segments,
            _segment_details,
        ),
    )
    patches.set(
        application.translator,
        "translate_indexed_image_lines",
        _async_wrapper(
            timeline,
            "indexed_vision_read_and_translate",
            application.translator.translate_indexed_image_lines,
            _indexed_details,
        ),
    )
    patches.set(
        application.translator,
        "read_indexed_image_lines",
        _async_wrapper(
            timeline,
            "indexed_vision_source_read",
            application.translator.read_indexed_image_lines,
            _indexed_details,
        ),
    )

    async def observed_post(path: str, payload: Dict[str, Any]):
        started_at = time.perf_counter()
        metadata = _payload_metadata(payload)
        try:
            response = await original_provider_post(path, payload)
        except BaseException as exc:
            timeline.record("model_http_request", started_at, status="error", error_type=type(exc).__name__, path=path, **metadata)
            raise
        timeline.record(
            "model_http_request",
            started_at,
            path=path,
            response_status=response.status_code,
            **metadata,
        )
        return response

    patches.set(application.translator.provider, "_post", observed_post)
    configuration = {
        "cad_ocr_mode": os.getenv("APP_CAD_OCR_MODE", "auto"),
        "pdf_ocr_concurrency": application.settings.pdf_ocr_concurrency,
        "rows_per_sheet": rows_per_sheet,
        "vision_concurrency": vision_concurrency,
        "model": application.settings.openai_model,
        "vision_model": application.settings.openai_vision_model,
    }
    started_at = time.perf_counter()
    pipeline_failure = None
    try:
        application.database.initialize()
        source_text, result = asyncio.run(
            application._translate_pdf_document_pipeline(
                content,
                f"{source.stem}-source-page-{source_page}.pdf",
                1,
                "th",
                "zh",
                "",
                report_progress,
            )
        )
        export_started_at = time.perf_counter()
        export_directory = OUTPUT_DIRECTORY / "exports"
        exported = create_document_export(
            item_id=(
                f"profile-page-{source_page}-rows-{rows_per_sheet}"
                f"-concurrency-{vision_concurrency}-{label}"
            ),
            source_filename=f"{source.stem}-source-page-{source_page}.pdf",
            source_content=content,
            translated_text=result.translated_text,
            target_language="zh",
            output_directory=export_directory,
            layout_segments=result.layout_segments,
            prepared_pdf_content=result.prepared_pdf_content,
        )
        timeline.record(
            "formatted_pdf_export",
            export_started_at,
            output_bytes=exported.path.stat().st_size,
            output_path=str(exported.path),
        )
        outcome = {
            "status": "completed",
            "layout_segment_count": len(result.layout_segments),
            "source_character_count": len(source_text),
            "provider": result.provider,
            "warning_count": len(result.warnings),
            "formatted_pdf_export": str(exported.path),
        }
    except BaseException as exc:
        outcome = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        pipeline_failure = exc
    finally:
        total_elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        patches.restore()

    report = {
        "source_file": source.name,
        "source_page": source_page,
        "isolated_page_bytes": len(content),
        "legacy_configuration": configuration,
        "outcome": outcome,
        "total_elapsed_ms": total_elapsed_ms,
        "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "progress": progress,
        "stage_aggregates": timeline.aggregates(),
        "events": timeline.events,
    }
    report["model_http_summary"] = _model_http_summary(timeline.events)
    if pipeline_failure is not None:
        report["pipeline_failure"] = True
    return report, timeline


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--page", type=int, default=7)
    parser.add_argument("--rows-per-sheet", type=int, default=6)
    parser.add_argument("--vision-concurrency", type=int, default=2)
    parser.add_argument(
        "--label",
        default="run",
        help="本次基准的 ASCII 标识，避免覆盖既有结果",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if arguments.rows_per_sheet < 1:
        raise SystemExit("--rows-per-sheet 必须大于 0")
    if arguments.vision_concurrency < 1:
        raise SystemExit("--vision-concurrency 必须大于 0")
    if not arguments.label.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("--label 仅支持字母、数字、连字符和下划线")
    print(
        "legacy_cad_profile_started "
        f"source={arguments.source.name} source_page={arguments.page} "
        f"rows_per_sheet={arguments.rows_per_sheet} "
        f"vision_concurrency={arguments.vision_concurrency}",
        flush=True,
    )
    try:
        report, _timeline = _profile_page(
            arguments.source,
            arguments.page,
            rows_per_sheet=arguments.rows_per_sheet,
            vision_concurrency=arguments.vision_concurrency,
            label=arguments.label,
        )
    except BaseException as exc:
        print(f"legacy_cad_profile_failed error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIRECTORY / (
        f"legacy-cad-page-{arguments.page}-rows-{arguments.rows_per_sheet}"
        f"-concurrency-{arguments.vision_concurrency}-{arguments.label}.json"
    )
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"legacy_cad_profile_completed elapsed_s={report['total_elapsed_ms'] / 1000:.2f} "
        f"report={destination}",
        flush=True,
    )
    for aggregate in report["stage_aggregates"]:
        print(
            f"stage={aggregate['stage']} calls={aggregate['call_count']} "
            f"sum_s={aggregate['sum_elapsed_ms'] / 1000:.2f} "
            f"max_s={aggregate['max_elapsed_ms'] / 1000:.2f}",
            flush=True,
        )
    return 0 if report["outcome"]["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
