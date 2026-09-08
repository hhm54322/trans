#!/usr/bin/env python3
"""Benchmark CAD source-reading plus one page-level text translation request.

This is an isolated call-organization experiment. It does not export a PDF,
change routing, or write document history. Vision keeps the same indexed crops
and coordinates; only translation moves out of the vision response.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app import main as application
from api.app.services.documents import DocumentSegment
from api.app.services.translator import DemoProvider
from api.app.services.visual_pdf import (
    extract_native_page_units,
    prepare_dense_cad_translation_sheets,
)


DEFAULT_PDF = ROOT / "Attach_TOR_1_260805_224430.pdf"
DEFAULT_BASELINE = ROOT / "output" / "vision-benchmarks" / "cad-resolution-page-7.json"
OUTPUT_DIRECTORY = ROOT / "output" / "call-organization-benchmarks"
THAI_PATTERN = re.compile(r"[\u0E00-\u0E7F]")


def _thai_key(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value or "")
    return "".join(character for character in normalized if THAI_PATTERN.fullmatch(character))


def _assign_global_ids(sheets: List[Dict[str, Any]]) -> List[str]:
    identifiers = []
    for sheet in sheets:
        for item_id, candidate in (sheet.get("entries") or {}).items():
            global_id = f"C{len(identifiers) + 1:03d}"
            candidate["benchmark_global_id"] = global_id
            candidate["benchmark_sheet_item_id"] = item_id
            identifiers.append(global_id)
    return identifiers


async def _read_sheet(
    sheet_number: int,
    sheet: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    total_sheets: int,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    expected_sources = {
        item_id: str(candidate.get("source_hint") or "")
        for item_id, candidate in (sheet.get("entries") or {}).items()
        if str(candidate.get("source_hint") or "").strip()
    }
    expected_ids = list((sheet.get("entries") or {}).keys())
    started_at = time.perf_counter()
    async with semaphore:
        items, route = await application.translator.provider.read_indexed_image_lines(
            sheet["content"],
            "image/png",
            "仅识别图片中的泰文 CAD 原文，后续会由独立文本模型整页翻译。",
            expected_sources=expected_sources,
        )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    returned: Dict[str, str] = {}
    for item in items:
        item_id = str(item.get("id") or "")
        source_text = str(item.get("source_text") or "").strip()
        if not item_id or item_id in returned or not source_text:
            raise RuntimeError(f"CAD 识字返回无效或重复 ID：{item_id or '(empty)'}")
        returned[item_id] = source_text
    expected = set(expected_ids)
    if set(returned) != expected:
        missing = sorted(expected - set(returned))
        unknown = sorted(set(returned) - expected)
        raise RuntimeError(
            f"ID_MISMATCH: 索引图 {sheet_number} 缺少 {missing[:8]}，多出 {unknown[:8]}"
        )
    source_by_global_id = {
        str(candidate["benchmark_global_id"]): returned[item_id]
        for item_id, candidate in (sheet.get("entries") or {}).items()
    }
    print(
        f"视觉识字 {sheet_number}/{total_sheets}: {elapsed_ms / 1000:.2f}s",
        flush=True,
    )
    return source_by_global_id, {
        "sheet_number": sheet_number,
        "entry_count": len(expected_ids),
        "image_bytes": len(sheet.get("content") or b""),
        "elapsed_ms": elapsed_ms,
        "route": route,
    }


async def _run(arguments: argparse.Namespace) -> Dict[str, Any]:
    if not arguments.pdf.is_file():
        raise ValueError("未找到待测 PDF")
    if isinstance(application.translator.provider, DemoProvider):
        raise ValueError("未配置真实模型，无法执行 CAD 调用组织基准")
    document = fitz.open(arguments.pdf)
    try:
        if not 1 <= arguments.page <= document.page_count:
            raise ValueError("页码超出范围")
        page = document[arguments.page - 1]
        native_units = extract_native_page_units(page, arguments.page)
        build_started_at = time.perf_counter()
        sheets = prepare_dense_cad_translation_sheets(
            page,
            arguments.page,
            desired_width=arguments.desired_width,
            rows_per_sheet=arguments.rows_per_sheet,
            native_units=native_units,
            detection_provider="tesseract",
        )
        build_elapsed_ms = round((time.perf_counter() - build_started_at) * 1000)
    finally:
        document.close()
    sheets = [sheet for sheet in sheets if sheet.get("entries")]
    identifiers = _assign_global_ids(sheets)
    if not identifiers:
        raise RuntimeError("未生成 CAD 文字候选")

    http_events: List[Dict[str, Any]] = []
    original_post = application.translator.provider._post

    async def observed_post(path: str, payload: Dict[str, Any]):
        started_at = time.perf_counter()
        input_value = payload.get("input")
        try:
            response = await original_post(path, payload)
        except BaseException as exc:
            http_events.append(
                {
                    "input_kind": "image" if isinstance(input_value, list) else "text",
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000),
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
            )
            raise
        http_events.append(
            {
                "input_kind": "image" if isinstance(input_value, list) else "text",
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000),
                "status": "ok",
                "response_status": response.status_code,
            }
        )
        return response

    application.translator.provider._post = observed_post
    total_started_at = time.perf_counter()
    try:
        vision_started_at = time.perf_counter()
        semaphore = asyncio.Semaphore(arguments.visual_concurrency)
        read_results = await asyncio.gather(
            *(
                _read_sheet(number, sheet, semaphore, len(sheets))
                for number, sheet in enumerate(sheets, start=1)
            )
        )
        vision_elapsed_ms = round((time.perf_counter() - vision_started_at) * 1000)
        source_by_id: Dict[str, str] = {}
        sheet_details = []
        for result, detail in read_results:
            source_by_id.update(result)
            sheet_details.append(detail)
        if set(source_by_id) != set(identifiers):
            raise RuntimeError("ID_MISMATCH: 视觉识字汇总与候选 ID 不一致")

        source_segments = [
            DocumentSegment(
                segment_id=f"cad:p{arguments.page}:{identifier}",
                page_number=arguments.page,
                text=source_by_id[identifier],
                source_kind="outline-text",
            )
            for identifier in identifiers
            if _thai_key(source_by_id[identifier])
        ]
        if not source_segments:
            raise RuntimeError("视觉识字未返回任何泰文")
        application.database.initialize()
        text_started_at = time.perf_counter()
        translations, providers, warnings = await application._translate_document_segment_batch(
            source_segments,
            "th",
            "zh",
            "",
            asyncio.Semaphore(1),
        )
        text_elapsed_ms = round((time.perf_counter() - text_started_at) * 1000)
        expected_translation_ids = {segment.segment_id for segment in source_segments}
        if set(translations) != expected_translation_ids:
            raise RuntimeError("ID_MISMATCH: 页面文本翻译返回的 ID 不完整")
    finally:
        application.translator.provider._post = original_post

    baseline_summary: Dict[str, Any] = {}
    if arguments.baseline.is_file():
        baseline = json.loads(arguments.baseline.read_text(encoding="utf-8"))
        baseline_by_id = {
            str(item.get("id") or ""): str(item.get("full_text") or "")
            for item in baseline.get("comparisons") or []
        }
        compared = [
            identifier
            for identifier in identifiers
            if _thai_key(baseline_by_id.get(identifier, ""))
        ]
        changed = [
            identifier
            for identifier in compared
            if _thai_key(baseline_by_id[identifier]) != _thai_key(source_by_id[identifier])
        ]
        baseline_summary = {
            "baseline_file": str(arguments.baseline),
            "baseline_thai_count": len(compared),
            "exact_thai_match_count": len(compared) - len(changed),
            "changed_thai_reading_count": len(changed),
        }

    total_elapsed_ms = round((time.perf_counter() - total_started_at) * 1000)
    return {
        "configuration": {
            "source_file": arguments.pdf.name,
            "source_page": arguments.page,
            "rows_per_sheet": arguments.rows_per_sheet,
            "desired_width": arguments.desired_width,
            "visual_concurrency": arguments.visual_concurrency,
        },
        "outcome": {
            "status": "completed",
            "candidate_count": len(identifiers),
            "thai_source_count": len(source_segments),
            "translated_count": len(translations),
            "translation_provider": "+".join(providers),
            "translation_warning_count": len(warnings),
        },
        "timing": {
            "indexed_sheet_build_ms": build_elapsed_ms,
            "vision_source_read_ms": vision_elapsed_ms,
            "page_text_translation_ms": text_elapsed_ms,
            "total_after_sheet_build_ms": total_elapsed_ms,
        },
        "vision_sheets": sheet_details,
        "http_summary": {
            "request_count": len(http_events),
            "image_request_count": sum(item["input_kind"] == "image" for item in http_events),
            "text_request_count": sum(item["input_kind"] == "text" for item in http_events),
            "error_count": sum(item["status"] == "error" for item in http_events),
            "sum_elapsed_ms": sum(item["elapsed_ms"] for item in http_events),
        },
        "baseline_comparison": baseline_summary,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--page", type=int, default=7)
    parser.add_argument("--rows-per-sheet", type=int, default=12)
    parser.add_argument("--desired-width", type=int, default=4000)
    parser.add_argument("--visual-concurrency", type=int, default=5)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if arguments.rows_per_sheet < 1 or arguments.visual_concurrency < 1:
        raise SystemExit("rows-per-sheet 和 visual-concurrency 必须大于 0")
    print(
        f"cad_page_level_started page={arguments.page} rows={arguments.rows_per_sheet} "
        f"concurrency={arguments.visual_concurrency}",
        flush=True,
    )
    try:
        report = asyncio.run(_run(arguments))
    except BaseException as exc:
        print(f"cad_page_level_failed error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIRECTORY / f"cad-page-level-page-{arguments.page}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    timing = report["timing"]
    outcome = report["outcome"]
    print(
        f"完成：候选 {outcome['candidate_count']}，泰文 {outcome['thai_source_count']}，"
        f"视觉识字 {timing['vision_source_read_ms'] / 1000:.2f}s，"
        f"整页文本翻译 {timing['page_text_translation_ms'] / 1000:.2f}s，"
        f"报告={destination}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
