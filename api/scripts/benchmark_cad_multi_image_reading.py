#!/usr/bin/env python3
"""Benchmark full-resolution CAD index sheets grouped into fewer HTTP calls.

The experiment does not alter production routing or export a document. Every
source sheet keeps the production 1800px width, 120px row height and 12 rows;
only the number of images carried by one model request changes.
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
from typing import Any, Dict, List, Sequence, Tuple

import fitz


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app import main as application
from api.app.services.translator import DemoProvider
from api.app.services.visual_pdf import (
    _overlap_smaller,
    extract_native_page_units,
    prepare_dense_cad_translation_sheets,
)


DEFAULT_PDF = ROOT / "Attach_TOR_1_260805_224430.pdf"
DEFAULT_BASELINE = ROOT / "output" / "vision-benchmarks" / "cad-resolution-page-7.json"
DEFAULT_ALIYUN_REPORT = (
    ROOT / "output" / "ocr-benchmarks" / "aliyun-thai-ocr-benchmark.json"
)
OUTPUT_DIRECTORY = ROOT / "output" / "vision-benchmarks"
THAI_PATTERN = re.compile(r"[\u0E00-\u0E7F]")


def _thai_key(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value or "")
    return "".join(character for character in normalized if THAI_PATTERN.fullmatch(character))


def _groups(values: Sequence[Dict[str, Any]], size: int):
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


async def _read_group(
    group_number: int,
    sheets: List[Dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    expected_ids = [
        item_id for sheet in sheets for item_id in (sheet.get("entries") or {})
    ]
    expected_sources = {
        item_id: str(candidate.get("source_hint") or "")
        for sheet in sheets
        for item_id, candidate in (sheet.get("entries") or {}).items()
        if str(candidate.get("source_hint") or "").strip()
    }
    started_at = time.perf_counter()
    async with semaphore:
        items, route = await application.translator.read_indexed_image_line_group(
            [sheet["content"] for sheet in sheets],
            "image/png",
            expected_ids,
            "CAD 多图识字基准，只读取原文。",
            expected_sources=expected_sources,
            require_complete=False,
        )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    print(
        f"多图识字 {group_number}: {len(sheets)} 张图、"
        f"{len(expected_ids)} 行、返回 {len(items)} 行、{elapsed_ms / 1000:.2f} 秒",
        flush=True,
    )
    return items, {
        "group_number": group_number,
        "image_count": len(sheets),
        "expected_count": len(expected_ids),
        "returned_count": len(items),
        "request_bytes": sum(len(sheet.get("content") or b"") for sheet in sheets),
        "elapsed_ms": elapsed_ms,
        "route": route,
    }


async def _run(arguments: argparse.Namespace) -> Dict[str, Any]:
    if isinstance(application.translator.provider, DemoProvider):
        raise ValueError("未配置真实视觉模型")
    document = fitz.open(arguments.pdf)
    try:
        page = document[arguments.page - 1]
        native_units = extract_native_page_units(page, arguments.page)
        build_started_at = time.perf_counter()
        detected_sheets = prepare_dense_cad_translation_sheets(
            page,
            arguments.page,
            desired_width=arguments.desired_width,
            rows_per_sheet=arguments.rows_per_sheet,
            native_units=native_units,
            detection_provider="tesseract",
            globally_unique_ids=True,
        )
        detected_candidates = [
            candidate
            for sheet in detected_sheets
            for candidate in (sheet.get("entries") or {}).values()
        ]
        seed_candidates = [
            {
                "rect": tuple(candidate["pixel_rect"]),
                "source_text": str(candidate.get("source_hint") or ""),
                "source_confidence": float(candidate.get("source_confidence") or 0.0),
            }
            for candidate in detected_candidates
        ]
        aliyun_supplement_count = 0
        if arguments.aliyun_report:
            report = json.loads(arguments.aliyun_report.read_text(encoding="utf-8"))
            page_report = next(
                item
                for item in report.get("pages") or []
                if int(item.get("page_number") or 0) == arguments.page
            )
            rendered = page_report.get("rendered_image") or {}
            report_width = float(rendered.get("width") or 0)
            report_height = float(rendered.get("height") or 0)
            if report_width <= 0 or report_height <= 0:
                raise ValueError("阿里云 OCR 基准缺少渲染尺寸")
            base_rects = [fitz.Rect(item["rect"]) for item in seed_candidates]
            for block in page_report.get("blocks") or []:
                source_text = str(block.get("text") or "").strip()
                if not THAI_PATTERN.search(source_text):
                    continue
                points = block.get("points") or []
                if len(points) < 4:
                    continue
                xs = [float(point["X"]) for point in points]
                ys = [float(point["Y"]) for point in points]
                rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
                if any(_overlap_smaller(rect, existing) >= 0.10 for existing in base_rects):
                    continue
                seed_candidates.append(
                    {
                        "rect": tuple(rect),
                        "source_text": source_text,
                        "source_confidence": float(block.get("confidence") or 0.0),
                    }
                )
                base_rects.append(rect)
                aliyun_supplement_count += 1
        sheets = prepare_dense_cad_translation_sheets(
            page,
            arguments.page,
            desired_width=arguments.desired_width,
            rows_per_sheet=arguments.rows_per_sheet,
            native_units=native_units,
            seed_candidates=seed_candidates,
            globally_unique_ids=True,
        )
        build_elapsed_ms = round((time.perf_counter() - build_started_at) * 1000)
    finally:
        document.close()
    indexed_sheets = [sheet for sheet in sheets if sheet.get("entries")]
    grouped = list(_groups(indexed_sheets, arguments.images_per_request))
    if arguments.only_group:
        if arguments.only_group > len(grouped):
            raise ValueError("指定的多图批次超出范围")
        grouped = [grouped[arguments.only_group - 1]]
    started_at = time.perf_counter()
    semaphore = asyncio.Semaphore(arguments.concurrency)
    results = await asyncio.gather(
        *(
            _read_group(number, group, semaphore)
            for number, group in enumerate(
                grouped,
                start=arguments.only_group or 1,
            )
        )
    )
    model_wall_elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    by_id = {
        item["id"]: item
        for items, _detail in results
        for item in items
    }
    comparisons = []
    baseline_items = []
    baseline = arguments.baseline
    if baseline is None and arguments.page == 7:
        baseline = DEFAULT_BASELINE
    if baseline is not None and baseline.is_file():
        baseline_items = json.loads(baseline.read_text(encoding="utf-8")).get(
            "comparisons", []
        )
    expected_ids = [
        item_id for sheet in indexed_sheets for item_id in sheet["entries"]
    ]
    for index, item_id in enumerate(expected_ids):
        item = by_id.get(item_id) or {}
        source_text = str(item.get("source_text") or "")
        baseline_text = (
            str(baseline_items[index].get("full_text") or "")
            if index < len(baseline_items)
            else ""
        )
        source_key = _thai_key(source_text)
        baseline_key = _thai_key(baseline_text)
        comparisons.append(
            {
                "id": item_id,
                "grouped_text": source_text,
                "baseline_text": baseline_text,
                "grouped_has_thai": bool(source_key),
                "baseline_has_thai": bool(baseline_key),
                "exact_thai_match": bool(source_key) and source_key == baseline_key,
            }
        )
    details = [detail for _items, detail in results]
    return {
        "configuration": {
            "page": arguments.page,
            "rows_per_sheet": arguments.rows_per_sheet,
            "images_per_request": arguments.images_per_request,
            "concurrency": arguments.concurrency,
            "desired_width": arguments.desired_width,
            "only_group": arguments.only_group,
            "aliyun_supplement": bool(arguments.aliyun_report),
        },
        "summary": {
            "candidate_count": len(expected_ids),
            "aliyun_supplement_count": aliyun_supplement_count,
            "sheet_count": len(indexed_sheets),
            "request_count": len(grouped),
            "returned_count": len(by_id),
            "grouped_thai_count": sum(item["grouped_has_thai"] for item in comparisons),
            "baseline_thai_count": sum(item["baseline_has_thai"] for item in comparisons),
            "exact_thai_match_count": sum(item["exact_thai_match"] for item in comparisons),
            "sheet_build_elapsed_ms": build_elapsed_ms,
            "model_wall_elapsed_ms": model_wall_elapsed_ms,
            "request_sum_elapsed_ms": sum(item["elapsed_ms"] for item in details),
        },
        "requests": details,
        "comparisons": comparisons,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--page", type=int, default=7)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--desired-width", type=int, default=4000)
    parser.add_argument("--rows-per-sheet", type=int, default=12)
    parser.add_argument("--images-per-request", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--only-group", type=int, default=0)
    parser.add_argument(
        "--aliyun-report",
        type=Path,
        default=None,
        const=DEFAULT_ALIYUN_REPORT,
        nargs="?",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if not arguments.pdf.is_file():
        raise SystemExit("未找到待测 PDF")
    with fitz.open(arguments.pdf) as document:
        if not 1 <= arguments.page <= document.page_count:
            raise SystemExit("页码超出范围")
    if min(
        arguments.rows_per_sheet,
        arguments.images_per_request,
        arguments.concurrency,
    ) < 1:
        raise SystemExit("批次参数必须大于 0")
    report = asyncio.run(_run(arguments))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    only_group_suffix = (
        f"-only-{arguments.only_group}" if arguments.only_group else ""
    )
    supplement_suffix = "-aliyun" if arguments.aliyun_report else ""
    destination = OUTPUT_DIRECTORY / (
        f"cad-multi-image-page-{arguments.page}-group-{arguments.images_per_request}"
        f"{only_group_suffix}{supplement_suffix}.json"
    )
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"结果已保存: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CAD 多图基准失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
