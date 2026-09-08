#!/usr/bin/env python3
"""Benchmark Alibaba Thai OCR on enlarged, indexed CAD text sheets.

This is a QA experiment only. It does not alter production routing or export.
Each CAD candidate remains at the same enlarged pixel size as the legacy
vision route; only the sheet height changes from six rows to twelve rows.
Alibaba OCR is compared with an independent visual reread of the same sheet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence

import fitz


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app.config import settings
from api.app.services.translator import OpenAIProvider
from api.app.services.visual_pdf import (
    extract_native_page_units,
    prepare_dense_cad_translation_sheets,
)
from benchmark_aliyun_thai_ocr import (
    _block_summary,
    _build_client,
    _read_access_key,
    _request_ocr,
    _write_coordinate_overlay,
)


DEFAULT_PDF = ROOT / "Attach_TOR_1_260805_224430.pdf"
DEFAULT_CREDENTIALS = ROOT / "AccessKey.csv"
OUTPUT_DIRECTORY = ROOT / "output" / "ocr-benchmarks"
THAI_PATTERN = re.compile(r"[\u0E00-\u0E7F]")


def _thai_key(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value or "")
    return "".join(char for char in normalized if THAI_PATTERN.fullmatch(char))


def _point_center_y(block: Dict[str, Any]) -> float:
    points = block.get("points") or []
    if len(points) != 4:
        return -1.0
    return sum(float(point.get("Y") or 0.0) for point in points) / 4.0


def _point_left(block: Dict[str, Any]) -> float:
    points = block.get("points") or []
    if not points:
        return 0.0
    return min(float(point.get("X") or 0.0) for point in points)


def _map_ocr_rows(sheet: Dict[str, Any], blocks: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Assign Thai OCR blocks to the known indexed-sheet rows.

    The sheet generator already assigns exactly one candidate per row. This
    mapping is geometry-only; it never trusts the OCR's reading of the printed
    ID label.
    """
    row_height = float(sheet.get("row_height") or 120)
    by_row: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if not _thai_key(text):
            continue
        row = int(max(0.0, _point_center_y(block)) // row_height)
        by_row[row].append(block)

    output: Dict[str, Dict[str, Any]] = {}
    for item_id, candidate in (sheet.get("entries") or {}).items():
        row = int(candidate.get("sheet_row") or 0)
        row_blocks = sorted(by_row.get(row, []), key=_point_left)
        texts = [str(block.get("text") or "").strip() for block in row_blocks]
        confidences = []
        for block in row_blocks:
            try:
                confidences.append(float(block.get("confidence")))
            except (TypeError, ValueError):
                continue
        output[str(item_id)] = {
            "text": " ".join(text for text in texts if text).strip(),
            "block_count": len(row_blocks),
            "confidence": round(median(confidences), 4) if confidences else None,
        }
    return output


async def _visual_reread(sheet: Dict[str, Any], provider: OpenAIProvider) -> Dict[str, str]:
    """Read the same indexed sheet without supplying OCR hints."""
    items, _route = await provider.translate_indexed_image_lines(
        sheet["content"],
        "image/png",
        "zh",
        "OCR 准确率审计：仅依据索引图右侧原始文字复读 source_text。",
    )
    return {
        str(item.get("id") or ""): str(item.get("source_text") or "").strip()
        for item in items
        if str(item.get("id") or "").strip()
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--page", type=int, default=7)
    parser.add_argument("--rows-per-sheet", type=int, default=12)
    parser.add_argument("--desired-width", type=int, default=4000)
    parser.add_argument("--visual-concurrency", type=int, default=2)
    return parser.parse_args()


async def _run(arguments: argparse.Namespace) -> Dict[str, Any]:
    if not arguments.pdf.is_file():
        raise ValueError("未找到待测 PDF")
    if arguments.page < 1:
        raise ValueError("页码从 1 开始")
    if arguments.rows_per_sheet < 2 or arguments.rows_per_sheet > 20:
        raise ValueError("每张索引图行数必须在 2 至 20 之间")
    if not settings.openai_api_key:
        raise ValueError("未配置用于独立复读核验的模型 API Key")

    access_key_id, access_key_secret = _read_access_key(arguments.credentials)
    client = _build_client(access_key_id, access_key_secret)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    document = fitz.open(arguments.pdf)
    try:
        if arguments.page > document.page_count:
            raise ValueError("指定页码超出 PDF 页数")
        page = document[arguments.page - 1]
        native_units = extract_native_page_units(page, arguments.page)
        prepare_started_at = time.perf_counter()
        sheets = prepare_dense_cad_translation_sheets(
            page,
            arguments.page,
            desired_width=arguments.desired_width,
            rows_per_sheet=arguments.rows_per_sheet,
            native_units=native_units,
            detection_provider="tesseract",
        )
        prepare_elapsed_ms = round((time.perf_counter() - prepare_started_at) * 1000)
    finally:
        document.close()

    indexed_sheets = [sheet for sheet in sheets if sheet.get("entries")]
    if not indexed_sheets:
        raise RuntimeError("未生成 CAD 索引图")
    candidate_count = sum(len(sheet["entries"]) for sheet in indexed_sheets)
    report_sheets: List[Dict[str, Any]] = []

    for sheet_number, sheet in enumerate(indexed_sheets, start=1):
        ocr_started_at = time.perf_counter()
        response = _request_ocr(client, sheet["content"])
        elapsed_ms = round((time.perf_counter() - ocr_started_at) * 1000)
        blocks = _block_summary(response)
        image_path = OUTPUT_DIRECTORY / (
            f"aliyun-indexed-page-{arguments.page}-rows-{arguments.rows_per_sheet}-"
            f"sheet-{sheet_number:03d}.png"
        )
        image_path.write_bytes(sheet["content"])
        overlay_path = image_path.with_name(f"{image_path.stem}-ocr-overlay.png")
        _write_coordinate_overlay(sheet["content"], blocks, overlay_path)
        report_sheets.append(
            {
                "sheet_number": sheet_number,
                "image": str(image_path.relative_to(ROOT)),
                "coordinate_overlay": str(overlay_path.relative_to(ROOT)),
                "image_bytes": len(sheet["content"]),
                "entry_count": len(sheet["entries"]),
                "ocr_elapsed_ms": elapsed_ms,
                "entries": {
                    item_id: {
                        "row": int(candidate.get("sheet_row") or 0),
                        "bbox": list(candidate.get("bbox") or []),
                        "local_source_hint": str(candidate.get("source_hint") or ""),
                    }
                    for item_id, candidate in sheet["entries"].items()
                },
                "aliyun_rows": _map_ocr_rows(sheet, blocks),
            }
        )
        print(
            f"阿里 OCR {sheet_number}/{len(indexed_sheets)}："
            f"{len(sheet['entries'])} 行，{elapsed_ms / 1000:.2f} 秒",
            flush=True,
        )

    provider = OpenAIProvider(settings)
    semaphore = asyncio.Semaphore(max(1, arguments.visual_concurrency))

    async def audit_sheet(index: int, sheet: Dict[str, Any]) -> Dict[str, str]:
        async with semaphore:
            result = await _visual_reread(sheet, provider)
            print(f"视觉复读 {index}/{len(indexed_sheets)} 完成", flush=True)
            return result

    visual_results = await asyncio.gather(
        *(audit_sheet(index, sheet) for index, sheet in enumerate(indexed_sheets, start=1))
    )

    comparisons = []
    for sheet_report, visual_rows in zip(report_sheets, visual_results):
        sheet_number = int(sheet_report["sheet_number"])
        for item_id in sheet_report["entries"]:
            aliyun = (sheet_report["aliyun_rows"].get(item_id) or {}).get("text") or ""
            visual = visual_rows.get(item_id) or ""
            comparisons.append(
                {
                    "sheet_number": sheet_number,
                    "id": item_id,
                    "aliyun_text": aliyun,
                    "visual_text": visual,
                    "aliyun_nonempty": bool(_thai_key(aliyun)),
                    "visual_nonempty": bool(_thai_key(visual)),
                    "exact_thai_match": bool(_thai_key(aliyun))
                    and _thai_key(aliyun) == _thai_key(visual),
                }
            )

    exact_count = sum(item["exact_thai_match"] for item in comparisons)
    aliyun_nonempty_count = sum(item["aliyun_nonempty"] for item in comparisons)
    visual_nonempty_count = sum(item["visual_nonempty"] for item in comparisons)
    summary = {
        "candidate_count": candidate_count,
        "sheet_count": len(indexed_sheets),
        "rows_per_sheet": arguments.rows_per_sheet,
        "prepare_elapsed_ms": prepare_elapsed_ms,
        "aliyun_nonempty_count": aliyun_nonempty_count,
        "visual_nonempty_count": visual_nonempty_count,
        "exact_thai_match_count": exact_count,
        "exact_thai_match_rate": round(exact_count / max(1, candidate_count), 4),
        "strict_pass": (
            candidate_count == aliyun_nonempty_count == visual_nonempty_count == exact_count
        ),
    }
    return {"summary": summary, "sheets": report_sheets, "comparisons": comparisons}


def main() -> int:
    arguments = _arguments()
    print(
        f"indexed_aliyun_ocr_started page={arguments.page} rows_per_sheet={arguments.rows_per_sheet}",
        flush=True,
    )
    report = asyncio.run(_run(arguments))
    destination = OUTPUT_DIRECTORY / (
        f"aliyun-indexed-page-{arguments.page}-rows-{arguments.rows_per_sheet}.json"
    )
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(
        f"完成：{summary['candidate_count']} 候选、{summary['sheet_count']} 张索引图；"
        f"阿里有结果 {summary['aliyun_nonempty_count']}，"
        f"与视觉复读一致 {summary['exact_thai_match_count']}，"
        f"严格验收 {'通过' if summary['strict_pass'] else '不通过'}",
        flush=True,
    )
    print(f"结果已保存: {destination}", flush=True)
    return 0 if summary["strict_pass"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"索引图阿里 OCR 基准失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
