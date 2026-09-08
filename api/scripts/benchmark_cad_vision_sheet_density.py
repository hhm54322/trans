#!/usr/bin/env python3
"""Compare legacy CAD visual reading with six versus twelve rows per sheet.

Both variants are rendered from one fixed Tesseract candidate set, so this
measures sheet density only. It never updates the production route or writes a
translated document.
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

from api.app.config import settings
from api.app.services.translator import OpenAIProvider
from api.app.services.visual_pdf import (
    extract_native_page_units,
    prepare_dense_cad_translation_sheets,
)


DEFAULT_PDF = ROOT / "Attach_TOR_1_260805_224430.pdf"
OUTPUT_DIRECTORY = ROOT / "output" / "vision-benchmarks"
THAI_PATTERN = re.compile(r"[\u0E00-\u0E7F]")


def _thai_key(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value or "")
    return "".join(char for char in normalized if THAI_PATTERN.fullmatch(char))


def _flatten_entries(sheets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        candidate
        for sheet in sheets
        for candidate in (sheet.get("entries") or {}).values()
    ]


def _seed_candidates(sheets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seeds = []
    for candidate in _flatten_entries(sheets):
        rect = candidate.get("pixel_rect")
        if rect is None:
            raise RuntimeError("CAD 索引候选缺少像素坐标，无法构造固定对照集")
        seeds.append(
            {
                "rect": tuple(rect),
                "source_text": str(candidate.get("source_hint") or ""),
                "source_confidence": float(candidate.get("source_confidence") or 0.0),
            }
        )
    return seeds


def _assign_global_ids(sheets: Sequence[Dict[str, Any]]) -> List[str]:
    global_ids = []
    for sheet in sheets:
        for item_id, candidate in (sheet.get("entries") or {}).items():
            global_id = f"C{len(global_ids) + 1:03d}"
            candidate["benchmark_global_id"] = global_id
            global_ids.append(global_id)
    return global_ids


async def _read_sheet(
    sheet_number: int,
    sheet: Dict[str, Any],
    provider: OpenAIProvider,
    semaphore: asyncio.Semaphore,
    total_sheets: int,
    label: str,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    started_at = time.perf_counter()
    async with semaphore:
        items, route = await provider.translate_indexed_image_lines(
            sheet["content"],
            "image/png",
            "zh",
            "A/B 识字基准：仅按索引图右侧原文读取 source_text；不得猜测或合并相邻行。",
        )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    by_item_id = {
        str(item.get("id") or ""): str(item.get("source_text") or "").strip()
        for item in items
        if str(item.get("id") or "").strip()
    }
    output = {
        str(candidate["benchmark_global_id"]): by_item_id.get(str(item_id), "")
        for item_id, candidate in (sheet.get("entries") or {}).items()
    }
    print(
        f"{label} 视觉识字 {sheet_number}/{total_sheets}：{elapsed_ms / 1000:.2f} 秒",
        flush=True,
    )
    return output, {
        "sheet_number": sheet_number,
        "entry_count": len(sheet.get("entries") or {}),
        "image_bytes": len(sheet.get("content") or b""),
        "elapsed_ms": elapsed_ms,
        "route": route,
    }


async def _read_variant(
    sheets: Sequence[Dict[str, Any]],
    provider: OpenAIProvider,
    concurrency: int,
    label: str,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results = await asyncio.gather(
        *(
            _read_sheet(index, sheet, provider, semaphore, len(sheets), label)
            for index, sheet in enumerate(sheets, start=1)
        )
    )
    output: Dict[str, str] = {}
    details = []
    for texts, sheet_detail in results:
        output.update(texts)
        details.append(sheet_detail)
    return output, details


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--page", type=int, default=7)
    parser.add_argument("--desired-width", type=int, default=4000)
    parser.add_argument("--visual-concurrency", type=int, default=2)
    return parser.parse_args()


async def _run(arguments: argparse.Namespace) -> Dict[str, Any]:
    if not arguments.pdf.is_file():
        raise ValueError("未找到待测 PDF")
    if not settings.openai_api_key:
        raise ValueError("未配置视觉模型 API Key")
    document = fitz.open(arguments.pdf)
    try:
        if arguments.page < 1 or arguments.page > document.page_count:
            raise ValueError("指定页码超出 PDF 页数")
        page = document[arguments.page - 1]
        native_units = extract_native_page_units(page, arguments.page)
        candidate_started_at = time.perf_counter()
        detected_sheets = prepare_dense_cad_translation_sheets(
            page,
            arguments.page,
            desired_width=arguments.desired_width,
            rows_per_sheet=12,
            native_units=native_units,
            detection_provider="tesseract",
        )
        candidate_elapsed_ms = round((time.perf_counter() - candidate_started_at) * 1000)
        seeds = _seed_candidates(detected_sheets)
        if not seeds:
            raise RuntimeError("未生成 CAD 候选")
        variants = {}
        for rows_per_sheet in (6, 12):
            sheets = prepare_dense_cad_translation_sheets(
                page,
                arguments.page,
                desired_width=arguments.desired_width,
                rows_per_sheet=rows_per_sheet,
                native_units=native_units,
                seed_candidates=seeds,
            )
            indexed = [sheet for sheet in sheets if sheet.get("entries")]
            global_ids = _assign_global_ids(indexed)
            if len(global_ids) != len(seeds):
                raise RuntimeError(
                    f"{rows_per_sheet} 行版本候选数变化：{len(global_ids)} != {len(seeds)}"
                )
            variants[rows_per_sheet] = indexed
    finally:
        document.close()

    provider = OpenAIProvider(settings)
    six_texts, six_details = await _read_variant(
        variants[6], provider, arguments.visual_concurrency, "6 行"
    )
    twelve_texts, twelve_details = await _read_variant(
        variants[12], provider, arguments.visual_concurrency, "12 行"
    )
    compared = []
    for candidate_number in range(1, len(seeds) + 1):
        global_id = f"C{candidate_number:03d}"
        six = six_texts.get(global_id, "")
        twelve = twelve_texts.get(global_id, "")
        six_key = _thai_key(six)
        twelve_key = _thai_key(twelve)
        compared.append(
            {
                "id": global_id,
                "six_row_text": six,
                "twelve_row_text": twelve,
                "six_row_thai": bool(six_key),
                "twelve_row_thai": bool(twelve_key),
                "exact_thai_match": bool(six_key) and six_key == twelve_key,
                "regression": bool(six_key) and six_key != twelve_key,
            }
        )
    baseline_count = sum(item["six_row_thai"] for item in compared)
    twelve_count = sum(item["twelve_row_thai"] for item in compared)
    regressions = [item for item in compared if item["regression"]]
    summary = {
        "candidate_count": len(seeds),
        "six_row_sheet_count": len(variants[6]),
        "twelve_row_sheet_count": len(variants[12]),
        "six_row_thai_count": baseline_count,
        "twelve_row_thai_count": twelve_count,
        "exact_thai_match_count": sum(item["exact_thai_match"] for item in compared),
        "regression_count": len(regressions),
        "strict_non_regression_pass": len(regressions) == 0,
        "candidate_detection_elapsed_ms": candidate_elapsed_ms,
        "six_row_model_elapsed_ms": sum(item["elapsed_ms"] for item in six_details),
        "twelve_row_model_elapsed_ms": sum(item["elapsed_ms"] for item in twelve_details),
    }
    return {
        "summary": summary,
        "six_row_sheets": six_details,
        "twelve_row_sheets": twelve_details,
        "comparisons": compared,
    }


def main() -> int:
    arguments = _arguments()
    print(f"cad_vision_density_started page={arguments.page}", flush=True)
    report = asyncio.run(_run(arguments))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIRECTORY / f"cad-vision-density-page-{arguments.page}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(
        f"完成：{summary['candidate_count']} 候选；6 行 {summary['six_row_sheet_count']} 张，"
        f"12 行 {summary['twelve_row_sheet_count']} 张；"
        f"6 行泰文 {summary['six_row_thai_count']}，12 行泰文 {summary['twelve_row_thai_count']}，"
        f"回归 {summary['regression_count']}",
        flush=True,
    )
    print(f"结果已保存: {destination}", flush=True)
    return 0 if summary["strict_non_regression_pass"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CAD 视觉密度基准失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
