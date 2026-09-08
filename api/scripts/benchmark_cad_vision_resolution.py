#!/usr/bin/env python3
"""Strictly compare full versus half-resolution CAD indexed-image reading.

The source candidate rectangles are detected once at full resolution, then
rescaled into both render variants. This isolates source-pixel density from
candidate recall, sheet row count, and model concurrency.
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


def _seed_candidates(sheets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = []
    for sheet in sheets:
        for item in (sheet.get("entries") or {}).values():
            rect = item.get("pixel_rect")
            if rect is None:
                raise RuntimeError("CAD 索引候选缺少像素坐标")
            candidates.append(
                {
                    "rect": tuple(rect),
                    "source_text": str(item.get("source_hint") or ""),
                    "source_confidence": float(item.get("source_confidence") or 0.0),
                }
            )
    return candidates


def _rescale_seeds(seeds: Sequence[Dict[str, Any]], ratio: float) -> List[Dict[str, Any]]:
    output = []
    for item in seeds:
        rect = fitz.Rect(item["rect"])
        output.append(
            {
                "rect": (rect.x0 * ratio, rect.y0 * ratio, rect.x1 * ratio, rect.y1 * ratio),
                "source_text": item["source_text"],
                "source_confidence": item["source_confidence"],
            }
        )
    return output


def _assign_global_ids(sheets: Sequence[Dict[str, Any]]) -> List[str]:
    identifiers = []
    for sheet in sheets:
        for candidate in (sheet.get("entries") or {}).values():
            global_id = f"C{len(identifiers) + 1:03d}"
            candidate["benchmark_global_id"] = global_id
            identifiers.append(global_id)
    return identifiers


async def _read_sheet(
    number: int,
    sheet: Dict[str, Any],
    provider: OpenAIProvider,
    semaphore: asyncio.Semaphore,
    total: int,
    label: str,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    started_at = time.perf_counter()
    async with semaphore:
        items, route = await provider.translate_indexed_image_lines(
            sheet["content"],
            "image/png",
            "zh",
            "分辨率 A/B 识字基准：只读取每个 ID 右侧的泰文 source_text；不得猜测、合并或翻译。",
        )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    result_by_id = {
        str(item.get("id") or ""): str(item.get("source_text") or "").strip()
        for item in items
        if str(item.get("id") or "").strip()
    }
    print(f"{label} 识字 {number}/{total}: {elapsed_ms / 1000:.2f}s", flush=True)
    return (
        {
            str(candidate["benchmark_global_id"]): result_by_id.get(str(item_id), "")
            for item_id, candidate in (sheet.get("entries") or {}).items()
        },
        {
            "sheet_number": number,
            "entry_count": len(sheet.get("entries") or {}),
            "image_bytes": len(sheet.get("content") or b""),
            "elapsed_ms": elapsed_ms,
            "route": route,
        },
    )


async def _read_variant(
    sheets: Sequence[Dict[str, Any]],
    provider: OpenAIProvider,
    concurrency: int,
    label: str,
) -> Tuple[Dict[str, str], List[Dict[str, Any]], int]:
    started_at = time.perf_counter()
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results = await asyncio.gather(
        *(
            _read_sheet(number, sheet, provider, semaphore, len(sheets), label)
            for number, sheet in enumerate(sheets, start=1)
        )
    )
    output: Dict[str, str] = {}
    details = []
    for texts, detail in results:
        output.update(texts)
        details.append(detail)
    return output, details, round((time.perf_counter() - started_at) * 1000)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--page", type=int, default=7)
    parser.add_argument("--rows-per-sheet", type=int, default=12)
    parser.add_argument("--full-width", type=int, default=4000)
    parser.add_argument("--half-width", type=int, default=2000)
    parser.add_argument("--visual-concurrency", type=int, default=5)
    return parser.parse_args()


async def _run(arguments: argparse.Namespace) -> Dict[str, Any]:
    if not arguments.pdf.is_file():
        raise ValueError("未找到待测 PDF")
    if not settings.openai_api_key:
        raise ValueError("未配置视觉模型 API Key")
    if arguments.half_width >= arguments.full_width:
        raise ValueError("half-width 必须小于 full-width")
    document = fitz.open(arguments.pdf)
    try:
        if not 1 <= arguments.page <= document.page_count:
            raise ValueError("指定页码超出 PDF 页数")
        page = document[arguments.page - 1]
        page_width = max(1.0, float(page.rect.width))
        full_scale = min(2.0, arguments.full_width / page_width)
        half_scale = min(2.0, arguments.half_width / page_width)
        native_units = extract_native_page_units(page, arguments.page)
        detection_started_at = time.perf_counter()
        detected_sheets = prepare_dense_cad_translation_sheets(
            page,
            arguments.page,
            desired_width=arguments.full_width,
            minimum_render_scale=0.1,
            rows_per_sheet=arguments.rows_per_sheet,
            native_units=native_units,
            detection_provider="tesseract",
        )
        detection_elapsed_ms = round((time.perf_counter() - detection_started_at) * 1000)
        seeds = _seed_candidates(detected_sheets)
        if not seeds:
            raise RuntimeError("未生成 CAD 候选")
        full_sheets = prepare_dense_cad_translation_sheets(
            page,
            arguments.page,
            desired_width=arguments.full_width,
            minimum_render_scale=0.1,
            rows_per_sheet=arguments.rows_per_sheet,
            native_units=native_units,
            seed_candidates=seeds,
        )
        half_sheets = prepare_dense_cad_translation_sheets(
            page,
            arguments.page,
            desired_width=arguments.half_width,
            minimum_render_scale=0.1,
            rows_per_sheet=arguments.rows_per_sheet,
            native_units=native_units,
            seed_candidates=_rescale_seeds(seeds, half_scale / full_scale),
        )
    finally:
        document.close()

    full_sheets = [sheet for sheet in full_sheets if sheet.get("entries")]
    half_sheets = [sheet for sheet in half_sheets if sheet.get("entries")]
    full_ids = _assign_global_ids(full_sheets)
    half_ids = _assign_global_ids(half_sheets)
    if full_ids != half_ids or len(full_ids) != len(seeds):
        raise RuntimeError(
            f"固定候选集未保持一致：full={len(full_ids)} half={len(half_ids)} source={len(seeds)}"
        )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / f"cad-resolution-page-{arguments.page}-full-sheet-001.png").write_bytes(
        full_sheets[0]["content"]
    )
    (OUTPUT_DIRECTORY / f"cad-resolution-page-{arguments.page}-half-sheet-001.png").write_bytes(
        half_sheets[0]["content"]
    )

    provider = OpenAIProvider(settings)
    full_texts, full_details, full_wall_ms = await _read_variant(
        full_sheets, provider, arguments.visual_concurrency, "全分辨率"
    )
    half_texts, half_details, half_wall_ms = await _read_variant(
        half_sheets, provider, arguments.visual_concurrency, "半分辨率"
    )
    comparisons = []
    for candidate_number in range(1, len(seeds) + 1):
        identifier = f"C{candidate_number:03d}"
        full_text = full_texts.get(identifier, "")
        half_text = half_texts.get(identifier, "")
        full_thai = _thai_key(full_text)
        half_thai = _thai_key(half_text)
        comparisons.append(
            {
                "id": identifier,
                "full_text": full_text,
                "half_text": half_text,
                "full_has_thai": bool(full_thai),
                "half_has_thai": bool(half_thai),
                "exact_thai_match": bool(full_thai) and full_thai == half_thai,
                "regression": bool(full_thai) and full_thai != half_thai,
            }
        )
    regressions = [item for item in comparisons if item["regression"]]
    summary = {
        "candidate_count": len(seeds),
        "rows_per_sheet": arguments.rows_per_sheet,
        "visual_concurrency": arguments.visual_concurrency,
        "full_render_width": round(page_width * full_scale),
        "half_render_width": round(page_width * half_scale),
        "full_sheet_count": len(full_sheets),
        "half_sheet_count": len(half_sheets),
        "full_thai_count": sum(item["full_has_thai"] for item in comparisons),
        "half_thai_count": sum(item["half_has_thai"] for item in comparisons),
        "exact_thai_match_count": sum(item["exact_thai_match"] for item in comparisons),
        "regression_count": len(regressions),
        "strict_non_regression_pass": not regressions,
        "candidate_detection_elapsed_ms": detection_elapsed_ms,
        "full_model_wall_elapsed_ms": full_wall_ms,
        "half_model_wall_elapsed_ms": half_wall_ms,
        "full_total_image_bytes": sum(item["image_bytes"] for item in full_details),
        "half_total_image_bytes": sum(item["image_bytes"] for item in half_details),
    }
    return {
        "summary": summary,
        "full_sheets": full_details,
        "half_sheets": half_details,
        "comparisons": comparisons,
    }


def main() -> int:
    arguments = _arguments()
    if arguments.rows_per_sheet < 1 or arguments.visual_concurrency < 1:
        raise SystemExit("rows-per-sheet 和 visual-concurrency 必须大于 0")
    print(
        f"cad_resolution_started page={arguments.page} rows={arguments.rows_per_sheet} "
        f"concurrency={arguments.visual_concurrency} full={arguments.full_width} half={arguments.half_width}",
        flush=True,
    )
    report = asyncio.run(_run(arguments))
    destination = OUTPUT_DIRECTORY / f"cad-resolution-page-{arguments.page}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(
        f"完成：候选 {summary['candidate_count']}；全分辨率泰文 {summary['full_thai_count']}，"
        f"半分辨率泰文 {summary['half_thai_count']}，回归 {summary['regression_count']}；"
        f"全分辨率 {summary['full_model_wall_elapsed_ms'] / 1000:.2f}s，"
        f"半分辨率 {summary['half_model_wall_elapsed_ms'] / 1000:.2f}s",
        flush=True,
    )
    print(f"结果已保存: {destination}", flush=True)
    return 0 if summary["strict_non_regression_pass"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CAD 分辨率基准失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
