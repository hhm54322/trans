#!/usr/bin/env python3
"""Visually audit Alibaba Cloud Thai OCR against representative PDF crops.

This is a QA-only tool. It never changes the production translation route.
Alibaba OCR provides the candidate text and coordinates; the configured vision
model re-reads enlarged crops so disagreements can be inspected before OCR is
trusted as the CAD source reader.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
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
from api.app.services.visual_pdf import prepare_dense_cad_translation_sheets


DEFAULT_PDF = ROOT / "Attach_TOR_1_260805_224430.pdf"
DEFAULT_REPORT = ROOT / "output" / "ocr-benchmarks" / "aliyun-thai-ocr-benchmark.json"
OUTPUT_DIRECTORY = ROOT / "output" / "ocr-audits"
THAI_PATTERN = re.compile(r"[\u0E00-\u0E7F]")


def _thai_key(value: str, *, ignore_marks: bool = False) -> str:
    normalized = unicodedata.normalize("NFC", value or "")
    return "".join(
        char
        for char in normalized
        if THAI_PATTERN.fullmatch(char)
        and (not ignore_marks or unicodedata.category(char) not in {"Mn", "Me"})
    )


def _similarity(first: str, second: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, _thai_key(first), _thai_key(second), autojunk=False).ratio()


def _sample_blocks(blocks: Sequence[Dict[str, Any]], maximum: int) -> List[Dict[str, Any]]:
    """Select low-confidence and spatially distributed Thai labels."""
    thai_blocks = [
        block
        for block in blocks
        if _thai_key(str(block.get("text") or ""))
        and len(block.get("points") or []) == 4
    ]
    if len(thai_blocks) <= maximum:
        return thai_blocks

    def confidence(block: Dict[str, Any]) -> float:
        try:
            return float(block.get("confidence"))
        except (TypeError, ValueError):
            return -1.0

    def center(block: Dict[str, Any]) -> tuple[float, float]:
        points = block["points"]
        return (
            sum(float(point["X"]) for point in points) / 4,
            sum(float(point["Y"]) for point in points) / 4,
        )

    selected: List[Dict[str, Any]] = []
    selected_keys = set()

    def add(block: Dict[str, Any]) -> None:
        key = tuple(
            (round(float(point["X"])), round(float(point["Y"])))
            for point in block["points"]
        )
        if key not in selected_keys and len(selected) < maximum:
            selected_keys.add(key)
            selected.append(block)

    low_confidence_count = max(3, maximum // 2)
    for block in sorted(thai_blocks, key=confidence)[:low_confidence_count]:
        add(block)

    spatial = sorted(thai_blocks, key=lambda block: (center(block)[1], center(block)[0]))
    remaining = maximum - len(selected)
    for index in range(remaining):
        location = (
            len(spatial) // 2
            if remaining == 1
            else round(index * (len(spatial) - 1) / (remaining - 1))
        )
        add(spatial[location])
    for block in spatial:
        add(block)
    return selected


def _seed_candidates(
    page: fitz.Page,
    source_width: int,
    source_height: int,
    blocks: Iterable[Dict[str, Any]],
    *,
    desired_width: int,
) -> List[Dict[str, Any]]:
    """Map OCR image coordinates into the indexed-sheet raster coordinates."""
    render_scale = min(
        2.0,
        max(1.0, desired_width / max(1.0, float(page.rect.width))),
    )
    target_width = float(page.rect.width) * render_scale
    target_height = float(page.rect.height) * render_scale
    candidates = []
    for block in blocks:
        points = block["points"]
        xs = [float(point["X"]) / source_width * target_width for point in points]
        ys = [float(point["Y"]) / source_height * target_height for point in points]
        candidates.append(
            {
                "rect": (min(xs), min(ys), max(xs), max(ys)),
                "source_text": str(block.get("text") or "").strip(),
                "source_confidence": float(block.get("confidence") or 0.0),
            }
        )
    return candidates


async def _audit_page(
    document: fitz.Document,
    page_report: Dict[str, Any],
    sample_size: int,
) -> List[Dict[str, Any]]:
    page_number = int(page_report["page_number"])
    blocks = _sample_blocks(page_report.get("blocks") or [], sample_size)
    if not blocks:
        return []
    page = document[page_number - 1]
    desired_width = 4000
    rendered = page_report.get("rendered_image") or {}
    source_width = int(rendered.get("width") or 0)
    source_height = int(rendered.get("height") or 0)
    if source_width < 1 or source_height < 1:
        raise ValueError(f"第 {page_number} 页 OCR 结果缺少渲染尺寸")
    sheets = prepare_dense_cad_translation_sheets(
        page,
        page_number,
        desired_width=desired_width,
        rows_per_sheet=6,
        native_units=[],
        seed_candidates=_seed_candidates(
            page,
            source_width,
            source_height,
            blocks,
            desired_width=desired_width,
        ),
    )
    provider = OpenAIProvider(settings)
    results: List[Dict[str, Any]] = []
    for sheet_index, sheet in enumerate(sheets, start=1):
        expected_sources = {
            item_id: str(candidate.get("source_hint") or "")
            for item_id, candidate in (sheet.get("entries") or {}).items()
        }
        translated, route = await provider.translate_indexed_image_lines(
            sheet["content"],
            "image/png",
            "zh",
            "OCR 审计：仅依据图片复读每个编号右侧的原文。",
            expected_sources=expected_sources,
        )
        reviewed = {str(item.get("id") or ""): item for item in translated}
        audit_path = OUTPUT_DIRECTORY / f"page-{page_number}-sheet-{sheet_index}.png"
        audit_path.write_bytes(sheet["content"])
        for item_id, source_text in expected_sources.items():
            item = reviewed.get(item_id)
            visual_text = str((item or {}).get("source_text") or "")
            results.append(
                {
                    "page_number": page_number,
                    "sheet": sheet_index,
                    "id": item_id,
                    "aliyun_text": source_text,
                    "vision_text": visual_text,
                    "exact_thai_match": _thai_key(source_text) == _thai_key(visual_text),
                    "thai_similarity": round(_similarity(source_text, visual_text), 4),
                    "vision_route": route,
                    "audit_image": str(audit_path.relative_to(ROOT)),
                }
            )
    return results


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--pages", default="1,2,7,23")
    parser.add_argument("--sample-per-page", type=int, default=12)
    return parser.parse_args()


async def _run(arguments: argparse.Namespace) -> Dict[str, Any]:
    if not arguments.pdf.is_file():
        raise ValueError("未找到待核验 PDF")
    if not arguments.report.is_file():
        raise ValueError("未找到阿里云 OCR 基准结果")
    if not settings.openai_api_key:
        raise ValueError("未配置用于 OCR 复读核验的模型 API Key")
    selected_pages = {int(value.strip()) for value in arguments.pages.split(",") if value.strip()}
    if not selected_pages:
        raise ValueError("至少指定一个页码")
    if arguments.sample_per_page < 1:
        raise ValueError("每页抽检数量至少为 1")

    report = json.loads(arguments.report.read_text(encoding="utf-8"))
    reports = [
        page for page in report.get("pages") or [] if int(page.get("page_number") or 0) in selected_pages
    ]
    missing = selected_pages - {int(page["page_number"]) for page in reports}
    if missing:
        raise ValueError(f"OCR 基准结果缺少页面：{', '.join(map(str, sorted(missing)))}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    document = fitz.open(arguments.pdf)
    try:
        audited = []
        for page_report in reports:
            audited.extend(
                await _audit_page(document, page_report, arguments.sample_per_page)
            )
    finally:
        document.close()

    by_page = defaultdict(list)
    for item in audited:
        by_page[item["page_number"]].append(item)
    page_summary = []
    for page_number, values in sorted(by_page.items()):
        similarities = [item["thai_similarity"] for item in values]
        page_summary.append(
            {
                "page_number": page_number,
                "sample_count": len(values),
                "exact_match_count": sum(item["exact_thai_match"] for item in values),
                "similarity_median": round(median(similarities), 4) if similarities else 0.0,
                "similarity_min": round(min(similarities), 4) if similarities else 0.0,
                "review_failures": [
                    item["id"] for item in values if item["thai_similarity"] < 1.0
                ],
            }
        )
    return {"pages": page_summary, "items": audited}


def main() -> int:
    arguments = _arguments()
    report = asyncio.run(_run(arguments))
    destination = OUTPUT_DIRECTORY / "aliyun-thai-ocr-visual-audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for page in report["pages"]:
        print(
            f"第 {page['page_number']} 页：抽检 {page['sample_count']} 条，"
            f"完全一致 {page['exact_match_count']} 条，"
            f"最低相似度 {page['similarity_min']:.3f}"
        )
    print(f"审计结果已保存: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"OCR 审计失败: {exc}")
        raise SystemExit(1)
