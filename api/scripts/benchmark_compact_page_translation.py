#!/usr/bin/env python3
"""Benchmark one-page marked-text translation without JSON layout overhead.

This is an experiment only. It translates unique OCR strings in visual reading
order with compact anchors, then checks that each returned anchor is present
once. No source PDF is changed and no result enters the production pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app.config import settings
from api.app.services.translator import OpenAIProvider


DEFAULT_REPORT = ROOT / "output" / "ocr-benchmarks" / "aliyun-thai-ocr-benchmark.json"
OUTPUT_DIRECTORY = ROOT / "output" / "translation-benchmarks"
THAI_PATTERN = re.compile(r"[\u0E00-\u0E7F]")
ANCHOR_PATTERN = re.compile(r"\[\[m(\d+)\]\](.*?)\[\[/m\1\]\]", re.DOTALL)


def _page_entries(report_path: Path, page_number: int) -> List[Tuple[str, str]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    page = next(
        (item for item in report.get("pages") or [] if item.get("page_number") == page_number),
        None,
    )
    if page is None:
        raise ValueError(f"OCR 基准结果没有第 {page_number} 页")
    unique: Dict[str, str] = {}
    ordered = []
    for block in sorted(
        page.get("blocks") or [],
        key=lambda item: (
            min((float(point["Y"]) for point in item.get("points") or []), default=0.0),
            min((float(point["X"]) for point in item.get("points") or []), default=0.0),
        ),
    ):
        text = re.sub(r"\s+", " ", str(block.get("text") or "")).strip()
        if not text or not THAI_PATTERN.search(text):
            continue
        if text in unique:
            continue
        anchor = str(len(unique) + 1)
        unique[text] = anchor
        ordered.append((anchor, text))
    return ordered


def _payload(entries: List[Tuple[str, str]]) -> str:
    return "\n".join(f"[[m{anchor}]]{text}[[/m{anchor}]]" for anchor, text in entries)


def _instructions() -> str:
    return (
        "你是专业泰语到中文翻译引擎。输入是一页工程文件的全部泰文文字，"
        "已按视觉阅读顺序排列。理解整页上下文后准确翻译每个标记内的内容，"
        "保留原文中的英文、数字、型号和标点；不得解释、删减或补充。"
        "每个 [[m数字]] 与 [[/m数字]] 标记必须原样保留，且每个只出现一次。"
        "只能替换标记之间的文本。输出不得包含泰文字符，也不得使用 JSON 或 Markdown。"
    )


def _parse(output: str, expected_ids: List[str]) -> Dict[str, str]:
    found: Dict[str, str] = {}
    duplicate_ids = set()
    for item_id, value in ANCHOR_PATTERN.findall(output):
        if item_id in found:
            duplicate_ids.add(item_id)
        found[item_id] = value.strip()
    expected = set(expected_ids)
    if duplicate_ids or set(found) != expected or any(not found[item_id] for item_id in expected):
        raise RuntimeError(
            "ANCHOR_MISMATCH: "
            f"missing={len(expected - set(found))}, unknown={len(set(found) - expected)}, "
            f"duplicate={len(duplicate_ids)}"
        )
    return found


async def _run(arguments: argparse.Namespace) -> Dict[str, object]:
    if not settings.openai_api_key:
        raise ValueError("未配置模型 API Key")
    entries = _page_entries(arguments.report, arguments.page)
    if not entries:
        raise ValueError("未找到待翻译的泰文文字")
    source = _payload(entries)
    provider = OpenAIProvider(settings)
    started = time.perf_counter()
    print(f"compact_page_translation_started entries={len(entries)}", flush=True)
    output, route = await provider._generate_text(
        _instructions(), source, settings.openai_model
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    translated = _parse(output, [item_id for item_id, _ in entries])
    return {
        "page_number": arguments.page,
        "entry_count": len(entries),
        "source_characters": len(source),
        "translated_characters": sum(len(value) for value in translated.values()),
        "elapsed_ms": elapsed_ms,
        "route": route,
        "anchor_validation": "passed",
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--page", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    report = asyncio.run(_run(arguments))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIRECTORY / f"compact-page-{arguments.page}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"第 {arguments.page} 页：{report['entry_count']} 条，"
        f"耗时 {report['elapsed_ms'] / 1000:.2f} 秒，"
        f"锚点校验 {report['anchor_validation']}"
    )
    print(f"结果已保存: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"整页文本翻译基准失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
