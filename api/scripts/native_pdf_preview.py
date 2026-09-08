import argparse
import asyncio
import json
import sys
from pathlib import Path

import fitz


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.main import _translate_pdf_document_pipeline
from app.services.exports import build_adaptive_pdf_export


def _page_range(source: Path, start: int, count: int) -> bytes:
    document = fitz.open(source)
    try:
        subset = fitz.open()
        first = max(0, start - 1)
        last = min(first + count, document.page_count) - 1
        if first > last:
            raise ValueError("页码范围超出文档")
        subset.insert_pdf(document, from_page=first, to_page=last)
        try:
            return subset.tobytes(garbage=4, deflate=True)
        finally:
            subset.close()
    finally:
        document.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--target", choices=("zh", "en"), default="zh")
    parser.add_argument("--context", default="建筑工程、教育建筑和工程造价文档")
    args = parser.parse_args()

    content = _page_range(args.source, args.start_page, args.pages)
    document = fitz.open(stream=content, filetype="pdf")
    try:
        page_count = document.page_count
    finally:
        document.close()
    _, result = await _translate_pdf_document_pipeline(
        content,
        args.source.name,
        page_count,
        "th",
        args.target,
        args.context,
    )
    segments = result.layout_segments or []
    translations = {
        segment["segment_id"]: segment["translated_text"]
        for segment in segments
    }
    translation_path = args.output.with_suffix(".translations.json")
    translation_path.parent.mkdir(parents=True, exist_ok=True)
    translation_path.write_text(
        json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    layout_path = args.output.with_suffix(".layout.json")
    layout_path.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output = build_adaptive_pdf_export(content, segments, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(
        json.dumps(
            {
                "source": str(args.source),
                "output": str(args.output),
                "pages": page_count,
                "units": len(segments),
                "provider": result.provider,
                "warnings": result.warnings,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
