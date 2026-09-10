#!/usr/bin/env python3
"""Measure quality-safe CAD visual cache opportunities without model calls."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app.main import _cad_visual_source_cache_key
from api.app.services.visual_pdf import (
    detect_dense_cad_paddle_candidates,
    extract_native_page_units,
    prepare_dense_cad_review_sheets,
    render_dense_cad_page_png,
)


DEFAULT_PDF = ROOT / "Attach_TOR_1_260805_224430.pdf"
DEFAULT_OUTPUT = (
    ROOT / "output" / "vision-benchmarks" / "cad-visual-cache-reuse.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--pages", type=int, nargs="+", default=[51, 52])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = fitz.open(args.pdf)
    seen = set()
    page_results = []
    try:
        for page_number in args.pages:
            started_at = time.perf_counter()
            page = document[page_number - 1]
            native_units = extract_native_page_units(page, page_number)
            candidates = detect_dense_cad_paddle_candidates(
                page,
                desired_width=4800,
                native_units=native_units,
                existing_bboxes=[],
                defer_existing_filter=True,
                selective_recognition=False,
            )
            rendered = render_dense_cad_page_png(
                page,
                desired_width=6400,
                minimum_render_scale=2.5,
                maximum_render_scale=5.0,
            )
            sheets = prepare_dense_cad_review_sheets(
                page,
                candidates,
                desired_width=6400,
                rows_per_sheet=3,
                rendered_page_png=rendered,
            )
            keys = [
                _cad_visual_source_cache_key(candidate)
                for sheet in sheets
                for candidate in (sheet.get("entries") or {}).values()
            ]
            reusable_keys = [key for key in keys if key]
            within_counts = Counter(reusable_keys)
            within_duplicates = sum(count - 1 for count in within_counts.values())
            cross_page_hits = sum(key in seen for key in within_counts)
            seen.update(within_counts)
            page_results.append(
                {
                    "page_number": page_number,
                    "candidate_count": len(candidates),
                    "fingerprinted_count": len(reusable_keys),
                    "within_page_duplicate_count": within_duplicates,
                    "cross_page_cache_hit_count": cross_page_hits,
                    "model_row_count_after_warm_cache": (
                        len(candidates) - within_duplicates - cross_page_hits
                    ),
                    "local_audit_elapsed_seconds": round(
                        time.perf_counter() - started_at, 3
                    ),
                }
            )
            print(json.dumps(page_results[-1], ensure_ascii=False), flush=True)
    finally:
        document.close()
    payload = {
        "source": str(args.pdf),
        "pages": page_results,
        "totals": {
            "candidate_count": sum(item["candidate_count"] for item in page_results),
            "fingerprinted_count": sum(
                item["fingerprinted_count"] for item in page_results
            ),
            "within_page_duplicate_count": sum(
                item["within_page_duplicate_count"] for item in page_results
            ),
            "cross_page_cache_hit_count": sum(
                item["cross_page_cache_hit_count"] for item in page_results
            ),
            "model_row_count_after_warm_cache": sum(
                item["model_row_count_after_warm_cache"] for item in page_results
            ),
        },
        "note": (
            "Local timing is diagnostic only. Production benefit is based on "
            "candidate reduction, not workstation elapsed time."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
