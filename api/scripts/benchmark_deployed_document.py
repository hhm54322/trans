#!/usr/bin/env python3
"""Measure one document translation through a deployed META Trans instance.

This is an opt-in live benchmark: running it creates a normal translation
history item and incurs the configured model cost. The JSON report stores
timings and result metadata only, never source or translated document text.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIRECTORY = ROOT / "output" / "production-benchmarks"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Example: https://metatrans.example.com")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--source-language", choices=("auto", "zh", "th", "en"), default="auto")
    parser.add_argument("--target-language", choices=("zh", "th", "en"), required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=6 * 60 * 60)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--confirm-live-run",
        action="store_true",
        help="Required because this creates history and consumes model quota.",
    )
    return parser.parse_args()


def _snapshot(job: Dict[str, Any], elapsed_seconds: float) -> Dict[str, Any]:
    return {
        "elapsed_seconds": round(elapsed_seconds, 3),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "completed_pages": job.get("completed_pages"),
        "total_pages": job.get("total_pages"),
        "progress": job.get("progress"),
        "message": job.get("message"),
    }


def _state_key(snapshot: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(
        snapshot.get(key)
        for key in (
            "status",
            "stage",
            "completed_pages",
            "progress",
            "message",
        )
    )


def _run(arguments: argparse.Namespace) -> Dict[str, Any]:
    base_url = arguments.base_url.rstrip("/")
    started_wall = datetime.now(timezone.utc).isoformat()
    started_at = time.monotonic()
    timeline: List[Dict[str, Any]] = []
    timeout = httpx.Timeout(30.0, write=300.0)
    with httpx.Client(base_url=base_url, follow_redirects=True, timeout=timeout) as client:
        health_started = time.monotonic()
        health_response = client.get("/api/health")
        health_response.raise_for_status()
        health = health_response.json()
        health_elapsed = time.monotonic() - health_started

        media_type = mimetypes.guess_type(arguments.pdf.name)[0] or "application/pdf"
        upload_started = time.monotonic()
        with arguments.pdf.open("rb") as stream:
            response = client.post(
                "/api/translate/document/jobs",
                files={"file": (arguments.pdf.name, stream, media_type)},
                data={
                    "source_language": arguments.source_language,
                    "target_language": arguments.target_language,
                    "context": arguments.context,
                },
            )
        response.raise_for_status()
        job = response.json()
        upload_elapsed = time.monotonic() - upload_started
        job_id = str(job["job_id"])
        snapshot = _snapshot(job, time.monotonic() - started_at)
        timeline.append(snapshot)
        print(json.dumps(snapshot, ensure_ascii=False), flush=True)

        deadline = started_at + arguments.timeout_seconds
        last_key = _state_key(snapshot)
        while job.get("status") == "processing":
            if time.monotonic() >= deadline:
                raise TimeoutError(f"线上任务 {job_id} 超过基准超时")
            time.sleep(arguments.poll_seconds)
            try:
                response = client.get(f"/api/translate/document/jobs/{job_id}")
            except httpx.TimeoutException as exc:
                # Final PDF assembly can briefly keep a large CAD worker busy.
                # A lost status poll must not discard hours of benchmark data
                # after the server has already completed the translation.
                print(
                    f"状态轮询暂时超时，继续等待任务 {job_id}: "
                    f"{type(exc).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            response.raise_for_status()
            job = response.json()
            snapshot = _snapshot(job, time.monotonic() - started_at)
            current_key = _state_key(snapshot)
            if current_key != last_key:
                timeline.append(snapshot)
                last_key = current_key
                print(json.dumps(snapshot, ensure_ascii=False), flush=True)

    total_elapsed = time.monotonic() - started_at
    if job.get("status") != "completed":
        raise RuntimeError(str(job.get("error") or "线上文档翻译失败"))
    result = job.get("result") or {}
    page_count = max(1, int(job.get("total_pages") or 1))
    first_progress = next(
        (
            item["elapsed_seconds"]
            for item in timeline
            if int(item.get("completed_pages") or 0) > 0
        ),
        None,
    )
    return {
        "started_at_utc": started_wall,
        "base_url": base_url,
        "source_filename": arguments.pdf.name,
        "source_bytes": arguments.pdf.stat().st_size,
        "source_language": arguments.source_language,
        "target_language": arguments.target_language,
        "server": {
            "provider": health.get("provider"),
            "model": health.get("model"),
            "health_elapsed_seconds": round(health_elapsed, 3),
        },
        "job_id": job_id,
        "history_id": result.get("id"),
        "provider_route": result.get("provider"),
        "warning_count": len(result.get("warnings") or []),
        "page_count": page_count,
        "upload_elapsed_seconds": round(upload_elapsed, 3),
        "first_page_elapsed_seconds": first_progress,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "seconds_per_page": round(total_elapsed / page_count, 3),
        "timeline": timeline,
    }


def main() -> int:
    arguments = _arguments()
    if not arguments.confirm_live_run:
        raise SystemExit("必须显式加 --confirm-live-run；该基准会产生模型费用和线上历史")
    if not arguments.pdf.is_file() or arguments.pdf.suffix.lower() != ".pdf":
        raise SystemExit("请指定存在的 PDF 文件")
    if arguments.source_language == arguments.target_language:
        raise SystemExit("源语言和目标语言不能相同")
    if arguments.poll_seconds <= 0 or arguments.timeout_seconds <= 0:
        raise SystemExit("轮询和超时参数必须大于 0")

    report = _run(arguments)
    destination = arguments.output
    if destination is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = DEFAULT_OUTPUT_DIRECTORY / f"document-{timestamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("job_id", "page_count", "total_elapsed_seconds", "seconds_per_page")}, ensure_ascii=False, indent=2))
    print(f"基准报告已保存: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (httpx.HTTPError, RuntimeError, TimeoutError) as exc:
        print(f"线上基准失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
