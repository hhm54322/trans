#!/usr/bin/env python3
"""Run a coordinate-preserving Alibaba Cloud Thai OCR benchmark on PDF pages.

This tool is deliberately isolated from the production translation pipeline.
It measures the official OCR service before any routing or export behavior is
changed. Credentials are read from a local AccessKey CSV or environment only.
"""

import argparse
import csv
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import fitz


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = ROOT / "Attach_TOR_1_260805_224430.pdf"
DEFAULT_CREDENTIALS = ROOT / "AccessKey.csv"
OUTPUT_DIRECTORY = ROOT / "output" / "ocr-benchmarks"
MAX_IMAGE_BYTES = 9 * 1024 * 1024
OCR_ENDPOINT = "ocr-api.cn-hangzhou.aliyuncs.com"


def _read_access_key(path: Path) -> Tuple[str, str]:
    if not path.is_file():
        raise ValueError("未找到 AccessKey CSV 文件")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream), None)
    if not row:
        raise ValueError("AccessKey CSV 没有凭证数据")
    access_key_id = (row.get("AccessKey ID") or "").strip()
    access_key_secret = (row.get("AccessKey Secret") or "").strip()
    if not access_key_id or not access_key_secret:
        raise ValueError("AccessKey CSV 缺少 AccessKey ID 或 AccessKey Secret")
    return access_key_id, access_key_secret


def _build_client(access_key_id: str, access_key_secret: str):
    try:
        from alibabacloud_ocr_api20210707.client import Client as OcrClient
        from alibabacloud_tea_openapi import models as open_api_models
    except ImportError as exc:
        raise RuntimeError(
            "缺少阿里云 OCR SDK。请执行 pip install -r api/requirements.txt"
        ) from exc

    config = open_api_models.Config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
    )
    config.endpoint = OCR_ENDPOINT
    config.connect_timeout = 10_000
    config.read_timeout = 90_000
    return OcrClient(config)


def _render_page(page: fitz.Page, long_edge: int) -> Tuple[bytes, int, int]:
    scale = max(0.1, float(long_edge) / max(page.rect.width, page.rect.height))
    for _ in range(4):
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        image = pixmap.tobytes("png")
        if len(image) <= MAX_IMAGE_BYTES:
            return image, pixmap.width, pixmap.height
        scale *= max(0.45, (MAX_IMAGE_BYTES / len(image)) ** 0.5 * 0.92)
    raise RuntimeError("页面渲染图超过阿里云 OCR 10 MB 请求上限")


def _request_ocr(client, image: bytes) -> Dict[str, Any]:
    from alibabacloud_ocr_api20210707 import models as ocr_models

    # `tai` is the official Thai language code. Request original-image points
    # so later evaluation can measure whether every label can be rewritten at
    # its source location without changing vector drawings.
    request = ocr_models.RecognizeAllTextRequest(
        type="MultiLang",
        multi_lan_config=ocr_models.RecognizeAllTextRequestMultiLanConfig(
            languages="tai"
        ),
        output_coordinate="points",
        output_oricoord=True,
        body=BytesIO(image),
    )
    response = client.recognize_all_text(request)
    return response.to_map()


def _block_summary(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    data = response.get("body", {}).get("Data", {})
    for sub_image in data.get("SubImages") or []:
        details = ((sub_image.get("BlockInfo") or {}).get("BlockDetails") or [])
        for detail in details:
            text = str(detail.get("BlockContent") or "").strip()
            if not text:
                continue
            blocks.append(
                {
                    "text": text,
                    "confidence": detail.get("BlockConfidence"),
                    "points": detail.get("BlockPoints") or [],
                    "angle": detail.get("BlockAngle"),
                }
            )
    return blocks


def _write_coordinate_overlay(
    image: bytes, blocks: Iterable[Dict[str, Any]], destination: Path
) -> None:
    """Render returned point coordinates over the original OCR image for QA."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("缺少 OpenCV，无法生成 OCR 坐标核对图") from exc

    canvas = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if canvas is None:
        raise RuntimeError("无法生成 OCR 坐标核对图")
    for block in blocks:
        points = block.get("points") or []
        if len(points) != 4:
            continue
        polygon = np.array(
            [[int(point["X"]), int(point["Y"])] for point in points],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [polygon], True, (0, 0, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError("无法写入 OCR 坐标核对图")


def _parse_pages(value: str) -> Sequence[int]:
    page_numbers = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            page_number = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("页码必须是整数") from exc
        if page_number < 1:
            raise argparse.ArgumentTypeError("页码从 1 开始")
        page_numbers.append(page_number)
    if not page_numbers:
        raise argparse.ArgumentTypeError("至少指定一页")
    return tuple(dict.fromkeys(page_numbers))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--pages", type=_parse_pages, default=(1, 2, 7, 23))
    parser.add_argument("--long-edge", type=int, default=4000)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if arguments.long_edge < 800 or arguments.long_edge > 8192:
        raise ValueError("--long-edge 必须在 800 至 8192 之间")
    if not arguments.pdf.is_file():
        raise ValueError("未找到待测 PDF")

    access_key_id, access_key_secret = _read_access_key(arguments.credentials)
    client = _build_client(access_key_id, access_key_secret)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    report: Dict[str, Any] = {
        "provider": "aliyun-ocr-multilang-tai",
        "pdf": arguments.pdf.name,
        "requested_pages": list(arguments.pages),
        "long_edge": arguments.long_edge,
        "endpoint": OCR_ENDPOINT,
        "pages": [],
    }

    document = fitz.open(arguments.pdf)
    try:
        for page_number in arguments.pages:
            if page_number > document.page_count:
                raise ValueError("指定页码超出 PDF 页数")
            page = document[page_number - 1]
            image, width, height = _render_page(page, arguments.long_edge)
            request_started_at = time.perf_counter()
            response = _request_ocr(client, image)
            elapsed_ms = round((time.perf_counter() - request_started_at) * 1000)
            blocks = _block_summary(response)
            overlay_path = OUTPUT_DIRECTORY / f"aliyun-thai-ocr-page-{page_number}.png"
            _write_coordinate_overlay(image, blocks, overlay_path)
            page_report = {
                "page_number": page_number,
                "rendered_image": {
                    "width": width,
                    "height": height,
                    "bytes": len(image),
                },
                "ocr_elapsed_ms": elapsed_ms,
                "block_count": len(blocks),
                "coordinate_overlay": str(overlay_path.relative_to(ROOT)),
                "blocks": blocks,
                "raw_response": response,
            }
            report["pages"].append(page_report)
            print(
                f"第 {page_number} 页: {len(blocks)} 个文字块, "
                f"OCR {elapsed_ms / 1000:.2f} 秒"
            )
    finally:
        document.close()

    report["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000)
    destination = OUTPUT_DIRECTORY / "aliyun-thai-ocr-benchmark.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果已保存: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"OCR 基准失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
