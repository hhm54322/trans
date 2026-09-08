import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz
import pikepdf


# Paddle reads CPU backend flags during import. The production CentOS CPU
# cannot execute oneDNN's PIR ArrayAttribute path, so configure the backend
# before any lazy PaddleOCR import below.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("FLAGS_use_mkldnn", "0")


NATIVE_ENGINE_VERSION = 1
DENSE_VECTOR_DRAWING_THRESHOLD = 20_000
INDEXED_CAD_DRAWING_THRESHOLD = 20_000
_OCR_LOCK = threading.Lock()
_OCR_THREAD_LOCAL = threading.local()
_CAD_TEXT_DETECTOR_LOCK = threading.Lock()
_CAD_TEXT_DETECTOR = None
_TEXT_RECOGNIZER_LOCK = threading.Lock()
_TEXT_RECOGNIZER = None
_OCR_WORKER_PDF_CONTENT = None
# Each dense CAD page starts four Tesseract orientation/scale passes. Page
# concurrency can otherwise multiply this into 16 CPU-bound subprocesses;
# once they cross documents.ocr_image_text_blocks' timeout, the page silently
# loses every candidate from that pass. Keep enough parallelism to saturate a
# normal workstation without turning CPU contention into missing text.
_CAD_TESSERACT_SEMAPHORE = threading.BoundedSemaphore(4)
# Dense architectural sheets are frequently A1 or larger. Passing their
# complete 4k render through a 1600px detection model loses the smallest
# outlined labels before the vision model ever has a chance to read them.
# Overlapping tiles retain that glyph detail while remaining model-agnostic.
_CAD_DETECTION_TILE_SIDE = 1600
_CAD_DETECTION_TILE_OVERLAP = 256
_LETTER_PATTERN = re.compile(r"[A-Za-z\u0E00-\u0E7F\u4E00-\u9FFF]")
_CJK_PATTERN = re.compile(r"[\u4E00-\u9FFF]")
_THAI_LATIN_PATTERN = re.compile(r"[A-Za-z\u0E00-\u0E7F]")


def extract_native_page_units(page, page_number: int) -> List[Dict[str, Any]]:
    """Extract editable text into translation units without inspecting drawings."""
    raw = page.get_text("rawdict", sort=True)
    blocks = [block for block in raw.get("blocks", []) if block.get("type") == 0]
    parsed_blocks = []
    all_rects = []
    for block in blocks:
        lines = [_parse_native_line(line) for line in block.get("lines", [])]
        lines = [line for line in lines if line is not None]
        if not lines:
            continue
        parsed_blocks.append(lines)
        all_rects.extend(fitz.Rect(line["bbox"]) for line in lines)

    if not all_rects:
        return []
    content_rect = all_rects[0]
    for rect in all_rects[1:]:
        content_rect |= rect

    units = []
    for lines in parsed_blocks:
        for group in _group_native_lines(lines):
            unit_id = f"pdf:p{page_number}:u{len(units) + 1}"
            fragments = []
            for fragment_index, line in enumerate(group, start=1):
                fragments.append(
                    {
                        "fragment_id": f"{unit_id}:f{fragment_index}",
                        "parent_unit_id": unit_id,
                        "text": line["text"],
                        "bbox": line["bbox"],
                        "quad": line["quad"],
                        "origin": line["origin"],
                        "direction": line["direction"],
                        "wmode": line["wmode"],
                        "char_bboxes": line["char_bboxes"],
                    }
                )
            bbox = _union_rects(fitz.Rect(item["bbox"]) for item in fragments)
            sizes = [line["font_size"] for line in group]
            font_size = median(sizes)
            alignment = _native_alignment(group, bbox, content_rect)
            first_indent = _first_line_indent(group, bbox)
            text = "\n".join(line["text"] for line in group).strip()
            if not _is_translatable_text(text):
                continue
            units.append(
                {
                    "segment_id": unit_id,
                    "page_number": page_number,
                    "text": text,
                    "source_kind": "text",
                    "bbox": tuple(bbox),
                    "font_size": font_size,
                    "color": group[0]["color"],
                    "alignment": alignment,
                    "metadata": {
                        "native_pdf_version": NATIVE_ENGINE_VERSION,
                        "fragments": fragments,
                        "direction": group[0]["direction"],
                        "rotation": group[0]["rotation"],
                        "wmode": group[0]["wmode"],
                        "line_count": len(group),
                        "leading": _native_leading(group, font_size),
                        "first_line_indent": first_indent,
                        "bold": _dominant(group, "bold"),
                        "italic": _dominant(group, "italic"),
                    },
                }
            )
    return units


def extract_table_ocr_units(
    page,
    page_number: int,
    native_units: Sequence[Dict[str, Any]],
    *,
    table_regions: Optional[Sequence[fitz.Rect]] = None,
    desired_width: int = 4000,
    minimum_score: float = 0.78,
    high_accuracy: bool = False,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Use PaddleOCR for visible table text absent from the PDF text layer."""
    raster_scale = _full_page_raster_scale(page)
    scale_limit = raster_scale or 2.0
    scale = min(
        scale_limit,
        max(1.0, desired_width / max(1.0, page.rect.width)),
    )
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False
    )
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("PaddleOCR 运行环境不完整") from exc
    image = cv2.imdecode(np.frombuffer(pixmap.tobytes("png"), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("PDF 页面无法转换为 OCR 图像")
    if raster_scale:
        image = _remove_significant_blue_ink(image, scale)
    try:
        drawing_count = len(page.get_cdrawings())
    except Exception:
        drawing_count = len(page.get_drawings())
    dense_vector_page = drawing_count >= DENSE_VECTOR_DRAWING_THRESHOLD
    cad_ocr_mode = os.getenv("APP_CAD_OCR_MODE", "auto").strip().lower()
    if raster_scale and os.getenv("APP_SCAN_OCR_ENGINE", "tesseract") == "tesseract":
        from .documents import ocr_image_text_blocks

        ok, encoded = cv2.imencode(".png", image)
        blocks = (
            ocr_image_text_blocks(
                encoded.tobytes(),
                min_confidence=25.0,
                page_segmentation_mode=11,
            )
            if ok
            else []
        )
        candidates = []
        height, width = image.shape[:2]
        for block in blocks:
            value = _normalize_tesseract_source_hint(block.get("source_text") or "")
            if not re.search(r"[\u0E00-\u0E7F]", value):
                continue
            x0, y0, x1, y1 = [float(value) for value in block["bbox"]]
            polygon = [
                [x0 / 1000.0 * width, y0 / 1000.0 * height],
                [x1 / 1000.0 * width, y0 / 1000.0 * height],
                [x1 / 1000.0 * width, y1 / 1000.0 * height],
                [x0 / 1000.0 * width, y1 / 1000.0 * height],
            ]
            candidates.append(
                (
                    value,
                    float(block.get("confidence") or 0.0) / 100.0,
                    polygon,
                    int(page.rotation) % 360,
                    "tesseract-scan",
                )
            )
    elif dense_vector_page and cad_ocr_mode == "auto" and not high_accuracy:
        seed_candidates = _dense_tesseract_seed_candidates(image)
        if diagnostics is not None:
            diagnostics["seed_count"] = len(seed_candidates)
            diagnostics["seed_rects"] = [
                tuple(candidate["rect"]) for candidate in seed_candidates
            ]
            diagnostics["seed_candidates"] = [
                {
                    "rect": tuple(candidate["rect"]),
                    "source_text": candidate["source_text"],
                    "source_confidence": candidate["source_confidence"],
                }
                for candidate in seed_candidates
            ]
            diagnostics["recognized_candidate_count"] = 0
        candidates = []
    elif dense_vector_page and cad_ocr_mode != "deep" and not high_accuracy:
        candidates = _dense_fast_ocr_candidates(
            image,
            page,
            diagnostics=diagnostics,
        )
    else:
        ocr = _get_paddle_ocr()
        passes = (
            _dense_ocr_image_tiles(image)
            if dense_vector_page
            else [(image, 0, 0, "full")]
        )
        candidates = []
        with _paddle_ocr_predict_lock():
            if dense_vector_page:
                # The detector accepts a list of images. Submitting all CAD
                # tiles together avoids six independent predictor start-ups
                # while preserving the exact same 4000px tiles and thresholds.
                dense_results = list(
                    ocr.predict([pass_image for pass_image, _, _, _ in passes])
                )
                for (
                    pass_image,
                    offset_x,
                    offset_y,
                    pass_name,
                ), result in zip(passes, dense_results):
                    for text, score, polygon in zip(
                        list(result.get("rec_texts") or []),
                        [float(value) for value in (result.get("rec_scores") or [])],
                        list(result.get("rec_polys") or []),
                    ):
                        if _polygon_touches_internal_tile_edge(
                            polygon,
                            offset_x,
                            offset_y,
                            pass_image.shape[1],
                            pass_image.shape[0],
                            image.shape[1],
                            image.shape[0],
                        ):
                            continue
                        candidates.append(
                            (
                                str(text).strip(),
                                score,
                                _offset_polygon(polygon, offset_x, offset_y),
                                int(page.rotation) % 360,
                                pass_name,
                            )
                        )
            else:
                for pass_image, offset_x, offset_y, pass_name in passes:
                    results = list(ocr.predict(pass_image))
                    if results:
                        result = results[0]
                        for text, score, polygon in zip(
                            list(result.get("rec_texts") or []),
                            [float(value) for value in (result.get("rec_scores") or [])],
                            list(result.get("rec_polys") or []),
                        ):
                            candidates.append(
                                (
                                    str(text).strip(),
                                    score,
                                    _offset_polygon(polygon, offset_x, offset_y),
                                    int(page.rotation) % 360,
                                    pass_name,
                                )
                            )

                    rotated_image = cv2.rotate(pass_image, cv2.ROTATE_90_CLOCKWISE)
                    rotated_results = list(ocr.predict(rotated_image))
                    if not rotated_results:
                        continue
                    rotated_result = rotated_results[0]
                    vertical_rotation = _display_direction_to_unrotated_rotation(
                        page, (0.0, -1.0)
                    )
                    for text, score, polygon in zip(
                        list(rotated_result.get("rec_texts") or []),
                        [float(value) for value in (rotated_result.get("rec_scores") or [])],
                        list(rotated_result.get("rec_polys") or []),
                    ):
                        if not re.search(r"[\u0E00-\u0E7F]", str(text)):
                            continue
                        restored_polygon = _restore_clockwise_polygon(
                            polygon, pass_image.shape[0]
                        )
                        xs = [float(point[0]) for point in restored_polygon]
                        ys = [float(point[1]) for point in restored_polygon]
                        if max(ys) - min(ys) < (max(xs) - min(xs)) * 1.5:
                            continue
                        candidates.append(
                            (
                                str(text).strip(),
                                score,
                                _offset_polygon(restored_polygon, offset_x, offset_y),
                                vertical_rotation,
                                f"{pass_name}-rotated",
                            )
                        )
            if dense_vector_page:
                candidates.extend(_dense_vertical_header_candidates(ocr, image, page))
                candidates.extend(_dense_bottom_title_candidates(ocr, image, page))
                candidates = _recover_repeated_bottom_titles(candidates, image.shape[0])
                candidates.extend(
                    _dense_tesseract_paddle_fallback(ocr, image, page, candidates)
                )
    if not candidates:
        return []
    page_context_lines = []
    if raster_scale:
        for context_value, context_score, _, _, _ in candidates:
            context_value = re.sub(r"\s+", " ", str(context_value)).strip()
            if (
                context_score >= 0.80
                and not re.search(r"[\u0E00-\u0E7F]", context_value)
                and re.search(r"[0-9A-Za-z]", context_value)
                and len(context_value) <= 120
                and context_value not in page_context_lines
            ):
                page_context_lines.append(context_value)
            if len(page_context_lines) >= 80:
                break
    native_rects = [fitz.Rect(unit["bbox"]) for unit in native_units]
    fallback_barriers = (
        _ocr_structural_barriers(page)
        if any(str(item[4]).startswith("tesseract-local-") for item in candidates)
        else ([], [])
    )
    accepted = []
    for value, score, polygon, text_rotation, pass_name in candidates:
        value = _normalize_outline_ocr_text(value)
        thai_count = len(re.findall(r"[\u0E00-\u0E7F]", value))
        thai_consonants = re.findall(r"[\u0E01-\u0E2E]", value)
        effective_score = (
            0.45
            if str(pass_name).startswith("tesseract-batch-")
            else min(minimum_score, 0.65)
            if dense_vector_page
            else minimum_score
        )
        if (
            score < effective_score
            or thai_count == 0
            or (
                not dense_vector_page
                and score < 0.75
                and thai_count < 2
            )
            or (score < minimum_score and thai_count < 2)
            or (
                score < minimum_score
                and (
                    len(thai_consonants) < 2
                    or len(set(thai_consonants)) < 2
                )
            )
        ):
            continue
        rect = _ocr_polygon_to_unrotated_rect(
            page, polygon, image.shape[1], image.shape[0]
        )
        if str(pass_name).startswith("tesseract-local-"):
            rect = _clip_fallback_rect_at_barrier(
                rect, text_rotation, fallback_barriers
            )
        if rect.is_empty:
            continue
        if max(rect.width, rect.height) < 7.0:
            continue
        if (
            text_rotation in {0, 180}
            and rect.height > rect.width * 1.5
        ) or (
            text_rotation in {90, 270}
            and rect.width > rect.height * 1.5
        ):
            continue
        if table_regions and not any(
            region.contains(rect.tl)
            or region.contains(rect.br)
            or region.contains(fitz.Point(rect.x0 + rect.width / 2, rect.y0 + rect.height / 2))
            for region in table_regions
        ):
            continue
        if any(
            _overlap_smaller(rect, existing) >= 0.55
            for existing in native_rects
        ):
            continue
        duplicate_index = next(
            (
                existing_index
                for existing_index, existing in enumerate(accepted)
                if existing["rotation"] == text_rotation
                and _same_ocr_line(rect, existing["rect"], text_rotation)
            ),
            None,
        )
        candidate = {
            "value": value,
            "score": score,
            "rect": rect,
            "rotation": text_rotation,
            "pass_name": pass_name,
            "pixel_polygon": polygon,
        }
        if duplicate_index is None:
            accepted.append(candidate)
        elif _ocr_candidate_quality(candidate) > _ocr_candidate_quality(
            accepted[duplicate_index]
        ):
            accepted[duplicate_index] = candidate

    accepted = _suppress_circular_text_fragments(accepted, page.rect)
    if dense_vector_page and accepted:
        accepted_tuples = [
            (
                item["value"],
                item["score"],
                item["pixel_polygon"],
                item["rotation"],
                item["pass_name"],
            )
            for item in accepted
        ]
        for value, score, polygon, text_rotation, pass_name in (
            _recover_repeated_vector_labels(accepted_tuples, image)
        ):
            rect = _ocr_polygon_to_unrotated_rect(
                page, polygon, image.shape[1], image.shape[0]
            )
            if rect.is_empty or max(rect.width, rect.height) < 7.0:
                continue
            if table_regions and not any(
                region.contains(rect.tl)
                or region.contains(rect.br)
                or region.contains(
                    fitz.Point(
                        rect.x0 + rect.width / 2,
                        rect.y0 + rect.height / 2,
                    )
                )
                for region in table_regions
            ):
                continue
            if any(
                _overlap_smaller(rect, existing) >= 0.55
                for existing in native_rects
            ):
                continue
            if any(
                item["rotation"] == text_rotation
                and _same_ocr_line(rect, item["rect"], text_rotation)
                for item in accepted
            ):
                continue
            accepted.append(
                {
                    "value": value,
                    "score": score,
                    "rect": rect,
                    "rotation": text_rotation,
                    "pass_name": pass_name,
                    "pixel_polygon": polygon,
                }
            )

    if diagnostics is not None:
        diagnostics["accepted_count"] = len(accepted)

    units = []
    for index, candidate in enumerate(
        sorted(accepted, key=lambda item: (item["rect"].y0, item["rect"].x0)),
        start=1,
    ):
        value = candidate["value"]
        score = candidate["score"]
        rect = candidate["rect"]
        text_rotation = candidate["rotation"]
        unit_id = f"pdf:p{page_number}:o{index}"
        cross_size = rect.width if text_rotation in {90, 270} else rect.height
        font_size = max(3.0, cross_size * 0.72)
        units.append(
            {
                "segment_id": unit_id,
                "page_number": page_number,
                "text": value,
                "source_kind": "outline-text",
                "bbox": tuple(rect),
                "font_size": font_size,
                "color": "#000000",
                "alignment": "left",
                "metadata": {
                    "native_pdf_version": NATIVE_ENGINE_VERSION,
                    "ocr_provider": (
                        "tesseract"
                        if candidate["pass_name"] == "tesseract-scan"
                        else "paddleocr"
                    ),
                    "ocr_score": score,
                    "ocr_source_text": value,
                    "ocr_pass": candidate["pass_name"],
                    "rotation": text_rotation,
                    "line_count": 1,
                    "leading": max(font_size, rect.height),
                    "cover_bbox": list(rect),
                    "table_region": True,
                    "ocr_page_context": page_context_lines,
                },
            }
        )
    units.extend(
        _diagonal_colored_watermark_units(
            page,
            page_number,
            image,
            start_index=len(units) + 1,
        )
    )
    return units


def _full_page_raster_scale(page) -> float:
    """Return the useful source scale for a page-sized embedded scan.

    OCR at the old fixed 2x ceiling discards half the pixels of common
    300-DPI scans and misses small form labels. Vector and CAD pages return
    zero and keep the existing 2x path.
    """
    page_area = max(1.0, float(page.rect.get_area()))
    useful_scale = 0.0
    for image_info in page.get_images(full=True):
        try:
            xref = int(image_info[0])
            image_width = float(image_info[2])
            image_height = float(image_info[3])
            for image_rect in page.get_image_rects(xref):
                if image_rect.is_empty:
                    continue
                if image_rect.get_area() / page_area < 0.65:
                    continue
                useful_scale = max(
                    useful_scale,
                    image_width / max(1.0, image_rect.width),
                    image_height / max(1.0, image_rect.height),
                )
        except Exception:
            continue
    return min(4.0, useful_scale) if useful_scale >= 2.0 else 0.0


def _diagonal_colored_watermark_units(
    page,
    page_number: int,
    image,
    *,
    start_index: int,
) -> List[Dict[str, Any]]:
    """Read a diagonal outlined revision phrase framed by two colored rules.

    CAD exports commonly place revision stamps in a dedicated optional-content
    layer. Cardinal OCR misses the oblique outline text, while an axis-aligned
    cover would destroy the drawing below it. Detecting the layer and its two
    framing rules lets the exporter remove only that marked-content body and
    reconstruct the translated stamp without touching unrelated geometry.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    try:
        drawings = page.get_cdrawings()
    except Exception:
        return []
    page_area = max(1.0, page.rect.get_area())
    groups: Dict[Tuple[str, Tuple[float, float, float]], List[Dict[str, Any]]] = (
        defaultdict(list)
    )
    for drawing in drawings:
        color = drawing.get("color") or drawing.get("fill")
        layer = str(drawing.get("layer") or "").strip()
        if not layer or not color or len(color) < 3:
            continue
        rgb = tuple(float(value) for value in color[:3])
        if max(rgb) - min(rgb) < 0.15:
            continue
        groups[(layer, tuple(round(value, 3) for value in rgb))].append(drawing)

    output = []
    for (layer, rounded_color), group in groups.items():
        long_segments = []
        for drawing in group:
            rect = fitz.Rect(drawing.get("rect") or ())
            if rect.is_empty or rect.get_area() < page_area * 0.18:
                continue
            segment = _longest_drawing_line(drawing)
            if segment is None:
                continue
            start, end = segment
            length = math.hypot(end.x - start.x, end.y - start.y)
            if length >= math.hypot(page.rect.width, page.rect.height) * 0.55:
                long_segments.append((length, start, end))
        if len(long_segments) < 2 or len(group) < 10:
            continue
        first, second = sorted(
            long_segments, key=lambda item: item[0], reverse=True
        )[:2]
        line_a = _orient_line_left_to_right(first[1], first[2])
        line_b = _orient_line_left_to_right(second[1], second[2])
        vector_a = fitz.Point(line_a[1].x - line_a[0].x, line_a[1].y - line_a[0].y)
        vector_b = fitz.Point(line_b[1].x - line_b[0].x, line_b[1].y - line_b[0].y)
        length_a = math.hypot(vector_a.x, vector_a.y)
        length_b = math.hypot(vector_b.x, vector_b.y)
        cosine = (vector_a.x * vector_b.x + vector_a.y * vector_b.y) / max(
            1.0, length_a * length_b
        )
        angle = math.degrees(math.atan2(vector_a.y, vector_a.x))
        if cosine < 0.985 or not 10.0 <= abs(angle) <= 80.0:
            continue
        middle_a = fitz.Point(
            (line_a[0].x + line_a[1].x) / 2,
            (line_a[0].y + line_a[1].y) / 2,
        )
        middle_b = fitz.Point(
            (line_b[0].x + line_b[1].x) / 2,
            (line_b[0].y + line_b[1].y) / 2,
        )
        separation = abs(
            (middle_b.x - middle_a.x) * (-vector_a.y / length_a)
            + (middle_b.y - middle_a.y) * (vector_a.x / length_a)
        )
        if not page.rect.width * 0.015 <= separation <= page.rect.width * 0.12:
            continue
        marked_content_tag = _optional_content_tag(page, layer)
        if not marked_content_tag:
            continue

        image_height, image_width = image.shape[:2]
        target_rgb = np.asarray(rounded_color, dtype=np.float32)
        target_bgr = target_rgb[::-1] * 255.0
        pixels = image.astype(np.float32)
        distance = np.linalg.norm(pixels - target_bgr, axis=2)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        target_pixel = np.uint8([[target_bgr]])
        target_hue = int(cv2.cvtColor(target_pixel, cv2.COLOR_BGR2HSV)[0, 0, 0])
        hue = hsv[:, :, 0].astype(np.int16)
        hue_distance = np.minimum(abs(hue - target_hue), 180 - abs(hue - target_hue))
        mask = np.where(
            ((distance <= 95.0) | (hue_distance <= 8))
            & (hsv[:, :, 1] >= 25),
            255,
            0,
        ).astype(np.uint8)
        display_line_a = _orient_line_left_to_right(
            line_a[0] * page.rotation_matrix,
            line_a[1] * page.rotation_matrix,
        )
        display_vector = fitz.Point(
            display_line_a[1].x - display_line_a[0].x,
            display_line_a[1].y - display_line_a[0].y,
        )
        display_angle = math.degrees(
            math.atan2(display_vector.y, display_vector.x)
        )
        center = (image_width / 2.0, image_height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, display_angle, 1.0)
        rotated = cv2.warpAffine(
            mask,
            matrix,
            (image_width, image_height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )

        def transformed_y(point: fitz.Point) -> float:
            display_point = point * page.rotation_matrix
            x = display_point.x / max(1.0, page.rect.width) * image_width
            y = display_point.y / max(1.0, page.rect.height) * image_height
            return float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])

        line_y = sorted(
            [
                (transformed_y(line_a[0]) + transformed_y(line_a[1])) / 2,
                (transformed_y(line_b[0]) + transformed_y(line_b[1])) / 2,
            ]
        )
        inset = max(4, int(round((line_y[1] - line_y[0]) * 0.06)))
        top = max(0, int(math.ceil(line_y[0])) + inset)
        bottom = min(image_height, int(math.floor(line_y[1])) - inset)
        if bottom - top < 20:
            continue
        text_mask = rotated[top:bottom]
        text_mask = cv2.dilate(
            text_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)),
            iterations=1,
        )
        ys, xs = np.where(text_mask > 0)
        if not len(xs):
            continue
        left = max(0, int(xs.min()) - 20)
        right = min(image_width, int(xs.max()) + 21)
        text_crop = 255 - text_mask[:, left:right]
        ok, encoded = cv2.imencode(".png", text_crop)
        if not ok:
            continue
        executable = shutil.which("tesseract")
        if not executable:
            continue
        try:
            with _CAD_TESSERACT_SEMAPHORE:
                recognized = subprocess.run(
                    [
                        executable,
                        "stdin",
                        "stdout",
                        "-l",
                        "tha+eng",
                        "--psm",
                        "13",
                    ],
                    input=encoded.tobytes(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=30,
                )
        except (OSError, subprocess.SubprocessError):
            continue
        if recognized.returncode != 0:
            continue
        source_text = re.sub(
            r"\s+",
            " ",
            recognized.stdout.decode("utf-8", errors="ignore"),
        ).strip()
        if len(re.findall(r"[\u0E01-\u0E2E]", source_text)) < 5:
            continue
        baseline_start = fitz.Point(
            (line_a[0].x + line_b[0].x) / 2,
            (line_a[0].y + line_b[0].y) / 2,
        )
        baseline_end = fitz.Point(
            (line_a[1].x + line_b[1].x) / 2,
            (line_a[1].y + line_b[1].y) / 2,
        )
        band_rect = fitz.Rect(
            min(line_a[0].x, line_a[1].x, line_b[0].x, line_b[1].x),
            min(line_a[0].y, line_a[1].y, line_b[0].y, line_b[1].y),
            max(line_a[0].x, line_a[1].x, line_b[0].x, line_b[1].x),
            max(line_a[0].y, line_a[1].y, line_b[0].y, line_b[1].y),
        )
        output.append(
            {
                "segment_id": f"pdf:p{page_number}:w{start_index + len(output)}",
                "page_number": page_number,
                "text": source_text,
                "source_kind": "outline-text",
                "bbox": tuple(band_rect),
                "font_size": max(8.0, separation * 0.48),
                "color": "#" + "".join(
                    f"{max(0, min(255, round(value * 255))):02x}"
                    for value in rounded_color
                ),
                "alignment": "center",
                "metadata": {
                    "visual_pdf_version": 1,
                    "translation_unit": "complete-line",
                    "ocr_provider": "tesseract-diagonal-layer",
                    "rotation": angle,
                    "line_count": 1,
                    "leading": max(8.0, separation * 0.48),
                    "diagonal_watermark": True,
                    "marked_content_tag": marked_content_tag,
                    "watermark_lines": [
                        [[line_a[0].x, line_a[0].y], [line_a[1].x, line_a[1].y]],
                        [[line_b[0].x, line_b[0].y], [line_b[1].x, line_b[1].y]],
                    ],
                    "baseline_start": [baseline_start.x, baseline_start.y],
                    "baseline_end": [baseline_end.x, baseline_end.y],
                },
            }
        )
    return output


def _longest_drawing_line(drawing) -> Optional[Tuple[fitz.Point, fitz.Point]]:
    longest = None
    for item in drawing.get("items") or []:
        if not item or item[0] != "l":
            continue
        start, end = fitz.Point(item[1]), fitz.Point(item[2])
        length = math.hypot(end.x - start.x, end.y - start.y)
        if longest is None or length > longest[0]:
            longest = (length, start, end)
    return None if longest is None else (longest[1], longest[2])


def _orient_line_left_to_right(
    start: fitz.Point, end: fitz.Point
) -> Tuple[fitz.Point, fitz.Point]:
    if end.x < start.x or (end.x == start.x and end.y < start.y):
        return end, start
    return start, end


def _optional_content_tag(page, layer_name: str) -> Optional[str]:
    try:
        for tag, xref, _kind in page.get_oc_items():
            value_type, value = page.parent.xref_get_key(int(xref), "Name")
            if value_type == "string" and value == layer_name:
                return str(tag)
    except Exception:
        return None
    return None


def _remove_significant_blue_ink(image, scale: float):
    """Hide handwritten blue marks from OCR while retaining printed text."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_core = np.where(
        (hsv[:, :, 0] >= 90)
        & (hsv[:, :, 0] <= 140)
        & (hsv[:, :, 1] >= 70)
        & (hsv[:, :, 2] <= 245),
        255,
        0,
    ).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        blue_core,
        connectivity=8,
    )
    minimum_area = max(18, int(round(scale * scale * 5.0)))
    handwriting = np.zeros_like(blue_core)
    for component_index in range(1, component_count):
        if int(stats[component_index, cv2.CC_STAT_AREA]) >= minimum_area:
            handwriting[labels == component_index] = 255
    if not np.any(handwriting):
        return image
    handwriting = cv2.dilate(
        handwriting,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    blue_edges = np.where(
        (hsv[:, :, 0] >= 85)
        & (hsv[:, :, 0] <= 145)
        & (hsv[:, :, 1] >= 20)
        & (hsv[:, :, 2] <= 250),
        255,
        0,
    ).astype(np.uint8)
    handwriting = cv2.bitwise_and(handwriting, blue_edges)
    cleaned = image.copy()
    cleaned[handwriting > 0] = 255
    return cleaned


def _dense_ocr_image_tiles(image):
    """Keep small vector text above PaddleOCR's detector downscale limit."""
    height, width = image.shape[:2]
    columns, rows = 3, 2
    overlap_x = max(24, int(width / columns * 0.32))
    overlap_y = max(24, int(height / rows * 0.16))
    tiles = []
    for row in range(rows):
        base_y0 = round(row * height / rows)
        base_y1 = round((row + 1) * height / rows)
        y0 = max(0, base_y0 - (overlap_y if row else 0))
        y1 = min(height, base_y1 + (overlap_y if row < rows - 1 else 0))
        for column in range(columns):
            base_x0 = round(column * width / columns)
            base_x1 = round((column + 1) * width / columns)
            x0 = max(0, base_x0 - (overlap_x if column else 0))
            x1 = min(width, base_x1 + (overlap_x if column < columns - 1 else 0))
            tiles.append((image[y0:y1, x0:x1], x0, y0, f"tile-r{row + 1}c{column + 1}"))
    return tiles


def _dense_fast_ocr_candidates(
    image,
    page,
    *,
    batch_size: int = 32,
    diagnostics: Optional[Dict[str, Any]] = None,
):
    """Recognize Tesseract-located CAD lines in one batched Paddle pass."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("PaddleOCR 运行环境不完整") from exc

    seed_rects = _dense_tesseract_seed_rects(image)
    if diagnostics is not None:
        diagnostics["seed_count"] = len(seed_rects)
        # Keep the same pixel-space candidates for the selective vision gap
        # pass. The caller renders at the same scale, so there is no reason to
        # run Tesseract a second time for the page.
        diagnostics["seed_rects"] = [tuple(rect) for rect in seed_rects]
    if not seed_rects:
        return []
    height, width = image.shape[:2]
    crops = []
    prepared = []
    page_rotation = int(page.rotation) % 360
    vertical_rotation = _display_direction_to_unrotated_rotation(
        page, (0.0, -1.0)
    )
    for seed_index, seed in enumerate(seed_rects, start=1):
        rect = fitz.Rect(seed) & fitz.Rect(0, 0, width, height)
        vertical = rect.height > rect.width * 1.5
        pad_x = max(3.0, rect.width * (0.08 if vertical else 0.025))
        pad_y = max(3.0, rect.height * (0.025 if vertical else 0.18))
        crop_rect = (rect + (-pad_x, -pad_y, pad_x, pad_y)) & fitz.Rect(
            0, 0, width, height
        )
        x0 = max(0, int(math.floor(crop_rect.x0)))
        y0 = max(0, int(math.floor(crop_rect.y0)))
        x1 = min(width, int(math.ceil(crop_rect.x1)))
        y1 = min(height, int(math.ceil(crop_rect.y1)))
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        if vertical:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        recognition_scale = min(
            4.0,
            max(1.0, 56.0 / max(1, crop.shape[0])),
        )
        if recognition_scale > 1.01:
            crop = cv2.resize(
                crop,
                None,
                fx=recognition_scale,
                fy=recognition_scale,
                interpolation=cv2.INTER_CUBIC,
            )
        crop = cv2.copyMakeBorder(
            crop,
            6,
            6,
            8,
            8,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        crops.append(crop)
        prepared.append(
            (
                seed_index,
                rect,
                vertical_rotation if vertical else page_rotation,
            )
        )

    if not crops:
        return []
    recognizer = _get_text_recognizer()
    with _TEXT_RECOGNIZER_LOCK:
        results = list(recognizer.predict(crops, batch_size=max(1, batch_size)))
    candidates = []
    for (seed_index, rect, rotation), result in zip(prepared, results):
        value = str(result.get("rec_text") or "").strip()
        score = float(result.get("rec_score") or 0.0)
        if not re.search(r"[\u0E00-\u0E7F]", value):
            continue
        polygon = [
            [rect.x0, rect.y0],
            [rect.x1, rect.y0],
            [rect.x1, rect.y1],
            [rect.x0, rect.y1],
        ]
        candidates.append(
            (
                value,
                score,
                polygon,
                rotation,
                f"tesseract-batch-{seed_index}",
            )
        )
    if diagnostics is not None:
        diagnostics["recognized_candidate_count"] = len(candidates)
    return candidates


def prepare_dense_cad_translation_sheets(
    page,
    page_number: int,
    *,
    desired_width: int = 4000,
    minimum_render_scale: float = 1.0,
    rows_per_sheet: int = 12,
    row_height: int = 120,
    native_units: Optional[Sequence[Dict[str, Any]]] = None,
    seed_rects: Optional[Sequence[Sequence[float]]] = None,
    seed_candidates: Optional[Sequence[Dict[str, Any]]] = None,
    detection_provider: str = "tesseract",
):
    """Render indexed CAD line crops for GPT recognition and translation."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("CAD 索引图运行环境不完整") from exc

    if minimum_render_scale <= 0:
        raise ValueError("CAD 索引图最小渲染倍率必须大于 0")
    scale = min(
        2.0,
        max(minimum_render_scale, desired_width / max(1.0, page.rect.width)),
    )
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False
    )
    image = cv2.imdecode(
        np.frombuffer(pixmap.tobytes("png"), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise RuntimeError("PDF 页面无法转换为 CAD 索引图")
    supplemental_units = _diagonal_colored_watermark_units(
        page,
        page_number,
        image,
        start_index=1,
    )
    if seed_candidates is not None:
        located_seeds = [
            {
                "rect": fitz.Rect(candidate["rect"]),
                "source_text": str(candidate.get("source_text") or "").strip(),
                "source_confidence": float(
                    candidate.get("source_confidence") or 0.0
                ),
            }
            for candidate in seed_candidates
        ]
    elif seed_rects is None and detection_provider == "paddle-detector":
        located_seeds = _dense_paddle_detection_seeds(image)
    elif seed_rects is None and detection_provider == "tesseract":
        located_seeds = _dense_tesseract_seed_candidates(image, thai_only=True)
    elif seed_rects is not None:
        located_seeds = [
            {
                "rect": fitz.Rect(rect),
                "source_text": "",
                "source_confidence": 0.0,
            }
            for rect in seed_rects
        ]
    else:
        raise ValueError(f"不支持的 CAD 文字检测器: {detection_provider}")
    if not located_seeds:
        return (
            [{"content": b"", "entries": {}, "supplemental_units": supplemental_units}]
            if supplemental_units
            else []
        )

    if native_units is None:
        native_units = extract_native_page_units(page, page_number)
    native_rects = [fitz.Rect(unit["bbox"]) for unit in native_units]
    image_height, image_width = image.shape[:2]
    page_rotation = int(page.rotation) % 360
    vertical_rotation = _display_direction_to_unrotated_rotation(
        page, (0.0, -1.0)
    )
    candidates = []
    for seed in located_seeds:
        rect = fitz.Rect(seed["rect"])
        polygon = [
            [rect.x0, rect.y0],
            [rect.x1, rect.y0],
            [rect.x1, rect.y1],
            [rect.x0, rect.y1],
        ]
        page_rect = _ocr_polygon_to_unrotated_rect(
            page, polygon, image_width, image_height
        )
        if page_rect.is_empty or max(page_rect.width, page_rect.height) < 7.0:
            continue
        if any(_overlap_smaller(page_rect, native) >= 0.55 for native in native_rects):
            continue
        vertical = rect.height > rect.width * 1.5
        candidates.append(
            {
                "pixel_rect": fitz.Rect(rect),
                "bbox": tuple(page_rect),
                    "rotation": vertical_rotation if vertical else page_rotation,
                    "vertical": vertical,
                    "source_hint": seed["source_text"],
                    "source_confidence": float(seed.get("source_confidence") or 0.0),
                }
            )

    if detection_provider == "paddle-detector":
        # Paddle does not promise an output ordering. Stable spatial IDs make
        # a retry, a model response, and a QA finding refer to the same CAD
        # label across independent detector runs.
        candidates.sort(
            key=lambda item: (
                round(item["pixel_rect"].y0, 2),
                round(item["pixel_rect"].x0, 2),
                round(item["pixel_rect"].y1, 2),
                round(item["pixel_rect"].x1, 2),
            )
        )
    sheets = []
    # Keep each source line legible after the vision provider resizes tall
    # images. With 20 rows this produces a 1800 x 2400 sheet, rather than
    # squeezing the same text into the old 1800 x 3200, 40-row sheet.
    row_height = max(80, int(row_height))
    label_width = 110
    sheet_width = 1800
    for sheet_index in range(0, len(candidates), max(1, rows_per_sheet)):
        group = candidates[sheet_index : sheet_index + max(1, rows_per_sheet)]
        canvas = np.full(
            (row_height * len(group), sheet_width, 3), 255, dtype=np.uint8
        )
        entries = {}
        for row_index, candidate in enumerate(group, start=1):
            rect = candidate["pixel_rect"]
            x0 = max(0, int(math.floor(rect.x0)) - 3)
            y0 = max(0, int(math.floor(rect.y0)) - 3)
            x1 = min(image_width, int(math.ceil(rect.x1)) + 3)
            y1 = min(image_height, int(math.ceil(rect.y1)) + 3)
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            if candidate["vertical"]:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            crop_scale = min(
                4.0,
                (row_height * 0.8) / max(1, crop.shape[0]),
                (sheet_width - label_width - 40.0) / max(1, crop.shape[1]),
            )
            crop = cv2.resize(
                crop,
                None,
                fx=crop_scale,
                fy=crop_scale,
                interpolation=cv2.INTER_CUBIC,
            )
            top = (row_index - 1) * row_height + (row_height - crop.shape[0]) // 2
            canvas[
                top : top + crop.shape[0],
                label_width : label_width + crop.shape[1],
            ] = crop
            item_id = f"ID{row_index:03d}"
            candidate["sheet_row"] = row_index - 1
            cv2.putText(
                canvas,
                item_id,
                (4, (row_index - 1) * row_height + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            entries[item_id] = candidate
        ok, encoded = cv2.imencode(".png", canvas)
        if ok and entries:
            sheets.append(
                {
                    "content": encoded.tobytes(),
                    "entries": entries,
                    "row_height": row_height,
                    "label_width": label_width,
                }
            )
    if supplemental_units:
        if sheets:
            sheets[0]["supplemental_units"] = supplemental_units
        else:
            sheets.append(
                {
                    "content": b"",
                    "entries": {},
                    "supplemental_units": supplemental_units,
                }
            )
    return sheets


def subset_indexed_translation_sheet(
    sheet: Dict[str, Any], item_ids: Sequence[str]
) -> Dict[str, Any]:
    """Build a compact retry image containing only unresolved CAD rows."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("CAD 索引图运行环境不完整") from exc
    image = cv2.imdecode(
        np.frombuffer(sheet.get("content") or b"", dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise RuntimeError("CAD 索引图无法生成定向重试图")
    row_height = int(sheet.get("row_height") or 80)
    entries = sheet.get("entries") or {}
    selected = [item_id for item_id in item_ids if item_id in entries]
    if not selected:
        return {"content": b"", "entries": {}}
    rows = []
    retry_entries = {}
    for retry_row, item_id in enumerate(selected):
        candidate = entries[item_id]
        source_row = int(candidate.get("sheet_row") or 0)
        y0 = source_row * row_height
        y1 = min(image.shape[0], y0 + row_height)
        if y1 <= y0:
            continue
        rows.append(image[y0:y1].copy())
        retry_candidate = dict(candidate)
        retry_candidate["sheet_row"] = retry_row
        retry_entries[item_id] = retry_candidate
    if not rows:
        return {"content": b"", "entries": {}}
    retry_image = np.concatenate(rows, axis=0)
    ok, encoded = cv2.imencode(".png", retry_image)
    if not ok:
        raise RuntimeError("CAD 索引图无法生成定向重试图")
    return {
        "content": encoded.tobytes(),
        "entries": retry_entries,
        "row_height": row_height,
        "label_width": int(sheet.get("label_width") or 110),
    }


def detect_dense_cad_paddle_candidates(
    page,
    *,
    desired_width: int = 4800,
    native_units: Optional[Sequence[Dict[str, Any]]] = None,
    existing_bboxes: Optional[Sequence[Sequence[float]]] = None,
    defer_existing_filter: bool = False,
) -> List[Dict[str, Any]]:
    """Find CAD text missed by Tesseract using one full-page Paddle pass.

    Native text is independent of the indexed Tesseract candidates. Callers
    that build those candidates concurrently can defer just the final
    candidate-to-candidate comparison without changing Paddle's input or its
    detected text boxes.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("PaddleOCR 运行环境不完整") from exc

    scale = min(2.0, max(1.0, desired_width / max(1.0, page.rect.width)))
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False
    )
    image = cv2.imdecode(
        np.frombuffer(pixmap.tobytes("png"), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise RuntimeError("PDF 页面无法转换为 CAD 检测图")
    ocr = _get_paddle_ocr()
    with _paddle_ocr_predict_lock():
        results = list(ocr.predict(image))
    if not results:
        return []

    result = results[0]
    native_rects = [
        fitz.Rect(unit["bbox"])
        for unit in native_units or []
        if isinstance(unit.get("bbox"), (list, tuple)) and len(unit["bbox"]) == 4
    ]
    page_rotation = int(page.rotation) % 360
    vertical_rotation = _display_direction_to_unrotated_rotation(
        page, (0.0, -1.0)
    )
    candidates = []
    for text, score, polygon in zip(
        list(result.get("rec_texts") or []),
        [float(value) for value in (result.get("rec_scores") or [])],
        list(result.get("rec_polys") or []),
    ):
        source_text = _normalize_tesseract_source_hint(str(text or ""))
        if score < 0.40 or not re.search(r"[\u0E00-\u0E7F]", source_text):
            continue
        pixel_rect = _polygon_rect(polygon)
        if pixel_rect.is_empty:
            continue
        page_rect = _ocr_polygon_to_unrotated_rect(
            page, polygon, image.shape[1], image.shape[0]
        )
        if page_rect.is_empty or max(page_rect.width, page_rect.height) < 7.0:
            continue
        if any(_overlap_smaller(page_rect, rect) >= 0.50 for rect in native_rects):
            continue
        vertical = pixel_rect.height > pixel_rect.width * 1.5
        candidate = {
            "bbox": tuple(page_rect),
            "rotation": vertical_rotation if vertical else page_rotation,
            "vertical": vertical,
            "source_hint": source_text,
            "source_confidence": score * 100.0,
            "candidate_provider": "paddle-full-page",
        }
        candidates.append(candidate)
    if defer_existing_filter:
        return candidates
    return filter_dense_cad_paddle_candidates(
        candidates,
        existing_bboxes=existing_bboxes,
        page_rotation=page_rotation,
    )


def filter_dense_cad_paddle_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    existing_bboxes: Optional[Sequence[Sequence[float]]] = None,
    page_rotation: int = 0,
) -> List[Dict[str, Any]]:
    """Apply the indexed-candidate overlap rule after Paddle detection.

    This is deliberately separate from Paddle inference so the detector can
    run while Tesseract creates indexed candidates. The logic is identical to
    the original in-loop comparison and remains the sole authority on whether
    a Paddle candidate is new or merely provides a larger cover rectangle.
    """
    covered_rects = [fitz.Rect(values) for values in existing_bboxes or []]
    filtered = []
    for original in candidates:
        candidate = dict(original)
        page_rect = fitz.Rect(candidate["bbox"])
        matching_existing_index = next(
            (
                index
                for index, rect in enumerate(covered_rects)
                if _same_ocr_line(page_rect, rect, page_rotation)
            ),
            None,
        )
        if matching_existing_index is not None:
            existing_rect = covered_rects[matching_existing_index]
            paddle_along = page_rect.width
            existing_along = existing_rect.width
            materially_larger = (
                page_rect.get_area() >= existing_rect.get_area() * 1.18
                or paddle_along >= existing_along * 1.12
            )
            if not materially_larger:
                continue
            candidate["matching_existing_index"] = matching_existing_index
        filtered.append(candidate)
    return _deduplicate_page_candidates(filtered)


def _deduplicate_page_candidates(
    candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output = []
    for candidate in sorted(
        candidates,
        key=lambda item: fitz.Rect(item["bbox"]).get_area(),
        reverse=True,
    ):
        rect = fitz.Rect(candidate["bbox"])
        if any(
            int(existing.get("rotation") or 0) % 360
            == int(candidate.get("rotation") or 0) % 360
            and _same_ocr_line(rect, fitz.Rect(existing["bbox"]), int(candidate["rotation"]) % 360)
            for existing in output
        ):
            continue
        output.append(dict(candidate))
    return output


def prepare_dense_cad_review_sheets(
    page,
    candidates: Sequence[Dict[str, Any]],
    *,
    desired_width: int = 6400,
    rows_per_sheet: int = 3,
    wide_context: bool = False,
) -> List[Dict[str, Any]]:
    """Render unresolved CAD lines sharply with marked surrounding context.

    ``wide_context`` retains the normal enlarged target and adds a second,
    broader panel for the rare labels whose glyphs require nearby CAD context
    to read correctly. The red frame remains the only text eligible for
    translation.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("CAD 复核图运行环境不完整") from exc

    if not candidates:
        return []
    scale = min(5.0, max(2.5, desired_width / max(1.0, page.rect.width)))
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False
    )
    image = cv2.imdecode(
        np.frombuffer(pixmap.tobytes("png"), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise RuntimeError("PDF 页面无法转换为 CAD 高清复核图")

    image_height, image_width = image.shape[:2]
    display_scale_x = image_width / max(1.0, page.rect.width)
    display_scale_y = image_height / max(1.0, page.rect.height)
    prepared = []
    for candidate_index, source in enumerate(candidates, start=1):
        display_rect = fitz.Rect(source["bbox"]) * page.rotation_matrix
        target = fitz.Rect(
            display_rect.x0 * display_scale_x,
            display_rect.y0 * display_scale_y,
            display_rect.x1 * display_scale_x,
            display_rect.y1 * display_scale_y,
        )
        target &= fitz.Rect(0, 0, image_width, image_height)
        if target.is_empty:
            continue
        vertical = bool(source.get("vertical"))
        cross_size = target.width if vertical else target.height
        along_size = target.height if vertical else target.width
        cross_padding = max(28.0, cross_size * 1.8)
        along_padding = max(70.0, min(240.0, along_size * 0.12))
        if vertical:
            crop_rect = fitz.Rect(
                target.x0 - cross_padding,
                target.y0 - along_padding,
                target.x1 + cross_padding,
                target.y1 + along_padding,
            )
        else:
            crop_rect = fitz.Rect(
                target.x0 - along_padding,
                target.y0 - cross_padding,
                target.x1 + along_padding,
                target.y1 + cross_padding,
            )
        crop_rect &= fitz.Rect(0, 0, image_width, image_height)
        x0, y0, x1, y1 = (
            int(math.floor(crop_rect.x0)),
            int(math.floor(crop_rect.y0)),
            int(math.ceil(crop_rect.x1)),
            int(math.ceil(crop_rect.y1)),
        )
        crop = image[y0:y1, x0:x1].copy()
        if crop.size == 0:
            continue
        local_target = fitz.Rect(
            target.x0 - x0,
            target.y0 - y0,
            target.x1 - x0,
            target.y1 - y0,
        )
        border_padding = max(3, round(cross_size * 0.18))
        cv2.rectangle(
            crop,
            (
                max(0, int(math.floor(local_target.x0)) - border_padding),
                max(0, int(math.floor(local_target.y0)) - border_padding),
            ),
            (
                min(crop.shape[1] - 1, int(math.ceil(local_target.x1)) + border_padding),
                min(crop.shape[0] - 1, int(math.ceil(local_target.y1)) + border_padding),
            ),
            (0, 0, 230),
            max(2, round(scale * 0.45)),
            cv2.LINE_AA,
        )
        if vertical:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        target_line_height = max(1.0, cross_size)
        crop_scale = min(
            5.0,
            96.0 / target_line_height,
            2200.0 / max(1, crop.shape[1]),
        )
        if crop_scale != 1.0:
            crop = cv2.resize(
                crop,
                None,
                fx=crop_scale,
                fy=crop_scale,
                interpolation=cv2.INTER_CUBIC,
            )
        context_crop = None
        if wide_context:
            context_along_padding = max(480.0, min(1800.0, along_size * 8.0))
            context_cross_padding = max(320.0, min(1200.0, cross_size * 16.0))
            if vertical:
                context_rect = fitz.Rect(
                    target.x0 - context_cross_padding,
                    target.y0 - context_along_padding,
                    target.x1 + context_cross_padding,
                    target.y1 + context_along_padding,
                )
            else:
                context_rect = fitz.Rect(
                    target.x0 - context_along_padding,
                    target.y0 - context_cross_padding,
                    target.x1 + context_along_padding,
                    target.y1 + context_cross_padding,
                )
            context_rect &= fitz.Rect(0, 0, image_width, image_height)
            cx0, cy0, cx1, cy1 = (
                int(math.floor(context_rect.x0)),
                int(math.floor(context_rect.y0)),
                int(math.ceil(context_rect.x1)),
                int(math.ceil(context_rect.y1)),
            )
            context_crop = image[cy0:cy1, cx0:cx1].copy()
            if context_crop.size:
                local_context_target = fitz.Rect(
                    target.x0 - cx0,
                    target.y0 - cy0,
                    target.x1 - cx0,
                    target.y1 - cy0,
                )
                cv2.rectangle(
                    context_crop,
                    (
                        max(0, int(math.floor(local_context_target.x0)) - border_padding),
                        max(0, int(math.floor(local_context_target.y0)) - border_padding),
                    ),
                    (
                        min(
                            context_crop.shape[1] - 1,
                            int(math.ceil(local_context_target.x1)) + border_padding,
                        ),
                        min(
                            context_crop.shape[0] - 1,
                            int(math.ceil(local_context_target.y1)) + border_padding,
                        ),
                    ),
                    (0, 0, 230),
                    max(2, round(scale * 0.45)),
                    cv2.LINE_AA,
                )
            else:
                context_crop = None
        prepared.append(
            {
                "review_id": f"RID{candidate_index:04d}",
                "crop": crop,
                "context_crop": context_crop,
                "candidate": dict(source),
            }
        )

    row_height = 460 if wide_context else 280
    label_width = 150
    sheet_width = 3600 if wide_context else 2400
    sheets = []
    for offset in range(0, len(prepared), max(1, rows_per_sheet)):
        group = prepared[offset : offset + max(1, rows_per_sheet)]
        canvas = np.full(
            (row_height * len(group), sheet_width, 3), 255, dtype=np.uint8
        )
        entries = {}
        for row_index, item in enumerate(group):
            crop = item["crop"]
            maximum_height = row_height - 40 if wide_context else row_height - 20
            maximum_width = (
                1700 if wide_context else sheet_width - label_width - 20
            )
            fit_scale = min(
                1.0,
                maximum_height / max(1, crop.shape[0]),
                maximum_width / max(1, crop.shape[1]),
            )
            if fit_scale < 1.0:
                crop = cv2.resize(
                    crop,
                    None,
                    fx=fit_scale,
                    fy=fit_scale,
                    interpolation=cv2.INTER_AREA,
                )
            top = row_index * row_height + (row_height - crop.shape[0]) // 2
            canvas[
                top : top + crop.shape[0],
                label_width : label_width + crop.shape[1],
            ] = crop
            context_crop = item.get("context_crop")
            if wide_context and context_crop is not None and context_crop.size:
                context_max_height = row_height - 40
                context_max_width = sheet_width - label_width - 1760
                context_scale = min(
                    1.0,
                    context_max_height / max(1, context_crop.shape[0]),
                    context_max_width / max(1, context_crop.shape[1]),
                )
                if context_scale < 1.0:
                    context_crop = cv2.resize(
                        context_crop,
                        None,
                        fx=context_scale,
                        fy=context_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                context_left = label_width + 1740
                context_top = row_index * row_height + (
                    row_height - context_crop.shape[0]
                ) // 2
                canvas[
                    context_top : context_top + context_crop.shape[0],
                    context_left : context_left + context_crop.shape[1],
                ] = context_crop
                cv2.putText(
                    canvas,
                    "CONTEXT",
                    (context_left, row_index * row_height + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            review_id = item["review_id"]
            candidate = item["candidate"]
            candidate["sheet_row"] = row_index
            cv2.putText(
                canvas,
                review_id,
                (4, row_index * row_height + row_height // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.74,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            if wide_context:
                cv2.putText(
                    canvas,
                    "TARGET",
                    (label_width, row_index * row_height + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            entries[review_id] = candidate
        ok, encoded = cv2.imencode(".png", canvas)
        if ok and entries:
            sheets.append(
                {
                    "content": encoded.tobytes(),
                    "entries": entries,
                    "row_height": row_height,
                    "label_width": label_width,
                    "focused_review": True,
                    "wide_context": wide_context,
                }
            )
    return sheets


def recognize_indexed_sheet_rows(
    sheet: Dict[str, Any], item_ids: Sequence[str]
) -> Dict[str, str]:
    """Confirm unresolved index rows with the local Thai recognizer."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("PaddleOCR 运行环境不完整") from exc
    image = cv2.imdecode(
        np.frombuffer(sheet.get("content") or b"", dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        return {}
    entries = sheet.get("entries") or {}
    row_height = int(sheet.get("row_height") or 80)
    label_width = int(sheet.get("label_width") or 110)
    prepared = []
    crops = []
    for item_id in item_ids:
        candidate = entries.get(item_id)
        if not candidate:
            continue
        source_row = int(candidate.get("sheet_row") or 0)
        y0 = source_row * row_height
        y1 = min(image.shape[0], y0 + row_height)
        crop = image[y0:y1, min(label_width, image.shape[1]) :]
        if crop.size == 0:
            continue
        prepared.append(item_id)
        crops.append(crop)
    if not crops:
        return {}
    recognizer = _get_text_recognizer()
    with _TEXT_RECOGNIZER_LOCK:
        results = list(recognizer.predict(crops, batch_size=min(32, len(crops))))
    confirmed = {}
    for item_id, result in zip(prepared, results):
        value = _normalize_outline_ocr_text(str(result.get("rec_text") or ""))
        score = float(result.get("rec_score") or 0.0)
        consonants = re.findall(r"[\u0E01-\u0E2E]", value)
        if (
            score >= 0.55
            and len(consonants) >= 3
            and len(set(consonants)) >= 2
            and re.search(r"[\u0E00-\u0E7F]", value)
        ):
            confirmed[item_id] = value
    return confirmed


def recover_dense_cad_review_text_lines(
    page,
    candidates: Sequence[Dict[str, Any]],
    *,
    desired_width: int = 12000,
) -> List[Dict[str, Any]]:
    """Recover complete Thai lines around an unresolved CAD OCR fragment.

    Tesseract sometimes locates only a glyph or a piece of a compact CAD
    label. This targeted fallback uses Paddle's detector on that small local
    neighborhood, then its Thai recognizer on each detected line. It avoids a
    costly full-page Paddle pass and returns only high-confidence Thai text.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("CAD 局部复核运行环境不完整") from exc

    if not candidates:
        return []
    scale = min(5.0, max(2.5, desired_width / max(1.0, page.rect.width)))
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False
    )
    image = cv2.imdecode(
        np.frombuffer(pixmap.tobytes("png"), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise RuntimeError("PDF 页面无法转换为 CAD 局部复核图")

    image_height, image_width = image.shape[:2]
    display_scale_x = image_width / max(1.0, page.rect.width)
    display_scale_y = image_height / max(1.0, page.rect.height)
    detector = _get_cad_text_detector()
    prepared = []
    for source_index, source in enumerate(candidates):
        display_rect = fitz.Rect(source["bbox"]) * page.rotation_matrix
        target = fitz.Rect(
            display_rect.x0 * display_scale_x,
            display_rect.y0 * display_scale_y,
            display_rect.x1 * display_scale_x,
            display_rect.y1 * display_scale_y,
        ) & fitz.Rect(0, 0, image_width, image_height)
        if target.is_empty:
            continue
        vertical = bool(source.get("vertical"))
        if vertical:
            pad_x = max(160.0, target.width * 1.4)
            pad_y = max(240.0, target.height * 0.42)
        else:
            pad_x = max(240.0, target.width * 0.42)
            pad_y = max(160.0, target.height * 1.4)
        neighborhood = (
            target + (-pad_x, -pad_y, pad_x, pad_y)
        ) & fitz.Rect(0, 0, image_width, image_height)
        x0, y0, x1, y1 = (
            int(round(neighborhood.x0)),
            int(round(neighborhood.y0)),
            int(round(neighborhood.x1)),
            int(round(neighborhood.y1)),
        )
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        with _CAD_TEXT_DETECTOR_LOCK:
            results = list(detector.predict(crop))
        if not results:
            continue
        result = results[0]
        polygons = result.get("dt_polys")
        scores = result.get("dt_scores")
        if polygons is None or scores is None:
            continue
        for score, polygon in zip(scores, polygons):
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                continue
            if confidence < 0.80:
                continue
            polygon_array = np.asarray(polygon, dtype=np.float32)
            if polygon_array.ndim != 2 or polygon_array.shape[1] < 2:
                continue
            local_rect = fitz.Rect(
                float(polygon_array[:, 0].min()),
                float(polygon_array[:, 1].min()),
                float(polygon_array[:, 0].max()),
                float(polygon_array[:, 1].max()),
            )
            if local_rect.is_empty or min(local_rect.width, local_rect.height) < 5:
                continue
            padding = 0
            ix0 = max(0, int(local_rect.x0) - padding)
            iy0 = max(0, int(local_rect.y0) - padding)
            ix1 = min(crop.shape[1], int(local_rect.x1) + padding)
            iy1 = min(crop.shape[0], int(local_rect.y1) + padding)
            line_image = crop[iy0:iy1, ix0:ix1]
            if line_image.size == 0:
                continue
            display_vertical = local_rect.height > local_rect.width * 1.5
            if display_vertical:
                line_image = cv2.rotate(line_image, cv2.ROTATE_90_CLOCKWISE)
            recognition_scale = min(4.0, max(1.0, 160.0 / max(1, line_image.shape[0])))
            if recognition_scale > 1.01:
                line_image = cv2.resize(
                    line_image,
                    None,
                    fx=recognition_scale,
                    fy=recognition_scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            prepared.append(
                {
                    "source": source,
                    "source_index": source_index,
                    "polygon": polygon_array,
                    "crop_offset": (x0, y0),
                    "display_vertical": display_vertical,
                    "detector_confidence": confidence,
                    "image": line_image,
                }
            )
    if not prepared:
        return []

    recognizer = _get_text_recognizer()
    with _TEXT_RECOGNIZER_LOCK:
        recognized = list(
            recognizer.predict(
                [item["image"] for item in prepared],
                batch_size=min(32, len(prepared)),
            )
        )
    recovered = []
    page_rotation = int(page.rotation) % 360
    vertical_rotation = _display_direction_to_unrotated_rotation(page, (0.0, -1.0))
    for item, result in zip(prepared, recognized):
        source_text = _normalize_outline_ocr_text(str(result.get("rec_text") or ""))
        recognition_confidence = float(result.get("rec_score") or 0.0)
        consonants = re.findall(r"[\u0E01-\u0E2E]", source_text)
        if (
            recognition_confidence < 0.70
            or len(consonants) < 3
            or len(set(consonants)) < 2
        ):
            continue
        offset_x, offset_y = item["crop_offset"]
        display_points = [
            fitz.Point(
                (float(point[0]) + offset_x) / display_scale_x,
                (float(point[1]) + offset_y) / display_scale_y,
            )
            for point in item["polygon"]
        ]
        page_points = [point * page.derotation_matrix for point in display_points]
        rect = fitz.Rect(
            min(point.x for point in page_points),
            min(point.y for point in page_points),
            max(point.x for point in page_points),
            max(point.y for point in page_points),
        )
        if rect.is_empty:
            continue
        vertical = bool(item["display_vertical"])
        output = {
            "bbox": tuple(rect),
            "rotation": vertical_rotation if vertical else page_rotation,
            "vertical": vertical,
            "source_hint": source_text,
            "source_confidence": recognition_confidence * 100.0,
            "candidate_provider": "paddle-local-review",
            "origin_result_index": item["source"].get("origin_result_index"),
            "origin_item_id": item["source"].get("origin_item_id"),
        }
        duplicate = next(
            (
                existing
                for existing in recovered
                if int(existing["origin_result_index"] or -1)
                == int(output["origin_result_index"] or -1)
                and str(existing["origin_item_id"] or "")
                == str(output["origin_item_id"] or "")
                and _overlap_smaller(
                    fitz.Rect(existing["bbox"]), rect
                ) >= 0.75
            ),
            None,
        )
        if duplicate is None:
            recovered.append(output)
        elif output["source_confidence"] > duplicate["source_confidence"]:
            duplicate.update(output)
    return recovered


def _dense_tesseract_paddle_fallback(ocr, image, page, candidates):
    """Use Tesseract only to seed local PaddleOCR retries on missed regions."""
    seed_rects = _dense_tesseract_seed_rects(image)
    if not seed_rects:
        return []
    covered = [
        _polygon_rect(candidate[2])
        for candidate in candidates
        if _candidate_can_cover_seed(candidate)
    ]
    uncovered = [
        rect
        for rect in seed_rects
        if not any(_seed_is_covered(rect, existing) for existing in covered)
    ]
    if not uncovered:
        return []

    height, width = image.shape[:2]
    crops = _merge_seed_crops(
        [_seed_crop_rect(rect, width, height) for rect in uncovered]
    )
    recovered = []
    page_rotation = int(page.rotation) % 360
    vertical_rotation = _display_direction_to_unrotated_rotation(page, (0.0, -1.0))
    for crop_index, crop_rect in enumerate(crops[:96], start=1):
        x0, y0, x1, y1 = [int(round(value)) for value in crop_rect]
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        shortest = max(1, min(crop.shape[:2]))
        resize_scale = min(4.0, max(2.0, 360.0 / shortest))
        try:
            import cv2
        except ImportError:
            return recovered
        enlarged = cv2.resize(
            crop,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        if crop_rect.width >= crop_rect.height * 0.8:
            recovered.extend(
                _local_paddle_candidates(
                    ocr,
                    enlarged,
                    crop_rect,
                    resize_scale,
                    page_rotation,
                    f"tesseract-local-{crop_index}",
                    rotate=False,
                )
            )
        if crop_rect.height >= crop_rect.width * 0.8:
            rotated = cv2.rotate(enlarged, cv2.ROTATE_90_CLOCKWISE)
            recovered.extend(
                _local_paddle_candidates(
                    ocr,
                    rotated,
                    crop_rect,
                    resize_scale,
                    vertical_rotation,
                    f"tesseract-local-{crop_index}-rotated",
                    rotate=True,
                    original_height=enlarged.shape[0],
                )
            )
    return recovered


def _candidate_direction_matches_polygon(candidate) -> bool:
    rect = _polygon_rect(candidate[2])
    rotation = int(candidate[3]) % 360
    if rotation in {0, 180}:
        return rect.height <= rect.width * 1.5
    return rect.width <= rect.height * 1.5


def _candidate_can_cover_seed(candidate) -> bool:
    value = str(candidate[0])
    score = float(candidate[1])
    thai_count = len(re.findall(r"[\u0E00-\u0E7F]", value))
    return (
        _candidate_direction_matches_polygon(candidate)
        and score >= 0.40
        and thai_count > 0
        and (score >= 0.78 or thai_count >= 2)
    )


def _dense_vertical_header_candidates(ocr, image, page):
    """Read short vertical schedule headers from a compact high-scale band."""
    try:
        import cv2
    except ImportError:
        return []
    height, width = image.shape[:2]
    band_height = max(1, int(height * 0.24))
    overlap = max(24, int(width * 0.02))
    rotation = _display_direction_to_unrotated_rotation(page, (0.0, -1.0))
    recovered = []
    for column in range(4):
        x0 = max(0, round(column * width / 4) - (overlap if column else 0))
        x1 = min(
            width,
            round((column + 1) * width / 4) + (overlap if column < 3 else 0),
        )
        crop_rect = fitz.Rect(x0, 0, x1, band_height)
        crop = image[:band_height, x0:x1]
        scale = 1.6
        enlarged = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        rotated = cv2.rotate(enlarged, cv2.ROTATE_90_CLOCKWISE)
        recovered.extend(
            _local_paddle_candidates(
                ocr,
                rotated,
                crop_rect,
                scale,
                rotation,
                f"header-c{column + 1}-rotated",
                rotate=True,
                original_height=enlarged.shape[0],
            )
        )
    return recovered


def _dense_bottom_title_candidates(ocr, image, page):
    """Read large CAD drawing titles without a full-page detector downscale."""
    try:
        import cv2
    except ImportError:
        return []
    height, width = image.shape[:2]
    band_top = max(0, int(height * 0.70))
    overlap = max(48, int(width * 0.06))
    rotation = int(page.rotation) % 360
    recovered = []
    for column in range(4):
        x0 = max(0, round(column * width / 4) - (overlap if column else 0))
        x1 = min(
            width,
            round((column + 1) * width / 4) + (overlap if column < 3 else 0),
        )
        crop_rect = fitz.Rect(x0, band_top, x1, height)
        crop = image[band_top:height, x0:x1]
        scale = 1.6
        enlarged = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        recovered.extend(
            _local_paddle_candidates(
                ocr,
                enlarged,
                crop_rect,
                scale,
                rotation,
                f"bottom-c{column + 1}",
                rotate=False,
            )
        )
    return recovered


def _recover_repeated_bottom_titles(candidates, image_height):
    """Use a clean title-block transcription for a fuzzy large drawing title."""
    anchors = []
    for candidate in candidates:
        value, score, polygon, _, pass_name = candidate
        consonants = "".join(re.findall(r"[\u0E01-\u0E2E]", str(value)))
        if (
            str(pass_name).startswith("bottom-c")
            and float(score) >= 0.85
            and len(consonants) >= 5
        ):
            anchors.append((str(value), consonants))
    if not anchors:
        return candidates

    output = []
    for candidate in candidates:
        value, score, polygon, rotation, pass_name = candidate
        rect = _polygon_rect(polygon)
        consonants = "".join(re.findall(r"[\u0E01-\u0E2E]", str(value)))
        is_large_fuzzy_title = (
            str(pass_name).startswith("bottom-c")
            and 0.35 <= float(score) < 0.65
            and len(consonants) >= 5
            and rect.height >= image_height * 0.012
            and rect.width >= rect.height * 2.0
        )
        best = None
        if is_large_fuzzy_title:
            for anchor_value, anchor_consonants in anchors:
                similarity = SequenceMatcher(
                    None, consonants, anchor_consonants, autojunk=False
                ).ratio()
                if similarity >= 0.82 and (best is None or similarity > best[0]):
                    best = (similarity, anchor_value)
        if best is not None:
            output.append(
                (
                    best[1],
                    0.80,
                    polygon,
                    rotation,
                    "bottom-title-recovered",
                )
            )
        else:
            output.append(candidate)
    return output


def _recover_repeated_vector_labels(candidates, image):
    """Recover identical CAD labels missed among already recognized copies."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    grouped = defaultdict(list)
    for candidate in candidates:
        value, score, polygon, rotation, _ = candidate
        consonants = "".join(re.findall(r"[\u0E01-\u0E2E]", str(value)))
        if (
            float(score) >= 0.85
            and len(consonants) >= 2
            and _candidate_direction_matches_polygon(candidate)
        ):
            grouped[(int(rotation) % 360, consonants)].append(candidate)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = np.ascontiguousarray(255 - gray)
    image_height, image_width = ink.shape[:2]
    existing_rects = [
        _polygon_rect(candidate[2])
        for candidate in candidates
        if _dense_candidate_can_reach_acceptance(candidate)
    ]
    recovered = []

    ordered_groups = sorted(grouped.items(), key=lambda item: -len(item[1]))[:8]
    for (rotation, _), group in ordered_groups:
        exact_counts = Counter(str(candidate[0]) for candidate in group)
        canonical, exact_count = exact_counts.most_common(1)[0]
        if exact_count < 2:
            continue
        templates = [
            candidate
            for candidate in sorted(group, key=lambda item: -float(item[1]))
            if str(candidate[0]) == canonical
        ][:5]
        if len(templates) < 2:
            continue

        matches = []
        for template_index, template_candidate in enumerate(templates):
            template_rect = _polygon_rect(template_candidate[2])
            x0 = max(0, int(math.floor(template_rect.x0)))
            y0 = max(0, int(math.floor(template_rect.y0)))
            x1 = min(image_width, int(math.ceil(template_rect.x1)))
            y1 = min(image_height, int(math.ceil(template_rect.y1)))
            if x1 - x0 < 5 or y1 - y0 < 5:
                continue
            template = ink[y0:y1, x0:x1]
            if not 0.025 <= float(np.mean(template > 64)) <= 0.55:
                continue
            scores = cv2.matchTemplate(ink, template, cv2.TM_CCOEFF_NORMED)
            working = scores.copy()
            template_height, template_width = template.shape[:2]
            for _ in range(120):
                _, best_score, _, location = cv2.minMaxLoc(working)
                if best_score < 0.82:
                    break
                left, top = location
                rect = fitz.Rect(
                    left,
                    top,
                    left + template_width,
                    top + template_height,
                )
                matches.append((rect, float(best_score), template_index))
                working[
                    max(0, top - template_height // 2) : min(
                        working.shape[0], top + template_height // 2
                    ),
                    max(0, left - template_width // 2) : min(
                        working.shape[1], left + template_width // 2
                    ),
                ] = -1.0

        clusters = []
        for rect, score, template_index in sorted(
            matches, key=lambda item: -item[1]
        ):
            center = fitz.Point(rect.x0 + rect.width / 2, rect.y0 + rect.height / 2)
            cluster = next(
                (
                    item
                    for item in clusters
                    if abs(item["center"].x - center.x) <= 18
                    and abs(item["center"].y - center.y) <= 18
                ),
                None,
            )
            if cluster is None:
                clusters.append(
                    {
                        "center": center,
                        "matches": [(rect, score, template_index)],
                    }
                )
            else:
                cluster["matches"].append((rect, score, template_index))
                centers = [
                    fitz.Point(
                        item[0].x0 + item[0].width / 2,
                        item[0].y0 + item[0].height / 2,
                    )
                    for item in cluster["matches"]
                ]
                cluster["center"] = fitz.Point(
                    sum(point.x for point in centers) / len(centers),
                    sum(point.y for point in centers) / len(centers),
                )

        for cluster in clusters:
            distinct_templates = {
                item[2] for item in cluster["matches"]
            }
            if len(distinct_templates) < 2:
                continue
            best_rect, best_score, _ = max(
                cluster["matches"], key=lambda item: item[1]
            )
            if any(
                _overlap_smaller(best_rect, existing) >= 0.45
                for existing in existing_rects
            ):
                continue
            polygon = [
                [best_rect.x0, best_rect.y0],
                [best_rect.x1, best_rect.y0],
                [best_rect.x1, best_rect.y1],
                [best_rect.x0, best_rect.y1],
            ]
            recovered.append(
                (
                    canonical,
                    min(0.95, best_score),
                    polygon,
                    rotation,
                    "repeated-template",
                )
            )
            existing_rects.append(best_rect)
    return recovered


def _dense_candidate_can_reach_acceptance(candidate) -> bool:
    value = str(candidate[0])
    score = float(candidate[1])
    thai_count = len(re.findall(r"[\u0E00-\u0E7F]", value))
    consonants = re.findall(r"[\u0E01-\u0E2E]", value)
    return (
        score >= 0.65
        and thai_count > 0
        and _candidate_direction_matches_polygon(candidate)
        and (
            score >= 0.78
            or (
                thai_count >= 2
                and len(consonants) >= 2
                and len(set(consonants)) >= 2
            )
        )
    )


def _suppress_circular_text_fragments(candidates, page_rect):
    """Drop curved logo fragments already represented by a nearby full phrase."""
    full_phrases = []
    for candidate in candidates:
        value = str(candidate["value"])
        thai = re.findall(r"[\u0E00-\u0E7F]", value)
        rect = candidate["rect"]
        if (
            len(thai) >= 8
            and float(candidate["score"]) >= 0.85
            and rect.width >= rect.height * 3
            and (
                rect.x0 <= page_rect.x0 + page_rect.width * 0.15
                or rect.x1 >= page_rect.x1 - page_rect.width * 0.15
            )
        ):
            full_phrases.append((candidate, set(thai)))
    if not full_phrases:
        return candidates

    output = []
    for candidate in candidates:
        value = str(candidate["value"])
        thai = re.findall(r"[\u0E00-\u0E7F]", value)
        consonants = "".join(re.findall(r"[\u0E01-\u0E2E]", value))
        rect = candidate["rect"]
        suppressed = False
        for phrase, phrase_chars in full_phrases:
            if candidate is phrase or not thai or len(thai) >= len(
                re.findall(r"[\u0E00-\u0E7F]", str(phrase["value"]))
            ):
                continue
            phrase_rect = phrase["rect"]
            phrase_consonants = "".join(
                re.findall(r"[\u0E01-\u0E2E]", str(phrase["value"]))
            )
            horizontal_overlap = max(
                0.0, min(rect.x1, phrase_rect.x1) - max(rect.x0, phrase_rect.x0)
            )
            matching_consonants = sum(
                block.size
                for block in SequenceMatcher(
                    None, consonants, phrase_consonants, autojunk=False
                ).get_matching_blocks()
            )
            near_phrase = (
                horizontal_overlap >= rect.width * 0.6
                and rect.width <= phrase_rect.width * 0.45
                and rect.y0 >= phrase_rect.y0 - phrase_rect.height * 0.5
                and rect.y1 <= phrase_rect.y1 + phrase_rect.height * 3.5
            )
            character_overlap = sum(char in phrase_chars for char in thai) / len(thai)
            ordered_overlap = matching_consonants / max(1, len(consonants))
            centered_short_arc_fragment = (
                rect.width <= phrase_rect.width * 0.25
                and rect.height <= phrase_rect.height * 1.75
                and phrase_rect.x0 + phrase_rect.width * 0.18 <= rect.x0
                and rect.x1 <= phrase_rect.x1 - phrase_rect.width * 0.18
                and rect.y0 >= phrase_rect.y1 + phrase_rect.height * 0.45
                and rect.y1 <= phrase_rect.y1 + phrase_rect.height * 3.75
            )
            if (
                near_phrase
                and (
                    (
                        character_overlap >= 0.70
                        and ordered_overlap >= 0.65
                    )
                    or centered_short_arc_fragment
                )
            ):
                suppressed = True
                break
        if not suppressed:
            output.append(candidate)
    return output


def _dense_tesseract_seed_rects(image, *, thai_only: bool = True):
    return [
        candidate["rect"]
        for candidate in _dense_tesseract_seed_candidates(
            image, thai_only=thai_only
        )
    ]


def _dense_paddle_detection_seeds(
    image,
    *,
    tile_side: int = _CAD_DETECTION_TILE_SIDE,
    tile_overlap: int = _CAD_DETECTION_TILE_OVERLAP,
):
    """Locate CAD labels with Paddle's detector before asking GPT to read them.

    The detector supplies only geometry. This deliberately avoids using a
    local OCR transcription as a gate: low-quality Thai recognition should
    not decide whether a visible CAD label reaches the translation model.

    The detector itself accepts a maximum-side input. Run it on overlapping
    source-image tiles rather than allowing its preprocessor to downscale an
    entire large drawing. Coordinates are restored to the original rendered
    page before candidates are deduplicated.
    """
    detector = _get_cad_text_detector()
    candidates = []
    image_height, image_width = image.shape[:2]
    for top in _cad_detection_tile_offsets(
        image_height, tile_side, tile_overlap
    ):
        for left in _cad_detection_tile_offsets(
            image_width, tile_side, tile_overlap
        ):
            tile = image[
                top : min(image_height, top + tile_side),
                left : min(image_width, left + tile_side),
            ]
            if tile.size == 0:
                continue
            # Paddle predictors are not documented as thread-safe. Dense PDF
            # pages may run concurrently, so serialize just this light local
            # inference rather than risking intermittent lost detections.
            with _CAD_TEXT_DETECTOR_LOCK:
                results = list(detector.predict(tile))
            for result in results:
                polygons = result.get("dt_polys")
                scores = result.get("dt_scores")
                if polygons is None or scores is None:
                    continue
                for polygon, score in zip(polygons, scores):
                    try:
                        confidence = float(score)
                    except (TypeError, ValueError):
                        continue
                    if confidence < 0.35:
                        continue
                    rect = _polygon_rect(polygon)
                    if rect.is_empty or rect.width < 4 or rect.height < 4:
                        continue
                    rect += (left, top, left, top)
                    candidates.append(
                        {
                            "rect": rect,
                            "source_text": "",
                            "source_confidence": confidence * 100.0,
                        }
                    )
    return _deduplicate_seed_candidates(candidates)


def _cad_detection_tile_offsets(
    length: int,
    tile_side: int,
    tile_overlap: int,
) -> List[int]:
    """Return overlapping tile starts while always covering the final pixel."""
    size = max(1, int(tile_side))
    extent = max(0, int(length))
    if extent <= size:
        return [0]
    stride = max(1, size - max(0, min(size - 1, int(tile_overlap))))
    last_start = extent - size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _dense_tesseract_seed_candidates(image, *, thai_only: bool = True):
    """Return deduplicated CAD line rectangles with Tesseract text hints."""
    try:
        import cv2
        from .documents import ocr_image_text_blocks
    except ImportError:
        return []

    height, width = image.shape[:2]
    enlarged_scale = max(1.0, 4800.0 / max(1, width))
    scales = [1.0]
    if enlarged_scale >= 1.05:
        scales.append(enlarged_scale)
    passes = []
    for detection_scale in scales:
        detection_image = (
            cv2.resize(
                image,
                None,
                fx=detection_scale,
                fy=detection_scale,
                interpolation=cv2.INTER_CUBIC,
            )
            if detection_scale > 1.0
            else image
        )
        passes.append((detection_image, False, detection_scale))
        passes.append(
            (
                cv2.rotate(detection_image, cv2.ROTATE_90_CLOCKWISE),
                True,
                detection_scale,
            )
        )

    def detect(pass_item):
        pass_image, rotated, detection_scale = pass_item
        ok, encoded = cv2.imencode(".png", pass_image)
        if not ok:
            return pass_image, rotated, detection_scale, []
        with _CAD_TESSERACT_SEMAPHORE:
            blocks = ocr_image_text_blocks(encoded.tobytes(), min_confidence=25.0)
        return pass_image, rotated, detection_scale, blocks

    with ThreadPoolExecutor(max_workers=len(passes)) as executor:
        detected_passes = list(executor.map(detect, passes))

    seeds = []
    for pass_image, rotated, detection_scale, blocks in detected_passes:
        detection_height = round(height * detection_scale)
        for block in blocks:
            source_text = str(block.get("source_text") or "")
            thai_letters = re.findall(r"[\u0E01-\u0E3A\u0E40-\u0E4E]", source_text)
            if (thai_only and not thai_letters) or not source_text or len(source_text) > 250:
                continue
            values = block.get("bbox")
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                continue
            pass_height, pass_width = pass_image.shape[:2]
            rect = fitz.Rect(
                float(values[0]) / 1000.0 * pass_width,
                float(values[1]) / 1000.0 * pass_height,
                float(values[2]) / 1000.0 * pass_width,
                float(values[3]) / 1000.0 * pass_height,
            )
            if rotated:
                polygon = _restore_clockwise_polygon(
                    [
                        [rect.x0, rect.y0],
                        [rect.x1, rect.y0],
                        [rect.x1, rect.y1],
                        [rect.x0, rect.y1],
                    ],
                    detection_height,
                )
                rect = _polygon_rect(polygon)
            if detection_scale != 1.0:
                rect = fitz.Rect(
                    rect.x0 / detection_scale,
                    rect.y0 / detection_scale,
                    rect.x1 / detection_scale,
                    rect.y1 / detection_scale,
                )
            rect &= fitz.Rect(0, 0, width, height)
            minimum_line_length = min(width, height) * 0.015
            if (
                rect.width >= 5
                and rect.height >= 5
                and max(rect.width, rect.height) >= minimum_line_length
            ):
                seeds.append(
                    {
                        "rect": rect,
                        "source_text": _normalize_tesseract_source_hint(
                            source_text
                        ),
                        "source_confidence": float(block.get("confidence") or 0.0),
                    }
                )
    return _deduplicate_seed_candidates(seeds)


def _deduplicate_seed_candidates(candidates):
    output = []
    for candidate in sorted(
        candidates,
        key=lambda item: fitz.Rect(item["rect"]).get_area(),
        reverse=True,
    ):
        rect = fitz.Rect(candidate["rect"])
        duplicate = next(
            (
                existing
                for existing in output
                if _overlap_smaller(rect, fitz.Rect(existing["rect"])) >= 0.70
            ),
            None,
        )
        if duplicate is None:
            output.append(
                {
                    "rect": rect,
                    "source_text": str(candidate.get("source_text") or "").strip(),
                    "source_confidence": float(
                        candidate.get("source_confidence") or 0.0
                    ),
                }
            )
            continue
        current_text = str(candidate.get("source_text") or "").strip()
        existing_text = str(duplicate.get("source_text") or "").strip()
        current_quality = (
            float(candidate.get("source_confidence") or 0.0),
            len(re.findall(r"[\u0E01-\u0E2E]", current_text)),
            len(current_text),
        )
        existing_quality = (
            float(duplicate.get("source_confidence") or 0.0),
            len(re.findall(r"[\u0E01-\u0E2E]", existing_text)),
            len(existing_text),
        )
        if current_quality > existing_quality:
            duplicate["source_text"] = current_text
            duplicate["source_confidence"] = float(
                candidate.get("source_confidence") or 0.0
            )
    return output


def _normalize_tesseract_source_hint(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    # Outline fonts are frequently segmented as one TSV word per glyph.
    # Thai normally does not require spaces between words, so joining only
    # Thai-to-Thai gaps reconstructs the line without touching Latin codes or
    # numeric spacing.
    return re.sub(
        r"(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])",
        "",
        value,
    )


def _local_paddle_candidates(
    ocr,
    image,
    crop_rect,
    resize_scale,
    rotation,
    pass_name,
    *,
    rotate,
    original_height=0,
):
    results = list(ocr.predict(image))
    if not results:
        return []
    result = results[0]
    output = []
    for text, score, polygon in zip(
        list(result.get("rec_texts") or []),
        [float(value) for value in (result.get("rec_scores") or [])],
        list(result.get("rec_polys") or []),
    ):
        value = str(text).strip()
        thai_count = len(re.findall(r"[\u0E00-\u0E7F]", value))
        minimum_score = 0.70 if str(pass_name).startswith("tesseract-local-") else 0.30
        if thai_count == 0 or score < minimum_score or (score < 0.40 and thai_count < 2):
            continue
        restored = (
            _restore_clockwise_polygon(polygon, original_height)
            if rotate
            else polygon
        )
        mapped = [
            [
                crop_rect.x0 + float(point[0]) / resize_scale,
                crop_rect.y0 + float(point[1]) / resize_scale,
            ]
            for point in restored
        ]
        output.append((value, score, mapped, rotation, pass_name))
    return output


def _polygon_rect(polygon) -> fitz.Rect:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))


def _seed_is_covered(seed: fitz.Rect, existing: fitz.Rect) -> bool:
    if _overlap_smaller(seed, existing) >= 0.18:
        return True
    center = fitz.Point(seed.x0 + seed.width / 2, seed.y0 + seed.height / 2)
    return (existing + (-4, -4, 4, 4)).contains(center)


def _seed_crop_rect(seed: fitz.Rect, width: int, height: int) -> fitz.Rect:
    if seed.height > seed.width * 1.5:
        pad_x = max(18.0, seed.width * 0.8)
        pad_y = max(32.0, min(96.0, seed.height * 0.12))
    else:
        pad_x = max(32.0, min(96.0, seed.width * 0.12))
        pad_y = max(18.0, seed.height * 0.8)
    return (seed + (-pad_x, -pad_y, pad_x, pad_y)) & fitz.Rect(0, 0, width, height)


def _merge_seed_crops(rects: Sequence[fitz.Rect]) -> List[fitz.Rect]:
    merged: List[fitz.Rect] = []
    for rect in sorted(rects, key=lambda item: item.get_area(), reverse=True):
        match = next(
            (
                existing
                for existing in merged
                if _overlap_smaller(rect, existing) >= 0.45
            ),
            None,
        )
        if match is None:
            merged.append(fitz.Rect(rect))
        else:
            match |= rect
    return merged


def _deduplicate_rects(rects: Sequence[fitz.Rect]) -> List[fitz.Rect]:
    output: List[fitz.Rect] = []
    for rect in sorted(rects, key=lambda item: item.get_area(), reverse=True):
        if any(_overlap_smaller(rect, existing) >= 0.70 for existing in output):
            continue
        output.append(rect)
    return output


def _offset_polygon(polygon, offset_x: int, offset_y: int):
    return [
        [float(point[0]) + offset_x, float(point[1]) + offset_y]
        for point in polygon
    ]


def _polygon_touches_internal_tile_edge(
    polygon,
    offset_x: int,
    offset_y: int,
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
) -> bool:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    margin = 8.0
    return (
        (offset_x > 0 and min(xs) <= margin)
        or (offset_x + tile_width < image_width and max(xs) >= tile_width - margin)
        or (offset_y > 0 and min(ys) <= margin)
        or (offset_y + tile_height < image_height and max(ys) >= tile_height - margin)
    )


def _same_ocr_line(first: fitz.Rect, second: fitz.Rect, rotation: int) -> bool:
    if _overlap_smaller(first, second) >= 0.45:
        return True
    intersection = first & second
    if intersection.is_empty:
        return False
    if rotation in {0, 180}:
        cross_overlap = intersection.height / max(1.0, min(first.height, second.height))
        along_overlap = intersection.width / max(1.0, min(first.width, second.width))
    else:
        cross_overlap = intersection.width / max(1.0, min(first.width, second.width))
        along_overlap = intersection.height / max(1.0, min(first.height, second.height))
    return cross_overlap >= 0.72 and along_overlap >= 0.12


def _ocr_candidate_quality(candidate: Dict[str, Any]) -> Tuple[int, int, float, float]:
    value = re.sub(r"\s+", "", str(candidate["value"]))
    rect = candidate["rect"]
    pass_name = str(candidate.get("pass_name") or "")
    source_rank = (
        0
        if pass_name.startswith("tesseract-local-")
        else 2
        if pass_name.startswith("header-")
        else 1
    )
    return source_rank, len(value), float(candidate["score"]), rect.get_area()


def _normalize_outline_ocr_text(value: str) -> str:
    """Repair a narrow mixed-script ambiguity without splitting the OCR line."""
    value = re.sub(
        r"^พ(?=\d{1,2}(?:\s|[\u0E00-\u0E7F]))",
        "W",
        value.strip(),
    )
    value = re.sub(r"^WO(?=\s+[\u0E00-\u0E7F])", "W0", value)
    value = re.sub(r"(?<=[\u0E00-\u0E7F])L$", "ม", value)
    return re.sub(
        r"(?:E(?:LE)?CTRICAL|MECHANICAL|CIVIL|STRUCTURAL)\s+ENGINEERS?\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).rstrip()


def _ocr_structural_barriers(page):
    try:
        drawings = page.get_cdrawings()
    except Exception:
        drawings = page.get_drawings()
    vertical = []
    horizontal = []
    for drawing in drawings:
        for item in drawing.get("items", []):
            if not item or item[0] != "l":
                continue
            start, end = fitz.Point(item[1]), fitz.Point(item[2])
            if abs(start.x - end.x) <= 0.5 and abs(start.y - end.y) >= page.rect.height * 0.02:
                vertical.append(((start.x + end.x) / 2, min(start.y, end.y), max(start.y, end.y)))
            elif abs(start.y - end.y) <= 0.5 and abs(start.x - end.x) >= page.rect.width * 0.02:
                horizontal.append(((start.y + end.y) / 2, min(start.x, end.x), max(start.x, end.x)))
    return vertical, horizontal


def _clip_fallback_rect_at_barrier(rect: fitz.Rect, rotation: int, barriers) -> fitz.Rect:
    vertical, horizontal = barriers
    if rotation in {0, 180}:
        center = (rect.y0 + rect.y1) / 2
        cuts = sorted(
            x
            for x, top, bottom in vertical
            if rect.x0 + 5 < x < rect.x1 - 5 and top - 1 <= center <= bottom + 1
        )
        bounds = [rect.x0, *cuts, rect.x1]
        x0, x1 = max(zip(bounds, bounds[1:]), key=lambda pair: pair[1] - pair[0])
        return fitz.Rect(x0, rect.y0, x1, rect.y1)
    center = (rect.x0 + rect.x1) / 2
    cuts = sorted(
        y
        for y, left, right in horizontal
        if rect.y0 + 5 < y < rect.y1 - 5 and left - 1 <= center <= right + 1
    )
    bounds = [rect.y0, *cuts, rect.y1]
    y0, y1 = max(zip(bounds, bounds[1:]), key=lambda pair: pair[1] - pair[0])
    return fitz.Rect(rect.x0, y0, rect.x1, y1)


def extract_visual_page_units(
    page,
    page_number: int,
    *,
    desired_width: int = 4000,
    minimum_score: float = 0.78,
    native_units: Optional[Sequence[Dict[str, Any]]] = None,
    high_accuracy: bool = False,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Extract one complete OCR line per visual translation unit.

    Native text coordinates are supplied only to suppress duplicate OCR boxes;
    those native units are not returned or rewritten by this adapter.
    """
    if native_units is None:
        native_units = extract_native_page_units(page, page_number)
    units = extract_table_ocr_units(
        page,
        page_number,
        native_units,
        table_regions=None,
        desired_width=desired_width,
        minimum_score=minimum_score,
        high_accuracy=high_accuracy,
        diagnostics=diagnostics,
    )
    for unit in units:
        metadata = unit.setdefault("metadata", {})
        metadata.pop("native_pdf_version", None)
        metadata["visual_pdf_version"] = 1
        metadata["translation_unit"] = "complete-line"
    return units


def initialize_pdf_ocr_worker(content: bytes) -> None:
    global _OCR_WORKER_PDF_CONTENT
    _OCR_WORKER_PDF_CONTENT = content


def extract_visual_page_units_in_worker(
    page_number: int,
    desired_width: int,
    minimum_score: float,
    native_units: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if _OCR_WORKER_PDF_CONTENT is None:
        raise RuntimeError("PDF OCR 工作进程未初始化")
    document = fitz.open(stream=_OCR_WORKER_PDF_CONTENT, filetype="pdf")
    try:
        return extract_visual_page_units(
            document[page_number - 1],
            page_number,
            desired_width=desired_width,
            minimum_score=minimum_score,
            native_units=native_units,
            high_accuracy=True,
        )
    finally:
        document.close()


def find_dense_table_regions(page) -> List[fitz.Rect]:
    """Find ruled tables and stacked title blocks that contain editable text."""
    minimum_width = max(80.0, page.rect.width * 0.08)
    horizontal = []
    vertical = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if not item or item[0] != "l":
                continue
            start, end = fitz.Point(item[1]), fitz.Point(item[2])
            dx, dy = abs(end.x - start.x), abs(end.y - start.y)
            if dy <= 0.5 and dx >= minimum_width:
                horizontal.append(
                    (min(start.x, end.x), max(start.x, end.x), (start.y + end.y) / 2)
                )
            elif dx <= 0.5 and dy >= page.rect.height * 0.2:
                vertical.append(
                    ((start.x + end.x) / 2, min(start.y, end.y), max(start.y, end.y))
                )

    groups: List[List[Tuple[float, float, float]]] = []
    for line in sorted(horizontal, key=lambda value: (value[0], value[1], value[2])):
        match = next(
            (
                group
                for group in groups
                if abs(group[0][0] - line[0]) <= 2.0
                and abs(group[0][1] - line[1]) <= 2.0
            ),
            None,
        )
        if match is None:
            groups.append([line])
        else:
            match.append(line)

    regions = []
    page_area = max(1.0, page.rect.width * page.rect.height)
    for group in groups:
        distinct_y = sorted({round(line[2], 1) for line in group})
        if len(distinct_y) < 10:
            continue
        x0 = median(line[0] for line in group)
        x1 = median(line[1] for line in group)
        y0, y1 = distinct_y[0], distinct_y[-1]
        region = fitz.Rect(x0, y0, x1, y1)
        crossing_verticals = [
            (x, top, bottom)
            for x, top, bottom in vertical
            if x0 - 2 <= x <= x1 + 2 and top <= y0 + 2 and bottom >= y1 - 2
        ]
        if (
            len({round(item[0], 1) for item in crossing_verticals}) < 3
            or region.get_area() / page_area < 0.08
        ):
            continue
        y0 = min([y0, *(item[1] for item in crossing_verticals)])
        y1 = max([y1, *(item[2] for item in crossing_verticals)])
        title_margin = page.rect.height * 0.04
        regions.append(
            fitz.Rect(x0 - 2.0, y0 - title_margin, x1 + 2.0, y1 + title_margin)
            & page.rect
        )
    return regions


def requires_deep_table_ocr(page) -> bool:
    """Select deep OCR only for pages dominated by multiple ruled tables."""
    regions = find_dense_table_regions(page)
    if len(regions) < 2:
        return False
    page_area = max(1.0, page.rect.width * page.rect.height)
    return sum(region.get_area() for region in regions) / page_area >= 0.20


_VECTOR_PATH_OPERATORS = {"m", "l", "c", "v", "y", "h", "re"}
_VECTOR_PAINT_OPERATORS = {"S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n"}


def _remove_dense_cad_outline_paths(
    source_content: bytes,
    segments: Sequence[Dict[str, Any]],
) -> bytes:
    """Surgically remove OCR-confirmed outline glyph paths from CAD PDFs.

    CAD programs frequently convert text to many small stroked paths. PDF
    redaction can only remove whole graphical objects and therefore either
    leaves glyph strokes or removes table lines. Here paths are parsed from the
    original content stream and only glyph-sized paths intersecting a confirmed
    OCR text region are omitted. All other PDF operations are preserved.
    """
    targets_by_page: Dict[int, List[Tuple[fitz.Rect, int]]] = defaultdict(list)
    for segment in segments:
        page_number = segment.get("page_number")
        if not isinstance(page_number, int):
            continue
        metadata = segment.get("metadata") or {}
        raw_rect = metadata.get("cover_bbox", segment.get("bbox"))
        try:
            rect = fitz.Rect(raw_rect)
        except (TypeError, ValueError):
            continue
        if rect.is_empty:
            continue
        rotation = int(metadata.get("rotation") or 0) % 360
        # OCR rectangles are measured on a raster and are occasionally one
        # anti-aliased pixel tight. This remains far below a table cell gap.
        padding = max(1.0, min(2.5, (rect.width if rotation in {90, 270} else rect.height) * 0.16))
        targets_by_page[page_number].append(
            (rect + (-padding, -padding, padding, padding), rotation)
        )
    if not targets_by_page:
        return source_content

    output = BytesIO()
    with fitz.open(stream=source_content, filetype="pdf") as fitz_document:
        page_geometry = {
            page_number: (
                float(fitz_document[page_number - 1].mediabox.height),
                fitz_document[page_number - 1].derotation_matrix,
            )
            for page_number in targets_by_page
        }
    with pikepdf.Pdf.open(BytesIO(source_content)) as document:
        for page_number, targets in targets_by_page.items():
            if page_number < 1 or page_number > len(document.pages):
                raise ValueError(f"CAD 路径删除页码无效: {page_number}")
            page = document.pages[page_number - 1]
            page_height, derotation_matrix = page_geometry[page_number]
            contents = page.obj.get("/Contents")
            if contents is None:
                continue
            streams = list(contents) if isinstance(contents, pikepdf.Array) else [contents]
            for stream in streams:
                instructions = pikepdf.parse_content_stream(stream)
                retained, removed_count = _filter_outline_path_instructions(
                    instructions,
                    targets,
                    page_height,
                    derotation_matrix,
                )
                if removed_count:
                    stream.write(pikepdf.unparse_content_stream(retained))
        document.save(output)
    return output.getvalue()


def _filter_outline_path_instructions(
    instructions: Sequence[Any],
    targets: Sequence[Tuple[fitz.Rect, int]],
    page_height: float,
    derotation_matrix: fitz.Matrix,
) -> Tuple[List[Any], int]:
    """Keep all PDF instructions except small text-like paths in target boxes."""
    graphics_stack = [(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)]
    matrix = graphics_stack[-1]
    path_indices: List[int] = []
    path_points: List[Tuple[float, float]] = []
    path_has_clip = False
    remove_indices = set()

    for index, instruction in enumerate(instructions):
        operator = str(instruction.operator)
        if operator == "q":
            graphics_stack.append(matrix)
            continue
        if operator == "Q":
            matrix = graphics_stack.pop() if len(graphics_stack) > 1 else graphics_stack[0]
            continue
        if operator == "cm":
            values = tuple(float(value) for value in instruction.operands)
            if len(values) == 6:
                matrix = _multiply_pdf_matrices(matrix, values)
            continue
        if operator in _VECTOR_PATH_OPERATORS:
            path_indices.append(index)
            path_points.extend(
                _pdf_path_points(operator, instruction.operands, matrix)
            )
            continue
        if operator in {"W", "W*"}:
            path_has_clip = True
            continue
        if operator not in _VECTOR_PAINT_OPERATORS:
            continue
        if path_points and not path_has_clip:
            path_rect = _pdf_points_to_page_rect(
                path_points,
                page_height,
                derotation_matrix,
            )
            if _is_outline_glyph_path(path_rect, targets):
                remove_indices.update(path_indices)
                remove_indices.add(index)
        path_indices = []
        path_points = []
        path_has_clip = False

    return (
        [instruction for index, instruction in enumerate(instructions) if index not in remove_indices],
        len(remove_indices),
    )


def _multiply_pdf_matrices(
    first: Tuple[float, float, float, float, float, float],
    second: Tuple[float, float, float, float, float, float],
) -> Tuple[float, float, float, float, float, float]:
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _pdf_transform_point(
    matrix: Tuple[float, float, float, float, float, float],
    x: float,
    y: float,
) -> Tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _pdf_path_points(
    operator: str,
    operands: Sequence[Any],
    matrix: Tuple[float, float, float, float, float, float],
) -> Iterable[Tuple[float, float]]:
    values = [float(value) for value in operands]
    if operator in {"m", "l"}:
        yield _pdf_transform_point(matrix, values[0], values[1])
    elif operator in {"c", "v", "y"}:
        for index in range(0, len(values), 2):
            yield _pdf_transform_point(matrix, values[index], values[index + 1])
    elif operator == "re":
        x, y, width, height = values
        for point_x, point_y in (
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ):
            yield _pdf_transform_point(matrix, point_x, point_y)


def _pdf_points_to_page_rect(
    points: Sequence[Tuple[float, float]],
    page_height: float,
    derotation_matrix: fitz.Matrix,
) -> fitz.Rect:
    unrotated = [
        fitz.Point(x, page_height - y) * derotation_matrix for x, y in points
    ]
    return fitz.Rect(
        min(point.x for point in unrotated),
        min(point.y for point in unrotated),
        max(point.x for point in unrotated),
        max(point.y for point in unrotated),
    )


def _is_outline_glyph_path(
    path_rect: fitz.Rect,
    targets: Sequence[Tuple[fitz.Rect, int]],
) -> bool:
    if path_rect.is_empty:
        return False
    for target, rotation in targets:
        if (path_rect & target).is_empty:
            continue
        cross_size = target.width if rotation in {90, 270} else target.height
        glyph_limit = max(8.0, cross_size * 2.5)
        # CAD borders and dimensions are long compared with a glyph. Leaving
        # them untouched is what avoids the white holes and broken table lines
        # caused by rectangle redaction.
        if path_rect.width <= glyph_limit and path_rect.height <= glyph_limit:
            return True
    return False


def build_visual_pdf_export(
    source_content: bytes,
    layout_segments: Sequence[Dict[str, Any]],
    target_language: str,
) -> bytes:
    """Replace OCR-located text without rasterizing the complete PDF page."""
    try:
        document = fitz.open(stream=source_content, filetype="pdf")
    except Exception as exc:
        raise ValueError("源 PDF 无法用于生成译文文件") from exc
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for segment in layout_segments:
        page_number = segment.get("page_number")
        metadata = segment.get("metadata") or {}
        if not isinstance(page_number, int) or segment.get("source_kind") != "outline-text":
            document.close()
            raise ValueError("PDF 视觉译文缺少局部排版数据")
        by_page.setdefault(page_number, []).append(segment)
    base_font_path = _find_font(target_language)
    font_path, temporary_font_path = _subset_export_font(
        base_font_path,
        "\n".join(
            str(segment.get("translated_text", ""))
            for segment in layout_segments
            if _needs_replace(segment)
        )
        + "\u00a0",
    )
    try:
        for page_index, page in enumerate(document, start=1):
            segments = [item for item in by_page.get(page_index, []) if _needs_replace(item)]
            if not segments:
                continue
            diagonal_watermarks = [
                segment
                for segment in segments
                if (segment.get("metadata") or {}).get("diagonal_watermark")
            ]
            outlined = [
                segment for segment in segments if segment not in diagonal_watermarks
            ]
            outlined = _deduplicate_overlapping_outline_segments(outlined)
            write_segments = [
                part
                for segment in outlined
                for part in _split_positioned_segment(segment)
            ]
            table_layout = list(write_segments)
            page.insert_font(fontname="MetaTransVisual", fontfile=str(font_path))
            native_content_rect = fitz.Rect()
            tight_cad_page = all(
                (segment.get("metadata") or {}).get("dense_cad_tight_cover")
                or str(
                    (segment.get("metadata") or {}).get("ocr_provider", "")
                ).startswith(("gpt-indexed-image", "paddle-detect+gpt-review"))
                for segment in table_layout
            )
            # CAD pages can contain hundreds of thousands of drawing objects.
            # Indexed crops are tight source-line boxes, so a full vector scan
            # just to restore distant table lines is both unnecessary and slow.
            drawings = [] if tight_cad_page else page.get_drawings()
            if not tight_cad_page:
                _assign_table_write_rects(page, table_layout, drawings)
            restored_lines = _cover_outline_text(
                page, outlined, _structural_lines(page, drawings)
            )
            _restore_table_lines(page, restored_lines)
            for segment in write_segments:
                _insert_translation(
                    page,
                    segment,
                    target_language,
                    font_path,
                    native_content_rect,
                )
            for segment in diagonal_watermarks:
                _replace_diagonal_watermark(page, segment, font_path)
        output = document.tobytes(garbage=4, deflate=True)
    finally:
        document.close()
        if temporary_font_path is not None:
            temporary_font_path.unlink(missing_ok=True)
    _validate_written_pdf(output, layout_segments)
    return output


def _deduplicate_overlapping_outline_segments(
    segments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep one write unit when detectors found the same label in one place."""
    kept: List[Dict[str, Any]] = []
    for segment in segments:
        translated = _normalized_outline_translation(segment.get("translated_text"))
        rect = _outline_segment_rect(segment)
        rotation = int((segment.get("metadata") or {}).get("rotation") or 0) % 360
        if not translated or rect.is_empty:
            kept.append(segment)
            continue
        duplicate_index = None
        for index, existing in enumerate(kept):
            existing_translated = _normalized_outline_translation(
                existing.get("translated_text")
            )
            if translated != existing_translated:
                continue
            existing_rotation = int(
                (existing.get("metadata") or {}).get("rotation") or 0
            ) % 360
            if rotation != existing_rotation:
                continue
            existing_rect = _outline_segment_rect(existing)
            intersection = rect & existing_rect
            if intersection.is_empty:
                continue
            coverage = intersection.get_area() / max(
                1.0, min(rect.get_area(), existing_rect.get_area())
            )
            if coverage >= 0.50:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(segment)
            continue
        existing = kept[duplicate_index]
        existing_rect = _outline_segment_rect(existing)
        if rect.get_area() > existing_rect.get_area():
            kept[duplicate_index] = segment
    return kept


def _normalized_outline_translation(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _outline_segment_rect(segment: Dict[str, Any]) -> fitz.Rect:
    bbox = segment.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return fitz.Rect()
    try:
        return fitz.Rect(*(float(value) for value in bbox))
    except (TypeError, ValueError):
        return fitz.Rect()


def _parse_native_line(line: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    direction = tuple(float(value) for value in line.get("dir", (1.0, 0.0)))
    rotation = _cardinal_rotation(direction)
    if rotation is None:
        return None
    chars = []
    spans = []
    for span in line.get("spans", []):
        span_chars = list(span.get("chars", []))
        if not span_chars:
            continue
        spans.append(span)
        chars.extend(span_chars)
    text = "".join(str(char.get("c", "")) for char in chars).strip()
    if not text:
        return None
    visible = [char for char in chars if str(char.get("c", "")).strip()]
    boxes = [fitz.Rect(char["bbox"]) for char in visible if char.get("bbox")]
    if not boxes:
        return None
    rect = _union_rects(boxes)
    try:
        quad = fitz.recover_line_quad(line)
        quad_values = [[point.x, point.y] for point in quad]
    except Exception:
        quad_values = [[rect.x0, rect.y0], [rect.x1, rect.y0], [rect.x0, rect.y1], [rect.x1, rect.y1]]
    sizes = [float(span.get("size", 11.0)) for span in spans]
    color = next((int(span.get("color", 0)) for span in spans), 0)
    origin = spans[0].get("origin") or visible[0].get("origin") or (rect.x0, rect.y1)
    return {
        "text": text,
        "bbox": tuple(rect),
        "quad": quad_values,
        "origin": [float(origin[0]), float(origin[1])],
        "direction": [direction[0], direction[1]],
        "rotation": rotation,
        "wmode": int(line.get("wmode", 0)),
        "font_size": median(sizes) if sizes else 11.0,
        "color": f"#{color & 0xFFFFFF:06x}",
        "bold": _span_flag(spans, 16),
        "italic": _span_flag(spans, 2),
        "char_bboxes": [list(char["bbox"]) for char in visible if char.get("bbox")],
    }


def _group_native_lines(lines: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for line in lines:
        if current and not _lines_compatible(current[-1], line):
            groups.append(current)
            current = []
        current.append(line)
    if current:
        groups.append(current)
    return groups


def _lines_compatible(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    first_size = first["font_size"]
    second_size = second["font_size"]
    style_matches = (
        first["rotation"] == second["rotation"]
        and first["wmode"] == second["wmode"]
        and abs(first_size - second_size) <= max(0.35, min(first_size, second_size) * 0.08)
        and first["bold"] == second["bold"]
        and first["italic"] == second["italic"]
    )
    if not style_matches:
        return False
    first_rect = fitz.Rect(first["bbox"])
    second_rect = fitz.Rect(second["bbox"])
    if first["rotation"] in {0, 180}:
        center_distance = abs(
            (first_rect.y0 + first_rect.y1 - second_rect.y0 - second_rect.y1) / 2
        )
        if center_distance < min(first_rect.height, second_rect.height) * 0.55:
            return False
        gap = max(0.0, max(first_rect.y0, second_rect.y0) - min(first_rect.y1, second_rect.y1))
        overlap = max(0.0, min(first_rect.x1, second_rect.x1) - max(first_rect.x0, second_rect.x0))
        aligned = overlap >= min(first_rect.width, second_rect.width) * 0.15
    else:
        center_distance = abs(
            (first_rect.x0 + first_rect.x1 - second_rect.x0 - second_rect.x1) / 2
        )
        if center_distance < min(first_rect.width, second_rect.width) * 0.55:
            return False
        gap = max(0.0, max(first_rect.x0, second_rect.x0) - min(first_rect.x1, second_rect.x1))
        overlap = max(0.0, min(first_rect.y1, second_rect.y1) - max(first_rect.y0, second_rect.y0))
        aligned = overlap >= min(first_rect.height, second_rect.height) * 0.15
    return aligned and gap <= max(first_size, second_size) * 1.25


def _remove_native_text(page, segments: Sequence[Dict[str, Any]]) -> None:
    raw_targets = []
    protected = []
    target_ids = {item["segment_id"] for item in segments}
    for segment in segments:
        for fragment in (segment.get("metadata") or {}).get("fragments", []):
            rect = _fragment_rect(fragment)
            padding = max(
                0.75,
                min(2.5, float(segment.get("font_size") or 11.0) * 0.1),
            )
            raw_targets.append((segment["segment_id"], rect, padding))
    for unit in extract_native_page_units(page, page.number + 1):
        if unit["segment_id"] in target_ids:
            continue
        for fragment in (unit.get("metadata") or {}).get("fragments", []):
            protected.append(_fragment_rect(fragment))
    targets = []
    for unit_id, base_rect, desired_padding in raw_targets:
        rect = None
        for padding in (desired_padding, desired_padding / 2, 0.35, 0.0):
            candidate = base_rect + (-padding, -padding, padding, padding)
            if not any(_rects_collide(candidate, item) for item in protected):
                rect = candidate
                break
        if rect is None:
            raise ValueError(f"FRAGMENT_COLLISION: {unit_id}")
        targets.append((unit_id, rect))
        page.add_redact_annot(rect, fill=False, cross_out=False)
    if targets:
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            text=fitz.PDF_REDACT_TEXT_REMOVE,
        )


def _cover_outline_text(page, segments: Sequence[Dict[str, Any]], source_lines):
    if not segments:
        return []
    touched = []
    for segment in segments:
        rect = fitz.Rect((segment.get("metadata") or {}).get("cover_bbox", segment["bbox"]))
        rotation = int((segment.get("metadata") or {}).get("rotation", 0)) % 360
        cross_size = rect.width if rotation in {90, 270} else rect.height
        padding = (
            max(0.8, min(2.0, cross_size * 0.16))
            if (segment.get("metadata") or {}).get("dense_cad_tight_cover")
            else max(1.5, min(12.0, cross_size * 0.22))
        )
        cover = rect + (-padding, -padding, padding, padding)
        page.draw_rect(cover, color=None, fill=(1, 1, 1), overlay=True)
        for line in source_lines:
            clipped = _clip_line_to_rect(line, cover)
            if clipped is not None:
                touched.append(clipped)
    return _deduplicate_lines(touched)


def _replace_diagonal_watermark(page, segment, font_path: Path) -> None:
    metadata = segment.get("metadata") or {}
    tag = str(metadata.get("marked_content_tag") or "").strip()
    line_values = metadata.get("watermark_lines") or []
    if not tag or len(line_values) != 2:
        raise ValueError(f"UNSUPPORTED_DIRECTION: {segment['segment_id']}")
    if not _remove_marked_content_body(page, tag):
        raise ValueError(f"RESIDUAL_SOURCE_TEXT: {segment['segment_id']}")

    color = str(segment.get("color") or "#000000")
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
        color = "#000000"
    rgb = tuple(int(color[index : index + 2], 16) / 255 for index in (1, 3, 5))
    lines = []
    for value in line_values:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"UNSUPPORTED_DIRECTION: {segment['segment_id']}")
        lines.append((fitz.Point(value[0]), fitz.Point(value[1])))
    line_width = max(2.0, min(8.0, float(segment.get("font_size") or 20.0) * 0.1))
    for start, end in lines:
        page.draw_line(start, end, color=rgb, width=line_width, overlay=True)

    translated = str(segment.get("translated_text") or "").strip()
    if not translated:
        raise ValueError(f"ID_MISMATCH: {segment['segment_id']}")
    start_values = metadata.get("baseline_start")
    end_values = metadata.get("baseline_end")
    if not start_values or not end_values:
        raise ValueError(f"UNSUPPORTED_DIRECTION: {segment['segment_id']}")
    start, end = fitz.Point(start_values), fitz.Point(end_values)
    dx, dy = end.x - start.x, end.y - start.y
    baseline_length = math.hypot(dx, dy)
    if baseline_length < 1.0:
        raise ValueError(f"UNSUPPORTED_DIRECTION: {segment['segment_id']}")
    direction_x, direction_y = dx / baseline_length, dy / baseline_length
    angle = math.degrees(math.atan2(dy, dx))
    font_size = max(4.0, float(segment.get("font_size") or 20.0))
    font = fitz.Font(fontfile=str(font_path))
    available = baseline_length * 0.75
    text_length = font.text_length(translated, fontsize=font_size)
    if text_length > available:
        font_size *= available / max(1.0, text_length)
        text_length = font.text_length(translated, fontsize=font_size)
    if font_size < 3.0:
        raise ValueError(f"LAYOUT_OVERFLOW: {segment['segment_id']}")
    center = fitz.Point((start.x + end.x) / 2, (start.y + end.y) / 2)
    origin = fitz.Point(
        center.x - direction_x * text_length / 2,
        center.y - direction_y * text_length / 2,
    )
    page.insert_text(
        origin,
        translated,
        fontname="MetaTransVisual",
        fontfile=str(font_path),
        fontsize=font_size,
        color=rgb,
        morph=(origin, fitz.Matrix(-angle)),
        overlay=True,
    )


def _remove_marked_content_body(page, tag: str) -> bool:
    document = page.parent
    marker = re.compile(rb"/OC\s+/" + re.escape(tag.encode("ascii")) + rb"\s+BDC\b")
    operator = re.compile(rb"\b(?:BDC|BMC|EMC)\b")
    removed = False
    for xref in page.get_contents():
        data = document.xref_stream(xref)
        cursor = 0
        output = bytearray()
        stream_changed = False
        while True:
            match = marker.search(data, cursor)
            if match is None:
                output.extend(data[cursor:])
                break
            output.extend(data[cursor : match.end()])
            depth = 1
            closing = None
            for token in operator.finditer(data, match.end()):
                if token.group(0) in {b"BDC", b"BMC"}:
                    depth += 1
                else:
                    depth -= 1
                    if depth == 0:
                        closing = token
                        break
            if closing is None:
                raise ValueError("RESIDUAL_SOURCE_TEXT: marked content is malformed")
            output.extend(b"\n")
            cursor = closing.start()
            stream_changed = True
            removed = True
        if stream_changed:
            document.update_stream(xref, bytes(output))
    return removed


def _restore_table_lines(page, lines) -> None:
    for start, end, width, color in lines:
        page.draw_line(start, end, color=color, width=width, overlay=True)


def _insert_translation(
    page,
    segment,
    target_language,
    font_path: Path,
    native_content_rect: fitz.Rect,
) -> None:
    metadata = segment.get("metadata") or {}
    rect = fitz.Rect(metadata.get("write_bbox", segment["bbox"]))
    rotation = int(metadata.get("rotation", 0)) % 360
    if rotation not in {0, 90, 180, 270}:
        raise ValueError(f"UNSUPPORTED_DIRECTION: {segment['segment_id']}")
    translated = str(segment.get("translated_text", "")).strip()
    if not translated:
        raise ValueError(f"ID_MISMATCH: {segment['segment_id']}")
    original_size = max(3.0, float(segment.get("font_size") or 11.0))
    if (
        segment.get("source_kind") == "text"
        and metadata.get("page_type") == "table"
    ):
        original_size *= 0.75
    if int(metadata.get("line_count") or 1) > 1:
        original_size *= 0.98
    if metadata.get("page_type") == "text" and not native_content_rect.is_empty:
        if rotation in {0, 180}:
            if segment.get("alignment") == "center":
                rect.x0, rect.x1 = native_content_rect.x0, native_content_rect.x1
            elif segment.get("alignment") == "right":
                rect.x0 = native_content_rect.x0
            else:
                rect.x1 = native_content_rect.x1
        else:
            if segment.get("alignment") == "center":
                rect.y0, rect.y1 = native_content_rect.y0, native_content_rect.y1
            elif segment.get("alignment") == "right":
                rect.y0 = native_content_rect.y0
            else:
                rect.y1 = native_content_rect.y1
    leading = _segment_leading(metadata, original_size, rotation)
    alignment = str(segment.get("alignment") or "left")
    color = str(segment.get("color") or "#000000")
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
        color = "#000000"
    plain_text = _translation_plain_text(translated, metadata)
    align_value = {"left": 0, "center": 1, "right": 2, "justify": 3}.get(
        alignment, 0
    )
    if rotation == 180 and metadata.get("page_type") != "text":
        align_value = 0
    rgb = tuple(int(color[index : index + 2], 16) / 255 for index in (1, 3, 5))
    lineheight = max(1.0, min(1.5, leading / original_size))
    minimum_ratio = (
        0.15
        if segment.get("source_kind") == "outline-text"
        else 0.5
        if metadata.get("table_region") or metadata.get("page_type") == "table"
        else 0.55
    )
    minimum_size = max(
        0.8 if segment.get("source_kind") == "outline-text" else 3.0,
        original_size * minimum_ratio,
    )
    fitted_size = _largest_fitting_font_size(
        page,
        rect,
        plain_text,
        original_size,
        minimum_size,
        lineheight,
        align_value,
        rotation,
        font_path,
    )
    if fitted_size is None:
        raise ValueError(f"LAYOUT_OVERFLOW: {segment['segment_id']}")
    spare = page.insert_textbox(
        rect,
        plain_text,
        fontname="MetaTransVisual",
        fontfile=str(font_path),
        fontsize=fitted_size,
        lineheight=lineheight,
        color=rgb,
        align=align_value,
        rotate=rotation,
        overlay=True,
    )
    if spare < 0:
        raise ValueError(f"LAYOUT_OVERFLOW: {segment['segment_id']}")


def _translation_plain_text(translated: str, metadata: Dict[str, Any]) -> str:
    translated = re.sub(
        r"(?m)^(/?\d+(?:\.\d+)+)\s+",
        lambda match: match.group(1) + "\u00a0",
        translated,
    )
    if metadata.get("page_type") != "text" or "\n" not in translated:
        return translated
    lines = translated.splitlines()
    output = []
    for line in lines:
        value = line.strip()
        if not value:
            continue
        if not output:
            output.append(value)
        elif re.match(r"^/?\d+(?:\.\d+)+\b", value):
            output.append("\n" + value)
        else:
            output.append(" " + value)
    return "".join(output)


def _split_positioned_segment(segment: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = segment.get("metadata") or {}
    fragments = metadata.get("fragments") or []
    translated_parts = re.split(r"\s{2,}", str(segment.get("translated_text", "")).strip())
    if len(fragments) != 1 or len(translated_parts) < 2:
        return [segment]
    fragment = fragments[0]
    source_parts = re.split(r"\s{2,}", str(fragment.get("text", "")).strip())
    char_boxes = [fitz.Rect(value) for value in fragment.get("char_bboxes") or []]
    if len(source_parts) != len(translated_parts) or not char_boxes:
        return [segment]

    counts = [len(re.sub(r"\s", "", value)) for value in source_parts]
    if any(count < 1 for count in counts) or sum(counts) != len(char_boxes):
        return [segment]
    direction = tuple(float(value) for value in metadata.get("direction", (1.0, 0.0)))
    boundaries = []
    offset = 0
    for count in counts[:-1]:
        offset += count
        first, second = char_boxes[offset - 1], char_boxes[offset]
        if abs(direction[0]) >= abs(direction[1]):
            gap = (
                second.x0 - first.x1
                if direction[0] >= 0
                else first.x0 - second.x1
            )
        else:
            gap = (
                second.y0 - first.y1
                if direction[1] >= 0
                else first.y0 - second.y1
            )
        boundaries.append(gap)
    minimum_gap = max(3.0, float(segment.get("font_size") or 11.0) * 0.75)
    if any(gap < minimum_gap for gap in boundaries):
        return [segment]

    output = []
    offset = 0
    for index, (source_text, translated_text, count) in enumerate(
        zip(source_parts, translated_parts, counts), start=1
    ):
        boxes = char_boxes[offset : offset + count]
        offset += count
        rect = _union_rects(boxes)
        part = dict(segment)
        part["segment_id"] = f"{segment['segment_id']}:part{index}"
        part["text"] = source_text
        part["translated_text"] = translated_text
        part["bbox"] = tuple(rect)
        part["alignment"] = "left"
        part_metadata = dict(metadata)
        part_metadata.pop("write_bbox", None)
        part_metadata["positioned_part"] = True
        part["metadata"] = part_metadata
        output.append(part)
    return output


def _largest_fitting_font_size(
    page,
    rect: fitz.Rect,
    text: str,
    original_size: float,
    minimum_size: float,
    lineheight: float,
    align: int,
    rotation: int,
    font_path: Path,
) -> Optional[float]:
    def fits(font_size: float) -> bool:
        shape = page.new_shape()
        spare = shape.insert_textbox(
            rect,
            text,
            fontname="MetaTransVisual",
            fontfile=str(font_path),
            fontsize=font_size,
            lineheight=lineheight,
            align=align,
            rotate=rotation,
        )
        return spare >= 0

    if fits(original_size):
        return original_size
    if not fits(minimum_size):
        return None
    low, high = minimum_size, original_size
    for _ in range(9):
        middle = (low + high) / 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low * 0.995


def _fragment_rect(fragment: Dict[str, Any]) -> fitz.Rect:
    quad = fragment.get("quad")
    if isinstance(quad, list) and len(quad) == 4:
        try:
            points = [fitz.Point(float(point[0]), float(point[1])) for point in quad]
            return fitz.Rect(
                min(point.x for point in points),
                min(point.y for point in points),
                max(point.x for point in points),
                max(point.y for point in points),
            )
        except (TypeError, ValueError):
            pass
    return fitz.Rect(fragment["bbox"])


def _assign_table_write_rects(page, segments, drawings) -> None:
    grid = _table_grid(page, drawings)
    for segment in segments:
        source_rect = fitz.Rect(
            (segment.get("metadata") or {}).get("cover_bbox", segment["bbox"])
        )
        rotation = int((segment.get("metadata") or {}).get("rotation", 0)) % 360
        if segment.get("source_kind") == "outline-text" and rotation in {90, 270}:
            metadata = segment.setdefault("metadata", {})
            metadata["write_bbox"] = tuple(
                source_rect + (-0.5, -0.5, 0.5, 0.5)
            )
            segment["alignment"] = "left"
            continue
        cell = _table_cell_rect(page, source_rect, grid)
        if cell is None:
            continue
        inset = max(1.0, min(2.0, cell.height * 0.08))
        interior = cell + (inset, inset, -inset, -inset)
        if rotation in {0, 180}:
            x0 = max(interior.x0, source_rect.x0) if rotation == 0 else interior.x0
            x1 = interior.x1 if rotation == 0 else min(interior.x1, source_rect.x1)
            write_rect = fitz.Rect(
                x0,
                max(page.rect.y0, source_rect.y0 - 0.5),
                x1,
                min(page.rect.y1, source_rect.y1 + 0.5),
            )
        else:
            y0 = max(interior.y0, source_rect.y0) if rotation == 90 else interior.y0
            y1 = interior.y1 if rotation == 90 else min(interior.y1, source_rect.y1)
            write_rect = fitz.Rect(
                max(page.rect.x0, source_rect.x0 - 0.5),
                y0,
                min(page.rect.x1, source_rect.x1 + 0.5),
                y1,
            )
        if write_rect.is_empty:
            continue
        metadata = segment.setdefault("metadata", {})
        metadata["write_bbox"] = tuple(write_rect)
        segment["alignment"] = "left"


def _table_grid(page, drawings):
    cells = []
    try:
        finder = page.find_tables()
        for table in finder.tables:
            for cell in table.cells:
                if cell:
                    cells.append(fitz.Rect(cell) * page.derotation_matrix)
    except Exception:
        cells = []
    vertical = []
    horizontal = []
    for drawing in drawings:
        for item in drawing.get("items", []):
            if not item or item[0] != "l":
                continue
            start, end = fitz.Point(item[1]), fitz.Point(item[2])
            dx, dy = abs(end.x - start.x), abs(end.y - start.y)
            if dx <= 0.5 and dy >= page.rect.height * 0.08:
                vertical.append(
                    ((start.x + end.x) / 2, min(start.y, end.y), max(start.y, end.y))
                )
            elif dy <= 0.5 and dx >= page.rect.width * 0.08:
                horizontal.append(
                    ((start.y + end.y) / 2, min(start.x, end.x), max(start.x, end.x))
                )
    return cells, vertical, horizontal


def _table_cell_rect(page, source_rect: fitz.Rect, grid) -> Optional[fitz.Rect]:
    center = fitz.Point(
        source_rect.x0 + source_rect.width / 2,
        source_rect.y0 + source_rect.height / 2,
    )
    tolerance = max(2.0, min(6.0, source_rect.height * 0.2))
    cells, vertical, horizontal = grid
    containing_cells = [cell for cell in cells if cell.contains(center)]
    if containing_cells:
        cell = min(containing_cells, key=lambda value: value.get_area())
        if (
            cell.width + tolerance >= source_rect.width
            and cell.height + tolerance >= source_rect.height
            and _cell_is_local(page, source_rect, cell)
        ):
            return cell
    active_vertical = [
        x for x, top, bottom in vertical if top - 1 <= center.y <= bottom + 1
    ]
    active_horizontal = [
        y for y, left, right in horizontal if left - 1 <= center.x <= right + 1
    ]
    lefts = [value for value in active_vertical if value <= center.x]
    rights = [value for value in active_vertical if value >= center.x]
    tops = [value for value in active_horizontal if value <= center.y]
    bottoms = [value for value in active_horizontal if value >= center.y]
    if not lefts or not rights or not tops or not bottoms:
        return None
    cell = fitz.Rect(max(lefts), max(tops), min(rights), min(bottoms))
    if (
        cell.width + tolerance < source_rect.width
        or cell.height + tolerance < source_rect.height
        or not _cell_is_local(page, source_rect, cell)
    ):
        return None
    return cell


def _cell_is_local(page, source_rect: fitz.Rect, cell: fitz.Rect) -> bool:
    """Reject page-spanning pseudo-cells returned for compound ruled layouts."""
    maximum_width = max(source_rect.width * 8.0, page.rect.width * 0.12)
    maximum_height = max(source_rect.height * 6.0, page.rect.height * 0.08)
    return cell.width <= maximum_width and cell.height <= maximum_height


def _segment_leading(metadata: Dict[str, Any], font_size: float, rotation: int) -> float:
    origins = [
        fragment.get("origin")
        for fragment in metadata.get("fragments", [])
        if isinstance(fragment.get("origin"), (list, tuple))
        and len(fragment["origin"]) == 2
    ]
    horizontal = rotation in {0, 180}
    distances = []
    for first, second in zip(origins, origins[1:]):
        distance = (
            abs(float(second[1]) - float(first[1]))
            if horizontal
            else abs(float(second[0]) - float(first[0]))
        )
        if distance > 0:
            distances.append(distance)
    if distances:
        return max(font_size, median(distances))
    return max(font_size, float(metadata.get("leading") or font_size * 1.15))


def _validate_written_pdf(content: bytes, segments: Sequence[Dict[str, Any]]) -> None:
    expected = {item["segment_id"] for item in segments if _needs_replace(item)}
    if not expected:
        return
    document = fitz.open(stream=content, filetype="pdf")
    try:
        if document.page_count < 1:
            raise ValueError("PDF 译文生成失败")
        for page in document:
            page.get_text("text")
    finally:
        document.close()


def _get_paddle_ocr():
    engine = getattr(_OCR_THREAD_LOCAL, "engine", None)
    if engine is not None:
        return engine
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("未安装 PaddleOCR，无法识别表格轮廓文字") from exc
    # OCR work is already bounded by APP_PDF_OCR_CONCURRENCY. Give each
    # executor thread its own predictor so independent pages can use separate
    # CPU cores without sharing a non-thread-safe Paddle pipeline.
    with _OCR_LOCK:
        engine = getattr(_OCR_THREAD_LOCAL, "engine", None)
        if engine is None:
            engine = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="th_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=2400,
                text_det_limit_type="max",
                text_det_thresh=0.2,
                text_det_box_thresh=0.4,
                text_rec_score_thresh=0.3,
                enable_mkldnn=False,
            )
            _OCR_THREAD_LOCAL.engine = engine
            _OCR_THREAD_LOCAL.predict_lock = threading.Lock()
    return engine


def _get_cad_text_detector():
    """Return the lightweight, geometry-only PaddleOCR detector for CAD."""
    global _CAD_TEXT_DETECTOR
    if _CAD_TEXT_DETECTOR is not None:
        return _CAD_TEXT_DETECTOR
    try:
        from paddleocr import TextDetection
    except ImportError as exc:
        raise RuntimeError("未安装 PaddleOCR，无法定位 CAD 文字") from exc
    with _CAD_TEXT_DETECTOR_LOCK:
        if _CAD_TEXT_DETECTOR is None:
            # PP-OCR v5 is already used by the application and works with the
            # supported Paddle runtime. The v6 default currently is not
            # compatible with the pinned Python 3.8/Paddle combination.
            _CAD_TEXT_DETECTOR = TextDetection(
                model_name="PP-OCRv5_mobile_det",
                limit_side_len=1600,
                limit_type="max",
                enable_mkldnn=False,
            )
    return _CAD_TEXT_DETECTOR


def _paddle_ocr_predict_lock():
    lock = getattr(_OCR_THREAD_LOCAL, "predict_lock", None)
    if lock is None:
        lock = threading.Lock()
        _OCR_THREAD_LOCAL.predict_lock = lock
    return lock


def _get_text_recognizer():
    global _TEXT_RECOGNIZER
    if _TEXT_RECOGNIZER is not None:
        return _TEXT_RECOGNIZER
    try:
        from paddleocr import TextRecognition
    except ImportError as exc:
        raise RuntimeError("未安装 PaddleOCR，无法识别 CAD 文字") from exc
    with _TEXT_RECOGNIZER_LOCK:
        if _TEXT_RECOGNIZER is None:
            _TEXT_RECOGNIZER = TextRecognition(
                model_name="th_PP-OCRv5_mobile_rec",
                enable_mkldnn=False,
            )
    return _TEXT_RECOGNIZER


def _ocr_polygon_to_unrotated_rect(page, polygon, width: int, height: int) -> fitz.Rect:
    points = [fitz.Point(float(item[0]), float(item[1])) for item in polygon]
    display = [
        fitz.Point(point.x / width * page.rect.width, point.y / height * page.rect.height)
        for point in points
    ]
    unrotated = [point * page.derotation_matrix for point in display]
    return fitz.Rect(
        min(point.x for point in unrotated),
        min(point.y for point in unrotated),
        max(point.x for point in unrotated),
        max(point.y for point in unrotated),
    )


def _restore_clockwise_polygon(polygon, original_height: int):
    """Map a polygon from a clockwise-rotated image back to source pixels."""
    return [
        [float(point[1]), float(original_height - 1 - point[0])]
        for point in polygon
    ]


def _display_direction_to_unrotated_rotation(page, direction) -> int:
    origin = fitz.Point(0.0, 0.0) * page.derotation_matrix
    endpoint = fitz.Point(float(direction[0]), float(direction[1])) * page.derotation_matrix
    rotation = _cardinal_rotation((endpoint.x - origin.x, endpoint.y - origin.y))
    if rotation is None:
        raise ValueError("UNSUPPORTED_DIRECTION")
    return rotation


def _structural_lines(page, drawings=None):
    lines = []
    horizontal_minimum = max(40.0, page.rect.width * 0.1)
    vertical_minimum = max(40.0, page.rect.height * 0.1)
    for drawing in drawings if drawings is not None else page.get_drawings():
        width = max(0.15, float(drawing.get("width") or 0.5))
        color = drawing.get("color") or (0.45, 0.45, 0.45)
        for item in drawing.get("items", []):
            if not item or item[0] != "l":
                continue
            start, end = fitz.Point(item[1]), fitz.Point(item[2])
            dx, dy = abs(end.x - start.x), abs(end.y - start.y)
            if (dx <= 0.4 and dy >= vertical_minimum) or (
                dy <= 0.4 and dx >= horizontal_minimum
            ):
                lines.append((start, end, width, color))
    return lines


def _clip_line_to_rect(line, rect: fitz.Rect):
    start, end, width, color = line
    dx, dy = end.x - start.x, end.y - start.y
    p = (-dx, dx, -dy, dy)
    q = (start.x - rect.x0, rect.x1 - start.x, start.y - rect.y0, rect.y1 - start.y)
    lower, upper = 0.0, 1.0
    for coefficient, distance in zip(p, q):
        if abs(coefficient) < 1e-9:
            if distance < 0:
                return None
            continue
        ratio = distance / coefficient
        if coefficient < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    clipped_start = fitz.Point(start.x + lower * dx, start.y + lower * dy)
    clipped_end = fitz.Point(start.x + upper * dx, start.y + upper * dy)
    if math.dist(clipped_start, clipped_end) < 0.5:
        return None
    return clipped_start, clipped_end, width, color


def _deduplicate_lines(lines):
    output = []
    seen = set()
    for line in lines:
        start, end, width, color = line
        key = tuple(round(value, 2) for value in (start.x, start.y, end.x, end.y, width))
        if key in seen:
            continue
        seen.add(key)
        output.append(line)
    return output


def _find_font(target_language: str) -> Path:
    environment = {
        "zh": "APP_EXPORT_FONT_ZH",
        "th": "APP_EXPORT_FONT_TH",
        "en": "APP_EXPORT_FONT_EN",
    }[target_language]
    candidates = [
        os.getenv(environment, ""),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate) if candidate else None
        if path and path.is_file():
            return path
    raise ValueError("未找到可用于生成译文 PDF 的字体")


def _subset_export_font(font_path: Path, text: str) -> Tuple[Path, Optional[Path]]:
    if not text.strip():
        return font_path, None
    temporary_path = None
    font = None
    try:
        from fontTools import subset
        from fontTools.ttLib import TTFont

        options = subset.Options()
        options.retain_gids = True
        kwargs = {"fontNumber": 0} if font_path.suffix.lower() in {".ttc", ".otc"} else {}
        font = TTFont(str(font_path), **kwargs)
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(text=text)
        subsetter.subset(font)
        temporary = tempfile.NamedTemporaryFile(
            prefix="meta-trans-font-", suffix=".ttf", delete=False
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        font.save(str(temporary_path))
        font.close()
        return temporary_path, temporary_path
    except Exception:
        if font is not None:
            font.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        return font_path, None


def _cardinal_rotation(direction: Tuple[float, float]) -> Optional[int]:
    angle = math.degrees(math.atan2(direction[1], direction[0])) % 360
    candidates = {0: 0.0, 90: 90.0, 180: 180.0, 270: 270.0}
    rotation = min(candidates, key=lambda value: abs(((angle - value + 180) % 360) - 180))
    delta = abs(((angle - rotation + 180) % 360) - 180)
    return rotation if delta <= 1.5 else None


def _native_alignment(group, bbox: fitz.Rect, content: fitz.Rect) -> str:
    if len(group) > 1:
        return "left"
    width = max(1.0, content.width)
    if bbox.width >= width * 0.8:
        return "left"
    center_delta = abs((bbox.x0 + bbox.x1 - content.x0 - content.x1) / 2)
    if center_delta <= width * 0.06:
        return "center"
    if abs(bbox.x1 - content.x1) <= width * 0.025:
        return "right"
    return "left"


def _first_line_indent(group, bbox: fitz.Rect) -> float:
    if len(group) < 2 or group[0]["rotation"] != 0:
        return 0.0
    return max(0.0, fitz.Rect(group[0]["bbox"]).x0 - bbox.x0)


def _native_leading(group, font_size: float) -> float:
    if len(group) < 2:
        return max(font_size, fitz.Rect(group[0]["bbox"]).height)
    origins = [line["origin"] for line in group]
    horizontal = group[0]["rotation"] in {0, 180}
    distances = []
    for first, second in zip(origins, origins[1:]):
        distance = abs(second[1] - first[1]) if horizontal else abs(second[0] - first[0])
        if distance > 0:
            distances.append(distance)
    return max(font_size, median(distances) if distances else font_size * 1.15)


def _span_flag(spans, flag: int) -> bool:
    weighted = [(max(1, len(span.get("chars", []))), int(span.get("flags", 0))) for span in spans]
    total = sum(weight for weight, _ in weighted)
    return total > 0 and sum(weight for weight, value in weighted if value & flag) * 2 >= total


def _dominant(group, field: str) -> bool:
    return sum(1 for line in group if line[field]) * 2 >= len(group)


def _union_rects(rects: Iterable[fitz.Rect]) -> fitz.Rect:
    values = list(rects)
    if not values:
        return fitz.Rect()
    result = fitz.Rect(values[0])
    for rect in values[1:]:
        result |= rect
    return result


def _coverage(first: fitz.Rect, second: fitz.Rect) -> float:
    intersection = first & second
    return 0.0 if intersection.is_empty else intersection.get_area() / max(1.0, first.get_area())


def _overlap_smaller(first: fitz.Rect, second: fitz.Rect) -> float:
    intersection = first & second
    if intersection.is_empty:
        return 0.0
    return intersection.get_area() / max(1.0, min(first.get_area(), second.get_area()))


def _is_translatable_text(text: str) -> bool:
    value = re.sub(r"\s+", "", text)
    if not value or not _LETTER_PATTERN.search(value):
        return False
    if _CJK_PATTERN.search(value):
        return True
    return len(_THAI_LATIN_PATTERN.findall(value)) >= 2


def _rects_collide(first: fitz.Rect, second: fitz.Rect) -> bool:
    intersection = first & second
    if intersection.is_empty:
        return False
    return intersection.get_area() > max(0.5, min(first.get_area(), second.get_area()) * 0.08)


def _needs_replace(segment: Dict[str, Any]) -> bool:
    source = re.sub(r"\s+", "", str(segment.get("text", "")))
    translated = re.sub(r"\s+", "", str(segment.get("translated_text", "")))
    return bool(translated and source != translated)
