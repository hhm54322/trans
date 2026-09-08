import html
import os
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import fitz
from lxml import etree
from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml.ns import qn
from docx.shared import Pt
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from .native_pdf import build_native_pdf_export, build_text_layer_fallback_export
from .visual_pdf import build_visual_pdf_export

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:  # Optional local image repair for scanned/CAD text.
    cv2 = None
    np = None
    Image = None
else:
    # Avoid OpenCV spawning a large worker pool for small text patches.
    cv2.setNumThreads(1)


PAGE_MARKER = re.compile(r"【第\s*(\d+)\s*页】")
REPORTLAB_CHINESE_FONT = "MetaTransChinese"
REPORTLAB_THAI_FONT = "MetaTransThai"
REPORTLAB_ENGLISH_FONT = "MetaTransEnglish"
_REPORTLAB_FONT_LOCK = threading.Lock()
WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DRAWING_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/main"
PRESENTATION_NAMESPACE = "http://schemas.openxmlformats.org/presentationml/2006/main"


@dataclass
class ExportedDocument:
    filename: str
    path: Path
    media_type: str


def create_document_export(
    *,
    item_id: str,
    source_filename: str,
    source_content: bytes,
    translated_text: str,
    target_language: str,
    output_directory: Path,
    layout_segments: Optional[List[Dict[str, Any]]] = None,
    prepared_pdf_content: Optional[bytes] = None,
) -> ExportedDocument:
    extension = Path(source_filename).suffix.lower()
    if extension not in {".pdf", ".docx", ".pptx", ".txt", ".md"}:
        raise ValueError("该文档格式暂不支持导出")

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{item_id}{extension}"
    export_filename = f"{Path(source_filename).stem}-译文{extension}"

    if extension == ".pdf":
        output_path.write_bytes(
            build_adaptive_pdf_export(
                source_content,
                layout_segments,
                target_language,
                prepared_pdf_content=prepared_pdf_content,
            )
            if layout_segments
            else build_pdf_export(source_content, translated_text, target_language)
        )
        media_type = "application/pdf"
    elif extension == ".docx":
        output_path.write_bytes(
            build_layout_docx_export(source_content, layout_segments)
            if layout_segments
            else build_docx_export(source_content, translated_text, target_language)
        )
        media_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    elif extension == ".pptx":
        if not layout_segments:
            raise ValueError("PPTX 译文缺少版式定位信息")
        output_path.write_bytes(
            build_layout_pptx_export(source_content, layout_segments)
        )
        media_type = (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
    else:
        output_path.write_text(translated_text.strip() + "\n", encoding="utf-8")
        media_type = "text/markdown" if extension == ".md" else "text/plain"

    return ExportedDocument(export_filename, output_path, media_type)


def build_adaptive_pdf_export(
    source_content: bytes,
    layout_segments: List[Dict[str, Any]],
    target_language: str,
    *,
    prepared_pdf_content: Optional[bytes] = None,
) -> bytes:
    native_segments = [
        segment
        for segment in layout_segments
        if (segment.get("metadata") or {}).get("native_pdf_version")
    ]
    text_layer_fallback_segments = [
        segment
        for segment in layout_segments
        if (segment.get("metadata") or {}).get("text_layer_fallback")
    ]
    visual_segments = [
        segment
        for segment in layout_segments
        if not (segment.get("metadata") or {}).get("native_pdf_version")
        and not (segment.get("metadata") or {}).get("text_layer_fallback")
    ]
    output = prepared_pdf_content or source_content
    if native_segments:
        output = build_native_pdf_export(
            output,
            native_segments,
            target_language,
            text_already_removed=prepared_pdf_content is not None,
            validation_source_content=source_content,
        )
    if text_layer_fallback_segments:
        output = build_text_layer_fallback_export(
            output, text_layer_fallback_segments, target_language
        )
    if visual_segments:
        legacy_outline_segments = [
            segment
            for segment in visual_segments
            if segment.get("source_kind") == "outline-text"
        ]
        legacy_layout_segments = [
            segment
            for segment in visual_segments
            if segment.get("source_kind") != "outline-text"
        ]
        if legacy_outline_segments:
            output = build_visual_pdf_export(
                output, legacy_outline_segments, target_language
            )
        if legacy_layout_segments:
            output = build_layout_pdf_export(
                output, legacy_layout_segments, target_language
            )
    return output


def create_unformatted_document_export(
    *,
    item_id: str,
    source_filename: str,
    source_content: bytes,
    translated_text: str,
    target_language: str,
    output_directory: Path,
) -> ExportedDocument:
    extension = Path(source_filename).suffix.lower()
    if extension not in {".pdf", ".docx"}:
        raise ValueError("该文档格式暂不支持未排版导出")

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{item_id}-unformatted{extension}"
    export_filename = f"{Path(source_filename).stem}-未排版译文{extension}"

    if extension == ".pdf":
        output_path.write_bytes(
            build_pdf_export(source_content, translated_text, target_language)
        )
        media_type = "application/pdf"
    else:
        output_path.write_bytes(
            build_docx_export(source_content, translated_text, target_language)
        )
        media_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    return ExportedDocument(export_filename, output_path, media_type)


def build_layout_pdf_export(
    source_content: bytes,
    layout_segments: List[Dict[str, Any]],
    target_language: str,
) -> bytes:
    try:
        source = fitz.open(stream=source_content, filetype="pdf")
    except Exception as exc:
        raise ValueError("源 PDF 无法用于生成译文文件") from exc

    segments_by_page = _segments_by_page(
        _coalesce_visual_bilingual_segments(layout_segments)
    )
    try:
        # PyMuPDF exposes text coordinates in the page's unrotated coordinate
        # space. Normalize rotated pages before redacting and drawing, then
        # apply the same matrix to every translated text box. This keeps the
        # visual page unchanged without rotating or mirroring the overlay.
        for page_index, page in enumerate(source):
            page_segments = segments_by_page.get(page_index + 1, [])
            # Layout caches created by the early CAD prototype used the
            # already-rotated display coordinate space and carry ``cad:`` ids.
            # New parser/vision segments are in the PDF's unrotated space.
            # Detect the legacy form so it is not rotated a second time.
            legacy_display_coords = any(
                str(segment.get("segment_id", "")).startswith("cad:")
                and not (segment.get("metadata") or {}).get("normalized_bbox")
                for segment in page_segments
            )
            if not page.rotation:
                continue
            transform = page.remove_rotation()
            if legacy_display_coords:
                # These coordinates already describe the visible upright
                # page. Removing /Rotate preserves that appearance, so the
                # boxes must remain unchanged.
                continue
            transformed_segments = []
            for segment in page_segments:
                transformed = dict(segment)
                bbox = segment.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    metadata = dict(segment.get("metadata") or {})
                    if metadata.get("display_bbox"):
                        transformed_segments.append(transformed)
                        continue
                    # Removing /Rotate transforms the page content itself, so
                    # cleanup and overlay must use the same transformed box.
                    # Keeping the pre-transform box here leaves source text
                    # untouched on 90/180/270 degree pages.
                    transformed["metadata"] = metadata
                    rect = fitz.Rect(bbox) * transform
                    transformed["bbox"] = [rect.x0, rect.y0, rect.x1, rect.y1]
                transformed_segments.append(transformed)
            segments_by_page[page_index + 1] = transformed_segments

        # Normalize visual blocks against the actual page grid after page
        # rotation has been removed. This collapses overlapping crop results
        # into the table cells used by the proven v2 export path.
        for page_index, page in enumerate(source):
            page_segments = segments_by_page.get(page_index + 1, [])
            if any(_is_visual_layout_segment(item) for item in page_segments):
                scanned_page = any(
                    str((item.get("metadata") or {}).get("page_type", ""))
                    in {"image", "mixed"}
                    for item in page_segments
                )
                segments_by_page[page_index + 1] = (
                    _deduplicate_visual_segments(page, page_segments)
                    if scanned_page
                    else _coalesce_visual_page_cells(page, page_segments)
                )

        for page_index, page in enumerate(source):
            page_segments = segments_by_page.get(page_index + 1, [])
            visual_segments = [
                segment
                for segment in page_segments
                if _is_visual_layout_segment(segment)
                and _segment_needs_replacement(segment)
            ]
            repair_visual_segments = bool(visual_segments) and not all(
                str((segment.get("metadata") or {}).get("visual_provider", ""))
                .lower()
                .startswith("demo")
                for segment in visual_segments
            )
            # A scan is already raster data. Repair it once at page resolution
            # and place one clean backing over the source image. Per-line image
            # patches create seams and compound coordinate errors. Vector/CAD
            # pages remain on their separate graphics-preserving path.
            scanned_page = any(
                str((segment.get("metadata") or {}).get("page_type", ""))
                in {"image", "mixed"}
                for segment in visual_segments
            )
            clean_raster_backing = scanned_page and _insert_clean_visual_raster_backing(
                page,
                visual_segments,
            )
            cleanup_grid_lines = _extract_pdf_grid_lines(page) if visual_segments else ([], [])
            has_redactions = False
            for segment in ([] if clean_raster_backing else page_segments):
                if not _segment_needs_replacement(segment):
                    continue
                metadata = segment.get("metadata") or {}
                page_type = metadata.get("page_type")
                is_visual = _is_visual_layout_segment(segment)
                # Pure scanned pages are image-backed and need the local
                # inpainting path above; transparent vector redaction would
                # not remove those pixels. Visual CAD boxes are handled only
                # by that path so their graphics are never redacted.
                can_redact_visual = page_type in {"vector", "vector_mixed"}
                if is_visual:
                    # Pixel repair handles outlined glyphs. Also remove any
                    # selectable text objects inside the same box, but keep
                    # every graphic object untouched.
                    rect = _pdf_segment_cleanup_rect(segment, page.rect)
                    if rect.is_empty:
                        continue
                    if rect.get_area() / max(1.0, page.rect.get_area()) > 0.20:
                        continue
                    font_size = max(1.0, float(segment.get("font_size") or 11.0))
                    padding_x = max(12.0, min(16.0, font_size * 1.05))
                    cell = _nearest_grid_bounds(rect, cleanup_grid_lines)
                    table_like = bool(
                        cell
                        and cell.height <= max(rect.height * 2.5, 40.0)
                        and cell.width >= rect.width * 0.75
                    )
                    header_like = bool(
                        40.0 <= rect.height <= 60.0
                        and 100.0 <= rect.width <= 500.0
                    )
                    # Vision boxes on rotated CAD tables can be one row off.
                    # Expand only small, clearly tabular rows; title blocks and
                    # logos keep the tight source box to avoid collateral loss.
                    padding_y = (
                        max(20.0, min(36.0, font_size * 2.5))
                        if table_like
                        else (
                            max(20.0, min(28.0, rect.height * 0.55))
                            if header_like
                            else max(8.0, min(14.0, font_size * 0.9))
                        )
                    )
                    redact_rect = rect + (
                        -padding_x,
                        -padding_y,
                        padding_x,
                        padding_y,
                    )
                    redact_rect.intersect(page.rect)
                    page.add_redact_annot(redact_rect, fill=False)
                    has_redactions = True
                    continue
                if segment.get("source_kind") != "text" and not can_redact_visual:
                    continue
                rect = _pdf_segment_cleanup_rect(segment, page.rect)
                if rect.is_empty:
                    continue
                if can_redact_visual and rect.get_area() / max(1.0, page.rect.get_area()) > 0.20:
                    # A full-page/large visual box is usually a model
                    # detection fallback rather than a text line. Redacting
                    # it could remove legitimate CAD geometry, so leave the
                    # source artwork untouched and draw only the translation.
                    continue
                # Vision/OCR boxes around CAD outline glyphs are approximate.
                # Expand only the redaction area (not the draw area) slightly
                # so native glyph contours outside the text box are removed.
                redact_rect = rect
                if metadata.get("text_layer_fallback"):
                    # This box comes from selectable PDF text, not approximate
                    # OCR geometry. Keep it tight so adjacent labels, numbers
                    # and table cells cannot be removed.
                    font_size = max(0.75, float(segment.get("font_size") or 11.0))
                    padding = max(0.35, min(1.25, font_size * 0.18))
                    redact_rect = rect + (-padding, -padding, padding, padding)
                    redact_rect.intersect(page.rect)
                elif metadata.get("visual_residual") or metadata.get("tile"):
                    # Vector-outline glyphs frequently extend beyond the
                    # model's visual bbox by a few points. Give CAD visual
                    # blocks a little more clearance; graphics crossing the
                    # box remain protected by graphics=1 below.
                    # Keep the clearance tight on CAD drawings. A large
                    # transparent redaction can remove hatch fills that are
                    # fully contained in the box and appear as white holes.
                    padding = 3.0 if metadata.get("visual_pass") else 2.0
                    redact_rect = rect + (-padding, -padding, padding, padding)
                    redact_rect.intersect(page.rect)
                elif page_type in {"vector", "vector_mixed"}:
                    # Native PDF text on CAD pages may be backed by outlined
                    # glyphs. A small expansion removes those contours while
                    # preserving crossing dimension/table graphics.
                    redact_rect = rect + (-4.0, -4.0, 4.0, 4.0)
                    redact_rect.intersect(page.rect)
                elif str(segment.get("segment_id", "")).startswith("cad:"):
                    # Legacy CAD boxes follow the visible glyph bounds very
                    # tightly. Include accents and curve extrema so contained
                    # outline objects are removed in full, while crossing
                    # table/dimension lines remain protected by graphics=1.
                    font_size = max(1.0, float(segment.get("font_size") or 11.0))
                    padding = max(3.0, min(20.0, font_size * 0.20))
                    redact_rect = rect + (-padding, -padding, padding, padding)
                    redact_rect.intersect(page.rect)
                page.add_redact_annot(redact_rect, fill=False)
                has_redactions = True
            if has_redactions:
                # Layout replacement must remove text only. Images and vector
                # graphics may cross a text box in CAD drawings, tables and
                # diagrams; removing them creates visible white holes even
                # when the redaction itself has no fill.
                # Remove graphic objects fully contained by a detected visual
                # text box (outlined glyphs), while preserving table and CAD
                # lines that cross the box. Native text pages never enter this
                # visual branch and continue to use content-stream deletion.
                remove_contained_graphics = any(
                    not (segment.get("metadata") or {}).get("text_layer_fallback")
                    for segment in page_segments
                    if _segment_needs_replacement(segment)
                )
                page.apply_redactions(
                    images=0,
                    graphics=1 if remove_contained_graphics else 0,
                    text=0,
                )
            # Insert transparent repair patches only after redaction. Otherwise
            # the redaction engine may also rewrite the newly inserted image,
            # exposing the light vector outlines underneath it.
            if repair_visual_segments and not clean_raster_backing:
                _repair_visual_text_regions(page, visual_segments)

        overlay_bytes = _build_pdf_layout_overlay(
            source,
            segments_by_page,
            target_language,
        )
        overlay = fitz.open(stream=overlay_bytes, filetype="pdf")
        try:
            for page_index, page in enumerate(source):
                page.show_pdf_page(
                    page.rect,
                    overlay,
                    page_index,
                    overlay=True,
                )
        finally:
            overlay.close()
        # Level 4 performs an all-object duplicate scan and becomes prohibitively
        # slow on CAD PDFs with tens of thousands of objects. Level 3 still
        # removes unreachable objects while preserving page content.
        return source.tobytes(garbage=3, deflate=True)
    finally:
        source.close()


def _build_pdf_layout_overlay(
    source,
    segments_by_page: Dict[int, List[Dict[str, Any]]],
    target_language: str,
) -> bytes:
    font_names = {
        (False, False): _register_reportlab_export_font(target_language),
        (True, False): _register_reportlab_export_font(
            target_language, bold=True
        ),
        (False, True): _register_reportlab_export_font(
            target_language, italic=True
        ),
        (True, True): _register_reportlab_export_font(
            target_language, bold=True, italic=True
        ),
    }
    stream = BytesIO()
    pdf = canvas.Canvas(stream, pageCompression=1)
    alignment_values = {
        "left": TA_LEFT,
        "center": TA_CENTER,
        "right": TA_RIGHT,
        "justify": TA_JUSTIFY,
    }

    for page_index, source_page in enumerate(source):
        page_width = float(source_page.rect.width)
        page_height = float(source_page.rect.height)
        pdf.setPageSize((page_width, page_height))
        page_segments = segments_by_page.get(page_index + 1, [])
        scanned_page = any(
            str((item.get("metadata") or {}).get("page_type", ""))
            in {"image", "mixed"}
            for item in page_segments
        )
        colored_ink_obstacles = (
            _scan_colored_ink_obstacles(source_page) if scanned_page else []
        )
        grid_lines = (
            _extract_pdf_grid_lines(source_page)
            if any(_is_visual_layout_segment(item) for item in page_segments)
            else ([], [])
        )
        page_scale = _find_pdf_page_scale(
            page_segments,
            source_page.rect,
            font_names,
            alignment_values,
            page_index,
        )
        draw_index = 0
        for segment_index, segment in enumerate(page_segments, start=1):
            translated_text = str(segment.get("translated_text", "")).strip()
            if not translated_text or not _segment_needs_replacement(segment):
                continue
            rect = _pdf_segment_rect(segment, source_page.rect)
            if rect.is_empty or rect.width < 2 or rect.height < 2:
                continue
            if segment.get("source_kind") == "ocr":
                # OCR replacement is allowed only on raster/scanned pages.
                # Never paint a white rectangle on a CAD/vector page.
                pass

            color = str(segment.get("color") or "#000000")
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                color = "#000000"
            draw_items = _pdf_segment_draw_items(
                segment,
                rect,
                source_page.rect,
                grid_lines,
            )
            for draw_segment, draw_rect, vertically_centered in draw_items:
                draw_index += 1
                segment_scale = page_scale
                if _is_visual_layout_segment(segment):
                    # OCR boxes are approximate and one tiny label must not force
                    # the whole CAD page's typography to shrink. Fit visual text
                    # independently inside its own box instead.
                    segment_scale = _fit_pdf_segment_scale(
                        draw_segment,
                        draw_rect,
                        1.0,
                        font_names,
                        alignment_values,
                        page_index,
                        draw_index,
                    )
                    if scanned_page:
                        segment_scale *= _scan_ink_scale(
                            draw_rect,
                            colored_ink_obstacles,
                            float(draw_segment.get("font_size") or 11.0),
                            segment_scale,
                        )
                paragraph, paragraph_height = _make_pdf_layout_paragraph(
                    draw_segment,
                    draw_rect,
                    segment_scale,
                    font_names,
                    alignment_values,
                    page_index,
                    draw_index,
                )
                draw_y = draw_rect.y0
                if vertically_centered:
                    draw_y += max(0.0, (draw_rect.height - paragraph_height) / 2)
                paragraph.drawOn(
                    pdf,
                    draw_rect.x0,
                    page_height - draw_y - paragraph_height,
                )
        pdf.showPage()

    pdf.save()
    return stream.getvalue()


def _scan_colored_ink_obstacles(page) -> List[fitz.Rect]:
    """Locate significant handwriting on a scan without treating color noise as ink."""

    if cv2 is None or np is None:
        return []
    scale = 2.0
    try:
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )[:, :, :3]
    except Exception:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    colored = np.where(
        (hsv[:, :, 0] >= 85)
        & (hsv[:, :, 0] <= 145)
        & (hsv[:, :, 1] >= 50)
        & (hsv[:, :, 2] <= 245),
        255,
        0,
    ).astype(np.uint8)
    colored = cv2.dilate(
        colored,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(3, int(round(scale * 4.0)) | 1), max(3, int(round(scale * 2.0)) | 1)),
        ),
        iterations=1,
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        colored,
        connectivity=8,
    )
    minimum_area = int(round(scale * scale * 60.0))
    obstacles = []
    for component_index in range(1, component_count):
        x, y, width, height, area = stats[component_index]
        if area < minimum_area:
            continue
        if width < scale * 8.0 or height < scale * 5.0:
            continue
        rect = fitz.Rect(
            x / scale,
            y / scale,
            (x + width) / scale,
            (y + height) / scale,
        )
        rect += (-1.0, -1.0, 1.0, 1.0)
        rect.intersect(page.rect)
        if not rect.is_empty:
            obstacles.append(rect)
    return obstacles


def _rect_overlap_area(rect: fitz.Rect, obstacles: List[fitz.Rect]) -> float:
    return sum(
        intersection.get_area()
        for obstacle in obstacles
        for intersection in [rect & obstacle]
        if not intersection.is_empty
    )


def _scan_ink_scale(
    rect: fitz.Rect,
    obstacles: List[fitz.Rect],
    font_size: float,
    fitted_scale: float,
) -> float:
    """Reduce scan text only when handwriting materially covers its box.

    OCR boxes often graze a signature by one or two points. Treating every
    non-zero intersection as a collision made otherwise readable labels tiny.
    Large intersections cannot be solved by shrinking alone, so preserve a
    minimum readable size instead of repeatedly sacrificing legibility.
    """
    rect_area = max(1.0, rect.get_area())
    overlap_ratio = _rect_overlap_area(rect, obstacles) / rect_area
    if overlap_ratio < 0.08:
        return 1.0
    if overlap_ratio < 0.20:
        collision_scale = 0.85
    elif overlap_ratio < 0.40:
        collision_scale = 0.75
    else:
        collision_scale = 0.65

    original_size = max(0.01, float(font_size))
    effective_size = original_size * max(0.01, float(fitted_scale))
    minimum_readable_scale = min(1.0, 6.0 / effective_size)
    return min(1.0, max(collision_scale, minimum_readable_scale))


def _extract_pdf_grid_lines(page):
    """Return substantial horizontal and vertical vector lines on a page."""
    horizontal = []
    vertical = []
    # CAD fonts are often converted to short vector strokes. Keep the initial
    # threshold conservative so glyph strokes never become synthetic rules;
    # the merge step below still joins genuine fragmented long borders.
    minimum_horizontal = max(12.0, float(page.rect.width) * 0.02)
    minimum_vertical = max(12.0, float(page.rect.height) * 0.02)

    def add_line(first, second):
        dx = abs(float(second.x) - float(first.x))
        dy = abs(float(second.y) - float(first.y))
        if dy <= 0.8 and dx >= minimum_horizontal:
            horizontal.append(
                (
                    (float(first.y) + float(second.y)) / 2,
                    min(float(first.x), float(second.x)),
                    max(float(first.x), float(second.x)),
                )
            )
        elif dx <= 0.8 and dy >= minimum_vertical:
            vertical.append(
                (
                    (float(first.x) + float(second.x)) / 2,
                    min(float(first.y), float(second.y)),
                    max(float(first.y), float(second.y)),
                )
            )

    try:
        drawings = page.get_drawings()
    except Exception:
        return horizontal, vertical
    for drawing in drawings:
        for item in drawing.get("items", []):
            if not item:
                continue
            if item[0] == "l":
                add_line(item[1], item[2])
            elif item[0] == "re":
                rect = fitz.Rect(item[1])
                add_line(rect.tl, rect.tr)
                add_line(rect.bl, rect.br)
                add_line(rect.tl, rect.bl)
                add_line(rect.tr, rect.br)
    if cv2 is not None and np is not None:
        raster_horizontal, raster_vertical = _extract_raster_grid_lines(page)
        horizontal.extend(raster_horizontal)
        vertical.extend(raster_vertical)
    return (
        _merge_grid_lines(horizontal, float(page.rect.width)),
        _merge_grid_lines(vertical, float(page.rect.height)),
    )


def _extract_raster_grid_lines(page):
    """Detect visible rules missing from a PDF's vector drawing inventory."""
    try:
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(1.0, 1.0),
            colorspace=fitz.csGRAY,
            alpha=False,
        )
        gray = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
        )
    except Exception:
        return [], []
    ink = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal_mask = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(35, int(pixmap.width * 0.018)), 1),
        ),
    )
    vertical_mask = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(35, int(pixmap.height * 0.018))),
        ),
    )

    def components(mask, horizontal_axis):
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        lines = []
        for label in range(1, count):
            x, y, width, height, _ = stats[label]
            if horizontal_axis:
                if width < max(45, pixmap.width * 0.022) or width < height * 12:
                    continue
                lines.append((y + height / 2, x, x + width))
            else:
                if height < max(45, pixmap.height * 0.022) or height < width * 12:
                    continue
                lines.append((x + width / 2, y, y + height))
        return lines

    return components(horizontal_mask, True), components(vertical_mask, False)


def _merge_grid_lines(lines, page_extent):
    """Merge fragmented collinear rules and reject isolated glyph strokes."""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda item: (item[0], item[1], item[2]))
    groups = []
    for position, start, end in ordered:
        matching = None
        for group in reversed(groups):
            group_position = group[0][0]
            if position - group_position > 1.2:
                break
            if start <= max(item[2] for item in group) + 4.0 and end >= min(
                item[1] for item in group
            ) - 4.0:
                matching = group
                break
        if matching is None:
            groups.append([(position, start, end)])
        else:
            matching.append((position, start, end))

    merged = []
    substantial_length = max(16.0, page_extent * 0.02)
    for group in groups:
        position = sum(item[0] for item in group) / len(group)
        start = min(item[1] for item in group)
        end = max(item[2] for item in group)
        length = end - start
        # A repeated short rule is a table/grid signal. One-off short rules
        # are overwhelmingly likely to be a letter or CAD glyph stroke.
        if length < substantial_length and len(group) < 2:
            continue
        merged.append((position, start, end))
    return merged


def _nearest_grid_bounds(rect, grid_lines):
    horizontal, vertical = grid_lines
    center_x = (rect.x0 + rect.x1) / 2
    center_y = (rect.y0 + rect.y1) / 2
    tolerance = 2.0
    horizontal_at_text = [
        (position, start, end)
        for position, start, end in horizontal
        if start - tolerance <= center_x <= end + tolerance
    ]
    vertical_at_text = [
        (position, start, end)
        for position, start, end in vertical
        if start - tolerance <= center_y <= end + tolerance
    ]
    # Some CAD exporters encode only the internal vertical dividers while the
    # outer border is implied by identical horizontal endpoints.
    inferred_vertical = []
    for _, start, end in horizontal_at_text:
        inferred_vertical.extend(
            ((start, rect.y0, rect.y1), (end, rect.y0, rect.y1))
        )
    vertical_at_text.extend(inferred_vertical)
    top = max(
        (position for position, _, _ in horizontal_at_text if position <= center_y),
        default=None,
    )
    bottom = min(
        (position for position, _, _ in horizontal_at_text if position >= center_y),
        default=None,
    )
    left = max(
        (position for position, _, _ in vertical_at_text if position <= center_x),
        default=None,
    )
    right = min(
        (position for position, _, _ in vertical_at_text if position >= center_x),
        default=None,
    )
    if None in {top, bottom, left, right} or bottom - top < 4 or right - left < 8:
        return None
    return fitz.Rect(left, top, right, bottom)


def _normalized_layout_value(value: Any) -> str:
    return re.sub(
        r"[^0-9A-Za-z\u3400-\u9fff\u0E00-\u0E7F]+",
        "",
        str(value or "").lower(),
    )


def _coalesce_visual_page_cells(page, segments: List[Dict[str, Any]]):
    """Collapse repeated visual blocks using detected page structure."""
    grid_lines = _extract_pdf_grid_lines(page)
    passthrough = []
    cell_groups: Dict[tuple, List[tuple]] = defaultdict(list)
    loose_visual: List[tuple] = []
    page_area = max(1.0, page.rect.get_area())

    for segment in segments:
        if not _is_visual_layout_segment(segment):
            passthrough.append(segment)
            continue
        rect = _pdf_segment_rect(segment, page.rect)
        if rect.is_empty:
            continue
        cell = _nearest_grid_bounds(rect, grid_lines)
        if (
            cell is not None
            and cell.get_area() <= page_area * 0.035
            and cell.height <= max(rect.height * 8.0, page.rect.height * 0.10)
        ):
            key = tuple(round(value, 1) for value in (cell.x0, cell.y0, cell.x1, cell.y1))
            cell_groups[key].append((segment, rect, cell))
        else:
            loose_visual.append((segment, rect))

    merged = list(passthrough)
    for values in cell_groups.values():
        values.sort(key=lambda item: (item[1].y0, item[1].x0))
        unique = []
        seen_translations = set()
        for segment, rect, _ in values:
            translation_key = _normalized_layout_value(segment.get("translated_text"))
            if translation_key and translation_key in seen_translations:
                continue
            if translation_key:
                seen_translations.add(translation_key)
            unique.append((segment, rect))
        if not unique:
            continue
        base = dict(unique[0][0])
        cell = values[0][2]
        translations = [
            str(segment.get("translated_text") or "").strip()
            for segment, _ in unique
            if str(segment.get("translated_text") or "").strip()
        ]
        sources = [
            str(segment.get("text") or "").strip()
            for segment, _ in unique
            if str(segment.get("text") or "").strip()
        ]
        base["translated_text"] = "\n".join(translations)
        base["text"] = "\n".join(sources)
        visual_union = unique[0][1]
        for _, rect in unique[1:]:
            visual_union |= rect
        full_clear_cell = cell.height <= max(
            42.0,
            max(rect.height for _, rect in unique) * 3.2,
        )
        padding_x = max(1.8, min(5.0, cell.width * 0.035))
        padding_y = max(1.2, min(3.5, cell.height * 0.10))
        if full_clear_cell:
            draw_box = fitz.Rect(
                cell.x0 + padding_x,
                cell.y0 + padding_y,
                cell.x1 - padding_x,
                cell.y1 - padding_y,
            )
        else:
            draw_box = fitz.Rect(visual_union)
            draw_box.intersect(
                fitz.Rect(
                    cell.x0 + padding_x,
                    cell.y0 + padding_y,
                    cell.x1 - padding_x,
                    cell.y1 - padding_y,
                )
            )
        base["bbox"] = list(draw_box)
        base["alignment"] = "center" if all(
            abs(((rect.x0 + rect.x1) / 2) - ((cell.x0 + cell.x1) / 2))
            <= cell.width * 0.18
            for _, rect in unique
        ) else "left"
        base["font_size"] = min(
            max(1.0, float(segment.get("font_size") or 11.0))
            for segment, _ in unique
        )
        metadata = dict(base.get("metadata") or {})
        metadata["table_cell_bbox"] = list(cell)
        metadata["visual_union_bbox"] = list(visual_union)
        metadata["table_cell_coalesced"] = len(values)
        metadata["table_cell_full_clear"] = full_clear_cell
        metadata["line_count"] = max(1, len(translations))
        base["metadata"] = metadata
        merged.append(base)

    # Crop overlap can duplicate labels outside tables too. Keep the more
    # accurately OCR-anchored instance when source/translation and geometry
    # agree, while preserving genuinely repeated labels at different places.
    kept_loose: List[tuple] = []
    for segment, rect in loose_visual:
        source_key = _normalized_layout_value(segment.get("text"))
        translation_key = _normalized_layout_value(segment.get("translated_text"))
        code_match = re.match(
            r"^([A-Za-z]{1,3}\d+[a-z]?)\b",
            str(segment.get("text") or "").strip(),
        )
        leading_code = code_match.group(1).lower() if code_match else ""
        duplicate_index = None
        for index, (candidate, candidate_rect) in enumerate(kept_loose):
            intersection = rect & candidate_rect
            overlap = (
                0.0
                if intersection.is_empty
                else intersection.get_area()
                / max(1.0, min(rect.get_area(), candidate_rect.get_area()))
            )
            same_value = (
                translation_key
                and translation_key
                == _normalized_layout_value(candidate.get("translated_text"))
            ) or (
                source_key
                and source_key == _normalized_layout_value(candidate.get("text"))
            )
            candidate_code_match = re.match(
                r"^([A-Za-z]{1,3}\d+[a-z]?)\b",
                str(candidate.get("text") or "").strip(),
            )
            same_nearby_code = bool(
                leading_code
                and candidate_code_match
                and leading_code == candidate_code_match.group(1).lower()
                and abs(
                    (rect.y0 + rect.y1)
                    - (candidate_rect.y0 + candidate_rect.y1)
                ) / 2 <= max(28.0, rect.height * 1.8, candidate_rect.height * 1.8)
                and max(0.0, min(rect.x1, candidate_rect.x1) - max(rect.x0, candidate_rect.x0))
                >= min(rect.width, candidate_rect.width) * 0.45
            )
            if (same_value and overlap >= 0.42) or same_nearby_code:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept_loose.append((segment, rect))
            continue
        candidate, _ = kept_loose[duplicate_index]
        candidate_anchored = bool((candidate.get("metadata") or {}).get("ocr_anchored"))
        segment_anchored = bool((segment.get("metadata") or {}).get("ocr_anchored"))
        if (
            segment_anchored
            and not candidate_anchored
            or segment_anchored == candidate_anchored
            and len(str(segment.get("text") or ""))
            > len(str(candidate.get("text") or ""))
        ):
            kept_loose[duplicate_index] = (segment, rect)
    merged.extend(segment for segment, _ in kept_loose)
    return merged


def _deduplicate_visual_segments(page, segments: List[Dict[str, Any]]):
    """Remove only crop-overlap duplicates without guessing table cells."""
    passthrough = [
        segment for segment in segments if not _is_visual_layout_segment(segment)
    ]
    kept: List[tuple] = []
    for segment in segments:
        if not _is_visual_layout_segment(segment):
            continue
        rect = _pdf_segment_rect(segment, page.rect)
        if rect.is_empty:
            continue
        source_key = _normalized_layout_value(segment.get("text"))
        translation_key = _normalized_layout_value(segment.get("translated_text"))
        duplicate_index = None
        for index, (candidate, candidate_rect) in enumerate(kept):
            if source_key != _normalized_layout_value(candidate.get("text")):
                continue
            if translation_key != _normalized_layout_value(
                candidate.get("translated_text")
            ):
                continue
            overlap = rect & candidate_rect
            if overlap.is_empty:
                continue
            coverage = overlap.get_area() / max(
                1.0, min(rect.get_area(), candidate_rect.get_area())
            )
            if coverage >= 0.45:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append((segment, rect))
            continue
        candidate, candidate_rect = kept[duplicate_index]
        candidate_anchored = bool((candidate.get("metadata") or {}).get("ocr_anchored"))
        segment_anchored = bool((segment.get("metadata") or {}).get("ocr_anchored"))
        if segment_anchored and not candidate_anchored:
            kept[duplicate_index] = (segment, rect)
        elif segment_anchored == candidate_anchored and rect.get_area() < candidate_rect.get_area():
            kept[duplicate_index] = (segment, rect)
    return passthrough + [segment for segment, _ in kept]


def _drawing_code_parts(text: str):
    match = re.match(
        r"^\s*([A-Za-z]{1,4}\d*\s*[-\u2013\u2014]\s*\d+"
        r"(?:\s*[-\u2013\u2014]\s*\d+)?)"
        r"(?:\s+|(?=[\u3400-\u9fff]))(.+?)\s*$",
        text,
    )
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _pdf_segment_draw_items(segment, rect, page_rect, grid_lines):
    """Anchor visual text to detected cells without document-specific coordinates."""
    if not _is_visual_layout_segment(segment):
        return [(segment, rect, False)]
    if str((segment.get("metadata") or {}).get("page_type", "")) in {
        "image",
        "mixed",
    }:
        # OCR refinement already provides a tight glyph box on scanned pages.
        # CAD cell snapping would move it onto neighboring rows and rules.
        return [(segment, rect, False)]

    # Keep narrow vertical labels in their own source column. Expanding them
    # to the surrounding table cell makes multiple translated headers collide.
    if rect.height >= max(24.0, rect.width * 1.6):
        vertical_segment = dict(segment)
        vertical_segment["alignment"] = "center"
        return [(vertical_segment, rect, True)]

    horizontal, vertical = grid_lines
    center_y = (rect.y0 + rect.y1) / 2
    internal_dividers = sorted(
        position
        for position, start, end in vertical
        if rect.x0 + 4 < position < rect.x1 - 4
        and start - 2 <= center_y <= end + 2
    )
    code_parts = _drawing_code_parts(
        str(segment.get("translated_text", "")).strip()
    )
    if internal_dividers and code_parts:
        divider = internal_dividers[0]
        left_lines = [
            position
            for position, start, end in vertical
            if position < divider - 2 and start - 2 <= center_y <= end + 2
        ]
        right_lines = [
            position
            for position, start, end in vertical
            if position > divider + 2 and start - 2 <= center_y <= end + 2
        ]
        row_lines = [
            (position, start, end)
            for position, start, end in horizontal
            if start - 2 <= divider <= end + 2
        ]
        top_lines = [item for item in row_lines if item[0] <= center_y]
        bottom_lines = [item for item in row_lines if item[0] >= center_y]
        inferred_left = [start for _, start, _ in row_lines if start < divider - 2]
        inferred_right = [end for _, _, end in row_lines if end > divider + 2]
        if not left_lines:
            left_lines = inferred_left
        if not right_lines:
            right_lines = inferred_right
        if left_lines and right_lines and top_lines and bottom_lines:
            top = max(item[0] for item in top_lines)
            bottom = min(item[0] for item in bottom_lines)
            if bottom - top >= 4:
                padding_x = max(
                    2.5,
                    min(8.0, float(segment.get("font_size") or 11.0) * 0.55),
                )
                padding_y = max(1.0, min(3.0, (bottom - top) * 0.10))
                code_segment = dict(segment)
                code_segment["translated_text"] = code_parts[0]
                code_segment["alignment"] = "center"
                body_segment = dict(segment)
                body_segment["translated_text"] = code_parts[1]
                body_segment["alignment"] = "left"
                return [
                    (
                        code_segment,
                        fitz.Rect(
                            max(left_lines) + padding_x,
                            top + padding_y,
                            divider - padding_x,
                            bottom - padding_y,
                        ),
                        True,
                    ),
                    (
                        body_segment,
                        fitz.Rect(
                            divider + padding_x,
                            top + padding_y,
                            min(right_lines) - padding_x,
                            bottom - padding_y,
                        ),
                        True,
                    ),
                ]

    cell = _nearest_grid_bounds(rect, grid_lines)
    if cell is None:
        return [(segment, rect, False)]
    # Reject large framing boxes. A text block belongs to a table cell only
    # when the detected row is reasonably close to its visible height.
    if cell.height > max(rect.height * 6.0, page_rect.height * 0.12):
        return [(segment, rect, False)]
    padding_x = max(
        1.5,
        min(4.0, float(segment.get("font_size") or 11.0) * 0.30),
    )
    padding_y = max(0.8, min(2.5, cell.height * 0.07))
    left_margin = max(0.0, rect.x0 - cell.x0)
    right_margin = max(0.0, cell.x1 - rect.x1)
    centered = (
        min(left_margin, right_margin) >= cell.width * 0.04
        and abs(left_margin - right_margin) <= cell.width * 0.18
    )
    anchored = dict(segment)
    if centered:
        anchored["alignment"] = "center"
    else:
        anchored["alignment"] = str(segment.get("alignment") or "left")
    # Keep the original visual footprint and only clamp it to the detected
    # cell. Expanding every label to a full table cell is the main cause of
    # cross-column text and inconsistent apparent font sizes.
    draw_rect = fitz.Rect(
        max(rect.x0, cell.x0 + padding_x),
        max(rect.y0, cell.y0 + padding_y),
        min(rect.x1, cell.x1 - padding_x),
        min(rect.y1, cell.y1 - padding_y),
    )
    if draw_rect.is_empty or draw_rect.width < 2 or draw_rect.height < 2:
        return [(segment, rect, False)]
    return [(anchored, draw_rect, True)]


def _is_visual_layout_segment(segment: Dict[str, Any]) -> bool:
    """Return true for text located by a visual/OCR pass.

    Older cached layout files incorrectly changed ``source_kind`` to ``text``
    during style normalization. The metadata flag is therefore intentionally
    treated as the authoritative signal as well.
    """
    metadata = segment.get("metadata") or {}
    return bool(
        metadata.get("visual_pass")
        or metadata.get("visual_residual")
        or metadata.get("ocr_audit")
        or metadata.get("tile")
        or segment.get("source_kind") in {"ocr", "ocr-audit", "visual"}
    )


def _segment_needs_replacement(segment: Dict[str, Any]) -> bool:
    source = re.sub(r"\s+", " ", str(segment.get("text", "")).strip())
    translated = re.sub(
        r"\s+", " ", str(segment.get("translated_text", "")).strip()
    )
    return bool(translated and translated != source)


def _needs_visual_raster_backing(page, segments: List[Dict[str, Any]]) -> bool:
    """Return true only for pages whose source text lives in raster pixels."""
    if not segments:
        return False
    return any(
        str((segment.get("metadata") or {}).get("page_type", ""))
        in {"image", "mixed"}
        for segment in segments
    )


def _insert_clean_visual_raster_backing(
    page,
    segments: List[Dict[str, Any]],
) -> bool:
    """Cover a scanned page with one clean, faithful raster backing.

    Local per-label patches produce visible seams when a sheet contains tens
    or hundreds of labels. Building one page-sized image avoids those seams.
    Only neutral text pixels inside detected text boxes are cleared; long
    raster and vector rules are restored from a protection mask.
    """
    if cv2 is None or np is None or Image is None or not segments:
        return False
    page_width = max(1.0, float(page.rect.width))
    source_scale = 0.0
    for image_info in page.get_images(full=True):
        try:
            xref = int(image_info[0])
            image_width = float(image_info[2])
            image_height = float(image_info[3])
            for image_rect in page.get_image_rects(xref):
                if image_rect.width > 0 and image_rect.height > 0:
                    source_scale = max(
                        source_scale,
                        image_width / image_rect.width,
                        image_height / image_rect.height,
                    )
        except Exception:
            continue
    # Preserve the scan's effective source resolution while bounding memory.
    scale = min(4.0, max(2.0, source_scale or (1800.0 / page_width)))
    try:
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )[:, :, :3].copy()
    except Exception:
        return False

    # Raster-only pages do not have authoritative PDF grid objects. Running
    # the generic raster grid detector here can classify text strokes as cell
    # borders and clip a cleanup box down to a few points.
    grid_lines = ([], [])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    white_page = float(np.mean(gray >= 225)) >= 0.78
    channel_spread = image.max(axis=2) - image.min(axis=2)
    neutral = np.where(channel_spread <= 42, 255, 0).astype(np.uint8)
    ink = cv2.bitwise_and(
        cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1],
        neutral,
    )
    horizontal_candidates = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(220, int(pixmap.width * 0.30)), max(1, int(round(scale * 0.6)))),
        ),
    )
    vertical_candidates = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(1, int(round(scale * 0.6))), max(120, int(pixmap.height * 0.06))),
        ),
    )
    # Raster continuity is a safer structural signal than CAD vector paths:
    # outlined fonts are themselves made from vector paths and otherwise get
    # mistaken for table rules, which restores the source text after cleanup.
    line_mask = cv2.bitwise_or(horizontal_candidates, vertical_candidates)
    # On an image-only page ``_extract_pdf_grid_lines`` is itself based on
    # raster morphology and can mistake Thai headline strokes for PDF rules.
    # The stricter full-resolution ``line_mask`` above is the sole source of
    # structural protection for scans.
    pdf_grid_mask = np.zeros_like(ink)

    box_mask = np.zeros((pixmap.height, pixmap.width), dtype=np.uint8)
    for segment in segments:
        rect = _pdf_segment_cleanup_rect(segment, page.rect)
        if rect.is_empty or rect.get_area() / max(1.0, page.rect.get_area()) > 0.08:
            continue
        metadata = segment.get("metadata") or {}
        visual = _is_visual_layout_segment(segment)
        if str(metadata.get("page_type", "")) in {"image", "mixed"}:
            # OCR polygons are tight to the recognized core and can miss the
            # first/last glyph by several points, especially around signatures.
            # The mask still targets neutral ink only, so a wider horizontal
            # margin does not erase blue handwriting.
            padding_x = max(3.0, min(10.0, rect.height * 0.60))
            padding_y = max(2.0, min(6.0, rect.height * 0.40))
        elif visual:
            padding_x = max(1.5, min(12.0, rect.height * 0.60))
            padding_y = max(1.0, min(7.0, rect.height * 0.38))
        else:
            padding_x = max(0.6, min(3.0, rect.height * 0.16))
            padding_y = max(0.5, min(2.5, rect.height * 0.12))
        expanded = rect + (-padding_x, -padding_y, padding_x, padding_y)
        cell = _nearest_grid_bounds(rect, grid_lines)
        if cell and cell.height <= max(rect.height * 5.0, 90.0):
            cell_inner = fitz.Rect(
                cell.x0 + 1.2,
                cell.y0 + 1.2,
                cell.x1 - 1.2,
                cell.y1 - 1.2,
            )
            expanded.intersect(cell_inner)
        expanded.intersect(page.rect)
        if expanded.is_empty:
            continue
        x0 = max(0, min(pixmap.width - 1, int(expanded.x0 * scale)))
        y0 = max(0, min(pixmap.height - 1, int(expanded.y0 * scale)))
        x1 = max(x0 + 1, min(pixmap.width, int(expanded.x1 * scale) + 1))
        y1 = max(y0 + 1, min(pixmap.height, int(expanded.y1 * scale) + 1))
        box_mask[y0:y1, x0:x1] = 255
        # Never clear the full OCR rectangle on scans. It can include a blue
        # signature, a handwritten date, or a dotted form rule. The glyph mask
        # below removes only neutral printed ink.

    glyph_mask = cv2.bitwise_and(ink, box_mask)
    glyph_mask = cv2.bitwise_and(glyph_mask, cv2.bitwise_not(line_mask))
    if not np.any(glyph_mask) and not np.any(box_mask):
        return False
    dilation = max(5, int(round(scale * 3.8)))
    if dilation % 2 == 0:
        dilation += 1
    glyph_mask = cv2.dilate(
        glyph_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation)),
        iterations=1,
    )
    mask = (
        cv2.bitwise_and(box_mask, cv2.bitwise_not(line_mask))
        if white_page
        else glyph_mask
    )

    if white_page:
        # Inpainting a dark glyph from its immediate neighborhood frequently
        # reconstructs the same stroke. On white scanned paper, confirmed text
        # boxes should be restored directly to paper white. Restore colored
        # pixels afterwards so blue signatures and handwritten dates survive.
        repaired = image.copy()
        repaired[mask > 0] = 255
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        colored_ink = np.where(
            (hsv[:, :, 0] >= 90)
            & (hsv[:, :, 0] <= 140)
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] <= 245),
            255,
            0,
        ).astype(np.uint8)
        # Scanners often add a few blue fringe pixels around otherwise black
        # printed glyphs. Restoring every saturated pixel brings those glyphs
        # back after cleanup. Handwritten blue ink forms much larger connected
        # strokes, so discard only tiny color components before restoring it.
        component_count, component_labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(colored_ink, connectivity=8)
        )
        minimum_colored_area = max(18, int(round(scale * scale * 5.0)))
        significant_colored_ink = np.zeros_like(colored_ink)
        for component_index in range(1, component_count):
            if (
                int(component_stats[component_index, cv2.CC_STAT_AREA])
                >= minimum_colored_area
            ):
                significant_colored_ink[
                    component_labels == component_index
                ] = 255
        colored_ink = significant_colored_ink
        colored_ink = cv2.dilate(
            colored_ink,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        blue_edge_support = np.where(
            (hsv[:, :, 0] >= 85)
            & (hsv[:, :, 0] <= 145)
            & (hsv[:, :, 1] >= 20)
            & (hsv[:, :, 2] <= 250),
            255,
            0,
        ).astype(np.uint8)
        colored_ink = cv2.bitwise_and(colored_ink, blue_edge_support)
        colored_inside_boxes = cv2.bitwise_and(box_mask, colored_ink)
        repaired[colored_inside_boxes > 0] = image[colored_inside_boxes > 0]
    else:
        repaired = cv2.inpaint(image, mask, 2.0, cv2.INPAINT_TELEA)
    restore_mask = cv2.bitwise_or(pdf_grid_mask, line_mask)
    repaired[restore_mask > 0] = image[restore_mask > 0]

    try:
        stream = BytesIO()
        Image.fromarray(repaired, mode="RGB").save(
            stream,
            format="PNG",
            optimize=True,
        )
        page.insert_image(page.rect, stream=stream.getvalue(), overlay=True)
    except Exception:
        return False
    return True


def _repair_visual_text_regions(page, segments: List[Dict[str, Any]]) -> None:
    """Repair source pixels inside visual text boxes without white rectangles.

    CAD labels are frequently outlined vector paths rather than PDF text
    objects, so a normal PDF text redaction cannot remove them. Rendering one
    page locally and inpainting only the detected boxes lets us place a
    transparent patch back over the original page while retaining all artwork
    outside those boxes. This is deliberately best-effort and is skipped when
    the optional image dependencies are unavailable.
    """
    if cv2 is None or np is None or Image is None or not segments:
        return

    page_width = max(1.0, float(page.rect.width))
    # Keep patches sharp enough for small CAD labels without rasterising the
    # whole page. Clustering nearby boxes bounds memory on dense title blocks.
    scale = min(2.0, max(1.25, 1400.0 / page_width))
    grid_lines = _extract_pdf_grid_lines(page)
    visual_rects = []
    for segment in segments:
        rect = _pdf_segment_cleanup_rect(segment, page.rect)
        if rect.is_empty or rect.get_area() <= 0:
            continue
        if rect.get_area() / max(1.0, page.rect.get_area()) > 0.20:
            continue
        # Native CAD text boxes are usually more accurate than OCR boxes. A
        # tight margin prevents the repair from touching nearby dimensions or
        # table borders.
        metadata = segment.get("metadata") or {}
        cell = _nearest_grid_bounds(rect, grid_lines)
        if segment.get("source_kind") in {"ocr", "ocr-audit"}:
            if rect.height >= max(24.0, rect.width * 1.6):
                padding_x = max(2.0, min(8.0, rect.width * 0.35))
                padding_y = max(4.0, min(24.0, rect.width * 1.25))
            else:
                padding_x = max(5.0, min(24.0, rect.height * 1.20))
                padding_y = max(3.0, min(10.0, rect.height * 0.45))
        elif metadata.get("visual_residual") or metadata.get("visual_pass"):
            padding_x = max(4.0, min(20.0, rect.height))
            padding_y = max(3.0, min(9.0, rect.height * 0.40))
        else:
            padding_x, padding_y = 0.8, 0.8
        expanded = rect + (-padding_x, -padding_y, padding_x, padding_y)
        if cell and cell.height <= max(rect.height * 6.0, 80.0):
            cell_inner = fitz.Rect(
                cell.x0 + 1.5,
                cell.y0 + 1.5,
                cell.x1 - 1.5,
                cell.y1 - 1.5,
            )
            expanded.intersect(cell_inner)
        expanded.intersect(page.rect)
        if not expanded.is_empty:
            visual_rects.append(expanded)
    if not visual_rects:
        return
    clusters: List[List[Any]] = []
    for rect in visual_rects:
        expanded = rect + (-18, -18, 18, 18)
        merged = []
        for index, cluster in enumerate(clusters):
            bounds = cluster[0]
            for item in cluster[1:]:
                bounds |= item
            if (bounds & expanded).is_empty:
                continue
            merged.append(index)
        if not merged:
            clusters.append([rect])
            continue
        first = merged[0]
        clusters[first].append(rect)
        for index in reversed(merged[1:]):
            clusters[first].extend(clusters.pop(index))

    for cluster in clusters:
        _repair_visual_cluster(page, cluster, scale, grid_lines)


def _repair_visual_cluster(
    page,
    visual_rects: List[Any],
    scale: float,
    grid_lines,
) -> None:
    repair_rect = visual_rects[0]
    for rect in visual_rects[1:]:
        repair_rect |= rect
    # A pathological OCR box must never trigger a near-full-page raster patch.
    if repair_rect.get_area() / max(1.0, page.rect.get_area()) > 0.30:
        return
    try:
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            clip=repair_rect,
            colorspace=fitz.csRGB,
            alpha=False,
        )
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )[:, :, :3].copy()
    except Exception:
        return


    box_mask = np.zeros((pixmap.height, pixmap.width), dtype=np.uint8)
    for rect in visual_rects:
        x0 = max(0, min(pixmap.width - 1, int((rect.x0 - repair_rect.x0) * scale)))
        y0 = max(0, min(pixmap.height - 1, int((rect.y0 - repair_rect.y0) * scale)))
        x1 = max(x0 + 1, min(pixmap.width, int((rect.x1 - repair_rect.x0) * scale) + 1))
        y1 = max(y0 + 1, min(pixmap.height, int((rect.y1 - repair_rect.y0) * scale) + 1))
        box_mask[y0:y1, x0:x1] = 255
    if not np.any(box_mask):
        return
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    channel_spread = image.max(axis=2) - image.min(axis=2)
    neutral = np.where(channel_spread <= 34, 255, 0).astype(np.uint8)
    ink = cv2.bitwise_and(cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)[1], neutral)
    horizontal_candidates = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, int(18 * scale)), 1)),
    )
    vertical_candidates = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, int(18 * scale)))),
    )
    horizontal = _long_axis_components(horizontal_candidates, horizontal_axis=True)
    vertical = _long_axis_components(vertical_candidates, horizontal_axis=False)
    line_mask = cv2.bitwise_or(horizontal, vertical)
    line_mask = cv2.bitwise_or(
        line_mask,
        _vector_grid_line_mask(
            grid_lines,
            repair_rect,
            scale,
            pixmap.width,
            pixmap.height,
        ),
    )
    candidates = cv2.bitwise_and(ink, box_mask)
    candidates = cv2.bitwise_and(candidates, cv2.bitwise_not(line_mask))
    # The outlined glyphs produced by CAD exporters can join into one large
    # word component. Filtering by component area leaves complete pale word
    # contours behind, so remove every detected neutral-ink pixel inside the
    # text boxes and rely on the structural-line mask for preservation.
    mask = candidates
    if not np.any(mask):
        return
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(line_mask))
    border_pixels = np.concatenate(
        (image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]),
        axis=0,
    )
    background = np.median(border_pixels, axis=0)
    if np.all(background >= 245):
        # Most architectural sheets use white cell backgrounds. Repainting
        # only the detected glyph pixels avoids any rectangular patch edge.
        repaired = image.copy()
        repaired[mask > 0] = background.astype(np.uint8)
    else:
        repaired = cv2.inpaint(image, mask, 1.2, cv2.INPAINT_TELEA)
    repaired[line_mask > 0] = image[line_mask > 0]
    # Use an opaque local source patch. The patch contains the original
    # background and protected linework, while only glyph pixels were changed;
    # alpha shaped like the glyph mask creates visible halo/box contours after
    # PDF rasterisation.
    rgba = np.dstack((repaired, np.full_like(mask, 255)))
    try:
        stream = BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(stream, format="PNG", optimize=True)
        page.insert_image(repair_rect, stream=stream.getvalue(), overlay=True)
    except Exception:
        return


def _long_axis_components(mask, *, horizontal_axis: bool):
    """Keep raster lines that span a meaningful part of a repair patch."""
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    result = np.zeros_like(mask)
    patch_height, patch_width = mask.shape
    for label in range(1, component_count):
        _, _, width, height, _ = stats[label]
        if horizontal_axis:
            structural = width >= max(30, patch_width * 0.35) and width >= height * 6
        else:
            structural = height >= max(30, patch_height * 0.35) and height >= width * 6
        if structural:
            result[labels == label] = 255
    return result


def _vector_grid_line_mask(
    grid_lines,
    repair_rect,
    scale: float,
    width: int,
    height: int,
):
    """Rasterize detected PDF grid lines into a local protection mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    horizontal, vertical = grid_lines
    thickness = max(2, int(round(scale * 1.8)))
    for position, start, end in horizontal:
        if position < repair_rect.y0 - 1 or position > repair_rect.y1 + 1:
            continue
        clipped_start = max(start, repair_rect.x0)
        clipped_end = min(end, repair_rect.x1)
        if clipped_end <= clipped_start:
            continue
        y = int(round((position - repair_rect.y0) * scale))
        x0 = int(round((clipped_start - repair_rect.x0) * scale))
        x1 = int(round((clipped_end - repair_rect.x0) * scale))
        cv2.line(mask, (x0, y), (x1, y), 255, thickness)
    for position, start, end in vertical:
        if position < repair_rect.x0 - 1 or position > repair_rect.x1 + 1:
            continue
        clipped_start = max(start, repair_rect.y0)
        clipped_end = min(end, repair_rect.y1)
        if clipped_end <= clipped_start:
            continue
        x = int(round((position - repair_rect.x0) * scale))
        y0 = int(round((clipped_start - repair_rect.y0) * scale))
        y1 = int(round((clipped_end - repair_rect.y0) * scale))
        cv2.line(mask, (x, y0), (x, y1), 255, thickness)
    return mask


def _find_pdf_page_scale(
    page_segments: List[Dict[str, Any]],
    page_rect,
    font_names: Dict[tuple, str],
    alignment_values: Dict[str, int],
    page_index: int,
) -> float:
    """Find one scale for the complete page, preserving consistent typography."""
    visible_segments = []
    for segment in page_segments:
        if _is_visual_layout_segment(segment):
            continue
        if not str(segment.get("translated_text", "")).strip():
            continue
        rect = _pdf_segment_rect(segment, page_rect)
        if rect.is_empty or rect.width < 5 or rect.height < 4:
            continue
        visible_segments.append((segment, rect))
    if not visible_segments:
        return 1.0

    def fits(scale: float) -> bool:
        for segment_index, (segment, rect) in enumerate(visible_segments, start=1):
            _, paragraph_height = _make_pdf_layout_paragraph(
                segment,
                rect,
                scale,
                font_names,
                alignment_values,
                page_index,
                segment_index,
            )
            if paragraph_height > rect.height + 0.1:
                return False
        return True

    if fits(1.0):
        return 1.0

    lower = 0.01
    while lower > 0.0001 and not fits(lower):
        lower /= 2
    if not fits(lower):
        raise ValueError(f"第 {page_index + 1} 页译文无法排入原页面")

    upper = 1.0
    for _ in range(12):
        candidate = (lower + upper) / 2
        if fits(candidate):
            lower = candidate
        else:
            upper = candidate
    return max(0.0001, lower * 0.995)


def _fit_pdf_segment_scale(
    segment: Dict[str, Any],
    rect,
    upper: float,
    font_names: Dict[tuple, str],
    alignment_values: Dict[str, int],
    page_index: int,
    segment_index: int,
) -> float:
    upper = max(0.05, min(1.0, upper))
    metadata = segment.get("metadata") or {}
    translated_text = str(segment.get("translated_text", "")).strip()
    keep_single_line = bool(
        str(metadata.get("page_type", "")) in {"image", "mixed"}
        and metadata.get("translation_unit") == "complete-line"
        and int(metadata.get("rotation") or 0) % 360 == 0
        and "\n" not in translated_text
    )
    original_size = max(0.01, float(segment.get("font_size") or 11.0))
    font_name = font_names[
        (bool(metadata.get("bold")), bool(metadata.get("italic")))
    ]
    single_line_width = (
        pdfmetrics.stringWidth(translated_text, font_name, original_size)
        if keep_single_line
        else 0.0
    )

    def fits(scale: float) -> bool:
        _, height = _make_pdf_layout_paragraph(
            segment,
            rect,
            scale,
            font_names,
            alignment_values,
            page_index,
            segment_index,
        )
        return bool(
            height <= rect.height + 0.1
            and (
                not keep_single_line
                or single_line_width * scale <= rect.width * 0.98
            )
        )

    if fits(upper):
        return upper
    lower = 0.05
    if not fits(lower):
        return lower
    for _ in range(10):
        candidate = (lower + upper) / 2
        if fits(candidate):
            lower = candidate
        else:
            upper = candidate
    return max(0.05, lower * 0.995)


def _make_pdf_layout_paragraph(
    segment: Dict[str, Any],
    rect,
    scale: float,
    font_names: Dict[tuple, str],
    alignment_values: Dict[str, int],
    page_index: int,
    segment_index: int,
):
    translated_text = str(segment.get("translated_text", "")).strip()
    color = str(segment.get("color") or "#000000")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        color = "#000000"
    original_size = max(0.01, float(segment.get("font_size") or 11.0))
    metadata = segment.get("metadata") or {}
    original_leading = max(
        original_size,
        float(metadata.get("leading") or original_size * 1.15),
    )
    font_size = max(0.01, original_size * scale)
    leading = max(font_size, original_leading * scale)
    font_name = font_names[
        (bool(metadata.get("bold")), bool(metadata.get("italic")))
    ]
    style = ParagraphStyle(
        name=f"MetaTransLayout{page_index}_{segment_index}_{scale:.5f}",
        fontName=font_name,
        fontSize=font_size,
        leading=leading,
        textColor=HexColor(color),
        alignment=alignment_values.get(
            str(segment.get("alignment", "left")), TA_LEFT
        ),
        wordWrap="CJK",
        splitLongWords=1,
        spaceBefore=0,
        spaceAfter=0,
    )
    markup = html.escape(translated_text).replace("\n", "<br/>")
    paragraph = Paragraph(markup, style)
    _, paragraph_height = paragraph.wrap(max(2.0, rect.width), max(2.0, rect.height))
    return paragraph, paragraph_height


def _pdf_segment_rect(segment: Dict[str, Any], page_rect) -> fitz.Rect:
    bbox = segment.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return fitz.Rect()
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return fitz.Rect()
    if (segment.get("metadata") or {}).get("normalized_bbox"):
        values = [
            values[0] / 1000 * page_rect.width,
            values[1] / 1000 * page_rect.height,
            values[2] / 1000 * page_rect.width,
            values[3] / 1000 * page_rect.height,
        ]
    rect = fitz.Rect(values)
    rect.intersect(page_rect)
    return rect


def _pdf_segment_cleanup_rect(segment: Dict[str, Any], page_rect) -> fitz.Rect:
    """Return the source-space box used to remove pixels/text before drawing."""
    metadata = segment.get("metadata") or {}
    visual_union_bbox = metadata.get("visual_union_bbox")
    if isinstance(visual_union_bbox, (list, tuple)) and len(visual_union_bbox) == 4:
        source_segment = dict(segment)
        source_segment["bbox"] = list(visual_union_bbox)
        source_metadata = dict(metadata)
        source_metadata.pop("normalized_bbox", None)
        source_segment["metadata"] = source_metadata
        return _pdf_segment_rect(source_segment, page_rect)
    cleanup_bbox = metadata.get("cleanup_bbox")
    if isinstance(cleanup_bbox, (list, tuple)) and len(cleanup_bbox) == 4:
        source_segment = dict(segment)
        source_segment["bbox"] = list(cleanup_bbox)
        source_metadata = dict(metadata)
        source_metadata.pop("normalized_bbox", None)
        source_segment["metadata"] = source_metadata
        return _pdf_segment_rect(source_segment, page_rect)
    return _pdf_segment_rect(segment, page_rect)


def build_layout_docx_export(
    source_content: bytes,
    layout_segments: List[Dict[str, Any]],
) -> bytes:
    translations_by_part = _translations_by_part(layout_segments, "docx")

    def transform(part_name: str, content: bytes) -> bytes:
        translations = translations_by_part.get(part_name)
        if not translations:
            return content
        root = etree.fromstring(content)
        paragraphs = root.iter(f"{{{WORD_NAMESPACE}}}p")
        for paragraph_index, paragraph in enumerate(paragraphs):
            translated_text = translations.get(paragraph_index)
            if translated_text is None:
                continue
            _replace_ooxml_paragraph_text(
                paragraph,
                f"{{{WORD_NAMESPACE}}}t",
                translated_text,
                f"{{{WORD_NAMESPACE}}}sz",
                f"{{{WORD_NAMESPACE}}}val",
                units_per_point=2,
            )
        return etree.tostring(
            root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        )

    return _rewrite_zip(source_content, transform)


def build_layout_pptx_export(
    source_content: bytes,
    layout_segments: List[Dict[str, Any]],
) -> bytes:
    translations_by_part = _translations_by_part(layout_segments, "pptx")

    def transform(part_name: str, content: bytes) -> bytes:
        translations = translations_by_part.get(part_name)
        if not translations:
            return content
        root = etree.fromstring(content)
        paragraphs = root.iter(f"{{{DRAWING_NAMESPACE}}}p")
        for paragraph_index, paragraph in enumerate(paragraphs):
            translated_text = translations.get(paragraph_index)
            if translated_text is None:
                continue
            expanded = _replace_ooxml_paragraph_text(
                paragraph,
                f"{{{DRAWING_NAMESPACE}}}t",
                translated_text,
                f"{{{DRAWING_NAMESPACE}}}rPr",
                "sz",
                units_per_point=100,
            )
            if expanded:
                _enable_powerpoint_autofit(paragraph)
        return etree.tostring(
            root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        )

    return _rewrite_zip(source_content, transform)


def _replace_ooxml_paragraph_text(
    paragraph,
    text_tag: str,
    translated_text: str,
    size_tag: str,
    size_attribute: str,
    *,
    units_per_point: int,
) -> bool:
    text_nodes = list(paragraph.iter(text_tag))
    if not text_nodes:
        return False
    source_text = "".join(node.text or "" for node in text_nodes)
    text_nodes[0].text = translated_text
    for node in text_nodes[1:]:
        node.text = ""

    source_length = max(1, len(source_text.strip()))
    target_length = len(translated_text.strip())
    if target_length <= source_length * 1.08:
        return False
    scale = max(0.65, min(1.0, (source_length / target_length) ** 0.5))
    for size_node in paragraph.iter(size_tag):
        raw_value = size_node.get(size_attribute)
        if not raw_value:
            continue
        try:
            current_size = int(raw_value)
        except ValueError:
            continue
        minimum_size = int(5 * units_per_point)
        size_node.set(size_attribute, str(max(minimum_size, round(current_size * scale))))
    return True


def _enable_powerpoint_autofit(paragraph) -> None:
    current = paragraph
    while current is not None and current.tag != f"{{{PRESENTATION_NAMESPACE}}}txBody":
        current = current.getparent()
    if current is None:
        return
    body_properties = current.find(f"{{{DRAWING_NAMESPACE}}}bodyPr")
    if body_properties is None:
        return
    for child in list(body_properties):
        if child.tag in {
            f"{{{DRAWING_NAMESPACE}}}noAutofit",
            f"{{{DRAWING_NAMESPACE}}}spAutoFit",
            f"{{{DRAWING_NAMESPACE}}}normAutofit",
        }:
            body_properties.remove(child)
    etree.SubElement(body_properties, f"{{{DRAWING_NAMESPACE}}}normAutofit")


def _translations_by_part(
    layout_segments: List[Dict[str, Any]],
    prefix: str,
) -> Dict[str, Dict[int, str]]:
    result: Dict[str, Dict[int, str]] = defaultdict(dict)
    for segment in layout_segments:
        if not str(segment.get("segment_id", "")).startswith(f"{prefix}:"):
            continue
        part_name = segment.get("part_name")
        paragraph_index = segment.get("paragraph_index")
        translated_text = segment.get("translated_text")
        if (
            not isinstance(part_name, str)
            or not isinstance(paragraph_index, int)
            or not isinstance(translated_text, str)
        ):
            continue
        result[part_name][paragraph_index] = translated_text
    return result


def _rewrite_zip(source_content: bytes, transform) -> bytes:
    output = BytesIO()
    with ZipFile(BytesIO(source_content)) as source, ZipFile(
        output, "w", compression=ZIP_DEFLATED
    ) as destination:
        for item in source.infolist():
            content = source.read(item.filename)
            destination.writestr(item, transform(item.filename, content))
    return output.getvalue()


def _segments_by_page(
    layout_segments: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    result: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for segment in layout_segments:
        page_number = segment.get("page_number")
        if isinstance(page_number, int) and page_number > 0:
            result[page_number].append(segment)
    return result


def _coalesce_visual_bilingual_segments(
    layout_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge adjacent bilingual labels that translate to the same text.

    Engineering schedules commonly print Thai and English in the same cell.
    Vision returns both lines independently; drawing both Chinese translations
    creates duplicate, overlapping labels. The union box still clears both
    source lines and draws one translation in their shared position.
    """
    output: List[Dict[str, Any]] = []
    consumed = set()
    for index, segment in enumerate(layout_segments):
        if index in consumed or not _is_visual_layout_segment(segment):
            if index not in consumed:
                output.append(segment)
            continue
        translated = re.sub(
            r"[\s\u3000，。,:：;；()（）\[\]]+",
            "",
            str(segment.get("translated_text", "")).lower(),
        )
        rect = _raw_segment_rect(segment)
        if not translated or rect.is_empty:
            output.append(segment)
            continue
        merged = dict(segment)
        merged_rect = rect
        source_parts = [str(segment.get("text", "")).strip()]
        for candidate_index in range(index + 1, len(layout_segments)):
            if candidate_index in consumed:
                continue
            candidate = layout_segments[candidate_index]
            if (
                candidate.get("page_number") != segment.get("page_number")
                or not _is_visual_layout_segment(candidate)
            ):
                continue
            candidate_translation = re.sub(
                r"[\s\u3000，。,:：;；()（）\[\]]+",
                "",
                str(candidate.get("translated_text", "")).lower(),
            )
            if candidate_translation != translated:
                continue
            candidate_rect = _raw_segment_rect(candidate)
            if candidate_rect.is_empty:
                continue
            horizontal_overlap = max(
                0.0,
                min(merged_rect.x1, candidate_rect.x1)
                - max(merged_rect.x0, candidate_rect.x0),
            )
            vertical_gap = max(
                0.0,
                candidate_rect.y0 - merged_rect.y1,
                merged_rect.y0 - candidate_rect.y1,
            )
            if (
                horizontal_overlap
                < min(merged_rect.width, candidate_rect.width) * 0.35
                or vertical_gap > max(18.0, min(merged_rect.height, candidate_rect.height) * 1.5)
            ):
                continue
            merged_rect |= candidate_rect
            source_parts.append(str(candidate.get("text", "")).strip())
            consumed.add(candidate_index)
        if len(source_parts) > 1:
            merged["bbox"] = list(merged_rect)
            merged["text"] = "\n".join(part for part in source_parts if part)
            metadata = dict(merged.get("metadata") or {})
            metadata["bilingual_coalesced"] = True
            merged["metadata"] = metadata
        output.append(merged)
    return output


def _raw_segment_rect(segment: Dict[str, Any]) -> fitz.Rect:
    bbox = segment.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return fitz.Rect()
    try:
        return fitz.Rect(*(float(value) for value in bbox))
    except (TypeError, ValueError):
        return fitz.Rect()


def build_pdf_export(
    source_content: bytes, translated_text: str, target_language: str
) -> bytes:
    try:
        source = fitz.open(stream=source_content, filetype="pdf")
    except Exception as exc:
        raise ValueError("源 PDF 无法用于生成译文文件") from exc

    page_translations = split_page_translations(translated_text, source.page_count)
    if target_language == "zh":
        try:
            return _build_compact_chinese_pdf(source, page_translations)
        finally:
            source.close()

    output = fitz.open()
    font_path = _find_export_font(target_language)
    archive = fitz.Archive(str(font_path.parent)) if font_path else None
    font_css = (
        f"@font-face {{ font-family: ExportFont; src: url('{font_path.name}'); }}"
        if font_path
        else ""
    )
    css = (
        font_css
        + """
        body { font-family: ExportFont, sans-serif; font-size: 11pt;
               line-height: 1.48; color: #16231f; }
        .page-number { color: #65716c; font-size: 8.5pt; margin-bottom: 10pt; }
        .translation { white-space: pre-wrap; }
        """
    )
    try:
        for page_index, source_page in enumerate(source):
            page_number = page_index + 1
            rect = source_page.rect
            output_page = output.new_page(width=rect.width, height=rect.height)
            margin_x = max(32, min(52, rect.width * 0.075))
            margin_y = max(30, min(48, rect.height * 0.055))
            text_rect = fitz.Rect(
                margin_x,
                margin_y,
                rect.width - margin_x,
                rect.height - margin_y,
            )
            page_text = page_translations.get(page_number, "")
            markup = (
                f'<div class="page-number">{page_number} / {source.page_count}</div>'
                f'<div class="translation">{html.escape(page_text)}</div>'
            )
            spare_height, scale = output_page.insert_htmlbox(
                text_rect,
                markup,
                css=css,
                archive=archive,
                scale_low=0,
            )
            if spare_height < 0 or scale <= 0:
                raise ValueError(f"第 {page_number} 页译文无法排入对应页面")
        exported = output.tobytes(garbage=4, deflate=True)
        _require_pdf_page_count(exported, source.page_count)
        return exported
    finally:
        output.close()
        source.close()


def _build_compact_chinese_pdf(
    source,
    page_translations: Dict[int, str],
) -> bytes:
    font_name = _register_reportlab_chinese_font()
    stream = BytesIO()
    pdf = canvas.Canvas(stream, pageCompression=1)
    page_count = source.page_count

    for page_index, source_page in enumerate(source):
        page_number = page_index + 1
        rect = source_page.rect
        page_width = float(rect.width)
        page_height = float(rect.height)
        margin_x = max(32, min(52, page_width * 0.075))
        margin_y = max(30, min(48, page_height * 0.055))
        label_height = 24
        page_text = page_translations.get(page_number, "")
        available_width = page_width - margin_x * 2
        available_height = page_height - margin_y * 2 - label_height
        placements = _layout_unformatted_chinese_page(
            page_text,
            font_name,
            available_width,
            available_height,
        )
        if placements is None and page_text:
            raise ValueError(
                f"LAYOUT_OVERFLOW: 第 {page_number} 页未排版译文无法放入原页面"
            )

        _draw_unformatted_pdf_page_label(
            pdf,
            page_width,
            page_height,
            margin_x,
            margin_y,
            page_number,
            page_count,
            False,
        )
        if placements:
            column_count = max(item[0] for item in placements) + 1
            gutter = 18.0 if column_count > 1 else 0.0
            column_width = (
                available_width - gutter * (column_count - 1)
            ) / column_count
            column_tops = [
                page_height - margin_y - label_height
                for _ in range(column_count)
            ]
            for column_index, paragraph, paragraph_height in placements:
                column_tops[column_index] -= paragraph_height
                paragraph.drawOn(
                    pdf,
                    margin_x + column_index * (column_width + gutter),
                    column_tops[column_index],
                )
        pdf.showPage()

    pdf.save()
    exported = stream.getvalue()
    _require_pdf_page_count(exported, page_count)
    return exported


def _layout_unformatted_chinese_page(
    page_text: str,
    font_name: str,
    available_width: float,
    available_height: float,
):
    if not page_text:
        return []

    # Wide engineering sheets have enough horizontal room to keep dense
    # page-mapped text readable in columns. Narrow pages remain single-column.
    max_columns = max(1, min(5, int((available_width + 18.0) // 340.0)))
    lines = page_text.splitlines() or [page_text]
    for step in range(15):
        font_size = 11 - step * 0.5
        style = _unformatted_chinese_style(font_name, font_size, step)
        for column_count in range(1, max_columns + 1):
            gutter = 18.0 if column_count > 1 else 0.0
            column_width = (
                available_width - gutter * (column_count - 1)
            ) / column_count
            placements = []
            column_index = 0
            used_height = 0.0
            fits = True
            for line in lines:
                markup = html.escape(line) if line else "&#160;"
                paragraph = Paragraph(markup, style)
                _, paragraph_height = paragraph.wrap(
                    column_width,
                    available_height,
                )
                if paragraph_height > available_height:
                    fits = False
                    break
                if used_height + paragraph_height > available_height + 0.01:
                    column_index += 1
                    used_height = 0.0
                if column_index >= column_count:
                    fits = False
                    break
                placements.append((column_index, paragraph, paragraph_height))
                used_height += paragraph_height
            if fits:
                return placements
    return None


def _require_pdf_page_count(content: bytes, expected: int) -> None:
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("PDF 译文生成失败") from exc
    try:
        if document.page_count != expected:
            raise ValueError(
                f"ID_MISMATCH: 输出页数 {document.page_count} 与源文件 {expected} 不一致"
            )
    finally:
        document.close()


def _unformatted_chinese_style(
    font_name: str,
    font_size: float,
    suffix,
) -> ParagraphStyle:
    return ParagraphStyle(
        name=f"MetaTransChinese{suffix}",
        fontName=font_name,
        fontSize=font_size,
        leading=font_size * 1.48,
        textColor=HexColor("#16231f"),
        wordWrap="CJK",
        splitLongWords=1,
    )


def _draw_unformatted_pdf_page_label(
    pdf,
    page_width: float,
    page_height: float,
    margin_x: float,
    margin_y: float,
    source_page_number: int,
    source_page_count: int,
    continuation: bool,
) -> None:
    pdf.setPageSize((page_width, page_height))
    pdf.setFillColor(HexColor("#65716c"))
    pdf.setFont("Helvetica", 8.5)
    suffix = " cont." if continuation else ""
    pdf.drawString(
        margin_x,
        page_height - margin_y - 8.5,
        f"{source_page_number} / {source_page_count}{suffix}",
    )


def _register_reportlab_chinese_font() -> str:
    try:
        pdfmetrics.getFont(REPORTLAB_CHINESE_FONT)
        return REPORTLAB_CHINESE_FONT
    except KeyError:
        pass

    with _REPORTLAB_FONT_LOCK:
        try:
            pdfmetrics.getFont(REPORTLAB_CHINESE_FONT)
            return REPORTLAB_CHINESE_FONT
        except KeyError:
            pass

        candidates = [
            os.getenv("APP_EXPORT_FONT_ZH", ""),
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
        for value in candidates:
            if not value or not Path(value).is_file():
                continue
            try:
                pdfmetrics.registerFont(
                    TTFont(REPORTLAB_CHINESE_FONT, value, subfontIndex=0)
                )
                return REPORTLAB_CHINESE_FONT
            except Exception:
                continue
    raise ValueError("未找到可用于生成中文 PDF 的 TrueType 字体")


def _register_reportlab_export_font(
    target_language: str,
    *,
    bold: bool = False,
    italic: bool = False,
) -> str:
    if target_language == "zh" and not bold and not italic:
        return _register_reportlab_chinese_font()
    base_name = {
        "zh": REPORTLAB_CHINESE_FONT,
        "th": REPORTLAB_THAI_FONT,
        "en": REPORTLAB_ENGLISH_FONT,
    }[target_language]
    variant = "BoldItalic" if bold and italic else "Bold" if bold else "Italic" if italic else ""
    font_name = f"{base_name}{variant}"
    try:
        pdfmetrics.getFont(font_name)
        return font_name
    except KeyError:
        pass

    candidates = _reportlab_export_font_candidates(
        target_language,
        bold=bold,
        italic=italic,
    )
    with _REPORTLAB_FONT_LOCK:
        try:
            pdfmetrics.getFont(font_name)
            return font_name
        except KeyError:
            pass
        for value in candidates:
            if not value or not Path(value).is_file():
                continue
            try:
                kwargs = {"subfontIndex": 0} if Path(value).suffix.lower() == ".ttc" else {}
                pdfmetrics.registerFont(TTFont(font_name, value, **kwargs))
                return font_name
            except Exception:
                continue
    raise ValueError("未找到可用于生成译文 PDF 的字体")


def _reportlab_export_font_candidates(
    target_language: str,
    *,
    bold: bool,
    italic: bool,
) -> List[str]:
    style = "BOLD_ITALIC" if bold and italic else "BOLD" if bold else "ITALIC" if italic else ""
    environment_name = f"APP_EXPORT_FONT_{target_language.upper()}"
    styled_environment = f"{environment_name}_{style}" if style else environment_name
    candidates = [os.getenv(styled_environment, "")]
    if target_language == "zh":
        if bold:
            candidates.extend(
                [
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                    "/System/Library/Fonts/STHeiti Medium.ttc",
                ]
            )
        candidates.extend(
            [
                os.getenv("APP_EXPORT_FONT_ZH", ""),
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            ]
        )
    elif target_language == "th":
        suffix = "BoldItalic" if bold and italic else "Bold" if bold else "Italic" if italic else "Regular"
        candidates.extend(
            [
                f"/usr/share/fonts/truetype/noto/NotoSansThai-{suffix}.ttf",
                "/System/Library/Fonts/Supplemental/Thonburi Bold.ttf" if bold else "",
                "/System/Library/Fonts/Supplemental/Thonburi.ttf",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            ]
        )
    else:
        suffix = "-BoldOblique" if bold and italic else "-Bold" if bold else "-Oblique" if italic else ""
        arial_name = "Arial Bold Italic.ttf" if bold and italic else "Arial Bold.ttf" if bold else "Arial Italic.ttf" if italic else "Arial.ttf"
        candidates.extend(
            [
                f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf",
                f"/System/Library/Fonts/Supplemental/{arial_name}",
            ]
        )
    return candidates


def build_docx_export(
    source_content: bytes, translated_text: str, target_language: str
) -> bytes:
    document = Document()
    try:
        source_document = Document(BytesIO(source_content))
        source_section = source_document.sections[0]
        target_section = document.sections[0]
        target_section.page_width = source_section.page_width
        target_section.page_height = source_section.page_height
        target_section.top_margin = source_section.top_margin
        target_section.right_margin = source_section.right_margin
        target_section.bottom_margin = source_section.bottom_margin
        target_section.left_margin = source_section.left_margin
    except Exception:
        pass

    font_name = {
        "zh": "PingFang SC",
        "th": "Thonburi",
        "en": "Arial",
    }[target_language]
    normal = document.styles["Normal"]
    normal.font.name = font_name
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:ascii"), font_name)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), font_name)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
    normal._element.rPr.rFonts.set(qn("w:cs"), font_name)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    pages = split_page_translations(translated_text)
    if pages and PAGE_MARKER.search(translated_text):
        for index, page_number in enumerate(sorted(pages)):
            if index:
                document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            _append_docx_text(document, pages[page_number], font_name)
    else:
        _append_docx_text(document, translated_text, font_name)

    stream = BytesIO()
    document.save(stream)
    return stream.getvalue()


def split_page_translations(
    translated_text: str, page_count: Optional[int] = None
) -> Dict[int, str]:
    matches = list(PAGE_MARKER.finditer(translated_text))
    pages: Dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(translated_text)
        pages[int(match.group(1))] = translated_text[match.end() : end].strip()
    if not matches:
        pages[1] = translated_text.strip()
    if page_count:
        return {page_number: pages.get(page_number, "") for page_number in range(1, page_count + 1)}
    return pages


def _append_docx_text(document: Document, text: str, font_name: str) -> None:
    blocks = re.split(r"\n\s*\n", text.strip()) if text.strip() else [""]
    for block in blocks:
        paragraph = document.add_paragraph()
        lines = block.splitlines()
        for index, line in enumerate(lines):
            if index:
                paragraph.add_run().add_break()
            run = paragraph.add_run(line)
            run.font.name = font_name
            run._element.rPr.rFonts.set(qn("w:ascii"), font_name)
            run._element.rPr.rFonts.set(qn("w:hAnsi"), font_name)
            run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
            run._element.rPr.rFonts.set(qn("w:cs"), font_name)


def _find_export_font(target_language: str) -> Optional[Path]:
    environment_name = f"APP_EXPORT_FONT_{target_language.upper()}"
    candidates = [os.getenv(environment_name, "")]
    if target_language == "zh":
        candidates.extend(
            [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            ]
        )
    elif target_language == "th":
        candidates.extend(
            [
                "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
                "/System/Library/Fonts/Supplemental/Thonburi.ttf",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
            ]
        )
    return next((Path(value) for value in candidates if value and Path(value).is_file()), None)
