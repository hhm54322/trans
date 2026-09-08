import csv
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterator, List, Optional, Tuple
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import fitz

from .native_pdf import NativePdfExtractor
from .pdf_routing import freeze_pdf_page_route


WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DRAWING_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/main"
SLIDE_PART_PATTERN = re.compile(r"^ppt/slides/slide(\d+)\.xml$")


@dataclass
class DocumentSegment:
    segment_id: str
    page_number: int
    text: str
    source_kind: str = "text"
    bbox: Optional[Tuple[float, float, float, float]] = None
    font_size: float = 11.0
    color: str = "#000000"
    alignment: str = "left"
    part_name: Optional[str] = None
    paragraph_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    filename: str
    text: str
    page_count: int
    warnings: list
    ocr_required: bool = False
    ocr_pages: List[int] = field(default_factory=list)
    page_texts: List[str] = field(default_factory=list)
    segments: List[DocumentSegment] = field(default_factory=list)
    # A page-level profile lets callers choose a safe extraction/export path
    # instead of treating every PDF as either selectable text or a scan.
    page_types: List[str] = field(default_factory=list)
    page_profiles: List[Dict[str, Any]] = field(default_factory=list)
    visual_pages: List[int] = field(default_factory=list)
    page_routes: List[str] = field(default_factory=list)


@dataclass
class RenderedPdfPage:
    page_number: int
    content: bytes
    # Absolute PDF coordinates for a cropped visual input. None means the
    # content is a full page render.
    clip: Optional[Tuple[float, float, float, float]] = None


@dataclass
class ParsedPdfPage:
    page_number: int
    text: str
    segments: List[DocumentSegment]
    ocr_required: bool = False
    page_type: str = "text"
    profile: Dict[str, Any] = field(default_factory=dict)
    processing_route: str = ""


def parse_document(
    filename: str, content: bytes, source_language: str = "auto"
) -> ParsedDocument:
    extension = Path(filename).suffix.lower()
    if extension in {".txt", ".md"}:
        text = decode_text(content)
        return ParsedDocument(filename, text, 1, [], page_texts=[text])
    if extension == ".docx":
        return parse_docx(filename, content)
    if extension == ".pptx":
        return parse_pptx(filename, content)
    if extension == ".pdf":
        return parse_pdf(filename, content, source_language)
    raise ValueError(
        "暂不支持该文件格式，请上传 DOCX、PPTX、PDF、TXT 或 Markdown 文件"
    )


def parse_docx(filename: str, content: bytes) -> ParsedDocument:
    segments: List[DocumentSegment] = []
    try:
        with ZipFile(BytesIO(content)) as archive:
            names = set(archive.namelist())
            if "word/document.xml" not in names:
                raise ValueError("DOCX 文件缺少正文内容")
            part_names = ["word/document.xml"] + sorted(
                name
                for name in names
                if re.match(
                    r"^word/(?:header\d+|footer\d+|footnotes|endnotes)\.xml$",
                    name,
                )
            )
            for part_name in part_names:
                root = ElementTree.fromstring(archive.read(part_name))
                for paragraph_index, paragraph in enumerate(
                    root.iter(f"{{{WORD_NAMESPACE}}}p")
                ):
                    if _contains_word_field(paragraph):
                        continue
                    text = "".join(
                        node.text or ""
                        for node in paragraph.iter(f"{{{WORD_NAMESPACE}}}t")
                    ).strip()
                    if not text:
                        continue
                    segments.append(
                        DocumentSegment(
                            segment_id=f"docx:{part_name}:{paragraph_index}",
                            page_number=1,
                            text=text,
                            part_name=part_name,
                            paragraph_index=paragraph_index,
                        )
                    )
    except ValueError:
        raise
    except (BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ValueError("DOCX 文件已损坏或格式不正确") from exc

    text = "\n\n".join(segment.text for segment in segments)
    return ParsedDocument(
        filename,
        text,
        1,
        [],
        page_texts=[text],
        segments=segments,
    )


def parse_pptx(filename: str, content: bytes) -> ParsedDocument:
    segments: List[DocumentSegment] = []
    page_texts: List[str] = []
    try:
        with ZipFile(BytesIO(content)) as archive:
            slide_parts = sorted(
                (
                    (int(match.group(1)), name)
                    for name in archive.namelist()
                    if (match := SLIDE_PART_PATTERN.match(name))
                ),
                key=lambda value: value[0],
            )
            if not slide_parts:
                raise ValueError("PPTX 文件没有幻灯片")
            for page_number, part_name in slide_parts:
                root = ElementTree.fromstring(archive.read(part_name))
                page_segments = []
                for paragraph_index, paragraph in enumerate(
                    root.iter(f"{{{DRAWING_NAMESPACE}}}p")
                ):
                    if paragraph.find(f".//{{{DRAWING_NAMESPACE}}}fld") is not None:
                        continue
                    text = "".join(
                        node.text or ""
                        for node in paragraph.iter(f"{{{DRAWING_NAMESPACE}}}t")
                    ).strip()
                    if not text:
                        continue
                    segment = DocumentSegment(
                        segment_id=f"pptx:{part_name}:{paragraph_index}",
                        page_number=page_number,
                        text=text,
                        part_name=part_name,
                        paragraph_index=paragraph_index,
                    )
                    segments.append(segment)
                    page_segments.append(text)
                page_texts.append("\n".join(page_segments))
    except ValueError:
        raise
    except (BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ValueError("PPTX 文件已损坏或格式不正确") from exc

    text = "\n\n".join(page for page in page_texts if page)
    return ParsedDocument(
        filename,
        text,
        len(page_texts),
        [],
        page_texts=page_texts,
        segments=segments,
    )


def parse_pdf(
    filename: str, content: bytes, source_language: str = "auto"
) -> ParsedDocument:
    pages = list(iter_pdf_pages(content, source_language))
    segments = [segment for page in pages for segment in page.segments]
    page_texts = [page.text for page in pages]
    ocr_pages = [page.page_number for page in pages if page.ocr_required]
    page_types = [page.page_type for page in pages]
    page_profiles = [page.profile for page in pages]
    visual_pages = [
        page.page_number
        for page in pages
        if page.profile.get("visual_required")
    ]
    page_routes = [page.processing_route for page in pages]
    text = "\n\n".join(page for page in page_texts if page)
    warnings = native_pdf_profile_warnings(page_profiles)
    type_counts: Dict[str, int] = {}
    for page_type in page_types:
        type_counts[page_type] = type_counts.get(page_type, 0) + 1
    if len(type_counts) > 1:
        summary = "、".join(
            f"{page_type} {count} 页" for page_type, count in type_counts.items()
        )
        warnings.append(f"PDF 页面内容分类：{summary}")
    return ParsedDocument(
        filename,
        text,
        len(page_texts),
        warnings,
        bool(ocr_pages),
        ocr_pages,
        page_texts,
        segments,
        page_types,
        page_profiles,
        visual_pages,
        page_routes,
    )


def native_pdf_profile_warnings(page_profiles: List[Dict[str, Any]]) -> List[str]:
    no_text_layer_pages = [
        index
        for index, profile in enumerate(page_profiles, start=1)
        if not profile.get("text_blocks") and not profile.get("visual_required")
    ]
    visual_pages = [
        index
        for index, profile in enumerate(page_profiles, start=1)
        if profile.get("visual_required") and not profile.get("native_text_complete")
    ]
    preserved_pages = [
        (
            index,
            max(
                0,
                int(profile.get("preserved_source_chars") or profile.get("preserved_thai_chars") or 0)
                - int(
                    profile.get("protected_logo_source_chars")
                    or profile.get("protected_logo_thai_chars")
                    or 0
                ),
            ),
        )
        for index, profile in enumerate(page_profiles, start=1)
        if int(profile.get("preserved_source_chars") or profile.get("preserved_thai_chars") or 0)
        > int(
            profile.get("protected_logo_source_chars")
            or profile.get("protected_logo_thai_chars")
            or 0
        )
    ]
    logo_pages = [
        (
            index,
            int(
                profile.get("protected_logo_source_chars")
                or profile.get("protected_logo_thai_chars")
                or 0
            ),
        )
        for index, profile in enumerate(page_profiles, start=1)
        if int(
            profile.get("protected_logo_source_chars")
            or profile.get("protected_logo_thai_chars")
            or 0
        ) > 0
    ]
    warnings = []
    if visual_pages:
        warnings.append(
            "以下页为扫描图或缺少完整文字层的高密度矢量页，将使用视觉识别翻译："
            + ", ".join(map(str, visual_pages))
            + " 页"
        )
    if no_text_layer_pages:
        warnings.append(
            "以下页未发现可靠映射的源语言文字层，且不是扫描/轮廓页，已原样保留："
            + ", ".join(map(str, no_text_layer_pages))
            + " 页"
        )
    if preserved_pages:
        warnings.append(
            "以下页存在无法稳定回写的短小/任意角度源语言文字，已保留原文："
            + "、".join(f"{page} 页 {count} 字符" for page, count in preserved_pages)
        )
    if logo_pages:
        warnings.append(
            "以下页的环形 Logo 文字已作为图形标识保护，不拆分翻译："
            + "、".join(f"{page} 页 {count} 字符" for page, count in logo_pages)
        )
    return warnings


def pdf_page_count(content: bytes) -> int:
    document = _open_pdf(content)
    try:
        return document.page_count
    finally:
        document.close()


def iter_pdf_pages(
    content: bytes, source_language: str = "auto"
) -> Iterator[ParsedPdfPage]:
    try:
        with NativePdfExtractor(content, source_language) as extractor:
            for page_number, units, profile in extractor.iter_pages():
                # Vector-heavy pages may combine selectable title-block text
                # with outlined table text. Keep the reliable native units and
                # let the visual pass add only geometry that is absent from the
                # text layer.
                page_segments = [DocumentSegment(**unit) for unit in units]
                plan = freeze_pdf_page_route(profile)
                yield ParsedPdfPage(
                    page_number=page_number,
                    text="\n".join(segment.text for segment in page_segments),
                    segments=page_segments,
                    ocr_required=profile["page_type"] == "image",
                    page_type=profile["page_type"],
                    profile=profile,
                    processing_route=plan.route.value,
                )
    except Exception as exc:
        raise ValueError("PDF 页面内容解析失败，请检查文件内容") from exc


def _open_pdf(content: bytes):
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("PDF 文件已损坏或格式不正确") from exc
    if document.page_count < 1:
        document.close()
        raise ValueError("PDF 文件没有页面")
    return document


def _contains_word_field(paragraph) -> bool:
    return (
        paragraph.find(f".//{{{WORD_NAMESPACE}}}fldChar") is not None
        or paragraph.find(f".//{{{WORD_NAMESPACE}}}instrText") is not None
    )


def render_pdf_pages(
    content: bytes, page_numbers: list, desired_width: int
) -> list:
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("PDF 页面渲染失败，文件可能已损坏") from exc

    rendered = []
    try:
        for page_number in page_numbers:
            if page_number < 1 or page_number > document.page_count:
                raise ValueError(f"PDF 第 {page_number} 页不存在")
            page = document.load_page(page_number - 1)
            scale = min(4.0, max(1.0, desired_width / max(page.rect.width, 1)))
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                colorspace=fitz.csRGB,
                alpha=False,
            )
            rendered.append(
                RenderedPdfPage(page_number=page_number, content=pixmap.tobytes("png"))
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF 页面渲染失败，请检查文件内容") from exc
    finally:
        document.close()
    return rendered


def render_pdf_tiles(
    content: bytes,
    page_numbers: list,
    desired_width: int,
    tile_mode: str = "quadrants",
) -> list:
    """Render focused crops for vector-heavy pages with dense CAD labels."""
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("PDF 页面渲染失败，文件可能已损坏") from exc

    rendered = []
    try:
        for page_number in page_numbers:
            if page_number < 1 or page_number > document.page_count:
                raise ValueError(f"PDF 第 {page_number} 页不存在")
            page = document.load_page(page_number - 1)
            width = float(page.rect.width)
            height = float(page.rect.height)
            if tile_mode in {"grid", "directory"}:
                # Dense schedules need compact crops: very tall strips make
                # normalized vision coordinates drift by an entire row. A
                # small overlap prevents labels on a crop boundary from being
                # dropped; duplicate blocks are collapsed later by geometry.
                columns = 3
                rows = 3 if tile_mode == "directory" else 2
                overlap_x = width * 0.012
                overlap_y = height * 0.018
                clips = []
                for row in range(rows):
                    for column in range(columns):
                        x0 = width * column / columns
                        x1 = width * (column + 1) / columns
                        y0 = height * row / rows
                        y1 = height * (row + 1) / rows
                        clips.append(
                            (
                                max(0.0, x0 - (overlap_x if column else 0.0)),
                                max(0.0, y0 - (overlap_y if row else 0.0)),
                                min(width, x1 + (overlap_x if column + 1 < columns else 0.0)),
                                min(height, y1 + (overlap_y if row + 1 < rows else 0.0)),
                            )
                        )
            elif tile_mode == "scan":
                columns = 2
                rows = 2
                overlap_x = width * 0.02
                overlap_y = height * 0.02
                clips = []
                for row in range(rows):
                    for column in range(columns):
                        x0 = width * column / columns
                        x1 = width * (column + 1) / columns
                        y0 = height * row / rows
                        y1 = height * (row + 1) / rows
                        clips.append(
                            (
                                max(0.0, x0 - (overlap_x if column else 0.0)),
                                max(0.0, y0 - (overlap_y if row else 0.0)),
                                min(width, x1 + (overlap_x if column + 1 < columns else 0.0)),
                                min(height, y1 + (overlap_y if row + 1 < rows else 0.0)),
                            )
                        )
            elif tile_mode == "columns":
                # Directory/schedule sheets are dominated by long, narrow
                # rows. Full-height columns keep each row legible and avoid
                # spending vision context on empty drawing space.
                left = width * 0.03
                right = width * 0.88
                third = (right - left) / 3.0
                clips = (
                    (left, 0.0, left + third, height),
                    (left + third, 0.0, left + third * 2.0, height),
                    (left + third * 2.0, 0.0, right, height),
                )
            else:
                mid_x = width / 2.0
                mid_y = height / 2.0
                clips = (
                    (0.0, 0.0, mid_x, mid_y),
                    (mid_x, 0.0, width, mid_y),
                    (0.0, mid_y, mid_x, height),
                    (mid_x, mid_y, width, height),
                )
            for values in clips:
                clip = fitz.Rect(values)
                scale = min(4.0, max(1.0, desired_width / max(clip.width, 1)))
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    clip=clip,
                    colorspace=fitz.csRGB,
                    alpha=False,
                )
                rendered.append(
                    RenderedPdfPage(
                        page_number=page_number,
                        content=pixmap.tobytes("png"),
                        clip=values,
                    )
                )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF 页面分块渲染失败，请检查文件内容") from exc
    finally:
        document.close()
    return rendered


def ocr_image_text_blocks(
    content: bytes,
    language: str = "tha+eng",
    min_confidence: float = 35.0,
    page_segmentation_mode: int = 11,
) -> List[Dict[str, Any]]:
    """Extract candidate text boxes locally for vector pages.

    CAD PDFs frequently contain outline glyphs or nested form XObjects that
    PyMuPDF cannot expose as a text layer. Tesseract is used only as a
    detection pass; translation still goes through the configured GPT model.
    The helper is optional and returns an empty list when Tesseract is not
    installed, so the normal visual path remains fully portable.
    """
    executable = shutil.which("tesseract")
    if not executable or not content:
        return []
    try:
        languages = subprocess.run(
            [executable, "--list-langs"],
            input=b"",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        ).stdout.decode("utf-8", errors="ignore")
        available = set(languages.split())
        selected_language = language
        if "tha" not in available and "eng" in available:
            selected_language = "eng"
        elif "tha" not in available and "eng" not in available:
            return []
        result = subprocess.run(
            [
                executable,
                "stdin",
                "stdout",
                "-l",
                selected_language,
                "--psm",
                str(max(3, min(13, int(page_segmentation_mode)))),
                "tsv",
            ],
            input=content,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=45,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []

    try:
        csv.field_size_limit(max(csv.field_size_limit(), 10_000_000))
        rows = csv.DictReader(
            StringIO(result.stdout.decode("utf-8", errors="ignore")),
            delimiter="\t",
        )
    except (TypeError, ValueError):
        return []
    words_by_line: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    image_width = image_height = 0
    try:
        # ``Page.rect`` reports physical points and respects PNG DPI metadata;
        # Tesseract TSV coordinates are pixels. Using points scales every OCR
        # box (typically by 4/3 at 96 DPI) and moves labels into adjacent rows.
        image_pixmap = fitz.Pixmap(content)
        image_width = int(image_pixmap.width)
        image_height = int(image_pixmap.height)
    except Exception:
        pass
    for row in rows:
        try:
            text = str(row.get("text") or "").strip()
            confidence = float(row.get("conf") or -1)
            left = int(row.get("left") or 0)
            top = int(row.get("top") or 0)
            width = int(row.get("width") or 0)
            height = int(row.get("height") or 0)
            if not text or confidence < min_confidence or width <= 1 or height <= 1:
                continue
            image_width = max(image_width, left + width)
            image_height = max(image_height, top + height)
            key = (
                str(row.get("block_num") or "0"),
                str(row.get("par_num") or "0"),
                str(row.get("line_num") or "0"),
            )
            words_by_line.setdefault(key, []).append(
                {
                    "text": text,
                    "confidence": confidence,
                    "left": left,
                    "top": top,
                    "right": left + width,
                    "bottom": top + height,
                }
            )
        except (TypeError, ValueError):
            continue
    if image_width <= 0 or image_height <= 0:
        return []

    blocks: List[Dict[str, Any]] = []
    for words in words_by_line.values():
        # Tesseract has already grouped these words into one logical line.
        # Sorting by top first scrambles Thai glyphs when their accents cause
        # small vertical bbox differences, so preserve left-to-right order.
        words.sort(key=lambda item: item["left"])
        source_text = " ".join(item["text"] for item in words).strip()
        if not source_text:
            continue
        left = min(item["left"] for item in words)
        top = min(item["top"] for item in words)
        right = max(item["right"] for item in words)
        bottom = max(item["bottom"] for item in words)
        # Ignore isolated one-pixel glyph noise. Thai text is retained even
        # when it is only one character because CAD labels are often short.
        if right - left < 3 or bottom - top < 3:
            continue
        blocks.append(
            {
                "source_text": source_text,
                "confidence": sum(item["confidence"] for item in words)
                / max(1, len(words)),
                "bbox": [
                    max(0.0, min(1000.0, left / image_width * 1000.0)),
                    max(0.0, min(1000.0, top / image_height * 1000.0)),
                    max(0.0, min(1000.0, right / image_width * 1000.0)),
                    max(0.0, min(1000.0, bottom / image_height * 1000.0)),
                ],
            }
        )
    return blocks


def decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文本文件编码")
