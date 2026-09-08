import math
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import fitz
from fontTools import subset
from fontTools.ttLib import TTFont
from pypdf import PdfReader
from pypdf._cmap import build_char_map, build_font_width_map, compute_font_width
from pypdf.generic import (
    ArrayObject,
    ByteStringObject,
    ContentStream,
    FloatObject,
    IndirectObject,
)

try:
    import regex as unicode_regex
except ImportError:  # Keep local development usable when the optional wheel is unavailable.
    unicode_regex = None


NATIVE_ENGINE_VERSION = 4
VECTOR_SCAN_DRAWING_THRESHOLD = 20_000
SUPPORTED_DOCUMENT_LANGUAGES = {"zh", "th", "en"}


@dataclass(frozen=True)
class _CodeRef:
    stream_xref: int
    operation_index: int
    array_index: int
    token_index: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "stream_xref": self.stream_xref,
            "operation_index": self.operation_index,
            "array_index": self.array_index,
            "token_index": self.token_index,
        }


@dataclass(frozen=True)
class _CodeToken:
    ref: _CodeRef
    value: str
    raw: bytes
    glyph_key: str
    width: float


class NativePdfExtractor:
    """Extract selected source-language text that maps to PDF text-show operators."""

    def __init__(self, content: bytes, source_language: str = "auto"):
        if source_language not in {"auto", *SUPPORTED_DOCUMENT_LANGUAGES}:
            raise ValueError("PDF 源语言无效")
        self.source_language = source_language
        try:
            self.document = fitz.open(stream=content, filetype="pdf")
            self.reader = PdfReader(BytesIO(content))
        except Exception as exc:
            raise ValueError("PDF 文件已损坏或格式不正确") from exc
        if self.document.page_count != len(self.reader.pages):
            self.document.close()
            raise ValueError("PDF 页面结构不一致")

    def close(self) -> None:
        self.document.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def iter_pages(self) -> Iterator[Tuple[int, List[Dict[str, Any]], Dict[str, Any]]]:
        for page_index in range(self.document.page_count):
            page_number = page_index + 1
            units, profile = self._extract_page(
                self.document[page_index],
                self.reader.pages[page_index],
                page_number,
            )
            yield page_number, units, profile

    def _extract_page(self, page, pdf_page, page_number: int):
        tokens, unsupported_show_ops = _page_code_tokens(pdf_page)
        content_characters = [
            (character, token.ref)
            for token in tokens
            for character in token.value
        ]
        raw_lines = _raw_page_lines(page)
        try:
            drawings = page.get_cdrawings()
        except Exception:
            drawings = page.get_drawings()
        table_cells = _native_pdf_table_cells(page)
        effective_source_language = _resolve_page_source_language(
            raw_lines, self.source_language
        )
        source_characters = [
            entry
            for line in raw_lines
            for entry in line["entries"]
            if _is_language_character(entry["char"], effective_source_language)
        ]
        raw_characters = [entry for line in raw_lines for entry in line["entries"]]
        mapping_reliable = _map_raw_characters_to_content_tokens(
            raw_characters, content_characters
        )
        if mapping_reliable:
            raw_lines = _merge_fragmented_raw_lines(raw_lines)
        units: List[Dict[str, Any]] = []
        skipped_directions = 0
        protected_logo_lines = _circular_logo_line_indexes(
            raw_lines, page.rect, effective_source_language
        )
        if mapping_reliable:
            # Every character in a mixed-language line must map back to the
            # content stream. The complete line is removed and written back as
            # one unit so English, numbers and punctuation remain part of the
            # sentence presented to the translation model.
            for line_index, line in enumerate(raw_lines, start=1):
                if line_index in protected_logo_lines:
                    continue
                line_units, skipped = _line_translation_units(
                    line, page_number, effective_source_language
                )
                units.extend(line_units)
                skipped_directions += skipped
        else:
            # The PDF text layer is still authoritative even when its decoded
            # characters cannot be mapped back to individual content-stream
            # glyph codes. Keep complete mixed-language lines and let the
            # layout adapter remove only their tight text boxes.
            for line_index, line in enumerate(raw_lines, start=1):
                if line_index in protected_logo_lines:
                    continue
                line_units, skipped = _line_text_layer_fallback_units(
                    line, page_number, line_index, effective_source_language
                )
                units.extend(line_units)
                skipped_directions += skipped

        if mapping_reliable:
            units = _annotate_native_table_cells(units, table_cells, page.rect)
            units = _merge_native_paragraph_units(units)

        image_count = len(page.get_images(full=True))
        drawing_count = len(drawings)
        vector_scan = drawing_count >= VECTOR_SCAN_DRAWING_THRESHOLD
        image_scan = image_count > 0 and not units
        visual_required = vector_scan or image_scan
        table_like = bool(
            units
            and not vector_scan
            and _looks_like_native_table(drawings, page.rect)
        )
        processable_source_chars = sum(
            sum(_is_language_character(char, effective_source_language) for char in unit["text"])
            for unit in units
        )
        protected_logo_source_chars = sum(
            sum(
                _is_language_character(entry["char"], effective_source_language)
                for entry in raw_lines[index - 1]["entries"]
            )
            for index in protected_logo_lines
        )
        # A vector-heavy page can still have a complete, editable text layer.
        # Treat that as the authoritative source for translation.  Drawing
        # density alone is not evidence that the page needs image OCR.
        native_text_complete = bool(
            mapping_reliable
            and source_characters
            and processable_source_chars + protected_logo_source_chars
            >= len(source_characters)
            and unsupported_show_ops == 0
        )
        profile = {
            "page_type": (
                "vector"
                if vector_scan
                else "image"
                if image_scan
                else "native_text"
                if units
                else "no_native_text"
            ),
            "source_language": effective_source_language,
            "text_chars": len(source_characters),
            "text_blocks": len(units),
            "image_count": image_count,
            "drawing_count": drawing_count,
            "page_rotation": int(page.rotation) % 360,
            "table_like": table_like,
            "visual_required": visual_required,
            "mapping_reliable": mapping_reliable,
            "unsupported_text_show_ops": unsupported_show_ops,
            "skipped_directions": skipped_directions,
            "processable_source_chars": processable_source_chars,
            "preserved_source_chars": max(0, len(source_characters) - processable_source_chars),
            "protected_logo_source_chars": protected_logo_source_chars,
            # Compatibility fields keep historical diagnostics readable for
            # Thai documents while all runtime decisions use the generic keys.
            "processable_thai_chars": (
                processable_source_chars if effective_source_language == "th" else 0
            ),
            "preserved_thai_chars": (
                max(0, len(source_characters) - processable_source_chars)
                if effective_source_language == "th"
                else 0
            ),
            "protected_logo_thai_chars": (
                protected_logo_source_chars if effective_source_language == "th" else 0
            ),
            "native_text_complete": native_text_complete,
            "native_pdf_version": NATIVE_ENGINE_VERSION,
        }
        return units, profile


def _looks_like_native_table(drawings, page_rect) -> bool:
    """Detect ruled/filled table structure without OCR or page rendering."""

    horizontal_edges = 0
    vertical_edges = 0
    filled_cells = 0
    minimum_horizontal = max(20.0, float(page_rect.width) * 0.025)
    minimum_vertical = max(8.0, float(page_rect.height) * 0.015)
    for drawing in drawings:
        for item in drawing.get("items", []):
            if not item:
                continue
            if item[0] == "re":
                rect = fitz.Rect(item[1])
                if rect.width >= minimum_horizontal and rect.height <= 2.0:
                    horizontal_edges += 1
                if rect.height >= minimum_vertical and rect.width <= 2.0:
                    vertical_edges += 1
                if rect.width >= minimum_horizontal and rect.height >= 4.0:
                    filled_cells += 1
            elif item[0] == "l":
                start = fitz.Point(item[1])
                end = fitz.Point(item[2])
                if (
                    abs(start.y - end.y) <= 0.8
                    and abs(start.x - end.x) >= minimum_horizontal
                ):
                    horizontal_edges += 1
                elif (
                    abs(start.x - end.x) <= 0.8
                    and abs(start.y - end.y) >= minimum_vertical
                ):
                    vertical_edges += 1
            if filled_cells >= 6 or (
                horizontal_edges >= 4 and vertical_edges >= 3
            ):
                return True
    return False


def build_native_pdf_export(
    source_content: bytes,
    layout_segments: Sequence[Dict[str, Any]],
    target_language: str,
    *,
    text_already_removed: bool = False,
    validation_source_content: Optional[bytes] = None,
) -> bytes:
    """Remove mapped source glyph codes and write translations at their baselines."""
    if target_language not in SUPPORTED_DOCUMENT_LANGUAGES:
        raise ValueError("PDF 原生文字引擎不支持该目标语言")
    segments = [dict(segment) for segment in layout_segments]
    _validate_segments(segments)
    selected_by_page: Dict[int, set] = {}
    for segment in segments:
        page_number = int(segment["page_number"])
        selected = selected_by_page.setdefault(page_number, set())
        for value in (segment.get("metadata") or {}).get("code_refs", []):
            selected.add(_code_ref_key(value))

    font_source = _find_system_font(target_language)
    font_path, temporary_font = _subset_font(
        font_source,
        "".join(_translation_text(segment) for segment in segments),
    )
    font = fitz.Font(fontfile=str(font_path))
    placements = [_prepare_placement(segment, font) for segment in segments]

    try:
        reader = None if text_already_removed else PdfReader(BytesIO(source_content))
        document = fitz.open(stream=source_content, filetype="pdf")
        try:
            if reader is not None:
                for page_number, selected in selected_by_page.items():
                    if page_number < 1 or page_number > document.page_count:
                        raise ValueError(f"ID_MISMATCH: page {page_number}")
                    _rewrite_page_text_streams(
                        document,
                        reader.pages[page_number - 1],
                        selected,
                    )

            page_shapes = {}
            for placement in placements:
                page = document[placement["page_number"] - 1]
                origin = fitz.Point(placement["origin"])
                shape = page_shapes.setdefault(placement["page_number"], page.new_shape())
                for line_index, text_line in enumerate(placement["lines"]):
                    line_origin = fitz.Point(
                        origin.x,
                        origin.y + line_index * placement["leading"],
                    )
                    shape.insert_text(
                        line_origin,
                        text_line,
                        fontname="MetaTrans",
                        fontfile=str(font_path),
                        fontsize=placement["font_size"],
                        color=placement["line_colors"][line_index],
                        morph=(line_origin, fitz.Matrix(placement["angle"])),
                    )
            for shape in page_shapes.values():
                shape.commit(overlay=True)
            # Rewriting already updates only referenced page streams. Deep
            # object deduplication is expensive on large PDFs and does not
            # affect rendering or text coverage, so keep lightweight cleanup.
            output = document.tobytes(garbage=1, deflate=True)
        finally:
            document.close()
    finally:
        if temporary_font is not None:
            temporary_font.unlink(missing_ok=True)

    _validate_written_pdf(validation_source_content or source_content, output, segments)
    return output


def prepare_native_pdf_source(
    source_content: bytes,
    layout_segments: Sequence[Dict[str, Any]],
) -> bytes:
    """Remove mapped source text so this work can overlap model requests."""
    segments = [dict(segment) for segment in layout_segments]
    _validate_segments(segments)
    selected_by_page: Dict[int, set] = {}
    for segment in segments:
        page_number = int(segment["page_number"])
        selected = selected_by_page.setdefault(page_number, set())
        for value in (segment.get("metadata") or {}).get("code_refs", []):
            selected.add(_code_ref_key(value))

    reader = PdfReader(BytesIO(source_content))
    document = fitz.open(stream=source_content, filetype="pdf")
    try:
        for page_number, selected in selected_by_page.items():
            if page_number < 1 or page_number > document.page_count:
                raise ValueError(f"ID_MISMATCH: page {page_number}")
            _rewrite_page_text_streams(
                document,
                reader.pages[page_number - 1],
                selected,
            )
        # This is an in-memory intermediate. Final output is compressed and
        # fully validated after translated text is inserted.
        return document.tobytes(garbage=0, deflate=False)
    finally:
        document.close()


def _page_code_tokens(pdf_page) -> Tuple[List[_CodeToken], int]:
    fonts = _font_maps(pdf_page)
    width_maps = {
        name: build_font_width_map(font_map[4], 1000.0)
        for name, font_map in fonts.items()
    }
    tokens: List[_CodeToken] = []
    unsupported_show_ops = 0
    current_font = None
    current_width_map = None
    for stream_xref, stream in _page_content_streams(pdf_page):
        data = stream.get_data()
        if not _contains_text_show_operator(data):
            continue
        content = ContentStream(stream, pdf_page.pdf)
        for operation_index, (operands, operator) in enumerate(content.operations):
            if operator == b"Tf":
                current_font = fonts.get(operands[0])
                current_width_map = width_maps.get(operands[0])
                continue
            if operator in {b"'", b'"'}:
                unsupported_show_ops += 1
                continue
            if current_font is None:
                continue
            if operator == b"Tj" and operands:
                tokens.extend(
                    _string_tokens(
                        operands[0],
                        current_font,
                        _CodeRef(stream_xref, operation_index, -1, 0),
                        current_width_map,
                    )
                )
            elif operator == b"TJ" and operands and isinstance(operands[0], list):
                for array_index, value in enumerate(operands[0]):
                    if isinstance(value, (str, bytes)):
                        tokens.extend(
                            _string_tokens(
                                value,
                                current_font,
                                _CodeRef(stream_xref, operation_index, array_index, 0),
                                current_width_map,
                            )
                        )
    return tokens, unsupported_show_ops


def _raw_page_lines(page):
    raw_lines: List[Dict[str, Any]] = []
    raw = page.get_text("rawdict", sort=False)
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            entries = []
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    value = str(char.get("c", ""))
                    entry = {
                        "char": value,
                        "bbox": tuple(char.get("bbox", ())),
                        "origin": tuple(char.get("origin", span.get("origin", (0, 0)))),
                        "size": float(span.get("size", 11.0)),
                        "color": int(span.get("color", 0)),
                    }
                    entries.append(entry)
            if entries:
                raw_lines.append(
                    {
                        "entries": entries,
                        "direction": tuple(float(value) for value in line.get("dir", (1, 0))),
                    }
                )
    return raw_lines


def _merge_fragmented_raw_lines(raw_lines: Sequence[Dict[str, Any]]):
    """Join adjacent same-baseline text objects emitted by slide/PDF exporters."""

    merged: List[Dict[str, Any]] = []
    for source_line in raw_lines:
        entries = list(source_line.get("entries") or [])
        while entries and entries[0]["char"].isspace():
            entries.pop(0)
        while entries and entries[-1]["char"].isspace():
            entries.pop()
        line = {
            **source_line,
            "entries": entries,
        }
        if not merged or not _raw_line_fragments_can_merge(merged[-1], line):
            merged.append(line)
            continue
        previous = merged[-1]
        previous_entries = previous["entries"]
        next_entries = line["entries"]
        previous_rect = _union_rects(
            fitz.Rect(entry["bbox"]) for entry in previous_entries
        )
        next_rect = _union_rects(
            fitz.Rect(entry["bbox"]) for entry in next_entries
        )
        template = previous_entries[-1]
        previous_entries.append(
            {
                "char": " ",
                "bbox": (
                    previous_rect.x1,
                    min(previous_rect.y0, next_rect.y0),
                    next_rect.x0,
                    max(previous_rect.y1, next_rect.y1),
                ),
                "origin": (previous_rect.x1, template["origin"][1]),
                "size": template["size"],
                "color": template["color"],
            }
        )
        previous_entries.extend(next_entries)
    return merged


def _raw_line_fragments_can_merge(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    first_entries = list(first.get("entries") or [])
    second_entries = list(second.get("entries") or [])
    if not first_entries or not second_entries:
        return False
    first_direction = tuple(first.get("direction") or (1.0, 0.0))
    second_direction = tuple(second.get("direction") or (1.0, 0.0))
    if (
        abs(float(first_direction[1])) > 0.05
        or abs(float(second_direction[1])) > 0.05
        or abs(float(first_direction[0]) - float(second_direction[0])) > 0.05
    ):
        return False
    first_sizes = [float(entry["size"]) for entry in first_entries]
    second_sizes = [float(entry["size"]) for entry in second_entries]
    first_size = median(first_sizes)
    second_size = median(second_sizes)
    if abs(first_size - second_size) > max(0.75, min(first_size, second_size) * 0.12):
        return False
    first_colors = {
        int(entry["color"]) & 0xFFFFFF
        for entry in first_entries
        if not entry["char"].isspace()
    }
    second_colors = {
        int(entry["color"]) & 0xFFFFFF
        for entry in second_entries
        if not entry["char"].isspace()
    }
    if first_colors != second_colors:
        return False
    first_rect = _union_rects(fitz.Rect(entry["bbox"]) for entry in first_entries)
    second_rect = _union_rects(fitz.Rect(entry["bbox"]) for entry in second_entries)
    baseline_difference = abs(
        median(float(entry["origin"][1]) for entry in first_entries)
        - median(float(entry["origin"][1]) for entry in second_entries)
    )
    if baseline_difference > max(0.75, min(first_size, second_size) * 0.12):
        return False
    gap = second_rect.x0 - first_rect.x1
    return -0.5 <= gap <= max(8.0, max(first_size, second_size) * 2.25)


def _line_translation_units(
    line: Dict[str, Any], page_number: int, source_language: str
):
    entries = list(line["entries"])
    while entries and entries[0]["char"].isspace():
        entries.pop(0)
    while entries and entries[-1]["char"].isspace():
        entries.pop()
    text = "".join(entry["char"] for entry in entries)
    if not entries or not any(
        _is_language_character(char, source_language) for char in text
    ):
        return [], 0
    if any(
        "code_ref" not in entry
        for entry in entries
        if not entry["char"].isspace()
    ):
        return [], 1

    direction = line["direction"]
    dx, dy = float(direction[0]), float(direction[1])
    if not all(math.isfinite(value) for value in (dx, dy)) or math.hypot(dx, dy) < 0.5:
        return [], 1

    refs = list(
        dict.fromkeys(
            entry["code_ref"] for entry in entries if "code_ref" in entry
        )
    )
    if not refs:
        return [], 1
    first_ref = refs[0]
    rect = _union_rects(fitz.Rect(entry["bbox"]) for entry in entries)
    width, height = _projected_dimensions(entries, direction)
    visible_entries = [entry for entry in entries if not entry["char"].isspace()]
    first_visible = visible_entries[0] if visible_entries else entries[0]
    color = first_visible["color"] & 0xFFFFFF
    segment_id = (
        f"pdf:p{page_number}:s{first_ref.stream_xref}:"
        f"o{first_ref.operation_index}:t{first_ref.token_index}"
    )
    return [
        {
            "segment_id": segment_id,
            "page_number": page_number,
            "text": text,
            "source_kind": "text",
            "bbox": tuple(rect),
            "font_size": median(entry["size"] for entry in visible_entries),
            "color": f"#{color:06x}",
            "alignment": "left",
            "metadata": {
                "native_pdf_version": NATIVE_ENGINE_VERSION,
                "engine": "content-stream",
                "code_refs": [reference.as_dict() for reference in refs],
                "origin": list(entries[0]["origin"]),
                "direction": list(direction),
                "available_width": width,
                "available_height": height,
                "grapheme_count": _grapheme_count(text),
                "translation_unit": "complete-line",
                "source_language": source_language,
            },
        }
    ], 0


def _merge_native_paragraph_units(units: Sequence[Dict[str, Any]]):
    """Restore paragraphs that presentation exporters split into visual lines."""

    paragraphs: List[List[Dict[str, Any]]] = []
    for unit in units:
        if (
            paragraphs
            and _native_units_belong_to_same_paragraph(paragraphs[-1][-1], unit)
        ):
            paragraphs[-1].append(unit)
        else:
            paragraphs.append([unit])

    merged_units = []
    for lines in paragraphs:
        if len(lines) == 1:
            merged_units.append(lines[0])
            continue
        first = lines[0]
        line_rects = [fitz.Rect(line["bbox"]) for line in lines]
        paragraph_rect = _union_rects(line_rects)
        sizes = [float(line.get("font_size") or 11.0) for line in lines]
        origins = [
            tuple(float(value) for value in (line.get("metadata") or {})["origin"])
            for line in lines
        ]
        leading_values = [
            origins[index][1] - origins[index - 1][1]
            for index in range(1, len(origins))
            if origins[index][1] > origins[index - 1][1]
        ]
        metadata = dict(first.get("metadata") or {})
        table_cell_bbox = metadata.get("table_cell_bbox")
        if table_cell_bbox:
            available_width = float(metadata.get("available_width") or paragraph_rect.width)
            available_height = float(
                metadata.get("available_height") or paragraph_rect.height
            )
        else:
            available_width = paragraph_rect.width
            available_height = paragraph_rect.height
        metadata.update(
            {
                "code_refs": [
                    reference
                    for line in lines
                    for reference in (line.get("metadata") or {}).get(
                        "code_refs", []
                    )
                ],
                "available_width": available_width,
                "available_height": available_height,
                "source_line_count": len(lines),
                "source_line_colors": [
                    str(line.get("color") or "#000000") for line in lines
                ],
                "source_leading": (
                    median(leading_values)
                    if leading_values
                    else median(sizes) * 1.2
                ),
                "translation_unit": "complete-paragraph",
            }
        )
        text = ""
        for line in lines:
            line_text = str(line.get("text") or "").strip()
            separator = "" if text.endswith("-") else " "
            text = f"{text}{separator}{line_text}".strip()
        metadata["grapheme_count"] = _grapheme_count(text)
        merged_units.append(
            {
                **first,
                "text": text,
                "bbox": tuple(paragraph_rect),
                "font_size": median(sizes),
                "metadata": metadata,
            }
        )
    return merged_units


def _native_units_belong_to_same_paragraph(
    first: Dict[str, Any], second: Dict[str, Any]
) -> bool:
    first_metadata = first.get("metadata") or {}
    second_metadata = second.get("metadata") or {}
    first_cell = first_metadata.get("table_cell_bbox")
    second_cell = second_metadata.get("table_cell_bbox")
    if bool(first_cell) != bool(second_cell):
        return False
    if first_cell and tuple(first_cell) != tuple(second_cell):
        return False
    if _native_units_are_separate_list_items(first, second):
        return False
    if (
        first_metadata.get("engine") != "content-stream"
        or second_metadata.get("engine") != "content-stream"
        or first.get("page_number") != second.get("page_number")
    ):
        return False
    if (
        first.get("color") != second.get("color")
        and not _native_unit_is_bullet_continuation(first, second)
    ):
        return False
    first_direction = tuple(first_metadata.get("direction") or (1.0, 0.0))
    second_direction = tuple(second_metadata.get("direction") or (1.0, 0.0))
    if (
        abs(float(first_direction[1])) > 0.05
        or abs(float(second_direction[1])) > 0.05
        or abs(float(first_direction[0]) - float(second_direction[0])) > 0.05
    ):
        return False
    first_size = float(first.get("font_size") or 11.0)
    second_size = float(second.get("font_size") or 11.0)
    if abs(first_size - second_size) > max(
        0.75, min(first_size, second_size) * 0.12
    ):
        return False
    first_rect = fitz.Rect(first["bbox"])
    second_rect = fitz.Rect(second["bbox"])
    if second_rect.y0 < first_rect.y0:
        return False
    vertical_gap = second_rect.y0 - first_rect.y1
    if vertical_gap < -max(1.0, min(first_size, second_size) * 0.55):
        return False
    if vertical_gap > max(4.0, max(first_size, second_size) * 0.8):
        return False
    start_tolerance = max(8.0, max(first_size, second_size) * 2.0)
    return abs(second_rect.x0 - first_rect.x0) <= start_tolerance


def _native_units_are_separate_list_items(
    first: Dict[str, Any], second: Dict[str, Any]
) -> bool:
    first_text = str(first.get("text") or "").lstrip()
    second_text = str(second.get("text") or "").lstrip()
    bullet_pattern = r"^[❖◆◇•●▪▫◊✓✔☑➢➤►]"
    timed_step_pattern = r"^\(\s*\d+\s+[A-Za-z\u0E00-\u0E7F\u3400-\u9FFF]+\s*\)"
    return bool(
        (re.match(bullet_pattern, first_text) and re.match(bullet_pattern, second_text))
        or (
            re.match(timed_step_pattern, first_text)
            and re.match(timed_step_pattern, second_text)
        )
    )


def _native_unit_is_bullet_continuation(
    first: Dict[str, Any], second: Dict[str, Any]
) -> bool:
    bullet_pattern = r"^[❖◆◇•●▪▫◊✓✔☑➢➤►]"
    first_text = str(first.get("text") or "").lstrip()
    second_text = str(second.get("text") or "").lstrip()
    if not re.match(bullet_pattern, first_text) or re.match(
        bullet_pattern, second_text
    ):
        return False
    first_rect = fitz.Rect(first["bbox"])
    second_rect = fitz.Rect(second["bbox"])
    font_size = max(
        float(first.get("font_size") or 11.0),
        float(second.get("font_size") or 11.0),
    )
    indent = second_rect.x0 - first_rect.x0
    return -1.0 <= indent <= font_size * 4.0


def _annotate_native_table_cells(
    units: Sequence[Dict[str, Any]], table_cells, page_rect: fitz.Rect
):
    """Constrain extractable text to its vector table cell when one exists."""

    if not table_cells:
        return list(units)
    annotated = []
    for source_unit in units:
        unit = dict(source_unit)
        rect = fitz.Rect(unit["bbox"])
        center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
        candidates = [
            fitz.Rect(value)
            for value in table_cells
            if fitz.Rect(value).contains(center)
        ]
        cell = min(candidates, key=lambda value: value.get_area()) if candidates else None
        if (
            cell is None
            or rect.x0 < cell.x0 - 2.0
            or rect.x1 > cell.x1 + 2.0
            or rect.y0 < cell.y0 - 2.0
            or rect.y1 > cell.y1 + 2.0
            or cell.height > max(rect.height * 8.0, page_rect.height * 0.15)
            or cell.width > page_rect.width * 0.75
        ):
            annotated.append(unit)
            continue
        font_size = max(0.75, float(unit.get("font_size") or 11.0))
        padding_x = max(1.0, min(4.0, font_size * 0.25))
        padding_y = max(0.8, min(3.0, font_size * 0.18))
        inner_right = cell.x1 - padding_x
        inner_bottom = cell.y1 - padding_y
        available_width = inner_right - rect.x0
        available_height = inner_bottom - rect.y0
        if available_width < 2.0 or available_height < font_size * 0.7:
            annotated.append(unit)
            continue
        metadata = dict(unit.get("metadata") or {})
        metadata.update(
            {
                "table_cell_bbox": [
                    round(float(value), 3)
                    for value in (cell.x0, cell.y0, cell.x1, cell.y1)
                ],
                "layout_container": "table-cell",
                "available_width": available_width,
                "available_height": available_height,
            }
        )
        unit["metadata"] = metadata
        annotated.append(unit)
    return annotated


def _native_pdf_table_cells(page):
    """Return cells recognized by the PDF engine, excluding empty decoration."""

    try:
        finder = page.find_tables()
    except Exception:
        return []
    cells = []
    page_area = max(1.0, page.rect.get_area())
    for table in finder.tables:
        table_rect = fitz.Rect(table.bbox)
        if table_rect.get_area() > page_area * 0.80:
            continue
        for value in table.cells:
            if value is None:
                continue
            cell = fitz.Rect(value)
            if cell.width >= 8.0 and cell.height >= 4.0:
                cells.append(tuple(cell))
    return cells


def _line_text_layer_fallback_units(
    line: Dict[str, Any], page_number: int, line_index: int, source_language: str
):
    entries = list(line["entries"])
    while entries and entries[0]["char"].isspace():
        entries.pop(0)
    while entries and entries[-1]["char"].isspace():
        entries.pop()
    text = "".join(entry["char"] for entry in entries)
    if not entries or not any(
        _is_language_character(char, source_language) for char in text
    ):
        return [], 0

    direction = line["direction"]
    dx, dy = float(direction[0]), float(direction[1])
    if (
        not all(math.isfinite(value) for value in (dx, dy))
        or math.hypot(dx, dy) < 0.5
    ):
        return [], 1
    angle = math.degrees(math.atan2(dy, dx)) % 360
    rotation = min(
        (0, 90, 180, 270),
        key=lambda value: abs(((angle - value + 180) % 360) - 180),
    )
    rect = _union_rects(fitz.Rect(entry["bbox"]) for entry in entries)
    width, height = _projected_dimensions(entries, direction)
    visible_entries = [entry for entry in entries if not entry["char"].isspace()]
    first_visible = visible_entries[0] if visible_entries else entries[0]
    color = first_visible["color"] & 0xFFFFFF
    return [
        {
            "segment_id": f"pdf:p{page_number}:text-layer:{line_index}",
            "page_number": page_number,
            "text": text,
            "source_kind": "text",
            "bbox": tuple(rect),
            "font_size": median(entry["size"] for entry in visible_entries),
            "color": f"#{color:06x}",
            "alignment": "left",
            "metadata": {
                "text_layer_fallback": True,
                "rotation": rotation,
                "angle": angle,
                "arbitrary_direction": not _is_cardinal(direction),
                "direction": list(direction),
                "origin": list(entries[0]["origin"]),
                "available_width": width,
                "available_height": height,
                "line_count": 1,
                "translation_unit": "complete-line",
                "source_language": source_language,
            },
        }
    ], 0


def _circular_logo_line_indexes(
    raw_lines, page_rect: fitz.Rect, source_language: str = "th"
) -> set:
    candidates = []
    for index, line in enumerate(raw_lines, start=1):
        entries = [entry for entry in line["entries"] if not entry["char"].isspace()]
        text = "".join(entry["char"] for entry in entries)
        if not entries or not any(
            _is_language_character(char, source_language) for char in text
        ):
            continue
        direction = line["direction"]
        if _grapheme_count(text) > 2:
            continue
        rect = _union_rects(fitz.Rect(entry["bbox"]) for entry in entries)
        angle = math.degrees(math.atan2(float(direction[1]), float(direction[0]))) % 360
        candidates.append((index, rect, angle))
    if len(candidates) < 8:
        return set()

    protected = set()
    for _, seed_rect, _ in candidates:
        seed_center = ((seed_rect.x0 + seed_rect.x1) / 2, (seed_rect.y0 + seed_rect.y1) / 2)
        neighborhood = [
            item
            for item in candidates
            if abs((item[1].x0 + item[1].x1) / 2 - seed_center[0]) <= 70
            and abs((item[1].y0 + item[1].y1) / 2 - seed_center[1]) <= 70
        ]
        if len(neighborhood) < 8:
            continue
        union = _union_rects(item[1] for item in neighborhood)
        angles = [item[2] for item in neighborhood]
        compact = (
            union.width <= min(180.0, page_rect.width * 0.10)
            and union.height <= min(180.0, page_rect.height * 0.10)
        )
        if compact and max(angles) - min(angles) >= 70.0:
            protected.update(item[0] for item in neighborhood)
    return protected


def build_text_layer_fallback_export(
    source_content: bytes,
    layout_segments: Sequence[Dict[str, Any]],
    target_language: str,
) -> bytes:
    """Replace decoded text-layer lines that lack stable content-stream IDs."""
    segments = [dict(segment) for segment in layout_segments]
    if not segments:
        return source_content
    font_source = _find_system_font(target_language)
    font_path, temporary_font = _subset_font(
        font_source,
        "".join(_translation_text(segment) for segment in segments),
    )
    font = fitz.Font(fontfile=str(font_path))
    document = fitz.open(stream=source_content, filetype="pdf")
    try:
        segments_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for segment in segments:
            metadata = segment.get("metadata") or {}
            if not metadata.get("text_layer_fallback"):
                raise ValueError(f"ID_MISMATCH: {segment.get('segment_id')}")
            translated = _translation_text(segment)
            source_language = str(metadata.get("source_language") or "th")
            if not translated or (
                source_language == "th" and any(_is_thai(char) for char in translated)
            ):
                raise ValueError(f"RESIDUAL_SOURCE_TEXT: {segment.get('segment_id')}")
            segments_by_page.setdefault(int(segment["page_number"]), []).append(segment)

        for page_number, page_segments in segments_by_page.items():
            page = document[page_number - 1]
            for segment in page_segments:
                rect = fitz.Rect(segment["bbox"])
                if rect.is_empty:
                    raise ValueError(f"ID_MISMATCH: {segment.get('segment_id')}")
                metadata = segment.get("metadata") or {}
                padding = 0.0 if metadata.get("arbitrary_direction") else max(
                    0.25, min(0.8, float(segment.get("font_size") or 4.0) * 0.10)
                )
                redaction = rect + (-padding, -padding, padding, padding)
                redaction.intersect(page.rect)
                page.add_redact_annot(redaction, fill=False, cross_out=False)
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                text=fitz.PDF_REDACT_TEXT_REMOVE,
            )

            shape = page.new_shape()
            for segment in page_segments:
                metadata = segment.get("metadata") or {}
                translated = _translation_text(segment)
                original_size = max(0.75, float(segment.get("font_size") or 4.0))
                available_width = max(1.0, float(metadata.get("available_width") or 0.0))
                target_width = max(0.001, font.text_length(translated, fontsize=original_size))
                fitted_size = min(original_size, original_size * available_width / target_width)
                if fitted_size < max(0.65, min(1.5, original_size * 0.35)):
                    raise ValueError(f"LAYOUT_OVERFLOW: {segment.get('segment_id')}")
                origin = fitz.Point(metadata["origin"])
                direction = metadata.get("direction") or [1.0, 0.0]
                angle = math.degrees(
                    math.atan2(-float(direction[1]), float(direction[0]))
                )
                color = str(segment.get("color") or "#000000")
                if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
                    color = "#000000"
                rgb = tuple(int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
                shape.insert_text(
                    origin,
                    translated,
                    fontname="MetaTransFallback",
                    fontfile=str(font_path),
                    fontsize=fitted_size,
                    color=rgb,
                    morph=(origin, fitz.Matrix(angle)),
                )
            shape.commit(overlay=True)
        output = document.tobytes(garbage=1, deflate=True)
    finally:
        document.close()
        if temporary_font is not None:
            temporary_font.unlink(missing_ok=True)

    reopened = fitz.open(stream=output, filetype="pdf")
    try:
        for page_number, page_segments in segments_by_page.items():
            output_text = reopened[page_number - 1].get_text("text")
            for segment in page_segments:
                if _translation_text(segment) not in output_text:
                    raise ValueError(f"ID_MISMATCH: {segment['segment_id']} was not written")
    finally:
        reopened.close()
    return output


def _map_raw_characters_to_content_tokens(raw_characters, content_characters) -> bool:
    """Map display characters while tolerating extractor-inserted whitespace."""
    raw_index = 0
    content_index = 0
    while raw_index < len(raw_characters) and content_index < len(content_characters):
        raw_item = raw_characters[raw_index]
        raw_value = raw_item["char"]
        content_value, reference = content_characters[content_index]
        if raw_value == content_value:
            raw_item["code_ref"] = reference
            raw_index += 1
            content_index += 1
        elif raw_value.isspace():
            raw_index += 1
        elif content_value.isspace():
            content_index += 1
        else:
            return False
    return all(
        item["char"].isspace() for item in raw_characters[raw_index:]
    ) and all(
        value.isspace() for value, _ in content_characters[content_index:]
    )


def _font_maps(pdf_page):
    resources = pdf_page.get("/Resources") or {}
    fonts = resources.get("/Font") or {}
    return {name: build_char_map(name, 200.0, pdf_page) for name in fonts}


def _page_content_streams(pdf_page):
    raw_contents = pdf_page.raw_get("/Contents")
    pending = list(raw_contents) if isinstance(raw_contents, list) else [raw_contents]
    while pending:
        value = pending.pop(0)
        if value is None:
            continue
        stream = value.get_object() if hasattr(value, "get_object") else value
        if isinstance(stream, list):
            pending[0:0] = list(stream)
            continue
        reference = value if isinstance(value, IndirectObject) else getattr(stream, "indirect_reference", None)
        if reference is not None:
            yield int(reference.idnum), stream


def _contains_text_show_operator(data: bytes) -> bool:
    return b"Tj" in data or b"TJ" in data or b"'" in data or b'"' in data


def _string_tokens(
    value,
    font_map,
    base_ref: _CodeRef,
    width_map=None,
) -> List[_CodeToken]:
    raw = getattr(value, "original_bytes", None)
    if raw is None:
        raw = bytes(value)
    encoding = font_map[2]
    unicode_map = font_map[3]
    code_size = int(unicode_map.get(-1, 1) or 1)
    if width_map is None:
        width_map = build_font_width_map(font_map[4], 1000.0)
    tokens = []
    for token_index, offset in enumerate(range(0, len(raw), code_size)):
        chunk = raw[offset : offset + code_size]
        if len(chunk) != code_size:
            return []
        try:
            if isinstance(encoding, str):
                glyph_key = chunk.decode(encoding, "surrogatepass")
            else:
                glyph_key = "".join(
                    encoding.get(byte, bytes((byte,)).decode("latin1"))
                    for byte in chunk
                )
        except Exception:
            return []
        decoded = "".join(unicode_map.get(char, char) for char in glyph_key)
        tokens.append(
            _CodeToken(
                ref=_CodeRef(
                    base_ref.stream_xref,
                    base_ref.operation_index,
                    base_ref.array_index,
                    token_index,
                ),
                value=decoded,
                raw=chunk,
                glyph_key=glyph_key,
                width=float(compute_font_width(width_map, glyph_key)),
            )
        )
    return tokens


def _rewrite_page_text_streams(document, pdf_page, selected: set) -> None:
    fonts = _font_maps(pdf_page)
    width_maps = {
        name: build_font_width_map(font_map[4], 1000.0)
        for name, font_map in fonts.items()
    }
    found = set()
    current_font = None
    current_width_map = None
    for stream_xref, stream in _page_content_streams(pdf_page):
        data = stream.get_data()
        if not _contains_text_show_operator(data):
            continue
        content = ContentStream(stream, pdf_page.pdf)
        rewritten = []
        changed = False
        for operation_index, (operands, operator) in enumerate(content.operations):
            if operator == b"Tf":
                current_font = fonts.get(operands[0])
                current_width_map = width_maps.get(operands[0])
                rewritten.append((operands, operator))
                continue
            if current_font is None:
                rewritten.append((operands, operator))
                continue
            if operator == b"Tj" and operands:
                tokens = _string_tokens(
                    operands[0],
                    current_font,
                    _CodeRef(stream_xref, operation_index, -1, 0),
                    current_width_map,
                )
                replacement, removed = _rewrite_string_tokens(tokens, selected)
                if removed:
                    rewritten.append(([ArrayObject(replacement)], b"TJ"))
                    found.update(removed)
                    changed = True
                    continue
            elif operator == b"TJ" and operands and isinstance(operands[0], list):
                replacement = ArrayObject()
                removed = set()
                for array_index, value in enumerate(operands[0]):
                    if isinstance(value, (str, bytes)):
                        tokens = _string_tokens(
                            value,
                            current_font,
                            _CodeRef(stream_xref, operation_index, array_index, 0),
                            current_width_map,
                        )
                        items, item_removed = _rewrite_string_tokens(tokens, selected)
                        replacement.extend(items)
                        removed.update(item_removed)
                    else:
                        replacement.append(value)
                if removed:
                    rewritten.append(([replacement], b"TJ"))
                    found.update(removed)
                    changed = True
                    continue
            rewritten.append((operands, operator))
        if changed:
            content.operations = rewritten
            document.update_stream(stream_xref, content.get_data(), compress=True)
    missing = selected - found
    if missing:
        raise ValueError(f"ID_MISMATCH: {len(missing)} PDF glyph references were not found")


def _rewrite_string_tokens(tokens: Sequence[_CodeToken], selected: set):
    output = ArrayObject()
    raw_buffer = bytearray()
    removed = set()
    pending_advance = 0.0

    def flush_raw():
        nonlocal raw_buffer
        if raw_buffer:
            output.append(ByteStringObject(bytes(raw_buffer)))
            raw_buffer = bytearray()

    def flush_advance():
        nonlocal pending_advance
        if abs(pending_advance) > 1e-6:
            output.append(FloatObject(-pending_advance))
            pending_advance = 0.0

    for token in tokens:
        key = _code_ref_key(token.ref.as_dict())
        if key in selected:
            flush_raw()
            pending_advance += token.width
            removed.add(key)
        else:
            flush_advance()
            raw_buffer.extend(token.raw)
    flush_raw()
    flush_advance()
    if not output:
        output.append(ByteStringObject(b""))
    return output, removed


def _prepare_placement(segment: Dict[str, Any], font: fitz.Font):
    metadata = segment.get("metadata") or {}
    translated = _translation_text(segment)
    if not translated:
        raise ValueError(f"ID_MISMATCH: {segment.get('segment_id')}")
    source_language = str(metadata.get("source_language") or "th")
    if source_language == "th" and any(_is_thai(char) for char in translated):
        raise ValueError(f"RESIDUAL_SOURCE_TEXT: {segment.get('segment_id')}")
    original_size = max(0.75, float(segment.get("font_size") or 11.0))
    available_width = max(1.0, float(metadata.get("available_width") or 0.0))
    source_line_count = max(1, int(metadata.get("source_line_count") or 1))
    if source_line_count > 1 or metadata.get("layout_container"):
        fitted_size, leading, lines = _fit_paragraph_text(
            translated,
            font,
            original_size,
            available_width,
            max(1.0, float(metadata.get("available_height") or 0.0)),
            max(
                original_size,
                float(metadata.get("source_leading") or original_size * 1.2),
            ),
        )
    else:
        target_width = max(
            0.001, font.text_length(translated, fontsize=original_size)
        )
        fitted_size = min(
            original_size, original_size * available_width / target_width
        )
        leading = fitted_size
        lines = [translated]
    # A1/A0 CAD title blocks legitimately contain source text below 1.5 pt.
    # Keep a proportional readability floor for those labels while retaining
    # the existing 1.5 pt hard floor for ordinary document text.
    minimum_size = max(0.75, min(1.5, original_size * 0.5))
    if fitted_size + 1e-6 < minimum_size:
        raise ValueError(f"LAYOUT_OVERFLOW: {segment.get('segment_id')}")
    direction = metadata.get("direction") or [1.0, 0.0]
    angle = math.degrees(math.atan2(-float(direction[1]), float(direction[0])))
    color = str(segment.get("color") or "#000000")
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
        color = "#000000"
    rgb = tuple(int(color[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
    source_line_colors = metadata.get("source_line_colors") or [color]
    parsed_line_colors = []
    for source_color in source_line_colors:
        source_color = str(source_color or "#000000")
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", source_color):
            source_color = "#000000"
        parsed_line_colors.append(
            tuple(
                int(source_color[index : index + 2], 16) / 255.0
                for index in (1, 3, 5)
            )
        )
    line_colors = [
        parsed_line_colors[min(index, len(parsed_line_colors) - 1)]
        for index in range(len(lines))
    ]
    return {
        "page_number": int(segment["page_number"]),
        "segment_id": segment["segment_id"],
        "text": translated,
        "lines": lines,
        "origin": tuple(float(value) for value in metadata["origin"]),
        "angle": angle,
        "font_size": fitted_size,
        "leading": leading,
        "color": rgb,
        "line_colors": line_colors,
    }


def _fit_paragraph_text(
    text: str,
    font: fitz.Font,
    original_size: float,
    available_width: float,
    available_height: float,
    original_leading: float,
):
    minimum_size = max(0.75, min(1.5, original_size * 0.5))

    def layout(font_size: float):
        lines = _wrap_text_to_width(text, font, font_size, available_width)
        scale = font_size / original_size
        leading = max(font_size, original_leading * scale)
        required_height = font_size + max(0, len(lines) - 1) * leading
        return lines, leading, required_height

    lines, leading, required_height = layout(original_size)
    if required_height <= available_height + original_size * 0.25:
        return original_size, leading, lines

    low = minimum_size
    high = original_size
    best = None
    for _ in range(16):
        candidate = (low + high) / 2.0
        candidate_lines, candidate_leading, candidate_height = layout(candidate)
        if candidate_height <= available_height + candidate * 0.25:
            best = (candidate, candidate_leading, candidate_lines)
            low = candidate
        else:
            high = candidate
    if best is None:
        raise ValueError("LAYOUT_OVERFLOW: paragraph")
    return best


def _wrap_text_to_width(
    text: str, font: fitz.Font, font_size: float, available_width: float
) -> List[str]:
    tokens = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9&./+_'\-]*|[\u3400-\u9FFF]|[^\s]",
        text,
    )
    if not tokens:
        return [text]
    lines: List[str] = []
    current = ""
    for token in tokens:
        separator = (
            " "
            if current
            and current[-1].isascii()
            and current[-1].isalnum()
            and token[0].isascii()
            and token[0].isalnum()
            else ""
        )
        candidate = f"{current}{separator}{token}"
        if not current and font.text_length(token, fontsize=font_size) > available_width:
            token_parts = []
            part = ""
            for character in token:
                candidate_part = f"{part}{character}"
                if part and font.text_length(
                    candidate_part, fontsize=font_size
                ) > available_width:
                    token_parts.append(part)
                    part = character
                else:
                    part = candidate_part
            if part:
                token_parts.append(part)
            if len(token_parts) > 1:
                lines.extend(token_parts[:-1])
                current = token_parts[-1]
                continue
        if (
            current
            and font.text_length(candidate, fontsize=font_size) > available_width
        ):
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [text]


def _validate_segments(segments: Sequence[Dict[str, Any]]) -> None:
    if not segments:
        raise ValueError("PDF 译文缺少可回写的文字片段")
    ids = [str(segment.get("segment_id") or "") for segment in segments]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("ID_MISMATCH: duplicate or empty segment id")
    for segment in segments:
        metadata = segment.get("metadata") or {}
        source_language = str(metadata.get("source_language") or "th")
        if (
            metadata.get("native_pdf_version") != NATIVE_ENGINE_VERSION
            or metadata.get("engine") != "content-stream"
            or not metadata.get("code_refs")
            or source_language not in SUPPORTED_DOCUMENT_LANGUAGES
            or not any(
                _is_language_character(char, source_language)
                for char in str(segment.get("text") or "")
            )
        ):
            raise ValueError(f"ID_MISMATCH: {segment.get('segment_id')}")


def _validate_written_pdf(
    source_content: bytes,
    output_content: bytes,
    segments: Sequence[Dict[str, Any]],
) -> None:
    source = fitz.open(stream=source_content, filetype="pdf")
    output = fitz.open(stream=output_content, filetype="pdf")
    try:
        if source.page_count != output.page_count:
            raise ValueError("ID_MISMATCH: output page count changed")
        segments_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for segment in segments:
            segments_by_page.setdefault(int(segment["page_number"]), []).append(segment)
        for page_number, page_segments in segments_by_page.items():
            source_page = source[page_number - 1]
            output_page = output[page_number - 1]
            output_text = output_page.get_text("text")
            normalized_output_text = re.sub(r"\s+", " ", output_text).strip()
            compact_output_text = re.sub(r"\s+", "", output_text)
            for segment in page_segments:
                translated = _translation_text(segment)
                normalized_translation = re.sub(r"\s+", " ", translated).strip()
                compact_translation = re.sub(r"\s+", "", translated)
                if (
                    normalized_translation not in normalized_output_text
                    and compact_translation not in compact_output_text
                ):
                    raise ValueError(f"ID_MISMATCH: {segment['segment_id']} was not written")
            source_language = str(
                (page_segments[0].get("metadata") or {}).get("source_language")
                or "th"
            )
            if any(
                str((segment.get("metadata") or {}).get("source_language") or "th")
                != source_language
                for segment in page_segments
            ):
                raise ValueError(f"ID_MISMATCH: inconsistent source language on page {page_number}")
            removed_source = sum(
                sum(
                    _is_language_character(char, source_language)
                    for char in str(segment["text"])
                )
                for segment in page_segments
            )
            if source_language == "th":
                source_characters = sum(
                    _is_thai(char) for char in source_page.get_text("text")
                )
                output_characters = sum(_is_thai(char) for char in output_text)
                if output_characters != source_characters - removed_source:
                    raise ValueError(f"RESIDUAL_SOURCE_TEXT: page {page_number}")
            if len(source_page.get_images(full=True)) != len(output_page.get_images(full=True)):
                raise ValueError(f"ID_MISMATCH: page {page_number} image resources changed")
    finally:
        output.close()
        source.close()


def _projected_dimensions(entries: Sequence[Dict[str, Any]], direction):
    dx, dy = direction
    length = math.hypot(dx, dy) or 1.0
    dx, dy = dx / length, dy / length
    nx, ny = -dy, dx
    parallel = []
    perpendicular = []
    for entry in entries:
        rect = fitz.Rect(entry["bbox"])
        points = (rect.tl, rect.tr, rect.bl, rect.br)
        parallel.extend(point.x * dx + point.y * dy for point in points)
        perpendicular.extend(point.x * nx + point.y * ny for point in points)
    return max(parallel) - min(parallel), max(perpendicular) - min(perpendicular)


def _is_cardinal(direction, tolerance: float = 1.5) -> bool:
    angle = math.degrees(math.atan2(float(direction[1]), float(direction[0]))) % 90
    return min(angle, 90 - angle) <= tolerance


def _is_thai(value: str) -> bool:
    return bool(value and len(value) == 1 and "\u0E00" <= value <= "\u0E7F")


def _is_chinese(value: str) -> bool:
    return bool(
        value
        and len(value) == 1
        and (
            "\u3400" <= value <= "\u4DBF"
            or "\u4E00" <= value <= "\u9FFF"
            or "\uF900" <= value <= "\uFAFF"
        )
    )


def _is_english(value: str) -> bool:
    return bool(value and len(value) == 1 and ("A" <= value <= "Z" or "a" <= value <= "z"))


def _is_language_character(value: str, language: str) -> bool:
    if language == "th":
        return _is_thai(value)
    if language == "zh":
        return _is_chinese(value)
    if language == "en":
        return _is_english(value)
    return False


def _resolve_page_source_language(raw_lines: Sequence[Dict[str, Any]], requested: str) -> str:
    if requested in SUPPORTED_DOCUMENT_LANGUAGES:
        return requested
    counts = {
        language: sum(
            _is_language_character(entry["char"], language)
            for line in raw_lines
            for entry in line["entries"]
        )
        for language in ("th", "zh", "en")
    }
    detected = max(counts, key=counts.get)
    # Image-only and outline-text pages have no usable native characters.
    # Keep them as automatic instead of arbitrarily choosing the first
    # language (historically Thai), so the visual reader can identify Chinese,
    # Thai or English from the rendered page itself.
    return detected if counts[detected] else "auto"


def _grapheme_count(value: str) -> int:
    if unicode_regex is not None:
        return sum(
            1
            for cluster in unicode_regex.findall(r"\X", value)
            if not cluster.isspace()
        )
    count = 0
    for char in value:
        if char.isspace():
            continue
        if unicodedata.category(char) in {"Mn", "Me"}:
            continue
        count += 1
    return count


def _code_ref_key(value: Dict[str, Any]):
    return (
        int(value["stream_xref"]),
        int(value["operation_index"]),
        int(value["array_index"]),
        int(value["token_index"]),
    )


def _translation_text(segment: Dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", str(segment.get("translated_text") or "")).strip()


def _union_rects(rects: Iterable[fitz.Rect]) -> fitz.Rect:
    values = list(rects)
    if not values:
        return fitz.Rect()
    result = fitz.Rect(values[0])
    for rect in values[1:]:
        result |= rect
    return result


def _find_system_font(target_language: str) -> Path:
    if target_language not in SUPPORTED_DOCUMENT_LANGUAGES:
        raise ValueError("PDF 原生文字引擎不支持该目标语言")
    environment = f"APP_EXPORT_FONT_{target_language.upper()}"
    candidates = [os.getenv(environment, "")]
    if target_language == "th":
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Thonburi.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansThai-Regular.ttf",
            ]
        )
    elif target_language == "zh":
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "/System/Library/Fonts/PingFang.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    for candidate in candidates:
        path = Path(candidate) if candidate else None
        if path and path.is_file():
            return path
    raise ValueError("未找到可用的系统字体")


def _subset_font(font_path: Path, text: str) -> Tuple[Path, Optional[Path]]:
    temporary_path = None
    font = None
    try:
        options = subset.Options()
        options.retain_gids = True
        kwargs = {"fontNumber": 0} if font_path.suffix.lower() in {".ttc", ".otc"} else {}
        font = TTFont(str(font_path), **kwargs)
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(text=text + " ")
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
