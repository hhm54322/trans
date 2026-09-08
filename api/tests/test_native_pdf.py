import math

import fitz
import pytest

from app.services.exports import build_adaptive_pdf_export

from app.services.native_pdf import (
    NativePdfExtractor,
    _circular_logo_line_indexes,
    _find_system_font,
    _map_raw_characters_to_content_tokens,
    _prepare_placement,
    build_native_pdf_export,
    prepare_native_pdf_source,
)


def _source_pdf(text="ABC อาคาร 123", angle=0):
    document = fitz.open()
    page = document.new_page(width=420, height=300)
    font_path = str(_find_system_font("zh"))
    page.insert_font(fontname="SourceFont", fontfile=font_path)
    origin = fitz.Point(40, 80)
    page.insert_text(
        origin,
        text,
        fontname="SourceFont",
        fontfile=font_path,
        fontsize=14,
        morph=(origin, fitz.Matrix(angle)),
    )
    page.draw_line((20, 110), (400, 110), width=0.7)
    content = document.tobytes()
    document.close()
    return content


def _units(content, source_language="auto"):
    with NativePdfExtractor(content, source_language) as extractor:
        _, units, profile = next(extractor.iter_pages())
    return units, profile


def _char_origins(content, characters):
    document = fitz.open(stream=content, filetype="pdf")
    try:
        return [
            (char["c"], tuple(round(value, 3) for value in char["origin"]))
            for block in document[0].get_text("rawdict")["blocks"]
            if block.get("type") == 0
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            for char in span.get("chars", [])
            if char["c"] in characters
        ]
    finally:
        document.close()


def test_extracts_complete_mixed_line_with_content_stream_ids():
    units, profile = _units(_source_pdf())

    assert [unit["text"] for unit in units] == ["ABC อาคาร 123"]
    assert profile["mapping_reliable"] is True
    assert profile["native_text_complete"] is True
    assert profile["visual_required"] is False
    assert units[0]["segment_id"].startswith("pdf:p1:s")
    assert len(units[0]["metadata"]["code_refs"]) == len("ABC อาคาร 123")
    assert units[0]["metadata"]["translation_unit"] == "complete-line"


def test_unmapped_text_layer_falls_back_to_tight_complete_line_redaction(monkeypatch):
    source = _source_pdf()
    monkeypatch.setattr(
        "app.services.native_pdf._map_raw_characters_to_content_tokens",
        lambda *_args: False,
    )
    units, profile = _units(source)

    assert profile["mapping_reliable"] is False
    assert profile["preserved_thai_chars"] == 0
    assert [unit["text"] for unit in units] == ["ABC อาคาร 123"]
    assert units[0]["metadata"]["text_layer_fallback"] is True
    assert "native_pdf_version" not in units[0]["metadata"]

    units[0]["metadata"]["page_type"] = "vector"
    units[0]["translated_text"] = "ABC 建筑 123"
    output = build_adaptive_pdf_export(source, units, "zh")
    document = fitz.open(stream=output, filetype="pdf")
    try:
        text = document[0].get_text("text")
        assert "อาคาร" not in text
        assert "ABC 建筑 123" in text
    finally:
        document.close()


def test_unmapped_text_layer_redaction_uses_rotated_page_coordinates(monkeypatch):
    source_document = fitz.open(stream=_source_pdf(), filetype="pdf")
    source_document[0].set_rotation(180)
    source = source_document.tobytes()
    source_document.close()
    monkeypatch.setattr(
        "app.services.native_pdf._map_raw_characters_to_content_tokens",
        lambda *_args: False,
    )
    units, _ = _units(source)
    units[0]["metadata"]["page_type"] = "vector"
    units[0]["translated_text"] = "ABC 建筑 123"

    output = build_adaptive_pdf_export(source, units, "zh")
    document = fitz.open(stream=output, filetype="pdf")
    try:
        text = document[0].get_text("text")
        assert "อาคาร" not in text
        assert "ABC 建筑 123" in text
    finally:
        document.close()


def test_export_replaces_complete_line_and_preserves_mixed_content():
    source = _source_pdf()
    units, _ = _units(source)
    units[0]["translated_text"] = "ABC 建筑 123"
    source_document = fitz.open(stream=source, filetype="pdf")
    drawing_count = len(source_document[0].get_drawings())
    source_document.close()

    output = build_native_pdf_export(source, units, "zh")

    document = fitz.open(stream=output, filetype="pdf")
    try:
        text = document[0].get_text("text")
        assert "อาคาร" not in text
        assert "建筑" in text
        assert "ABC" in text
        assert "123" in text
        assert len(document[0].get_drawings()) == drawing_count
    finally:
        document.close()


def test_export_does_not_classify_or_validate_non_thai_content():
    source = _source_pdf()
    units, _ = _units(source)
    units[0]["translated_text"] = "建筑"

    output = build_native_pdf_export(source, units, "zh")

    document = fitz.open(stream=output, filetype="pdf")
    try:
        assert "建筑" in document[0].get_text("text")
        assert "อาคาร" not in document[0].get_text("text")
    finally:
        document.close()


@pytest.mark.parametrize(
    ("source_text", "source_language", "target_language", "translated_text"),
    [
        ("Project schedule", "en", "th", "กำหนดการโครงการ"),
        ("项目进度", "zh", "en", "Project schedule"),
        ("Project schedule", "auto", "th", "กำหนดการโครงการ"),
    ],
)
def test_native_pdf_supports_all_three_languages(
    source_text, source_language, target_language, translated_text
):
    source = _source_pdf(text=source_text)
    units, profile = _units(source, source_language)
    assert len(units) == 1
    assert profile["source_language"] == (
        "en" if source_language == "auto" else source_language
    )
    units[0]["translated_text"] = translated_text

    output = build_native_pdf_export(source, units, target_language)

    document = fitz.open(stream=output, filetype="pdf")
    try:
        text = document[0].get_text("text")
        assert source_text not in text
        assert translated_text in text
        assert document.page_count == 1
    finally:
        document.close()


def test_export_translates_a_single_thai_grapheme_line():
    source = _source_pdf(text="ปี")
    units, _ = _units(source)
    assert len(units) == 1
    units[0]["translated_text"] = "年"

    output = build_native_pdf_export(source, units, "zh")

    document = fitz.open(stream=output, filetype="pdf")
    try:
        text = document[0].get_text("text")
        assert "ปี" not in text
        assert "年" in text
    finally:
        document.close()


def test_cad_label_below_one_point_five_uses_its_real_source_size():
    font = fitz.Font(fontfile=str(_find_system_font("zh")))
    placement = _prepare_placement(
        {
            "segment_id": "pdf:p56:tiny",
            "page_number": 56,
            "translated_text": "楼梯 ST3 详图",
            "font_size": 1.448759913444519,
            "color": "#000000",
            "metadata": {
                "origin": [211.08, 1660.36],
                "direction": [1.0, 0.0],
                "available_width": 8.192001342773438,
            },
        },
        font,
    )

    assert 0.75 <= placement["font_size"] < 1.5


def test_circular_single_character_logo_lines_are_protected():
    lines = []
    for index, angle in enumerate(range(0, 360, 30)):
        radians = math.radians(angle)
        x = 100 + math.cos(radians) * 30
        y = 100 + math.sin(radians) * 30
        lines.append(
            {
                "direction": (math.cos(radians), math.sin(radians)),
                "entries": [
                    {
                        "char": "ก",
                        "bbox": (x, y, x + 5, y + 7),
                        "origin": (x, y + 6),
                        "size": 7,
                        "color": 0,
                    }
                ],
            }
        )

    assert _circular_logo_line_indexes(lines, fitz.Rect(0, 0, 1000, 1000)) == set(
        range(1, 13)
    )


def test_prepared_source_produces_the_same_valid_translation():
    source = _source_pdf()
    units, _ = _units(source)
    units[0]["translated_text"] = "ABC 建筑 123"

    prepared = prepare_native_pdf_source(source, units)
    output = build_native_pdf_export(
        prepared,
        units,
        "zh",
        text_already_removed=True,
        validation_source_content=source,
    )

    document = fitz.open(stream=output, filetype="pdf")
    try:
        assert document.page_count == 1
        text = document[0].get_text("text")
        assert "อาคาร" not in text
        assert "ABC 建筑 123" in text
    finally:
        document.close()


def test_translation_keeps_arbitrary_baseline_direction():
    source = _source_pdf(text="อาคาร", angle=27)
    units, _ = _units(source)
    units[0]["translated_text"] = "建筑"

    output = build_native_pdf_export(source, units, "zh")

    document = fitz.open(stream=output, filetype="pdf")
    try:
        translated_line = next(
            line
            for block in document[0].get_text("dict")["blocks"]
            if block.get("type") == 0
            for line in block.get("lines", [])
            if any("建筑" in span.get("text", "") for span in line.get("spans", []))
        )
        expected = (math.cos(math.radians(27)), -math.sin(math.radians(27)))
        assert translated_line["dir"] == pytest.approx(expected, abs=0.002)
    finally:
        document.close()


def test_long_translation_is_shrunk_without_wrapping():
    source = _source_pdf(text="อาคารปฏิบัติการ")
    units, _ = _units(source)
    units[0]["translated_text"] = "建筑实践教学中心"
    original_size = units[0]["font_size"]

    output = build_native_pdf_export(source, units, "zh")

    document = fitz.open(stream=output, filetype="pdf")
    try:
        target_sizes = [
            span["size"]
            for block in document[0].get_text("dict")["blocks"]
            if block.get("type") == 0
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if "建筑实践" in span.get("text", "")
        ]
        assert target_sizes
        assert target_sizes[0] <= original_size
        assert "\n" not in units[0]["translated_text"]
    finally:
        document.close()


def test_missing_content_reference_fails_explicitly():
    source = _source_pdf()
    units, _ = _units(source)
    units[0]["translated_text"] = "建筑"
    units[0]["metadata"]["code_refs"][0]["operation_index"] += 1000

    with pytest.raises(ValueError, match="ID_MISMATCH"):
        build_native_pdf_export(source, units, "zh")


def test_character_mapping_allows_extractor_inserted_spaces():
    raw = [{"char": value} for value in "ABC อาคาร 123"]
    references = [object() for _ in "ABCอาคาร123"]
    content = list(zip("ABCอาคาร123", references))

    assert _map_raw_characters_to_content_tokens(raw, content) is True
    mapped = [item["code_ref"] for item in raw if "code_ref" in item]
    assert mapped == references


def test_character_mapping_rejects_non_whitespace_differences():
    raw = [{"char": value} for value in "ABC อาคาร 123"]
    content = [(value, object()) for value in "ABCอาคXร123"]

    assert _map_raw_characters_to_content_tokens(raw, content) is False
