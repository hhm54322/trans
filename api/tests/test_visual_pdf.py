import cv2
import fitz
import numpy as np
import pytest
from types import SimpleNamespace

from app.services import documents, visual_pdf
from app.services.visual_pdf import (
    build_visual_pdf_export,
    extract_visual_page_units,
    prepare_dense_cad_translation_sheets,
)


class _FakeOcr:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def predict(self, _image):
        self.calls += 1
        return [self.result] if self.calls == 1 else []


class _FakeRecognizer:
    def __init__(self, results):
        self.results = results
        self.images = []
        self.batch_size = None

    def predict(self, images, *, batch_size):
        self.images = images
        self.batch_size = batch_size
        return self.results


class _FakeCadDetector:
    def __init__(self, polygons, scores):
        self.polygons = np.asarray(polygons)
        self.scores = np.asarray(scores)
        self.images = []

    def predict(self, image):
        self.images.append(image)
        return [{"dt_polys": self.polygons, "dt_scores": self.scores}]


def _blank_pdf(width=420, height=300):
    document = fitz.open()
    document.new_page(width=width, height=height)
    content = document.tobytes()
    document.close()
    return content


def _outline_segment(text="ABC อาคาร 123", translated="ABC 建筑 123"):
    return {
        "segment_id": "pdf:p1:o1",
        "page_number": 1,
        "text": text,
        "translated_text": translated,
        "source_kind": "outline-text",
        "bbox": [40.0, 60.0, 210.0, 82.0],
        "font_size": 12.0,
        "color": "#000000",
        "alignment": "left",
        "metadata": {
            "visual_pdf_version": 1,
            "translation_unit": "complete-line",
            "rotation": 0,
            "line_count": 1,
            "leading": 14.0,
            "cover_bbox": [40.0, 60.0, 210.0, 82.0],
        },
    }


def _text_spans(content):
    document = fitz.open(stream=content, filetype="pdf")
    try:
        return [
            span
            for block in document[0].get_text("dict")["blocks"]
            if block.get("type") == 0
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]
    finally:
        document.close()


def test_cad_png_network_encoding_is_pixel_exact():
    image = np.full((120, 640, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "ID001 CAD 123",
        (8, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    ok, encoded = visual_pdf._encode_cad_png(image)
    restored = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    assert ok is True
    assert np.array_equal(restored, image)


def test_pixmap_direct_conversion_matches_lossless_png_decode():
    document = fitz.open()
    page = document.new_page(width=160, height=90)
    page.draw_rect(
        fitz.Rect(8, 12, 95, 64),
        color=(0.1, 0.4, 0.8),
        fill=(0.9, 0.7, 0.2),
    )
    pixmap = page.get_pixmap(colorspace=fitz.csRGB, alpha=False)
    expected = cv2.imdecode(
        np.frombuffer(pixmap.tobytes("png"), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    actual = visual_pdf._pixmap_to_bgr_image(pixmap)
    document.close()

    assert np.array_equal(actual, expected)


def test_paddle_runtime_kwargs_are_optional_and_bounded(monkeypatch):
    monkeypatch.setattr(visual_pdf, "_PADDLE_ENABLE_MKLDNN", True)
    monkeypatch.setenv("APP_PADDLE_DEVICE", "gpu:0")
    monkeypatch.setenv("APP_PADDLE_CPU_THREADS", "200")

    assert visual_pdf._paddle_runtime_kwargs() == {
        "enable_mkldnn": True,
        "device": "gpu:0",
        "cpu_threads": 64,
    }

    monkeypatch.delenv("APP_PADDLE_DEVICE")
    monkeypatch.delenv("APP_PADDLE_CPU_THREADS")
    assert visual_pdf._paddle_runtime_kwargs() == {
        "enable_mkldnn": True,
        "cpu_threads": max(
            1,
            min(
                8,
                (
                    visual_pdf._CPU_COUNT - visual_pdf._CAD_TESSERACT_WORKERS
                )
                // visual_pdf._CAD_PADDLE_WORKERS,
            ),
        ),
    }


def test_cad_paddle_worker_recommendation_requires_cpu_and_memory_headroom():
    gibibyte = 1024**3

    assert visual_pdf._recommended_cad_paddle_workers(8, 12 * gibibyte) == 2
    assert visual_pdf._recommended_cad_paddle_workers(7, 32 * gibibyte) == 1
    assert visual_pdf._recommended_cad_paddle_workers(16, 11 * gibibyte) == 1


def test_cad_paddle_worker_count_accepts_auto_and_explicit_override(monkeypatch):
    monkeypatch.setattr(visual_pdf, "_CPU_COUNT", 8)
    monkeypatch.setattr(visual_pdf, "_MEMORY_LIMIT_BYTES", 16 * 1024**3)
    monkeypatch.setenv("APP_CAD_PADDLE_WORKERS", "auto")
    assert visual_pdf.cad_paddle_worker_count() == 2

    monkeypatch.setenv("APP_CAD_PADDLE_WORKERS", "9")
    assert visual_pdf.cad_paddle_worker_count() == 4


def test_normalize_tesseract_source_hint_joins_only_thai_gaps():
    assert visual_pdf._normalize_tesseract_source_hint(
        "  ร า ย ก า ร   ABC 123  "
    ) == "รายการ ABC 123"
    assert visual_pdf._normalize_tesseract_source_hint(
        "A B 1 2 ก ข"
    ) == "A B 1 2 กข"


def test_local_cad_review_recovers_complete_lines_from_a_partial_candidate(monkeypatch):
    document = fitz.open()
    page = document.new_page(width=420, height=300)
    detector = _FakeCadDetector(
        [
            [[400, 180], [700, 180], [700, 220], [400, 220]],
            [[410, 235], [650, 235], [650, 275], [410, 275]],
        ],
        [0.92, 0.89],
    )
    recognizer = _FakeRecognizer(
        [
            {"rec_text": "ทางลาด 1:12", "rec_score": 0.99},
            {"rec_text": "ผิวขัดเรียบเซาะร่อง", "rec_score": 0.88},
        ]
    )
    monkeypatch.setattr(visual_pdf, "_get_cad_text_detector", lambda: detector)
    monkeypatch.setattr(visual_pdf, "_get_text_recognizer", lambda: recognizer)

    recovered = visual_pdf.recover_dense_cad_review_text_lines(
        page,
        [
            {
                "bbox": (50.0, 50.0, 180.0, 72.0),
                "rotation": 0,
                "vertical": False,
                "origin_result_index": 2,
                "origin_item_id": "ID007",
            }
        ],
    )

    document.close()
    assert [item["source_hint"] for item in recovered] == [
        "ทางลาด 1:12",
        "ผิวขัดเรียบเซาะร่อง",
    ]
    assert {item["origin_item_id"] for item in recovered} == {"ID007"}
    assert all(item["candidate_provider"] == "paddle-local-review" for item in recovered)


def test_tesseract_blocks_include_average_line_confidence(monkeypatch):
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
        "left\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t80\tราย\n"
        "5\t1\t1\t1\t1\t2\t45\t20\t25\t12\t90\tการ\n"
    ).encode("utf-8")
    calls = iter(
        [
            SimpleNamespace(stdout=b"List of available languages:\ntha\neng\n"),
            SimpleNamespace(stdout=tsv, returncode=0),
        ]
    )
    monkeypatch.setattr(documents.shutil, "which", lambda _name: "/usr/bin/tesseract")
    documents._TESSERACT_LANGUAGES_BY_EXECUTABLE.clear()
    monkeypatch.setattr(documents.subprocess, "run", lambda *args, **kwargs: next(calls))
    monkeypatch.setattr(
        documents.fitz,
        "Pixmap",
        lambda _content: SimpleNamespace(width=100, height=100),
    )

    blocks = documents.ocr_image_text_blocks(b"png", min_confidence=25.0)

    assert len(blocks) == 1
    assert blocks[0]["source_text"] == "ราย การ"
    assert blocks[0]["confidence"] == pytest.approx(85.0)


def test_tesseract_language_probe_is_cached(monkeypatch):
    documents._TESSERACT_LANGUAGES_BY_EXECUTABLE.clear()
    calls = []

    def fake_run(*_args, **_kwargs):
        calls.append(True)
        return SimpleNamespace(stdout=b"List of available languages:\ntha\neng\n")

    monkeypatch.setattr(documents.subprocess, "run", fake_run)

    assert documents._available_tesseract_languages("/usr/bin/tesseract") == {
        "List",
        "of",
        "available",
        "languages:",
        "tha",
        "eng",
    }
    assert documents._available_tesseract_languages("/usr/bin/tesseract")
    assert len(calls) == 1


def test_paddle_cad_detector_uses_geometry_without_local_transcription(monkeypatch):
    detector = _FakeCadDetector(
        [
            [[10, 20], [120, 20], [120, 42], [10, 42]],
            [[150, 50], [153, 50], [153, 53], [150, 53]],
        ],
        [0.91, 0.99],
    )
    monkeypatch.setattr(visual_pdf, "_get_cad_text_detector", lambda: detector)

    seeds = visual_pdf._dense_paddle_detection_seeds(
        np.full((100, 200, 3), 255, dtype=np.uint8)
    )

    assert len(seeds) == 1
    assert seeds[0]["rect"] == fitz.Rect(10, 20, 120, 42)
    assert seeds[0]["source_text"] == ""
    assert seeds[0]["source_confidence"] == pytest.approx(91.0)


def test_paddle_cad_detector_tiles_large_pages_and_restores_coordinates(monkeypatch):
    detector = _FakeCadDetector(
        [[[10, 20], [120, 20], [120, 42], [10, 42]]],
        [0.91],
    )
    monkeypatch.setattr(visual_pdf, "_get_cad_text_detector", lambda: detector)

    seeds = visual_pdf._dense_paddle_detection_seeds(
        np.full((100, 3000, 3), 255, dtype=np.uint8),
        tile_side=1600,
        tile_overlap=200,
    )

    # The second tile starts at x=1400, and the same local detector box must
    # be translated back into the source-image coordinate system.
    assert [tuple(seed["rect"]) for seed in seeds] == [
        (10.0, 20.0, 120.0, 42.0),
        (1410.0, 20.0, 1520.0, 42.0),
    ]
    assert [image.shape[:2] for image in detector.images] == [(100, 1600), (100, 1600)]


def test_dense_cad_translation_sheets_accept_paddle_detection(monkeypatch):
    monkeypatch.setattr(
        visual_pdf,
        "_dense_paddle_detection_seeds",
        lambda _image: [
            {
                "rect": fitz.Rect(40, 60, 180, 82),
                "source_text": "",
                "source_confidence": 92.0,
            }
        ],
    )
    document = fitz.open(stream=_blank_pdf(), filetype="pdf")
    try:
        sheets = prepare_dense_cad_translation_sheets(
            document[0],
            1,
            desired_width=420,
            rows_per_sheet=18,
            detection_provider="paddle-detector",
        )
    finally:
        document.close()

    assert len(sheets) == 1
    candidate = sheets[0]["entries"]["ID001"]
    assert candidate["bbox"] == pytest.approx((40, 60, 180, 82))
    assert candidate["source_hint"] == ""


def test_visual_ocr_keeps_complete_mixed_line_and_ignores_non_thai(monkeypatch):
    fake_ocr = _FakeOcr(
        {
            "rec_texts": [
                "ABC อาคาร 123",
                "PURE ENGLISH 456",
                "รายการ (A-01)",
                "(M) THGIEH GNILIECว",
            ],
            "rec_scores": [0.99, 0.99, 0.99, 0.99],
            "rec_polys": [
                [[40, 60], [210, 60], [210, 82], [40, 82]],
                [[40, 100], [210, 100], [210, 122], [40, 122]],
                [[40, 140], [210, 140], [210, 162], [40, 162]],
                [[250, 50], [260, 50], [260, 180], [250, 180]],
            ],
        }
    )
    monkeypatch.setattr(visual_pdf, "_get_paddle_ocr", lambda: fake_ocr)
    document = fitz.open(stream=_blank_pdf(), filetype="pdf")
    try:
        units = extract_visual_page_units(
            document[0], 1, desired_width=420, minimum_score=0.78
        )
    finally:
        document.close()

    assert [unit["text"] for unit in units] == [
        "ABC อาคาร 123",
        "รายการ (A-01)",
    ]
    assert all(
        unit["metadata"]["translation_unit"] == "complete-line"
        for unit in units
    )


def test_scan_ocr_rejects_low_confidence_single_thai_handwriting(monkeypatch):
    fake_ocr = _FakeOcr(
        {
            "rec_texts": ["25-ค", "วันที่"],
            "rec_scores": [0.65, 0.65],
            "rec_polys": [
                [[40, 60], [100, 60], [100, 82], [40, 82]],
                [[40, 100], [120, 100], [120, 122], [40, 122]],
            ],
        }
    )
    monkeypatch.setattr(visual_pdf, "_get_paddle_ocr", lambda: fake_ocr)
    document = fitz.open(stream=_blank_pdf(), filetype="pdf")
    try:
        units = extract_visual_page_units(
            document[0], 1, desired_width=420, minimum_score=0.60
        )
    finally:
        document.close()

    assert [unit["text"] for unit in units] == ["วันที่"]


def test_scan_ocr_hides_only_significant_blue_handwriting():
    image = np.full((80, 160, 3), 255, dtype=np.uint8)
    image[10:12, 10:12] = (128, 0, 0)
    image[30:35, 30:70] = (128, 0, 0)

    cleaned = visual_pdf._remove_significant_blue_ink(image, scale=2.0)

    assert np.array_equal(cleaned[10:12, 10:12], image[10:12, 10:12])
    assert np.all(cleaned[30:35, 30:70] == 255)


def test_visual_export_is_vector_and_uses_distinct_font_resource():
    output = build_visual_pdf_export(_blank_pdf(), [_outline_segment()], "zh")

    document = fitz.open(stream=output, filetype="pdf")
    try:
        page = document[0]
        assert not page.get_images(full=True)
        assert "ABC 建筑 123" in page.get_text("text")
        assert any("MetaTransVisual" in str(item) for item in page.get_fonts(full=True))
    finally:
        document.close()


def test_visual_export_writes_overlapping_duplicate_translation_once():
    primary = _outline_segment(text="ทางลาด 1:12", translated="坡道 1:12")
    duplicate = _outline_segment(text="ทางลาด 1:12", translated="坡道 1:12")
    duplicate["segment_id"] = "pdf:p1:o2"
    duplicate["bbox"] = [72.0, 63.0, 160.0, 75.0]
    duplicate["font_size"] = 6.0
    duplicate["metadata"] = dict(duplicate["metadata"])
    duplicate["metadata"]["cover_bbox"] = list(duplicate["bbox"])

    output = build_visual_pdf_export(_blank_pdf(), [primary, duplicate], "zh")

    spans = [span for span in _text_spans(output) if span["text"] == "坡道 1:12"]
    assert len(spans) == 1
    assert spans[0]["size"] == pytest.approx(12.0, abs=0.05)


def test_visual_export_starts_at_source_size_and_only_shrinks_on_overflow():
    normal = build_visual_pdf_export(_blank_pdf(), [_outline_segment()], "zh")
    normal_span = next(
        span for span in _text_spans(normal) if "ABC 建筑 123" in span["text"]
    )
    assert normal_span["size"] == pytest.approx(12.0, abs=0.05)

    long_segment = _outline_segment(
        translated="这是一段需要在原始文字区域内完整排入的较长中文译文"
    )
    shrunk = build_visual_pdf_export(_blank_pdf(), [long_segment], "zh")
    shrunk_span = next(
        span for span in _text_spans(shrunk) if "这是一段" in span["text"]
    )
    assert shrunk_span["size"] < 12.0


def test_indexed_cad_export_can_fit_a_long_translation_in_a_tiny_box():
    segment = _outline_segment(
        text="ข้อความ",
        translated="钢筋混凝土构件详图",
    )
    segment["bbox"] = [40.0, 60.0, 78.0, 66.0]
    segment["font_size"] = 4.3
    segment["metadata"]["cover_bbox"] = list(segment["bbox"])
    segment["metadata"]["ocr_provider"] = "gpt-indexed-image"

    output = build_visual_pdf_export(_blank_pdf(), [segment], "zh")

    assert "钢筋混凝土构件详图" in "".join(
        span["text"] for span in _text_spans(output)
    )


def test_indexed_cad_export_does_not_scan_all_page_drawings(monkeypatch):
    segment = _outline_segment(text="ข้อความ", translated="构件")
    segment["metadata"]["ocr_provider"] = "gpt-indexed-image"

    monkeypatch.setattr(
        fitz.Page,
        "get_drawings",
        lambda _page: pytest.fail("indexed CAD export scanned all drawings"),
    )

    assert build_visual_pdf_export(_blank_pdf(), [segment], "zh")


def test_dense_tiles_overlap_without_leaving_page_bounds():
    image = np.zeros((1000, 1500, 3), dtype=np.uint8)

    tiles = visual_pdf._dense_ocr_image_tiles(image)

    assert len(tiles) == 6
    assert all(tile.size for tile, _, _, _ in tiles)
    assert all(x >= 0 and y >= 0 for _, x, y, _ in tiles)
    assert tiles[0][0].shape[1] > 1500 / 3
    assert tiles[0][0].shape[0] > 1000 / 2


def test_internal_tile_edge_fragments_are_rejected_but_page_edges_are_kept():
    assert visual_pdf._polygon_touches_internal_tile_edge(
        [[1, 20], [40, 20], [40, 40], [1, 40]],
        100,
        0,
        500,
        500,
        1500,
        1000,
    )
    assert not visual_pdf._polygon_touches_internal_tile_edge(
        [[1, 20], [40, 20], [40, 40], [1, 40]],
        0,
        0,
        500,
        500,
        1500,
        1000,
    )


def test_overlapping_tile_detections_on_one_baseline_are_duplicates():
    complete = fitz.Rect(100, 100, 300, 120)
    clipped = fitz.Rect(250, 101, 330, 121)
    next_row = fitz.Rect(250, 125, 330, 145)

    assert visual_pdf._same_ocr_line(complete, clipped, 0)
    assert not visual_pdf._same_ocr_line(complete, next_row, 0)


def test_dense_page_accepts_low_confidence_thai_phrase_not_single_glyph(
    monkeypatch,
):
    fake_ocr = _FakeOcr(
        {
            "rec_texts": ["กข", "Aก"],
            "rec_scores": [0.65, 0.65],
            "rec_polys": [
                [[20, 20], [80, 20], [80, 40], [20, 40]],
                [[20, 60], [80, 60], [80, 80], [20, 80]],
            ],
        }
    )
    monkeypatch.setenv("APP_CAD_OCR_MODE", "deep")
    monkeypatch.setattr(visual_pdf, "_get_paddle_ocr", lambda: fake_ocr)
    monkeypatch.setattr(visual_pdf, "DENSE_VECTOR_DRAWING_THRESHOLD", 0)
    document = fitz.open(stream=_blank_pdf(), filetype="pdf")
    try:
        units = extract_visual_page_units(
            document[0], 1, desired_width=420, minimum_score=0.78
        )
    finally:
        document.close()

    assert [unit["text"] for unit in units] == ["กข"]


def test_dense_page_uses_fast_batch_path_without_full_paddle(monkeypatch):
    monkeypatch.setenv("APP_CAD_OCR_MODE", "fast")
    monkeypatch.setattr(visual_pdf, "DENSE_VECTOR_DRAWING_THRESHOLD", 0)
    monkeypatch.setattr(
        visual_pdf,
        "_get_paddle_ocr",
        lambda: pytest.fail("dense fast mode initialized the full Paddle pipeline"),
    )
    monkeypatch.setattr(
        visual_pdf,
        "_dense_fast_ocr_candidates",
        lambda _image, _page, **_kwargs: [
            (
                "ABC อาคาร 123",
                0.95,
                [[20, 20], [180, 20], [180, 40], [20, 40]],
                0,
                "tesseract-batch-1",
            )
        ],
    )
    document = fitz.open(stream=_blank_pdf(), filetype="pdf")
    try:
        units = extract_visual_page_units(
            document[0], 1, desired_width=420, minimum_score=0.78
        )
    finally:
        document.close()

    assert [unit["text"] for unit in units] == ["ABC อาคาร 123"]
    assert units[0]["metadata"]["ocr_pass"] == "tesseract-batch-1"


def test_deep_table_selector_requires_multiple_large_regions(monkeypatch):
    document = fitz.open(stream=_blank_pdf(width=1000, height=1000), filetype="pdf")
    try:
        monkeypatch.setattr(
            visual_pdf,
            "find_dense_table_regions",
            lambda _page: [fitz.Rect(0, 0, 1000, 110)],
        )
        assert visual_pdf.requires_deep_table_ocr(document[0]) is False

        monkeypatch.setattr(
            visual_pdf,
            "find_dense_table_regions",
            lambda _page: [
                fitz.Rect(0, 0, 450, 500),
                fitz.Rect(500, 0, 950, 500),
            ],
        )
        assert visual_pdf.requires_deep_table_ocr(document[0]) is True
    finally:
        document.close()


def test_dense_fast_batch_maps_horizontal_and_vertical_complete_lines(monkeypatch):
    recognizer = _FakeRecognizer(
        [
            {"rec_text": "W6 ผนัง 123", "rec_score": 0.96},
            {"rec_text": "ชั้น 2", "rec_score": 0.91},
            {"rec_text": "PURE ENGLISH", "rec_score": 0.99},
        ]
    )
    monkeypatch.setattr(
        visual_pdf,
        "_dense_tesseract_seed_rects",
        lambda _image: [
            fitz.Rect(10, 20, 110, 40),
            fitz.Rect(150, 10, 170, 90),
            fitz.Rect(20, 60, 120, 80),
        ],
    )
    monkeypatch.setattr(visual_pdf, "_get_text_recognizer", lambda: recognizer)
    document = fitz.open(stream=_blank_pdf(width=200, height=100), filetype="pdf")
    try:
        diagnostics = {}
        candidates = visual_pdf._dense_fast_ocr_candidates(
            np.zeros((100, 200, 3), dtype=np.uint8),
            document[0],
            batch_size=16,
            diagnostics=diagnostics,
        )
    finally:
        document.close()

    assert recognizer.batch_size == 16
    assert recognizer.images[0].shape[0] >= 60
    assert recognizer.images[1].shape[0] >= 60
    assert [item[0] for item in candidates] == ["W6 ผนัง 123", "ชั้น 2"]
    assert candidates[0][2] == [
        [10.0, 20.0],
        [110.0, 20.0],
        [110.0, 40.0],
        [10.0, 40.0],
    ]
    assert candidates[0][3] == 0
    assert candidates[1][3] == 270
    assert diagnostics == {
        "seed_count": 3,
        "seed_rects": [
            (10.0, 20.0, 110.0, 40.0),
            (150.0, 10.0, 170.0, 90.0),
            (20.0, 60.0, 120.0, 80.0),
        ],
        "recognized_candidate_count": 2,
    }


def test_dense_cad_sheet_keeps_stable_ids_and_source_geometry(monkeypatch):
    monkeypatch.setattr(
        visual_pdf,
        "_dense_tesseract_seed_candidates",
        lambda _image, *, thai_only=True: [
            {"rect": fitz.Rect(10, 20, 110, 40), "source_text": "ผนัง"},
            {"rect": fitz.Rect(150, 10, 170, 90), "source_text": "ชั้น"},
        ],
    )
    document = fitz.open(stream=_blank_pdf(width=200, height=100), filetype="pdf")
    try:
        sheets = visual_pdf.prepare_dense_cad_translation_sheets(
            document[0],
            1,
            desired_width=200,
            rows_per_sheet=25,
            native_units=[],
        )
    finally:
        document.close()

    assert len(sheets) == 1
    assert list(sheets[0]["entries"]) == ["ID001", "ID002"]
    assert sheets[0]["content"].startswith(b"\x89PNG")
    first = sheets[0]["entries"]["ID001"]
    second = sheets[0]["entries"]["ID002"]
    assert first["rotation"] == 0
    assert second["rotation"] == 270
    assert fitz.Rect(first["bbox"]) == fitz.Rect(10, 20, 110, 40)


def test_dense_cad_sheet_can_use_global_ids_across_images(monkeypatch):
    monkeypatch.setattr(
        visual_pdf,
        "_dense_tesseract_seed_candidates",
        lambda _image, *, thai_only=True: [
            {
                "rect": fitz.Rect(5 + index * 4, 20, 30 + index * 4, 40),
                "source_text": "ผนัง",
                "source_confidence": 90.0,
            }
            for index in range(5)
        ],
    )
    document = fitz.open(stream=_blank_pdf(width=200, height=100), filetype="pdf")
    try:
        sheets = visual_pdf.prepare_dense_cad_translation_sheets(
            document[0],
            1,
            desired_width=200,
            rows_per_sheet=2,
            native_units=[],
            globally_unique_ids=True,
        )
    finally:
        document.close()

    assert [list(sheet["entries"]) for sheet in sheets] == [
        ["ID0001", "ID0002"],
        ["ID0003", "ID0004"],
        ["ID0005"],
    ]


def test_dense_cad_sheet_reuses_supplied_seed_rects(monkeypatch):
    monkeypatch.setattr(
        visual_pdf,
        "_dense_tesseract_seed_rects",
        lambda *_args, **_kwargs: pytest.fail("Tesseract candidates were recomputed"),
    )
    document = fitz.open(stream=_blank_pdf(width=200, height=100), filetype="pdf")
    try:
        sheets = visual_pdf.prepare_dense_cad_translation_sheets(
            document[0],
            1,
            desired_width=200,
            rows_per_sheet=25,
            native_units=[],
            seed_rects=[(10, 20, 110, 40)],
        )
    finally:
        document.close()

    assert len(sheets) == 1
    assert list(sheets[0]["entries"]) == ["ID001"]


def test_dense_cad_review_sheet_marks_target_and_preserves_origin_mapping():
    document = fitz.open()
    page = document.new_page(width=240, height=140)
    page.insert_text((48, 65), "CONTEXT TARGET 123", fontsize=8)
    try:
        sheets = visual_pdf.prepare_dense_cad_review_sheets(
            page,
            [
                {
                    "bbox": (80, 54, 145, 68),
                    "rotation": 0,
                    "vertical": False,
                    "source_hint": "ผนัง",
                    "source_confidence": 80.0,
                    "origin_result_index": 2,
                    "origin_item_id": "ID004",
                }
            ],
            desired_width=1200,
            rows_per_sheet=3,
        )
    finally:
        document.close()

    assert len(sheets) == 1
    assert sheets[0]["focused_review"] is True
    assert list(sheets[0]["entries"]) == ["RID0001"]
    entry = sheets[0]["entries"]["RID0001"]
    assert entry["origin_result_index"] == 2
    assert entry["origin_item_id"] == "ID004"
    import cv2

    decoded = cv2.imdecode(
        np.frombuffer(sheets[0]["content"], dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded.shape == (280, 2400, 3)
    assert np.count_nonzero(
        (decoded[:, :, 2] > 180)
        & (decoded[:, :, 2] > decoded[:, :, 1] * 1.5)
    ) > 20


def test_dense_cad_review_sheet_accepts_exact_cached_page_render():
    document = fitz.open()
    page = document.new_page(width=240, height=140)
    page.insert_text((48, 65), "CONTEXT TARGET 123", fontsize=8)
    candidate = {
        "bbox": (80, 54, 145, 68),
        "rotation": 0,
        "vertical": False,
        "source_hint": "ผนัง",
        "source_confidence": 80.0,
    }
    try:
        cached_png = visual_pdf.render_dense_cad_page_png(
            page,
            desired_width=1200,
            minimum_render_scale=2.5,
            maximum_render_scale=5.0,
        )
        direct = visual_pdf.prepare_dense_cad_review_sheets(
            page,
            [candidate],
            desired_width=1200,
            rows_per_sheet=3,
        )

        class CachedPageProxy:
            def __getattr__(self, name):
                return getattr(page, name)

            def get_pixmap(self, *_args, **_kwargs):
                pytest.fail("cached review render unexpectedly rerendered the PDF page")

        cached = visual_pdf.prepare_dense_cad_review_sheets(
            CachedPageProxy(),
            [candidate],
            desired_width=1200,
            rows_per_sheet=3,
            rendered_page_png=cached_png,
        )
    finally:
        document.close()

    assert cached[0]["content"] == direct[0]["content"]
    assert cached[0]["entries"] == direct[0]["entries"]


def test_dense_cad_outline_covers_share_one_page_content_stream():
    document = fitz.open()
    page = document.new_page(width=240, height=140)
    page.draw_rect(page.rect, color=None, fill=(0, 0, 0))
    before = len(page.get_contents())
    segments = [
        {
            "bbox": (20.0 + index * 60.0, 30.0, 60.0 + index * 60.0, 50.0),
            "metadata": {"rotation": 0, "dense_cad_tight_cover": True},
        }
        for index in range(3)
    ]
    try:
        restored = visual_pdf._cover_outline_text(page, segments, [])
        stream_count = len(page.get_contents())
        pixmap = page.get_pixmap(colorspace=fitz.csRGB, alpha=False)
    finally:
        document.close()

    assert restored == []
    assert stream_count - before == 1
    assert tuple(pixmap.pixel(30, 40)) == (255, 255, 255)
    assert tuple(pixmap.pixel(5, 5)) == (0, 0, 0)


def test_paddle_full_page_candidates_only_add_uncovered_thai_lines(monkeypatch):
    fake_ocr = _FakeOcr(
        {
            "rec_texts": ["ระดับพื้นชั้น 4", "PURE ENGLISH", "ผนัง"],
            "rec_scores": [0.92, 0.99, 0.91],
            "rec_polys": [
                [[20, 20], [130, 20], [130, 40], [20, 40]],
                [[20, 55], [150, 55], [150, 75], [20, 75]],
                [[160, 20], [220, 20], [220, 40], [160, 40]],
            ],
        }
    )
    monkeypatch.setattr(visual_pdf, "_get_paddle_ocr", lambda: fake_ocr)
    monkeypatch.setattr(visual_pdf, "_paddle_ocr_predict_lock", lambda: visual_pdf._OCR_LOCK)
    document = fitz.open(stream=_blank_pdf(width=240, height=140), filetype="pdf")
    try:
        candidates = visual_pdf.detect_dense_cad_paddle_candidates(
            document[0],
            desired_width=240,
            native_units=[],
            existing_bboxes=[(18, 18, 132, 42)],
        )
    finally:
        document.close()

    assert len(candidates) == 1
    assert candidates[0]["source_hint"] == "ผนัง"
    assert candidates[0]["candidate_provider"] == "paddle-full-page"


def test_deferred_paddle_candidate_filter_matches_inline_result(monkeypatch):
    fake_ocr = _FakeOcr(
        {
            "rec_texts": ["ระดับพื้นชั้น 4", "ผนัง"],
            "rec_scores": [0.92, 0.91],
            "rec_polys": [
                [[20, 20], [130, 20], [130, 40], [20, 40]],
                [[160, 20], [220, 20], [220, 40], [160, 40]],
            ],
        }
    )
    monkeypatch.setattr(visual_pdf, "_get_paddle_ocr", lambda: fake_ocr)
    monkeypatch.setattr(visual_pdf, "_paddle_ocr_predict_lock", lambda: visual_pdf._OCR_LOCK)
    document = fitz.open(stream=_blank_pdf(width=240, height=140), filetype="pdf")
    try:
        inline = visual_pdf.detect_dense_cad_paddle_candidates(
            document[0],
            desired_width=240,
            native_units=[],
            existing_bboxes=[(18, 18, 132, 42)],
        )
        # Compare identical OCR input through the deferred-filter route.
        fake_ocr.calls = 0
        raw = visual_pdf.detect_dense_cad_paddle_candidates(
            document[0],
            desired_width=240,
            native_units=[],
            defer_existing_filter=True,
        )
        deferred = visual_pdf.filter_dense_cad_paddle_candidates(
            raw,
            existing_bboxes=[(18, 18, 132, 42)],
            page_rotation=0,
        )
    finally:
        document.close()

    assert deferred == inline


def test_selective_paddle_recognizes_only_new_or_larger_geometry(monkeypatch):
    polygons = [
        [[20, 20], [130, 20], [130, 40], [20, 40]],
        [[160, 20], [220, 20], [220, 40], [160, 40]],
        [[15, 17], [145, 17], [145, 43], [15, 43]],
    ]

    class FakeSelectivePipeline:
        def __init__(self):
            self.recognized_images = []

        @staticmethod
        def get_text_det_params(*_args):
            return {}

        @staticmethod
        def text_det_model(_images, **_kwargs):
            return [{"dt_polys": np.asarray(polygons)}]

        @staticmethod
        def _sort_boxes(values):
            return values

        @staticmethod
        def _crop_by_polys(_image, values):
            return [
                np.zeros(
                    (
                        int(max(point[1] for point in polygon) - min(point[1] for point in polygon)),
                        int(max(point[0] for point in polygon) - min(point[0] for point in polygon)),
                        3,
                    ),
                    dtype=np.uint8,
                )
                for polygon in values
            ]

        def text_rec_model(self, images, *, return_word_box):
            assert return_word_box is False
            self.recognized_images = images
            return [
                {"rec_text": "ผนัง", "rec_score": 0.95},
                {"rec_text": "แปลนหลังคา", "rec_score": 0.96},
            ]

    pipeline = FakeSelectivePipeline()
    fake_ocr = SimpleNamespace(
        paddlex_pipeline=SimpleNamespace(_pipeline=pipeline)
    )
    monkeypatch.setattr(visual_pdf, "_get_paddle_ocr", lambda: fake_ocr)
    monkeypatch.setattr(
        visual_pdf,
        "_paddle_ocr_predict_lock",
        lambda: visual_pdf._OCR_LOCK,
    )
    document = fitz.open(stream=_blank_pdf(width=240, height=140), filetype="pdf")
    try:
        candidates = visual_pdf.detect_dense_cad_paddle_candidates(
            document[0],
            desired_width=240,
            native_units=[],
            existing_bboxes=[(18, 18, 132, 42)],
            defer_existing_filter=True,
            selective_recognition=True,
        )
    finally:
        document.close()

    # The exactly covered first polygon never reaches recognition. The new
    # polygon and the materially larger cover polygon both remain.
    assert len(pipeline.recognized_images) == 2
    assert len(candidates) == 2
    assert {candidate["source_hint"] for candidate in candidates} == {
        "ผนัง",
        "แปลนหลังคา",
    }
    assert sum("matching_existing_index" in candidate for candidate in candidates) == 1


def test_circular_logo_fragments_are_suppressed_below_full_phrase():
    candidates = [
        {
            "value": "มหาวิทยาลัยสงขลานครินทร์",
            "score": 0.91,
            "rect": fitz.Rect(100, 100, 260, 120),
        },
        {
            "value": "าลัยสงขลา",
            "score": 0.79,
            "rect": fitz.Rect(150, 126, 210, 150),
        },
        {
            "value": "เครืน",
            "score": 0.81,
            "rect": fitz.Rect(168, 151, 188, 176),
        },
        {
            "value": "รายละเอียด",
            "score": 0.95,
            "rect": fitz.Rect(300, 126, 380, 145),
        },
    ]

    filtered = visual_pdf._suppress_circular_text_fragments(
        candidates, fitz.Rect(0, 0, 1000, 700)
    )

    assert [item["value"] for item in filtered] == [
        "มหาวิทยาลัยสงขลานครินทร์",
        "รายละเอียด",
    ]


def test_fuzzy_large_bottom_title_reuses_clean_title_block_text():
    candidates = [
        (
            "แปลนพื้นชั้น 1",
            0.99,
            [[3500, 2400], [3630, 2400], [3630, 2425], [3500, 2425]],
            180,
            "bottom-c4",
        ),
        (
            "ลนพืนชน",
            0.55,
            [[2940, 2405], [3200, 2405], [3200, 2465], [2940, 2465]],
            180,
            "bottom-c3",
        ),
    ]

    recovered = visual_pdf._recover_repeated_bottom_titles(candidates, 2826)

    assert recovered[1][0] == "แปลนพื้นชั้น 1"
    assert recovered[1][1] == 0.80
    assert recovered[1][4] == "bottom-title-recovered"


def test_repeated_vector_label_recovers_only_consensus_match():
    import cv2
    import numpy as np

    image = np.full((180, 360, 3), 255, dtype=np.uint8)
    pattern = np.array(
        [
            [0, 0, 255, 255, 0, 0, 255, 255, 0, 0],
            [0, 255, 0, 0, 255, 255, 0, 0, 255, 0],
            [255, 0, 255, 255, 0, 0, 255, 255, 0, 255],
            [255, 0, 0, 0, 255, 255, 0, 0, 0, 255],
            [0, 255, 255, 255, 0, 0, 255, 255, 255, 0],
            [0, 0, 255, 255, 255, 255, 255, 255, 0, 0],
        ],
        dtype=np.uint8,
    )
    for left, top in ((20, 20), (120, 20), (220, 20), (70, 110)):
        image[top : top + 6, left : left + 10] = cv2.cvtColor(
            pattern, cv2.COLOR_GRAY2BGR
        )
    candidates = [
        (
            "ดูแบบขยาย",
            0.95,
            [[left, 20], [left + 10, 20], [left + 10, 26], [left, 26]],
            0,
            "tile",
        )
        for left in (20, 120, 220)
    ]
    candidates.append(
        (
            "กข",
            0.20,
            [[70, 110], [80, 110], [80, 116], [70, 116]],
            0,
            "low-confidence",
        )
    )

    recovered = visual_pdf._recover_repeated_vector_labels(candidates, image)

    assert len(recovered) == 1
    assert recovered[0][0] == "ดูแบบขยาย"
    assert recovered[0][4] == "repeated-template"
    assert fitz.Rect(recovered[0][2][0], recovered[0][2][2]).intersects(
        fitz.Rect(70, 110, 80, 116)
    )


def test_local_paddle_fallback_maps_crop_coordinates_and_discards_non_thai():
    fake_ocr = _FakeOcr(
        {
            "rec_texts": ["W6 ผนัง", "PURE ENGLISH"],
            "rec_scores": [0.91, 0.99],
            "rec_polys": [
                [[20, 30], [220, 30], [220, 70], [20, 70]],
                [[20, 90], [220, 90], [220, 130], [20, 130]],
            ],
        }
    )

    candidates = visual_pdf._local_paddle_candidates(
        fake_ocr,
        np.zeros((200, 300, 3), dtype=np.uint8),
        fitz.Rect(100, 200, 250, 300),
        2.0,
        0,
        "tesseract-local-1",
        rotate=False,
    )

    assert len(candidates) == 1
    assert candidates[0][0] == "W6 ผนัง"
    assert candidates[0][2] == [
        [110.0, 215.0],
        [210.0, 215.0],
        [210.0, 235.0],
        [110.0, 235.0],
    ]


def test_outline_ocr_repairs_wall_schedule_prefix_without_splitting_line():
    assert visual_pdf._normalize_outline_ocr_text(
        "พ6 ผนังก่อคอนกรีตมวลเบา"
    ) == "W6 ผนังก่อคอนกรีตมวลเบา"
    assert visual_pdf._normalize_outline_ocr_text(
        "พ6 ผนังก่อคอนกรีต ทำสีECTRICAL ENGINEERS"
    ) == "W6 ผนังก่อคอนกรีต ทำสี"
    assert visual_pdf._normalize_outline_ocr_text("พ1ผนังก่อคอนกรีต") == "W1ผนังก่อคอนกรีต"
    assert visual_pdf._normalize_outline_ocr_text("WO ผนังดูแบบ") == "W0 ผนังดูแบบ"
    assert visual_pdf._normalize_outline_ocr_text("W6 ผนังทำสีย้อL") == "W6 ผนังทำสีย้อม"
    assert visual_pdf._normalize_outline_ocr_text("พื้นที่ 6 ม.") == "พื้นที่ 6 ม."


def test_fallback_rect_is_clipped_to_largest_structural_column():
    clipped = visual_pdf._clip_fallback_rect_at_barrier(
        fitz.Rect(100, 50, 400, 70),
        0,
        ([(170, 0, 100)], []),
    )

    assert clipped == fitz.Rect(170, 50, 400, 70)


def test_primary_ocr_candidate_beats_longer_tesseract_fallback():
    primary = {
        "value": "W5 ผนังก่อคอนกรีต",
        "score": 0.90,
        "rect": fitz.Rect(0, 0, 100, 10),
        "pass_name": "tile-r1c3",
    }
    fallback = {
        "value": "พ5 ผนังก่อคอนกรีต ELECTRICAL ENGINEER",
        "score": 0.95,
        "rect": fitz.Rect(0, 0, 160, 10),
        "pass_name": "tesseract-local-1",
    }

    assert visual_pdf._ocr_candidate_quality(primary) > visual_pdf._ocr_candidate_quality(
        fallback
    )


def test_misoriented_header_candidate_does_not_suppress_fallback_seed():
    candidate = (
        "พื้น ค.ส.ล.",
        0.9,
        [[0, 0], [200, 0], [200, 20], [0, 20]],
        90,
        "header-c3-rotated",
    )

    assert not visual_pdf._candidate_direction_matches_polygon(candidate)


def test_low_score_primary_candidate_does_not_suppress_fallback_seed():
    candidate = (
        "พ6 ผนังก่อคอนกรีต",
        0.31,
        [[0, 0], [200, 0], [200, 20], [0, 20]],
        0,
        "tile-r1c3",
    )

    assert not visual_pdf._candidate_can_cover_seed(candidate)


def test_low_confidence_tesseract_local_text_is_rejected():
    fake_ocr = _FakeOcr(
        {
            "rec_texts": ["ยลยสงขa"],
            "rec_scores": [0.47],
            "rec_polys": [[[0, 0], [100, 0], [100, 20], [0, 20]]],
        }
    )

    assert visual_pdf._local_paddle_candidates(
        fake_ocr,
        np.zeros((50, 120, 3), dtype=np.uint8),
        fitz.Rect(0, 0, 120, 50),
        1.0,
        0,
        "tesseract-local-1",
        rotate=False,
    ) == []
