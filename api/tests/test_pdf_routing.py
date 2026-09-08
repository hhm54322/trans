import fitz
import pytest

from app.services.native_pdf import _looks_like_native_table
from app.services.pdf_routing import (
    PdfPageRoute,
    freeze_pdf_page_route,
    select_pdf_page_plan,
)


@pytest.mark.parametrize(
    ("profile", "route", "native", "visual"),
    [
        (
            {"page_type": "native_text", "table_like": False},
            PdfPageRoute.NATIVE_TEXT,
            True,
            None,
        ),
        (
            {"page_type": "native_text", "table_like": True},
            PdfPageRoute.NATIVE_TABLE,
            True,
            None,
        ),
        (
            {"page_type": "image"},
            PdfPageRoute.SCANNED_IMAGE,
            False,
            "scan",
        ),
        (
            {"page_type": "vector"},
            PdfPageRoute.DENSE_VECTOR,
            True,
            "cad",
        ),
        (
            {"page_type": "no_native_text"},
            PdfPageRoute.PRESERVE,
            False,
            None,
        ),
    ],
)
def test_pdf_page_routes_are_exclusive(profile, route, native, visual):
    plan = select_pdf_page_plan(profile)

    assert plan.route == route
    assert plan.translate_native is native
    assert plan.visual_strategy == visual


def test_frozen_pdf_route_is_not_reclassified_from_later_profile_changes():
    profile = {"page_type": "native_text", "table_like": True}
    plan = freeze_pdf_page_route(profile)

    profile["page_type"] = "vector"
    profile["drawing_count"] = 999_999

    assert plan.route == PdfPageRoute.NATIVE_TABLE
    assert select_pdf_page_plan(profile).route == PdfPageRoute.NATIVE_TABLE


def test_dense_vector_with_complete_native_text_skips_visual_primary_route():
    plan = select_pdf_page_plan(
        {
            "page_type": "vector",
            "native_text_complete": True,
        }
    )

    assert plan.route == PdfPageRoute.DENSE_VECTOR
    assert plan.translate_native is True
    assert plan.visual_strategy is None


def test_frozen_dense_vector_with_complete_native_text_keeps_native_plan():
    profile = {
        "page_type": "vector",
        "native_text_complete": True,
    }
    freeze_pdf_page_route(profile)

    plan = select_pdf_page_plan(profile)

    assert plan.route == PdfPageRoute.DENSE_VECTOR
    assert plan.translate_native is True
    assert plan.visual_strategy is None


def test_invalid_frozen_pdf_route_fails_instead_of_using_another_branch():
    with pytest.raises(ValueError, match="PDF 页面处理路由无效"):
        select_pdf_page_plan(
            {"page_type": "native_text", "processing_route": "unknown"}
        )


def test_native_table_hint_uses_page_geometry_without_ocr():
    drawings = []
    for row in range(4):
        y = 40.0 + row * 20.0
        drawings.append(
            {"items": [("l", (20.0, y), (380.0, y))]}
        )
    for column in range(3):
        x = 20.0 + column * 180.0
        drawings.append(
            {"items": [("l", (x, 40.0), (x, 100.0))]}
        )

    assert _looks_like_native_table(drawings, fitz.Rect(0, 0, 400, 300))
    assert not _looks_like_native_table(
        [{"items": [("l", (20.0, 40.0), (380.0, 40.0))]}],
        fitz.Rect(0, 0, 400, 300),
    )
