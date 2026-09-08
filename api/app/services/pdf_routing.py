from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class PdfPageRoute(str, Enum):
    """One exclusive processing route selected before page translation."""

    NATIVE_TEXT = "native_text"
    NATIVE_TABLE = "native_table"
    SCANNED_IMAGE = "scanned_image"
    DENSE_VECTOR = "dense_vector"
    PRESERVE = "preserve"


@dataclass(frozen=True)
class PdfPagePlan:
    route: PdfPageRoute
    translate_native: bool
    visual_strategy: Optional[str] = None

    @property
    def translate_visual(self) -> bool:
        return self.visual_strategy is not None


_PAGE_PLANS = {
    PdfPageRoute.NATIVE_TEXT: PdfPagePlan(
        route=PdfPageRoute.NATIVE_TEXT,
        translate_native=True,
    ),
    PdfPageRoute.NATIVE_TABLE: PdfPagePlan(
        route=PdfPageRoute.NATIVE_TABLE,
        translate_native=True,
    ),
    PdfPageRoute.SCANNED_IMAGE: PdfPagePlan(
        route=PdfPageRoute.SCANNED_IMAGE,
        translate_native=False,
        visual_strategy="scan",
    ),
    PdfPageRoute.DENSE_VECTOR: PdfPagePlan(
        route=PdfPageRoute.DENSE_VECTOR,
        translate_native=True,
        visual_strategy="cad",
    ),
    PdfPageRoute.PRESERVE: PdfPagePlan(
        route=PdfPageRoute.PRESERVE,
        translate_native=False,
    ),
}


def select_pdf_page_plan(profile: Dict[str, Any]) -> PdfPagePlan:
    """Resolve a page to one stable route from extraction-only signals."""

    route_value = profile.get("processing_route")
    if route_value:
        try:
            route = PdfPageRoute(str(route_value))
        except (KeyError, ValueError):
            raise ValueError(f"PDF 页面处理路由无效: {route_value}")
        if route == PdfPageRoute.DENSE_VECTOR and profile.get(
            "native_text_complete"
        ):
            return PdfPagePlan(route=route, translate_native=True)
        return _PAGE_PLANS[route]

    page_type = str(profile.get("page_type") or "")
    if page_type in {"vector", "vector_mixed"}:
        route = PdfPageRoute.DENSE_VECTOR
    elif page_type in {"image", "mixed"}:
        route = PdfPageRoute.SCANNED_IMAGE
    elif page_type in {"native_text", "text"}:
        route = (
            PdfPageRoute.NATIVE_TABLE
            if profile.get("table_like")
            else PdfPageRoute.NATIVE_TEXT
        )
    else:
        route = PdfPageRoute.PRESERVE
    if route == PdfPageRoute.DENSE_VECTOR and profile.get("native_text_complete"):
        return PdfPagePlan(route=route, translate_native=True)
    return _PAGE_PLANS[route]


def freeze_pdf_page_route(profile: Dict[str, Any]) -> PdfPagePlan:
    """Select and persist the route so downstream code cannot reclassify it."""

    plan = select_pdf_page_plan(profile)
    profile["processing_route"] = plan.route.value
    return plan
