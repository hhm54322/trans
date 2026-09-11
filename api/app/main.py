import asyncio
import difflib
import csv
import hashlib
import inspect
import json
import logging
import os
import re
import traceback
import fitz
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from functools import partial
from io import StringIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import monotonic
from typing import Callable, Dict, List, Optional
from urllib.parse import quote
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import Database
from .schemas import (
    DocumentJobResponse,
    HealthResponse,
    KnowledgeImportResponse,
    KnowledgeEntryResponse,
    KnowledgeListResponse,
    KnowledgeUpsertRequest,
    TranslationRequest,
    TranslationResponse,
)
from .services.documents import (
    DocumentSegment,
    ParsedDocument,
    ParsedPdfPage,
    iter_pdf_pages,
    native_pdf_profile_warnings,
    parse_document,
    pdf_page_count,
    ocr_image_text_blocks,
    render_pdf_pages,
    render_pdf_tiles,
)
from .services.exports import (
    create_document_export,
    create_unformatted_document_export,
)
from .services.knowledge import parse_knowledge_xlsx
from .services.visual_pdf import (
    INDEXED_CAD_DRAWING_THRESHOLD,
    # Retained as an explicit diagnostic hook and for backwards-compatible
    # test instrumentation. Auto CAD routing no longer invokes it.
    detect_dense_cad_paddle_candidates,
    cad_paddle_worker_count,
    extract_visual_page_units,
    extract_visual_page_units_in_worker,
    filter_dense_cad_paddle_candidates,
    initialize_pdf_ocr_worker,
    prepare_dense_cad_review_sheets,
    prepare_dense_cad_translation_sheets,
    render_dense_cad_page_png,
    recover_dense_cad_review_text_lines,
    recognize_indexed_sheet_rows,
    repack_indexed_translation_sheets,
    requires_deep_table_ocr,
    subset_indexed_translation_sheet,
)
from .services.translator import (
    DemoProvider,
    TranslationResult,
    TranslationService,
    detect_language,
    normalize_translation_text,
)
from .services.native_pdf import prepare_native_pdf_source
from .services.pdf_routing import PdfPageRoute, select_pdf_page_plan

PDF_TEXT_PAGE_CHUNK_SIZE = 5
LAYOUT_SEGMENT_MAX_PAGES = 5
LAYOUT_SEGMENT_MAX_CHARACTERS = 6000
LAYOUT_SEGMENT_MAX_ITEMS = 180
# Dense CAD labels are independent layout units but must retain page-level
# terminology context. Smaller parallel output batches avoid one very long
# structured response without changing any source-reading or write-back rule.
CAD_PARALLEL_TEXT_BATCH_ITEMS = 60
CAD_BALANCED_TEXT_BATCH_ITEMS = 35
CAD_BALANCED_TEXT_BATCH_MAX_PAGE_ITEMS = 120
CAD_PARALLEL_TEXT_BATCH_CHARACTERS = 2000
CAD_PARALLEL_TEXT_BATCH_MIN_ITEMS = 40
CAD_GAP_ROWS_PER_SHEET = 6
# Crops retain a fixed 80px row height after compaction. More candidates in a
# source sheet therefore do not shrink the uncertain rows sent to vision, but
# they do avoid serial model calls for sparse low-confidence candidates.
# Missing IDs still go through the separate 3-row and 1-row review passes.
# A CAD index row is already an enlarged source crop. Combining more than a
# dozen rows forces the vision provider to downsample its input and loses Thai
# glyph detail, so do not trade recognition accuracy for fewer requests here.
# Six enlarged rows preserve Thai glyph detail without forcing the vision
# model to reason across a page-sized index. Direct gateway probes complete
# two independent batches concurrently in about ten seconds, so this remains
# faster than a large, serial image while keeping each line legible.
CAD_PADDLE_DETECTOR_ROWS_PER_SHEET = 12
CAD_INDEXED_IMAGES_PER_REQUEST = max(
    1, min(4, int(os.getenv("APP_CAD_INDEXED_IMAGES_PER_REQUEST", "3")))
)
# Focused review sheets default to four enlarged rows. The deployed gateway
# returned every indexed row in the four-page CAD benchmark while reducing
# requests by roughly one quarter. Five and six rows remain opt-in so larger
# batches must pass the same no-loss production check before becoming default.
# Missing IDs are retried as exact row subsets and a rejected group is split
# recursively before the final independent-sheet fallback.
CAD_REVIEW_IMAGES_PER_REQUEST = max(
    1, min(4, int(os.getenv("APP_CAD_REVIEW_IMAGES_PER_REQUEST", "4")))
)
CAD_REVIEW_ROWS_PER_SHEET = max(
    1, min(6, int(os.getenv("APP_CAD_REVIEW_ROWS_PER_SHEET", "4")))
)
# Indexed sheets retain their original rendering density. A request may carry
# several separate sheets, but no sheet is stitched, resized or recompressed.
# Keep the CAD share independently tunable for the deployed gateway while
# never exceeding the provider-wide request limit.
CAD_INDEXED_MODEL_CONCURRENCY = max(
    1,
    min(
        settings.openai_max_concurrency,
        int(os.getenv("APP_CAD_INDEXED_MODEL_CONCURRENCY", "5")),
    ),
)
CAD_VISUAL_PAGE_CONCURRENCY = max(
    1,
    min(
        CAD_INDEXED_MODEL_CONCURRENCY,
        int(os.getenv("APP_CAD_VISUAL_PAGE_CONCURRENCY", "4")),
    ),
)
CAD_PADDLE_WORKERS = cad_paddle_worker_count()
# This is a per-request guard, not a page-level service target. Gateway queue
# time can exceed 30 seconds for a healthy high-detail image request, so allow
# it to drain before invoking the bounded retry/review path. The provider's
# own HTTP timeout remains the final protection for an unreachable upstream.
CAD_INDEXED_MODEL_TIMEOUT_SECONDS = 120.0
CAD_INDEXED_MODEL_MAX_ATTEMPTS = 3
CAD_INDEXED_HEDGE_DELAY_SECONDS = max(
    0.0,
    float(os.getenv("APP_CAD_INDEXED_HEDGE_DELAY_SECONDS", "25")),
)
CAD_INDEXED_HEDGE_CONCURRENCY = max(
    0,
    min(3, int(os.getenv("APP_CAD_INDEXED_HEDGE_CONCURRENCY", "2"))),
)
TEXT_MODEL_HEDGE_DELAY_SECONDS = max(
    0.0,
    float(os.getenv("APP_TEXT_MODEL_HEDGE_DELAY_SECONDS", "60")),
)
# Text batches are not the current bottleneck. Keep enough overlap for normal
# documents without letting text traffic occupy every provider slot while
# dense CAD visual sheets are in flight.
LAYOUT_TRANSLATION_CONCURRENCY = max(
    1,
    min(
        settings.openai_max_concurrency,
        int(os.getenv("APP_LAYOUT_TRANSLATION_CONCURRENCY", "5")),
    ),
)
DOCUMENT_JOB_TTL_SECONDS = 3600
CLIENT_ID_COOKIE = "metatrans_client_id"
CLIENT_ID_MAX_AGE_SECONDS = 365 * 24 * 60 * 60
KNOWLEDGE_IMPORT_MAX_ROWS = 10000
KNOWLEDGE_CONTEXT_MAX_CHARACTERS = 4000
# Visual processing is selected per page. This keeps plain text PDFs on the
# cheaper native-text path while giving scans and CAD/vector drawings a
# vision pass only where it is useful.
VISUAL_PAGE_TYPES = {"image", "mixed", "vector", "vector_mixed"}

ProgressCallback = Optional[Callable[[int, str], None]]


@dataclass
class DocumentJobState:
    job_id: str
    total_pages: int
    owner_id: str
    diagnostic_id: str = ""
    completed_pages: int = 0
    status: str = "processing"
    stage: str = "queued"
    message: str = "正在准备文档"
    error: Optional[str] = None
    result: Optional[TranslationResponse] = None
    updated_at: float = 0.0


@dataclass
class _SegmentBatchBuilder:
    current_batch: List[DocumentSegment] = field(default_factory=list)
    current_characters: int = 0
    current_pages: set = field(default_factory=set)
    max_items: int = LAYOUT_SEGMENT_MAX_ITEMS
    max_characters: int = LAYOUT_SEGMENT_MAX_CHARACTERS
    max_pages: int = LAYOUT_SEGMENT_MAX_PAGES

    def add(self, segment: DocumentSegment) -> Optional[List[DocumentSegment]]:
        closed_batch = None
        segment_characters = len(segment.text)
        adds_page = segment.page_number not in self.current_pages
        if self.current_batch and (
            len(self.current_batch) >= self.max_items
            or self.current_characters + segment_characters
            > self.max_characters
            or (adds_page and len(self.current_pages) >= self.max_pages)
        ):
            closed_batch = self.flush()
        self.current_batch.append(segment)
        self.current_characters += segment_characters
        self.current_pages.add(segment.page_number)
        return closed_batch

    def flush_page_limit(self) -> Optional[List[DocumentSegment]]:
        if len(self.current_pages) >= self.max_pages:
            return self.flush()
        return None

    def flush(self) -> Optional[List[DocumentSegment]]:
        if not self.current_batch:
            return None
        batch = self.current_batch
        self.current_batch = []
        self.current_characters = 0
        self.current_pages = set()
        return batch


database = Database(settings.database_path)
translator = TranslationService(settings)
logger = logging.getLogger(__name__)
document_jobs: Dict[str, DocumentJobState] = {}
document_job_tasks: Dict[str, asyncio.Task] = {}


class _CadLocalOcrCoordinator:
    """Serialize Tesseract page builds while allowing bounded Paddle overlap."""

    def __init__(self, paddle_limit: int):
        self._condition = asyncio.Condition()
        self._paddle_limit = max(1, int(paddle_limit))
        self._exclusive_active = False
        self._active_paddles = 0

    @asynccontextmanager
    async def exclusive(self):
        async with self._condition:
            await self._condition.wait_for(lambda: not self._exclusive_active)
            self._exclusive_active = True
        try:
            yield
        finally:
            async with self._condition:
                self._exclusive_active = False
                self._condition.notify_all()

    @asynccontextmanager
    async def paddle(self):
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._active_paddles < self._paddle_limit
            )
            self._active_paddles += 1
        try:
            yield
        finally:
            async with self._condition:
                self._active_paddles -= 1
                self._condition.notify_all()


class _CadTextTranslationCache:
    """Coalesce identical CAD text translations within one document job."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._values: Dict[str, str] = {}
        self._pending: Dict[str, asyncio.Future] = {}

    @staticmethod
    def key(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    async def claim(self, segments: List[DocumentSegment]):
        owners = []
        owner_keys = {}
        cached = {}
        waiters = {}
        async with self._lock:
            loop = asyncio.get_running_loop()
            for segment in segments:
                cache_key = self.key(segment.text)
                if cache_key in self._values:
                    cached[segment.segment_id] = self._values[cache_key]
                    continue
                pending = self._pending.get(cache_key)
                if pending is not None:
                    waiters[segment.segment_id] = pending
                    continue
                pending = loop.create_future()
                self._pending[cache_key] = pending
                owners.append(segment)
                owner_keys[segment.segment_id] = cache_key
        return owners, owner_keys, cached, waiters

    async def resolve(self, owner_keys: Dict[str, str], translations: Dict[str, str]):
        async with self._lock:
            missing_segment_ids = [
                segment_id
                for segment_id in owner_keys
                if segment_id not in translations
            ]
            if missing_segment_ids:
                raise RuntimeError(
                    "ID_MISMATCH: CAD 文档缓存缺少已请求的译文"
                )
            for segment_id, cache_key in owner_keys.items():
                translated = translations[segment_id]
                self._values[cache_key] = translated
                pending = self._pending.pop(cache_key, None)
                if pending is not None and not pending.done():
                    pending.set_result(translated)

    async def fail(self, owner_keys: Dict[str, str], error: BaseException):
        async with self._lock:
            for cache_key in owner_keys.values():
                pending = self._pending.pop(cache_key, None)
                if pending is not None and not pending.done():
                    pending.set_exception(error)
                    # Retrieve owner-only failures while preserving the same
                    # exception for any page already awaiting this future.
                    pending.add_done_callback(
                        lambda future: future.exception()
                        if not future.cancelled()
                        else None
                    )


class _CadVisualSourceCache:
    """Coalesce exact CAD glyph reads across concurrent document pages.

    A row is shareable only when the conservative pixel fingerprint and the
    local Thai OCR hint both match.  A failed/no-text owner publishes ``None``
    so every waiting page can run its own normal high-resolution review.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._values: Dict[str, str] = {}
        self._pending: Dict[str, asyncio.Future] = {}

    @property
    def values(self) -> Dict[str, str]:
        return self._values

    def get(self, cache_key: str) -> Optional[str]:
        return self._values.get(cache_key)

    async def claim_sheets(self, sheets: List[Dict]):
        reduced_sheets = []
        cached_rows = []
        duplicate_rows = []
        waiting_rows = []
        owner_rows = []
        selected_sheets = []
        original_count = 0
        sent_count = 0
        seen_keys = set()

        async with self._lock:
            loop = asyncio.get_running_loop()
            for sheet in sheets:
                entries = sheet.get("entries") or {}
                original_count += len(entries)
                selected_ids = []
                for item_id, candidate in entries.items():
                    cache_key = _cad_visual_source_cache_key(candidate)
                    if cache_key:
                        candidate["source_cache_key"] = cache_key
                    cached_source = (
                        self._values.get(cache_key) if cache_key else None
                    )
                    if cached_source:
                        cached_rows.append((candidate, cached_source))
                        continue
                    if cache_key and cache_key in seen_keys:
                        duplicate_rows.append((candidate, cache_key))
                        continue
                    if cache_key:
                        seen_keys.add(cache_key)
                        pending = self._pending.get(cache_key)
                        if pending is not None:
                            waiting_rows.append((candidate, cache_key, pending))
                            continue
                        pending = loop.create_future()
                        self._pending[cache_key] = pending
                        owner_rows.append((candidate, cache_key))
                    selected_ids.append(item_id)
                if not selected_ids:
                    continue
                sent_count += len(selected_ids)
                selected_sheets.append((sheet, selected_ids))

        if sent_count == original_count:
            reduced_sheets = [sheet for sheet, _ids in selected_sheets]
        elif selected_sheets:
            reduced_sheets = await loop.run_in_executor(
                None,
                repack_indexed_translation_sheets,
                selected_sheets,
            )

        owner_task = asyncio.current_task()
        if owner_rows and owner_task is not None:
            # If this page exits before publishing (provider error, job
            # cancellation, etc.), release other pages instead of leaving
            # their exact-match futures pending inside ``asyncio.gather``.
            def release_unpublished_owners(_completed_task):
                asyncio.create_task(self.abandon(owner_rows))

            owner_task.add_done_callback(release_unpublished_owners)

        return (
            reduced_sheets,
            cached_rows,
            duplicate_rows,
            waiting_rows,
            owner_rows,
            {
                "original_count": original_count,
                "sent_count": sent_count,
                "cache_hit_count": len(cached_rows),
                "duplicate_count": len(duplicate_rows),
                "coalesced_count": len(waiting_rows),
                "original_sheet_count": len(sheets),
                "sent_sheet_count": len(reduced_sheets),
            },
        )

    async def resolve(self, owner_rows, recognized_rows):
        recognized_by_key = {}
        for candidate, item in recognized_rows:
            cache_key = _cad_visual_source_cache_key(candidate)
            source_text = str(item.get("source_text") or "").strip()
            if cache_key and re.search(r"[\u0E00-\u0E7F]", source_text):
                recognized_by_key[cache_key] = source_text

        async with self._lock:
            for _candidate, cache_key in owner_rows:
                source_text = recognized_by_key.get(cache_key)
                if source_text:
                    self._values[cache_key] = source_text
                pending = self._pending.pop(cache_key, None)
                if pending is not None and not pending.done():
                    pending.set_result(source_text)

    async def abandon(self, owner_rows):
        """Release peers after an owner fails without caching a decision."""
        async with self._lock:
            for _candidate, cache_key in owner_rows:
                pending = self._pending.pop(cache_key, None)
                if pending is not None and not pending.done():
                    pending.set_result(None)

    async def wait(self, waiting_rows):
        if not waiting_rows:
            return [], []
        values = await asyncio.gather(*(row[2] for row in waiting_rows))
        resolved_rows = []
        unresolved_candidates = []
        for (candidate, _cache_key, _pending), source_text in zip(
            waiting_rows, values
        ):
            if source_text:
                resolved_rows.append((candidate, source_text))
            else:
                unresolved_candidates.append(candidate)
        return resolved_rows, unresolved_candidates

    async def remember(self, recognized_rows):
        async with self._lock:
            for candidate, item in recognized_rows:
                cache_key = _cad_visual_source_cache_key(candidate)
                source_text = str(item.get("source_text") or "").strip()
                if cache_key and re.search(r"[\u0E00-\u0E7F]", source_text):
                    self._values[cache_key] = source_text


_cad_local_ocr_coordinators_by_loop: Dict[int, _CadLocalOcrCoordinator] = {}
# Process-wide workers retain thread-local Paddle models between documents.
# The safe default is one because every additional configured worker keeps a
# separate large predictor resident; capable production servers can opt in.
_cad_paddle_executor = ThreadPoolExecutor(
    max_workers=CAD_PADDLE_WORKERS,
    thread_name_prefix="cad-paddle",
)


def _cad_local_ocr_coordinator_for_current_loop() -> _CadLocalOcrCoordinator:
    loop_id = id(asyncio.get_running_loop())
    coordinator = _cad_local_ocr_coordinators_by_loop.get(loop_id)
    if coordinator is None:
        coordinator = _CadLocalOcrCoordinator(CAD_PADDLE_WORKERS)
        _cad_local_ocr_coordinators_by_loop[loop_id] = coordinator
    return coordinator


def _configure_document_logging() -> None:
    """Keep operational document events after Docker's console log rotates."""
    if any(
        getattr(handler, "_metatrans_document_log", False)
        for handler in logger.handlers
    ):
        return
    try:
        settings.document_log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            settings.document_log_path,
            maxBytes=20 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        handler._metatrans_document_log = True
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    except OSError:
        # Docker stdout remains available even if the mounted data disk is
        # temporarily unavailable. Do not prevent startup for diagnostics alone.
        logger.exception("Unable to configure persistent document diagnostics log")


def _log_document_event(event: str, **fields) -> None:
    """Emit diagnostic metadata without writing document text to production logs."""
    logger.info(
        "document_event=%s fields=%s",
        event,
        json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _configure_document_logging()
    database.initialize()
    try:
        yield
    finally:
        tasks = list(document_job_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await translator.aclose()


app = FastAPI(
    title="META Trans API",
    version="0.3.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalized_client_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return str(UUID(value))
    except (ValueError, AttributeError):
        return None


def _request_owner_id(request: Request) -> str:
    owner_id = getattr(request.state, "owner_id", None)
    if not owner_id:
        raise RuntimeError("请求缺少客户端身份")
    return owner_id


@app.middleware("http")
async def assign_client_identity(request: Request, call_next):
    cookie_id = _normalized_client_id(request.cookies.get(CLIENT_ID_COOKIE))
    owner_id = cookie_id or str(uuid4())
    request.state.owner_id = owner_id
    response = await call_next(request)
    if cookie_id != owner_id:
        forwarded_proto = request.headers.get("x-forwarded-proto", "")
        response.set_cookie(
            CLIENT_ID_COOKIE,
            owner_id,
            max_age=CLIENT_ID_MAX_AGE_SECONDS,
            httponly=True,
            secure=request.url.scheme == "https" or forwarded_proto == "https",
            samesite="lax",
            path="/",
        )
    return response


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    provider = "demo" if isinstance(translator.provider, DemoProvider) else "openai"
    return HealthResponse(status="ok", provider=provider, model=settings.openai_model)


@app.post("/api/translate", response_model=TranslationResponse)
async def translate_text(payload: TranslationRequest, request: Request) -> TranslationResponse:
    try:
        result = await _translate_text_with_knowledge(
            payload.text,
            payload.source_language,
            payload.target_language,
            payload.context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    return _save_result(
        owner_id=_request_owner_id(request),
        kind="text",
        source_text=payload.text,
        target_language=payload.target_language,
        result=result,
    )


@app.post("/api/translate/image", response_model=TranslationResponse)
async def translate_image(
    request: Request,
    file: UploadFile = File(...),
    source_language: str = Form("auto"),
    target_language: str = Form(...),
    context: str = Form(""),
) -> TranslationResponse:
    _validate_languages(source_language, target_language)
    if len(context) > 5000:
        raise HTTPException(status_code=422, detail="本次翻译背景不能超过 5000 个字符")
    media_type = file.content_type or ""
    if media_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=422, detail="暂不支持该图片格式，请上传 JPG、PNG 或 WebP")
    content = await _read_upload(file)
    try:
        source_text, result = await _translate_image_with_knowledge(
            content, media_type, source_language, target_language, context
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return _save_result(
        owner_id=_request_owner_id(request),
        kind="image",
        source_text=source_text,
        target_language=target_language,
        result=result,
        filename=_safe_filename(file.filename),
    )


@app.post("/api/translate/document", response_model=TranslationResponse)
async def translate_document(
    request: Request,
    file: UploadFile = File(...),
    source_language: str = Form("auto"),
    target_language: str = Form(...),
    context: str = Form(""),
) -> TranslationResponse:
    _validate_languages(source_language, target_language)
    if len(context) > 5000:
        raise HTTPException(status_code=422, detail="本次翻译背景不能超过 5000 个字符")
    filename = _safe_filename(file.filename)
    content = await _read_upload(file)
    attempt_id = str(uuid4())
    await _create_document_attempt(
        owner_id=_request_owner_id(request),
        attempt_id=attempt_id,
        filename=filename,
        content_type=file.content_type or "",
        content=content,
        source_language=source_language,
        target_language=target_language,
    )
    try:
        document = parse_document(filename, content, source_language)
        _validate_document(document)
        source_text, result = await _translate_parsed_document(
            content,
            document,
            source_language,
            target_language,
            context,
        )
        response = await _save_document_result(
            owner_id=_request_owner_id(request),
            kind="document",
            source_text=source_text,
            target_language=target_language,
            result=result,
            filename=filename,
            source_content=content,
        )
    except ValueError as exc:
        _record_document_attempt_failure(attempt_id, "parsing_or_translation", exc)
        logger.exception("Document translation request %s failed", attempt_id)
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        _record_document_attempt_failure(attempt_id, "parsing_or_translation", exc)
        logger.exception("Document translation request %s failed", attempt_id)
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        _record_document_attempt_failure(attempt_id, "parsing_or_translation", exc)
        logger.exception("Document translation request %s failed", attempt_id)
        raise
    _update_document_attempt_safely(
        attempt_id,
        status="completed",
        stage="completed",
        history_id=response.id,
        completed=True,
    )
    _log_document_event(
        "document_completed",
        attempt_id=attempt_id,
        filename=filename,
        history_id=response.id,
    )
    return response


@app.post(
    "/api/translate/document/jobs",
    response_model=DocumentJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_document_job(
    request: Request,
    file: UploadFile = File(...),
    source_language: str = Form("auto"),
    target_language: str = Form(...),
    context: str = Form(""),
) -> DocumentJobResponse:
    _validate_languages(source_language, target_language)
    if len(context) > 5000:
        raise HTTPException(status_code=422, detail="本次翻译背景不能超过 5000 个字符")
    filename = _safe_filename(file.filename)
    content = await _read_upload(file)
    job_id = str(uuid4())
    archived = await _create_document_attempt(
        owner_id=_request_owner_id(request),
        attempt_id=job_id,
        filename=filename,
        content_type=file.content_type or "",
        content=content,
        source_language=source_language,
        target_language=target_language,
    )
    document = None
    try:
        if Path(filename).suffix.lower() == ".pdf":
            total_pages = await asyncio.get_running_loop().run_in_executor(
                None, pdf_page_count, content
            )
            completed_pages = 0
        else:
            document = parse_document(filename, content, source_language)
            _validate_document(document)
            total_pages = document.page_count
            completed_pages = _initial_completed_pages(document)
    except ValueError as exc:
        _record_document_attempt_failure(job_id, "preflight", exc)
        logger.exception("Document preflight %s failed", job_id)
        raise HTTPException(status_code=422, detail=str(exc))

    database.update_document_attempt(
        job_id,
        status="processing",
        stage="queued",
        total_pages=total_pages,
        completed_pages=completed_pages,
    )

    _prune_document_jobs()
    job = DocumentJobState(
        job_id=job_id,
        total_pages=total_pages,
        owner_id=_request_owner_id(request),
        diagnostic_id=job_id,
        completed_pages=completed_pages,
        updated_at=monotonic(),
    )
    document_jobs[job_id] = job
    _log_document_event(
        "job_queued",
        job_id=job_id,
        filename=filename,
        file_bytes=len(content),
        page_count=total_pages,
        source_language=source_language,
        target_language=target_language,
        content_sha256=archived["content_sha256"],
    )
    task = asyncio.create_task(
        _run_document_job(
            job,
            content,
            document,
            source_language,
            target_language,
            context,
            filename,
        )
    )
    document_job_tasks[job_id] = task
    task.add_done_callback(lambda _: document_job_tasks.pop(job_id, None))
    return _document_job_response(job)


@app.get(
    "/api/translate/document/jobs/{job_id}",
    response_model=DocumentJobResponse,
)
async def get_document_job(job_id: str, request: Request) -> DocumentJobResponse:
    _prune_document_jobs()
    job = document_jobs.get(job_id)
    if job is None or job.owner_id != _request_owner_id(request):
        raise HTTPException(status_code=404, detail="文档翻译任务不存在或已过期")
    return _document_job_response(job)


async def _run_document_job(
    job: DocumentJobState,
    content: bytes,
    document: Optional[ParsedDocument],
    source_language: str,
    target_language: str,
    context: str,
    filename: str,
) -> None:
    started_at = monotonic()
    job.stage = "translation"
    _update_document_attempt_safely(
        job.diagnostic_id,
        status="processing",
        stage=job.stage,
    )
    _log_document_event(
        "job_started",
        job_id=job.job_id,
        filename=filename,
        total_pages=job.total_pages,
    )

    def report_progress(completed_pages: int, message: str) -> None:
        job.completed_pages = min(
            job.total_pages, job.completed_pages + completed_pages
        )
        job.message = message
        job.updated_at = monotonic()
        if completed_pages or message.startswith("正在流水线"):
            _update_document_attempt_safely(
                job.diagnostic_id,
                status="processing",
                stage=job.stage,
                completed_pages=job.completed_pages,
            )
            _log_document_event(
                "job_progress",
                job_id=job.job_id,
                completed_pages=job.completed_pages,
                total_pages=job.total_pages,
                message=message,
                elapsed_ms=round((job.updated_at - started_at) * 1000),
            )

    try:
        if document is None:
            source_text, result = await _translate_pdf_document_pipeline(
                content,
                filename,
                job.total_pages,
                source_language,
                target_language,
                context,
                report_progress,
            )
        else:
            source_text, result = await _translate_parsed_document(
                content,
                document,
                source_language,
                target_language,
                context,
                report_progress,
            )
        job.message = "正在合并并保存结果"
        job.stage = "export"
        job.updated_at = monotonic()
        _update_document_attempt_safely(
            job.diagnostic_id,
            status="processing",
            stage=job.stage,
            completed_pages=job.completed_pages,
        )
        _log_document_event(
            "job_export_started",
            job_id=job.job_id,
            completed_pages=job.completed_pages,
            elapsed_ms=round((job.updated_at - started_at) * 1000),
        )
        job.result = await _save_document_result(
            owner_id=job.owner_id,
            kind="document",
            source_text=source_text,
            target_language=target_language,
            result=result,
            filename=filename,
            source_content=content,
        )
        job.completed_pages = job.total_pages
        job.status = "completed"
        job.message = "翻译完成"
        job.stage = "completed"
        _update_document_attempt_safely(
            job.diagnostic_id,
            status="completed",
            stage=job.stage,
            completed_pages=job.completed_pages,
            history_id=job.result.id if job.result else None,
            completed=True,
        )
    except asyncio.CancelledError:
        job.stage = "cancelled"
        _update_document_attempt_safely(
            job.diagnostic_id,
            status="cancelled",
            stage=job.stage,
            completed_pages=job.completed_pages,
            completed=True,
        )
        _log_document_event(
            "job_cancelled",
            job_id=job.job_id,
            filename=filename,
            elapsed_ms=round((monotonic() - started_at) * 1000),
        )
        raise
    except (ValueError, RuntimeError) as exc:
        _record_document_attempt_failure(job.diagnostic_id, job.stage, exc)
        _log_document_event(
            "job_failed",
            job_id=job.job_id,
            filename=filename,
            stage=job.stage,
            error_type=type(exc).__name__,
            error_code=str(exc).split(":", 1)[0][:120],
            elapsed_ms=round((monotonic() - started_at) * 1000),
        )
        logger.exception("Document translation job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
        job.message = "翻译失败"
    except Exception as exc:
        _record_document_attempt_failure(job.diagnostic_id, job.stage, exc)
        _log_document_event(
            "job_failed",
            job_id=job.job_id,
            filename=filename,
            stage=job.stage,
            error_type=type(exc).__name__,
            error_code="unhandled",
            elapsed_ms=round((monotonic() - started_at) * 1000),
        )
        logger.exception("Document translation job %s failed", job.job_id)
        job.status = "failed"
        job.error = "文档翻译失败，请稍后重试"
        job.message = "翻译失败"
    finally:
        job.updated_at = monotonic()
        if job.status == "completed":
            _log_document_event(
                "job_completed",
                job_id=job.job_id,
                filename=filename,
                result_id=job.result.id if job.result else None,
                provider=job.result.provider if job.result else None,
                warning_count=len(job.result.warnings) if job.result else 0,
                elapsed_ms=round((job.updated_at - started_at) * 1000),
            )


async def _translate_pdf_document_pipeline(
    content: bytes,
    filename: str,
    page_count: int,
    source_language: str,
    target_language: str,
    context: str,
    progress_callback: ProgressCallback = None,
):
    pipeline_started_at = monotonic()
    loop = asyncio.get_running_loop()
    page_queue = asyncio.Queue()
    text_tasks = []
    ocr_tasks = []
    native_prepare_future = None
    prepared_pdf_content = None
    # Creating the Paddle worker pool eagerly is costly on macOS and used to
    # leave idle worker processes behind for documents that never need deep
    # OCR. Keep it lazy: native CAD pages use Tesseract plus focused vision
    # review, while scans create this pool only when their route needs it.
    deep_ocr_executor = None

    def get_deep_ocr_executor() -> ProcessPoolExecutor:
        nonlocal deep_ocr_executor
        if deep_ocr_executor is None:
            deep_ocr_executor = ProcessPoolExecutor(
                max_workers=max(1, settings.pdf_ocr_concurrency),
                initializer=initialize_pdf_ocr_worker,
                initargs=(content,),
            )
        return deep_ocr_executor

    def produce_pages() -> None:
        def receive_prepared_pdf(prepared: Optional[bytes]) -> None:
            loop.call_soon_threadsafe(
                page_queue.put_nowait, ("prepared", prepared)
            )

        try:
            for parsed_page in _iter_pdf_pages_with_source(
                content,
                source_language,
                receive_prepared_pdf,
            ):
                loop.call_soon_threadsafe(
                    page_queue.put_nowait, ("page", parsed_page)
                )
        except Exception as exc:
            loop.call_soon_threadsafe(page_queue.put_nowait, ("error", exc))
        finally:
            loop.call_soon_threadsafe(page_queue.put_nowait, ("done", None))

    producer = loop.run_in_executor(None, produce_pages)
    pages: List[ParsedPdfPage] = []
    all_segments: List[DocumentSegment] = []
    ocr_pages: List[int] = []
    visual_pages: List[int] = []
    page_types: List[str] = []
    page_profiles: List[Dict] = []
    page_routes: List[str] = []
    buffered_pages: List[ParsedPdfPage] = []
    translations: Dict[str, str] = {}
    providers: List[str] = []
    warnings: List[str] = []
    remaining_by_page: Dict[int, int] = {}
    exact_pages_waiting = set()
    batch_builder = _SegmentBatchBuilder()
    text_semaphore = asyncio.Semaphore(LAYOUT_TRANSLATION_CONCURRENCY)
    cad_vision_semaphore = asyncio.Semaphore(
        min(CAD_INDEXED_MODEL_CONCURRENCY, LAYOUT_TRANSLATION_CONCURRENCY)
    )
    cad_hedge_semaphore = asyncio.Semaphore(CAD_INDEXED_HEDGE_CONCURRENCY)
    # Candidate location is the quality gate. It remains exclusive and takes
    # priority over Paddle so inference cannot starve Tesseract and silently
    # reduce recall. Once no locator is waiting, configured production workers
    # may run independent Paddle models concurrently across pages.
    cad_local_ocr = _cad_local_ocr_coordinator_for_current_loop()
    # This limits concurrent visual-page workflows, not individual OCR calls.
    # Local CAD OCR has the stricter quality gate above, while this still lets
    # text batches and remote CAD sheets overlap across pages.
    visual_page_semaphore = asyncio.Semaphore(
        min(CAD_VISUAL_PAGE_CONCURRENCY, LAYOUT_TRANSLATION_CONCURRENCY)
    )
    # Every dense CAD page that reaches Paddle review needs the same 6400px
    # source render. Build one page at a time while local OCR is running so the
    # expensive render no longer creates a serial gap after detection. The
    # resolution and the downstream crops remain unchanged.
    cad_review_render_semaphore = asyncio.Semaphore(1)
    # These caches live for one uploaded document only. Reuse therefore cannot
    # leak terminology or OCR decisions between customers. Vision reuse also
    # requires a matching glyph fingerprint and local Thai OCR hint.
    cad_visual_source_cache = _CadVisualSourceCache()
    cad_text_translation_cache = _CadTextTranslationCache()
    page_stream_complete = asyncio.Event()
    language_confirmed = asyncio.Event()
    provisional_language = source_language if source_language != "auto" else None
    if provisional_language is not None:
        language_confirmed.set()
    batch_count = 0

    async def translate_text_batch(batch: List[DocumentSegment], batch_index: int):
        batch_started_at = monotonic()
        batch_pages = sorted({segment.page_number for segment in batch})
        _log_document_event(
            "pdf_text_batch_started",
            filename=filename,
            batch_index=batch_index,
            segment_count=len(batch),
            character_count=sum(len(segment.text) for segment in batch),
            pages=batch_pages,
        )
        try:
            batch_result = await _translate_document_segment_batch(
                batch,
                provisional_language,
                target_language,
                context,
                text_semaphore,
            )
        except Exception as exc:
            _log_document_event(
                "pdf_text_batch_failed",
                filename=filename,
                batch_index=batch_index,
                error_type=type(exc).__name__,
                elapsed_ms=round((monotonic() - batch_started_at) * 1000),
            )
            raise
        _log_document_event(
            "pdf_text_batch_completed",
            filename=filename,
            batch_index=batch_index,
            segment_count=len(batch),
            pages=batch_pages,
            elapsed_ms=round((monotonic() - batch_started_at) * 1000),
        )
        await language_confirmed.wait()
        completed_pages = 0
        for segment in batch:
            remaining_by_page[segment.page_number] -= 1
            if (
                remaining_by_page[segment.page_number] == 0
                and segment.page_number not in visual_pages
            ):
                completed_pages += 1
        if progress_callback and completed_pages:
            progress_callback(completed_pages, "正在翻译并保留原排版")
        return batch_result

    def schedule_batch(batch: Optional[List[DocumentSegment]]) -> None:
        nonlocal batch_count
        if not batch:
            return
        batch_count += 1
        text_tasks.append(
            asyncio.create_task(translate_text_batch(batch, batch_count))
        )

    def process_text_page(page: ParsedPdfPage) -> None:
        exact_matches = database.find_exact_knowledge_many(
            provisional_language,
            target_language,
            [segment.text for segment in page.segments],
        )
        remaining_segments = []
        for segment in page.segments:
            exact = exact_matches.get(segment.text)
            if exact:
                translations[segment.segment_id] = normalize_translation_text(
                    exact["translated_text"]
                )
                providers.append("knowledge-base")
            else:
                remaining_segments.append(segment)

        if page.segments and not remaining_segments:
            if language_confirmed.is_set():
                if progress_callback and page.page_number not in visual_pages:
                    progress_callback(1, "正在应用知识库译文")
            elif page.page_number not in visual_pages:
                exact_pages_waiting.add(page.page_number)
        elif remaining_segments:
            remaining_by_page[page.page_number] = len(remaining_segments)

        for segment in remaining_segments:
            schedule_batch(batch_builder.add(segment))
        schedule_batch(batch_builder.flush_page_limit())

    def initialize_provisional_language() -> None:
        nonlocal provisional_language
        if provisional_language is not None:
            return
        buffered_source = "\n".join(
            segment.text for page in buffered_pages for segment in page.segments
        )
        provisional_language = detect_language(buffered_source)
        for buffered_page in buffered_pages:
            process_text_page(buffered_page)
        buffered_pages.clear()

    async def translate_visual_page(
        page_number: int,
        page_type: str,
        page_profile: Dict,
        native_segments: List[DocumentSegment],
        processing_route: str,
    ):
        async with visual_page_semaphore:
            native_units = [asdict(segment) for segment in native_segments]
            visual_source_language = str(
                page_profile.get("source_language") or source_language
            )
            page_plan = select_pdf_page_plan(
                {"processing_route": processing_route}
            )
            # The indexed CAD reader is Thai-specialized. Chinese, English and
            # pages without a native text layer use the generic visual layout
            # reader instead of being forced through Thai OCR rules. For an
            # automatic image/outline page, the vision response supplies the
            # actual source language.
            if visual_source_language in {"auto", "zh", "en"}:
                return await _translate_pdf_layout_pages(
                    content,
                    [page_number],
                    visual_source_language,
                    target_language,
                    context,
                    progress_callback,
                    page_types={page_number: page_type},
                    page_profiles={page_number: page_profile},
                )
            page_rotation = int(page_profile.get("page_rotation") or 0) % 360
            cad_ocr_mode = os.getenv("APP_CAD_OCR_MODE", "auto").strip().lower()
            # Tesseract is the default locator for every CAD page. It is fast,
            # local and filters directly to Thai candidates. Paddle's full-page
            # detector is retained only as an explicit diagnostic mode: it is
            # expensive and rediscovered editable native text on these pages.
            cad_detection_provider = (
                "paddle-detector" if cad_ocr_mode == "paddle" else "tesseract"
            )
            deep_table_page = False
            native_text_complete = bool(page_profile.get("native_text_complete"))
            indexed_cad = (
                page_plan.route == PdfPageRoute.DENSE_VECTOR
                and int(page_profile.get("drawing_count") or 0)
                >= INDEXED_CAD_DRAWING_THRESHOLD
                and cad_ocr_mode in {"auto", "indexed", "paddle"}
                and visual_source_language == "th"
                and not isinstance(translator.provider, DemoProvider)
            )
            deep_fallback_units = None
            deep_fallback_warnings = []

            if indexed_cad:
                # Parsing/profile extraction is CPU-heavy on vector PDFs and
                # used to run beside Tesseract, reducing its recall under load.
                # Native text translation can still stream, but CAD detection
                # begins only once the page stream is fully classified.
                await page_stream_complete.wait()
                if progress_callback:
                    progress_callback(
                        0,
                        f"正在流水线定位 CAD 文字（第 {page_number} 页）",
                    )
                review_page_png_task = None

                async def render_review_page_png_once():
                    async with cad_review_render_semaphore:
                        render_started_at = monotonic()

                        def render_review_page_png():
                            document = fitz.open(stream=content, filetype="pdf")
                            try:
                                return render_dense_cad_page_png(
                                    document[page_number - 1],
                                    desired_width=6400,
                                    minimum_render_scale=2.5,
                                    maximum_render_scale=5.0,
                                )
                            finally:
                                document.close()

                        rendered = await loop.run_in_executor(
                            None, render_review_page_png
                        )
                        _log_document_event(
                            "cad_review_page_render_completed",
                            filename=filename,
                            page_number=page_number,
                            elapsed_ms=round(
                                (monotonic() - render_started_at) * 1000
                            ),
                        )
                        return rendered

                def start_review_page_png_render():
                    nonlocal review_page_png_task
                    if review_page_png_task is None:
                        review_page_png_task = asyncio.create_task(
                            render_review_page_png_once()
                        )
                    return review_page_png_task

                async def get_review_page_png():
                    """Render the exact 6400px review page at most once."""
                    start_review_page_png_render()
                    return await asyncio.shield(review_page_png_task)
                # The full-page Paddle pass needs only the PDF page and native
                # text. Run it beside Tesseract, then apply the unchanged
                # overlap filter once both candidate sets are available.
                def locate_raw_paddle_candidates():
                    document = fitz.open(stream=content, filetype="pdf")
                    try:
                        return detect_dense_cad_paddle_candidates(
                            document[page_number - 1],
                            desired_width=4800,
                            native_units=native_units,
                            existing_bboxes=[],
                            defer_existing_filter=True,
                            selective_recognition=False,
                        )
                    finally:
                        document.close()

                async def locate_raw_paddle_candidates_once():
                    queue_started_at = monotonic()
                    async with cad_local_ocr.paddle():
                        worker_started_at = monotonic()
                        _log_document_event(
                            "cad_paddle_worker_started",
                            filename=filename,
                            page_number=page_number,
                            queue_ms=round(
                                (worker_started_at - queue_started_at) * 1000
                            ),
                            worker_limit=CAD_PADDLE_WORKERS,
                        )
                        candidates = await loop.run_in_executor(
                            _cad_paddle_executor,
                            locate_raw_paddle_candidates,
                        )
                        _log_document_event(
                            "cad_paddle_worker_completed",
                            filename=filename,
                            page_number=page_number,
                            candidate_count=len(candidates),
                            inference_ms=round(
                                (monotonic() - worker_started_at) * 1000
                            ),
                        )
                        return candidates

                paddle_supplement_started_at = monotonic()
                paddle_supplement_task = asyncio.create_task(
                    locate_raw_paddle_candidates_once()
                )
                # Rendering used to begin only after Paddle finished. Start it
                # now and let the single render lane fill otherwise-idle CPU
                # time while OCR and native text translation continue.
                start_review_page_png_render()
                await asyncio.sleep(0)

                def prepare_sheets():
                    document = fitz.open(stream=content, filetype="pdf")
                    try:
                        return prepare_dense_cad_translation_sheets(
                            document[page_number - 1],
                            page_number,
                            desired_width=4000,
                            rows_per_sheet=CAD_PADDLE_DETECTOR_ROWS_PER_SHEET,
                            native_units=native_units,
                            detection_provider=cad_detection_provider,
                            globally_unique_ids=CAD_INDEXED_IMAGES_PER_REQUEST > 1,
                        )
                    finally:
                        document.close()

                locator_started_at = monotonic()
                try:
                    async with cad_local_ocr.exclusive():
                        sheets = await loop.run_in_executor(None, prepare_sheets)
                except BaseException:
                    paddle_supplement_task.cancel()
                    await asyncio.gather(
                        paddle_supplement_task, return_exceptions=True
                    )
                    raise
                indexed_entries = [
                    entry
                    for sheet in sheets
                    for entry in (sheet.get("entries") or {}).values()
                ]
                reliable_entry_count = sum(
                    _cad_source_hint_is_reliable(entry) for entry in indexed_entries
                )
                _log_document_event(
                    "cad_outline_candidates_located",
                    filename=filename,
                    page_number=page_number,
                    detection_provider=cad_detection_provider,
                    native_text_complete=native_text_complete,
                    native_unit_count=len(native_units),
                    candidate_count=len(indexed_entries),
                    text_candidate_count=reliable_entry_count,
                    vision_candidate_count=len(indexed_entries) - reliable_entry_count,
                    elapsed_ms=round((monotonic() - locator_started_at) * 1000),
                )
                if progress_callback:
                    progress_callback(
                        0,
                        f"正在流水线读取 CAD 原文（第 {page_number} 页，"
                        f"{len(indexed_entries)} 处候选）",
                    )

                # Dense CAD is deliberately a two-stage operation. The vision
                # model reads compact source rows only; a text request then
                # translates the confirmed page vocabulary with full context.
                # This avoids asking every vision response to generate a long
                # bilingual JSON payload while retaining stable row IDs.
                indexed_sheets = [
                    sheet for sheet in sheets if sheet.get("entries")
                ]
                # Keep a stable identity through cache subsetting/repacking and
                # high-resolution review. Paddle can then replace one noisy
                # Tesseract crop only after its own focused read succeeds.
                for sheet_index, sheet in enumerate(indexed_sheets):
                    for item_id, candidate in sheet["entries"].items():
                        candidate.setdefault(
                            "origin_item_id", f"{sheet_index}:{item_id}"
                        )
                supplemental_segments = [
                    DocumentSegment(**unit)
                    for sheet in sheets
                    for unit in (sheet.get("supplemental_units") or [])
                ]
                (
                    vision_indexed_sheets,
                    indexed_cached_rows,
                    indexed_duplicate_rows,
                    indexed_waiting_rows,
                    indexed_owner_rows,
                    indexed_cache_stats,
                ) = await cad_visual_source_cache.claim_sheets(
                    indexed_sheets,
                )
                if (
                    indexed_cache_stats["cache_hit_count"]
                    or indexed_cache_stats["duplicate_count"]
                    or indexed_cache_stats["coalesced_count"]
                ):
                    _log_document_event(
                        "cad_visual_source_cache_reused",
                        filename=filename,
                        page_number=page_number,
                        stage="CAD 原文识别",
                        **indexed_cache_stats,
                    )

                async def request_indexed_with_hedge(
                    request_factory,
                    *,
                    group_number,
                    image_count,
                ):
                    """Run one indexed read with one bounded tail-latency hedge."""

                    async def request_once(
                        started_event=None,
                        *,
                        owns_provider_slot=False,
                        bypass_cad_slot=False,
                    ):
                        if not owns_provider_slot and not bypass_cad_slot:
                            await cad_vision_semaphore.acquire()
                        try:
                            if started_event is not None:
                                started_event.set()
                            return await asyncio.wait_for(
                                request_factory(),
                                timeout=CAD_INDEXED_MODEL_TIMEOUT_SECONDS,
                            )
                        finally:
                            if not bypass_cad_slot:
                                cad_vision_semaphore.release()

                    primary_started = asyncio.Event()
                    primary = asyncio.create_task(request_once(primary_started))
                    tasks = {primary}
                    hedge_slot = None
                    try:
                        if CAD_INDEXED_HEDGE_DELAY_SECONDS > 0:
                            # Model latency begins only after the request owns a
                            # provider slot. Queue time never creates duplicates.
                            await primary_started.wait()
                            done, _pending = await asyncio.wait(
                                tasks,
                                timeout=CAD_INDEXED_HEDGE_DELAY_SECONDS,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if not done:
                                hedge_mode = "queued"
                                if not cad_hedge_semaphore.locked():
                                    await cad_hedge_semaphore.acquire()
                                    if primary.done():
                                        cad_hedge_semaphore.release()
                                    else:
                                        async def request_reserved_hedge():
                                            try:
                                                return await request_once(
                                                    bypass_cad_slot=True
                                                )
                                            finally:
                                                cad_hedge_semaphore.release()

                                        hedge = asyncio.create_task(
                                            request_reserved_hedge()
                                        )
                                        tasks.add(hedge)
                                        hedge_mode = "reserved"
                                        _log_document_event(
                                            "cad_indexed_vision_hedge_started",
                                            filename=filename,
                                            page_number=page_number,
                                            group_number=group_number,
                                            image_count=image_count,
                                            mode=hedge_mode,
                                        )
                                elif CAD_INDEXED_HEDGE_CONCURRENCY <= 0:
                                    # An explicit zero keeps the previous FIFO
                                    # hedge behavior for conservative rollout.
                                    hedge_slot = asyncio.create_task(
                                        cad_vision_semaphore.acquire()
                                    )
                                    slot_done, _slot_pending = await asyncio.wait(
                                        {primary, hedge_slot},
                                        return_when=asyncio.FIRST_COMPLETED,
                                    )
                                    if hedge_slot in slot_done:
                                        hedge_slot.result()
                                        if primary.done():
                                            cad_vision_semaphore.release()
                                            hedge_slot = None
                                        else:
                                            hedge = asyncio.create_task(
                                                request_once(
                                                    owns_provider_slot=True
                                                )
                                            )
                                            tasks.add(hedge)
                                            hedge_slot = None
                                            _log_document_event(
                                                "cad_indexed_vision_hedge_started",
                                                filename=filename,
                                                page_number=page_number,
                                                group_number=group_number,
                                                image_count=image_count,
                                                mode=hedge_mode,
                                            )
                        last_error = None
                        while tasks:
                            done, tasks = await asyncio.wait(
                                tasks,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            for completed in done:
                                try:
                                    result = completed.result()
                                except (RuntimeError, asyncio.TimeoutError) as exc:
                                    last_error = exc
                                    continue
                                if len(done) > 1 or tasks:
                                    _log_document_event(
                                        "cad_indexed_vision_hedge_won",
                                        filename=filename,
                                        page_number=page_number,
                                        group_number=group_number,
                                    )
                                return result
                        if last_error is not None:
                            raise last_error
                        raise RuntimeError("CAD 视觉请求未返回结果")
                    finally:
                        if hedge_slot is not None:
                            if not hedge_slot.done():
                                hedge_slot.cancel()
                                await asyncio.gather(
                                    hedge_slot, return_exceptions=True
                                )
                            elif (
                                not hedge_slot.cancelled()
                                and hedge_slot.exception() is None
                            ):
                                cad_vision_semaphore.release()
                        for task in tasks:
                            task.cancel()
                        if tasks:
                            await asyncio.gather(*tasks, return_exceptions=True)

                async def read_indexed_sheet(
                    sheet_number,
                    sheet,
                    *,
                    require_complete=True,
                    stage="CAD 原文识别",
                ):
                    sheet["sheet_number"] = sheet_number
                    expected_ids = list(sheet["entries"])
                    accepted_items = {}
                    routes = []
                    pending_ids = list(expected_ids)
                    pending_sheet = sheet
                    last_error = None
                    maximum_attempts = (
                        CAD_INDEXED_MODEL_MAX_ATTEMPTS if require_complete else 1
                    )
                    for attempt in range(maximum_attempts):
                        expected_sources = {
                            item_id: str(
                                pending_sheet["entries"][item_id].get(
                                    "source_hint"
                                )
                                or ""
                            )
                            for item_id in pending_ids
                            if str(
                                pending_sheet["entries"][item_id].get(
                                    "source_hint"
                                )
                                or ""
                            ).strip()
                        }
                        try:
                            items, route = await request_indexed_with_hedge(
                                lambda: translator.read_indexed_image_lines(
                                    pending_sheet["content"],
                                    "image/png",
                                    pending_ids,
                                    context,
                                    expected_sources=expected_sources,
                                    require_complete=False,
                                ),
                                group_number=sheet_number,
                                image_count=1,
                            )
                            if route:
                                routes.append(route)
                            accepted_items.update(
                                {item["id"]: item for item in items}
                            )
                            pending_ids = [
                                item_id
                                for item_id in expected_ids
                                if item_id not in accepted_items
                            ]
                            if not pending_ids:
                                return (
                                    sheet,
                                    [accepted_items[item_id] for item_id in expected_ids],
                                    "+".join(dict.fromkeys(routes)),
                                )
                            last_error = RuntimeError(
                                "ID_MISMATCH: CAD 索引图原文识别不完整"
                            )
                        except (asyncio.TimeoutError, RuntimeError) as exc:
                            last_error = exc
                        if attempt + 1 < maximum_attempts:
                            # Retry exactly the unresolved source rows. Their
                            # pixels are copied from the initial high-density
                            # sheet, so this removes completed work without
                            # trading away any recognition detail.
                            pending_sheet = sheet
                            if len(pending_ids) != len(expected_ids):
                                pending_sheet = await loop.run_in_executor(
                                    None,
                                    subset_indexed_translation_sheet,
                                    sheet,
                                    pending_ids,
                                )
                            if not pending_sheet.get("entries"):
                                break
                    sheet["unresolved_ids"] = list(pending_ids)
                    if not require_complete:
                        return (
                            sheet,
                            [
                                accepted_items[item_id]
                                for item_id in expected_ids
                                if item_id in accepted_items
                            ],
                            "+".join(dict.fromkeys(routes)),
                        )
                    raise RuntimeError(
                        f"ID_MISMATCH: 第 {page_number} 页 {stage}第 "
                        f"{sheet_number} 批失败："
                        f"{last_error}"
                    ) from last_error

                # Paddle runs once per CAD page as a leak detector. New rows
                # and rows matching a Tesseract candidate receive a focused
                # high-resolution visual source read before they are accepted.
                existing_refs = [
                    (sheet, item_id)
                    for sheet in indexed_sheets
                    for item_id in sheet["entries"]
                ]
                existing_bboxes = [
                    sheet["entries"][item_id]["bbox"]
                    for sheet, item_id in existing_refs
                ]
                # Paddle was started before Tesseract sheet preparation. Its
                # raw result is still filtered against every indexed box here,
                # preserving the same candidate recall and merge rules.

                async def read_indexed_sheet_group(
                    group_number,
                    numbered_sheets,
                    *,
                    require_complete,
                    stage="CAD 原文识别",
                    emit_completion=True,
                ):
                    group_started_at = monotonic()
                    if len(numbered_sheets) == 1:
                        sheet_number, sheet = numbered_sheets[0]
                        single_output = [
                            await read_indexed_sheet(
                                sheet_number,
                                sheet,
                                require_complete=require_complete,
                                stage=stage,
                            )
                        ]
                        if emit_completion:
                            _log_document_event(
                                "cad_indexed_vision_group_completed",
                                filename=filename,
                                page_number=page_number,
                                group_number=group_number,
                                stage=stage,
                                image_count=1,
                                expected_count=len(sheet["entries"]),
                                returned_count=len(single_output[0][1]),
                                fallback=False,
                                elapsed_ms=round(
                                    (monotonic() - group_started_at) * 1000
                                ),
                            )
                        return single_output
                    expected_ids = [
                        item_id
                        for _sheet_number, sheet in numbered_sheets
                        for item_id in sheet["entries"]
                    ]
                    expected_sources = {
                        item_id: str(candidate.get("source_hint") or "")
                        for _sheet_number, sheet in numbered_sheets
                        for item_id, candidate in sheet["entries"].items()
                        if str(candidate.get("source_hint") or "").strip()
                    }

                    try:
                        items, route = await request_indexed_with_hedge(
                            lambda: translator.read_indexed_image_line_group(
                                [
                                    sheet["content"]
                                    for _sheet_number, sheet in numbered_sheets
                                ],
                                "image/png",
                                expected_ids,
                                context,
                                expected_sources=expected_sources,
                                # Preserve a valid partial response so a
                                # strict review can retry only omitted rows.
                                # Unknown or duplicate IDs still fail in the
                                # service and take the safe fallback.
                                require_complete=False,
                            ),
                            group_number=group_number,
                            image_count=len(numbered_sheets),
                        )
                        by_id = {item["id"]: item for item in items}
                        routes = [route] if route else []
                        if require_complete:
                            retry_sheets = []
                            for sheet_number, sheet in numbered_sheets:
                                missing_ids = [
                                    item_id
                                    for item_id in sheet["entries"]
                                    if item_id not in by_id
                                ]
                                if not missing_ids:
                                    continue
                                retry_sheet = await loop.run_in_executor(
                                    None,
                                    subset_indexed_translation_sheet,
                                    sheet,
                                    missing_ids,
                                )
                                retry_sheets.append((sheet_number, retry_sheet))
                            retry_results = await asyncio.gather(
                                *(
                                    read_indexed_sheet(
                                        sheet_number,
                                        retry_sheet,
                                        require_complete=True,
                                        stage=stage,
                                    )
                                    for sheet_number, retry_sheet in retry_sheets
                                )
                            )
                            for _retry_sheet, retry_items, retry_route in retry_results:
                                by_id.update(
                                    {item["id"]: item for item in retry_items}
                                )
                                if retry_route:
                                    routes.append(retry_route)
                        combined_route = "+".join(dict.fromkeys(routes))
                        output = []
                        for sheet_number, sheet in numbered_sheets:
                            sheet_ids = list(sheet["entries"])
                            sheet_items = [
                                by_id[item_id]
                                for item_id in sheet_ids
                                if item_id in by_id
                            ]
                            sheet["sheet_number"] = sheet_number
                            sheet["unresolved_ids"] = [
                                item_id
                                for item_id in sheet_ids
                                if item_id not in by_id
                            ]
                            output.append((sheet, sheet_items, combined_route))
                        if emit_completion:
                            _log_document_event(
                                "cad_indexed_vision_group_completed",
                                filename=filename,
                                page_number=page_number,
                                group_number=group_number,
                                stage=stage,
                                image_count=len(numbered_sheets),
                                expected_count=len(expected_ids),
                                returned_count=sum(
                                    len(sheet_items)
                                    for _, sheet_items, _ in output
                                ),
                                fallback=False,
                                elapsed_ms=round(
                                    (monotonic() - group_started_at) * 1000
                                ),
                            )
                        return output
                    except (asyncio.TimeoutError, RuntimeError) as exc:
                        if emit_completion:
                            _log_document_event(
                                "cad_multi_image_group_fallback",
                                filename=filename,
                                page_number=page_number,
                                group_number=group_number,
                                image_count=len(numbered_sheets),
                                error_type=type(exc).__name__,
                            )
                        if len(numbered_sheets) > 2:
                            # A timed-out six-image request usually succeeds as
                            # two three-image requests. Split recursively before
                            # falling all the way back to one request per sheet;
                            # source pixels and completeness checks are unchanged.
                            midpoint = (len(numbered_sheets) + 1) // 2
                            fallback_groups = await asyncio.gather(
                                read_indexed_sheet_group(
                                    group_number,
                                    numbered_sheets[:midpoint],
                                    require_complete=require_complete,
                                    stage=stage,
                                    emit_completion=False,
                                ),
                                read_indexed_sheet_group(
                                    group_number,
                                    numbered_sheets[midpoint:],
                                    require_complete=require_complete,
                                    stage=stage,
                                    emit_completion=False,
                                )
                            )
                            fallback_output = [
                                result
                                for fallback_group in fallback_groups
                                for result in fallback_group
                            ]
                        else:
                            fallback_output = await asyncio.gather(
                                *(
                                    read_indexed_sheet(
                                        sheet_number,
                                        sheet,
                                        require_complete=require_complete,
                                        stage=stage,
                                    )
                                    for sheet_number, sheet in numbered_sheets
                                )
                            )
                        if emit_completion:
                            _log_document_event(
                                "cad_indexed_vision_group_completed",
                                filename=filename,
                                page_number=page_number,
                                group_number=group_number,
                                stage=stage,
                                image_count=len(numbered_sheets),
                                expected_count=len(expected_ids),
                                returned_count=sum(
                                    len(sheet_items)
                                    for _, sheet_items, _ in fallback_output
                                ),
                                fallback=True,
                                elapsed_ms=round(
                                    (monotonic() - group_started_at) * 1000
                                ),
                            )
                        return fallback_output

                numbered_sheets = list(
                    enumerate(vision_indexed_sheets, start=1)
                )
                grouped_sheets = [
                    numbered_sheets[index : index + CAD_INDEXED_IMAGES_PER_REQUEST]
                    for index in range(
                        0,
                        len(numbered_sheets),
                        CAD_INDEXED_IMAGES_PER_REQUEST,
                    )
                ]
                source_result_groups = await asyncio.gather(
                    *(
                        read_indexed_sheet_group(
                            group_number,
                            group,
                            require_complete=False,
                        )
                        for group_number, group in enumerate(grouped_sheets, start=1)
                    )
                )
                source_results = [
                    result
                    for group_results in source_result_groups
                    for result in group_results
                ]
                recognized_rows = []
                reader_providers = []
                unresolved_for_review = []
                preserved_tiny_label_candidates = 0
                for sheet, items, route in source_results:
                    if route:
                        reader_providers.append(route)
                    for item in items:
                        candidate = sheet["entries"][item["id"]]
                        if _cad_indexed_read_needs_review(candidate, item):
                            candidate = dict(candidate)
                            candidate["origin_item_id"] = item["id"]
                            unresolved_for_review.append(candidate)
                            continue
                        recognized_rows.append((candidate, item))
                    for item_id in sheet.get("unresolved_ids") or []:
                        candidate = dict(sheet["entries"][item_id])
                        if _cad_candidate_is_tiny_unreadable_label(candidate):
                            preserved_tiny_label_candidates += 1
                            continue
                        candidate["origin_item_id"] = item_id
                        unresolved_for_review.append(candidate)

                # A duplicate is reused only after its representative produced
                # confirmed Thai source text. If that representative needs a
                # high-resolution retry, send the duplicate through the same
                # conservative review path instead of silently dropping it.
                await cad_visual_source_cache.remember(recognized_rows)

                async def review_cad_candidates(candidates):
                    if not candidates:
                        return []
                    review_page_png = await get_review_page_png()
                    review_prepare_started_at = monotonic()
                    _log_document_event(
                        "cad_review_sheets_prepare_started",
                        filename=filename,
                        page_number=page_number,
                        stage="CAD 高清复核",
                        candidate_count=len(candidates),
                    )

                    def prepare_unresolved_review_sheets():
                        document = fitz.open(stream=content, filetype="pdf")
                        try:
                            return prepare_dense_cad_review_sheets(
                                document[page_number - 1],
                                candidates,
                                desired_width=6400,
                                rows_per_sheet=CAD_REVIEW_ROWS_PER_SHEET,
                                rendered_page_png=review_page_png,
                            )
                        finally:
                            document.close()

                    unresolved_sheets = await loop.run_in_executor(
                        None, prepare_unresolved_review_sheets
                    )
                    _log_document_event(
                        "cad_review_sheets_prepare_completed",
                        filename=filename,
                        page_number=page_number,
                        stage="CAD 高清复核",
                        candidate_count=len(candidates),
                        sheet_count=len(unresolved_sheets),
                        elapsed_ms=round(
                            (monotonic() - review_prepare_started_at) * 1000
                        ),
                    )
                    numbered_unresolved_sheets = [
                        (sheet_number, sheet)
                        for sheet_number, sheet in enumerate(
                            unresolved_sheets, start=1
                        )
                        if sheet.get("entries")
                    ]
                    unresolved_groups = [
                        numbered_unresolved_sheets[
                            index : index + CAD_REVIEW_IMAGES_PER_REQUEST
                        ]
                        for index in range(
                            0,
                            len(numbered_unresolved_sheets),
                            CAD_REVIEW_IMAGES_PER_REQUEST,
                        )
                    ]
                    unresolved_group_results = await asyncio.gather(
                        *(
                            read_indexed_sheet_group(
                                group_number,
                                group,
                                require_complete=True,
                                stage="CAD 高清复核",
                            )
                            for group_number, group in enumerate(
                                unresolved_groups, start=1
                            )
                        )
                    )
                    unresolved_results = [
                        result
                        for group_results in unresolved_group_results
                        for result in group_results
                    ]
                    reviewed_rows = []
                    for sheet, items, route in unresolved_results:
                        if route:
                            reader_providers.append(route)
                        for item in items:
                            reviewed_rows.append(
                                (sheet["entries"][item["id"]], item)
                            )
                    return reviewed_rows

                recognized_rows.extend(
                    await review_cad_candidates(unresolved_for_review)
                )
                await cad_visual_source_cache.resolve(
                    indexed_owner_rows,
                    recognized_rows,
                )
                (
                    indexed_coalesced_rows,
                    indexed_unresolved_waiters,
                ) = await cad_visual_source_cache.wait(indexed_waiting_rows)
                if indexed_unresolved_waiters:
                    waiter_review_rows = await review_cad_candidates(
                        indexed_unresolved_waiters
                    )
                    recognized_rows.extend(waiter_review_rows)
                    await cad_visual_source_cache.remember(waiter_review_rows)
                indexed_cached_rows.extend(indexed_coalesced_rows)

                resolved_indexed_duplicates = []
                unresolved_indexed_duplicates = []
                for candidate, cache_key in indexed_duplicate_rows:
                    if cad_visual_source_cache.get(cache_key):
                        resolved_indexed_duplicates.append((candidate, cache_key))
                    else:
                        unresolved_indexed_duplicates.append(
                            (dict(candidate), cache_key)
                        )
                if unresolved_indexed_duplicates:
                    duplicate_review_rows = await review_cad_candidates(
                        [candidate for candidate, _ in unresolved_indexed_duplicates]
                    )
                    recognized_rows.extend(duplicate_review_rows)
                    await cad_visual_source_cache.remember(duplicate_review_rows)
                indexed_duplicate_rows = resolved_indexed_duplicates

                recognized_rows = _commit_and_expand_cad_source_cache(
                    recognized_rows,
                    indexed_cached_rows,
                    indexed_duplicate_rows,
                    cad_visual_source_cache.values,
                )

                if progress_callback:
                    progress_callback(
                        0,
                        f"正在流水线复核 CAD 漏检文字（第 {page_number} 页）",
                    )
                paddle_wait_started_at = monotonic()
                raw_paddle_candidates = await paddle_supplement_task
                _log_document_event(
                    "cad_paddle_supplement_completed",
                    filename=filename,
                    page_number=page_number,
                    candidate_count=len(raw_paddle_candidates),
                    wait_ms=round(
                        (monotonic() - paddle_wait_started_at) * 1000
                    ),
                    elapsed_ms=round(
                        (monotonic() - paddle_supplement_started_at) * 1000
                    ),
                )
                paddle_candidates = filter_dense_cad_paddle_candidates(
                    raw_paddle_candidates,
                    existing_bboxes=existing_bboxes,
                    page_rotation=page_rotation,
                    include_existing_matches=True,
                )
                paddle_only_candidates = []
                paddle_replacement_candidate_count = 0
                paddle_replaced_count = 0
                paddle_added_count = 0
                for candidate in paddle_candidates:
                    matching_index = candidate.get("matching_existing_index")
                    if matching_index is None:
                        paddle_only_candidates.append(candidate)
                        continue
                    _source_sheet, source_item_id = existing_refs[
                        int(matching_index)
                    ]
                    source_candidate = _source_sheet["entries"][source_item_id]
                    candidate["replaces_origin_item_id"] = source_candidate[
                        "origin_item_id"
                    ]
                    candidate["cover_bbox"] = tuple(candidate["bbox"])
                    paddle_only_candidates.append(candidate)
                    paddle_replacement_candidate_count += 1

                if paddle_only_candidates:
                    review_page_png = await get_review_page_png()
                    paddle_prepare_started_at = monotonic()
                    _log_document_event(
                        "cad_review_sheets_prepare_started",
                        filename=filename,
                        page_number=page_number,
                        stage="CAD Paddle 补漏",
                        candidate_count=len(paddle_only_candidates),
                    )

                    def prepare_paddle_review_sheets():
                        document = fitz.open(stream=content, filetype="pdf")
                        try:
                            return prepare_dense_cad_review_sheets(
                                document[page_number - 1],
                                paddle_only_candidates,
                                desired_width=6400,
                                rows_per_sheet=CAD_REVIEW_ROWS_PER_SHEET,
                                rendered_page_png=review_page_png,
                            )
                        finally:
                            document.close()

                    paddle_sheets = await loop.run_in_executor(
                        None, prepare_paddle_review_sheets
                    )
                    _log_document_event(
                        "cad_review_sheets_prepare_completed",
                        filename=filename,
                        page_number=page_number,
                        stage="CAD Paddle 补漏",
                        candidate_count=len(paddle_only_candidates),
                        sheet_count=len(paddle_sheets),
                        elapsed_ms=round(
                            (monotonic() - paddle_prepare_started_at) * 1000
                        ),
                    )
                    (
                        vision_paddle_sheets,
                        paddle_cached_rows,
                        paddle_duplicate_rows,
                        paddle_waiting_rows,
                        paddle_owner_rows,
                        paddle_cache_stats,
                    ) = await cad_visual_source_cache.claim_sheets(
                        [sheet for sheet in paddle_sheets if sheet.get("entries")],
                    )
                    if (
                        paddle_cache_stats["cache_hit_count"]
                        or paddle_cache_stats["duplicate_count"]
                        or paddle_cache_stats["coalesced_count"]
                    ):
                        _log_document_event(
                            "cad_visual_source_cache_reused",
                            filename=filename,
                            page_number=page_number,
                            stage="CAD Paddle 补漏",
                            **paddle_cache_stats,
                        )
                    numbered_paddle_sheets = [
                        (sheet_number, sheet)
                        for sheet_number, sheet in enumerate(
                            vision_paddle_sheets, start=1
                        )
                        if sheet.get("entries")
                    ]
                    paddle_groups = [
                        numbered_paddle_sheets[
                            index : index + CAD_REVIEW_IMAGES_PER_REQUEST
                        ]
                        for index in range(
                            0,
                            len(numbered_paddle_sheets),
                            CAD_REVIEW_IMAGES_PER_REQUEST,
                        )
                    ]
                    paddle_group_results = await asyncio.gather(
                        *(
                            read_indexed_sheet_group(
                                group_number,
                                group,
                                require_complete=True,
                            )
                            for group_number, group in enumerate(
                                paddle_groups, start=1
                            )
                        )
                    )
                    paddle_results = [
                        result
                        for group_results in paddle_group_results
                        for result in group_results
                    ]
                    paddle_recognized_rows = []
                    for sheet, items, route in paddle_results:
                        if route:
                            reader_providers.append(route)
                        for item in items:
                            paddle_recognized_rows.append(
                                (sheet["entries"][item["id"]], item)
                            )
                    await cad_visual_source_cache.resolve(
                        paddle_owner_rows,
                        paddle_recognized_rows,
                    )
                    (
                        paddle_coalesced_rows,
                        paddle_unresolved_waiters,
                    ) = await cad_visual_source_cache.wait(paddle_waiting_rows)
                    if paddle_unresolved_waiters:
                        waiter_review_rows = await review_cad_candidates(
                            paddle_unresolved_waiters
                        )
                        paddle_recognized_rows.extend(waiter_review_rows)
                        await cad_visual_source_cache.remember(waiter_review_rows)
                    paddle_cached_rows.extend(paddle_coalesced_rows)
                    merged_paddle_rows = _commit_and_expand_cad_source_cache(
                        paddle_recognized_rows,
                        paddle_cached_rows,
                        paddle_duplicate_rows,
                        cad_visual_source_cache.values,
                    )
                    replaced_origin_ids = {
                        str(candidate.get("replaces_origin_item_id"))
                        for candidate, item in merged_paddle_rows
                        if candidate.get("replaces_origin_item_id")
                        and re.search(
                            r"[\u0E00-\u0E7F]",
                            str(item.get("source_text") or ""),
                        )
                    }
                    paddle_replaced_count = len(replaced_origin_ids)
                    paddle_added_count = sum(
                        1
                        for candidate, item in merged_paddle_rows
                        if not candidate.get("replaces_origin_item_id")
                        and re.search(
                            r"[\u0E00-\u0E7F]",
                            str(item.get("source_text") or ""),
                        )
                    )
                    if replaced_origin_ids:
                        recognized_rows = [
                            (candidate, item)
                            for candidate, item in recognized_rows
                            if str(candidate.get("origin_item_id") or "")
                            not in replaced_origin_ids
                        ]
                        _log_document_event(
                            "cad_paddle_existing_candidate_replaced",
                            filename=filename,
                            page_number=page_number,
                            candidate_count=len(replaced_origin_ids),
                            attempted_count=paddle_replacement_candidate_count,
                        )
                    recognized_rows.extend(merged_paddle_rows)

                unique_segments = []
                translation_segment_by_source = {}
                for candidate, item in recognized_rows:
                    source_value = str(item.get("source_text") or "").strip()
                    if not re.search(r"[\u0E00-\u0E7F]", source_value):
                        continue
                    source_key = re.sub(r"\s+", " ", source_value)
                    if source_key in translation_segment_by_source:
                        continue
                    segment = DocumentSegment(
                        segment_id=(
                            f"cad:p{page_number}:source:"
                            f"{len(unique_segments) + 1:03d}"
                        ),
                        page_number=page_number,
                        text=source_value,
                        source_kind="outline-text",
                        bbox=tuple(candidate["bbox"]),
                        metadata={"ocr_provider": "gpt-indexed-source"},
                    )
                    unique_segments.append(segment)
                    translation_segment_by_source[source_key] = segment.segment_id

                translation_segments = [*unique_segments, *supplemental_segments]
                if not translation_segments:
                    if progress_callback:
                        progress_callback(1, "正在检查页面视觉文字")
                    empty_warnings = []
                    if preserved_tiny_label_candidates:
                        empty_warnings.append(
                            f"第 {page_number} 页保留 "
                            f"{preserved_tiny_label_candidates} 个"
                            "多工具未确认的极小 CAD 标签原文"
                        )
                    return [], empty_warnings

                if progress_callback:
                    progress_callback(
                        0,
                        f"正在流水线翻译并回写 CAD 文字（第 {page_number} 页）",
                    )
                (
                    pending_translation_segments,
                    translation_cache_keys,
                    cached_text_translations,
                    waiting_text_translations,
                ) = await cad_text_translation_cache.claim(translation_segments)
                if pending_translation_segments:
                    try:
                        text_translations, text_providers, text_warnings = (
                            # Adaptive CAD batches all receive the complete page
                            # source context. Concurrent identical source strings
                            # are coalesced into the first owning page request.
                            await _translate_document_segments(
                                pending_translation_segments,
                                "th",
                                target_language,
                                context,
                            )
                        )
                        await cad_text_translation_cache.resolve(
                            translation_cache_keys,
                            text_translations,
                        )
                    except BaseException as exc:
                        await cad_text_translation_cache.fail(
                            translation_cache_keys,
                            exc,
                        )
                        raise
                else:
                    text_translations, text_providers, text_warnings = (
                        {},
                        [],
                        [],
                    )
                if waiting_text_translations:
                    waited_values = await asyncio.gather(
                        *waiting_text_translations.values()
                    )
                    text_translations.update(
                        dict(zip(waiting_text_translations, waited_values))
                    )
                text_translations.update(cached_text_translations)
                reused_translation_count = (
                    len(cached_text_translations)
                    + len(waiting_text_translations)
                )
                if reused_translation_count:
                    text_providers = [*text_providers, "document-cache"]
                    _log_document_event(
                        "cad_text_translation_cache_reused",
                        filename=filename,
                        page_number=page_number,
                        cached_count=len(cached_text_translations),
                        coalesced_count=len(waiting_text_translations),
                        requested_count=len(pending_translation_segments),
                    )
                page_layout = []
                for candidate, item in recognized_rows:
                    source_value = str(item.get("source_text") or "").strip()
                    if not re.search(r"[\u0E00-\u0E7F]", source_value):
                        continue
                    source_key = re.sub(r"\s+", " ", source_value)
                    translated_value = text_translations[
                        translation_segment_by_source[source_key]
                    ]
                    rect = fitz.Rect(candidate["bbox"])
                    rotation = int(candidate.get("rotation") or 0) % 360
                    cross_size = rect.width if rotation in {90, 270} else rect.height
                    line_count = max(1, translated_value.count("\n") + 1)
                    thai_consonant_count = len(
                        re.findall(r"[\u0E01-\u0E2E]", source_value)
                    )
                    maximum_font_size = 7.0 if thai_consonant_count <= 2 else 14.0
                    font_size = max(
                        2.0,
                        min(maximum_font_size, cross_size / line_count * 0.72),
                    )
                    page_layout.append(
                        {
                            "segment_id": f"pdf:p{page_number}:g{len(page_layout) + 1}",
                            "page_number": page_number,
                            "text": source_value,
                            "translated_text": translated_value,
                            "source_kind": "outline-text",
                            "bbox": tuple(rect),
                            "font_size": font_size,
                            "color": "#000000",
                            "alignment": "left",
                            "metadata": {
                                "visual_pdf_version": 1,
                                "translation_unit": "complete-line",
                                "ocr_provider": candidate.get("candidate_provider")
                                or "gpt-indexed-source",
                                "page_type": page_type,
                                "rotation": rotation,
                                "line_count": line_count,
                                "leading": max(font_size, cross_size / line_count),
                                "cover_bbox": list(
                                    candidate.get("cover_bbox") or rect
                                ),
                                "table_region": True,
                                "dense_cad_tight_cover": True,
                            },
                        }
                    )

                for segment in supplemental_segments:
                    exported = asdict(segment)
                    exported["translated_text"] = text_translations[
                        segment.segment_id
                    ]
                    page_layout.append(exported)

                source_text = "\n".join(item["text"] for item in page_layout)
                translated_text = "\n".join(
                    item["translated_text"] for item in page_layout
                )
                page_warnings = list(text_warnings)
                if preserved_tiny_label_candidates:
                    page_warnings.append(
                        f"第 {page_number} 页保留 "
                        f"{preserved_tiny_label_candidates} 个"
                        "多工具未确认的极小 CAD 标签原文"
                    )
                repeated_source_count = len(recognized_rows) - len(unique_segments)
                if repeated_source_count:
                    page_warnings.append(
                        f"第 {page_number} 页复用 {repeated_source_count} 个"
                        "页内完全相同的 CAD 原文译文"
                    )
                if paddle_added_count or paddle_replaced_count:
                    page_warnings.append(
                        f"第 {page_number} 页 Paddle 补回 "
                        f"{paddle_added_count} 个 CAD 文字候选，"
                        f"校正 {paddle_replaced_count} 个"
                        " Tesseract 候选框"
                    )
                provider = "+".join(
                    _unique_provider_names([*reader_providers, *text_providers])
                )
                page_result = TranslationResult(
                    source_language="th",
                    translated_text=translated_text,
                    provider=provider,
                    warnings=page_warnings,
                    layout_segments=page_layout,
                )
                if progress_callback:
                    progress_callback(1, "正在翻译并保留原排版")
                return [(
                    page_number,
                    source_text,
                    page_result,
                    page_layout,
                )], page_warnings

                # Tesseract provides fast recall but misses outlined labels
                # such as compact door codes and ceiling annotations. Paddle
                # is used here only as a leak detector. Its results still go
                # through the indexed-image model before any text is replaced,
                # and only candidates outside an existing line are added to
                # the document.
                existing_refs = [
                    (sheet, item_id)
                    for sheet in sheets
                    for item_id in (sheet.get("entries") or {})
                ]
                existing_bboxes = [
                    sheet["entries"][item_id]["bbox"]
                    for sheet, item_id in existing_refs
                ]

                def locate_paddle_supplements():
                    document = fitz.open(stream=content, filetype="pdf")
                    try:
                        return (
                            detect_dense_cad_paddle_candidates(
                                document[page_number - 1],
                                desired_width=4800,
                                native_units=native_units,
                                existing_bboxes=existing_bboxes,
                            ),
                            existing_refs,
                        )
                    finally:
                        document.close()

                async def translate_sheet(sheet_number, sheet):
                    if not sheet["entries"]:
                        return sheet, [], "", 0
                    try:
                        sheet["sheet_number"] = sheet_number
                        expected_ids = list(sheet["entries"])
                        # Tesseract is a fast locator, not an authoritative
                        # Thai transcription engine. Its confidence measures
                        # image legibility, and can remain high after marks or
                        # consonants were misread. Every visible CAD line is
                        # therefore read from the indexed image by the vision
                        # model. The local reading remains a deliberately
                        # non-authoritative hint for difficult glyphs.
                        reliable_ids = []
                        uncertain_ids = expected_ids
                        accepted = {}
                        route = ""

                        # A reliable local OCR line has all the context needed
                        # for translation. Preserve the entire mixed-language
                        # line and use the normal structured text endpoint in
                        # parallel, instead of sending a visual request for it.
                        if reliable_ids:
                            reliable_segments = [
                                DocumentSegment(
                                    segment_id=(
                                        f"cad:p{page_number}:s{sheet_number}:"
                                        f"{item_id}"
                                    ),
                                    page_number=page_number,
                                    text=str(
                                        sheet["entries"][item_id].get(
                                            "source_hint"
                                        )
                                        or ""
                                    ),
                                    source_kind="outline-text",
                                    bbox=tuple(sheet["entries"][item_id]["bbox"]),
                                    metadata={"ocr_provider": "tesseract"},
                                )
                                for item_id in reliable_ids
                            ]
                            translated, reliable_providers, _warnings = (
                                await _translate_document_segments(
                                    reliable_segments,
                                    "th",
                                    target_language,
                                    context,
                                )
                            )
                            route = "+".join(
                                _unique_provider_names(reliable_providers)
                            )
                            for segment, item_id in zip(
                                reliable_segments, reliable_ids
                            ):
                                accepted[item_id] = {
                                    "id": item_id,
                                    "source_text": segment.text,
                                    "translated_text": translated[segment.segment_id],
                                }

                        # OCR rows that do not meet the reliability gate get a
                        # compact indexed image. Only these rows use the
                        # slower vision path; omissions then proceed to the
                        # explicit high-resolution review pass below.
                        if uncertain_ids:
                            uncertain_sheet = sheet
                            if len(uncertain_ids) != len(expected_ids):
                                uncertain_sheet = await loop.run_in_executor(
                                    None,
                                    subset_indexed_translation_sheet,
                                    sheet,
                                    uncertain_ids,
                                )
                            async with cad_vision_semaphore:
                                items, vision_route = await asyncio.wait_for(
                                    translator.translate_indexed_image_lines(
                                        uncertain_sheet["content"],
                                        "image/png",
                                        list(uncertain_sheet.get("entries") or {}),
                                        target_language,
                                        context,
                                        require_complete=False,
                                        include_non_thai=True,
                                        verify_thai_source=True,
                                        expected_sources={
                                            item_id: str(
                                                uncertain_sheet["entries"][item_id].get(
                                                    "source_hint"
                                                )
                                                or ""
                                            )
                                            for item_id in uncertain_sheet.get(
                                                "entries"
                                            )
                                            if str(
                                                uncertain_sheet["entries"][item_id].get(
                                                    "source_hint"
                                                )
                                                or ""
                                            ).strip()
                                        },
                                    ),
                                    timeout=CAD_INDEXED_MODEL_TIMEOUT_SECONDS,
                                )
                            accepted.update({item["id"]: item for item in items})
                            if vision_route:
                                route = (
                                    f"{route}+{vision_route}"
                                    if route
                                    else vision_route
                                )
                        sheet["unresolved_ids"] = [
                            item_id for item_id in expected_ids if item_id not in accepted
                        ]
                        items = [
                            accepted[item_id]
                            for item_id in expected_ids
                            if item_id in accepted
                        ]
                    except asyncio.TimeoutError as exc:
                        raise RuntimeError(
                            f"MODEL_TIMEOUT: 第 {page_number} 页 CAD 索引图"
                            f"第 {sheet_number} 批超过 "
                            f"{CAD_INDEXED_MODEL_TIMEOUT_SECONDS:.0f} 秒"
                        ) from exc
                    except RuntimeError as exc:
                        raise RuntimeError(
                            f"第 {page_number} 页 CAD 索引图第 {sheet_number} 批：{exc}"
                        ) from exc
                    return sheet, items, route, len(sheet["entries"])

                sheet_jobs = list(enumerate(sheets, start=1))
                first_pass = await asyncio.gather(
                    *(
                        translate_sheet(sheet_number, sheet)
                        for sheet_number, sheet in sheet_jobs
                    ),
                    return_exceptions=True,
                )
                translated_sheets = []
                retry_jobs = []
                for job, result in zip(sheet_jobs, first_pass):
                    if isinstance(result, Exception):
                        retry_jobs.append(job)
                    else:
                        translated_sheets.append(result)
                if retry_jobs:
                    # Let the main request wave drain before retrying only the
                    # incomplete sheets. This absorbs occasional model/gateway
                    # truncation without re-OCRing or retranslating the page.
                    translated_sheets.extend(
                        await asyncio.gather(
                            *(
                                translate_sheet(sheet_number, sheet)
                                for sheet_number, sheet in retry_jobs
                            )
                        )
                    )

                unresolved_for_review = []
                translated_cad_covers = [
                    fitz.Rect(sheet["entries"][item["id"]]["bbox"])
                    for sheet, items, _route, _count in translated_sheets
                    for item in items
                    if re.search(r"[\u0E00-\u0E7F]", item["source_text"])
                ]
                suppressed_duplicate_candidates = 0
                preserved_tiny_label_candidates = 0
                for result_index, (sheet, _items, _route, _count) in enumerate(
                    translated_sheets
                ):
                    for item_id in sheet.get("unresolved_ids") or []:
                        candidate = dict(sheet["entries"][item_id])
                        if _cad_candidate_is_covered_by_translation(
                            candidate["bbox"], translated_cad_covers
                        ):
                            sheet["unresolved_ids"] = [
                                unresolved_id
                                for unresolved_id in sheet["unresolved_ids"]
                                if unresolved_id != item_id
                            ]
                            suppressed_duplicate_candidates += 1
                            continue
                        if _cad_candidate_is_tiny_unreadable_label(candidate):
                            # A single-glyph outlined CAD code can be located
                            # but not read reliably by either local OCR or the
                            # first vision pass. The user opted to preserve
                            # such labels rather than erase and guess them.
                            sheet["unresolved_ids"] = [
                                unresolved_id
                                for unresolved_id in sheet["unresolved_ids"]
                                if unresolved_id != item_id
                            ]
                            preserved_tiny_label_candidates += 1
                            continue
                        candidate["origin_result_index"] = result_index
                        candidate["origin_item_id"] = item_id
                        unresolved_for_review.append(candidate)
                if unresolved_for_review:
                    def prepare_review_sheets():
                        document = fitz.open(stream=content, filetype="pdf")
                        try:
                            return prepare_dense_cad_review_sheets(
                                document[page_number - 1],
                                unresolved_for_review,
                                desired_width=6400,
                                rows_per_sheet=3,
                            )
                        finally:
                            document.close()

                    review_sheets = await loop.run_in_executor(
                        None, prepare_review_sheets
                    )

                    async def translate_review_sheet(review_sheet):
                        review_ids = list(review_sheet["entries"])
                        for attempt in range(CAD_INDEXED_MODEL_MAX_ATTEMPTS):
                            try:
                                async with cad_vision_semaphore:
                                    items, route = await asyncio.wait_for(
                                        translator.translate_indexed_image_lines(
                                            review_sheet["content"],
                                            "image/png",
                                            review_ids,
                                            target_language,
                                            context,
                                            require_complete=False,
                                            include_non_thai=True,
                                            verify_thai_source=True,
                                        ),
                                        timeout=CAD_INDEXED_MODEL_TIMEOUT_SECONDS,
                                    )
                                return review_sheet, items, route
                            except asyncio.TimeoutError:
                                if attempt + 1 >= CAD_INDEXED_MODEL_MAX_ATTEMPTS:
                                    raise RuntimeError(
                                        f"MODEL_TIMEOUT: 第 {page_number} 页 CAD 高清复核批次超过 "
                                        f"{CAD_INDEXED_MODEL_TIMEOUT_SECONDS:.0f} 秒，已重试 "
                                        f"{CAD_INDEXED_MODEL_MAX_ATTEMPTS} 次"
                                    ) from None

                    reviewed = await asyncio.gather(
                        *(translate_review_sheet(sheet) for sheet in review_sheets)
                    )
                    for review_sheet, review_items, review_route in reviewed:
                        for item in review_items:
                            candidate = review_sheet["entries"][item["id"]]
                            result_index = int(candidate["origin_result_index"])
                            origin_item_id = str(candidate["origin_item_id"])
                            sheet, items, route, count = translated_sheets[result_index]
                            if origin_item_id not in {entry["id"] for entry in items}:
                                replacement = dict(item)
                                replacement["id"] = origin_item_id
                                items.append(replacement)
                            sheet["entries"][origin_item_id]["focused_review"] = True
                            sheet["unresolved_ids"] = [
                                unresolved_id
                                for unresolved_id in sheet.get("unresolved_ids") or []
                                if unresolved_id != origin_item_id
                            ]
                            if review_route and review_route not in route:
                                route = (
                                    f"{route}+{review_route}" if route else review_route
                                )
                            translated_sheets[result_index] = (
                                sheet,
                                items,
                                route,
                                count,
                            )
                local_recovery_items = []
                local_recovery_routes = []
                local_recovery_warnings = []
                final_review_candidates = []
                for result_index, (sheet, _items, _route, _count) in enumerate(
                    translated_sheets
                ):
                    for item_id in sheet.get("unresolved_ids") or []:
                        candidate = dict(sheet["entries"][item_id])
                        candidate["origin_result_index"] = result_index
                        candidate["origin_item_id"] = item_id
                        final_review_candidates.append(candidate)
                if final_review_candidates:
                    def recover_local_review_lines():
                        document = fitz.open(stream=content, filetype="pdf")
                        try:
                            return recover_dense_cad_review_text_lines(
                                document[page_number - 1],
                                final_review_candidates,
                            )
                        finally:
                            document.close()

                    recovered_candidates = await loop.run_in_executor(
                        None, recover_local_review_lines
                    )
                    if recovered_candidates:
                        recovered_segments = [
                            DocumentSegment(
                                segment_id=(
                                    f"cad:p{page_number}:local:{index:03d}"
                                ),
                                page_number=page_number,
                                text=str(candidate["source_hint"]),
                                source_kind="outline-text",
                                bbox=tuple(candidate["bbox"]),
                                metadata={"ocr_provider": "paddle-local-review"},
                            )
                            for index, candidate in enumerate(
                                recovered_candidates, start=1
                            )
                        ]
                        recovered_translations, recovered_providers, recovered_warnings = (
                            await _translate_document_segments(
                                recovered_segments,
                                "th",
                                target_language,
                                context,
                            )
                        )
                        local_recovery_routes.extend(recovered_providers)
                        local_recovery_warnings.extend(recovered_warnings)
                        recovered_origins = set()
                        recovery_route = "+".join(
                            _unique_provider_names(recovered_providers)
                        )
                        for candidate, segment in zip(
                            recovered_candidates, recovered_segments
                        ):
                            local_recovery_items.append(
                                (
                                    candidate,
                                    {
                                        "id": str(candidate.get("origin_item_id") or ""),
                                        "source_text": segment.text,
                                        "translated_text": recovered_translations[
                                            segment.segment_id
                                        ],
                                    },
                                    recovery_route,
                                )
                            )
                            recovered_origins.add(
                                (
                                    int(candidate.get("origin_result_index") or 0),
                                    str(candidate.get("origin_item_id") or ""),
                                )
                            )
                        for result_index, item_id in recovered_origins:
                            sheet, items, route, count = translated_sheets[result_index]
                            sheet["unresolved_ids"] = [
                                unresolved_id
                                for unresolved_id in sheet.get("unresolved_ids") or []
                                if unresolved_id != item_id
                            ]
                            translated_sheets[result_index] = (
                                sheet,
                                items,
                                route,
                                count,
                            )

                # Only candidates that neither the regular OCR path nor the
                # targeted Paddle recovery could read need a final vision pass.
                final_review_candidates = []
                for result_index, (sheet, _items, _route, _count) in enumerate(
                    translated_sheets
                ):
                    for item_id in sheet.get("unresolved_ids") or []:
                        candidate = dict(sheet["entries"][item_id])
                        candidate["origin_result_index"] = result_index
                        candidate["origin_item_id"] = item_id
                        final_review_candidates.append(candidate)
                if final_review_candidates:
                    def prepare_final_review_sheets():
                        document = fitz.open(stream=content, filetype="pdf")
                        try:
                            return prepare_dense_cad_review_sheets(
                                document[page_number - 1],
                                final_review_candidates,
                                desired_width=12000,
                                rows_per_sheet=1,
                                wide_context=True,
                            )
                        finally:
                            document.close()

                    final_review_sheets = await loop.run_in_executor(
                        None, prepare_final_review_sheets
                    )

                    async def translate_final_review_sheet(review_sheet):
                        review_ids = list(review_sheet["entries"])
                        candidate = review_sheet["entries"][review_ids[0]]
                        for attempt in range(CAD_INDEXED_MODEL_MAX_ATTEMPTS):
                            try:
                                async with cad_vision_semaphore:
                                    items, route = await asyncio.wait_for(
                                        translator.translate_indexed_image_lines(
                                            review_sheet["content"],
                                            "image/png",
                                            review_ids,
                                            target_language,
                                            context,
                                            require_complete=True,
                                            include_non_thai=True,
                                            verify_thai_source=True,
                                            accept_single_thai_reading=True,
                                        ),
                                        timeout=CAD_INDEXED_MODEL_TIMEOUT_SECONDS,
                                    )
                                return review_sheet, items, route
                            except asyncio.TimeoutError:
                                if attempt + 1 < CAD_INDEXED_MODEL_MAX_ATTEMPTS:
                                    continue
                                origin_sheet_index = int(
                                    candidate.get("origin_result_index") or 0
                                )
                                origin_sheet = translated_sheets[origin_sheet_index][0]
                                raise RuntimeError(
                                    f"MODEL_TIMEOUT: 第 {page_number} 页 CAD 最终高清复核批次超过 "
                                    f"{CAD_INDEXED_MODEL_TIMEOUT_SECONDS:.0f} 秒，已重试 "
                                    f"{CAD_INDEXED_MODEL_MAX_ATTEMPTS} 次；CAD 候选 "
                                    f"S{int(origin_sheet.get('sheet_number') or 0):03d}:"
                                    f"{candidate.get('origin_item_id')} bbox={candidate.get('bbox')}"
                                ) from None
                            except RuntimeError as exc:
                                origin_sheet_index = int(
                                    candidate.get("origin_result_index") or 0
                                )
                                origin_sheet = translated_sheets[origin_sheet_index][0]
                                raise RuntimeError(
                                    f"{exc}; CAD 候选 S{int(origin_sheet.get('sheet_number') or 0):03d}:"
                                    f"{candidate.get('origin_item_id')} bbox={candidate.get('bbox')}"
                                ) from exc

                    final_reviews = await asyncio.gather(
                        *(
                            translate_final_review_sheet(sheet)
                            for sheet in final_review_sheets
                        )
                    )
                    for review_sheet, review_items, review_route in final_reviews:
                        for item in review_items:
                            candidate = review_sheet["entries"][item["id"]]
                            result_index = int(candidate["origin_result_index"])
                            origin_item_id = str(candidate["origin_item_id"])
                            sheet, items, route, count = translated_sheets[result_index]
                            if origin_item_id not in {entry["id"] for entry in items}:
                                replacement = dict(item)
                                replacement["id"] = origin_item_id
                                items.append(replacement)
                            sheet["entries"][origin_item_id]["focused_review"] = True
                            sheet["unresolved_ids"] = [
                                unresolved_id
                                for unresolved_id in sheet.get("unresolved_ids") or []
                                if unresolved_id != origin_item_id
                            ]
                            if review_route and review_route not in route:
                                route = (
                                    f"{route}+{review_route}" if route else review_route
                                )
                            translated_sheets[result_index] = (
                                sheet,
                                items,
                                route,
                                count,
                            )
                unresolved_after_review = [
                    (int(sheet.get("sheet_number") or 0), item_id)
                    for sheet, _items, _route, _count in translated_sheets
                    for item_id in sheet.get("unresolved_ids") or []
                ]
                if unresolved_after_review:
                    preview = ", ".join(
                        f"S{sheet_number:03d}:{item_id}"
                        for sheet_number, item_id in unresolved_after_review[:8]
                    )
                    suffix = "..." if len(unresolved_after_review) > 8 else ""
                    raise RuntimeError(
                        f"ID_MISMATCH: 第 {page_number} 页 CAD 高清复核后仍缺少 "
                        f"{len(unresolved_after_review)} 个候选 ID: {preview}{suffix}"
                    )
                paddle_review_items = list(local_recovery_items)
                paddle_unresolved_count = 0
                # Full-page Paddle inference is CPU and memory intensive. On
                # CAD sheets, running it alongside the first wave of image
                # requests has caused otherwise small gateway calls to stall.
                # Run it after those requests drain; it remains a mandatory
                # leak pass and does not reduce candidate coverage.
                async with cad_local_ocr.paddle():
                    paddle_candidates, existing_refs = await loop.run_in_executor(
                        None, locate_paddle_supplements
                    )
                if paddle_candidates:
                    paddle_candidates_to_review = []
                    for candidate in paddle_candidates:
                        matching_index = candidate.get("matching_existing_index")
                        if matching_index is None:
                            paddle_candidates_to_review.append(candidate)
                            continue
                        source_sheet, source_item_id = existing_refs[int(matching_index)]
                        source_candidate = source_sheet["entries"][source_item_id]
                        source_candidate["bbox"] = tuple(candidate["bbox"])
                        source_candidate["cover_bbox"] = tuple(candidate["bbox"])
                        source_already_translated = any(
                            source_sheet is translated_sheet
                            and any(
                                item["id"] == source_item_id
                                for item in translated_items
                            )
                            for translated_sheet, translated_items, _route, _count
                            in translated_sheets
                        )
                        if not source_already_translated:
                            candidate["matching_source_sheet"] = source_sheet
                            candidate["matching_source_item_id"] = source_item_id
                            paddle_candidates_to_review.append(candidate)

                    def prepare_paddle_review_sheets():
                        document = fitz.open(stream=content, filetype="pdf")
                        try:
                            return prepare_dense_cad_review_sheets(
                                document[page_number - 1],
                                paddle_candidates_to_review,
                                desired_width=6400,
                                rows_per_sheet=3,
                            )
                        finally:
                            document.close()

                    paddle_review_sheets = await loop.run_in_executor(
                        None, prepare_paddle_review_sheets
                    )

                    async def translate_paddle_review_sheet(review_sheet):
                        review_ids = list(review_sheet["entries"])
                        for attempt in range(CAD_INDEXED_MODEL_MAX_ATTEMPTS):
                            try:
                                async with cad_vision_semaphore:
                                    items, route = await asyncio.wait_for(
                                        translator.translate_indexed_image_lines(
                                            review_sheet["content"],
                                            "image/png",
                                            review_ids,
                                            target_language,
                                            context,
                                            require_complete=False,
                                            include_non_thai=True,
                                            verify_thai_source=True,
                                        ),
                                        timeout=CAD_INDEXED_MODEL_TIMEOUT_SECONDS,
                                    )
                                return review_sheet, items, route
                            except asyncio.TimeoutError:
                                if attempt + 1 >= CAD_INDEXED_MODEL_MAX_ATTEMPTS:
                                    raise RuntimeError(
                                        f"MODEL_TIMEOUT: 第 {page_number} 页 CAD Paddle "
                                        f"复核批次超过 {CAD_INDEXED_MODEL_TIMEOUT_SECONDS:.0f} 秒，"
                                        f"已重试 {CAD_INDEXED_MODEL_MAX_ATTEMPTS} 次"
                                    )

                    paddle_reviews = await asyncio.gather(
                        *(
                            translate_paddle_review_sheet(sheet)
                            for sheet in paddle_review_sheets
                        )
                    )
                    paddle_candidates_needing_final_review = []
                    for review_sheet, items, route in paddle_reviews:
                        accepted_ids = {item["id"] for item in items}
                        for item_id, candidate in review_sheet["entries"].items():
                            if item_id in accepted_ids:
                                continue
                            if _cad_candidate_is_tiny_unreadable_label(candidate):
                                paddle_unresolved_count += 1
                                continue
                            paddle_candidates_needing_final_review.append(candidate)
                        for item in items:
                            candidate = review_sheet["entries"][item["id"]]
                            matching_source_sheet = candidate.get(
                                "matching_source_sheet"
                            )
                            matching_source_item_id = candidate.get(
                                "matching_source_item_id"
                            )
                            if matching_source_sheet is not None:
                                for result_index, (
                                    source_sheet,
                                    source_items,
                                    source_route,
                                    source_count,
                                ) in enumerate(translated_sheets):
                                    if source_sheet is not matching_source_sheet:
                                        continue
                                    replacement = dict(item)
                                    replacement["id"] = matching_source_item_id
                                    source_items.append(replacement)
                                    source_sheet["unresolved_ids"] = [
                                        unresolved_id
                                        for unresolved_id in source_sheet.get(
                                            "unresolved_ids"
                                        )
                                        or []
                                        if unresolved_id != matching_source_item_id
                                    ]
                                    translated_sheets[result_index] = (
                                        source_sheet,
                                        source_items,
                                        source_route,
                                        source_count,
                                    )
                                    break
                            else:
                                paddle_review_items.append((candidate, item, route))
                    if paddle_candidates_needing_final_review:
                        def prepare_final_paddle_review_sheets():
                            document = fitz.open(stream=content, filetype="pdf")
                            try:
                                return prepare_dense_cad_review_sheets(
                                    document[page_number - 1],
                                    paddle_candidates_needing_final_review,
                                    desired_width=12000,
                                    rows_per_sheet=1,
                                    wide_context=True,
                                )
                            finally:
                                document.close()

                        final_paddle_review_sheets = await loop.run_in_executor(
                            None, prepare_final_paddle_review_sheets
                        )

                        async def translate_final_paddle_review_sheet(review_sheet):
                            review_ids = list(review_sheet["entries"])
                            for attempt in range(CAD_INDEXED_MODEL_MAX_ATTEMPTS):
                                try:
                                    async with cad_vision_semaphore:
                                        items, route = await asyncio.wait_for(
                                            translator.translate_indexed_image_lines(
                                                review_sheet["content"],
                                                "image/png",
                                                review_ids,
                                                target_language,
                                                context,
                                                require_complete=True,
                                                include_non_thai=True,
                                                verify_thai_source=True,
                                                accept_single_thai_reading=True,
                                            ),
                                            timeout=CAD_INDEXED_MODEL_TIMEOUT_SECONDS,
                                        )
                                    return review_sheet, items, route
                                except asyncio.TimeoutError:
                                    if attempt + 1 >= CAD_INDEXED_MODEL_MAX_ATTEMPTS:
                                        raise RuntimeError(
                                            f"MODEL_TIMEOUT: 第 {page_number} 页 CAD Paddle "
                                            f"最终复核批次超过 "
                                            f"{CAD_INDEXED_MODEL_TIMEOUT_SECONDS:.0f} 秒，"
                                            f"已重试 {CAD_INDEXED_MODEL_MAX_ATTEMPTS} 次"
                                        )

                        final_paddle_reviews = await asyncio.gather(
                            *(
                                translate_final_paddle_review_sheet(sheet)
                                for sheet in final_paddle_review_sheets
                            )
                        )
                        for review_sheet, items, route in final_paddle_reviews:
                            for item in items:
                                candidate = review_sheet["entries"][item["id"]]
                                matching_source_sheet = candidate.get(
                                    "matching_source_sheet"
                                )
                                matching_source_item_id = candidate.get(
                                    "matching_source_item_id"
                                )
                                if matching_source_sheet is None:
                                    paddle_review_items.append((candidate, item, route))
                                    continue
                                # Paddle can enlarge an existing Tesseract
                                # candidate so it needs the final one-row
                                # review. That is still the original layout
                                # unit, not an ID conflict: promote the
                                # verified reading back to its stable source ID.
                                for result_index, (
                                    source_sheet,
                                    source_items,
                                    source_route,
                                    source_count,
                                ) in enumerate(translated_sheets):
                                    if source_sheet is not matching_source_sheet:
                                        continue
                                    replacement = dict(item)
                                    replacement["id"] = matching_source_item_id
                                    source_items.append(replacement)
                                    source_sheet["unresolved_ids"] = [
                                        unresolved_id
                                        for unresolved_id in source_sheet.get(
                                            "unresolved_ids"
                                        )
                                        or []
                                        if unresolved_id != matching_source_item_id
                                    ]
                                    if route and route not in source_route:
                                        source_route = (
                                            f"{source_route}+{route}"
                                            if source_route
                                            else route
                                        )
                                    translated_sheets[result_index] = (
                                        source_sheet,
                                        source_items,
                                        source_route,
                                        source_count,
                                    )
                                    break
                                else:
                                    raise RuntimeError(
                                        "ID_MISMATCH: CAD Paddle 漏字复核"
                                        "找不到原始候选映射"
                                    )
                page_layout = []
                indexed_providers = []
                indexed_providers.extend(local_recovery_routes)
                supplemental_units = []
                unresolved_candidate_count = 0
                for sheet, items, route, _expected_count in translated_sheets:
                    if route:
                        indexed_providers.append(route)
                    supplemental_units.extend(sheet.get("supplemental_units") or [])
                    unresolved_candidate_count += len(sheet.get("unresolved_ids") or [])
                    for item in items:
                        if not re.search(r"[\u0E00-\u0E7F]", item["source_text"]):
                            continue
                        candidate = sheet["entries"][item["id"]]
                        rect = fitz.Rect(candidate["bbox"])
                        rotation = int(candidate["rotation"]) % 360
                        cross_size = rect.width if rotation in {90, 270} else rect.height
                        line_count = max(
                            1, str(item["translated_text"]).count("\n") + 1
                        )
                        thai_consonant_count = len(
                            re.findall(r"[\u0E01-\u0E2E]", item["source_text"])
                        )
                        maximum_font_size = (
                            7.0 if thai_consonant_count <= 2 else 14.0
                        )
                        font_size = max(
                            2.0,
                            min(
                                maximum_font_size,
                                cross_size / line_count * 0.72,
                            ),
                        )
                        page_layout.append(
                            {
                                "segment_id": f"pdf:p{page_number}:g{len(page_layout) + 1}",
                                "page_number": page_number,
                                "text": item["source_text"],
                                "translated_text": item["translated_text"],
                                "source_kind": "outline-text",
                                "bbox": tuple(rect),
                                "font_size": font_size,
                                "color": "#000000",
                                "alignment": "left",
                                "metadata": {
                                    "visual_pdf_version": 1,
                                    "translation_unit": "complete-line",
                                    "ocr_provider": (
                                        "gpt-indexed-image-review"
                                        if candidate.get("focused_review")
                                        else "gpt-indexed-image"
                                    ),
                                    "page_type": page_type,
                                    "rotation": rotation,
                                    "line_count": line_count,
                                    "leading": max(font_size, cross_size / line_count),
                                    "cover_bbox": list(rect),
                                    "table_region": True,
                                    "dense_cad_tight_cover": True,
                                },
                            }
                        )
                for candidate, item, route in paddle_review_items:
                    if not re.search(r"[\u0E00-\u0E7F]", item["source_text"]):
                        continue
                    if route:
                        indexed_providers.append(route)
                    rect = fitz.Rect(candidate["bbox"])
                    rotation = int(candidate["rotation"]) % 360
                    cross_size = rect.width if rotation in {90, 270} else rect.height
                    line_count = max(1, str(item["translated_text"]).count("\n") + 1)
                    thai_consonant_count = len(
                        re.findall(r"[\u0E01-\u0E2E]", item["source_text"])
                    )
                    maximum_font_size = 7.0 if thai_consonant_count <= 2 else 14.0
                    font_size = max(
                        2.0,
                        min(maximum_font_size, cross_size / line_count * 0.72),
                    )
                    page_layout.append(
                        {
                            "segment_id": f"pdf:p{page_number}:k{len(page_layout) + 1}",
                            "page_number": page_number,
                            "text": item["source_text"],
                            "translated_text": item["translated_text"],
                            "source_kind": "outline-text",
                            "bbox": tuple(rect),
                            "font_size": font_size,
                            "color": "#000000",
                            "alignment": "left",
                            "metadata": {
                                "visual_pdf_version": 1,
                                "translation_unit": "complete-line",
                                "ocr_provider": candidate.get(
                                    "candidate_provider"
                                )
                                or "paddle-detect+gpt-review",
                                "page_type": page_type,
                                "rotation": rotation,
                                "line_count": line_count,
                                "leading": max(font_size, cross_size / line_count),
                                "cover_bbox": list(rect),
                                "table_region": True,
                                "dense_cad_tight_cover": True,
                            },
                        }
                    )
                page_warnings = []
                page_warnings.extend(local_recovery_warnings)
                if local_recovery_items:
                    page_warnings.append(
                        f"第 {page_number} 页局部 Paddle 复核补回 "
                        f"{len(local_recovery_items)} 条 CAD 文字"
                    )
                if suppressed_duplicate_candidates:
                    page_warnings.append(
                        f"第 {page_number} 页合并 {suppressed_duplicate_candidates} 个"
                        "已由成功译文框覆盖的 CAD 重复检测候选"
                    )
                if preserved_tiny_label_candidates:
                    page_warnings.append(
                        f"第 {page_number} 页保留 {preserved_tiny_label_candidates} 个"
                        "多工具未确认的极小 CAD 标签原文"
                    )
                    _log_document_event(
                        "cad_tiny_labels_preserved",
                        filename=filename,
                        page_number=page_number,
                        candidate_count=preserved_tiny_label_candidates,
                    )
                if unresolved_candidate_count:
                    page_warnings.append(
                        f"第 {page_number} 页排除 {unresolved_candidate_count} 个"
                        "经视觉重试和本地识别均未确认的 CAD 误检候选"
                    )
                if paddle_unresolved_count:
                    page_warnings.append(
                        f"第 {page_number} 页排除 {paddle_unresolved_count} 个"
                        "Paddle 补充检测中经高清视觉双读未确认的候选"
                    )
                if supplemental_units:
                    supplemental_segments = [
                        DocumentSegment(**unit) for unit in supplemental_units
                    ]
                    supplemental_translations, supplemental_providers, supplemental_warnings = (
                        await _translate_document_segments(
                            supplemental_segments,
                            "th",
                            target_language,
                            context,
                        )
                    )
                    indexed_providers.extend(supplemental_providers)
                    page_warnings.extend(supplemental_warnings)
                    for segment in supplemental_segments:
                        exported = asdict(segment)
                        exported["translated_text"] = supplemental_translations[
                            segment.segment_id
                        ]
                        page_layout.append(exported)
                if not page_layout:
                    if progress_callback:
                        progress_callback(1, "正在检查页面视觉文字")
                    return [], page_warnings
                source_text = "\n".join(item["text"] for item in page_layout)
                translated_text = "\n".join(
                    item["translated_text"] for item in page_layout
                )
                provider = "+".join(_unique_provider_names(indexed_providers))
                page_result = TranslationResult(
                    source_language="th",
                    translated_text=translated_text,
                    provider=provider,
                    warnings=page_warnings,
                    layout_segments=page_layout,
                )
                if progress_callback:
                    progress_callback(1, "正在翻译并保留原排版")
                return [(
                    page_number,
                    source_text,
                    page_result,
                    page_layout,
                )], page_warnings

            high_accuracy = (
                native_text_complete
                or cad_ocr_mode == "deep"
                or deep_table_page
                or deep_fallback_units is not None
            )

            def extract_page_units():
                document = fitz.open(stream=content, filetype="pdf")
                try:
                    diagnostics = {}
                    units = extract_visual_page_units(
                        document[page_number - 1],
                        page_number,
                        desired_width=4000,
                        minimum_score=(
                            0.60 if page_type in {"image", "mixed"} else 0.78
                        ),
                        native_units=native_units,
                        high_accuracy=False,
                        diagnostics=diagnostics,
                    )
                    return units, diagnostics
                finally:
                    document.close()

            fast_diagnostics = {}
            if deep_fallback_units is not None:
                visual_units = deep_fallback_units
            elif high_accuracy:
                visual_units = await loop.run_in_executor(
                    get_deep_ocr_executor(),
                    extract_visual_page_units_in_worker,
                    page_number,
                    4000,
                    0.60 if page_type in {"image", "mixed"} else 0.78,
                    native_units,
                )
            else:
                visual_units, fast_diagnostics = await loop.run_in_executor(
                    None, extract_page_units
                )

            async def translate_cad_gaps():
                if not (
                    cad_ocr_mode == "auto"
                    and page_plan.route == PdfPageRoute.DENSE_VECTOR
                    and int(page_profile.get("drawing_count") or 0)
                    >= INDEXED_CAD_DRAWING_THRESHOLD
                    and not isinstance(translator.provider, DemoProvider)
                    and fast_diagnostics.get("seed_candidates")
                ):
                    return [], [], []

                def prepare_gap_sheets():
                    document = fitz.open(stream=content, filetype="pdf")
                    try:
                        return prepare_dense_cad_translation_sheets(
                            document[page_number - 1],
                            page_number,
                            desired_width=4000,
                            rows_per_sheet=CAD_GAP_ROWS_PER_SHEET,
                            native_units=[*native_units, *visual_units],
                            seed_candidates=fast_diagnostics["seed_candidates"],
                        )
                    finally:
                        document.close()

                async with cad_local_ocr.exclusive():
                    sheets = await loop.run_in_executor(None, prepare_gap_sheets)

                async def translate_gap_sheet(sheet):
                    expected_ids = list(sheet.get("entries") or {})
                    if not expected_ids:
                        return sheet, [], ""
                    expected_sources = {
                        item_id: str(candidate.get("source_hint") or "")
                        for item_id in expected_ids
                        for candidate in [sheet["entries"][item_id]]
                        if _cad_source_hint_is_reliable(candidate)
                    }
                    items, route = await translator.translate_indexed_image_lines(
                        sheet["content"],
                        "image/png",
                        expected_ids,
                        target_language,
                        context,
                        require_complete=False,
                        verify_thai_source=True,
                        expected_sources=expected_sources,
                    )
                    accepted = {item["id"]: item for item in items}
                    missing_ids = [
                        item_id for item_id in expected_ids if item_id not in accepted
                    ]
                    if missing_ids:
                        retry_sheet = await loop.run_in_executor(
                            None,
                            subset_indexed_translation_sheet,
                            sheet,
                            missing_ids,
                        )
                        retry_items, retry_route = (
                            await translator.translate_indexed_image_lines(
                                retry_sheet["content"],
                                "image/png",
                                list(retry_sheet.get("entries") or {}),
                                target_language,
                                context,
                                require_complete=False,
                                verify_thai_source=True,
                                expected_sources={
                                    item_id: expected_sources[item_id]
                                    for item_id in missing_ids
                                    if item_id in expected_sources
                                },
                            )
                        )
                        for item in retry_items:
                            accepted[item["id"]] = item
                        if retry_route and retry_route not in route:
                            route = f"{route}+{retry_route}" if route else retry_route
                    return sheet, [
                        accepted[item_id]
                        for item_id in expected_ids
                        if item_id in accepted
                    ], route

                results = await asyncio.gather(
                    *(translate_gap_sheet(sheet) for sheet in sheets)
                )
                gap_units = []
                fallback_segments = []
                providers = []
                warnings = []
                unresolved_count = 0
                repeated_source_count = 0
                supplemental_units = []
                canonical_sources = [
                    str(item.get("source_text") or "").strip()
                    for _sheet, items, _route in results
                    for item in items
                    if re.search(
                        r"[\u0E00-\u0E7F]",
                        str(item.get("source_text") or ""),
                    )
                ]
                for sheet, items, route in results:
                    if route:
                        providers.append(route)
                    supplemental_units.extend(sheet.get("supplemental_units") or [])
                    accepted_ids = {item["id"] for item in items}
                    for item in items:
                        if not re.search(r"[\u0E00-\u0E7F]", item["source_text"]):
                            continue
                        candidate = sheet["entries"][item["id"]]
                        rect = fitz.Rect(candidate["bbox"])
                        rotation = int(candidate["rotation"]) % 360
                        cross_size = rect.width if rotation in {90, 270} else rect.height
                        line_count = max(1, str(item["translated_text"]).count("\n") + 1)
                        thai_consonant_count = len(
                            re.findall(r"[\u0E01-\u0E2E]", item["source_text"])
                        )
                        maximum_font_size = 7.0 if thai_consonant_count <= 2 else 14.0
                        font_size = max(
                            2.0,
                            min(
                                maximum_font_size,
                                cross_size / line_count * 0.72,
                            ),
                        )
                        gap_units.append(
                            {
                                "segment_id": f"pdf:p{page_number}:h{len(gap_units) + 1}",
                                "page_number": page_number,
                                "text": item["source_text"],
                                "source_kind": "outline-text",
                                "bbox": tuple(rect),
                                "font_size": font_size,
                                "color": "#000000",
                                "alignment": "left",
                                "metadata": {
                                    "visual_pdf_version": 1,
                                    "translation_unit": "complete-line",
                                    "ocr_provider": "gpt-indexed-image",
                                    "page_type": page_type,
                                    "rotation": rotation,
                                    "line_count": line_count,
                                    "leading": max(font_size, cross_size / line_count),
                                    "cover_bbox": list(rect),
                                    "table_region": True,
                                    "dense_cad_tight_cover": True,
                                },
                                "translated_text": item["translated_text"],
                            }
                        )
                    for item_id, candidate in (sheet.get("entries") or {}).items():
                        if item_id in accepted_ids:
                            continue
                        source_hint = re.sub(
                            r"\s+", " ", str(candidate.get("source_hint") or "")
                        ).strip()
                        source_confidence = float(
                            candidate.get("source_confidence") or 0.0
                        )
                        repeated_source = _match_repeated_cad_source_hint(
                            source_hint,
                            canonical_sources,
                        )
                        if repeated_source:
                            source_hint = repeated_source
                            repeated_source_count += 1
                        thai_consonants = re.findall(
                            r"[\u0E01-\u0E2E]", source_hint
                        )
                        thai_runs = re.findall(r"[\u0E00-\u0E7F]+", source_hint)
                        consonant_diversity = len(set(thai_consonants)) / max(
                            1, len(thai_consonants)
                        )
                        non_space_length = len(re.sub(r"\s+", "", source_hint))
                        thai_ratio = sum(len(run) for run in thai_runs) / max(
                            1, non_space_length
                        )
                        if not repeated_source and (
                            source_confidence < 72.0
                            or len(thai_consonants) < 3
                            or len(set(thai_consonants)) < 2
                            or consonant_diversity < 0.30
                            or thai_ratio < 0.35
                            or len(source_hint) > 250
                        ):
                            unresolved_count += 1
                            continue
                        rect = fitz.Rect(candidate["bbox"])
                        rotation = int(candidate["rotation"]) % 360
                        cross_size = rect.width if rotation in {90, 270} else rect.height
                        font_size = max(2.0, min(12.0, cross_size * 0.72))
                        fallback_segments.append(
                            DocumentSegment(
                                segment_id=(
                                    f"pdf:p{page_number}:t"
                                    f"{len(fallback_segments) + 1}"
                                ),
                                page_number=page_number,
                                text=source_hint,
                                source_kind="outline-text",
                                bbox=tuple(rect),
                                font_size=font_size,
                                color="#000000",
                                alignment="left",
                                metadata={
                                    "visual_pdf_version": 1,
                                    "translation_unit": "complete-line",
                                    "ocr_provider": "tesseract+gpt-text",
                                    "page_type": page_type,
                                    "rotation": rotation,
                                    "line_count": 1,
                                    "leading": max(font_size, cross_size),
                                    "cover_bbox": list(rect),
                                    "table_region": True,
                                    "dense_cad_tight_cover": True,
                                    "source_confidence": source_confidence,
                                },
                            )
                        )
                remaining_segments = [
                    *[DocumentSegment(**unit) for unit in supplemental_units],
                    *fallback_segments,
                ]
                if remaining_segments:
                    translated, extra_providers, extra_warnings = (
                        await _translate_document_segments(
                            remaining_segments,
                            "th",
                            target_language,
                            context,
                        )
                    )
                    providers.extend(extra_providers)
                    warnings.extend(extra_warnings)
                    for segment in remaining_segments:
                        exported = asdict(segment)
                        exported["translated_text"] = translated[segment.segment_id]
                        gap_units.append(exported)
                if fallback_segments:
                    warnings.append(
                        f"第 {page_number} 页使用完整行文本补漏 "
                        f"{len(fallback_segments)} 个 CAD 文字块"
                    )
                if repeated_source_count:
                    warnings.append(
                        f"第 {page_number} 页使用页内已确认重复文案补漏 "
                        f"{repeated_source_count} 个 CAD 文字块"
                    )
                if unresolved_count:
                    warnings.append(
                        f"第 {page_number} 页排除 {unresolved_count} 个"
                        "单字符或重复噪声 CAD 候选"
                    )
                return gap_units, providers, warnings

            gap_task = asyncio.create_task(translate_cad_gaps())
            if not visual_units:
                gap_units, gap_providers, gap_warnings = await gap_task
                if not gap_units:
                    if progress_callback:
                        progress_callback(1, "正在检查页面视觉文字")
                    return [], gap_warnings
                source_text = "\n".join(item["text"] for item in gap_units)
                translated_text = "\n".join(
                    item["translated_text"] for item in gap_units
                )
                result = TranslationResult(
                    source_language="th",
                    translated_text=translated_text,
                    provider="+".join(_unique_provider_names(gap_providers)),
                    warnings=gap_warnings,
                    layout_segments=gap_units,
                )
                if progress_callback:
                    progress_callback(1, "正在翻译并保留原排版")
                return [(page_number, source_text, result, gap_units)], gap_warnings
            if page_plan.route == PdfPageRoute.SCANNED_IMAGE:
                # Scans use PaddleOCR's measured line boxes as the only layout
                # coordinates. GPT translates each complete line by ID, but it
                # is never asked to estimate a position on the page.
                for unit in visual_units:
                    unit["source_kind"] = "ocr"
                    metadata = unit.setdefault("metadata", {})
                    metadata["page_type"] = page_type
                    metadata["visual_pass"] = True
                    metadata["visual_provider"] = "paddleocr"
            visual_segments = [DocumentSegment(**unit) for unit in visual_units]
            local_translation_task = asyncio.create_task(
                _translate_document_segments(
                    visual_segments,
                    source_language,
                    target_language,
                    context,
                )
            )
            (
                (visual_translations, visual_providers, visual_warnings),
                (gap_units, gap_providers, gap_warnings),
            ) = await asyncio.gather(local_translation_task, gap_task)
            page_layout = []
            for segment in visual_segments:
                exported = asdict(segment)
                exported["translated_text"] = visual_translations[segment.segment_id]
                page_layout.append(exported)
            page_layout.extend(gap_units)
            source_text = "\n".join(item["text"] for item in page_layout)
            translated_text = "\n".join(item["translated_text"] for item in page_layout)
            visual_providers.extend(gap_providers)
            visual_warnings.extend(gap_warnings)
            provider = "+".join(_unique_provider_names(visual_providers))
            page_result = TranslationResult(
                source_language=source_language,
                translated_text=translated_text,
                provider=provider,
                warnings=visual_warnings + deep_fallback_warnings,
                layout_segments=page_layout,
            )
            if progress_callback:
                progress_callback(1, "正在翻译并保留原排版")
            return [(
                page_number,
                source_text,
                page_result,
                page_layout,
            )], visual_warnings + deep_fallback_warnings

    try:
        stream_error = None
        while True:
            item_kind, item = await page_queue.get()
            if item_kind == "done":
                break
            if item_kind == "error":
                stream_error = item
                continue
            if item_kind == "prepared":
                prepared_pdf_content = item
                continue

            page = item
            routing_profile = dict(page.profile)
            routing_profile.setdefault("page_type", page.page_type)
            if page.processing_route:
                routing_profile["processing_route"] = page.processing_route
            page_plan = select_pdf_page_plan(routing_profile)
            pages.append(page)
            for segment in page.segments:
                segment.metadata.setdefault("page_type", page.page_type)
                segment.metadata.setdefault(
                    "processing_route", page_plan.route.value
                )
            all_segments.extend(page.segments)
            page_types.append(page.page_type)
            page_profiles.append(page.profile)
            page_routes.append(page_plan.route.value)
            if page.ocr_required:
                ocr_pages.append(page.page_number)
            native_outline_supplement = (
                page_plan.route == PdfPageRoute.DENSE_VECTOR
                and bool(page.profile.get("native_text_complete"))
                and not isinstance(translator.provider, DemoProvider)
            )
            if page_plan.translate_visual or native_outline_supplement:
                visual_pages.append(page.page_number)
                ocr_tasks.append(
                    asyncio.create_task(
                        translate_visual_page(
                            page.page_number,
                            page.page_type,
                            page.profile,
                            list(page.segments),
                            page_plan.route.value,
                        )
                    )
                )
            elif not page.segments and progress_callback:
                progress_callback(1, "正在解析文档")

            if provisional_language is None:
                buffered_pages.append(page)
                if len(pages) >= LAYOUT_SEGMENT_MAX_PAGES and any(
                    buffered_page.segments for buffered_page in buffered_pages
                ):
                    initialize_provisional_language()
            elif page_plan.translate_native:
                process_text_page(page)

            if progress_callback and (
                page.page_number == 1
                or page.page_number % LAYOUT_SEGMENT_MAX_PAGES == 0
            ):
                progress_callback(
                    0,
                    f"正在流水线解析并翻译（已解析 {page.page_number}/{page_count} 页）",
                )
            await asyncio.sleep(0)

        await producer
        if stream_error:
            raise stream_error
        if len(pages) != page_count:
            raise ValueError("PDF 页面数量在解析过程中发生变化，请重新上传")
        page_stream_complete.set()
        _log_document_event(
            "pdf_stream_parse_completed",
            filename=filename,
            page_count=len(pages),
            prepared_pdf=prepared_pdf_content is not None,
            elapsed_ms=round((monotonic() - pipeline_started_at) * 1000),
        )

        if provisional_language is None and all_segments:
            initialize_provisional_language()
        schedule_batch(batch_builder.flush())

        preparable_segments = [
            asdict(segment)
            for segment in all_segments
            if segment.metadata.get("engine") == "content-stream"
            and segment.metadata.get("native_pdf_version")
            and segment.metadata.get("code_refs")
        ]
        if preparable_segments and prepared_pdf_content is None:

            async def prepare_native_source_without_cad_contention():
                async with cad_local_ocr.exclusive():
                    return await loop.run_in_executor(
                        None,
                        prepare_native_pdf_source,
                        content,
                        preparable_segments,
                    )

            native_prepare_future = asyncio.create_task(
                prepare_native_source_without_cad_contention()
            )

        page_texts = [page.text for page in pages]
        type_counts: Dict[str, int] = {}
        for page_type in page_types:
            type_counts[page_type] = type_counts.get(page_type, 0) + 1
        profile_warnings = native_pdf_profile_warnings(page_profiles)
        non_text_visual_pages = [
            page_number
            for page_number in visual_pages
            if page_types[page_number - 1] != "image"
        ]
        if non_text_visual_pages:
            profile_warnings.append(
                "检测到图文混排/矢量图页面，将按页面类型使用原生文字与视觉识别："
                + ", ".join(map(str, non_text_visual_pages))
                + " 页"
            )
        if len(type_counts) > 1:
            profile_warnings.append(
                "页面分类统计："
                + "、".join(
                    f"{page_type} {count} 页"
                    for page_type, count in type_counts.items()
                )
            )
        route_counts: Dict[str, int] = {}
        for route in page_routes:
            route_counts[route] = route_counts.get(route, 0) + 1
        _log_document_event(
            "pdf_route_summary",
            filename=filename,
            page_count=page_count,
            segment_count=len(all_segments),
            page_types=type_counts,
            routes=route_counts,
            visual_pages=visual_pages,
        )
        document = ParsedDocument(
            filename=filename,
            text="\n\n".join(text for text in page_texts if text),
            page_count=page_count,
            warnings=profile_warnings,
            ocr_required=bool(ocr_pages),
            ocr_pages=ocr_pages,
            page_texts=page_texts,
            segments=all_segments,
            page_types=page_types,
            page_profiles=page_profiles,
            visual_pages=visual_pages,
            page_routes=page_routes,
        )
        _validate_document(document)

        final_language = (
            source_language
            if source_language != "auto"
            else (provisional_language or "th")
        )
        if (
            source_language == "auto"
            and all_segments
            and final_language != provisional_language
        ):
            for task in text_tasks:
                task.cancel()
            await asyncio.gather(*text_tasks, return_exceptions=True)
            translations, providers, warnings = await _translate_document_segments(
                all_segments,
                final_language,
                target_language,
                context,
                progress_callback,
            )
        else:
            language_confirmed.set()
            if progress_callback and exact_pages_waiting:
                progress_callback(
                    len(exact_pages_waiting), "正在应用知识库译文"
                )
            batch_results = await asyncio.gather(*text_tasks)
            for batch_translations, batch_providers, batch_warnings in batch_results:
                translations.update(batch_translations)
                providers.extend(batch_providers)
                warnings.extend(batch_warnings)
            warnings.insert(
                0,
                f"文案按最多 {LAYOUT_SEGMENT_MAX_PAGES} 页分为 {batch_count} 个请求",
            )

        ocr_page_results = []
        ocr_task_results = await asyncio.gather(*ocr_tasks)
        if deep_ocr_executor is not None:
            deep_ocr_executor.shutdown(wait=True)
            deep_ocr_executor = None
        for page_results, page_warnings in ocr_task_results:
            ocr_page_results.extend(page_results)
            warnings.extend(page_warnings)
            providers.extend(result.provider for _, _, result, _ in page_results)
        ocr_page_results.sort(key=lambda value: value[0])

        source_text, result = _build_layout_translation_result(
            document,
            source_language,
            translations,
            providers,
            warnings,
            ocr_page_results,
        )
        result.warnings = document.warnings + result.warnings
        if native_prepare_future is not None:
            result.prepared_pdf_content = await native_prepare_future
        elif prepared_pdf_content is not None:
            result.prepared_pdf_content = prepared_pdf_content
        return source_text, result
    except BaseException:
        for task in [*text_tasks, *ocr_tasks]:
            if not task.done():
                task.cancel()
        await asyncio.gather(*text_tasks, *ocr_tasks, return_exceptions=True)
        if native_prepare_future is not None:
            await asyncio.gather(native_prepare_future, return_exceptions=True)
        await producer
        raise
    finally:
        if deep_ocr_executor is not None:
            deep_ocr_executor.shutdown(wait=True)


def _iter_pdf_pages_with_source(
    content: bytes,
    source_language: str,
    prepared_callback=None,
):
    """Keep parser test doubles compatible while production receives the source."""
    try:
        parameters = inspect.signature(iter_pdf_pages).parameters.values()
        accepts_varargs = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        positional_count = sum(
            parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            for parameter in parameters
        )
    except (TypeError, ValueError):
        accepts_varargs = True
        positional_count = 3
    if accepts_varargs or positional_count >= 3:
        return iter_pdf_pages(content, source_language, prepared_callback)
    if positional_count >= 2:
        return iter_pdf_pages(content, source_language)
    return iter_pdf_pages(content)


async def _translate_parsed_document(
    content: bytes,
    document: ParsedDocument,
    source_language: str,
    target_language: str,
    context: str,
    progress_callback: ProgressCallback = None,
):
    extension = Path(document.filename).suffix.lower()
    is_pdf = extension == ".pdf"
    if document.segments or (is_pdf and (document.ocr_pages or document.visual_pages)):
        source_text, result = await _translate_layout_document(
            content,
            document,
            source_language,
            target_language,
            context,
            progress_callback,
        )
    elif is_pdf and document.ocr_required and not document.text.strip():
        source_text, result = await _translate_scanned_pdf(
            content,
            document.page_count,
            source_language,
            target_language,
            context,
            document.ocr_pages,
            progress_callback,
        )
    elif is_pdf and document.page_count > 1:
        source_text, result = await _translate_hybrid_pdf(
            content,
            document,
            source_language,
            target_language,
            context,
            progress_callback,
        )
    else:
        source_text = document.text
        result = await _translate_text_with_knowledge(
            document.text, source_language, target_language, context
        )
        if progress_callback:
            progress_callback(document.page_count, "正在翻译文档")
    result.warnings = document.warnings + result.warnings
    return source_text, result


def _validate_document(document: ParsedDocument) -> None:
    if not document.text.strip() and not document.ocr_required and not document.visual_pages:
        if Path(document.filename).suffix.lower() == ".pdf":
            raise ValueError(
                "PDF 中未发现可靠映射的源语言文字层；"
                "页面也未达到扫描图或高密度轮廓文字的视觉处理条件"
            )
        raise ValueError("文档中没有可翻译的文字")
    extension = Path(document.filename).suffix.lower()
    character_limit = (
        50_000
        if extension in {".txt", ".md"}
        else settings.max_document_characters
    )
    if len(document.text) > character_limit:
        raise ValueError(
            f"文档文字超过 {character_limit:,} 个字符，请拆分后上传"
        )
    if len(document.ocr_pages) > settings.pdf_ocr_max_pages:
        raise ValueError(
            f"扫描 PDF 共 {len(document.ocr_pages)} 个图片页，当前最多处理 "
            f"{settings.pdf_ocr_max_pages} 页，请拆分后上传"
        )


def _cad_source_hint_is_reliable(candidate: Dict) -> bool:
    value = str(candidate.get("source_hint") or "").strip()
    confidence = float(candidate.get("source_confidence") or 0.0)
    consonants = re.findall(r"[\u0E01-\u0E2E]", value)
    thai_marks = re.findall(r"[\u0E30-\u0E3A\u0E40-\u0E4E]", value)
    consonant_diversity = len(set(consonants)) / max(1, len(consonants))
    non_space_length = len(re.sub(r"\s+", "", value))
    thai_length = len(re.findall(r"[\u0E00-\u0E7F]", value))
    return (
        confidence >= 60.0
        and len(consonants) >= 2
        and len(set(consonants)) >= 2
        and thai_length / max(1, non_space_length) >= 0.30
        and (bool(thai_marks) or consonant_diversity >= 0.60)
    )


def _cad_visual_source_cache_key(candidate: Dict) -> Optional[str]:
    """Return a conservative key for reusing a confirmed CAD source row."""
    inherited_key = str(candidate.get("source_cache_key") or "").strip()
    if re.fullmatch(r"[0-9a-f]{64}:.*[\u0E00-\u0E7F].*", inherited_key):
        return inherited_key
    fingerprint = str(candidate.get("visual_fingerprint") or "").strip().lower()
    source_hint = re.sub(
        r"\s+", " ", str(candidate.get("source_hint") or "").strip()
    )
    if (
        len(fingerprint) != 64
        or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        or not re.search(r"[\u0E00-\u0E7F]", source_hint)
        or float(candidate.get("source_confidence") or 0.0) < 40.0
    ):
        return None
    return f"{fingerprint}:{source_hint}"


def _compact_cad_sheets_with_source_cache(
    sheets: List[Dict],
    source_cache: Dict[str, str],
):
    """Remove only exact confirmed or within-batch duplicate visual rows.

    The glyph fingerprint is paired with the local Thai OCR hint. Rows without
    both signals remain untouched and always reach the vision provider.
    """
    seen_keys = set()
    reduced_sheets = []
    cached_rows = []
    duplicate_rows = []
    original_count = 0
    sent_count = 0
    for sheet in sheets:
        entries = sheet.get("entries") or {}
        original_count += len(entries)
        selected_ids = []
        for item_id, candidate in entries.items():
            cache_key = _cad_visual_source_cache_key(candidate)
            if cache_key:
                candidate["source_cache_key"] = cache_key
            cached_source = source_cache.get(cache_key) if cache_key else None
            if cached_source:
                cached_rows.append((candidate, cached_source))
                continue
            if cache_key and cache_key in seen_keys:
                duplicate_rows.append((candidate, cache_key))
                continue
            if cache_key:
                seen_keys.add(cache_key)
            selected_ids.append(item_id)
        if not selected_ids:
            continue
        sent_count += len(selected_ids)
        reduced_sheets.append(
            sheet
            if len(selected_ids) == len(entries)
            else subset_indexed_translation_sheet(sheet, selected_ids)
        )
    return (
        reduced_sheets,
        cached_rows,
        duplicate_rows,
        {
            "original_count": original_count,
            "sent_count": sent_count,
            "cache_hit_count": len(cached_rows),
            "duplicate_count": len(duplicate_rows),
        },
    )


def _commit_and_expand_cad_source_cache(
    recognized_rows,
    cached_rows,
    duplicate_rows,
    source_cache: Dict[str, str],
):
    """Commit vision-confirmed Thai text, then restore reused layout rows."""
    output = list(recognized_rows)
    for candidate, item in recognized_rows:
        source_text = str(item.get("source_text") or "").strip()
        cache_key = _cad_visual_source_cache_key(candidate)
        if cache_key and re.search(r"[\u0E00-\u0E7F]", source_text):
            source_cache[cache_key] = source_text
    for candidate, source_text in cached_rows:
        output.append((candidate, {"id": "CACHE", "source_text": source_text}))
    for candidate, cache_key in duplicate_rows:
        source_text = source_cache.get(cache_key)
        if source_text:
            output.append(
                (candidate, {"id": "DUPLICATE", "source_text": source_text})
            )
    return output


def _cad_indexed_read_needs_review(candidate: Dict, item: Dict) -> bool:
    """Reject a visual no-text result that contradicts reliable local OCR.

    The indexed model may occasionally emit ``[NO_TEXT]`` or an English-only
    reading for a legible Thai row. Treating that as a successful response
    silently drops the candidate. A focused high-resolution retry is cheaper
    than another full-page pass and preserves the local OCR's role as a
    quality alarm rather than an authoritative transcription source.
    """

    source_text = str(item.get("source_text") or "").strip()
    if re.search(r"[\u0E00-\u0E7F]", source_text):
        return False
    if _cad_source_hint_is_reliable(candidate):
        return True
    # A literal no-text answer is a stronger contradiction than an alternate
    # English reading. Preserve short real labels such as a one-consonant Thai
    # syllable when Tesseract saw at least two Thai code points confidently.
    source_hint = str(candidate.get("source_hint") or "")
    return (
        source_text.upper() == "[NO_TEXT]"
        and float(candidate.get("source_confidence") or 0.0) >= 80.0
        and len(re.findall(r"[\u0E00-\u0E7F]", source_hint)) >= 2
    )


def _cad_candidate_is_tiny_unreadable_label(candidate: Dict) -> bool:
    """Allow an explicitly tolerated unreadable CAD code to remain visible.

    This is deliberately narrower than an OCR confidence threshold. It never
    suppresses a normal Thai word or phrase, and is reached only after the
    candidate was absent from the initial vision translation result.
    """

    try:
        rect = fitz.Rect(candidate.get("bbox") or ())
    except (TypeError, ValueError):
        return False
    compact_glyph = max(rect.width, rect.height) <= 36.0
    compact_vertical_code = rect.width <= 28.0 and rect.height <= 80.0
    if rect.is_empty or not (compact_glyph or compact_vertical_code):
        return False
    value = re.sub(r"\s+", "", str(candidate.get("source_hint") or ""))
    thai_characters = re.findall(r"[\u0E00-\u0E7F]", value)
    has_number = bool(re.search(r"[0-9\u0E50-\u0E59]", value))
    return (
        0 < len(value) <= 8
        and len(thai_characters) <= 3
        and has_number
    )


def _match_repeated_cad_source_hint(
    source_hint: str,
    canonical_sources: List[str],
) -> str:
    hint_key = "".join(re.findall(r"[\u0E01-\u0E2E]", source_hint))
    if len(hint_key) < 4 or len(set(hint_key)) < 2:
        return ""
    best_source = ""
    best_score = 0.0
    for source in canonical_sources:
        source_key = "".join(re.findall(r"[\u0E01-\u0E2E]", source))
        if len(source_key) < 4:
            continue
        length_ratio = min(len(hint_key), len(source_key)) / max(
            len(hint_key), len(source_key)
        )
        if length_ratio < 0.55:
            continue
        score = difflib.SequenceMatcher(
            None,
            hint_key,
            source_key,
            autojunk=False,
        ).ratio()
        if score > best_score:
            best_source = source
            best_score = score
    return best_source if best_score >= 0.72 else ""


def _initial_completed_pages(document: ParsedDocument) -> int:
    visual_pages = set(document.visual_pages or document.ocr_pages)
    work_pages = len(visual_pages) + sum(
        1
        for page_number, page_text in enumerate(document.page_texts, start=1)
        if page_number not in visual_pages and page_text.strip()
    )
    return max(0, document.page_count - work_pages)


def _document_job_response(job: DocumentJobState) -> DocumentJobResponse:
    progress = min(100, round(job.completed_pages / job.total_pages * 100))
    return DocumentJobResponse(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        completed_pages=job.completed_pages,
        total_pages=job.total_pages,
        progress=progress,
        message=job.message,
        error=job.error,
        result=job.result,
    )


def _prune_document_jobs() -> None:
    cutoff = monotonic() - DOCUMENT_JOB_TTL_SECONDS
    expired_ids = [
        job_id
        for job_id, job in document_jobs.items()
        if job.status in {"completed", "failed"} and job.updated_at < cutoff
    ]
    for job_id in expired_ids:
        document_jobs.pop(job_id, None)


async def _translate_layout_document(
    content: bytes,
    document: ParsedDocument,
    source_language: str,
    target_language: str,
    context: str,
    progress_callback: ProgressCallback = None,
):
    translations: Dict[str, str] = {}
    providers: List[str] = []
    warnings: List[str] = []
    visual_pages = sorted(set(document.visual_pages or document.ocr_pages))
    page_types = {
        page_number: (
            document.page_types[page_number - 1]
            if page_number - 1 < len(document.page_types)
            else "image"
        )
        for page_number in visual_pages
    }

    tasks = []
    if document.segments:
        tasks.append(
            (
                "text",
                asyncio.create_task(
                    _translate_document_segments(
                        document.segments,
                        source_language,
                        target_language,
                        context,
                        progress_callback,
                    )
                ),
            )
        )
    if visual_pages:
        tasks.append(
            (
                "ocr",
                asyncio.create_task(
                    _translate_pdf_layout_pages(
                        content,
                        visual_pages,
                        source_language,
                        target_language,
                        context,
                        progress_callback,
                        page_types=page_types,
                    )
                ),
            )
        )

    task_results = await asyncio.gather(
        *(task for _, task in tasks), return_exceptions=True
    )
    ocr_page_results = []
    for (task_kind, _), task_result in zip(tasks, task_results):
        if isinstance(task_result, Exception):
            raise task_result
        if task_kind == "text":
            segment_translations, segment_providers, segment_warnings = task_result
            translations.update(segment_translations)
            providers.extend(segment_providers)
            warnings.extend(segment_warnings)
        else:
            page_results, ocr_warnings = task_result
            ocr_page_results.extend(page_results)
            warnings.extend(ocr_warnings)
            providers.extend(result.provider for _, _, result, _ in page_results)

    return _build_layout_translation_result(
        document,
        source_language,
        translations,
        providers,
        warnings,
        ocr_page_results,
    )


def _build_layout_translation_result(
    document: ParsedDocument,
    source_language: str,
    translations: Dict[str, str],
    providers: List[str],
    warnings: List[str],
    ocr_page_results: list,
):
    page_sources: Dict[int, List[str]] = {}
    page_translations: Dict[int, List[str]] = {}
    layout_segments = []
    translations_by_source: Dict[str, str] = {}
    native_segments_by_page: Dict[int, List[Dict]] = {}
    for segment in document.segments:
        translated = translations.get(segment.segment_id)
        if translated is None:
            continue
        exported_segment = asdict(segment)
        exported_segment["translated_text"] = translated
        layout_segments.append(exported_segment)
        native_segments_by_page.setdefault(segment.page_number, []).append(
            exported_segment
        )

    for page_number, source_text, page_result, page_layout in ocr_page_results:
        # A mixed/vector page can be seen by both native extraction and the
        # visual model. Keep the native block when the visual block covers the
        # same area; only add visual blocks that provide genuinely new text.
        # This prevents duplicate labels and duplicate translated numbers.
        for visual_segment in page_layout:
            source_key = re.sub(
                r"\s+",
                " ",
                str(visual_segment.get("text", "")).strip(),
            )
            translated_value = str(
                visual_segment.get("translated_text", "")
            ).strip()
            if source_key and source_key in translations_by_source:
                visual_segment["translated_text"] = translations_by_source[
                    source_key
                ]
            elif source_key and translated_value:
                translations_by_source[source_key] = translated_value
            visual_rect = _layout_segment_rect(visual_segment)
            if not visual_rect.is_empty and any(
                _layout_rect_coverage(visual_rect, _layout_segment_rect(native))
                >= 0.55
                for native in native_segments_by_page.get(page_number, [])
            ):
                continue
            layout_segments.append(visual_segment)

    for segment in layout_segments:
        source_value = str(segment.get("text", "")).strip()
        translated_value = str(segment.get("translated_text", "")).strip()
        if not source_value and not translated_value:
            continue
        page_number = segment.get("page_number")
        if not isinstance(page_number, int):
            continue
        page_sources.setdefault(page_number, []).append(source_value)
        page_translations.setdefault(page_number, []).append(translated_value)

    if not page_translations:
        raise RuntimeError("文档中没有可翻译的文字内容")

    source_text = _join_layout_pages(page_sources, document.page_count)
    translated_text = _join_layout_pages(page_translations, document.page_count)
    # Page markers are Chinese UI structure (for example ``【第 1 页】``), not
    # document content. Detect from the raw source blocks so a short English or
    # Thai document is not mislabeled as Chinese merely because it has pages.
    raw_source_text = "\n".join(
        value
        for page_number in sorted(page_sources)
        for value in page_sources[page_number]
        if value
    )
    detected = (
        detect_language(raw_source_text)
        if source_language == "auto"
        else source_language
    )
    unique_providers = _unique_provider_names(providers) or ["knowledge-base"]
    warnings.insert(0, f"已按原文件结构翻译并写回 {len(layout_segments)} 个文字块")
    return source_text, TranslationResult(
        source_language=detected,
        translated_text=translated_text,
        provider=f"{'+'.join(unique_providers)}:layout",
        warnings=list(dict.fromkeys(warnings)),
        layout_segments=layout_segments,
    )


def _layout_segment_rect(segment: Dict) -> fitz.Rect:
    bbox = segment.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return fitz.Rect()
    try:
        rect = fitz.Rect(*(float(value) for value in bbox))
    except (TypeError, ValueError):
        return fitz.Rect()
    metadata = segment.get("metadata") or {}
    if metadata.get("normalized_bbox"):
        width = float(metadata.get("page_width") or 1000.0)
        height = float(metadata.get("page_height") or 1000.0)
        rect = fitz.Rect(
            rect.x0 / 1000.0 * width,
            rect.y0 / 1000.0 * height,
            rect.x1 / 1000.0 * width,
            rect.y1 / 1000.0 * height,
        )
    return rect


def _layout_rect_overlap(first: fitz.Rect, second: fitz.Rect) -> float:
    if first.is_empty or second.is_empty:
        return 0.0
    intersection = first & second
    if intersection.is_empty:
        return 0.0
    return intersection.get_area() / max(
        1.0, min(first.get_area(), second.get_area())
    )


def _layout_rect_coverage(subject: fitz.Rect, covering: fitz.Rect) -> float:
    """Return how much of ``subject`` is covered by another text box."""
    if subject.is_empty or covering.is_empty:
        return 0.0
    intersection = subject & covering
    if intersection.is_empty:
        return 0.0
    return intersection.get_area() / max(1.0, subject.get_area())


def _unique_provider_names(providers: List[str]) -> List[str]:
    names = []
    for provider in providers:
        for name in provider.split("+"):
            if name and name not in names:
                names.append(name)
    return names


async def _translate_document_segments(
    segments: List[DocumentSegment],
    source_language: str,
    target_language: str,
    context: str,
    progress_callback: ProgressCallback = None,
):
    combined_source = "\n".join(segment.text for segment in segments)
    detected = (
        detect_language(combined_source) if source_language == "auto" else source_language
    )
    translations: Dict[str, str] = {}
    providers: List[str] = []
    warnings: List[str] = []
    remaining_segments = []
    exact_matches = database.find_exact_knowledge_many(
        detected,
        target_language,
        [segment.text for segment in segments],
    )

    for segment in segments:
        exact = exact_matches.get(segment.text)
        if exact:
            translations[segment.segment_id] = normalize_translation_text(
                exact["translated_text"]
            )
            providers.append("knowledge-base")
        else:
            remaining_segments.append(segment)

    dense_cad_segments = (
        len(remaining_segments) >= CAD_PARALLEL_TEXT_BATCH_MIN_ITEMS
        and len({segment.page_number for segment in remaining_segments}) == 1
        and any(segment.segment_id.startswith("cad:p") for segment in remaining_segments)
    )
    if dense_cad_segments:
        page_source_context = "\n".join(
            f"[{segment.segment_id}] {segment.text}" for segment in segments
        )
        context = (
            f"{context}\n\n" if context else ""
        ) + (
            "以下是同一 CAD 页完整原文，仅用于术语和语义一致性；"
            "只返回本次请求的 ID，不得额外输出其他 ID。\n"
            + page_source_context
        )
    batches = _build_document_segment_batches(
        remaining_segments,
        max_items=(
            (
                CAD_BALANCED_TEXT_BATCH_ITEMS
                if len(remaining_segments) <= CAD_BALANCED_TEXT_BATCH_MAX_PAGE_ITEMS
                else CAD_PARALLEL_TEXT_BATCH_ITEMS
            )
            if dense_cad_segments
            else LAYOUT_SEGMENT_MAX_ITEMS
        ),
        max_characters=(
            CAD_PARALLEL_TEXT_BATCH_CHARACTERS
            if dense_cad_segments
            else LAYOUT_SEGMENT_MAX_CHARACTERS
        ),
    )

    remaining_by_page: Dict[int, int] = {}
    for segment in remaining_segments:
        remaining_by_page[segment.page_number] = (
            remaining_by_page.get(segment.page_number, 0) + 1
        )
    segment_pages = {segment.page_number for segment in segments}
    exact_pages = segment_pages - set(remaining_by_page)
    if progress_callback and exact_pages:
        progress_callback(len(exact_pages), "正在应用知识库译文")

    semaphore = asyncio.Semaphore(LAYOUT_TRANSLATION_CONCURRENCY)

    async def translate_batch(batch):
        batch_translations, batch_providers, batch_warnings = (
            await _translate_document_segment_batch(
                batch,
                detected,
                target_language,
                context,
                semaphore,
            )
        )

        completed_pages = 0
        for segment in batch:
            remaining_by_page[segment.page_number] -= 1
            if remaining_by_page[segment.page_number] == 0:
                completed_pages += 1
        if progress_callback and completed_pages:
            progress_callback(completed_pages, "正在翻译并保留原排版")
        return batch_translations, batch_providers, batch_warnings

    batch_results = await asyncio.gather(
        *[translate_batch(batch) for batch in batches]
    )
    for batch_translations, batch_providers, batch_warnings in batch_results:
        translations.update(batch_translations)
        providers.extend(batch_providers)
        warnings.extend(batch_warnings)
    warnings.insert(
        0,
        f"文案按最多 {LAYOUT_SEGMENT_MAX_PAGES} 页分为 {len(batches)} 个请求",
    )
    return translations, providers, list(dict.fromkeys(warnings))


async def _translate_document_segment_batch(
    batch: List[DocumentSegment],
    source_language: str,
    target_language: str,
    context: str,
    semaphore: asyncio.Semaphore,
):
    batch_source = "\n".join(segment.text for segment in batch)
    entries = database.find_matching_knowledge(
        source_language, target_language, batch_source, limit=40
    )
    knowledge_context = _merge_knowledge_context(context, entries)
    if any(
        segment.source_kind in {"outline-text", "ocr"}
        for segment in batch
    ):
        ocr_page_context = []
        for segment in batch:
            for value in segment.metadata.get("ocr_page_context", []):
                value = str(value).strip()
                if value and value not in ocr_page_context:
                    ocr_page_context.append(value)
        knowledge_context = (
            knowledge_context
            + "\n部分文字由 OCR 提取。仅在同页重复结构或连续编号能明确证明时，"
            "修正明显的字符混淆；不要猜测不可辨认内容。图号、编号、专业缩写和型号保持原样。"
            "同一批次中的重复术语、职务和姓名称谓必须保持一致，译文在完整保留语义的前提下尽量简洁。"
            + (
                "\n同页未翻译的高置信数字和英文如下，仅用于核对金额、日期、编号及相邻泰文的语义，"
                "不得把它们添加到无关译文中：\n"
                + "\n".join(ocr_page_context)
                if ocr_page_context
                else ""
            )
        ).strip()

    def retry_context_for(request_batch, request_context):
        source_context = "\n".join(
            f"[{segment.segment_id}] {segment.text}"
            for segment in request_batch
        )
        return (
            f"{request_context}\n\n" if request_context else ""
        ) + (
            "以下是同一页的完整原文，仅用于理解本轮缺失文字块的上下文；"
            "不得把它们额外输出，仍只返回本次请求中的 ID。\n"
            + source_context
        )

    def can_split_after_failure(exc: RuntimeError) -> bool:
        message = str(exc)
        structural_failures = ("文字块", "完整返回", "ID_MISMATCH")
        transient_gateway_failures = (
            "模型服务请求失败 (500)",
            "模型服务请求失败 (502)",
            "模型服务请求失败 (503)",
            "模型服务请求失败 (504)",
            "模型服务请求失败 (520)",
            "模型服务请求失败 (522)",
            "模型服务请求失败 (524)",
            "模型服务响应超时",
        )
        return any(
            marker in message
            for marker in (*structural_failures, *transient_gateway_failures)
        )

    async def request_segments(request_batch, request_context=knowledge_context):
        async def translate_segments_once(
            target_batch,
            target_context,
            *,
            require_complete=False,
        ):
            async with semaphore:
                return await translator.translate_segments(
                    {
                        segment.segment_id: segment.text
                        for segment in target_batch
                    },
                    source_language,
                    target_language,
                    target_context,
                    require_complete=require_complete,
                )

        async def translate_segments_with_hedge(
            target_batch,
            target_context,
            *,
            require_complete=False,
        ):
            primary = asyncio.create_task(
                translate_segments_once(
                    target_batch,
                    target_context,
                    require_complete=require_complete,
                )
            )
            hedge = None
            partial_results = []
            failures = []
            try:
                if TEXT_MODEL_HEDGE_DELAY_SECONDS <= 0:
                    return await primary
                done, _pending = await asyncio.wait(
                    {primary},
                    timeout=TEXT_MODEL_HEDGE_DELAY_SECONDS,
                )
                if done:
                    return await primary
                hedge = asyncio.create_task(
                    translate_segments_once(
                        target_batch,
                        target_context,
                        require_complete=require_complete,
                    )
                )
                _log_document_event(
                    "text_translation_hedge_started",
                    segment_count=len(target_batch),
                    pages=sorted(
                        {segment.page_number for segment in target_batch}
                    ),
                )
                pending = {primary, hedge}
                expected_ids = {
                    segment.segment_id for segment in target_batch
                }
                while pending:
                    completed, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in completed:
                        try:
                            result = task.result()
                        except Exception as exc:
                            failures.append(exc)
                            continue
                        if expected_ids.issubset(result.translations):
                            _log_document_event(
                                "text_translation_hedge_completed",
                                winner=("hedge" if task is hedge else "primary"),
                                segment_count=len(target_batch),
                                pages=sorted(
                                    {
                                        segment.page_number
                                        for segment in target_batch
                                    }
                                ),
                            )
                            for waiting in pending:
                                waiting.cancel()
                            if pending:
                                await asyncio.gather(
                                    *pending, return_exceptions=True
                                )
                            return result
                        partial_results.append(result)
                if partial_results:
                    return max(
                        partial_results,
                        key=lambda result: len(result.translations),
                    )
                raise failures[0]
            finally:
                unfinished = [
                    task
                    for task in (primary, hedge)
                    if task is not None and not task.done()
                ]
                for task in unfinished:
                    task.cancel()
                if unfinished:
                    await asyncio.gather(*unfinished, return_exceptions=True)

        try:
            batch_result = await translate_segments_with_hedge(
                request_batch,
                request_context,
            )
        except RuntimeError as exc:
            if not can_split_after_failure(exc):
                raise
            if len(request_batch) == 1:
                segment = request_batch[0]
                async with semaphore:
                    result = await _translate_text_with_knowledge(
                        segment.text,
                        source_language,
                        target_language,
                        request_context,
                    )
                return (
                    {segment.segment_id: result.translated_text},
                    [result.provider],
                    result.warnings,
                    True,
                )
            midpoint = len(request_batch) // 2
            left, right = await asyncio.gather(
                request_segments(request_batch[:midpoint], request_context),
                request_segments(request_batch[midpoint:], request_context),
            )
            return (
                {**left[0], **right[0]},
                [*left[1], *right[1]],
                [*left[2], *right[2]],
                True,
            )

        missing_segments = [
            segment
            for segment in request_batch
            if segment.segment_id not in batch_result.translations
        ]
        if not missing_segments:
            return (
                batch_result.translations,
                [batch_result.provider],
                batch_result.warnings,
                False,
            )

        # The full page was already sent once.  Keep all accepted translations
        # and ask only for omitted IDs, while including the same page source as
        # read-only context.  This avoids retranslating completed rows or
        # breaking same-page terminology consistency through blind bisection.
        missing_context = retry_context_for(request_batch, request_context)
        try:
            retry_result = await translate_segments_with_hedge(
                missing_segments,
                missing_context,
                require_complete=True,
            )
            return (
                {**batch_result.translations, **retry_result.translations},
                [batch_result.provider, retry_result.provider],
                [*batch_result.warnings, *retry_result.warnings],
                True,
            )
        except RuntimeError as exc:
            if not can_split_after_failure(exc):
                raise
            retry = await request_segments(missing_segments, missing_context)
            return (
                {**batch_result.translations, **retry[0]},
                [batch_result.provider, *retry[1]],
                [*batch_result.warnings, *retry[2]],
                True,
            )

    (
        batch_translations,
        batch_providers,
        batch_warnings,
        used_split_retry,
    ) = await request_segments(batch)
    if entries:
        batch_providers.append("knowledge-base")
        batch_warnings.insert(0, f"已优先应用知识库匹配项 {len(entries)} 条")
    if used_split_retry:
        batch_warnings.insert(
            0,
            "模型请求超时或未完整返回文字块，"
            "已自动拆分并补发以保持排版对应",
        )
    return batch_translations, batch_providers, batch_warnings


def _build_document_segment_batches(
    segments: List[DocumentSegment],
    *,
    max_items: int = LAYOUT_SEGMENT_MAX_ITEMS,
    max_characters: int = LAYOUT_SEGMENT_MAX_CHARACTERS,
    max_pages: int = LAYOUT_SEGMENT_MAX_PAGES,
) -> List[List[DocumentSegment]]:
    batches: List[List[DocumentSegment]] = []
    builder = _SegmentBatchBuilder(
        max_items=max_items,
        max_characters=max_characters,
        max_pages=max_pages,
    )
    for segment in segments:
        closed_batch = builder.add(segment)
        if closed_batch:
            batches.append(closed_batch)
    final_batch = builder.flush()
    if final_batch:
        batches.append(final_batch)
    return batches


async def _translate_pdf_layout_pages(
    content: bytes,
    page_numbers: List[int],
    source_language: str,
    target_language: str,
    context: str,
    progress_callback: ProgressCallback = None,
    page_types: Optional[Dict[int, str]] = None,
    page_profiles: Optional[Dict[int, Dict]] = None,
):
    page_results = []
    warnings = []
    loop = asyncio.get_running_loop()
    concurrency = settings.pdf_ocr_concurrency
    page_sizes: Dict[int, tuple] = {}
    try:
        source_document = fitz.open(stream=content, filetype="pdf")
        try:
            for number in page_numbers:
                page = source_document.load_page(number - 1)
                page_sizes[number] = (float(page.rect.width), float(page.rect.height))
        finally:
            source_document.close()
    except Exception:
        # Rendering below will provide the user-facing error if the PDF is
        # invalid; keep this helper defensive for mocked test inputs.
        page_sizes = {}

    for start in range(0, len(page_numbers), concurrency):
        batch_numbers = page_numbers[start : start + concurrency]
        tile_numbers = []
        for number in batch_numbers:
            page_type = (page_types or {}).get(number)
            page_width, page_height = page_sizes.get(number, (0.0, 0.0))
            large_scan = (
                page_type in {"image", "mixed"}
                and max(page_width, page_height) >= 1200.0
            )
            if page_type in {"vector", "vector_mixed"} or large_scan:
                tile_numbers.append(number)
        full_numbers = [number for number in batch_numbers if number not in tile_numbers]
        rendered_pages = []
        if full_numbers:
            rendered_pages.extend(
                await loop.run_in_executor(
                    None,
                    render_pdf_pages,
                    content,
                    full_numbers,
                    max(settings.pdf_ocr_desired_width, 2000),
                )
            )
        if tile_numbers:
            for tile_number in tile_numbers:
                profile = (page_profiles or {}).get(tile_number, {})
                # A1/A0 schedules contain text too small for reliable OCR in a
                # quadrant render. Six overlapping 3x2 crops keep complete
                # table rows readable while preserving enough neighboring
                # cells for semantic context.
                page_type = (page_types or {}).get(tile_number)
                tile_mode = (
                    "scan"
                    if page_type in {"image", "mixed"}
                    else "directory"
                    if int(profile.get("drawing_count") or 0) < 30_000
                    else "grid"
                )
                rendered_pages.extend(
                    await loop.run_in_executor(
                        None,
                        render_pdf_tiles,
                        content,
                        [tile_number],
                        settings.pdf_ocr_desired_width,
                        tile_mode,
                    )
                )

        async def translate_page(page):
            try:
                page_type = (page_types or {}).get(page.page_number, "image")
                page_width, page_height = page_sizes.get(
                    page.page_number, (1000.0, 1000.0)
                )
                entries = database.list_knowledge_for_direction(
                    target_language,
                    None if source_language == "auto" else source_language,
                    limit=60,
                )
                knowledge_context = _merge_knowledge_context(context, entries)
                source_text, page_result = await translator.translate_image_layout(
                    page.content,
                    "image/png",
                    source_language,
                    target_language,
                    knowledge_context,
                    allow_empty=True,
                )
                page_layout = []
                tile_id = (
                    "full"
                    if page.clip is None
                    else f"x{int(page.clip[0])}y{int(page.clip[1])}"
                )
                for block_index, block in enumerate(
                    page_result.layout_segments, start=1
                ):
                    source_value = re.sub(
                        r"\s+", " ", str(block["source_text"]).strip()
                    )
                    translated_value = re.sub(
                        r"\s+", " ", str(block["translated_text"]).strip()
                    )
                    # Proper names, brands, model codes and other deliberately
                    # preserved text need no cover/write operation. Drawing an
                    # identical visual block over a flattened logo makes it
                    # darker or creates a duplicate even though no translation
                    # was requested.
                    if source_value == translated_value:
                        continue
                    exact = database.find_exact_knowledge(
                        page_result.source_language,
                        target_language,
                        block["source_text"],
                    )
                    if exact:
                        block["translated_text"] = normalize_translation_text(
                            exact["translated_text"]
                        )
                    metadata = {
                        "normalized_bbox": page.clip is None,
                        # Vision coordinates come from the rendered, upright
                        # page. They must not be rotated with native PDF text
                        # coordinates when /Rotate is normalized for export.
                        "display_bbox": True,
                        "page_type": page_type,
                        "visual_pass": True,
                        "visual_provider": page_result.provider,
                        "page_width": page_width,
                        "page_height": page_height,
                    }
                    bbox = block["bbox"]
                    if page.clip is not None:
                        clip_left, clip_top, clip_right, clip_bottom = page.clip
                        clip_width = clip_right - clip_left
                        clip_height = clip_bottom - clip_top
                        bbox = [
                            clip_left + float(bbox[0]) / 1000.0 * clip_width,
                            clip_top + float(bbox[1]) / 1000.0 * clip_height,
                            clip_left + float(bbox[2]) / 1000.0 * clip_width,
                            clip_top + float(bbox[3]) / 1000.0 * clip_height,
                        ]
                    line_count = max(
                        1, str(block["translated_text"]).count("\n") + 1
                    )
                    box_height = max(2.0, (float(bbox[3]) - float(bbox[1])) / line_count)
                    visual_font_size = max(2.5, min(24.0, box_height * 0.72))
                    metadata["line_count"] = line_count
                    metadata["leading"] = max(visual_font_size, box_height * 0.90)
                    page_layout.append(
                        {
                            "segment_id": f"pdf:p{page.page_number}:ocr{tile_id}:{block_index}",
                            "page_number": page.page_number,
                            "text": block["source_text"],
                            "translated_text": block["translated_text"],
                            "source_kind": "ocr",
                            "bbox": bbox,
                            "font_size": visual_font_size,
                            "color": "#000000",
                            "alignment": "left",
                            "metadata": metadata,
                        }
                    )

                # Vector CAD pages can contain outline glyphs that are absent
                # from both the native text layer and the vision response.
                # Run a local OCR detection pass only for those pages, then
                # send the uncovered OCR lines to the same GPT text translator.
                # This adds no work for ordinary text/image pages and avoids
                # asking the vision model to rediscover already covered text.
                if page_type in {"vector", "vector_mixed"} and not isinstance(
                    translator.provider, DemoProvider
                ):
                    ocr_blocks = await loop.run_in_executor(
                        None, ocr_image_text_blocks, page.content
                    )
                    _refine_visual_layout_with_ocr(
                        page_layout,
                        ocr_blocks,
                        page.clip,
                    )
                    missing_blocks = _select_uncovered_ocr_blocks(
                        ocr_blocks,
                        page_layout,
                        page.clip,
                    )
                    if missing_blocks:
                        audit_ids = {
                            f"ocr-audit:{page.page_number}:{index}": block["source_text"]
                            for index, block in enumerate(missing_blocks, start=1)
                        }
                        try:
                            audit_result = await translator.translate_segments(
                                audit_ids,
                                source_language,
                                target_language,
                                knowledge_context,
                            )
                        except Exception as exc:
                            warnings.append(
                                f"第 {page.page_number} 页局部文字复核失败，已保留视觉翻译结果：{exc}"
                            )
                        else:
                            for (segment_id, source_value), block in zip(
                                audit_ids.items(), missing_blocks
                            ):
                                translated_value = audit_result.translations.get(segment_id)
                                if not translated_value:
                                    continue
                                bbox = block["bbox"]
                                if page.clip is not None:
                                    clip_left, clip_top, clip_right, clip_bottom = page.clip
                                    clip_width = clip_right - clip_left
                                    clip_height = clip_bottom - clip_top
                                    bbox = [
                                        clip_left + float(bbox[0]) / 1000.0 * clip_width,
                                        clip_top + float(bbox[1]) / 1000.0 * clip_height,
                                        clip_left + float(bbox[2]) / 1000.0 * clip_width,
                                        clip_top + float(bbox[3]) / 1000.0 * clip_height,
                                        ]
                                line_count = max(
                                    1, str(translated_value).count("\n") + 1
                                )
                                box_height = max(
                                    2.0,
                                    (float(bbox[3]) - float(bbox[1])) / line_count,
                                )
                                audit_font_size = max(
                                    2.5, min(24.0, box_height * 0.72)
                                )
                                page_layout.append(
                                    {
                                        "segment_id": segment_id,
                                        "page_number": page.page_number,
                                        "text": source_value,
                                        "translated_text": translated_value,
                                        "source_kind": "ocr-audit",
                                        "bbox": bbox,
                                        "font_size": audit_font_size,
                                        "color": "#000000",
                                        "alignment": "left",
                                        "metadata": {
                                            "normalized_bbox": page.clip is None,
                                            "display_bbox": True,
                                            "page_type": page_type,
                                            "visual_pass": True,
                                            "visual_provider": audit_result.provider,
                                            "ocr_audit": True,
                                            "line_count": line_count,
                                            "leading": max(audit_font_size, box_height * 0.90),
                                            "page_width": page_width,
                                            "page_height": page_height,
                                        },
                                    }
                                )
                            if audit_result.provider not in page_result.provider:
                                page_result.provider = (
                                    f"{page_result.provider}+{audit_result.provider}"
                                )
                page_result.translated_text = "\n".join(
                    block["translated_text"] for block in page_layout
                )
                if any(
                    database.find_exact_knowledge(
                        page_result.source_language,
                        target_language,
                        block["source_text"],
                    )
                    for block in page_result.layout_segments
                ):
                    page_result.provider = f"{page_result.provider}+knowledge-base"
                return page.page_number, source_text, page_result, page_layout
            except Exception as exc:
                raise RuntimeError(
                    f"视觉翻译失败: page={page.page_number}, clip={page.clip}: {exc}"
                ) from exc
            finally:
                if progress_callback and (
                    page.clip is None
                    or (page.clip[0] == 0.0 and page.clip[1] == 0.0)
                ):
                    progress_callback(1, "正在识别图片页并保留文字位置")

        translated_pages = await asyncio.gather(
            *[translate_page(page) for page in rendered_pages]
        )
        page_results.extend(translated_pages)
        for _, _, page_result, _ in translated_pages:
            warnings.extend(page_result.warnings)

    return page_results, list(dict.fromkeys(warnings))


def _ocr_match_text(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9\u0E00-\u0E7F]+", "", value).lower()


def _cad_candidate_is_covered_by_translation(
    candidate_bbox, translated_covers: List[fitz.Rect]
) -> bool:
    """Whether a detected fragment is wholly covered by another Thai unit.

    Tiled text detection may report both a complete CAD label and a smaller
    sub-box around the same glyphs. Once the complete label has a confirmed
    translation, sending the contained fragment through another OCR pass can
    create a conflicting transcription without improving coverage.
    """
    candidate = fitz.Rect(candidate_bbox)
    candidate_area = max(1.0, candidate.get_area())
    for cover in translated_covers:
        overlap_ratio = (candidate & cover).get_area() / candidate_area
        if overlap_ratio >= 0.90:
            return True
        # The rotated Tesseract pass can rediscover only part of a horizontal
        # label as a thin vertical fragment. Its box often extends just past
        # the baseline, so full containment is too strict. Suppress only a
        # small, materially overlapping fragment whose center remains close
        # to the already translated complete label.
        expanded_cover = cover + (
            -max(2.0, cover.width * 0.04),
            -max(2.0, cover.height * 0.55),
            max(2.0, cover.width * 0.04),
            max(2.0, cover.height * 0.55),
        )
        center = fitz.Point(
            (candidate.x0 + candidate.x1) / 2,
            (candidate.y0 + candidate.y1) / 2,
        )
        if (
            overlap_ratio >= 0.30
            and candidate_area <= max(1.0, cover.get_area()) * 0.65
            and expanded_cover.contains(center)
        ):
            return True
    return False


def _refine_visual_layout_with_ocr(
    visual_blocks: List[Dict],
    ocr_blocks: List[Dict],
    clip: Optional[tuple],
) -> None:
    """Calibrate approximate model boxes against local OCR coordinates.

    Vision models are good at transcription and translation but their
    normalized coordinates drift on very tall or wide CAD tiles. Tesseract is
    used only as a geometric anchor; model text and translation remain the
    authoritative content. A robust affine transform also corrects unmatched
    model blocks, so layout quality does not depend on every Thai OCR line
    being transcribed perfectly.
    """
    if not visual_blocks or not ocr_blocks:
        return
    clip_left, clip_top, clip_right, clip_bottom = clip or (0.0, 0.0, 1000.0, 1000.0)
    clip_width = max(1.0, clip_right - clip_left)
    clip_height = max(1.0, clip_bottom - clip_top)
    # Make refinement idempotent for cached QA runs.
    for block in visual_blocks:
        metadata = dict(block.get("metadata") or {})
        vision_bbox = metadata.get("vision_bbox")
        if isinstance(vision_bbox, list) and len(vision_bbox) == 4:
            block["bbox"] = list(vision_bbox)
        metadata.pop("ocr_anchored", None)
        metadata.pop("ocr_affine", None)
        metadata.pop("ocr_transform", None)
        metadata.pop("ocr_similarity", None)
        block["metadata"] = metadata
    anchors = []
    for block in ocr_blocks:
        text = _ocr_match_text(str(block.get("source_text") or ""))
        bbox = block.get("bbox")
        if len(text) < 2 or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        rect = fitz.Rect(
            clip_left + float(bbox[0]) / 1000.0 * clip_width,
            clip_top + float(bbox[1]) / 1000.0 * clip_height,
            clip_left + float(bbox[2]) / 1000.0 * clip_width,
            clip_top + float(bbox[3]) / 1000.0 * clip_height,
        )
        anchors.append((text, rect))

    candidates = []
    for block_index, block in enumerate(visual_blocks):
        source = _ocr_match_text(str(block.get("text") or ""))
        bbox = block.get("bbox")
        if len(source) < 3 or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        original = fitz.Rect(*(float(value) for value in bbox))
        metadata = dict(block.get("metadata") or {})
        metadata.setdefault("vision_bbox", list(original))
        block["metadata"] = metadata
        source_has_thai = bool(re.search(r"[\u0E00-\u0E7F]", source))
        for anchor_index, (candidate_text, candidate_rect) in enumerate(anchors):
            if source_has_thai != bool(re.search(r"[\u0E00-\u0E7F]", candidate_text)):
                continue
            similarity = difflib.SequenceMatcher(None, source, candidate_text).ratio()
            if source in candidate_text or candidate_text in source:
                similarity = max(
                    similarity,
                    min(len(source), len(candidate_text)) / max(len(source), len(candidate_text)),
                )
            # Outlined Thai glyphs are often returned by Tesseract in partial
            # visual order. Even a modest character overlap remains useful as
            # a geometry anchor when language, tile and proximity all agree.
            if similarity < 0.28:
                continue
            dx = abs((candidate_rect.x0 + candidate_rect.x1) - (original.x0 + original.x1)) / 2
            dy = abs((candidate_rect.y0 + candidate_rect.y1) - (original.y0 + original.y1)) / 2
            # Keep matching local to this compact tile. The corrected OCR
            # pixel normalization can now safely repair moderate model drift;
            # one-to-one scoring below chooses the nearest repeated row.
            if (
                dx > min(100.0, clip_width * 0.15)
                or dy > min(80.0, clip_height * 0.10)
            ):
                continue
            distance_penalty = 0.10 * dx / clip_width + 0.15 * dy / clip_height
            score = similarity - distance_penalty
            if similarity >= 0.32:
                candidates.append(
                    (score, similarity, block_index, anchor_index, original, candidate_rect)
                )

    # One OCR line must not drag several nearby model labels onto the same
    # position. Select the strongest one-to-one matches first.
    matches = []
    used_blocks = set()
    used_anchors = set()
    for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        _, _, block_index, anchor_index, _, _ = candidate
        if block_index in used_blocks or anchor_index in used_anchors:
            continue
        used_blocks.add(block_index)
        used_anchors.add(anchor_index)
        matches.append(candidate)

    reliable = [match for match in matches if match[1] >= 0.68]
    x_transform = _fit_ocr_axis_transform(
        [
            ((match[4].x0 + match[4].x1) / 2, (match[5].x0 + match[5].x1) / 2)
            for match in reliable
        ],
        clip_width,
    )
    y_transform = _fit_ocr_axis_transform(
        [
            ((match[4].y0 + match[4].y1) / 2, (match[5].y0 + match[5].y1) / 2)
            for match in reliable
        ],
        clip_height,
    )

    matched_by_block = {match[2]: match for match in matches}
    for block_index, block in enumerate(visual_blocks):
        bbox = block.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        original = fitz.Rect(*(float(value) for value in bbox))
        metadata = dict(block.get("metadata") or {})
        metadata.setdefault("vision_bbox", list(original))
        match = matched_by_block.get(block_index)
        if match is None:
            if x_transform is None and y_transform is None:
                block["metadata"] = metadata
                continue
            ax, bx = x_transform or (1.0, 0.0)
            ay, by = y_transform or (1.0, 0.0)
            refined = fitz.Rect(
                ax * original.x0 + bx,
                ay * original.y0 + by,
                ax * original.x1 + bx,
                ay * original.y1 + by,
            )
            refined.intersect(fitz.Rect(clip_left, clip_top, clip_right, clip_bottom))
            if not refined.is_empty:
                block["bbox"] = list(refined)
                metadata["ocr_affine"] = True
                metadata["ocr_transform"] = [ax, bx, ay, by]
            block["metadata"] = metadata
            continue

        anchor = match[5]
        # OCR boxes are tight glyph bounds. Preserve the model box's line
        # height where it is larger, but use OCR's baseline and horizontal span.
        height = max(anchor.height, min(original.height, anchor.height * 1.8))
        refined = fitz.Rect(
            anchor.x0,
            anchor.y0 - max(0.0, (height - anchor.height) / 2),
            anchor.x1,
            anchor.y1 + max(0.0, (height - anchor.height) / 2),
        )
        block["bbox"] = list(refined)
        metadata["ocr_anchored"] = True
        metadata["ocr_similarity"] = round(float(match[1]), 4)
        block["metadata"] = metadata


def _fit_ocr_axis_transform(
    pairs: List[tuple],
    axis_extent: float,
) -> Optional[tuple]:
    """Fit a conservative per-tile translation with residual rejection.

    Compact overlapping tiles make scale drift negligible. Restricting the
    correction to translation prevents a handful of repeated table labels
    from stretching every unmatched box in the crop.
    """
    if len(pairs) < 4:
        return None
    current = [target - source for source, target in pairs]
    for _ in range(2):
        ordered_offsets = sorted(current)
        offset = ordered_offsets[len(ordered_offsets) // 2]
        residuals = [abs(value - offset) for value in current]
        ordered_residuals = sorted(residuals)
        median_residual = ordered_residuals[len(ordered_residuals) // 2]
        threshold = max(5.0, min(axis_extent * 0.035, median_residual * 2.5 + 2.0))
        filtered = [
            value
            for value, residual in zip(current, residuals)
            if residual <= threshold
        ]
        if len(filtered) < 4 or len(filtered) == len(current):
            break
        current = filtered
    if len(current) < 4:
        return None
    ordered_offsets = sorted(current)
    offset = ordered_offsets[len(ordered_offsets) // 2]
    offset = max(-axis_extent * 0.05, min(axis_extent * 0.05, offset))
    return 1.0, offset


def _select_uncovered_ocr_blocks(
    ocr_blocks: List[Dict],
    visual_blocks: List[Dict],
    clip: Optional[tuple],
) -> List[Dict]:
    """Keep every credible Thai OCR line not covered by the visual pass."""
    candidates = []
    for block in ocr_blocks:
        source_text = str(block.get("source_text") or "").strip()
        bbox = block.get("bbox")
        if not source_text or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if not re.search(r"[\u0E00-\u0E7F]", source_text):
            continue
        try:
            normalized_rect = fitz.Rect(*(float(value) for value in bbox))
        except (TypeError, ValueError):
            continue
        if normalized_rect.width < 8 or normalized_rect.height < 6:
            continue
        if clip is None:
            page_rect = normalized_rect
        else:
            clip_left, clip_top, clip_right, clip_bottom = clip
            clip_width = max(1.0, clip_right - clip_left)
            clip_height = max(1.0, clip_bottom - clip_top)
            page_rect = fitz.Rect(
                clip_left + normalized_rect.x0 / 1000.0 * clip_width,
                clip_top + normalized_rect.y0 / 1000.0 * clip_height,
                clip_left + normalized_rect.x1 / 1000.0 * clip_width,
                clip_top + normalized_rect.y1 / 1000.0 * clip_height,
            )
        if any(
            _layout_rect_overlap(page_rect, _layout_segment_rect(item)) >= 0.35
            for item in visual_blocks
        ):
            continue
        candidates.append({"source_text": source_text, "bbox": list(bbox)})

    candidates.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    deduplicated: List[Dict] = []
    for candidate in candidates:
        rect = fitz.Rect(*candidate["bbox"])
        if any(_layout_rect_overlap(rect, fitz.Rect(*existing["bbox"])) >= 0.55 for existing in deduplicated):
            continue
        deduplicated.append(candidate)
    return deduplicated


def _join_layout_pages(pages: Dict[int, List[str]], page_count: int) -> str:
    if page_count <= 1:
        return "\n\n".join(pages.get(1, [])).strip()
    return "\n\n".join(
        f"【第 {page_number} 页】\n"
        + "\n".join(pages.get(page_number, [])).strip()
        for page_number in range(1, page_count + 1)
        if pages.get(page_number)
    )


async def _translate_scanned_pdf(
    content: bytes,
    page_count: int,
    source_language: str,
    target_language: str,
    context: str,
    ocr_pages: Optional[List[int]] = None,
    progress_callback: ProgressCallback = None,
):
    page_numbers = sorted(
        set(range(1, page_count + 1) if ocr_pages is None else ocr_pages)
    )
    if len(page_numbers) > settings.pdf_ocr_max_pages:
        raise ValueError(
            f"扫描 PDF 共 {len(page_numbers)} 个图片页，当前最多处理 {settings.pdf_ocr_max_pages} 页，"
            "请拆分后上传"
        )

    page_outputs, warnings, failed_pages = await _translate_pdf_image_pages(
        content,
        page_numbers,
        source_language,
        target_language,
        context,
        progress_callback,
    )

    if not page_outputs:
        raise RuntimeError("扫描 PDF 未能识别出可翻译文字，请检查文件清晰度后重试")

    page_outputs.sort(key=lambda item: item[0])
    source_text = "\n\n".join(
        f"【第 {page_number} 页】\n{page_source.strip()}"
        for page_number, page_source, _ in page_outputs
    )
    translated_text = "\n\n".join(
        f"【第 {page_number} 页】\n{page_result.translated_text.strip()}"
        for page_number, _, page_result in page_outputs
    )
    warnings.insert(0, f"扫描 PDF 已通过视觉模型识别并翻译 {len(page_outputs)} 页")
    if failed_pages:
        warnings.append(
            "以下图片页未识别到可翻译文字：" + ", ".join(map(str, failed_pages))
        )

    detected_language = (
        detect_language(source_text) if source_language == "auto" else source_language
    )
    first_provider = page_outputs[0][2].provider
    return source_text, TranslationResult(
        source_language=detected_language,
        translated_text=translated_text,
        provider=f"{first_provider}:pdf-ocr",
        warnings=warnings,
    )


async def _translate_hybrid_pdf(
    content: bytes,
    document: ParsedDocument,
    source_language: str,
    target_language: str,
    context: str,
    progress_callback: ProgressCallback = None,
):
    text_pages = [
        (page_number, page_text.strip())
        for page_number, page_text in enumerate(document.page_texts, start=1)
        if page_number not in document.ocr_pages and page_text.strip()
    ]
    ocr_pages = sorted(document.ocr_pages)
    page_outputs = []
    warnings = []
    providers = []
    text_chunk_count = 0

    tasks = []
    if text_pages:
        tasks.append(
            (
                "text",
                asyncio.create_task(
                    _translate_text_page_chunks(
                        text_pages,
                        source_language,
                        target_language,
                        context,
                        progress_callback,
                    )
                ),
            )
        )
    if ocr_pages:
        tasks.append(
            (
                "image",
                asyncio.create_task(
                    _translate_pdf_image_pages(
                        content,
                        ocr_pages,
                        source_language,
                        target_language,
                        context,
                        progress_callback,
                    )
                ),
            )
        )

    task_results = await asyncio.gather(
        *(task for _, task in tasks),
        return_exceptions=True,
    )
    for (task_kind, _), task_result in zip(tasks, task_results):
        if isinstance(task_result, Exception):
            raise task_result
        if task_kind == "text":
            text_outputs, text_warnings, text_providers, text_chunk_count = task_result
            page_outputs.extend(text_outputs)
            warnings.extend(text_warnings)
            providers.extend(text_providers)
        else:
            image_outputs, image_warnings, failed_pages = task_result
            page_outputs.extend(image_outputs)
            warnings.extend(image_warnings)
            if failed_pages:
                warnings.append(
                    "以下图片页未识别到可翻译文字：" + ", ".join(map(str, failed_pages))
                )
            providers.extend(page_result.provider for _, _, page_result in image_outputs)

    if not page_outputs:
        raise RuntimeError("PDF 中没有可翻译的文案或图片内容")

    page_outputs.sort(key=lambda item: item[0])
    source_text = "\n\n".join(
        f"【第 {page_number} 页】\n{page_source.strip()}"
        for page_number, page_source, _ in page_outputs
    )
    translated_text = "\n\n".join(
        f"【第 {page_number} 页】\n{page_result.translated_text.strip()}"
        for page_number, _, page_result in page_outputs
    )
    warnings.insert(
        0,
        f"PDF 已按页面类型处理：文案页 {len(text_pages)} 页，图片页 {len(ocr_pages)} 页；"
        f"文案页分为 {text_chunk_count} 个请求",
    )
    unique_providers = list(dict.fromkeys(providers))
    route_suffix = "pdf-mixed" if ocr_pages else "pdf-text"
    return source_text, TranslationResult(
        source_language=(
            detect_language(source_text) if source_language == "auto" else source_language
        ),
        translated_text=translated_text,
        provider=f"{'+'.join(unique_providers)}:{route_suffix}",
        warnings=list(dict.fromkeys(warnings)),
    )


async def _translate_text_page_chunks(
    text_pages: List[tuple],
    source_language: str,
    target_language: str,
    context: str,
    progress_callback: ProgressCallback = None,
):
    chunks = [
        text_pages[start : start + PDF_TEXT_PAGE_CHUNK_SIZE]
        for start in range(0, len(text_pages), PDF_TEXT_PAGE_CHUNK_SIZE)
    ]

    async def translate_chunk(chunk):
        text_result = await _translate_text_with_knowledge(
            "\n\n".join(
                f"【第 {page_number} 页】\n{page_text}"
                for page_number, page_text in chunk
            ),
            source_language,
            target_language,
            context,
            preserve_page_markers=True,
        )
        translated_pages = _split_page_translations(text_result.translated_text)
        if all(page_number in translated_pages for page_number, _ in chunk):
            if progress_callback:
                progress_callback(len(chunk), "正在翻译文案页")
            return (
                [
                    (
                        page_number,
                        page_text,
                        TranslationResult(
                            source_language=text_result.source_language,
                            translated_text=translated_pages[page_number],
                            provider=text_result.provider,
                            warnings=text_result.warnings,
                        ),
                    )
                    for page_number, page_text in chunk
                ],
                list(text_result.warnings),
                [text_result.provider],
            )

        start_page = chunk[0][0]
        end_page = chunk[-1][0]
        retry_warning = (
            f"第 {start_page}-{end_page} 页翻译未保留页码标记，"
            "已自动按单页重试以保持页码对应"
        )

        async def translate_single_page(page_number, page_text):
            page_result = await _translate_text_with_knowledge(
                page_text,
                source_language,
                target_language,
                context,
            )
            if progress_callback:
                progress_callback(1, "正在按单页校正页码")
            return page_number, page_text, page_result

        single_page_outputs = await asyncio.gather(
            *[
                translate_single_page(page_number, page_text)
                for page_number, page_text in chunk
            ]
        )
        retry_warnings = list(text_result.warnings)
        retry_warnings.append(retry_warning)
        retry_providers = []
        for _, _, page_result in single_page_outputs:
            retry_warnings.extend(page_result.warnings)
            retry_providers.append(page_result.provider)
        return single_page_outputs, retry_warnings, retry_providers

    translated_chunks = await asyncio.gather(
        *[translate_chunk(chunk) for chunk in chunks]
    )

    page_outputs = []
    warnings = []
    providers = []
    for chunk_outputs, chunk_warnings, chunk_providers in translated_chunks:
        page_outputs.extend(chunk_outputs)
        warnings.extend(chunk_warnings)
        providers.extend(chunk_providers)

    return page_outputs, list(dict.fromkeys(warnings)), providers, len(chunks)


async def _translate_pdf_image_pages(
    content: bytes,
    page_numbers: List[int],
    source_language: str,
    target_language: str,
    context: str,
    progress_callback: ProgressCallback = None,
):
    page_outputs = []
    warnings = []
    failed_pages = []
    loop = asyncio.get_running_loop()
    concurrency = settings.pdf_ocr_concurrency

    for start in range(0, len(page_numbers), concurrency):
        batch_numbers = page_numbers[start : start + concurrency]
        rendered_pages = await loop.run_in_executor(
            None,
            render_pdf_pages,
            content,
            batch_numbers,
            settings.pdf_ocr_desired_width,
        )

        async def translate_page(page):
            try:
                translated = await _translate_image_with_knowledge(
                    page.content,
                    "image/png",
                    source_language,
                    target_language,
                    context,
                    allow_same_language=source_language == "auto",
                )
                return page, translated
            except Exception as exc:
                return page, exc
            finally:
                if progress_callback:
                    progress_callback(1, "正在识别图片页")

        translated_pages = await asyncio.gather(
            *[translate_page(page) for page in rendered_pages]
        )

        for page, translated in translated_pages:
            if isinstance(translated, Exception):
                failed_pages.append(page.page_number)
                continue
            page_source, page_result = translated
            page_outputs.append((page.page_number, page_source, page_result))
            for warning in page_result.warnings:
                if warning not in warnings:
                    warnings.append(warning)

    return page_outputs, warnings, failed_pages


def _split_page_translations(text: str) -> Dict[int, str]:
    marker = re.compile(r"【第\s*(\d+)\s*页】")
    matches = list(marker.finditer(text))
    pages = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages[int(match.group(1))] = text[match.end() : end].strip()
    return pages


@app.get("/api/history", response_model=List[TranslationResponse])
async def list_history(
    request: Request,
    limit: int = Query(30, ge=1, le=100),
) -> List[TranslationResponse]:
    owner_id = _request_owner_id(request)
    return [
        TranslationResponse(**item)
        for item in database.list_history(owner_id, limit)
    ]


@app.get("/api/history/{item_id}/export")
async def download_history_export(
    item_id: str,
    request: Request,
    inline: bool = Query(False),
):
    item = database.get_history(item_id, _request_owner_id(request))
    if item is None:
        raise HTTPException(status_code=404, detail="翻译记录不存在")
    export_filename = item.get("export_filename")
    if not export_filename:
        raise HTTPException(status_code=404, detail="该记录没有可下载的格式化译文")
    extension = Path(export_filename).suffix.lower()
    if extension not in {".pdf", ".docx", ".pptx", ".txt", ".md"}:
        raise HTTPException(status_code=404, detail="译文文件格式无效")
    export_path = _export_directory() / f"{item_id}{extension}"
    if not export_path.is_file():
        raise HTTPException(status_code=404, detail="译文文件不存在或已被清理")
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
    }
    disposition = "inline" if inline and extension == ".pdf" else "attachment"
    fallback_filename = f"translation{extension}"
    content_disposition = (
        f'{disposition}; filename="{fallback_filename}"; '
        f"filename*=UTF-8''{quote(export_filename)}"
    )
    return FileResponse(
        export_path,
        media_type=media_types[extension],
        headers={
            "Content-Disposition": content_disposition,
            "Cache-Control": "private, max-age=86400",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/history/{item_id}/export/text")
async def download_history_text_export(item_id: str, request: Request):
    item = database.get_history(item_id, _request_owner_id(request))
    if item is None:
        raise HTTPException(status_code=404, detail="翻译记录不存在")
    source_filename = item.get("filename") or "translation"
    export_filename = f"{Path(source_filename).stem}-译文.txt"
    content_disposition = (
        'attachment; filename="translation.txt"; '
        f"filename*=UTF-8''{quote(export_filename)}"
    )
    return Response(
        content=("\ufeff" + item["translated_text"].strip() + "\n").encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": content_disposition,
            "Cache-Control": "private, max-age=86400",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/history/{item_id}/export/unformatted")
def download_history_unformatted_export(
    item_id: str,
    request: Request,
    inline: bool = Query(False),
):
    item = database.get_history(item_id, _request_owner_id(request))
    if item is None:
        raise HTTPException(status_code=404, detail="翻译记录不存在")
    source_filename = item.get("filename")
    if not source_filename:
        raise HTTPException(status_code=404, detail="该记录没有可下载的文档译文")
    extension = Path(source_filename).suffix.lower()
    if extension not in {".pdf", ".docx"}:
        raise HTTPException(status_code=404, detail="该文档格式不支持未排版导出")

    export_directory = _export_directory()
    formatted_path = export_directory / f"{item_id}{extension}"
    source_path = export_directory / f"{item_id}-source{extension}"
    export_source_path = (
        source_path
        if source_path.is_file()
        else formatted_path
        if formatted_path.is_file()
        else None
    )
    if export_source_path is None:
        raise HTTPException(status_code=404, detail="译文文件不存在或已被清理")

    unformatted_path = export_directory / f"{item_id}-unformatted{extension}"
    unformatted_filename = f"{Path(source_filename).stem}-未排版译文{extension}"
    if not unformatted_path.is_file():
        try:
            exported = create_unformatted_document_export(
                item_id=item_id,
                source_filename=source_filename,
                source_content=export_source_path.read_bytes(),
                translated_text=item["translated_text"],
                target_language=item["target_language"],
                output_directory=export_directory,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        unformatted_path = exported.path
        unformatted_filename = exported.filename

    disposition = "inline" if inline and extension == ".pdf" else "attachment"
    fallback_filename = f"translation-unformatted{extension}"
    content_disposition = (
        f'{disposition}; filename="{fallback_filename}"; '
        f"filename*=UTF-8''{quote(unformatted_filename)}"
    )
    media_type = (
        "application/pdf"
        if extension == ".pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(
        unformatted_path,
        media_type=media_type,
        headers={
            "Content-Disposition": content_disposition,
            "Cache-Control": "private, max-age=86400",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/knowledge", response_model=KnowledgeListResponse)
async def list_knowledge(
    limit: int = Query(100, ge=1, le=500),
) -> KnowledgeListResponse:
    return KnowledgeListResponse(**database.list_knowledge(limit))


@app.get("/api/knowledge/template")
async def download_knowledge_template() -> FileResponse:
    template_path = Path(__file__).resolve().parent / "assets" / "knowledge-template.xlsx"
    return FileResponse(
        template_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="knowledge-template.xlsx",
    )


@app.post("/api/knowledge/import", response_model=KnowledgeImportResponse)
async def import_knowledge(file: UploadFile = File(...)) -> KnowledgeImportResponse:
    filename = _safe_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise HTTPException(status_code=422, detail="知识库仅支持 CSV 或 XLSX 文件")
    content = await _read_upload(file)
    try:
        if suffix == ".xlsx":
            rows, total_rows, invalid_rows = parse_knowledge_xlsx(content)
        else:
            rows, total_rows, invalid_rows = _parse_knowledge_csv(content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    counts = database.import_knowledge(rows, filename)
    return KnowledgeImportResponse(
        filename=filename,
        total_rows=total_rows,
        invalid_rows=invalid_rows,
        **counts,
    )


@app.post("/api/knowledge/entries", response_model=KnowledgeEntryResponse)
async def upsert_knowledge_entry(
    payload: KnowledgeUpsertRequest,
) -> KnowledgeEntryResponse:
    thai_text = payload.thai_text.strip()
    chinese_text = (payload.chinese_text or "").strip() or None
    english_text = (payload.english_text or "").strip() or None
    if not thai_text:
        raise HTTPException(status_code=422, detail="泰语内容不能为空")
    if not chinese_text and not english_text:
        raise HTTPException(status_code=422, detail="中文和英语至少填写一项")
    item = database.upsert_knowledge_item(
        thai_text=thai_text,
        chinese_text=chinese_text,
        english_text=english_text,
    )
    return KnowledgeEntryResponse(**item)


def _save_result(
    *,
    owner_id: str,
    kind: str,
    source_text: str,
    target_language: str,
    result: TranslationResult,
    filename: Optional[str] = None,
) -> TranslationResponse:
    item = database.add_history(
        {
            "owner_id": owner_id,
            "kind": kind,
            "source_language": result.source_language,
            "target_language": target_language,
            "source_text": source_text.strip(),
            "translated_text": result.translated_text,
            "provider": result.provider,
            "warnings": result.warnings,
            "filename": filename,
        }
    )
    return TranslationResponse(**item)


async def _save_document_result(
    *,
    owner_id: str,
    kind: str,
    source_text: str,
    target_language: str,
    result: TranslationResult,
    filename: str,
    source_content: bytes,
) -> TranslationResponse:
    response = _save_result(
        owner_id=owner_id,
        kind=kind,
        source_text=source_text,
        target_language=target_language,
        result=result,
        filename=filename,
    )
    try:
        exported = await asyncio.get_running_loop().run_in_executor(
            None,
            partial(
                create_document_export,
                item_id=response.id,
                source_filename=filename,
                source_content=source_content,
                translated_text=result.translated_text,
                target_language=target_language,
                output_directory=_export_directory(),
                layout_segments=result.layout_segments,
                prepared_pdf_content=result.prepared_pdf_content,
            ),
        )
        item = database.set_history_export(response.id, exported.filename)
    except Exception:
        logger.exception("Failed to create document export for history %s", response.id)
        extension = Path(filename).suffix.lower()
        if extension in {".pdf", ".docx"}:
            try:
                export_directory = _export_directory()
                export_directory.mkdir(parents=True, exist_ok=True)
                (export_directory / f"{response.id}-source{extension}").write_bytes(
                    source_content
                )
            except Exception:
                logger.exception(
                    "Failed to preserve source document for history %s",
                    response.id,
                )
        item = database.add_history_warning(
            response.id, "译文已生成，但格式化文件导出失败，请稍后重试"
        )
    return TranslationResponse(**item)


async def _translate_text_with_knowledge(
    text: str,
    source_language: str,
    target_language: str,
    context: str = "",
    preserve_page_markers: bool = False,
) -> TranslationResult:
    detected = detect_language(text) if source_language == "auto" else source_language
    if not preserve_page_markers:
        exact = database.find_exact_knowledge(detected, target_language, text)
        if exact:
            return TranslationResult(
                source_language=detected,
                translated_text=normalize_translation_text(exact["translated_text"]),
                provider="knowledge-base",
                warnings=["原文完整命中知识库，已直接使用既有译文"],
            )

    entries = database.find_matching_knowledge(
        detected, target_language, text, limit=40
    )
    knowledge_context = _merge_knowledge_context(context, entries)
    result = await translator.translate(
        text,
        detected,
        target_language,
        knowledge_context,
        preserve_page_markers,
    )
    if entries:
        result.provider = f"{result.provider}+knowledge-base"
        result.warnings.insert(0, f"已优先应用知识库匹配项 {len(entries)} 条")
    return result


async def _translate_image_with_knowledge(
    content: bytes,
    media_type: str,
    source_language: str,
    target_language: str,
    context: str = "",
    allow_same_language: bool = False,
):
    entries = database.list_knowledge_for_direction(
        target_language,
        None if source_language == "auto" else source_language,
        limit=60,
    )
    knowledge_context = _merge_knowledge_context(context, entries)
    source_text, result = await translator.translate_image(
        content,
        media_type,
        source_language,
        target_language,
        knowledge_context,
        allow_same_language,
    )
    exact = database.find_exact_knowledge(
        result.source_language, target_language, source_text
    )
    if exact:
        result.translated_text = normalize_translation_text(exact["translated_text"])
        result.provider = "knowledge-base:vision-ocr"
        result.warnings.insert(0, "图片原文完整命中知识库，已使用既有译文")
        return source_text, result

    matched_entries = database.find_matching_knowledge(
        result.source_language, target_language, source_text, limit=40
    )
    if matched_entries:
        result.provider = f"{result.provider}+knowledge-base"
        result.warnings.insert(
            0, f"图片原文命中知识库术语 {len(matched_entries)} 条"
        )
    return source_text, result


def _merge_knowledge_context(context: str, entries: List[Dict]) -> str:
    if not entries:
        return context
    lines = []
    used_characters = 0
    for entry in entries:
        line = f'{entry["source_text"]} => {entry["translated_text"]}'
        if used_characters + len(line) > KNOWLEDGE_CONTEXT_MAX_CHARACTERS:
            break
        lines.append(line)
        used_characters += len(line)
    if not lines:
        return context
    knowledge_block = (
        "知识库中已有以下对应译法。仅在原文出现对应词语时优先采用：\n"
        + "\n".join(lines)
    )
    return f"{context}\n\n{knowledge_block}".strip()


def _parse_knowledge_csv(content: bytes):
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("知识库 CSV 必须使用 UTF-8 编码") from exc
    reader = csv.DictReader(StringIO(decoded))
    required_fields = {
        "source_language",
        "target_language",
        "source_text",
        "translated_text",
    }
    fields = {field.strip() for field in (reader.fieldnames or []) if field}
    if not required_fields.issubset(fields):
        raise ValueError(
            "CSV 表头必须包含 source_language、target_language、source_text、translated_text"
        )

    rows = []
    total_rows = 0
    invalid_rows = 0
    for raw_row in reader:
        if not raw_row or not any(str(value or "").strip() for value in raw_row.values()):
            continue
        total_rows += 1
        if total_rows > KNOWLEDGE_IMPORT_MAX_ROWS:
            raise ValueError(
                f"单次最多导入 {KNOWLEDGE_IMPORT_MAX_ROWS:,} 条知识库记录"
            )
        row = {key.strip(): str(value or "").strip() for key, value in raw_row.items() if key}
        source = row.get("source_language", "").lower()
        target = row.get("target_language", "").lower()
        source_text = row.get("source_text", "")
        translated_text = row.get("translated_text", "")
        is_valid = (
            source in {"zh", "th", "en"}
            and target in {"zh", "th", "en"}
            and source != target
            and "th" in {source, target}
            and 0 < len(source_text) <= 2000
            and 0 < len(translated_text) <= 5000
        )
        if not is_valid:
            invalid_rows += 1
            continue
        rows.append(
            {
                "source_language": source,
                "target_language": target,
                "source_text": source_text,
                "translated_text": translated_text,
            }
        )
    return rows, total_rows, invalid_rows


def _export_directory() -> Path:
    return settings.database_path.parent / "exports"


async def _create_document_attempt(
    *,
    owner_id: str,
    attempt_id: str,
    filename: str,
    content_type: str,
    content: bytes,
    source_language: str,
    target_language: str,
) -> Dict[str, str]:
    """Archive source bytes before any parser or model can reject the file."""
    try:
        archived = await asyncio.get_running_loop().run_in_executor(
            None,
            partial(
                _archive_document_source,
                attempt_id=attempt_id,
                filename=filename,
                content=content,
            ),
        )
    except OSError as exc:
        logger.exception("Unable to archive uploaded document %s", attempt_id)
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail="无法保存原始文件，请稍后重试",
        ) from exc

    try:
        database.create_document_attempt(
            {
                "id": attempt_id,
                "owner_id": owner_id,
                "filename": filename,
                "content_type": content_type,
                "source_path": archived["source_path"],
                "content_sha256": archived["content_sha256"],
                "content_bytes": str(len(content)),
                "source_language": source_language,
                "target_language": target_language,
                "status": "processing",
                "stage": "preflight",
            }
        )
    except Exception as exc:
        logger.exception("Unable to persist document diagnostics for %s", attempt_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="无法创建文档诊断记录，请稍后重试",
        ) from exc

    _log_document_event(
        "upload_archived",
        attempt_id=attempt_id,
        filename=filename,
        content_type=content_type,
        content_bytes=len(content),
        content_sha256=archived["content_sha256"],
        source_path=archived["source_path"],
    )
    return archived


def _archive_document_source(
    *, attempt_id: str, filename: str, content: bytes
) -> Dict[str, str]:
    """Write the source into a private, deterministic per-attempt directory."""
    extension = Path(filename).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", extension):
        extension = ".bin"
    archive_directory = settings.upload_source_directory / attempt_id
    archive_directory.mkdir(parents=True, exist_ok=False)
    os.chmod(archive_directory, 0o700)
    destination = archive_directory / f"source{extension}"
    temporary = archive_directory / ".source-uploading"
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "source_path": str(destination.relative_to(settings.upload_source_directory)),
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }


def _update_document_attempt_safely(attempt_id: str, **values) -> None:
    if not attempt_id:
        return
    try:
        database.update_document_attempt(attempt_id, **values)
    except Exception:
        # A diagnostic storage outage must not destroy a translation that has
        # already started. The normal application log still records the issue.
        logger.exception("Unable to update document diagnostics for %s", attempt_id)


def _record_document_attempt_failure(
    attempt_id: str, stage: str, exc: Exception
) -> None:
    if not attempt_id:
        return
    error_trace = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    _update_document_attempt_safely(
        attempt_id,
        status="failed",
        stage=stage,
        error_type=type(exc).__name__,
        error_message=str(exc),
        error_trace=error_trace,
        completed=True,
    )


async def _read_upload(file: UploadFile) -> bytes:
    content = await file.read(settings.max_upload_bytes + 1)
    if not content:
        raise HTTPException(status_code=422, detail="上传文件为空")
    if len(content) > settings.max_upload_bytes:
        max_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"文件不能超过 {max_mb} MB")
    return content


def _safe_filename(filename: Optional[str] = None) -> str:
    return Path(filename or "upload").name


def _validate_languages(source_language: str, target_language: str) -> None:
    if source_language not in {"auto", "zh", "th", "en"}:
        raise HTTPException(status_code=422, detail="不支持的源语言")
    if target_language not in {"zh", "th", "en"}:
        raise HTTPException(status_code=422, detail="不支持的目标语言")
    if source_language == target_language:
        raise HTTPException(status_code=422, detail="源语言和目标语言不能相同")


web_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
if web_dist.exists():
    assets_dir = web_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_h5(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="接口不存在")
        candidate = (web_dist / full_path).resolve()
        if full_path and candidate.is_file() and web_dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(web_dist / "index.html")
