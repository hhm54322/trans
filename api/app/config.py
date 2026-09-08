from dataclasses import dataclass
import os
from pathlib import Path
from typing import List


_ENV_FILE_VALUES = {}


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        normalized_value = value.strip()
        _ENV_FILE_VALUES[normalized_key] = normalized_value
        os.environ.setdefault(normalized_key, normalized_value)


_load_env_file()


def _openai_setting(name: str, default: str = "") -> str:
    # Project connection settings must not be shadowed by unrelated global
    # OPENAI_* variables inherited from the desktop application.
    return _ENV_FILE_VALUES.get(name, os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    app_env: str
    database_path: Path
    upload_source_directory: Path
    document_log_path: Path
    cors_origins: List[str]
    max_upload_bytes: int
    max_document_characters: int
    ai_provider: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    openai_vision_model: str
    openai_api_mode: str
    openai_reasoning_effort: str
    openai_max_concurrency: int
    openai_max_retries: int
    openai_retry_base_seconds: float
    pdf_ocr_max_pages: int
    pdf_ocr_desired_width: int
    pdf_ocr_concurrency: int


def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    raw_database_path = Path(os.getenv("APP_DATABASE_PATH", "./data/app.db"))
    if not raw_database_path.is_absolute():
        raw_database_path = repo_root / raw_database_path
    raw_upload_source_directory = Path(
        os.getenv("APP_UPLOAD_SOURCE_DIRECTORY", str(raw_database_path.parent / "uploads"))
    )
    if not raw_upload_source_directory.is_absolute():
        raw_upload_source_directory = repo_root / raw_upload_source_directory
    raw_document_log_path = Path(
        os.getenv(
            "APP_DOCUMENT_LOG_PATH",
            str(raw_database_path.parent / "logs" / "document-jobs.log"),
        )
    )
    if not raw_document_log_path.is_absolute():
        raw_document_log_path = repo_root / raw_document_log_path
    max_upload_mb = max(1, min(500, int(os.getenv("APP_MAX_UPLOAD_MB", "200"))))
    max_document_characters = max(
        50_000,
        min(2_000_000, int(os.getenv("APP_MAX_DOCUMENT_CHARACTERS", "1000000"))),
    )
    pdf_ocr_max_pages = max(1, min(300, int(os.getenv("APP_PDF_OCR_MAX_PAGES", "50"))))
    pdf_ocr_desired_width = max(
        900, min(2400, int(os.getenv("APP_PDF_OCR_DESIRED_WIDTH", "1600")))
    )
    pdf_ocr_concurrency = max(
        1, min(4, int(os.getenv("APP_PDF_OCR_CONCURRENCY", "2")))
    )
    openai_max_concurrency = max(
        1, min(10, int(_openai_setting("OPENAI_MAX_CONCURRENCY", "10")))
    )
    openai_max_retries = max(
        0, min(6, int(_openai_setting("OPENAI_MAX_RETRIES", "3")))
    )
    openai_retry_base_seconds = max(
        0.1,
        min(10.0, float(_openai_setting("OPENAI_RETRY_BASE_SECONDS", "1.0"))),
    )
    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        database_path=raw_database_path,
        upload_source_directory=raw_upload_source_directory,
        document_log_path=raw_document_log_path,
        cors_origins=[
            origin.strip()
            for origin in os.getenv("APP_CORS_ORIGINS", "http://localhost:5173").split(",")
            if origin.strip()
        ],
        max_upload_bytes=max_upload_mb * 1024 * 1024,
        max_document_characters=max_document_characters,
        ai_provider=os.getenv("AI_PROVIDER", "demo").lower(),
        openai_api_key=_openai_setting("OPENAI_API_KEY"),
        openai_base_url=_openai_setting(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/"),
        openai_model=_openai_setting("OPENAI_MODEL", "gpt-5.6-terra"),
        openai_vision_model=_openai_setting("OPENAI_VISION_MODEL", "gpt-5.6-terra"),
        openai_api_mode=_openai_setting("OPENAI_API_MODE", "auto").lower(),
        openai_reasoning_effort=_openai_setting(
            "OPENAI_REASONING_EFFORT", "none"
        ).lower(),
        openai_max_concurrency=openai_max_concurrency,
        openai_max_retries=openai_max_retries,
        openai_retry_base_seconds=openai_retry_base_seconds,
        pdf_ocr_max_pages=pdf_ocr_max_pages,
        pdf_ocr_desired_width=pdf_ocr_desired_width,
        pdf_ocr_concurrency=pdf_ocr_concurrency,
    )


settings = get_settings()
