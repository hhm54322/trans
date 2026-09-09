from typing import List, Literal, Optional

from pydantic import BaseModel, Field


Language = Literal["auto", "zh", "th", "en"]
TargetLanguage = Literal["zh", "th", "en"]


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)
    source_language: Language = "auto"
    target_language: TargetLanguage
    context: str = Field(default="", max_length=5000)


class TranslationResponse(BaseModel):
    id: str
    source_language: TargetLanguage
    target_language: TargetLanguage
    source_text: str
    translated_text: str
    provider: str
    warnings: List[str] = Field(default_factory=list)
    filename: Optional[str] = None
    export_filename: Optional[str] = None
    kind: Literal["text", "image", "document"] = "text"
    created_at: str


class DocumentJobResponse(BaseModel):
    job_id: str
    status: Literal["processing", "completed", "failed"]
    stage: str
    completed_pages: int = Field(ge=0)
    total_pages: int = Field(ge=1)
    progress: int = Field(ge=0, le=100)
    message: str
    error: Optional[str] = None
    result: Optional[TranslationResponse] = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    provider: str
    model: str


class KnowledgeEntryResponse(BaseModel):
    id: str
    thai_text: str
    chinese_text: Optional[str] = None
    english_text: Optional[str] = None
    source_filename: str
    created_at: str
    updated_at: str


class KnowledgeListResponse(BaseModel):
    total: int = Field(ge=0)
    entries: List[KnowledgeEntryResponse] = Field(default_factory=list)


class KnowledgeImportResponse(BaseModel):
    filename: str
    total_rows: int = Field(ge=0)
    inserted: int = Field(ge=0)
    updated: int = Field(ge=0)
    invalid_rows: int = Field(ge=0)


class KnowledgeUpsertRequest(BaseModel):
    thai_text: str = Field(min_length=1, max_length=2000)
    chinese_text: Optional[str] = Field(default=None, max_length=5000)
    english_text: Optional[str] = Field(default=None, max_length=5000)
