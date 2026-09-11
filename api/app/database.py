import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS translation_history (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT,
                    kind TEXT NOT NULL,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    warnings TEXT NOT NULL DEFAULT '[]',
                    filename TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column(
                connection, "translation_history", "export_filename", "TEXT"
            )
            self._ensure_column(connection, "translation_history", "owner_id", "TEXT")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_translation_history_owner_created_at
                ON translation_history(owner_id, created_at DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_attempts (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    content_bytes INTEGER NOT NULL,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    total_pages INTEGER,
                    completed_pages INTEGER NOT NULL DEFAULT 0,
                    history_id TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    error_trace TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            self._ensure_column(connection, "document_attempts", "owner_id", "TEXT")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_document_attempts_created_at
                ON document_attempts(created_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_document_attempts_status
                ON document_attempts(status, updated_at DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id TEXT PRIMARY KEY,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    source_normalized TEXT NOT NULL,
                    translation_normalized TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(source_language, target_language, source_normalized)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_direction
                ON knowledge_entries(source_language, target_language)
                """
            )
            self._ensure_column(connection, "knowledge_entries", "thai_normalized", "TEXT")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_thai
                ON knowledge_entries(thai_normalized)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id TEXT PRIMARY KEY,
                    thai_text TEXT NOT NULL,
                    thai_normalized TEXT NOT NULL UNIQUE,
                    chinese_text TEXT,
                    english_text TEXT,
                    source_filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._migrate_knowledge_items(connection)

    def create_document_attempt(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Persist metadata for every uploaded document before parsing starts."""
        now = _now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO document_attempts
                (id, owner_id, filename, content_type, source_path, content_sha256, content_bytes,
                 source_language, target_language, status, stage, total_pages,
                 completed_pages, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    values["id"],
                    values["owner_id"],
                    values["filename"],
                    values.get("content_type", ""),
                    values["source_path"],
                    values["content_sha256"],
                    int(values["content_bytes"]),
                    values["source_language"],
                    values["target_language"],
                    values.get("status", "processing"),
                    values.get("stage", "queued"),
                    values.get("total_pages"),
                    int(values.get("completed_pages", 0)),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM document_attempts WHERE id = ?", (values["id"],)
            ).fetchone()
        return dict(row)

    def update_document_attempt(
        self,
        attempt_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        total_pages: Optional[int] = None,
        completed_pages: Optional[int] = None,
        history_id: Optional[str] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        error_trace: Optional[str] = None,
        completed: bool = False,
    ) -> Dict[str, Any]:
        fields: Dict[str, Any] = {"updated_at": _now()}
        for name, value in {
            "status": status,
            "stage": stage,
            "total_pages": total_pages,
            "completed_pages": completed_pages,
            "history_id": history_id,
            "error_type": error_type,
            "error_message": error_message,
            "error_trace": error_trace,
        }.items():
            if value is not None:
                fields[name] = value
        if completed:
            fields["completed_at"] = fields["updated_at"]
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE document_attempts SET {assignments} WHERE id = ?",
                [*fields.values(), attempt_id],
            )
            row = connection.execute(
                "SELECT * FROM document_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
        if row is None:
            raise ValueError("文档诊断记录不存在")
        return dict(row)

    def get_document_attempt(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM document_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
        return dict(row) if row else None

    def add_history(self, values: Dict[str, Any]) -> Dict[str, Any]:
        item_id = str(uuid4())
        created_at = _now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO translation_history
                (id, owner_id, kind, source_language, target_language, source_text, translated_text,
                 provider, warnings, filename, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    values["owner_id"],
                    values["kind"],
                    values["source_language"],
                    values["target_language"],
                    values["source_text"],
                    values["translated_text"],
                    values["provider"],
                    json.dumps(values.get("warnings", []), ensure_ascii=False),
                    values.get("filename"),
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM translation_history WHERE id = ?", (item_id,)
            ).fetchone()
        return self._history_row(row)

    def list_history(self, owner_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM translation_history
                WHERE owner_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (owner_id, limit),
            ).fetchall()
        return [self._history_row(row) for row in rows]

    def get_history(self, item_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM translation_history
                WHERE id = ? AND owner_id = ?
                """,
                (item_id, owner_id),
            ).fetchone()
        return self._history_row(row) if row else None

    def set_history_export(self, item_id: str, export_filename: str) -> Dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                "UPDATE translation_history SET export_filename = ? WHERE id = ?",
                (export_filename, item_id),
            )
            row = connection.execute(
                "SELECT * FROM translation_history WHERE id = ?", (item_id,)
            ).fetchone()
        if row is None:
            raise ValueError("翻译记录不存在")
        return self._history_row(row)

    def add_history_warning(self, item_id: str, warning: str) -> Dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT warnings FROM translation_history WHERE id = ?", (item_id,)
            ).fetchone()
            if row is None:
                raise ValueError("翻译记录不存在")
            warnings = json.loads(row["warnings"])
            if warning not in warnings:
                warnings.append(warning)
            connection.execute(
                "UPDATE translation_history SET warnings = ? WHERE id = ?",
                (json.dumps(warnings, ensure_ascii=False), item_id),
            )
            updated = connection.execute(
                "SELECT * FROM translation_history WHERE id = ?", (item_id,)
            ).fetchone()
        return self._history_row(updated)

    def import_knowledge(
        self, rows: List[Dict[str, str]], source_filename: str
    ) -> Dict[str, int]:
        inserted = 0
        updated = 0
        with self.connect() as connection:
            for values in rows:
                multilingual = (
                    values
                    if values.get("th")
                    else _knowledge_row_to_multilingual(values)
                )
                if multilingual is None:
                    continue
                _, created = self._upsert_knowledge_item(
                    connection,
                    thai_text=multilingual["th"],
                    chinese_text=multilingual.get("zh"),
                    english_text=multilingual.get("en"),
                    source_filename=source_filename,
                )
                if created:
                    inserted += 1
                else:
                    updated += 1
        return {
            "inserted": inserted,
            "updated": updated,
        }

    def upsert_knowledge_item(
        self,
        *,
        thai_text: str,
        chinese_text: Optional[str],
        english_text: Optional[str],
        source_filename: str = "手动录入",
    ) -> Dict[str, Any]:
        with self.connect() as connection:
            item, _ = self._upsert_knowledge_item(
                connection,
                thai_text=thai_text,
                chinese_text=chinese_text,
                english_text=english_text,
                source_filename=source_filename,
            )
        return item

    def list_knowledge(self, limit: int = 100) -> Dict[str, Any]:
        with self.connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM knowledge_items"
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT id, thai_text, chinese_text, english_text,
                       source_filename, created_at, updated_at
                FROM knowledge_items
                ORDER BY updated_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return {"total": total, "entries": [dict(row) for row in rows]}

    def _upsert_knowledge_item(
        self,
        connection: sqlite3.Connection,
        *,
        thai_text: str,
        chinese_text: Optional[str],
        english_text: Optional[str],
        source_filename: str,
    ) -> tuple:
        thai_text = thai_text.strip()
        chinese_text = (chinese_text or "").strip() or None
        english_text = (english_text or "").strip() or None
        thai_normalized = normalize_knowledge_text(thai_text)
        existing = connection.execute(
            "SELECT * FROM knowledge_items WHERE thai_normalized = ?",
            (thai_normalized,),
        ).fetchone()
        now = _now()
        if existing:
            item_id = existing["id"]
            chinese_text = chinese_text or existing["chinese_text"]
            english_text = english_text or existing["english_text"]
            connection.execute(
                """
                UPDATE knowledge_items
                SET thai_text = ?, chinese_text = ?, english_text = ?,
                    source_filename = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    thai_text,
                    chinese_text,
                    english_text,
                    source_filename,
                    now,
                    item_id,
                ),
            )
            created = False
        else:
            item_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO knowledge_items
                (id, thai_text, thai_normalized, chinese_text, english_text,
                 source_filename, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    thai_text,
                    thai_normalized,
                    chinese_text,
                    english_text,
                    source_filename,
                    now,
                    now,
                ),
            )
            created = True

        self._rebuild_directional_entries(
            connection,
            thai_normalized=thai_normalized,
            texts={"th": thai_text, "zh": chinese_text, "en": english_text},
            source_filename=source_filename,
            created_at=now,
        )
        row = connection.execute(
            """
            SELECT id, thai_text, chinese_text, english_text,
                   source_filename, created_at, updated_at
            FROM knowledge_items WHERE id = ?
            """,
            (item_id,),
        ).fetchone()
        return dict(row), created

    @staticmethod
    def _rebuild_directional_entries(
        connection: sqlite3.Connection,
        *,
        thai_normalized: str,
        texts: Dict[str, Optional[str]],
        source_filename: str,
        created_at: str,
    ) -> None:
        connection.execute(
            "DELETE FROM knowledge_entries WHERE thai_normalized = ?",
            (thai_normalized,),
        )
        available = [(language, text) for language, text in texts.items() if text]
        for source_language, source_text in available:
            for target_language, translated_text in available:
                if source_language == target_language:
                    continue
                connection.execute(
                    """
                    INSERT INTO knowledge_entries
                    (id, source_language, target_language, source_text, translated_text,
                     source_normalized, translation_normalized, source_filename,
                     created_at, thai_normalized)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_language, target_language, source_normalized)
                    DO UPDATE SET
                        translated_text = excluded.translated_text,
                        translation_normalized = excluded.translation_normalized,
                        source_filename = excluded.source_filename,
                        created_at = excluded.created_at,
                        thai_normalized = excluded.thai_normalized
                    """,
                    (
                        str(uuid4()),
                        source_language,
                        target_language,
                        source_text,
                        translated_text,
                        normalize_knowledge_text(source_text),
                        normalize_knowledge_text(translated_text),
                        source_filename,
                        created_at,
                        thai_normalized,
                    ),
                )

    def _migrate_knowledge_items(self, connection: sqlite3.Connection) -> None:
        if connection.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[0]:
            return
        rows = connection.execute(
            """
            SELECT source_language, target_language, source_text, translated_text,
                   source_filename
            FROM knowledge_entries
            WHERE source_language = 'th' OR target_language = 'th'
            ORDER BY created_at ASC, rowid ASC
            """
        ).fetchall()
        for row in rows:
            multilingual = _knowledge_row_to_multilingual(dict(row))
            if multilingual:
                self._upsert_knowledge_item(
                    connection,
                    thai_text=multilingual["th"],
                    chinese_text=multilingual.get("zh"),
                    english_text=multilingual.get("en"),
                    source_filename=row["source_filename"],
                )

    def find_exact_knowledge(
        self, source_language: str, target_language: str, source_text: str
    ) -> Optional[Dict[str, Any]]:
        normalized = normalize_knowledge_text(source_text)
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, source_language, target_language, source_text,
                       translated_text, source_filename, created_at
                FROM knowledge_entries
                WHERE source_language = ? AND target_language = ?
                  AND source_normalized = ?
                """,
                (source_language, target_language, normalized),
            ).fetchone()
        return dict(row) if row else None

    def find_exact_knowledge_many(
        self,
        source_language: str,
        target_language: str,
        source_texts: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        normalized_by_text = {
            source_text: normalize_knowledge_text(source_text)
            for source_text in source_texts
        }
        normalized_values = list(dict.fromkeys(normalized_by_text.values()))
        rows_by_normalized: Dict[str, Dict[str, Any]] = {}
        with self.connect() as connection:
            for start in range(0, len(normalized_values), 500):
                chunk = normalized_values[start : start + 500]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"""
                    SELECT id, source_language, target_language, source_text,
                           translated_text, source_normalized, source_filename, created_at
                    FROM knowledge_entries
                    WHERE source_language = ? AND target_language = ?
                      AND source_normalized IN ({placeholders})
                    """,
                    [source_language, target_language, *chunk],
                ).fetchall()
                rows_by_normalized.update(
                    {row["source_normalized"]: dict(row) for row in rows}
                )
        return {
            source_text: rows_by_normalized[normalized]
            for source_text, normalized in normalized_by_text.items()
            if normalized in rows_by_normalized
        }

    def find_matching_knowledge(
        self,
        source_language: str,
        target_language: str,
        source_text: str,
        limit: int = 40,
    ) -> List[Dict[str, Any]]:
        normalized_text = normalize_knowledge_text(source_text)
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, source_language, target_language, source_text,
                       translated_text, source_normalized, source_filename, created_at
                FROM knowledge_entries
                WHERE source_language = ? AND target_language = ?
                ORDER BY LENGTH(source_normalized) DESC, created_at DESC
                """,
                (source_language, target_language),
            ).fetchall()
        matches = [dict(row) for row in rows if row["source_normalized"] in normalized_text]
        return matches[:limit]

    def list_knowledge_for_direction(
        self,
        target_language: str,
        source_language: Optional[str] = None,
        limit: int = 60,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT id, source_language, target_language, source_text,
                   translated_text, source_filename, created_at
            FROM knowledge_entries
            WHERE target_language = ?
        """
        parameters: List[Any] = [target_language]
        if source_language:
            query += " AND source_language = ?"
            parameters.append(source_language)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        parameters.append(limit)
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection, table: str, column: str, definition: str
    ) -> None:
        columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _history_row(row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        result["warnings"] = json.loads(result["warnings"])
        return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_knowledge_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"\s+", " ", normalized)


def _knowledge_row_to_multilingual(
    values: Dict[str, str],
) -> Optional[Dict[str, str]]:
    source_language = values["source_language"]
    target_language = values["target_language"]
    if source_language == "th":
        return {
            "th": values["source_text"].strip(),
            target_language: values["translated_text"].strip(),
        }
    if target_language == "th":
        return {
            "th": values["translated_text"].strip(),
            source_language: values["source_text"].strip(),
        }
    return None
