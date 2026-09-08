"""Rebuild existing Chinese PDF exports with the current compact renderer."""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import fitz

from api.app.services.exports import build_pdf_export


def _page_geometry(pdf_bytes: bytes) -> list[tuple[float, float]]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        return [(float(page.rect.width), float(page.rect.height)) for page in document]


def rebuild_exports(data_directory: Path) -> None:
    database_path = data_directory / "app.db"
    export_directory = data_directory / "exports"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_directory = data_directory / "export-backups" / timestamp

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, translated_text
            FROM translation_history
            WHERE target_language = 'zh' AND export_filename LIKE '%.pdf'
            ORDER BY created_at
            """
        ).fetchall()

    if not rows:
        print("No Chinese PDF exports found.")
        return

    backup_directory.mkdir(parents=True, exist_ok=False)
    rebuilt = 0
    for item_id, translated_text in rows:
        export_path = export_directory / f"{item_id}.pdf"
        if not export_path.is_file():
            print(f"SKIP {item_id}: export file is missing")
            continue

        original_bytes = export_path.read_bytes()
        original_geometry = _page_geometry(original_bytes)
        backup_path = backup_directory / export_path.name
        shutil.copy2(export_path, backup_path)

        rebuilt_bytes = build_pdf_export(original_bytes, translated_text, "zh")
        rebuilt_geometry = _page_geometry(rebuilt_bytes)
        if len(rebuilt_geometry) != len(original_geometry):
            raise RuntimeError(f"{item_id}: page count changed")
        for page_number, (original, rebuilt_page) in enumerate(
            zip(original_geometry, rebuilt_geometry), start=1
        ):
            if any(abs(a - b) > 0.01 for a, b in zip(original, rebuilt_page)):
                raise RuntimeError(f"{item_id}: page {page_number} dimensions changed")

        temporary_path = export_path.with_suffix(".pdf.rebuilding")
        temporary_path.write_bytes(rebuilt_bytes)
        os.replace(temporary_path, export_path)
        rebuilt += 1
        print(
            f"OK {item_id}: {len(original_bytes)} -> {len(rebuilt_bytes)} bytes, "
            f"{len(rebuilt_geometry)} pages"
        )

    print(f"Rebuilt {rebuilt} export(s). Backups: {backup_directory}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    arguments = parser.parse_args()
    rebuild_exports(arguments.data_dir)


if __name__ == "__main__":
    main()
