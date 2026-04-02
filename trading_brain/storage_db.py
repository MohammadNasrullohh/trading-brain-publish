from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DB_FILENAME = "brain_state.db"
DB_LOCK = threading.RLock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def default_data_dir(root: Path) -> Path:
    if root.parent.name == "releases":
        return root.parent.parent / "shared"
    shared_dir = root.parent / "shared"
    return shared_dir if shared_dir.exists() else root / "logs"


def _connect(db_path: Path) -> sqlite3.Connection:
    _ensure_parent(db_path)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS module_documents (
            module TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return connection


def _read_legacy_json(path: Path | None, default_factory: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    if path is None or not path.exists():
        return default_factory()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_factory()
    return payload if isinstance(payload, dict) else default_factory()


def _read_legacy_db_document(path: Path | None, module: str, default_factory: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    if path is None or not path.exists():
        return default_factory()
    connection = sqlite3.connect(str(path))
    try:
        row = connection.execute(
            "SELECT payload_json FROM module_documents WHERE module = ?",
            (module,),
        ).fetchone()
    except sqlite3.Error:
        return default_factory()
    finally:
        connection.close()

    if row is None:
        return default_factory()
    try:
        payload = json.loads(str(row[0]))
    except json.JSONDecodeError:
        return default_factory()
    return payload if isinstance(payload, dict) else default_factory()


def read_document(
    *,
    db_path: Path,
    module: str,
    default_factory: Callable[[], dict[str, Any]],
    legacy_path: Path | None = None,
    legacy_paths: list[Path] | tuple[Path, ...] | None = None,
    legacy_db_paths: list[Path] | tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    with DB_LOCK:
        connection = _connect(db_path)
        try:
            row = connection.execute(
                "SELECT payload_json FROM module_documents WHERE module = ?",
                (module,),
            ).fetchone()
            if row is not None:
                try:
                    payload = json.loads(str(row["payload_json"]))
                except json.JSONDecodeError:
                    payload = default_factory()
                return payload if isinstance(payload, dict) else default_factory()

            payload = default_factory()
            for candidate_db in list(legacy_db_paths or []):
                if candidate_db == db_path:
                    continue
                payload = _read_legacy_db_document(candidate_db, module, default_factory)
                if payload != default_factory():
                    break

            candidates: list[Path] = []
            if legacy_paths:
                candidates.extend(list(legacy_paths))
            elif legacy_path is not None:
                candidates.append(legacy_path)

            if payload == default_factory():
                for candidate in candidates:
                    payload = _read_legacy_json(candidate, default_factory)
                    if payload != default_factory():
                        break

            connection.execute(
                """
                INSERT INTO module_documents (module, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(module) DO UPDATE
                SET payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (module, json.dumps(payload, ensure_ascii=False), _utc_now_iso()),
            )
            connection.commit()
            return payload
        finally:
            connection.close()


def write_document(*, db_path: Path, module: str, payload: dict[str, Any]) -> None:
    with DB_LOCK:
        connection = _connect(db_path)
        try:
            connection.execute(
                """
                INSERT INTO module_documents (module, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(module) DO UPDATE
                SET payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (module, json.dumps(payload, ensure_ascii=False), _utc_now_iso()),
            )
            connection.commit()
        finally:
            connection.close()
