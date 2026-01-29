from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import vectorlite_py


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    denom = float(np.linalg.norm(vec) + 1e-8)
    return vec / denom


@dataclass(frozen=True)
class ImageRecord:
    id: str
    # Backward compatible: older records used local files via rel_path.
    # New records should prefer `r2_key`.
    path: Path | None
    content_type: str
    original_filename: str | None
    size_bytes: int
    r2_key: str | None = None


class VectorIndex(Protocol):
    def upsert(self, *, item_id: str, vector: np.ndarray) -> None: ...

    def delete(self, *, item_id: str) -> None: ...

    def search(self, *, vector: np.ndarray, limit: int) -> list[tuple[str, float]]: ...

    def ids(self) -> list[str]: ...


class ImageRecordStore(Protocol):
    def upsert_record(self, record: ImageRecord) -> None: ...

    def get_record(self, *, image_id: str) -> ImageRecord | None: ...

    def delete_record(self, *, image_id: str) -> None: ...

    def list_records(self) -> list[ImageRecord]: ...


class VectorliteVectorIndex:
    """SQLite+vectorlite backed index.

    Mirrors test/vector_db.py:
    - virtual table v_images using vectorlite(embedding float32[d] cosine, hnsw(max_elements=...))
    - mapping table image_ids(rowid <-> image_id)
    - search via knn_search(v.embedding, knn_param(?, k))

    Additionally stores image metadata in `image_records` for list/get/delete.
    """

    def __init__(self, *, db_path: Path, base_dir: Path, vector_dim: int, max_elements: int) -> None:
        self.db_path = db_path
        self.base_dir = base_dir
        self.vector_dim = int(vector_dim)
        self.max_elements = int(max_elements)

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.enable_load_extension(True)
        try:
            ext_path = vectorlite_py.vectorlite_path()
            self.conn.load_extension(ext_path)
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                "Failed to load vectorlite SQLite extension. "
                "Ensure vectorlite is installed and your SQLite build allows extensions. "
                f"extension_path={ext_path}"
            ) from exc

        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("CREATE TABLE IF NOT EXISTS image_ids (image_id TEXT UNIQUE)")

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS image_records (
                image_id TEXT PRIMARY KEY,
                rel_path TEXT NOT NULL,
                r2_key TEXT,
                content_type TEXT NOT NULL,
                original_filename TEXT,
                size_bytes INTEGER NOT NULL
            )
            """
        )

        # Lightweight migration: older DBs won't have r2_key.
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(image_records)").fetchall()]
        if "r2_key" not in cols:
            self.conn.execute("ALTER TABLE image_records ADD COLUMN r2_key TEXT")

        self.conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS v_images USING vectorlite(
                embedding float32[{self.vector_dim}] cosine,
                hnsw(max_elements={self.max_elements})
            )
            """
        )
        self.conn.commit()

    def _rowid_for_image_id(self, image_id: str) -> int | None:
        row = self.conn.execute("SELECT rowid FROM image_ids WHERE image_id = ?", (image_id,)).fetchone()
        if not row:
            return None
        return int(row[0])

    def _ensure_rowid(self, image_id: str) -> int:
        self.conn.execute("INSERT OR IGNORE INTO image_ids(image_id) VALUES (?)", (image_id,))
        rowid = self._rowid_for_image_id(image_id)
        if rowid is None:
            raise RuntimeError("Failed to allocate rowid for image_id")
        return rowid

    def upsert_image(self, *, record: ImageRecord, vector: np.ndarray) -> None:
        vec = vector.astype(np.float32, copy=False)
        if vec.ndim != 1 or vec.shape[0] != self.vector_dim:
            raise ValueError(f"Unexpected vector shape: {tuple(vec.shape)}")

        rel_path = os.fspath(Path(record.path).name) if record.path is not None else ""
        blob = vec.tobytes()

        # One transaction: vector + metadata.
        with self.conn:
            rowid = self._ensure_rowid(record.id)
            self.conn.execute("DELETE FROM v_images WHERE rowid = ?", (rowid,))
            self.conn.execute("INSERT INTO v_images(rowid, embedding) VALUES (?, ?)", (rowid, blob))
            self.conn.execute(
                """
                INSERT INTO image_records(image_id, rel_path, r2_key, content_type, original_filename, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    rel_path=excluded.rel_path,
                    r2_key=excluded.r2_key,
                    content_type=excluded.content_type,
                    original_filename=excluded.original_filename,
                    size_bytes=excluded.size_bytes
                """,
                (
                    record.id,
                    rel_path,
                    record.r2_key,
                    record.content_type,
                    record.original_filename,
                    int(record.size_bytes),
                ),
            )

    def delete_image(self, *, image_id: str) -> None:
        # One transaction: vector + metadata.
        with self.conn:
            rowid = self._rowid_for_image_id(image_id)
            if rowid is not None:
                self.conn.execute("DELETE FROM v_images WHERE rowid = ?", (rowid,))
                self.conn.execute("DELETE FROM image_ids WHERE rowid = ?", (rowid,))
            self.conn.execute("DELETE FROM image_records WHERE image_id = ?", (image_id,))

    def upsert(self, *, item_id: str, vector: np.ndarray) -> None:
        vec = vector.astype(np.float32, copy=False)
        if vec.ndim != 1 or vec.shape[0] != self.vector_dim:
            raise ValueError(f"Unexpected vector shape: {tuple(vec.shape)}")

        rowid = self._ensure_rowid(item_id)

        blob = vec.tobytes()
        with self.conn:
            self.conn.execute("DELETE FROM v_images WHERE rowid = ?", (rowid,))
            self.conn.execute("INSERT INTO v_images(rowid, embedding) VALUES (?, ?)", (rowid, blob))

    def delete(self, *, item_id: str) -> None:
        rowid = self._rowid_for_image_id(item_id)
        if rowid is None:
            return
        with self.conn:
            self.conn.execute("DELETE FROM v_images WHERE rowid = ?", (rowid,))
            self.conn.execute("DELETE FROM image_ids WHERE rowid = ?", (rowid,))

    def upsert_record(self, record: ImageRecord) -> None:
        rel_path = os.fspath(Path(record.path).name) if record.path is not None else ""
        with self.conn:
            self.conn.execute(
            """
            INSERT INTO image_records(image_id, rel_path, r2_key, content_type, original_filename, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                rel_path=excluded.rel_path,
                r2_key=excluded.r2_key,
                content_type=excluded.content_type,
                original_filename=excluded.original_filename,
                size_bytes=excluded.size_bytes
            """,
            (
                record.id,
                rel_path,
                record.r2_key,
                record.content_type,
                record.original_filename,
                int(record.size_bytes),
            ),
            )

    def get_record(self, *, image_id: str) -> ImageRecord | None:
        row = self.conn.execute(
            "SELECT image_id, rel_path, r2_key, content_type, original_filename, size_bytes FROM image_records WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        if not row:
            return None
        image_id_s, rel_path, r2_key, content_type, original_filename, size_bytes = row
        rel_path_s = str(rel_path or "")
        path = (self.base_dir / rel_path_s) if rel_path_s else None
        return ImageRecord(
            id=str(image_id_s),
            path=path,
            content_type=str(content_type),
            original_filename=(None if original_filename is None else str(original_filename)),
            size_bytes=int(size_bytes),
            r2_key=(None if r2_key is None else str(r2_key)),
        )

    def delete_record(self, *, image_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM image_records WHERE image_id = ?", (image_id,))

    def list_records(self) -> list[ImageRecord]:
        rows = self.conn.execute(
            "SELECT image_id, rel_path, r2_key, content_type, original_filename, size_bytes FROM image_records ORDER BY image_id"
        ).fetchall()
        out: list[ImageRecord] = []
        for image_id, rel_path, r2_key, content_type, original_filename, size_bytes in rows:
            rel_path_s = str(rel_path or "")
            out.append(
                ImageRecord(
                    id=str(image_id),
                    path=(self.base_dir / rel_path_s) if rel_path_s else None,
                    content_type=str(content_type),
                    original_filename=(None if original_filename is None else str(original_filename)),
                    size_bytes=int(size_bytes),
                    r2_key=(None if r2_key is None else str(r2_key)),
                )
            )
        return out

    def search(self, *, vector: np.ndarray, limit: int) -> list[tuple[str, float]]:
        if limit <= 0:
            return []

        vec = vector.astype(np.float32, copy=False)
        if vec.ndim != 1 or vec.shape[0] != self.vector_dim:
            raise ValueError(f"Unexpected vector shape: {tuple(vec.shape)}")

        query_blob = vec.tobytes()

        sql = """
            SELECT ids.image_id, rec.rel_path, rec.r2_key, rec.content_type, rec.original_filename, rec.size_bytes, v.distance
            FROM v_images v
            JOIN image_ids ids ON v.rowid = ids.rowid
            JOIN image_records rec ON rec.image_id = ids.image_id
            WHERE knn_search(v.embedding, knn_param(?, ?))
        """
        rows = self.conn.execute(sql, (query_blob, int(limit))).fetchall()

        results: list[tuple[str, float]] = []
        for image_id, _rel_path, _r2_key, _content_type, _original_filename, _size_bytes, distance in rows:
            try:
                sim = 1.0 - float(distance)
            except Exception:
                sim = float(distance)
            results.append((str(image_id), sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search_records(self, *, vector: np.ndarray, limit: int) -> list[tuple[ImageRecord, float]]:
        if limit <= 0:
            return []

        vec = vector.astype(np.float32, copy=False)
        if vec.ndim != 1 or vec.shape[0] != self.vector_dim:
            raise ValueError(f"Unexpected vector shape: {tuple(vec.shape)}")

        query_blob = vec.tobytes()
        sql = """
            SELECT ids.image_id, rec.rel_path, rec.r2_key, rec.content_type, rec.original_filename, rec.size_bytes, v.distance
            FROM v_images v
            JOIN image_ids ids ON v.rowid = ids.rowid
            JOIN image_records rec ON rec.image_id = ids.image_id
            WHERE knn_search(v.embedding, knn_param(?, ?))
        """
        rows = self.conn.execute(sql, (query_blob, int(limit))).fetchall()

        out: list[tuple[ImageRecord, float]] = []
        for image_id, rel_path, r2_key, content_type, original_filename, size_bytes, distance in rows:
            rel_path_s = str(rel_path or "")
            path = (self.base_dir / rel_path_s) if rel_path_s else None
            record = ImageRecord(
                id=str(image_id),
                path=path,
                content_type=str(content_type),
                original_filename=(None if original_filename is None else str(original_filename)),
                size_bytes=int(size_bytes),
                r2_key=(None if r2_key is None else str(r2_key)),
            )
            try:
                sim = 1.0 - float(distance)
            except Exception:
                sim = float(distance)
            out.append((record, sim))

        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def ids(self) -> list[str]:
        rows = self.conn.execute("SELECT image_id FROM image_ids").fetchall()
        return [str(r[0]) for r in rows]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
