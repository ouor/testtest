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
    r2_key: str
    content_type: str
    original_filename: str | None
    size_bytes: int


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

    SCHEMA_VERSION = 2

    def __init__(self, *, db_path: Path, vector_dim: int, max_elements: int) -> None:
        self.db_path = db_path
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

        # Some SQLite virtual table extensions may keep index data outside the main DB file.
        # To make R2 snapshots robust, we persist embeddings in a normal table and can
        # rebuild the virtual index if it appears empty.
        self._ensure_vector_index_materialized()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL
            )
            """
        )

        row = self.conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
        current_version = int(row[0]) if row else 0
        if current_version != self.SCHEMA_VERSION:
            # Hard reset on incompatibility (we're intentionally dropping legacy support).
            self.conn.execute("DROP TABLE IF EXISTS v_images")
            self.conn.execute("DROP TABLE IF EXISTS image_ids")
            self.conn.execute("DROP TABLE IF EXISTS image_vectors")
            self.conn.execute("DROP TABLE IF EXISTS image_records")
            self.conn.execute("DELETE FROM schema_version WHERE id = 1")
            self.conn.execute("INSERT INTO schema_version(id, version) VALUES (1, ?)", (self.SCHEMA_VERSION,))

        self.conn.execute("CREATE TABLE IF NOT EXISTS image_ids (image_id TEXT UNIQUE)")

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS image_records (
                image_id TEXT PRIMARY KEY,
                r2_key TEXT NOT NULL,
                content_type TEXT NOT NULL,
                original_filename TEXT,
                size_bytes INTEGER NOT NULL
            )
            """
        )

        # Persist embeddings in a normal table so backups capture vectors reliably.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS image_vectors (
                image_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL
            )
            """
        )

        self._create_virtual_table_if_missing()
        self.conn.commit()

    def _create_virtual_table_if_missing(self) -> None:
        self.conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS v_images USING vectorlite(
                embedding float32[{self.vector_dim}] cosine,
                hnsw(max_elements={self.max_elements})
            )
            """
        )

    def _recreate_virtual_table(self) -> None:
        # Some vectorlite builds don't support bulk DELETEs; safest is drop+recreate.
        self.conn.execute("DROP TABLE IF EXISTS v_images")
        self._create_virtual_table_if_missing()

    def _vector_count(self) -> int:
        try:
            row = self.conn.execute("SELECT COUNT(*) FROM image_vectors").fetchone()
            return int(row[0] if row else 0)
        except Exception:
            return 0

    def _index_count(self) -> int:
        try:
            row = self.conn.execute("SELECT COUNT(*) FROM v_images").fetchone()
            return int(row[0] if row else 0)
        except Exception:
            return 0

    def _ensure_vector_index_materialized(self) -> None:
        # If we have persisted vectors but the virtual index is empty (common after restore
        # if the extension stores data out-of-band), rebuild it.
        vec_count = self._vector_count()
        if vec_count <= 0:
            return

        idx_count = self._index_count()
        if idx_count > 0:
            return

        self.rebuild_vector_index_from_vectors()

    def rebuild_vector_index_from_vectors(self) -> None:
        rows = self.conn.execute("SELECT image_id, embedding FROM image_vectors").fetchall()
        if not rows:
            return

        with self.conn:
            # Clear existing index + mapping.
            self._recreate_virtual_table()
            self.conn.execute("DELETE FROM image_ids")

            for image_id, embedding_blob in rows:
                image_id_s = str(image_id)
                rowid = self._ensure_rowid(image_id_s)
                self.conn.execute("INSERT INTO v_images(rowid, embedding) VALUES (?, ?)", (rowid, embedding_blob))

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

        blob = vec.tobytes()

        # One transaction: vector + metadata.
        with self.conn:
            rowid = self._ensure_rowid(record.id)
            self.conn.execute("DELETE FROM v_images WHERE rowid = ?", (rowid,))
            self.conn.execute("INSERT INTO v_images(rowid, embedding) VALUES (?, ?)", (rowid, blob))
            self.conn.execute(
                """
                INSERT INTO image_vectors(image_id, embedding) VALUES (?, ?)
                ON CONFLICT(image_id) DO UPDATE SET embedding=excluded.embedding
                """,
                (record.id, blob),
            )
            self.conn.execute(
                """
                INSERT INTO image_records(image_id, r2_key, content_type, original_filename, size_bytes)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    r2_key=excluded.r2_key,
                    content_type=excluded.content_type,
                    original_filename=excluded.original_filename,
                    size_bytes=excluded.size_bytes
                """,
                (
                    record.id,
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
            self.conn.execute("DELETE FROM image_vectors WHERE image_id = ?", (image_id,))
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
            self.conn.execute(
                """
                INSERT INTO image_vectors(image_id, embedding) VALUES (?, ?)
                ON CONFLICT(image_id) DO UPDATE SET embedding=excluded.embedding
                """,
                (item_id, blob),
            )

    def delete(self, *, item_id: str) -> None:
        rowid = self._rowid_for_image_id(item_id)
        if rowid is None:
            return
        with self.conn:
            self.conn.execute("DELETE FROM v_images WHERE rowid = ?", (rowid,))
            self.conn.execute("DELETE FROM image_ids WHERE rowid = ?", (rowid,))
            self.conn.execute("DELETE FROM image_vectors WHERE image_id = ?", (item_id,))

    def upsert_record(self, record: ImageRecord) -> None:
        with self.conn:
            self.conn.execute(
            """
            INSERT INTO image_records(image_id, r2_key, content_type, original_filename, size_bytes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                r2_key=excluded.r2_key,
                content_type=excluded.content_type,
                original_filename=excluded.original_filename,
                size_bytes=excluded.size_bytes
            """,
            (
                record.id,
                record.r2_key,
                record.content_type,
                record.original_filename,
                int(record.size_bytes),
            ),
            )

    def get_record(self, *, image_id: str) -> ImageRecord | None:
        row = self.conn.execute(
            "SELECT image_id, r2_key, content_type, original_filename, size_bytes FROM image_records WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        if not row:
            return None
        image_id_s, r2_key, content_type, original_filename, size_bytes = row
        return ImageRecord(
            id=str(image_id_s),
            r2_key=str(r2_key),
            content_type=str(content_type),
            original_filename=(None if original_filename is None else str(original_filename)),
            size_bytes=int(size_bytes),
        )

    def delete_record(self, *, image_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM image_records WHERE image_id = ?", (image_id,))

    def list_records(self) -> list[ImageRecord]:
        rows = self.conn.execute(
            "SELECT image_id, r2_key, content_type, original_filename, size_bytes FROM image_records ORDER BY image_id"
        ).fetchall()
        out: list[ImageRecord] = []
        for image_id, r2_key, content_type, original_filename, size_bytes in rows:
            out.append(
                ImageRecord(
                    id=str(image_id),
                    r2_key=str(r2_key),
                    content_type=str(content_type),
                    original_filename=(None if original_filename is None else str(original_filename)),
                    size_bytes=int(size_bytes),
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
            SELECT ids.image_id, rec.r2_key, rec.content_type, rec.original_filename, rec.size_bytes, v.distance
            FROM v_images v
            JOIN image_ids ids ON v.rowid = ids.rowid
            JOIN image_records rec ON rec.image_id = ids.image_id
            WHERE knn_search(v.embedding, knn_param(?, ?))
        """
        rows = self.conn.execute(sql, (query_blob, int(limit))).fetchall()

        results: list[tuple[str, float]] = []
        for image_id, _r2_key, _content_type, _original_filename, _size_bytes, distance in rows:
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
            SELECT ids.image_id, rec.r2_key, rec.content_type, rec.original_filename, rec.size_bytes, v.distance
            FROM v_images v
            JOIN image_ids ids ON v.rowid = ids.rowid
            JOIN image_records rec ON rec.image_id = ids.image_id
            WHERE knn_search(v.embedding, knn_param(?, ?))
        """
        rows = self.conn.execute(sql, (query_blob, int(limit))).fetchall()

        out: list[tuple[ImageRecord, float]] = []
        for image_id, r2_key, content_type, original_filename, size_bytes, distance in rows:
            record = ImageRecord(
                id=str(image_id),
                r2_key=str(r2_key),
                content_type=str(content_type),
                original_filename=(None if original_filename is None else str(original_filename)),
                size_bytes=int(size_bytes),
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

    def backup_to_path(self, *, dest_path: Path) -> None:
        """Write a consistent SQLite snapshot to dest_path."""
        # Ensure pending transactions are flushed.
        try:
            self.conn.commit()
        except Exception:
            pass

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_conn = sqlite3.connect(str(dest_path), check_same_thread=False)
        try:
            with dest_conn:
                self.conn.backup(dest_conn)
        finally:
            try:
                dest_conn.close()
            except Exception:
                pass
