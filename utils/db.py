import sqlite3
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path.cwd() / "data" / "index_dir" / "vector_store_metadata.db"
        else:
            db_path = Path(db_path)
        
        self.conn = sqlite3.connect(str(db_path))
        self.cursor = self.conn.cursor()

    def create_metadata_table(self):
        self.cursor.execute("""
             CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY,
                content TEXT,
                source TEXT,
                doc_id TEXT,
                format TEXT,
                chunk_index INTEGER,
                start_index INTEGER,
                end_index INTEGER,
                section TEXT,
                page INTEGER,
                created_at DATETIME,
                content_hash TEXT
            )
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Apply any pending schema migrations to existing databases."""
        try:
            self.cursor.execute("ALTER TABLE metadata ADD COLUMN content_hash TEXT")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

    def insert_metadata(self, metadatas: list[dict]):
        self.cursor.executemany("""
            INSERT INTO metadata (id, content, source, doc_id, format, chunk_index, start_index, end_index, section, page, created_at, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(m["id"], m["content"], m["source"], m["doc_id"], m["format"], m["chunk_index"], m["start_index"], m["end_index"], m["section"], m["page"], m["created_at"], m["content_hash"]) for m in metadatas])

    def get_max_id(self) -> int:
        """Get the maximum ID from the metadata table. Returns -1 if table is empty."""
        self.cursor.execute("SELECT MAX(id) FROM metadata")
        result = self.cursor.fetchone()[0]
        return result if result is not None else -1

    def get_stored_hash(self, source: str) -> str | None:
        """Retrieve the stored content hash for a given source document.

        Args:
            source: The source file path or identifier.

        Returns:
            The stored hash string, or None if the source has not been indexed.
        """
        self.cursor.execute("SELECT content_hash FROM metadata WHERE source = ? LIMIT 1", (source,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    def is_document_indexed(self, source: str) -> bool:
        """Check whether a document with the given source has already been indexed."""
        self.cursor.execute("SELECT 1 FROM metadata WHERE source = ? LIMIT 1", (source,))
        return self.cursor.fetchone() is not None

    def delete_by_source(self, source: str) -> list[int]:
        """Delete all metadata entries for a source document and return their IDs.

        Args:
            source: The source file path or identifier.

        Returns:
            A list of vector IDs that were removed, to be used for FAISS cleanup.
        """
        self.cursor.execute("SELECT id FROM metadata WHERE source = ?", (source,))
        ids = [int(row[0]) for row in self.cursor.fetchall()]
        self.cursor.execute("DELETE FROM metadata WHERE source = ?", (source,))
        self.conn.commit()
        return ids

    def get_metadata_by_id(self, id: int) -> dict:
        self.cursor.execute("SELECT content, source, doc_id, format, chunk_index, start_index, end_index, section, page, created_at FROM metadata WHERE id = ?", (int(id),))
        row = self.cursor.fetchone()
        if row is None:
            return None
        return {
            "content": row[0],
            "source": row[1],
            "doc_id": row[2],
            "format": row[3],
            "chunk_index": row[4],
            "start_index": row[5],
            "end_index": row[6],
            "section": row[7],
            "page": row[8],
            "created_at": row[9]
        }
    