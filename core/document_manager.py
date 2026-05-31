import json
import time
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from core.database import DatabaseManager, get_database, escape_like


class DocStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


@dataclass
class Document:
    doc_id: str = ""
    title: str = ""
    source: str = ""
    content_hash: str = ""
    file_type: str = "text"
    status: DocStatus = DocStatus.READY
    total_chars: int = 0
    node_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    tags: List[str] = field(default_factory=list)
    description: str = ""
    version: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "content_hash": self.content_hash,
            "file_type": self.file_type,
            "status": self.status.value if isinstance(self.status, DocStatus) else self.status,
            "total_chars": self.total_chars,
            "node_count": self.node_count,
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "tags": self.tags,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class DocumentVersion:
    version_id: str = ""
    doc_id: str = ""
    version: int = 1
    content_hash: str = ""
    change_description: str = ""
    total_chars: int = 0
    node_count: int = 0
    created_at: float = 0.0

    def __post_init__(self):
        if not self.version_id:
            self.version_id = f"ver_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "doc_id": self.doc_id,
            "version": self.version,
            "content_hash": self.content_hash,
            "change_description": self.change_description,
            "total_chars": self.total_chars,
            "node_count": self.node_count,
            "created_at": self.created_at,
        }


class DocumentManager:
    def __init__(self, db: Optional[DatabaseManager] = None, db_path: Optional[str] = None):
        if db is not None:
            self.db = db
        elif db_path is not None:
            self.db = get_database(db_path)
        else:
            self.db = get_database()
        self._init_db()

    def _init_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                file_type TEXT NOT NULL DEFAULT 'text',
                status TEXT NOT NULL DEFAULT 'ready',
                total_chars INTEGER NOT NULL DEFAULT 0,
                node_count INTEGER NOT NULL DEFAULT 0,
                entity_count INTEGER NOT NULL DEFAULT 0,
                relation_count INTEGER NOT NULL DEFAULT 0,
                tags TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL DEFAULT '',
                version INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS document_versions (
                version_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                content_hash TEXT NOT NULL DEFAULT '',
                change_description TEXT NOT NULL DEFAULT '',
                total_chars INTEGER NOT NULL DEFAULT 0,
                node_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL DEFAULT 0,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_tags ON documents(tags)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_ver_doc ON document_versions(doc_id)
        """)
        self.db.commit()

    def register_document(
        self,
        doc_id: str,
        title: str = "",
        source: str = "",
        content: str = "",
        file_type: str = "text",
        tags: Optional[List[str]] = None,
        description: str = "",
        metadata: Optional[Dict] = None,
    ) -> Document:
        content_hash = hashlib.md5(content.encode()).hexdigest() if content else ""

        existing = self.get_document(doc_id)
        version = 1
        if existing:
            if existing.content_hash == content_hash:
                return existing
            version = existing.version + 1

        doc = Document(
            doc_id=doc_id,
            title=title or (existing.title if existing else doc_id),
            source=source or (existing.source if existing else ""),
            content_hash=content_hash,
            file_type=file_type,
            status=DocStatus.READY,
            total_chars=len(content),
            tags=tags if tags is not None else (existing.tags if existing else []),
            description=description or (existing.description if existing else ""),
            version=version,
            metadata=metadata if metadata is not None else (existing.metadata if existing else {}),
        )
        self._save_document(doc)

        if content:
            self._save_version(doc_id, version, content_hash, "", len(content), 0)

        return doc

    def update_document_stats(
        self,
        doc_id: str,
        node_count: int = 0,
        entity_count: int = 0,
        relation_count: int = 0,
    ):
        doc = self.get_document(doc_id)
        if not doc:
            return

        doc.node_count = node_count
        doc.entity_count = entity_count
        doc.relation_count = relation_count
        doc.updated_at = time.time()
        self._save_document(doc)

    def get_document(self, doc_id: str) -> Optional[Document]:
        cursor = self.db.execute(
            "SELECT doc_id, title, source, content_hash, file_type, status, "
            "total_chars, node_count, entity_count, relation_count, "
            "tags, description, version, created_at, updated_at, metadata "
            "FROM documents WHERE doc_id = ?",
            (doc_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_document(row)

    def list_documents(
        self,
        status: Optional[DocStatus] = None,
        tag: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        query = "SELECT doc_id, title, source, file_type, status, total_chars, node_count, entity_count, tags, version, created_at, updated_at FROM documents"
        conditions = []
        params = []

        if status:
            conditions.append("status = ?")
            params.append(status.value if isinstance(status, DocStatus) else status)
        if tag:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self.db.execute(query, params)
        rows = cursor.fetchall()
        return [
            {
                "doc_id": row[0],
                "title": row[1],
                "source": row[2],
                "file_type": row[3],
                "status": row[4],
                "total_chars": row[5],
                "node_count": row[6],
                "entity_count": row[7],
                "tags": json.loads(row[8]) if row[8] else [],
                "version": row[9],
                "created_at": row[10],
                "updated_at": row[11],
            }
            for row in rows
        ]

    def delete_document(self, doc_id: str) -> bool:
        self.db.execute("DELETE FROM document_versions WHERE doc_id = ?", (doc_id,))
        self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self.db.commit()
        return True

    def search_documents(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        safe = escape_like(keyword)
        cursor = self.db.execute(
            "SELECT doc_id, title, source, file_type, status, total_chars, tags, version, created_at "
            "FROM documents "
            "WHERE title LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' OR source LIKE ? ESCAPE '\\' "
            "ORDER BY updated_at DESC LIMIT ?",
            (f"%{safe}%", f"%{safe}%", f"%{safe}%", limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "doc_id": row[0],
                "title": row[1],
                "source": row[2],
                "file_type": row[3],
                "status": row[4],
                "total_chars": row[5],
                "tags": json.loads(row[6]) if row[6] else [],
                "version": row[7],
                "created_at": row[8],
            }
            for row in rows
        ]

    def get_document_versions(self, doc_id: str) -> List[Dict[str, Any]]:
        cursor = self.db.execute(
            "SELECT version_id, doc_id, version, content_hash, change_description, "
            "total_chars, node_count, created_at "
            "FROM document_versions WHERE doc_id = ? ORDER BY version DESC",
            (doc_id,),
        )
        rows = cursor.fetchall()
        return [
            {
                "version_id": row[0],
                "doc_id": row[1],
                "version": row[2],
                "content_hash": row[3],
                "change_description": row[4],
                "total_chars": row[5],
                "node_count": row[6],
                "created_at": row[7],
            }
            for row in rows
        ]

    def get_stats(self) -> Dict[str, Any]:
        cursor = self.db.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        cursor = self.db.execute("SELECT status, COUNT(*) FROM documents GROUP BY status")
        status_dist = {row[0]: row[1] for row in cursor.fetchall()}
        cursor = self.db.execute("SELECT SUM(total_chars), SUM(node_count) FROM documents")
        row = cursor.fetchone()
        total_chars = row[0] or 0
        total_nodes = row[1] or 0
        cursor = self.db.execute(
            "SELECT tags FROM documents WHERE tags != '[]'"
        )
        all_tags = set()
        for row in cursor.fetchall():
            tags = json.loads(row[0]) if row[0] else []
            all_tags.update(tags)

        return {
            "total_documents": total_docs,
            "status_distribution": status_dist,
            "total_chars": total_chars,
            "total_nodes": total_nodes,
            "all_tags": sorted(list(all_tags)),
        }

    def _save_document(self, doc: Document):
        self.db.execute(
            """INSERT OR REPLACE INTO documents
            (doc_id, title, source, content_hash, file_type, status,
             total_chars, node_count, entity_count, relation_count,
             tags, description, version, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc.doc_id,
                doc.title,
                doc.source,
                doc.content_hash,
                doc.file_type,
                doc.status.value if isinstance(doc.status, DocStatus) else doc.status,
                doc.total_chars,
                doc.node_count,
                doc.entity_count,
                doc.relation_count,
                json.dumps(doc.tags, ensure_ascii=False),
                doc.description,
                doc.version,
                doc.created_at,
                doc.updated_at,
                json.dumps(doc.metadata, ensure_ascii=False),
            ),
        )
        self.db.commit()

    def _save_version(
        self,
        doc_id: str,
        version: int,
        content_hash: str,
        change_description: str,
        total_chars: int,
        node_count: int,
    ):
        ver = DocumentVersion(
            doc_id=doc_id,
            version=version,
            content_hash=content_hash,
            change_description=change_description,
            total_chars=total_chars,
            node_count=node_count,
        )
        self.db.execute(
            """INSERT OR REPLACE INTO document_versions
            (version_id, doc_id, version, content_hash, change_description, total_chars, node_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ver.version_id,
                ver.doc_id,
                ver.version,
                ver.content_hash,
                ver.change_description,
                ver.total_chars,
                ver.node_count,
                ver.created_at,
            ),
        )
        self.db.commit()

    def _row_to_document(self, row) -> Document:
        return Document(
            doc_id=row[0],
            title=row[1],
            source=row[2],
            content_hash=row[3],
            file_type=row[4],
            status=DocStatus(row[5]),
            total_chars=row[6],
            node_count=row[7],
            entity_count=row[8],
            relation_count=row[9],
            tags=json.loads(row[10]) if row[10] else [],
            description=row[11],
            version=row[12],
            created_at=row[13],
            updated_at=row[14],
            metadata=json.loads(row[15]) if row[15] else {},
        )

    def close(self):
        pass  # DatabaseManager is shared, closed at app shutdown
