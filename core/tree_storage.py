import json
import sqlite3
import uuid as uuid_mod
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from core.hierarchical_chunker import ChunkNode
from config import config


class TreeStorage:
    def __init__(
        self,
        sqlite_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.sqlite_path = sqlite_path or config.HIERARCHICAL_TREE_DB
        self.collection_name = collection_name or config.HIERARCHICAL_COLLECTION
        self._init_sqlite()
        self._init_qdrant(qdrant_url, qdrant_api_key)

    def _init_sqlite(self):
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tree_nodes (
                node_id TEXT PRIMARY KEY,
                level INTEGER NOT NULL,
                content TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                children_ids TEXT NOT NULL DEFAULT '[]',
                parent_id TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                doc_id TEXT NOT NULL DEFAULT '',
                start_char INTEGER NOT NULL DEFAULT 0,
                end_char INTEGER NOT NULL DEFAULT 0
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_level ON tree_nodes(level)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_parent ON tree_nodes(parent_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc ON tree_nodes(doc_id)
        """)
        self.conn.commit()

    def _init_qdrant(self, url: Optional[str], api_key: Optional[str]):
        self.qdrant = None
        cloud_url = url or config.QDRANT_URL
        cloud_key = api_key or config.QDRANT_API_KEY

        if cloud_url and cloud_key:
            try:
                self.qdrant = QdrantClient(
                    url=cloud_url,
                    api_key=cloud_key,
                    timeout=10,
                )
                existing = [c.name for c in self.qdrant.get_collections().collections]
                if self.collection_name not in existing:
                    self.qdrant.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=config.QDRANT_VECTOR_SIZE,
                            distance=Distance.COSINE,
                        ),
                    )
                print(f"[TreeStorage] Qdrant cloud connected: {self.collection_name}")
                return
            except Exception as e:
                print(f"[TreeStorage] Qdrant cloud failed ({type(e).__name__}), falling back to local in-memory")
                self.qdrant = None

        try:
            self.qdrant = QdrantClient(":memory:")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=config.QDRANT_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[TreeStorage] Qdrant in-memory: {self.collection_name}")
        except Exception as e:
            print(f"[TreeStorage] Qdrant in-memory failed: {e}, vector operations disabled")
            self.qdrant = None

    @staticmethod
    def _node_id_to_uuid(node_id: str) -> str:
        return str(uuid_mod.uuid5(uuid_mod.NAMESPACE_URL, node_id))

    def store_nodes(self, nodes: List[ChunkNode], embed_func=None):
        l1_l2_nodes = [n for n in nodes if n.level in (1, 2)]
        l3_nodes = [n for n in nodes if n.level == 3]

        for node in l1_l2_nodes:
            self._store_sqlite_node(node)

        if l3_nodes and embed_func:
            self._store_l3_to_qdrant(l3_nodes, embed_func)
        for node in l3_nodes:
            self._store_sqlite_node(node)

    def _store_sqlite_node(self, node: ChunkNode):
        self.conn.execute(
            """
            INSERT OR REPLACE INTO tree_nodes
            (node_id, level, content, title, children_ids, parent_id, metadata, doc_id, start_char, end_char)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.node_id,
                node.level,
                node.content,
                node.title,
                json.dumps(node.children_ids),
                node.parent_id,
                json.dumps(node.metadata),
                node.doc_id,
                node.start_char,
                node.end_char,
            ),
        )
        self.conn.commit()

    def _store_l3_to_qdrant(self, l3_nodes: List[ChunkNode], embed_func):
        if not self.qdrant:
            return
        texts = [n.content for n in l3_nodes]
        vectors = embed_func(texts)

        points = []
        for i, node in enumerate(l3_nodes):
            points.append(
                PointStruct(
                    id=self._node_id_to_uuid(node.node_id),
                    vector=vectors[i],
                    payload={
                        "node_id": node.node_id,
                        "level": 3,
                        "content": node.content,
                        "title": node.title,
                        "parent_id": node.parent_id or "",
                        "doc_id": node.doc_id,
                        "metadata": node.metadata,
                    },
                )
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

    def get_node(self, node_id: str) -> Optional[ChunkNode]:
        cursor = self.conn.execute(
            "SELECT * FROM tree_nodes WHERE node_id = ?", (node_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_node(row)

    def get_parent(self, node_id: str) -> Optional[ChunkNode]:
        node = self.get_node(node_id)
        if node and node.parent_id:
            return self.get_node(node.parent_id)
        return None

    def get_ancestors(self, node_id: str) -> List[ChunkNode]:
        ancestors = []
        current = self.get_node(node_id)
        while current and current.parent_id:
            parent = self.get_node(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors

    def get_children(self, node_id: str) -> List[ChunkNode]:
        node = self.get_node(node_id)
        if not node:
            return []
        children = []
        for cid in node.children_ids:
            child = self.get_node(cid)
            if child:
                children.append(child)
        return children

    def get_l3_context_chain(self, l3_node_id: str) -> Dict[str, Any]:
        l3_node = self.get_node(l3_node_id)
        if not l3_node:
            return {"l3": None, "l2": None, "l1": None}

        l2_node = self.get_parent(l3_node_id)
        l1_node = self.get_parent(l2_node.node_id) if l2_node else None

        return {
            "l3": l3_node,
            "l2": l2_node,
            "l1": l1_node,
        }

    def search_l3_by_vector(
        self, query_vector: List[float], top_k: int = 5, doc_id: Optional[str] = None
    ) -> List[Dict]:
        if not self.qdrant:
            return []
        query_filter = None
        if doc_id:
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )

        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

        return [
            {
                "node_id": r.payload.get("node_id", str(r.id)),
                "score": r.score,
                "content": r.payload.get("content", ""),
                "title": r.payload.get("title", ""),
                "parent_id": r.payload.get("parent_id", ""),
                "doc_id": r.payload.get("doc_id", ""),
                "metadata": r.payload.get("metadata", {}),
            }
            for r in response.points
        ]

    def get_all_l3_content(self, doc_id: Optional[str] = None) -> List[Dict]:
        cursor = self.conn.execute(
            "SELECT node_id, content, title, parent_id FROM tree_nodes WHERE level = 3"
            + (" AND doc_id = ?" if doc_id else ""),
            (doc_id,) if doc_id else (),
        )
        rows = cursor.fetchall()
        return [
            {
                "node_id": row[0],
                "content": row[1],
                "title": row[2],
                "parent_id": row[3],
            }
            for row in rows
        ]

    def get_nodes_by_level(self, level: int, doc_id: Optional[str] = None) -> List[ChunkNode]:
        if doc_id:
            cursor = self.conn.execute(
                "SELECT * FROM tree_nodes WHERE level = ? AND doc_id = ?",
                (level, doc_id),
            )
        else:
            cursor = self.conn.execute(
                "SELECT * FROM tree_nodes WHERE level = ?", (level,)
            )
        return [self._row_to_node(row) for row in cursor.fetchall()]

    def delete_doc(self, doc_id: str):
        l3_nodes = self.get_nodes_by_level(3, doc_id)
        if l3_nodes and self.qdrant:
            uuids = [self._node_id_to_uuid(n.node_id) for n in l3_nodes]
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=uuids,
            )
        self.conn.execute("DELETE FROM tree_nodes WHERE doc_id = ?", (doc_id,))
        self.conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        cursor = self.conn.execute(
            "SELECT level, COUNT(*) FROM tree_nodes GROUP BY level"
        )
        level_counts = {row[0]: row[1] for row in cursor.fetchall()}
        cursor = self.conn.execute("SELECT COUNT(DISTINCT doc_id) FROM tree_nodes")
        doc_count = cursor.fetchone()[0]
        return {
            "level_counts": level_counts,
            "total_nodes": sum(level_counts.values()),
            "doc_count": doc_count,
        }

    def _row_to_node(self, row) -> ChunkNode:
        return ChunkNode(
            node_id=row[0],
            level=row[1],
            content=row[2],
            title=row[3],
            children_ids=json.loads(row[4]),
            parent_id=row[5],
            metadata=json.loads(row[6]),
            doc_id=row[7],
            start_char=row[8],
            end_char=row[9],
        )

    def close(self):
        self.conn.close()
