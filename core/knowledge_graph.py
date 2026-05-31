import json
import uuid
import re
import difflib
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from core.database import DatabaseManager, get_database, escape_like


@dataclass
class Entity:
    entity_id: str = ""
    name: str = ""
    entity_type: str = "concept"
    description: str = ""
    source_node_id: str = ""
    doc_id: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.entity_id:
            self.entity_id = f"ent_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "source_node_id": self.source_node_id,
            "doc_id": self.doc_id,
            "properties": self.properties,
        }


@dataclass
class Relation:
    relation_id: str = ""
    source_entity_id: str = ""
    target_entity_id: str = ""
    relation_type: str = "related_to"
    description: str = ""
    weight: float = 1.0
    doc_id: str = ""

    def __post_init__(self):
        if not self.relation_id:
            self.relation_id = f"rel_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relation_type": self.relation_type,
            "description": self.description,
            "weight": self.weight,
            "doc_id": self.doc_id,
        }


class KnowledgeGraphBuilder:
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        llm_func: Optional[Callable] = None,
        db_path: Optional[str] = None,
        embed_func: Optional[Callable] = None,
    ):
        if db is not None:
            self.db = db
        elif db_path is not None:
            self.db = get_database(db_path)
        else:
            self.db = get_database()
        self.llm_func = llm_func
        self.embed_func = embed_func
        self._alias_map: Dict[str, str] = {}
        self._name_to_id: Dict[str, str] = {}
        self._init_db()
        self._load_name_index()

    def _init_db(self):
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL DEFAULT 'concept',
                description TEXT NOT NULL DEFAULT '',
                source_node_id TEXT NOT NULL DEFAULT '',
                doc_id TEXT NOT NULL DEFAULT '',
                properties TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                relation_id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL DEFAULT 'related_to',
                description TEXT NOT NULL DEFAULT '',
                weight REAL NOT NULL DEFAULT 1.0,
                doc_id TEXT NOT NULL DEFAULT ''
            )
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_doc ON entities(doc_id)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_entity_id)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_entity_id)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_type ON relations(relation_type)
        """)
        self.db.commit()

    def build_from_nodes(
        self,
        nodes: List[Dict],
        doc_id: str = "",
    ) -> Dict[str, Any]:
        if self.llm_func:
            return self._build_with_llm(nodes, doc_id)
        return self._build_with_rules(nodes, doc_id)

    def _build_with_llm(
        self,
        nodes: List[Dict],
        doc_id: str,
    ) -> Dict[str, Any]:
        # Phase 1: rule-based NER on all nodes (free, no LLM calls)
        for node in nodes:
            ne, nr = self._extract_with_ner(
                node.get("content", ""), node.get("node_id", ""), doc_id, node.get("title", "")
            )
            for e in ne:
                self._save_entity(e)
                self._name_to_id[e.name] = e.entity_id
            for r in nr:
                self._save_relation(r)

        all_entities = list(self._load_all_entities_for_doc(doc_id))
        all_relations = []
        existing_names = {e.name for e in all_entities}

        if not self.llm_func:
            return self._build_result(doc_id, all_entities, all_relations)

        # Phase 2: batch LLM extraction (BATCH_SIZE nodes per call)
        BATCH_SIZE = 8
        content_nodes = [n for n in nodes if n.get("content", "").strip()]
        node_index = {n["node_id"]: n for n in content_nodes}

        for i in range(0, len(content_nodes), BATCH_SIZE):
            batch = content_nodes[i:i + BATCH_SIZE]
            batch_text_parts = []
            for n in batch:
                batch_text_parts.append(
                    f"[node_id: {n['node_id']}]\n[title: {n.get('title', '')}]\n{n['content'][:800]}"
                )
            batch_text = "\n\n---\n\n".join(batch_text_parts)

            prompt = f"""请从以下多段文本中批量提取实体和关系，按JSON格式输出。

文本批次:
{batch_text}

请按以下格式输出（为每个实体标注其来源 node_id）:
{{
  "entities": [
    {{"name": "实体名", "type": "概念/技术/工具/人物/组织", "description": "简短描述", "node_id": "来源节点ID"}}
  ],
  "relations": [
    {{"source": "源实体名", "target": "目标实体名", "type": "包含/依赖/属于/相关/基于/实现", "description": "关系描述"}}
  ]
}}

只输出JSON，不要其他内容:"""

            try:
                response = self.llm_func(prompt)
                clean = response.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                parsed = json.loads(clean)

                for ent_data in parsed.get("entities", []):
                    name = ent_data.get("name", "").strip()
                    if not name or name in existing_names:
                        continue
                    src_node_id = ent_data.get("node_id", batch[0]["node_id"])
                    existing_id = self._resolve_entity_conflict(name)
                    if existing_id:
                        existing = self._load_entity(existing_id)
                        if existing and doc_id and not existing.doc_id:
                            existing.doc_id = doc_id
                            existing.source_node_id = src_node_id
                            self._save_entity(existing)
                        if existing:
                            all_entities.append(existing)
                            existing_names.add(name)
                    else:
                        entity = Entity(
                            name=name,
                            entity_type=ent_data.get("type", "concept"),
                            description=ent_data.get("description", ""),
                            source_node_id=src_node_id,
                            doc_id=doc_id,
                        )
                        self._save_entity(entity)
                        self._name_to_id[name] = entity.entity_id
                        all_entities.append(entity)
                        existing_names.add(name)

                name_to_id = {e.name: e.entity_id for e in all_entities}

                for rel_data in parsed.get("relations", []):
                    src_name = rel_data.get("source", "").strip()
                    tgt_name = rel_data.get("target", "").strip()
                    if not src_name or not tgt_name:
                        continue

                    for ent_name in (src_name, tgt_name):
                        eid = name_to_id.get(ent_name)
                        if not eid:
                            eid = self._resolve_entity_conflict(ent_name)
                        if not eid:
                            new_entity = Entity(name=ent_name, doc_id=doc_id)
                            self._save_entity(new_entity)
                            self._name_to_id[ent_name] = new_entity.entity_id
                            name_to_id[ent_name] = new_entity.entity_id
                            all_entities.append(new_entity)
                            existing_names.add(ent_name)

                    relation = Relation(
                        source_entity_id=name_to_id.get(src_name, ""),
                        target_entity_id=name_to_id.get(tgt_name, ""),
                        relation_type=rel_data.get("type", "related_to"),
                        description=rel_data.get("description", ""),
                        doc_id=doc_id,
                    )
                    self._save_relation(relation)
                    all_relations.append(relation)

            except (json.JSONDecodeError, Exception):
                # LLM batch failed — fall back to rule-based extraction for each node
                for node in batch:
                    ne, nr = self._extract_with_rules(
                        node.get("content", ""), node.get("node_id", ""), doc_id, node.get("title", "")
                    )
                    for e in ne:
                        self._save_entity(e)
                        self._name_to_id[e.name] = e.entity_id
                        all_entities.append(e)
                        existing_names.add(e.name)
                    for r in nr:
                        self._save_relation(r)
                        all_relations.append(r)
                continue

        return self._build_result(doc_id, all_entities, all_relations)

    def _build_result(self, doc_id, all_entities, all_relations):
        return {
            "doc_id": doc_id,
            "total_entities": len(all_entities),
            "total_relations": len(all_relations),
            "entity_types": self._count_entity_types(all_entities),
            "relation_types": self._count_relation_types(all_relations),
        }

    def _load_all_entities_for_doc(self, doc_id: str) -> List[Entity]:
        cursor = self.db.execute(
            "SELECT entity_id, name, entity_type, description, source_node_id, doc_id, properties "
            "FROM entities WHERE doc_id = ?", (doc_id,)
        )
        results = []
        for row in cursor.fetchall():
            results.append(Entity(
                entity_id=row[0], name=row[1], entity_type=row[2],
                description=row[3], source_node_id=row[4], doc_id=row[5],
                properties=json.loads(row[6]) if row[6] else {},
            ))
        return results

    def _build_with_rules(
        self,
        nodes: List[Dict],
        doc_id: str,
    ) -> Dict[str, Any]:
        all_entities = []
        all_relations = []

        for node in nodes:
            content = node.get("content", "")
            node_id = node.get("node_id", "")
            title = node.get("title", "")

            entities, relations = self._extract_with_rules(content, node_id, doc_id, title)
            all_entities.extend(entities)
            all_relations.extend(relations)

        return {
            "doc_id": doc_id,
            "total_entities": len(all_entities),
            "total_relations": len(all_relations),
            "entity_types": self._count_entity_types(all_entities),
            "relation_types": self._count_relation_types(all_relations),
        }

    def _extract_with_ner(
        self,
        content: str,
        node_id: str,
        doc_id: str,
        title: str = "",
    ):
        entities = []
        relations = []
        seen = set()

        if title and title not in seen:
            e = self._find_or_create_entity(title, "section", node_id, doc_id)
            entities.append(e)
            seen.add(title)

        quoted = re.findall(r'[""「»『]([^""「»』]{2,30})[""」』]', content)
        for name in quoted:
            if name in seen:
                continue
            seen.add(name)
            e = self._find_or_create_entity(name, "concept_term", node_id, doc_id)
            entities.append(e)

        camel_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', content)
        for name in camel_case:
            if name in seen or len(name) < 3:
                continue
            seen.add(name)
            e = self._find_or_create_entity(name, "technique", node_id, doc_id)
            entities.append(e)

        capitalized = re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3}\b', content)
        for name in capitalized:
            if name in seen or len(name) < 3:
                continue
            if any(kw in name.lower() for kw in ["the", "this", "that", "these", "those", "what", "how", "why"]):
                continue
            seen.add(name)
            e = self._find_or_create_entity(name, "concept", node_id, doc_id)
            entities.append(e)

        return entities, relations

    def _extract_with_rules(
        self,
        content: str,
        node_id: str,
        doc_id: str,
        title: str = "",
    ) -> Tuple[List[Entity], List[Relation]]:
        entities = []
        relations = []

        if title:
            title_entity = self._find_or_create_entity(title, "section", node_id, doc_id)
            entities.append(title_entity)

        patterns = [
            (r'(?:称为|叫做|定义为|是指|指的是)\s*[""「]?([^""」\n,，。；]{2,20})[""」]?', "definition"),
            (r'([^，。；\n]{2,15})(?:包括|包含|由.*组成|分为)', "composition"),
            (r'([^，。；\n]{2,15})(?:基于|依赖|使用|利用|借助)', "dependency"),
            (r'([^，。；\n]{2,15})(?:实现|实现了一个?|用于|用来)', "implementation"),
        ]

        extracted_names = set()
        for pattern, rel_type in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                name = match.strip()
                if len(name) < 2 or name in extracted_names:
                    continue
                extracted_names.add(name)

                entity = self._find_or_create_entity(name, "concept", node_id, doc_id)
                entities.append(entity)

                if title and title_entity:
                    relation = Relation(
                        source_entity_id=title_entity.entity_id,
                        target_entity_id=entity.entity_id,
                        relation_type=rel_type,
                        description=f"{title} {rel_type} {name}",
                        doc_id=doc_id,
                    )
                    self._save_relation(relation)
                    relations.append(relation)

        name_to_entity = {e.name: e for e in entities}
        desc_name_index = {}
        for e in entities:
            for other_name in name_to_entity:
                if other_name != e.name and other_name in e.description:
                    desc_name_index.setdefault(e.name, set()).add(other_name)

        for name, related_names in desc_name_index.items():
            entity = name_to_entity[name]
            for related_name in related_names:
                related_entity = name_to_entity[related_name]
                relation = Relation(
                    source_entity_id=entity.entity_id,
                    target_entity_id=related_entity.entity_id,
                    relation_type="related_to",
                    description=f"{entity.name} 与 {related_entity.name} 相关",
                    doc_id=doc_id,
                    weight=0.5,
                )
                self._save_relation(relation)
                relations.append(relation)

        return entities, relations

    def query_entity(self, name: str) -> Optional[Dict]:
        entity = self._find_entity_by_name(name)
        return entity.to_dict() if entity else None

    def get_entity_relations(self, entity_id: str, depth: int = 1) -> Dict[str, Any]:
        entity = self._load_entity(entity_id)
        if not entity:
            return {"entity": None, "relations": [], "neighbors": []}

        visited_entities = set()
        visited_relations = set()
        current_level = {entity_id}

        for _ in range(depth):
            next_level = set()
            for eid in current_level:
                if eid in visited_entities:
                    continue
                visited_entities.add(eid)

                cursor = self.db.execute(
                    "SELECT relation_id, source_entity_id, target_entity_id, relation_type, description, weight "
                    "FROM relations WHERE source_entity_id = ? OR target_entity_id = ?",
                    (eid, eid),
                )
                for row in cursor.fetchall():
                    rid = row[0]
                    if rid in visited_relations:
                        continue
                    visited_relations.add(rid)

                    neighbor_id = row[2] if row[1] == eid else row[1]
                    next_level.add(neighbor_id)

            current_level = next_level - visited_entities

        neighbor_entities = []
        for eid in visited_entities:
            e = self._load_entity(eid)
            if e:
                neighbor_entities.append(e.to_dict())

        neighbor_relations = []
        for rid in visited_relations:
            r = self._load_relation(rid)
            if r:
                neighbor_relations.append(r.to_dict())

        return {
            "entity": entity.to_dict(),
            "relations": neighbor_relations,
            "neighbors": neighbor_entities,
        }

    def multi_hop_query(
        self,
        source_name: str,
        target_name: str,
        max_hops: int = 3,
    ) -> List[Dict[str, Any]]:
        source = self._find_entity_by_name(source_name)
        target = self._find_entity_by_name(target_name)
        if not source or not target:
            return []

        paths = self._bfs_find_paths(
            source.entity_id, target.entity_id, max_hops
        )
        return paths

    def get_related_entities(self, entity_name: str, relation_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        entity = self._find_entity_by_name(entity_name)
        if not entity:
            return []

        if relation_type:
            cursor = self.db.execute(
                "SELECT r.relation_type, r.description, r.weight, e.entity_id, e.name, e.entity_type, e.description "
                "FROM relations r JOIN entities e ON "
                "(r.target_entity_id = e.entity_id AND r.source_entity_id = ?) OR "
                "(r.source_entity_id = e.entity_id AND r.target_entity_id = ?) "
                "WHERE r.relation_type = ? LIMIT ?",
                (entity.entity_id, entity.entity_id, relation_type, limit),
            )
        else:
            cursor = self.db.execute(
                "SELECT r.relation_type, r.description, r.weight, e.entity_id, e.name, e.entity_type, e.description "
                "FROM relations r JOIN entities e ON "
                "(r.target_entity_id = e.entity_id AND r.source_entity_id = ?) OR "
                "(r.source_entity_id = e.entity_id AND r.target_entity_id = ?) "
                "LIMIT ?",
                (entity.entity_id, entity.entity_id, limit),
            )

        rows = cursor.fetchall()
        return [
            {
                "relation_type": row[0],
                "relation_description": row[1],
                "weight": row[2],
                "entity_id": row[3],
                "name": row[4],
                "entity_type": row[5],
                "description": row[6],
            }
            for row in rows
        ]

    def get_graph_stats(self) -> Dict[str, Any]:
        cursor = self.db.execute("SELECT COUNT(*) FROM entities")
        entity_count = cursor.fetchone()[0]
        cursor = self.db.execute("SELECT COUNT(*) FROM relations")
        relation_count = cursor.fetchone()[0]
        cursor = self.db.execute("SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type")
        entity_types = dict(cursor.fetchall())
        cursor = self.db.execute("SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type")
        relation_types = dict(cursor.fetchall())
        return {
            "total_entities": entity_count,
            "total_relations": relation_count,
            "entity_types": entity_types,
            "relation_types": relation_types,
        }

    def _bfs_find_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int,
    ) -> List[Dict[str, Any]]:
        paths = []
        queue = [(source_id, [source_id], [])]
        visited = set()

        while queue:
            current_id, path, rel_path = queue.pop(0)

            if len(path) - 1 > max_hops:
                continue

            if current_id == target_id and len(path) > 1:
                entities = []
                for eid in path:
                    e = self._load_entity(eid)
                    if e:
                        entities.append({"entity_id": eid, "name": e.name, "type": e.entity_type})
                paths.append({
                    "path": entities,
                    "relations": rel_path,
                    "hops": len(path) - 1,
                })
                continue

            if current_id in visited:
                continue
            visited.add(current_id)

            cursor = self.db.execute(
                "SELECT relation_id, source_entity_id, target_entity_id, relation_type "
                "FROM relations WHERE source_entity_id = ? OR target_entity_id = ?",
                (current_id, current_id),
            )
            for row in cursor.fetchall():
                rid, src_id, tgt_id, rel_type = row
                neighbor_id = tgt_id if src_id == current_id else src_id
                if neighbor_id not in path:
                    new_rel_path = rel_path + [{"relation_id": rid, "type": rel_type}]
                    queue.append((neighbor_id, path + [neighbor_id], new_rel_path))

            if len(paths) >= 5:
                break

        return sorted(paths, key=lambda p: p["hops"])

    def get_related_entities_by_id(self, entity_id: str, relation_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        if relation_type:
            cursor = self.db.execute(
                "SELECT r.relation_type, r.description, r.weight, e.entity_id, e.name, e.entity_type, 'source' as direction "
                "FROM relations r JOIN entities e ON r.target_entity_id = e.entity_id "
                "WHERE r.source_entity_id = ? AND r.relation_type = ? "
                "UNION "
                "SELECT r.relation_type, r.description, r.weight, e.entity_id, e.name, e.entity_type, 'target' as direction "
                "FROM relations r JOIN entities e ON r.source_entity_id = e.entity_id "
                "WHERE r.target_entity_id = ? AND r.relation_type = ? "
                "LIMIT ?",
                (entity_id, relation_type, entity_id, relation_type, limit),
            )
        else:
            cursor = self.db.execute(
                "SELECT r.relation_type, r.description, r.weight, e.entity_id, e.name, e.entity_type, 'source' as direction "
                "FROM relations r JOIN entities e ON r.target_entity_id = e.entity_id "
                "WHERE r.source_entity_id = ? "
                "UNION "
                "SELECT r.relation_type, r.description, r.weight, e.entity_id, e.name, e.entity_type, 'target' as direction "
                "FROM relations r JOIN entities e ON r.source_entity_id = e.entity_id "
                "WHERE r.target_entity_id = ? "
                "LIMIT ?",
                (entity_id, entity_id, limit),
            )
        return [
            {
                "relation_type": row[0],
                "description": row[1],
                "weight": row[2],
                "entity_id": row[3],
                "name": row[4],
                "entity_type": row[5],
                "direction": row[6] if len(row) > 6 else "",
            }
            for row in cursor.fetchall()
        ]

    def _find_or_create_entity(
        self,
        name: str,
        entity_type: str,
        node_id: str,
        doc_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        existing_id = self._resolve_entity_conflict(name)
        if existing_id:
            existing = self._load_entity(existing_id)
            if existing:
                if properties:
                    existing.properties.update(properties)
                    self._save_entity(existing)
                return existing

        entity = Entity(
            name=name,
            entity_type=entity_type,
            source_node_id=node_id,
            doc_id=doc_id,
            properties=properties or {},
        )
        self._save_entity(entity)
        self._name_to_id[name] = entity.entity_id
        return entity

    def _find_entity_by_name(self, name: str) -> Optional[Entity]:
        cursor = self.db.execute(
            "SELECT entity_id, name, entity_type, description, source_node_id, doc_id, properties "
            "FROM entities WHERE name = ? LIMIT 1",
            (name,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return Entity(
            entity_id=row[0],
            name=row[1],
            entity_type=row[2],
            description=row[3],
            source_node_id=row[4],
            doc_id=row[5],
            properties=json.loads(row[6]) if row[6] else {},
        )

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        safe = escape_like(query)
        like = self.db.execute(
            "SELECT entity_id, name, entity_type, description FROM entities WHERE name LIKE ? ESCAPE '\\' LIMIT ?",
            (f"%{safe}%", limit * 2),
        )
        results = [{"entity_id": r[0], "name": r[1], "entity_type": r[2], "description": r[3]} for r in like.fetchall()]
        if results:
            return results[:limit]

        like_desc = self.db.execute(
            "SELECT entity_id, name, entity_type, description FROM entities WHERE description LIKE ? ESCAPE '\\' LIMIT ?",
            (f"%{safe}%", limit),
        )
        results = [{"entity_id": r[0], "name": r[1], "entity_type": r[2], "description": r[3]} for r in like_desc.fetchall()]
        if results:
            return results[:limit]

        import difflib
        all_names = self.db.execute("SELECT name FROM entities").fetchall()
        names = [r[0] for r in all_names]
        close = difflib.get_close_matches(query, names, n=limit, cutoff=0.3)
        if close:
            matched = set(close)
            rows = self.db.execute(
                f"SELECT entity_id, name, entity_type, description FROM entities WHERE name IN ({','.join('?' for _ in close)})",
                list(close),
            ).fetchall()
            return [{"entity_id": r[0], "name": r[1], "entity_type": r[2], "description": r[3]} for r in rows]
        return []

    def _load_entity(self, entity_id: str) -> Optional[Entity]:
        cursor = self.db.execute(
            "SELECT entity_id, name, entity_type, description, source_node_id, doc_id, properties "
            "FROM entities WHERE entity_id = ?",
            (entity_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return Entity(
            entity_id=row[0],
            name=row[1],
            entity_type=row[2],
            description=row[3],
            source_node_id=row[4],
            doc_id=row[5],
            properties=json.loads(row[6]) if row[6] else {},
        )

    def _load_relation(self, relation_id: str) -> Optional[Relation]:
        cursor = self.db.execute(
            "SELECT relation_id, source_entity_id, target_entity_id, relation_type, description, weight, doc_id "
            "FROM relations WHERE relation_id = ?",
            (relation_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return Relation(
            relation_id=row[0],
            source_entity_id=row[1],
            target_entity_id=row[2],
            relation_type=row[3],
            description=row[4],
            weight=row[5],
            doc_id=row[6],
        )

    def _save_entity(self, entity: Entity):
        self.db.execute(
            """INSERT OR REPLACE INTO entities
            (entity_id, name, entity_type, description, source_node_id, doc_id, properties)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entity.entity_id,
                entity.name,
                entity.entity_type,
                entity.description,
                entity.source_node_id,
                entity.doc_id,
                json.dumps(entity.properties, ensure_ascii=False),
            ),
        )
        self.db.commit()

    def _save_relation(self, relation: Relation):
        self.db.execute(
            """INSERT OR REPLACE INTO relations
            (relation_id, source_entity_id, target_entity_id, relation_type, description, weight, doc_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                relation.relation_id,
                relation.source_entity_id,
                relation.target_entity_id,
                relation.relation_type,
                relation.description,
                relation.weight,
                relation.doc_id,
            ),
        )
        self.db.commit()

    def _count_entity_types(self, entities: List[Entity]) -> Dict[str, int]:
        counts = defaultdict(int)
        for e in entities:
            counts[e.entity_type] += 1
        return dict(counts)

    def _count_relation_types(self, relations: List[Relation]) -> Dict[str, int]:
        counts = defaultdict(int)
        for r in relations:
            counts[r.relation_type] += 1
        return dict(counts)

    def close(self):
        pass  # DatabaseManager is shared, closed at app shutdown

    def _load_name_index(self):
        cursor = self.db.execute("SELECT entity_id, name FROM entities")
        for row in cursor.fetchall():
            self._name_to_id[row[1]] = row[0]

    def _resolve_entity_conflict(self, name: str) -> Optional[str]:
        exact = self._find_entity_by_name(name)
        if exact:
            return exact.entity_id

        alias_target = self._alias_map.get(name)
        if alias_target and alias_target in self._name_to_id:
            return self._name_to_id[alias_target]

        candidates = self.db.execute(
            "SELECT entity_id, name FROM entities ORDER BY length(name) ASC LIMIT 20"
        ).fetchall()

        # Phase 1: vector similarity (fast, batch)
        if self.embed_func:
            candidate_names = [cname for _, cname in candidates]
            try:
                all_texts = [name] + candidate_names
                vecs = self.embed_func(all_texts)
                if not vecs or len(vecs) == 0:
                    return self._llm_match(name, candidates, context_hints)
                name_vec = vecs[0]
                scored = []
                for i, (cid, cname) in enumerate(candidates):
                    sim = sum(a * b for a, b in zip(name_vec, vecs[i + 1])) / (
                        (sum(a * a for a in name_vec) ** 0.5) * (sum(b * b for b in vecs[i + 1]) ** 0.5) + 1e-10
                    )
                    scored.append((sim, cid, cname))
                scored.sort(key=lambda x: -x[0])
                best_sim, best_cid, best_cname = scored[0]
                if best_sim > 0.88:
                    self._alias_map[name] = best_cname
                    return best_cid
                # Borderline: top-3 with LLM judge with context
                if self.llm_func:
                    for sim, cid, cname in scored[:3]:
                        if sim > 0.72:
                            try:
                                if self._llm_judge_same_concept(name, cname):
                                    self._alias_map[name] = cname
                                    return cid
                            except Exception:
                                continue
                return None
            except Exception as e:
                logger.debug(f"Embedding-based entity matching failed: {e}")

        # Phase 2: substring + LLM fallback (only when embed_func unavailable)
        if self.llm_func:
            for cid, cname in candidates:
                if len(cname) < 2 or len(name) < 2:
                    continue
                if cname.lower() in name.lower() or name.lower() in cname.lower():
                    try:
                        if self._llm_judge_same_concept(name, cname):
                            self._alias_map[name] = cname
                            return cid
                    except Exception:
                        continue
        return None

    def _llm_judge_same_concept(self, name_a: str, name_b: str) -> bool:
        if not self.llm_func:
            return False
        prompt = f"""请判断以下两个名称是否指代同一个概念/事物。
只需回答"是"或"否"，不要其他内容。
名称1: {name_a}
名称2: {name_b}"""
        response = self.llm_func(prompt).strip().lower()
        return "是" in response or "yes" in response
