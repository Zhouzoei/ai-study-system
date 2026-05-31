import json
import logging
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

GRAPH_RAG_EXTRACT_TEMPLATE = """从以下问题中提取关键实体（概念、技术、术语），每个实体一行。
只输出实体名称列表，每行一个，不要编号。

问题: {question}

实体:"""


class GraphRAGAugmenter:
    """Knowledge-graph-augmented retrieval and query expansion.

    For a given query:
      1. Extract key entities via LLM
      2. Look up each entity in the KG → get description + relations
      3. Expand the query with related entity names
      4. Return structured KG context for injection into generation prompt
    """

    def __init__(
        self,
        knowledge_graph,
        llm_func: Optional[Callable] = None,
        embed_func: Optional[Callable] = None,
    ):
        self.kg = knowledge_graph
        self.llm_func = llm_func
        self.embed_func = embed_func

    def augment_query(self, question: str) -> Dict[str, Any]:
        entities = self._extract_entities(question)
        kg_info = self._lookup_entities(entities, question)
        expanded_terms = self._expand_terms(kg_info)
        return {
            "kg_context": kg_info,
            "expanded_terms": expanded_terms,
            "entities_found": [e for e in entities if any(
                e.lower() in k.get("name", "").lower()
                or k.get("name", "").lower() in e.lower()
                for k in kg_info
            )],
        }

    def build_kg_prompt_block(self, kg_result: Dict[str, Any]) -> str:
        kg_info = kg_result.get("kg_context", [])
        if not kg_info:
            return ""
        blocks = []
        for item in kg_info:
            name = item.get("name", "")
            desc = item.get("description", "")
            neighbors = item.get("neighbors", [])
            block = f"- **{name}**"
            if desc:
                block += f": {desc[:200]}"
            if neighbors:
                rel_text = "、".join(
                    f"{n.get('name', '?')}({n.get('relation', '相关')})"
                    for n in neighbors[:4]
                )
                block += f"\n  关联: {rel_text}"
            blocks.append(block)
        if not blocks:
            return ""
        return "\n\n[知识图谱上下文]:\n" + "\n".join(blocks)

    def _extract_entities(self, question: str) -> List[str]:
        if not self.llm_func:
            import re
            words = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z][a-zA-Z0-9_-]+", question)
            return [w for w in words if len(w) > 1][:5]
        try:
            response = self.llm_func(GRAPH_RAG_EXTRACT_TEMPLATE.format(question=question)).strip()
            entities = [
                line.strip().lstrip("1234567890.、-· ")
                for line in response.split("\n")
                if line.strip() and not line.startswith("```")
            ]
            return entities[:5] if entities else []
        except Exception as e:
            logger.debug(f"GraphRAG entity extraction failed: {e}")
            return []

    def _lookup_entities(self, entities: List[str], question: str) -> List[Dict]:
        results = []
        seen_names = set()
        if self.embed_func:
            try:
                kg_entities = self.kg.search_entities(question, limit=5)
                for e in kg_entities:
                    name = e.get("name", "")
                    if name and name not in seen_names:
                        seen_names.add(name)
                        entity_id = e.get("entity_id", "")
                        rel_data = self.kg.get_entity_relations(entity_id, depth=1) if entity_id else {}
                        neighbors = [
                            {"name": n.get("name", ""), "relation": r.get("relation_type", "相关")}
                            for n, r in zip(
                                rel_data.get("neighbors", []),
                                rel_data.get("relations", []),
                            )
                        ] if rel_data else []
                        results.append({
                            "name": name,
                            "entity_type": e.get("entity_type", "concept"),
                            "description": e.get("description", ""),
                            "neighbors": neighbors[:6],
                        })
            except Exception as e:
                logger.debug(f"GraphRAG embedding search failed: {e}")

        for entity_name in entities:
            if entity_name in seen_names:
                continue
            try:
                entity = self.kg.query_entity(entity_name)
                if entity:
                    seen_names.add(entity_name)
                    rel_data = self.kg.get_entity_relations(entity["entity_id"], depth=1)
                    neighbors = [
                        {"name": n.get("name", ""), "relation": r.get("relation_type", "相关")}
                        for n, r in zip(
                            rel_data.get("neighbors", []),
                            rel_data.get("relations", []),
                        )
                    ] if rel_data else []
                    results.append({
                        "name": entity_name,
                        "entity_type": entity.get("entity_type", "concept"),
                        "description": entity.get("description", ""),
                        "neighbors": neighbors[:6],
                    })
            except Exception as e:
                logger.debug(f"GraphRAG lookup failed for '{entity_name}': {e}")
        return results

    def _expand_terms(self, kg_info: List[Dict]) -> List[str]:
        terms = []
        for item in kg_info:
            name = item.get("name", "")
            if name:
                terms.append(name)
            for n in item.get("neighbors", [])[:3]:
                nname = n.get("name", "")
                if nname:
                    terms.append(nname)
        return terms[:10]
