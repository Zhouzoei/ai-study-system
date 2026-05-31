import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeUnit:
    unit_id: str = ""
    concept: str = ""
    definition: str = ""
    bloom_level: str = "理解"
    prerequisites: List[str] = field(default_factory=list)
    source_node_ids: List[str] = field(default_factory=list)
    doc_id: str = ""
    difficulty: float = 0.5
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.unit_id:
            self.unit_id = f"ku_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "concept": self.concept,
            "definition": self.definition,
            "bloom_level": self.bloom_level,
            "prerequisites": self.prerequisites,
            "source_node_ids": self.source_node_ids,
            "doc_id": self.doc_id,
            "difficulty": self.difficulty,
            "keywords": self.keywords,
            "examples": self.examples,
        }


class KnowledgeDistiller:
    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm_func = llm_func

    def distill(
        self,
        l3_nodes: List[Dict],
        doc_id: str = "",
        batch_size: int = 4,
    ) -> List[KnowledgeUnit]:
        if not l3_nodes:
            return []

        all_units = []
        for i in range(0, len(l3_nodes), batch_size):
            batch = l3_nodes[i:i + batch_size]
            try:
                units = self._batch_extract(batch, doc_id)
                all_units.extend(units)
            except Exception as e:
                logger.warning(f"Distillation batch failed at offset {i}: {e}")
                for node in batch:
                    fallback = self._fallback_extract(node, doc_id)
                    if fallback:
                        all_units.append(fallback)
        return all_units

    def _batch_extract(
        self,
        nodes_batch: List[Dict],
        doc_id: str,
    ) -> List[KnowledgeUnit]:
        if not self.llm_func:
            return [self._fallback_extract(n, doc_id) for n in nodes_batch if n.get("content", "").strip()]

        contents = []
        for node in nodes_batch:
            title = node.get("title", "")
            content = node.get("content", "")[:800]
            contents.append(f"--- 段落 {len(contents)+1} ---\n标题: {title}\n内容: {content}")

        batch_text = "\n\n".join(contents)
        prompt = f"""从以下文档段落中提取独立的"知识点"。每个知识点应该是一个可以独立学习和评估的原子概念。

输出 JSON 格式，不要其他内容:
{{
  "knowledge_units": [
    {{
      "concept": "概念名称",
      "definition": "精确定义（1-2句话）",
      "bloom_level": "记忆/理解/应用/分析/评价/创造",
      "prerequisites": ["前置概念1", "前置概念2"],
      "keywords": ["关键词1", "关键词2"],
      "difficulty": 0.5,
      "examples": ["示例1"]
    }}
  ]
}}

文档段落:
{batch_text}"""

        try:
            response = self.llm_func(prompt).strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(response)
            units_data = parsed.get("knowledge_units", [])
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM distillation parse failed: {e}")
            units_data = []

        if not units_data:
            return [self._fallback_extract(n, doc_id) for n in nodes_batch if n.get("content", "").strip()]

        units = []
        for ud in units_data:
            concept = ud.get("concept", "").lower()
            concept_words = [w for w in concept.split() if len(w) > 1]

            best_node_id = ""
            best_score = 0
            for node in nodes_batch:
                content = (node.get("title", "") + " " + node.get("content", "")).lower()
                score = sum(1 for kw in concept_words if kw in content)
                if score > best_score:
                    best_score = score
                    best_node_id = node.get("node_id", "")

            unit = KnowledgeUnit(
                concept=ud.get("concept", f"概念_{len(units)}"),
                definition=ud.get("definition", ""),
                bloom_level=ud.get("bloom_level", "理解"),
                prerequisites=ud.get("prerequisites", []),
                source_node_ids=[best_node_id] if best_node_id else [],
                doc_id=doc_id,
                difficulty=float(ud.get("difficulty", 0.5)),
                keywords=ud.get("keywords", []),
                examples=ud.get("examples", []),
            )
            units.append(unit)

        return units

    def _fallback_extract(
        self,
        node: Dict,
        doc_id: str,
    ) -> Optional[KnowledgeUnit]:
        content = node.get("content", "").strip()
        title = node.get("title", "").strip()
        if not content and not title:
            return None

        concept = title if title else content[:50]
        keywords = [w for w in concept.replace(":", "").split() if len(w) > 1][:5]
        return KnowledgeUnit(
            concept=concept,
            definition=content[:200],
            bloom_level="理解",
            source_node_ids=[node.get("node_id", "")] if node.get("node_id") else [],
            doc_id=doc_id,
            difficulty=0.5,
            keywords=keywords if keywords else [concept[:10]],
        )
