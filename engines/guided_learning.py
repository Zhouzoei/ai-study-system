import json
import re
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MASTERY_ORDER = {"unknown": 0, "exposed": 1, "familiar": 2, "proficient": 3, "mastered": 4}
READY_THRESHOLD = 2  # familiar


@dataclass
class ConceptCard:
    concept: str = ""
    definition: str = ""
    bloom_level: str = ""
    difficulty: float = 0.5
    keywords: List[str] = field(default_factory=list)
    prerequisites: List[Dict] = field(default_factory=list)
    mastery: str = "unknown"
    readiness: bool = False
    blocked_by: List[str] = field(default_factory=list)
    key_passages: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "concept": self.concept,
            "definition": self.definition,
            "bloom_level": self.bloom_level,
            "difficulty": self.difficulty,
            "keywords": self.keywords,
            "prerequisites": self.prerequisites,
            "mastery": self.mastery,
            "readiness": self.readiness,
            "blocked_by": self.blocked_by,
            "key_passages": self.key_passages,
            "examples": self.examples,
        }


@dataclass
class DailyPlanItem:
    concept: str = ""
    definition: str = ""
    difficulty: float = 0.5
    bloom_level: str = ""
    mastery: str = "unknown"
    readiness_pct: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "concept": self.concept,
            "definition": self.definition[:120] if self.definition else "",
            "difficulty": self.difficulty,
            "bloom_level": self.bloom_level,
            "mastery": self.mastery,
            "readiness_pct": self.readiness_pct,
            "reason": self.reason,
        }


class GuidedLearningEngine:
    def __init__(
        self,
        knowledge_graph=None,
        progress_tracker=None,
        storage=None,
        pipeline=None,
        llm_func: Optional[Callable] = None,
        embed_func: Optional[Callable] = None,
    ):
        self.knowledge_graph = knowledge_graph
        self.progress_tracker = progress_tracker
        self.storage = storage
        self.pipeline = pipeline
        self.llm_func = llm_func
        self.embed_func = embed_func

    # ── Readiness (就绪度评估) ──

    def compute_readiness(
        self, concept_name: str, user_id: str = "default", max_depth: int = 3
    ) -> Dict:
        entity = self._resolve_entity(concept_name)
        if not entity:
            return {"ready": False, "blocked_by": [], "readiness_pct": 0.0,
                    "reason": f"知识库中未找到「{concept_name}」"}

        prereqs = self._collect_prerequisites(entity["entity_id"], max_depth=max_depth)
        if not prereqs:
            return {"ready": True, "blocked_by": [], "readiness_pct": 100.0,
                    "reason": "无需前置知识，可直接学习"}

        blocked = []
        total = len(prereqs)
        mastered_count = 0
        for pid, pname, mastery_val in prereqs:
            if mastery_val < READY_THRESHOLD:
                blocked.append({"name": pname, "mastery": mastery_val})
            else:
                mastered_count += 1

        pct = round(mastered_count / total * 100) if total > 0 else 100
        ready = len(blocked) == 0
        reason = "所有前置知识已掌握" if ready else f"还需掌握 {len(blocked)} 个前置知识点"

        return {
            "ready": ready,
            "blocked_by": blocked,
            "readiness_pct": pct,
            "reason": reason,
        }

    # ── Daily Plan (今天学什么) ──

    def get_daily_plan(
        self, user_id: str = "default", limit: int = 5, doc_id: Optional[str] = None
    ) -> List[DailyPlanItem]:
        candidates = self._get_candidate_concepts(doc_id)
        if not candidates:
            return []

        scored = []
        for name in candidates:
            rec = self._get_mastery(name, user_id)
            mastery_val = MASTERY_ORDER.get(rec, 0)
            if mastery_val >= MASTERY_ORDER["proficient"]:
                continue

            readiness = self.compute_readiness(name, user_id)
            if not readiness["ready"]:
                continue

            entity = self._resolve_entity(name)
            difficulty = float(entity.get("properties", {}).get("difficulty", 0.5)) if entity else 0.5
            bloom = (entity.get("properties", {}).get("bloom_level", "") if entity else "")
            definition = (entity.get("properties", {}).get("definition", "") if entity else "")

            reason = f"掌握度: {rec}"
            if mastery_val <= MASTERY_ORDER["exposed"]:
                reason += "，建议开始学习"
            elif mastery_val == MASTERY_ORDER["familiar"]:
                reason += "，建议巩固"

            scored.append({
                "name": name,
                "mastery_val": mastery_val,
                "difficulty": difficulty,
                "mastery": rec,
                "bloom": bloom,
                "definition": definition,
                "reason": reason,
            })

        scored.sort(key=lambda x: (x["mastery_val"], x["difficulty"]))
        items = []
        for s in scored[:limit]:
            items.append(DailyPlanItem(
                concept=s["name"],
                definition=s["definition"],
                difficulty=s["difficulty"],
                bloom_level=s["bloom"],
                mastery=s["mastery"],
                readiness_pct=100.0,
                reason=s["reason"],
            ))
        return items

    # ── Concept Card (概念学习卡片) ──

    def get_concept_card(
        self, concept_name: str, user_id: str = "default"
    ) -> ConceptCard:
        entity = self._resolve_entity(concept_name)
        card = ConceptCard(concept=concept_name)

        if entity:
            props = entity.get("properties", {}) if isinstance(entity, dict) else {}
            card.definition = props.get("definition", "")
            card.bloom_level = props.get("bloom_level", "")
            try:
                card.difficulty = float(props.get("difficulty", 0.5))
            except (ValueError, TypeError):
                card.difficulty = 0.5
            try:
                card.keywords = json.loads(props.get("keywords", "[]"))
            except (json.JSONDecodeError, TypeError):
                card.keywords = []
            try:
                card.examples = json.loads(props.get("examples", "[]"))
            except (json.JSONDecodeError, TypeError):
                card.examples = []

        mastery_rec = self._get_mastery(concept_name, user_id)
        card.mastery = mastery_rec

        readiness = self.compute_readiness(concept_name, user_id)
        card.readiness = readiness["ready"]
        card.blocked_by = [b["name"] for b in readiness.get("blocked_by", [])]

        prereq_entities = self._collect_prerequisites(entity["entity_id"] if entity else "", max_depth=1)
        card.prerequisites = [
            {"name": p[1], "mastery": MASTERY_ORDER.get(p[2], 0)}
            for p in prereq_entities
        ]

        key_passages = self._retrieve_key_passages(concept_name, top_k=5)
        card.key_passages = [p.get("content", "")[:500] for p in key_passages[:3]]

        # If entity has no definition, try to extract from passages via LLM
        if not card.definition and self.llm_func and key_passages:
            passage_text = "\n\n".join(p.get("content", "")[:800] for p in key_passages[:3])
            try:
                resp = self.llm_func(
                    f"""请从以下文本中提取关于「{concept_name}」的核心信息，按格式输出。

文本:
{passage_text}

请按以下格式输出：
定义: [一句话定义]
Bloom认知层级: [记忆/理解/应用/分析/评价/创造]
难度: [0.1-1.0之间的数字]
关键词: [关键词1, 关键词2, 关键词3]"""
                )
                for line in resp.strip().split("\n"):
                    if line.startswith("定义:"):
                        card.definition = line[3:].strip()
                    elif line.startswith("Bloom认知层级:") or line.startswith("Bloom认知") or line.startswith("Bloom"):
                        card.bloom_level = line.split(":")[-1].strip()
                    elif line.startswith("难度:"):
                        try:
                            card.difficulty = float(line.split(":")[-1].strip())
                        except (ValueError, TypeError):
                            pass
                    elif line.startswith("关键词"):
                        kw_raw = line.split(":")[-1].strip()
                        card.keywords = [k.strip() for k in kw_raw.replace("，", ",").split(",") if k.strip()]
            except Exception as e:
                logger.debug(f"LLM concept extraction failed: {e}")

        # If still no definition, use the first passage content as definition
        if not card.definition and key_passages:
            first = key_passages[0].get("content", "")
            card.definition = first[:300] + ("..." if len(first) > 300 else "")

        return card

    # ── Concept Quiz (概念级测验) ──

    def generate_concept_quiz(
        self, concept_name: str, sub_type: str = "choice"
    ) -> Dict:
        entity = self._resolve_entity(concept_name)
        props = entity.get("properties", {}) if entity else {}
        definition = props.get("definition", "")
        bloom = props.get("bloom_level", "理解")
        keywords = props.get("keywords", [])

        passages = self._retrieve_key_passages(concept_name)
        context_text = ""
        if passages:
            parts = [f"[来源 {i+1}]\n{p['content']}" for i, p in enumerate(passages[:4])]
            context_text = "\n\n---\n\n".join(parts)

        type_prompts = {
            "choice": "请出一道关于此概念的单项选择题，四个选项，标注正确答案。",
            "judgment": "请出一道关于此概念的判断题。",
            "fill": "请出一道关于此概念的填空题。",
        }
        instruction = type_prompts.get(sub_type, type_prompts["choice"])

        difficulty_hint = ""
        try:
            diff = float(props.get("difficulty", 0.5))
            if diff < 0.4:
                difficulty_hint = "（基础难度）"
            elif diff < 0.7:
                difficulty_hint = "（中等难度）"
            else:
                difficulty_hint = "（较高难度）"
        except (ValueError, TypeError):
            pass

        if self.llm_func:
            prompt = f"""你是一个出题助手。请基于以下概念信息出一道测验题{difficulty_hint}。

概念: {concept_name}
定义: {definition}
认知层级(Bloom): {bloom}

参考资料:
{context_text}

{instruction}

格式要求：
## 题目
[题目内容]

## 选项（选择题）
A. [选项] B. [选项] C. [选项] D. [选项]

## 正确答案
[答案]

## 解析
[基于参考资料的解析]"""

            try:
                answer = self.llm_func(prompt)
                return {
                    "concept": concept_name,
                    "sub_type": sub_type,
                    "raw": answer,
                    "success": True,
                }
            except Exception as e:
                logger.warning(f"Generate concept quiz LLM failed: {e}")

        return {
            "concept": concept_name,
            "sub_type": sub_type,
            "raw": "",
            "success": False,
            "error": "LLM不可用",
        }

    # ── Helpers ──

    def _resolve_entity(self, name: str) -> Optional[Dict]:
        if not self.knowledge_graph:
            return None
        entity = self.knowledge_graph.query_entity(name)
        if entity:
            return entity
        cleaned = re.sub(r"^\d+(?:[.\-]\d+)*\s*", "", name).strip()
        if cleaned != name:
            entity = self.knowledge_graph.query_entity(cleaned)
            if entity:
                return entity
        candidates = self.knowledge_graph.search_entities(name, limit=5)
        if candidates:
            for c in candidates:
                e = self.knowledge_graph.query_entity(c["name"])
                if e:
                    props = e.get("properties", {}) if isinstance(e, dict) else {}
                    if props.get("definition"):
                        return e
            return self.knowledge_graph.query_entity(candidates[0]["name"])
        if cleaned != name:
            candidates = self.knowledge_graph.search_entities(cleaned, limit=5)
            if candidates:
                return self.knowledge_graph.query_entity(candidates[0]["name"])
        return None

    def _collect_prerequisites(
        self, entity_id: str, max_depth: int = 3, depth: int = 0
    ) -> List[tuple]:
        if depth >= max_depth or not self.knowledge_graph:
            return []
        results = []
        related = self.knowledge_graph.get_related_entities_by_id(
            entity_id, relation_type="prerequisite_of"
        )
        for rel in related:
            pid = rel.get("entity_id", "")
            pname = rel.get("name", "")
            pmastery = self._mastery_value(pname)
            results.append((pid, pname, pmastery))
            results.extend(self._collect_prerequisites(pid, max_depth, depth + 1))
        seen = set()
        deduped = []
        for pid, pname, pm in results:
            if pid not in seen:
                seen.add(pid)
                deduped.append((pid, pname, pm))
        return deduped

    def _get_mastery(self, concept_name: str, user_id: str) -> str:
        if not self.progress_tracker:
            return "unknown"
        record = self.progress_tracker._find_record(concept_name, user_id)
        if record:
            return record.mastery.value if hasattr(record.mastery, "value") else str(record.mastery)
        return "unknown"

    def _mastery_value(self, concept_name: str, user_id: str = "default") -> int:
        m = self._get_mastery(concept_name, user_id)
        return MASTERY_ORDER.get(m, 0)

    def _get_candidate_concepts(self, doc_id: Optional[str] = None) -> List[str]:
        names = set()
        if self.knowledge_graph:
            cursor = self.knowledge_graph.db.execute(
                "SELECT name FROM entities WHERE entity_type = 'distilled_concept'"
            )
            names.update(row[0] for row in cursor.fetchall())
        if not names and self.progress_tracker:
            cursor = self.progress_tracker.db.execute(
                "SELECT DISTINCT knowledge_node_id FROM knowledge_records LIMIT 50"
            )
            names.update(row[0] for row in cursor.fetchall())
        return sorted(names)

    def _retrieve_key_passages(
        self, query: str, top_k: int = 5
    ) -> List[Dict]:
        if self.pipeline:
            try:
                result = self.pipeline.query(query, use_hybrid=True, top_k=top_k)
                sources = result.get("context_sources", [])
                return [
                    {"content": s.get("excerpt", ""), "score": 1.0 - i * 0.1,
                     "title": s.get("level_title", ""), "node_id": s.get("node_id", "")}
                    for i, s in enumerate(sources)
                ]
            except Exception as e:
                logger.warning(f"Pipeline retrieval failed: {e}")
        if not self.embed_func or not self.storage:
            return []
        try:
            qvec = self.embed_func([query])
            if not qvec or len(qvec) == 0:
                return []
            results = self.storage.search_l3_by_vector(qvec[0], top_k=top_k)
            return [
                {"content": r.get("content", ""), "score": r.get("score", 0),
                 "title": r.get("title", ""), "node_id": r.get("node_id", "")}
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Vector-only retrieval failed: {e}")
            return []
