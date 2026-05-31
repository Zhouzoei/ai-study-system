import logging
from typing import List, Dict, Any, Optional, Callable

from core.knowledge_graph import KnowledgeGraphBuilder, Relation
from core.knowledge_distiller import KnowledgeDistiller
from core.learning_context import LearningContextBuilder
from core.course_manager import CourseManager
from core.progress_tracker import ProgressTracker
from core.document_manager import DocumentManager

logger = logging.getLogger(__name__)


class KnowledgeService:
    """Encapsulates knowledge graph, distillation, learning context, and course management."""

    def __init__(self, llm_func=None, embed_func=None):
        self.llm_func = llm_func
        self.embed_func = embed_func

        self.knowledge_graph = KnowledgeGraphBuilder(llm_func=llm_func, embed_func=embed_func)
        self.learning_context = None  # wired after progress_tracker is available
        self.course_manager = CourseManager()
        self._active_course_id: Optional[str] = None

    def wire(self, progress_tracker, document_manager, conversation_memory):
        self.learning_context = LearningContextBuilder(
            progress_tracker=progress_tracker,
            document_manager=document_manager,
            conversation_memory=conversation_memory,
            knowledge_graph=self.knowledge_graph,
        )

    def build_from_nodes(self, l2_dicts: List[Dict], doc_id: str) -> Dict[str, Any]:
        return self.knowledge_graph.build_from_nodes(l2_dicts, doc_id)

    def distill_knowledge(self, l3_dicts: List[Dict], doc_id: str) -> List:
        distiller = KnowledgeDistiller(llm_func=self.llm_func)
        return distiller.distill(l3_dicts, doc_id)

    def store_prerequisites(self, knowledge_units, doc_id: str):
        for unit in knowledge_units:
            if not unit.prerequisites:
                continue
            for prereq_name in unit.prerequisites:
                try:
                    self.knowledge_graph._find_or_create_entity(
                        prereq_name, "concept",
                        unit.source_node_ids[0] if unit.source_node_ids else "", doc_id
                    )
                    self.knowledge_graph._find_or_create_entity(
                        unit.concept, "concept",
                        unit.source_node_ids[0] if unit.source_node_ids else "", doc_id
                    )
                    prereq_entity = self.knowledge_graph._find_entity_by_name(prereq_name)
                    concept_entity = self.knowledge_graph._find_entity_by_name(unit.concept)
                    if prereq_entity and concept_entity:
                        relation = Relation(
                            source_entity_id=prereq_entity.entity_id,
                            target_entity_id=concept_entity.entity_id,
                            relation_type="prerequisite_of",
                            description=f"{prereq_name} 是 {unit.concept} 的前置知识",
                            doc_id=doc_id,
                        )
                        self.knowledge_graph._save_relation(relation)
                except Exception:
                    continue

    def query_entity(self, name: str) -> Optional[Dict]:
        return self.knowledge_graph.query_entity(name)

    def get_entity_relations(self, entity_name: str, depth: int = 1) -> Dict[str, Any]:
        entity = self.knowledge_graph.query_entity(entity_name)
        if not entity:
            return {"entity": None, "relations": [], "neighbors": []}
        return self.knowledge_graph.get_entity_relations(entity["entity_id"], depth)

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.knowledge_graph.search_entities(query, limit)

    def build_learning_context(self, user_id: str, session_id: str) -> str:
        if self.learning_context:
            return self.learning_context.build_system_context(user_id, session_id)
        return ""

    @property
    def active_course_id(self) -> str:
        if self._active_course_id:
            return self._active_course_id
        courses = self.course_manager.list_courses()
        if courses:
            self._active_course_id = courses[0]["course_id"]
            return self._active_course_id
        course = self.course_manager.create_course("通用学习")
        self._active_course_id = course.course_id
        return self._active_course_id

    def set_active_course(self, course_id: str):
        self._active_course_id = course_id

    def get_stats(self) -> Dict[str, Any]:
        return {
            "knowledge_graph": self.knowledge_graph.get_graph_stats(),
        }

    def close(self):
        self.knowledge_graph.close()
        self.course_manager.close()
