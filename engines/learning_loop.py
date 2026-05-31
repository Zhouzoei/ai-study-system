import logging
import time
from typing import Dict, Any, Optional, Generator, List

from engines.intent_router import IntentType
from engines.resilience import ErrorCode

logger = logging.getLogger(__name__)


class LearningLoop:
    """Thin scheduler that completes the learn → assess → update → plan cycle.

    Each turn:
      1. Pre-check user knowledge state (weak nodes, due reviews)
      2. Route to the right handler (quiz / review / RAG QA)
      3. Post-process: record exposures, update mastery, mark KG nodes

    This wires together existing modules without adding new algorithms.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._graph_rag = None
        self._self_rag = None

    @property
    def graph_rag(self):
        if self._graph_rag is None:
            self._graph_rag = getattr(self.pipeline, 'graph_rag', None)
        return self._graph_rag

    @property
    def self_rag(self):
        if self._self_rag is None:
            self._self_rag = getattr(self.pipeline, 'self_rag', None)
        return self._self_rag

    # ── Public entry ───────────────────────────────────────────────

    def process(
        self,
        message: str,
        session_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: str = "default",
        mode: str = "hybrid",
    ) -> Generator[Dict[str, Any], None, None]:
        intent_result = self.pipeline.intent_router.route(message)
        yield {
            "type": "intent",
            "intent": intent_result.intent.label,
            "confidence": intent_result.confidence,
            "topic": intent_result.topic,
        }

        # ── Phase 1: Pre-check knowledge state ──
        pre_check = self._pre_check(user_id)
        if pre_check["has_weak_nodes"] and intent_result.intent in (IntentType.QA, IntentType.CHAT):
            yield {"type": "progress", "content": "检测到薄弱知识点，正在针对性出题..."}
            for event in self.pipeline.quiz_agent.generate_stream(
                message=topic_text(pre_check["weak_nodes"]),
                topic=intent_result.topic or message,
                sub_type="choice",
                doc_id=doc_id,
            ):
                yield event
            yield {"type": "loop_done", "pre_check": pre_check}
            return

        if pre_check["has_due_reviews"] and intent_result.intent in (IntentType.QA, IntentType.CHAT, IntentType.SUMMARY):
            yield {"type": "progress", "content": "你有待复习的知识点，先复习一下吧..."}
            for event in self.pipeline.review_agent.generate_stream(
                message=message,
                topic=intent_result.topic or message,
                sub_type="scheduled",
                user_id=user_id,
                doc_id=doc_id,
            ):
                yield event
            yield {"type": "loop_done", "pre_check": pre_check}
            return

        # ── Phase 1.5: Graph RAG query augmentation ──
        graph_kg_context = {}
        if self.graph_rag and intent_result.intent in (IntentType.QA, IntentType.EXPLAIN, IntentType.COMPARE):
            try:
                graph_kg_context = self.graph_rag.augment_query(message)
                if graph_kg_context.get("expanded_terms"):
                    yield {"type": "progress", "content": "正在扩展知识图谱关联..."}
            except Exception as e:
                logger.debug(f"Graph RAG failed: {e}")

        # ── Phase 2: Route to handler ──
        yield {"type": "progress", "content": "正在分析你的问题..."}
        full_answer = ""
        for event in self._route(message, intent_result, session_id, doc_id, user_id, mode):
            et = event.get("type")
            if et in ("token", "full", "answer"):
                full_answer = event.get("content", full_answer) or full_answer
            elif et == "done":
                sources = event.get("sources", [])
                metadata = event.get("metadata", {})
                yield event  # pass through
                break
            yield event

        # ── Phase 2.5: Self-RAG quality check on QA answers ──
        if self.self_rag and full_answer and intent_result.intent in (IntentType.QA, IntentType.EXPLAIN):
            try:
                quality = self.self_rag.check_quality(message, full_answer)
                if quality.get("needs_regenerate"):
                    yield {"type": "reflection_token", "content": "\n\n_[正在优化回答质量...]_\n\n"}
                    improve_hint = self.self_rag.build_quality_prompt_suffix(quality)
                    if improve_hint:
                        new_answer = self.pipeline.llm_func(
                            f"用户问题: {message}\n\n之前回答: {full_answer}\n{improve_hint}\n\n请根据改进方向重新回答："
                        )
                        full_answer = new_answer
            except Exception as e:
                logger.debug(f"Self-RAG failed: {e}")

        # ── Phase 3: Post-process ──
        self._mark_learner_event(message, user_id, session_id)
        if not full_answer:
            return
        yield {"type": "loop_answer", "content": full_answer}

    # ── Pre-check ──────────────────────────────────────────────────

    def _pre_check(self, user_id: str) -> Dict[str, Any]:
        weak_nodes = []
        due_reviews = []
        try:
            weak_nodes = self.pipeline.progress_tracker.get_weak_nodes(user_id, threshold=2)
        except Exception as e:
            logger.debug(f"Pre-check weak nodes failed: {e}")
        try:
            due_reviews = self.pipeline.progress_tracker.get_due_reviews(user_id, limit=5)
        except Exception as e:
            logger.debug(f"Pre-check due reviews failed: {e}")
        return {
            "has_weak_nodes": bool(weak_nodes),
            "weak_nodes": weak_nodes[:3],
            "has_due_reviews": bool(due_reviews),
            "due_reviews": due_reviews[:3],
        }

    # ── Phase 2: Route ─────────────────────────────────────────────

    def _route(
        self,
        message: str,
        intent_result,
        session_id: Optional[str],
        doc_id: Optional[str],
        user_id: str,
        mode: str,
    ) -> Generator[Dict[str, Any], None, None]:
        if intent_result.intent == IntentType.QUIZ:
            for event in self.pipeline.quiz_agent.generate_stream(
                message, topic=intent_result.topic or message,
                sub_type=intent_result.sub_type or "choice", doc_id=doc_id,
            ):
                yield event
            return

        if intent_result.intent == IntentType.SUMMARY:
            for event in self.pipeline.summary_agent.generate_stream(
                message, topic=intent_result.topic or message,
                sub_type=intent_result.sub_type or "topic", doc_id=doc_id,
            ):
                yield event
            return

        if intent_result.intent == IntentType.REVIEW:
            for event in self.pipeline.review_agent.generate_stream(
                message, topic=intent_result.topic or message,
                sub_type=intent_result.sub_type or "scheduled",
                user_id=user_id, doc_id=doc_id,
            ):
                yield event
            return

        if intent_result.intent == IntentType.TUTOR:
            yield {"type": "progress", "content": "正在生成学习建议..."}
            try:
                path = self.pipeline.tutor_agent.generate_learning_path(
                    goal=intent_result.topic or message, user_id=user_id,
                )
                if path:
                    answer = "## 学习路径建议\n\n"
                    for i, step in enumerate(path[:5]):
                        title = step.get("title", step.get("concept", f"步骤{i+1}"))
                        desc = step.get("description", step.get("definition", ""))
                        answer += f"### {i+1}. {title}\n{desc}\n\n"
                    yield {"type": "full", "content": answer, "sources": []}
                else:
                    answer = self.pipeline.llm_func(
                        f"用户希望获得关于「{intent_result.topic or message}」的学习建议。请给出结构化的学习计划。"
                    )
                    yield {"type": "full", "content": answer, "sources": []}
            except Exception as e:
                logger.warning(f"Tutor failed: {e}")
                yield {"type": "error", "content": "无法生成学习建议"}
            return

        # Default: delegate to orchestrator (handles QA, EXPLAIN, COMPARE, CHAT, etc.)
        for event in self.pipeline.orchestrator.process(
            message, session_id=session_id, doc_id=doc_id,
            user_id=user_id, mode=mode,
        ):
            yield event

    # ── Phase 3: Post-process ──────────────────────────────────────

    def _mark_learner_event(self, message: str, user_id: str, session_id: Optional[str]):
        try:
            self.pipeline.learner_model.emit(
                surface="chat",
                kind="query",
                data={"message": message[:200]},
                session_id=session_id or "",
            )
        except Exception as e:
            logger.debug(f"LearnerModel emit failed: {e}")

    def record_answer_feedback(
        self,
        knowledge_node_ids: List[str],
        quality: int,
        user_id: str = "default",
    ):
        """Call after user answers a quiz / review question."""
        if not knowledge_node_ids:
            return
        for node_id in knowledge_node_ids:
            try:
                record = self.pipeline.progress_tracker.record_review(
                    knowledge_node_id=node_id,
                    quality=quality,
                    user_id=user_id,
                )
                node_title = record.title or node_id
                self.pipeline.learner_model.emit(
                    surface="quiz",
                    kind="review",
                    data={
                        "knowledge_node_id": node_id,
                        "title": node_title,
                        "quality": quality,
                        "mastery": record.mastery.value if hasattr(record.mastery, 'value') else record.mastery,
                        "next_review_interval": record.review_interval_days,
                    },
                    session_id="",
                )
            except Exception as e:
                logger.debug(f"Record review failed for {node_id}: {e}")

    def record_exposure(
        self,
        knowledge_node_ids: List[str],
        titles: Optional[Dict[str, str]] = None,
        user_id: str = "default",
    ):
        """Call after RAG retrieval to mark nodes as exposed."""
        if not knowledge_node_ids:
            return
        try:
            self.pipeline.progress_tracker.batch_record_exposure(
                node_ids=knowledge_node_ids,
                titles=titles or {},
                user_id=user_id,
            )
        except Exception as e:
            logger.debug(f"Batch record exposure failed: {e}")

    def get_learner_state(self, user_id: str = "default") -> Dict[str, Any]:
        weak_nodes = []
        due_reviews = []
        progress = {}
        try:
            weak_nodes = self.pipeline.progress_tracker.get_weak_nodes(user_id, threshold=2)
        except Exception:
            pass
        try:
            due_reviews = self.pipeline.progress_tracker.get_due_reviews(user_id, limit=10)
        except Exception:
            pass
        try:
            progress = self.pipeline.progress_tracker.get_progress_summary(user_id)
        except Exception:
            pass
        return {
            "has_weak_nodes": bool(weak_nodes),
            "weak_nodes": weak_nodes[:5],
            "has_due_reviews": bool(due_reviews),
            "due_reviews": due_reviews[:5],
            "progress": progress,
        }


def topic_text(weak_nodes: List[Dict]) -> str:
    return "请针对薄弱知识点出题：" + "；".join(
        f"{n.get('title', n.get('knowledge_node_id', ''))}（答错{n.get('wrong_count', 0)}次）"
        for n in weak_nodes[:3]
    )
