import json
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Generator

from core.conversation_memory import LayeredMemory
from core.document_manager import DocumentManager
from core.background_agent import BackgroundAgent
from core.knowledge_distiller import KnowledgeDistiller
from core.knowledge_graph import Relation
from utils.retrieval_utils import merge_retrieval_results
from core.database import DatabaseManager, get_database, escape_like
from engines.retrieval_service import RetrievalService
from engines.knowledge_service import KnowledgeService
from engines.progress_service import ProgressService
from engines.agent_service import AgentService
from engines.hierarchical_retriever import ContextStrategy
from engines.intent_router import IntentType, IntentResult
from engines.learner_model import LearnerModel
from engines.resilience import ErrorCode, ERROR_USER_MESSAGES, build_degraded_answer
from engines.evaluator import EvalSample
from engines.guided_learning import GuidedLearningEngine
from engines.adaptive_retriever import QueryClassifier
from engines.learning_loop import LearningLoop
from engines.graph_rag import GraphRAGAugmenter
from engines.self_rag import SelfRAGCritic
from config import config

logger = logging.getLogger(__name__)


class PipelineContext:
    """Thin service holder. All logic lives in the four domain services."""

    def __init__(self, embed_func=None, llm_func=None, llm_service=None):
        self.embed_func = embed_func
        self.llm_func = llm_func
        self.llm_service = llm_service

        self.document_manager = DocumentManager()
        self.conversation_memory = LayeredMemory(llm_func=llm_func, embed_func=embed_func)

        self.retrieval = RetrievalService(embed_func=embed_func, llm_func=llm_func)
        self.knowledge = KnowledgeService(llm_func=llm_func, embed_func=embed_func)
        self.progress = ProgressService(llm_func=llm_func)
        self.agent = AgentService(llm_func=llm_func)
        self.guided_learning = None
        self.learner_model = LearnerModel(user_id="default", llm_func=llm_func)

    def wire_pipeline(self, pipeline):
        self.retrieval.wire(pipeline, self.knowledge.knowledge_graph)
        self.knowledge.wire(
            self.progress.progress_tracker,
            self.document_manager,
            self.conversation_memory,
        )
        self.progress.wire(
            self.retrieval.storage,
            self.document_manager,
            self.knowledge.knowledge_graph,
        )
        self.agent.wire(pipeline, self.progress, self.knowledge)
        self.agent.register_core_tools(pipeline)
        if self.knowledge.learning_context:
            self.knowledge.learning_context.analytics = self.progress.analytics
        self.graph_rag = GraphRAGAugmenter(
            knowledge_graph=self.knowledge.knowledge_graph,
            llm_func=self.llm_func,
            embed_func=self.embed_func,
        )
        self.self_rag = SelfRAGCritic(llm_func=self.llm_func)
        self.guided_learning = GuidedLearningEngine(
            knowledge_graph=self.knowledge.knowledge_graph,
            progress_tracker=self.progress.progress_tracker,
            storage=self.retrieval.storage,
            pipeline=pipeline,
            llm_func=self.llm_func,
            embed_func=self.embed_func,
        )
        self.learning_loop = LearningLoop(pipeline=pipeline)


class EnhancedRAGPipeline:
    """Facade that coordinates four domain services: retrieval, knowledge, progress, agent."""

    def __init__(
        self,
        embed_func: Optional[Callable] = None,
        llm_func: Optional[Callable] = None,
        llm_service: Optional[Any] = None,
    ):
        self.embed_func = embed_func
        self.llm_func = llm_func
        self.llm_service = llm_service

        self.ctx = PipelineContext(embed_func, llm_func, llm_service)

        # Set domain services before wire (register_core_tools accesses them)
        self.retrieval = self.ctx.retrieval
        self.knowledge = self.ctx.knowledge
        self.progress = self.ctx.progress
        self.agent = self.ctx.agent

        self.ctx.wire_pipeline(self)

        self.background_agent = BackgroundAgent(pipeline_getter=lambda: self)
        self._active_course_id: Optional[str] = None
        self._REFLECTION_MAX_RETRIES = 2

    def __getattr__(self, name: str):
        known_aliases = {
            "document_manager", "conversation_memory", "storage", "chunker",
            "hybrid_retriever", "reranker", "mmr_reranker", "evaluator", "tracer",
            "query_classifier", "query_rewriter", "query_expander", "retriever",
            "adaptive_retriever", "knowledge_graph", "learning_context", "course_manager",
            "progress_tracker", "learning_planner", "learning_reminder", "analytics",
            "intent_router", "orchestrator", "quiz_agent", "summary_agent", "review_agent",
            "tutor_agent", "health_checker", "event_bus", "tool_registry", "prompt_manager",
            "guided_learning", "learner_model", "learning_loop",
        }
        if name in known_aliases:
            if hasattr(self.ctx, name):
                return getattr(self.ctx, name)
            for svc in ("retrieval", "knowledge", "progress", "agent"):
                svc_obj = getattr(self, svc, None)
                if svc_obj and hasattr(svc_obj, name):
                    return getattr(svc_obj, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def render_prompt(self, name: str, **kwargs) -> str:
        return self.agent.render_prompt(name, **kwargs)


    def dispatch(
        self,
        message: str,
        session_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: str = "default",
        force_intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        full_answer = ""
        sources = []
        intent_val = "qa"
        confidence = 0.0
        topic = ""
        sub_type = ""
        metadata = {}
        degraded = False
        degradation_note = ""
        error_code = "ok"

        for event in self.dispatch_stream(
            message, session_id=session_id, doc_id=doc_id,
            user_id=user_id, force_intent=force_intent,
        ):
            et = event.get("type", "")
            if et == "intent":
                intent_val = event.get("intent", "qa")
                confidence = event.get("confidence", 0.0)
                topic = event.get("topic", "")
            elif et == "token":
                full_answer += event.get("content", "")
            elif et == "reflection_token":
                full_answer += event.get("content", "")
            elif et == "full":
                full_answer = event.get("content", "")
                sources = event.get("sources", [])
            elif et == "done":
                sources = event.get("sources", [])
                metadata = event.get("metadata", {})
            elif et == "degraded":
                full_answer = event.get("content", "")
                sources = event.get("sources", [])
                degradation_note = event.get("degradation_note", "")
                degraded = True
            elif et == "error":
                full_answer = event.get("content", "处理失败")
                error_code = event.get("error_code", "error")

        return {
            "answer": full_answer,
            "sources": sources,
            "agent_type": "orchestrator",
            "intent": intent_val,
            "intent_confidence": confidence,
            "topic": topic,
            "sub_type": sub_type,
            "metadata": metadata,
            "degraded": degraded,
            "degradation_note": degradation_note,
            "error_code": error_code,
        }

    def dispatch_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: str = "default",
        force_intent: Optional[str] = None,
        mode: str = "hybrid",
    ) -> Generator[Dict[str, Any], None, None]:
        if force_intent:
            intent_map = {
                "qa": IntentType.QA, "quiz": IntentType.QUIZ,
                "summary": IntentType.SUMMARY, "review": IntentType.REVIEW,
                "chat": IntentType.CHAT, "tutor": IntentType.TUTOR,
                "explain": IntentType.EXPLAIN, "compare": IntentType.COMPARE,
            }
            itype = intent_map.get(force_intent.lower())
            if itype:
                logger.info(f"Force intent applied: {force_intent} for query: {message[:50]}")
                intent_result = IntentResult(intent=itype, confidence=1.0, topic=message)
                yield {"type": "intent", "intent": itype.label, "confidence": 1.0, "topic": message}
                if itype == IntentType.QUIZ:
                    for e in self.quiz_agent.generate_stream(message, topic=message, sub_type="choice", doc_id=doc_id):
                        yield e
                elif itype == IntentType.SUMMARY:
                    for e in self.summary_agent.generate_stream(message, topic=message, sub_type="topic", doc_id=doc_id):
                        yield e
                elif itype == IntentType.REVIEW:
                    for e in self.review_agent.generate_stream(message, topic=message, sub_type="scheduled", user_id=user_id, doc_id=doc_id):
                        yield e
                else:
                    for event in self.orchestrator.process(
                        message, session_id=session_id, doc_id=doc_id,
                        user_id=user_id, mode=mode,
                    ):
                        yield event
                return
            else:
                logger.warning(f"Force intent '{force_intent}' not recognized, falling back to router for query: {message[:50]}")
                intent_result = self.intent_router.route(message)
        else:
            intent_result = self.intent_router.route(message)

        if session_id:
            session = self.conversation_memory.get_session(session_id)
            if not session:
                self.conversation_memory.create_session(
                    user_id=user_id,
                    title=f"Session {session_id[:12]}",
                )
            self.conversation_memory.add_message(
                session_id, "user", message,
                metadata={"source": "dispatch_stream"},
            )

        for event in self.orchestrator.process(
            message, session_id=session_id, doc_id=doc_id,
            user_id=user_id, mode=mode,
        ):
            yield event

    def process_with_loop(
        self,
        message: str,
        session_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: str = "default",
        mode: str = "hybrid",
    ) -> Generator[Dict[str, Any], None, None]:
        if session_id:
            session = self.conversation_memory.get_session(session_id)
            if not session:
                self.conversation_memory.create_session(
                    user_id=user_id,
                    title=f"Session {session_id[:12]}",
                )
            self.conversation_memory.add_message(
                session_id, "user", message,
                metadata={"source": "learning_loop"},
            )
        for event in self.learning_loop.process(
            message, session_id=session_id, doc_id=doc_id,
            user_id=user_id, mode=mode,
        ):
            yield event

    def _should_reflect(self, answer: str, question: str) -> Optional[int]:
        if not answer or len(answer.strip()) < 30:
            return 0
        low_quality_patterns = [
            "我无法", "无法回答", "不知道", "没有相关信息",
            "上下文中没有", "无法提供", "I cannot", "I don't know",
        ]
        for pattern in low_quality_patterns:
            if pattern in answer:
                return 0
        if answer.count("来源") == 0 and len(answer) > 100:
            return 1
        return None

    def _rewrite_for_reflection(self, question: str, previous_answer: str, attempt: int) -> str:
        if not self.llm_func:
            return question
        try:
            prompt = self.render_prompt("reflection_rewrite",
                question=question,
                previous_answer=previous_answer[:200],
            )
            rewritten = self.llm_func(prompt).strip()
            if rewritten and len(rewritten) > 5 and rewritten != question:
                return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting for reflection failed: {e}")
        return question

    def ingest(self, text: str, doc_id: str = "", title: str = "", tags: Optional[List[str]] = None) -> Dict[str, Any]:
        start = time.time()
        nodes = self.chunker.chunk(text, doc_id)
        chunk_time = time.time() - start

        start = time.time()
        self.storage.store_nodes(nodes, self.embed_func)
        store_time = time.time() - start

        self.hybrid_retriever.build_bm25_index(doc_id)

        l3_nodes = [n for n in nodes if n.level == 3]
        l2_nodes = [n for n in nodes if n.level == 2]
        l2_titles = {n.node_id: n.title for n in l2_nodes}
        self.progress_tracker.batch_record_exposure(
            [n.node_id for n in l2_nodes],
            titles=l2_titles,
        )

        start = time.time()
        l2_dicts = [{"node_id": n.node_id, "title": n.title, "content": n.content} for n in l2_nodes]
        kg_result = self.knowledge_graph.build_from_nodes(l2_dicts, doc_id)
        kg_time = time.time() - start

        start = time.time()
        l3_dicts = [{"node_id": n.node_id, "title": n.title, "content": n.content} for n in l3_nodes]
        distiller = KnowledgeDistiller(llm_func=self.llm_func)
        knowledge_units = distiller.distill(l3_dicts, doc_id)
        concept_dicts = [u.to_dict() for u in knowledge_units]
        if concept_dicts and self.embed_func:
            self.storage.store_concepts(concept_dicts, self.embed_func)
        for unit in knowledge_units:
            self.progress_tracker.record_exposure(
                knowledge_node_id=unit.concept,
                title=unit.concept,
                metadata={
                    "bloom_level": unit.bloom_level,
                    "difficulty": unit.difficulty,
                    "source_nodes": unit.source_node_ids,
                    "keywords": unit.keywords,
                    "is_distilled": True,
                },
            )
        for unit in knowledge_units:
            entity_properties = {
                "definition": unit.definition,
                "bloom_level": unit.bloom_level,
                "difficulty": str(unit.difficulty),
                "keywords": json.dumps(unit.keywords, ensure_ascii=False),
                "examples": json.dumps(unit.examples, ensure_ascii=False),
                "source_node_ids": json.dumps(unit.source_node_ids, ensure_ascii=False),
                "is_distilled": "true",
            }
            self.knowledge_graph._find_or_create_entity(
                unit.concept, "distilled_concept",
                unit.source_node_ids[0] if unit.source_node_ids else "",
                doc_id,
                properties=entity_properties,
            )
            if unit.prerequisites:
                for prereq_name in unit.prerequisites:
                    try:
                        self.knowledge_graph._find_or_create_entity(
                            prereq_name, "distilled_concept",
                            unit.source_node_ids[0] if unit.source_node_ids else "",
                            doc_id,
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
                    except Exception as e:
                        logger.warning(f"Failed to save prerequisite relation for {prereq_name} -> {unit.concept}: {e}")
                        continue

        distill_time = time.time() - start

        self.document_manager.register_document(
            doc_id=doc_id,
            title=title or doc_id,
            content=text,
            tags=tags,
        )
        self.document_manager.update_document_stats(
            doc_id,
            node_count=len(nodes),
            entity_count=kg_result.get("total_entities", 0),
            relation_count=kg_result.get("total_relations", 0),
        )

        level_counts = {}
        for node in nodes:
            level_counts[node.level] = level_counts.get(node.level, 0) + 1

        self.event_bus.publish(
            "document:ingested", "pipeline",
            payload={
                "doc_id": doc_id,
                "title": title,
                "node_count": len(nodes),
                "level_counts": level_counts,
                "distilled_units": len(knowledge_units),
                "kg_entities": kg_result.get("total_entities", 0),
            },
        )

        return {
            "doc_id": doc_id,
            "total_nodes": len(nodes),
            "level_counts": level_counts,
            "exposed_knowledge_nodes": len(l2_nodes),
            "distilled_knowledge_units": len(knowledge_units),
            "kg_entities": kg_result.get("total_entities", 0),
            "kg_relations": kg_result.get("total_relations", 0),
            "chunk_time_ms": round(chunk_time * 1000, 2),
            "store_time_ms": round(store_time * 1000, 2),
            "kg_build_time_ms": round(kg_time * 1000, 2),
            "distill_time_ms": round(distill_time * 1000, 2),
        }

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        use_hybrid: bool = True,
        use_reranker: bool = False,
        use_conversation_context: bool = True,
        use_rewriting: bool = False,
        strategy: Optional[str] = None,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_total = time.time()

        if mode:
            mode_map = {
                "naive": {"use_hybrid": False, "use_reranker": False, "use_rewriting": False},
                "hybrid": {"use_hybrid": True, "use_reranker": False, "use_rewriting": False},
                "deep": {"use_hybrid": True, "use_reranker": True, "use_rewriting": True},
            }
            resolved = mode_map.get(mode, {})
            use_hybrid = resolved.get("use_hybrid", use_hybrid)
            use_reranker = resolved.get("use_reranker", use_reranker)
            use_rewriting = resolved.get("use_rewriting", use_rewriting)

        if session_id and use_conversation_context:
            conv_context = self.conversation_memory.get_full_context(
                session_id, query=question, max_tokens=4000,
                include_relevant_history=True,
                include_user_preferences=True,
            )
        else:
            conv_context = []

        context_strategy = ContextStrategy(strategy) if strategy else None
        mode_label = f"[{mode}] " if mode else ""
        retrieval_strategy = f"{mode_label}{'hybrid' if use_hybrid else 'vector_only'}"

        start = time.time()

        if use_rewriting and config.QUERY_REWRITING_ENABLED and self.llm_func:
            analysis = self.query_classifier.classify(question)
            rewritten = self.query_rewriter.rewrite(
                question,
                strategies=analysis.rewrite_strategies,
                num_queries=config.QUERY_EXPANSION_NUM_QUERIES,
            )
            expanded = self.query_expander.expand(
                question,
                strategies=analysis.expand_strategies,
            )
            queries_to_search = list(dict.fromkeys(rewritten + expanded))
            if len(queries_to_search) > 1:
                retrieval_strategy += "+rewrite"

            per_query_top_k = max(top_k, config.RETRIEVAL_TOP_K)
            all_candidates = []
            for q in queries_to_search:
                results = self.hybrid_retriever.search(
                    q, top_k=per_query_top_k, doc_id=doc_id, use_hybrid=use_hybrid
                )
                all_candidates.append(results)

            candidates = merge_retrieval_results(all_candidates, top_k=per_query_top_k)
        else:
            candidates = self.hybrid_retriever.search(
                question, top_k=top_k * 3, doc_id=doc_id, use_hybrid=use_hybrid
            )
        retrieval_time = time.time() - start

        self.tracer.trace_retrieval(
            query=question,
            strategy=retrieval_strategy,
            results=candidates,
            latency_ms=retrieval_time * 1000,
        )

        start = time.time()
        if config.MMR_ENABLED and len(candidates) > config.RETRIEVAL_TOP_K:
            mmr_top_k = min(top_k * 2, config.MMR_TOP_K)
            candidates = self.mmr_reranker.rerank(question, candidates, top_k=mmr_top_k)
            retrieval_strategy += "+mmr"
        mmr_time = time.time() - start

        start = time.time()
        should_rerank = use_reranker and candidates
        if config.RERANKER_AUTO_DISABLE and should_rerank:
            top_scores = [c.get("rrf_score", c.get("score", 0)) for c in candidates[:5]]
            avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0
            if avg_top_score < config.RERANKER_MIN_CONFIDENCE:
                should_rerank = False

        if should_rerank:
            reranked = self.reranker.rerank(question, candidates, top_k=top_k)
            retrieval_strategy += "+reranker"
        else:
            reranked = candidates[:top_k]
        rerank_time = time.time() - start

        start = time.time()
        enriched = self.retriever.retrieve_with_context(reranked, context_strategy)
        context_time = time.time() - start

        contexts = [r["assembled_context"] for r in enriched]
        context_chain_info = [
            {
                "l1_title": r["context_chain"]["l1_title"],
                "l2_title": r["context_chain"]["l2_title"],
                "l3_title": r["context_chain"]["l3_title"],
                "l3_score": r["l3_score"],
                "auto_merged": r.get("auto_merged", False),
            }
            for r in enriched
        ]
        context_sources = [
            {
                "index": i + 1,
                "excerpt": r["assembled_context"][:300],
                "level_title": " / ".join(filter(None, [
                    r["context_chain"]["l1_title"],
                    r["context_chain"]["l2_title"],
                    r["context_chain"]["l3_title"],
                ])),
                "node_id": r.get("l3_node_id", ""),
            }
            for i, r in enumerate(enriched)
        ]

        concept_sources = []
        if candidates and self.embed_func:
            query_vec = self.embed_func([question])
            if query_vec and len(query_vec) > 0:
                concept_results = self.storage.search_concepts(query_vec[0], top_k=3, doc_id=doc_id)
                for cr in concept_results:
                    concept_sources.append({
                        "concept": cr.get("concept", ""),
                        "definition": cr.get("definition", ""),
                        "bloom_level": cr.get("bloom_level", ""),
                        "prerequisites": cr.get("prerequisites", []),
                        "difficulty": cr.get("difficulty", 0),
                        "score": cr.get("score", 0),
                    })

        for r in enriched:
            node_id = r.get("l3_node_id", "")
            if node_id:
                self.progress_tracker.record_exposure(node_id, r.get("l3_title", ""))

        if session_id:
            self.conversation_memory.add_message(session_id, "user", question)
            self.conversation_memory.add_message(
                session_id, "system",
                f"[检索到{len(contexts)}个上下文]",
                metadata={"retrieval_strategy": retrieval_strategy},
            )

        total_time = time.time() - start_total

        return {
            "question": question,
            "session_id": session_id or "",
            "user_id": "default",
            "contexts": contexts,
            "context_chains": context_chain_info,
            "context_sources": context_sources,
            "concept_sources": concept_sources,
            "num_contexts": len(contexts),
            "strategy": retrieval_strategy,
            "conversation_context": conv_context if session_id else [],
            "retrieval_time_ms": round(retrieval_time * 1000, 2),
            "rerank_time_ms": round(rerank_time * 1000, 2),
            "context_assembly_time_ms": round(context_time * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2),
        }

    @staticmethod
    def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        use_adaptive: bool = True,
    ) -> Dict[str, Any]:
        return self.dispatch(
            question, session_id=session_id, doc_id=doc_id, force_intent="qa"
        )

    def generate_answer(self, question: str, pipeline_result: Dict) -> str:
        if not self.llm_func:
            return "[未配置LLM生成函数]"

        prompt = self._build_answer_prompt(question, pipeline_result)

        return self.llm_func(prompt)

    def generate_answer_stream(self, question: str, pipeline_result: Dict):
        if not self.llm_service:
            for chunk in "[未配置LLM，无法流式生成]":
                yield chunk
            return

        prompt = self._build_answer_prompt(question, pipeline_result)

        for chunk in self.llm_service.invoke_stream(prompt):
            yield chunk

    def _build_answer_prompt(self, question: str, pipeline_result: Dict) -> str:
        contexts = pipeline_result["contexts"]
        numbered = [f"[来源 {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
        context_text = "\n\n---\n\n".join(numbered)

        concept_text = ""
        concept_sources = pipeline_result.get("concept_sources", [])
        if concept_sources:
            concept_parts = []
            for c in concept_sources[:3]:
                name = c.get("concept", "")
                definition = c.get("definition", "")
                bloom = c.get("bloom_level", "")
                prereqs = c.get("prerequisites", [])
                if name and definition:
                    prereq_str = f"，前置: {'、'.join(prereqs[:3])}" if prereqs else ""
                    concept_parts.append(f"- **{name}** (Bloom: {bloom}{prereq_str}): {definition[:200]}")
            if concept_parts:
                concept_text = "\n\n[相关概念定义]\n" + "\n".join(concept_parts)

        conv_context = pipeline_result.get("conversation_context", [])
        conv_text = ""
        if conv_context:
            conv_parts = []
            for msg in conv_context:
                role = msg.get("role", "system")
                content = msg.get("content", "")
                if not content.strip():
                    continue
                label = {
                    "system": "记忆",
                    "user": "用户",
                    "assistant": "助手",
                }.get(role, role)
                conv_parts.append(f"[{label}]: {content}")
            if conv_parts:
                conv_text = "\n对话上下文:\n" + "\n".join(conv_parts) + "\n\n"

        learning_context = ""
        if self.learning_context:
            session_id = pipeline_result.get("session_id", "")
            user_id = pipeline_result.get("user_id", "default")
            learning_context = self.learning_context.build_system_context(user_id, session_id)

        review_hint = ""
        try:
            user_id = pipeline_result.get("user_id", "default")
            due = self.progress_tracker.get_due_reviews(user_id, limit=3)
            if due:
                items = [f"- {r['title']}（掌握度: {r['mastery']}）" for r in due]
                review_hint = "\n\n[复习提醒] 用户以下知识点待复习，可在回答末尾自然提及：\n" + "\n".join(items)
        except Exception as e:
            logger.warning(f"Failed to get review hints: {e}")

        system_prompt = self.render_prompt("system", learning_context=learning_context + review_hint)
        return self.render_prompt("qa",
            system_prompt=system_prompt,
            conv_text=conv_text,
            context_text=context_text + concept_text,
            kg_text="",
            question=question,
        )

    def create_session(self, user_id: str = "default", title: str = "") -> Dict[str, Any]:
        session = self.conversation_memory.create_session(user_id, title)
        return {
            "session_id": session.session_id,
            "title": session.title,
            "created_at": session.created_at,
        }

    def create_learning_plan(
        self,
        doc_id: str,
        user_id: str = "default",
        title: str = "",
        daily_minutes: int = 60,
    ) -> Dict[str, Any]:
        l2_nodes = self.storage.get_nodes_by_level(2, doc_id)
        l2_dicts = [
            {
                "node_id": n.node_id,
                "title": n.title,
                "content": n.content,
            }
            for n in l2_nodes
        ]

        plan = self.learning_planner.create_plan_from_doc(
            doc_id=doc_id,
            l2_nodes=l2_dicts,
            user_id=user_id,
            title=title,
            daily_minutes=daily_minutes,
        )
        return plan.to_dict()

    def create_learning_plan_from_goal(
        self,
        goal: str,
        user_id: str = "default",
        daily_minutes: int = 60,
        total_days: int = 7,
    ) -> Dict[str, Any]:
        plan = self.learning_planner.create_plan_from_goal(
            goal=goal,
            user_id=user_id,
            daily_minutes=daily_minutes,
            total_days=total_days,
        )
        return plan.to_dict()

    def get_learning_progress(self, user_id: str = "default") -> Dict[str, Any]:
        return self.progress_tracker.get_progress_summary(user_id)

    def get_due_reviews(self, user_id: str = "default", limit: int = 10) -> List[Dict]:
        return self.progress_tracker.get_due_reviews(user_id, limit)

    def record_review(
        self,
        knowledge_node_id: str,
        quality: int,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        record = self.progress_tracker.record_review(knowledge_node_id, quality, user_id)
        return record.to_dict()

    def query_knowledge_graph(self, entity_name: str) -> Optional[Dict]:
        entity = self.knowledge_graph.query_entity(entity_name)
        if not entity:
            return None
        return self.knowledge_graph.get_entity_relations(entity["entity_id"], depth=1)

    def search_kg_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.knowledge_graph.search_entities(query, limit)

    def unified_query(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 10,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        kg_entities = self.knowledge_graph.search_entities(query, limit=limit)

        concept_results = []
        if self.embed_func:
            try:
                qvec = self.embed_func([query])
                if qvec and len(qvec) > 0:
                    concept_results = self.storage.search_concepts(qvec[0], top_k=limit, doc_id=doc_id)
            except Exception as e:
                logger.warning(f"Concept search failed for query '{query[:30]}': {e}")

        tracker_results = []
        try:
            escaped = escape_like(query)
            cursor = self.progress_tracker.db.execute(
                "SELECT knowledge_node_id, title, mastery FROM knowledge_records "
                "WHERE user_id = ? AND (title LIKE ? ESCAPE '\\' OR knowledge_node_id LIKE ? ESCAPE '\\') LIMIT ?",
                (user_id, f"%{escaped}%", f"%{escaped}%", limit),
            )
            tracker_results = [
                {"node_id": r[0], "name": r[1], "mastery": r[2], "source": "progress_tracker"}
                for r in cursor.fetchall()
            ]
        except Exception as e:
            logger.warning(f"Progress tracker search failed for query '{query[:30]}': {e}")

        return {
            "query": query,
            "kg_entities": kg_entities,
            "concepts": concept_results,
            "tracker_records": tracker_results,
        }

    def get_entity_relations(self, entity_name: str, depth: int = 1) -> Dict[str, Any]:
        entity = self.knowledge_graph.query_entity(entity_name)
        if not entity:
            return {"entity": None, "relations": [], "neighbors": []}
        return self.knowledge_graph.get_entity_relations(entity["entity_id"], depth)

    def multi_hop_query(self, source: str, target: str, max_hops: int = 3) -> List[Dict]:
        return self.knowledge_graph.multi_hop_query(source, target, max_hops)

    def classify_query(self, question: str) -> Dict[str, Any]:
        classifier = QueryClassifier()
        analysis = classifier.classify(question)
        return {
            "query_type": analysis.query_type.value,
            "confidence": analysis.confidence,
            "keywords": analysis.keywords,
            "entities": analysis.entities,
            "intent": analysis.intent,
            "suggested_strategy": analysis.suggested_strategy,
            "suggested_top_k": analysis.suggested_top_k,
        }

    def get_due_reminders(self, user_id: str = "default", limit: int = 10) -> List[Dict]:
        return self.learning_reminder.get_due_reminders(user_id, limit)

    def create_review_reminder(
        self,
        knowledge_node_id: str,
        title: str = "",
        user_id: str = "default",
    ) -> Dict[str, Any]:
        reminder = self.learning_reminder.create_review_reminder(
            knowledge_node_id, title, user_id
        )
        return reminder.to_dict()

    def auto_generate_reminders(self, user_id: str = "default") -> List[Dict[str, Any]]:
        reminders = self.learning_reminder.auto_generate_reminders(user_id)
        return [r.to_dict() for r in reminders]

    def get_learning_dashboard(self, user_id: str = "default") -> Dict[str, Any]:
        return self.analytics.get_learning_dashboard(user_id)

    def get_study_recommendations(self, user_id: str = "default") -> List[Dict[str, Any]]:
        return self.analytics.get_study_recommendations(user_id)

    def get_next_learning_steps(self, user_id: str = "default", limit: int = 5) -> List[Dict[str, Any]]:
        return self.tutor_agent.get_next_steps(user_id, limit)

    def generate_learning_path(self, goal: str, user_id: str = "default") -> List[Dict[str, Any]]:
        return self.tutor_agent.generate_learning_path(goal, user_id)

    def get_knowledge_graph_insights(self, user_id: str = "default") -> Dict[str, Any]:
        return self.analytics.get_knowledge_graph_insights(user_id)

    def list_documents(self, limit: int = 20, tag: Optional[str] = None) -> List[Dict]:
        return self.document_manager.list_documents(limit=limit, tag=tag)

    def get_document(self, doc_id: str) -> Optional[Dict]:
        doc = self.document_manager.get_document(doc_id)
        return doc.to_dict() if doc else None

    def delete_document(self, doc_id: str) -> bool:
        self.storage.delete_doc(doc_id)
        return self.document_manager.delete_document(doc_id)

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

    def list_courses(self) -> List[Dict[str, Any]]:
        return self.course_manager.list_courses()

    def create_course(self, name: str, description: str = "") -> Dict[str, Any]:
        course = self.course_manager.create_course(name, description)
        self._active_course_id = course.course_id
        return course.to_dict()

    def modify_ingest_for_course(self, text: str, doc_id: str = "", title: str = "",
                                  tags: Optional[List[str]] = None,
                                  course_id: Optional[str] = None) -> Dict[str, Any]:
        if course_id is None:
            course_id = self.active_course_id
        if tags is None:
            tags = []
        if f"course:{course_id}" not in tags:
            tags.append(f"course:{course_id}")
        result = self.ingest(text, doc_id=doc_id, title=title, tags=tags)
        course = self.course_manager.get_course(course_id)
        if course:
            all_docs = self.document_manager.list_documents(tag=f"course:{course_id}", limit=100)
            total_nodes = sum(d.get("node_count", 0) for d in all_docs)
            total_entities = sum(d.get("entity_count", 0) for d in all_docs)
            self.course_manager.update_course_stats(
                course_id, doc_count=len(all_docs), node_count=total_nodes, entity_count=total_entities,
            )
        return result

    def get_background_notifications(self, clear: bool = True) -> list:
        return self.background_agent.get_notifications(clear)

    def start_background_agent(self):
        self.background_agent.start()

    def eval_intent_router(self, test_cases: List[tuple]) -> Dict[str, Any]:
        correct = 0
        results = []
        for msg, expected_label in test_cases:
            result = self.intent_router.route(msg)
            is_correct = result.intent.label == expected_label
            if is_correct:
                correct += 1
            results.append({
                "message": msg[:50],
                "expected": expected_label,
                "got": result.intent.label,
                "confidence": result.confidence,
                "correct": is_correct,
            })
        total = len(test_cases)
        return {
            "accuracy": round(correct / total, 3) if total > 0 else 0,
            "correct": correct,
            "total": total,
            "results": results,
        }

    def eval_orchestrator_tool_selection(self, test_cases: List[tuple]) -> Dict[str, Any]:
        results = []
        for msg, expected_tool in test_cases:
            intent_result = self.intent_router.route(msg)
            prompt = f"你是一个AI助手的决策核心。\n\n可用工具:\n{self.tool_registry.build_tools_description()}\n\n用户消息: {msg}\n\n输出JSON: {{\"action\": \"工具名\", \"params\": {{}}}}或{{\"answer\": \"直接回答\"}}"
            try:
                response = self.llm_func(prompt) if self.llm_func else ""
                import json
                clean = response.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                parsed = json.loads(clean)
                selected = parsed.get("action", "direct_answer")
            except Exception:
                selected = "parse_failed"
            results.append({
                "message": msg[:50],
                "expected_tool": expected_tool,
                "selected": selected,
                "correct": selected == expected_tool,
            })
        correct = sum(1 for r in results if r["correct"])
        return {
            "accuracy": round(correct / len(test_cases), 3) if test_cases else 0,
            "correct": correct,
            "total": len(test_cases),
            "results": results,
        }

    def evaluate(
        self,
        samples: List[EvalSample],
        use_hybrid: bool = True,
        use_reranker: bool = False,
        use_rewriting: bool = False,
    ) -> Dict[str, Any]:
        results = []
        for sample in samples:
            pipeline_result = self.query(
                sample.question,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                use_rewriting=use_rewriting,
                strategy="balanced",
            )
            sample.contexts = pipeline_result["contexts"]
            if not sample.answer and self.llm_func:
                sample.answer = self.generate_answer(sample.question, pipeline_result)

            eval_result = self.evaluator.evaluate_single(sample)
            combined = {
                "question": sample.question,
                "faithfulness": eval_result.faithfulness,
                "answer_relevancy": eval_result.answer_relevancy,
                "context_precision": eval_result.context_precision,
                "context_recall": eval_result.context_recall,
                "num_contexts": eval_result.num_contexts,
                "total_time_ms": pipeline_result["total_time_ms"],
            }
            results.append(combined)

        if not results:
            return {}

        summary = {
            "strategy": f"hybrid={use_hybrid},reranker={use_reranker}",
            "num_samples": len(results),
            "avg_faithfulness": round(sum(r["faithfulness"] for r in results) / len(results), 4),
            "avg_answer_relevancy": round(sum(r["answer_relevancy"] for r in results) / len(results), 4),
            "avg_context_precision": round(sum(r["context_precision"] for r in results) / len(results), 4),
            "avg_context_recall": round(sum(r["context_recall"] for r in results) / len(results), 4),
            "avg_total_time_ms": round(sum(r["total_time_ms"] for r in results) / len(results), 2),
        }

        return {"summary": summary, "details": results}

    def compare_strategies(self, samples: List[EvalSample]) -> Dict[str, Any]:
        strategies = [
            {"use_hybrid": False, "use_reranker": False, "use_rewriting": False, "name": "向量检索"},
            {"use_hybrid": True, "use_reranker": False, "use_rewriting": False, "name": "混合检索"},
            {"use_hybrid": True, "use_reranker": True, "use_rewriting": False, "name": "混合检索+Reranker"},
            {"use_hybrid": True, "use_reranker": False, "use_rewriting": True, "name": "混合检索+改写"},
            {"use_hybrid": True, "use_reranker": True, "use_rewriting": True, "name": "混合检索+改写+Reranker"},
        ]

        comparison = {}
        for s in strategies:
            result = self.evaluate(
                samples,
                use_hybrid=s["use_hybrid"],
                use_reranker=s["use_reranker"],
                use_rewriting=s["use_rewriting"],
            )
            comparison[s["name"]] = result.get("summary", {})

        return comparison

    def get_stats(self) -> Dict[str, Any]:
        storage_stats = self.storage.get_stats()
        trace_summary = self.tracer.get_summary()
        conv_stats = self.conversation_memory.get_stats()
        progress = self.progress_tracker.get_progress_summary()
        kg_stats = self.knowledge_graph.get_graph_stats()
        doc_stats = self.document_manager.get_stats()
        reminder_stats = self.learning_reminder.get_reminder_stats()
        return {
            "storage": storage_stats,
            "retrieval_traces": trace_summary,
            "conversation": conv_stats,
            "learning_progress": progress,
            "knowledge_graph": kg_stats,
            "documents": doc_stats,
            "reminders": reminder_stats,
        }

    def close(self):
        self.storage.close()
        self.conversation_memory.close()
        self.learning_planner.close()
        self.progress_tracker.close()
        self.knowledge_graph.close()
        self.document_manager.close()
        self.learning_reminder.close()
