import time
from typing import List, Dict, Any, Optional, Callable, Generator

from core.hierarchical_chunker import HierarchicalChunker, ChunkNode
from core.tree_storage import TreeStorage
from core.conversation_memory import ConversationMemory
from core.learning_planner import LearningPlanner, PlanStatus
from core.progress_tracker import ProgressTracker
from core.knowledge_graph import KnowledgeGraphBuilder
from core.document_manager import DocumentManager
from core.learning_reminder import LearningReminder
from engines.hierarchical_retriever import HierarchicalRetriever, ContextStrategy
from engines.hybrid_retriever import HybridRetriever
from engines.reranker import CrossEncoderReranker
from engines.evaluator import RAGASEvaluator, EvalSample, RetrievalTracer
from engines.adaptive_retriever import AdaptiveRetriever, QueryClassifier
from engines.query_rewriter import QueryRewriter
from engines.query_expander import QueryExpander
from engines.mmr_reranker import MMRReranker
from engines.qa_engine import QAEngine
from engines.learning_analytics import LearningAnalytics
from config import config


class EnhancedRAGPipeline:
    def __init__(
        self,
        embed_func: Optional[Callable] = None,
        llm_func: Optional[Callable] = None,
        llm_service: Optional[Any] = None,
    ):
        self.embed_func = embed_func
        self.llm_func = llm_func
        self.llm_service = llm_service

        self.chunker = HierarchicalChunker(
            l1_max_size=config.CHUNK_L1_MAX_SIZE,
            l2_max_size=config.CHUNK_L2_MAX_SIZE,
            l3_max_size=config.CHUNK_L3_MAX_SIZE,
            l3_min_size=config.CHUNK_L3_MIN_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )

        self.storage = TreeStorage()
        self.retriever = HierarchicalRetriever(
            tree_storage=self.storage,
            strategy=ContextStrategy(config.CONTEXT_STRATEGY),
            auto_merge_threshold=config.AUTO_MERGE_THRESHOLD,
        )
        self.hybrid_retriever = HybridRetriever(
            tree_storage=self.storage,
            embed_func=embed_func,
            bm25_top_k=config.BM25_TOP_K,
            vector_top_k=config.VECTOR_TOP_K,
            rrf_k=config.RRF_K,
            final_top_k=config.RETRIEVAL_TOP_K,
        )
        self.reranker = CrossEncoderReranker(
            model_name=config.RERANKER_MODEL,
            top_k=config.RERANKER_TOP_K,
        )
        self.evaluator = RAGASEvaluator(
            llm_func=llm_func,
            embed_func=embed_func,
        )
        self.tracer = RetrievalTracer()

        self.conversation_memory = ConversationMemory(llm_func=llm_func)
        self.learning_planner = LearningPlanner(llm_func=llm_func)
        self.progress_tracker = ProgressTracker()

        self.knowledge_graph = KnowledgeGraphBuilder(llm_func=llm_func)

        self.document_manager = DocumentManager()

        self.learning_reminder = LearningReminder(
            progress_tracker=self.progress_tracker,
            learning_planner=self.learning_planner,
            llm_func=llm_func,
        )

        self.query_classifier = QueryClassifier()
        self.query_rewriter = QueryRewriter(llm_func=llm_func)
        self.query_expander = QueryExpander(knowledge_graph=self.knowledge_graph)
        self.mmr_reranker = MMRReranker(
            lambda_param=config.MMR_LAMBDA,
            top_k=config.MMR_TOP_K,
            embed_func=embed_func,
        )

        self.adaptive_retriever = AdaptiveRetriever(
            pipeline=self,
            knowledge_graph=self.knowledge_graph,
            llm_func=llm_func,
        )

        self.qa_engine = QAEngine(
            pipeline=self,
            adaptive_retriever=self.adaptive_retriever,
            knowledge_graph=self.knowledge_graph,
            llm_func=llm_func,
        )

        self.analytics = LearningAnalytics(
            progress_tracker=self.progress_tracker,
            learning_planner=self.learning_planner,
            knowledge_graph=self.knowledge_graph,
            tree_storage=self.storage,
            document_manager=self.document_manager,
        )

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

        return {
            "doc_id": doc_id,
            "total_nodes": len(nodes),
            "level_counts": level_counts,
            "exposed_knowledge_nodes": len(l3_nodes),
            "kg_entities": kg_result.get("total_entities", 0),
            "kg_relations": kg_result.get("total_relations", 0),
            "chunk_time_ms": round(chunk_time * 1000, 2),
            "store_time_ms": round(store_time * 1000, 2),
            "kg_build_time_ms": round(kg_time * 1000, 2),
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
            conv_context = self.conversation_memory.get_context_window(session_id)
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

            candidates = self._merge_retrieval_results(all_candidates, top_k=per_query_top_k)
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
            }
            for i, r in enumerate(enriched)
        ]

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
            "contexts": contexts,
            "context_chains": context_chain_info,
            "context_sources": context_sources,
            "num_contexts": len(contexts),
            "strategy": retrieval_strategy,
            "conversation_context": conv_context if session_id else [],
            "retrieval_time_ms": round(retrieval_time * 1000, 2),
            "rerank_time_ms": round(rerank_time * 1000, 2),
            "context_assembly_time_ms": round(context_time * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2),
        }

    @staticmethod
    def _merge_retrieval_results(
        all_results: List[List[Dict]], top_k: int
    ) -> List[Dict]:
        seen = {}
        for results in all_results:
            for r in results:
                node_id = r["node_id"]
                score_key = "rrf_score" if "rrf_score" in r else "score"
                score = r.get(score_key, 0)
                if node_id not in seen or score > seen[node_id].get(score_key, 0):
                    if node_id not in seen:
                        seen[node_id] = dict(r)
                    else:
                        seen[node_id].update(r)
                        seen[node_id][score_key] = score

        sorted_results = sorted(
            seen.values(),
            key=lambda x: x.get("rrf_score", x.get("score", 0)),
            reverse=True,
        )
        return sorted_results[:top_k]

    def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        use_adaptive: bool = True,
    ) -> Dict[str, Any]:
        result = self.qa_engine.ask(
            question, session_id=session_id, doc_id=doc_id, use_adaptive=use_adaptive
        )
        return result.to_dict()

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

        conv_context = pipeline_result.get("conversation_context", [])
        conv_text = ""
        if conv_context:
            conv_parts = []
            for msg in conv_context[-6:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                conv_parts.append(f"[{role}]: {content[:200]}")
            conv_text = "\n对话历史:\n" + "\n".join(conv_parts) + "\n\n"

        prompt = f"""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明。

引用来源规则：
- 回答中引用某个来源时，请在句末标注 [来源 X]
- 如果一句话综合了多个来源，标注 [来源 1][来源 2]
- 每个来源至少引用一次

{conv_text}上下文:
{context_text}

问题: {question}

请给出详细、准确的回答:"""

        return prompt

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
