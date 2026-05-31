import time
import logging
from typing import List, Dict, Any, Optional, Callable

from core.hierarchical_chunker import HierarchicalChunker
from core.tree_storage import TreeStorage
from engines.hierarchical_retriever import HierarchicalRetriever, ContextStrategy
from engines.hybrid_retriever import HybridRetriever
from engines.reranker import CrossEncoderReranker
from engines.mmr_reranker import MMRReranker
from engines.evaluator import RAGASEvaluator, EvalSample, RetrievalTracer
from engines.adaptive_retriever import AdaptiveRetriever, QueryClassifier
from engines.query_rewriter import QueryRewriter
from engines.query_expander import QueryExpander
from config import config
from utils.retrieval_utils import merge_retrieval_results

logger = logging.getLogger(__name__)


class RetrievalService:
    """Encapsulates all document ingestion and retrieval concerns."""

    def __init__(self, embed_func=None, llm_func=None):
        self.embed_func = embed_func
        self.llm_func = llm_func

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
        self.mmr_reranker = MMRReranker(
            lambda_param=config.MMR_LAMBDA,
            top_k=config.MMR_TOP_K,
            embed_func=embed_func,
        )
        self.evaluator = RAGASEvaluator(llm_func=llm_func, embed_func=embed_func)
        self.tracer = RetrievalTracer()
        self.query_classifier = QueryClassifier()
        self.query_rewriter = QueryRewriter(llm_func=llm_func)
        self.query_expander = None  # set later with knowledge_graph dependency
        self.adaptive_retriever = AdaptiveRetriever(
            pipeline=None, knowledge_graph=None, llm_func=llm_func,
        )

    def wire(self, pipeline, knowledge_graph):
        self.adaptive_retriever.pipeline = pipeline
        self.adaptive_retriever.knowledge_graph = knowledge_graph
        self.query_expander = QueryExpander(knowledge_graph=knowledge_graph)

    def ingest_document(self, text: str, doc_id: str = "", title: str = "") -> Dict[str, Any]:
        start = time.time()
        nodes = self.chunker.chunk(text, doc_id)
        chunk_time = time.time() - start

        start = time.time()
        self.storage.store_nodes(nodes, self.embed_func)
        store_time = time.time() - start

        self.hybrid_retriever.build_bm25_index(doc_id)

        level_counts = {}
        for node in nodes:
            level_counts[node.level] = level_counts.get(node.level, 0) + 1

        l2_nodes = [n for n in nodes if n.level == 2]
        l3_nodes = [n for n in nodes if n.level == 3]

        return {
            "doc_id": doc_id,
            "total_nodes": len(nodes),
            "level_counts": level_counts,
            "l2_nodes": l2_nodes,
            "l3_nodes": l3_nodes,
            "chunk_time_ms": round(chunk_time * 1000, 2),
            "store_time_ms": round(store_time * 1000, 2),
        }

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        use_hybrid: bool = True,
        use_reranker: bool = False,
        use_rewriting: bool = False,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        mode: Optional[str] = None,
        use_conversation_context: bool = True,
        conversation_memory=None,
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

        conv_context = []
        if session_id and use_conversation_context and conversation_memory:
            conv_context = conversation_memory.get_full_context(
                session_id, query=question, max_tokens=4000,
                include_relevant_history=True,
                include_user_preferences=True,
            )

        retrieval_strategy = f"[{mode or 'default'}] {'hybrid' if use_hybrid else 'vector_only'}"

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
            ) if self.query_expander else []
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
            query=question, strategy=retrieval_strategy,
            results=candidates, latency_ms=retrieval_time * 1000,
        )

        start = time.time()
        if config.MMR_ENABLED and len(candidates) > config.RETRIEVAL_TOP_K:
            mmr_top_k = min(top_k * 2, config.MMR_TOP_K)
            candidates = self.mmr_reranker.rerank(question, candidates, top_k=mmr_top_k)
            retrieval_strategy += "+mmr"
        _ = time.time() - start

        start = time.time()
        should_rerank = use_reranker and candidates
        if config.RERANKER_AUTO_DISABLE and should_rerank:
            top_scores = [c.get("rrf_score", c.get("score", 0)) for c in candidates[:5]]
            avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0
            if avg_top_score < config.RERANKER_MIN_CONFIDENCE:
                should_rerank = False

        if should_rerank:
            candidates = self.reranker.rerank(question, candidates, top_k=top_k)
            retrieval_strategy += "+reranker"
        else:
            candidates = candidates[:top_k]
        rerank_time = time.time() - start

        start = time.time()
        enriched = self.retriever.retrieve_with_context(candidates)
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

        total_time = time.time() - start_total
        return {
            "question": question,
            "session_id": session_id or "",
            "user_id": "default",
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
    def get_stats(self) -> Dict[str, Any]:
        return {
            "storage": self.storage.get_stats(),
            "retrieval_traces": self.tracer.get_summary(),
        }

    def close(self):
        self.storage.close()
