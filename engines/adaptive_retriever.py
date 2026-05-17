import re
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass


class QueryType(str, Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    EXPLORATORY = "exploratory"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"


@dataclass
class QueryAnalysis:
    query_type: QueryType = QueryType.FACTUAL
    confidence: float = 0.5
    keywords: List[str] = None
    entities: List[str] = None
    intent: str = ""
    suggested_strategy: str = "hybrid"
    suggested_top_k: int = 5
    use_reranker: bool = True
    context_strategy: str = "balanced"
    rewrite_strategies: List[str] = None
    expand_strategies: List[str] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.entities is None:
            self.entities = []
        if self.rewrite_strategies is None:
            self.rewrite_strategies = ["expand"]
        if self.expand_strategies is None:
            self.expand_strategies = ["abbr"]


class QueryClassifier:
    FACTUAL_PATTERNS = [
        r'(?:什么|是什么|定义|含义|概念|指的是|称为|叫做)',
        r'(?:what\s+is|define|definition|meaning\s+of)',
        r'(?:谁|何时|哪里|多少|哪个)',
        r'(?:who|when|where|how\s+many|which)',
    ]

    REASONING_PATTERNS = [
        r'(?:为什么|为什么|原因|原理|机制|如何理解|怎么解释)',
        r'(?:why|reason|principle|mechanism|how\s+does|explain\s+why)',
        r'(?:区别|差异|不同|联系|关系|关联)',
        r'(?:difference|relationship|connection|between)',
    ]

    EXPLORATORY_PATTERNS = [
        r'(?:如何|怎么|怎样|方法|步骤|方案|做法)',
        r'(?:how\s+to|approach|method|solution|way\s+to)',
        r'(?:推荐|建议|最佳|最好|优化|改进)',
        r'(?:recommend|suggest|best|optimize|improve)',
    ]

    COMPARISON_PATTERNS = [
        r'(?:比较|对比|vs|versus|区别|差异|优劣|利弊)',
        r'(?:compare|vs|versus|difference|pros\s+and\s+cons|advantage)',
        r'(?:A还是B|哪个更好|选择)',
    ]

    PROCEDURAL_PATTERNS = [
        r'(?:步骤|流程|过程|教程|指南|手册)',
        r'(?:step|procedure|process|tutorial|guide|manual)',
        r'(?:第一.*第二|首先.*然后|1\..*2\.)',
    ]

    def classify(self, query: str) -> QueryAnalysis:
        scores = {
            QueryType.FACTUAL: self._score_patterns(query, self.FACTUAL_PATTERNS),
            QueryType.REASONING: self._score_patterns(query, self.REASONING_PATTERNS),
            QueryType.EXPLORATORY: self._score_patterns(query, self.EXPLORATORY_PATTERNS),
            QueryType.COMPARISON: self._score_patterns(query, self.COMPARISON_PATTERNS),
            QueryType.PROCEDURAL: self._score_patterns(query, self.PROCEDURAL_PATTERNS),
        }

        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        if confidence < 0.3:
            best_type = QueryType.FACTUAL
            confidence = 0.3

        keywords = self._extract_keywords(query)
        entities = self._extract_entities(query)

        strategy_config = self._get_strategy_config(best_type)
        rewrite_strategies, expand_strategies = self.decide_rewrite_strategy(best_type)

        return QueryAnalysis(
            query_type=best_type,
            confidence=round(confidence, 2),
            keywords=keywords,
            entities=entities,
            intent=self._infer_intent(query, best_type),
            suggested_strategy=strategy_config["strategy"],
            suggested_top_k=strategy_config["top_k"],
            use_reranker=strategy_config["use_reranker"],
            context_strategy=strategy_config["context_strategy"],
            rewrite_strategies=rewrite_strategies,
            expand_strategies=expand_strategies,
        )

    def _score_patterns(self, query: str, patterns: List[str]) -> float:
        score = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            score += len(matches) * 0.4
        return min(score, 1.0)

    def _extract_keywords(self, query: str) -> List[str]:
        stop_words = {
            "的", "了", "是", "在", "有", "和", "与", "或", "不", "也",
            "都", "这", "那", "个", "一", "我", "你", "他", "她", "它",
            "什么", "怎么", "如何", "为什么", "哪里", "哪个", "谁",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "what", "how", "why", "where", "when", "who", "which",
        }
        tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', query)
        return [t for t in tokens if t not in stop_words and len(t) > 1]

    def _extract_entities(self, query: str) -> List[str]:
        patterns = [
            r'[""「」]([^""「」]+)[""「」]',
            r'《([^》]+)》',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        return entities

    def _infer_intent(self, query: str, query_type: QueryType) -> str:
        intent_map = {
            QueryType.FACTUAL: "查找具体事实或定义",
            QueryType.REASONING: "理解原理或因果关系",
            QueryType.EXPLORATORY: "探索方法或解决方案",
            QueryType.COMPARISON: "比较不同选项的差异",
            QueryType.PROCEDURAL: "获取操作步骤或流程",
        }
        return intent_map.get(query_type, "通用查询")

    def _get_strategy_config(self, query_type: QueryType) -> Dict[str, Any]:
        configs = {
            QueryType.FACTUAL: {
                "strategy": "hybrid",
                "top_k": 3,
                "use_reranker": True,
                "context_strategy": "conservative",
            },
            QueryType.REASONING: {
                "strategy": "hybrid",
                "top_k": 7,
                "use_reranker": True,
                "context_strategy": "full",
            },
            QueryType.EXPLORATORY: {
                "strategy": "hybrid",
                "top_k": 5,
                "use_reranker": True,
                "context_strategy": "balanced",
            },
            QueryType.COMPARISON: {
                "strategy": "hybrid",
                "top_k": 8,
                "use_reranker": True,
                "context_strategy": "full",
            },
            QueryType.PROCEDURAL: {
                "strategy": "hybrid",
                "top_k": 5,
                "use_reranker": True,
                "context_strategy": "balanced",
            },
        }
        return configs.get(query_type, configs[QueryType.FACTUAL])

    def decide_rewrite_strategy(self, query_type: QueryType):
        strategy_map = {
            QueryType.FACTUAL: (["expand"], ["abbr"]),
            QueryType.REASONING: (["hyde", "expand"], ["abbr"]),
            QueryType.EXPLORATORY: (["expand", "hyde"], ["abbr"]),
            QueryType.COMPARISON: (["expand"], ["abbr"]),
            QueryType.PROCEDURAL: (["expand"], ["abbr"]),
        }
        return strategy_map.get(query_type, (["expand"], ["abbr"]))


class AdaptiveRetriever:
    def __init__(
        self,
        pipeline=None,
        knowledge_graph=None,
        llm_func: Optional[Callable] = None,
    ):
        self.pipeline = pipeline
        self.knowledge_graph = knowledge_graph
        self.llm_func = llm_func
        self.classifier = QueryClassifier()

    def adaptive_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        analysis = self.classifier.classify(question)

        kg_context = self._get_kg_context(analysis)

        pipeline_result = self.pipeline.query(
            question,
            session_id=session_id,
            use_hybrid=(analysis.suggested_strategy == "hybrid"),
            use_reranker=analysis.use_reranker,
            strategy=analysis.context_strategy,
            top_k=analysis.suggested_top_k,
            doc_id=doc_id,
        )

        if kg_context:
            pipeline_result["kg_context"] = kg_context
            pipeline_result["contexts"] = kg_context + pipeline_result["contexts"]

        pipeline_result["query_analysis"] = {
            "query_type": analysis.query_type.value,
            "confidence": analysis.confidence,
            "keywords": analysis.keywords,
            "entities": analysis.entities,
            "intent": analysis.intent,
            "suggested_strategy": analysis.suggested_strategy,
            "suggested_top_k": analysis.suggested_top_k,
        }

        return pipeline_result

    def _get_kg_context(self, analysis: QueryAnalysis) -> List[str]:
        if not self.knowledge_graph:
            return []

        kg_contexts = []

        for entity_name in analysis.entities:
            related = self.knowledge_graph.get_related_entities(entity_name, limit=3)
            if related:
                parts = [f"[知识图谱: {entity_name}]"]
                for r in related:
                    parts.append(
                        f"  - {r['name']}({r['entity_type']}): {r['relation_type']} -> {r.get('description', '')}"
                    )
                kg_contexts.append("\n".join(parts))

        for keyword in analysis.keywords[:2]:
            entity = self.knowledge_graph.query_entity(keyword)
            if entity:
                entity_relations = self.knowledge_graph.get_entity_relations(entity["entity_id"], depth=1)
                if entity_relations["relations"]:
                    parts = [f"[知识图谱: {keyword}]"]
                    for rel in entity_relations["relations"][:3]:
                        parts.append(f"  - {rel['relation_type']}: {rel['description']}")
                    kg_contexts.append("\n".join(parts))

        return kg_contexts
