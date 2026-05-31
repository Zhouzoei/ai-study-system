import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class AnswerResult:
    question: str = ""
    answer: str = ""
    query_type: str = "factual"
    confidence: float = 0.0
    sources: List[Dict[str, Any]] = None
    kg_facts: List[str] = None
    follow_up_questions: List[str] = None
    total_time_ms: float = 0.0

    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.kg_facts is None:
            self.kg_facts = []
        if self.follow_up_questions is None:
            self.follow_up_questions = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "query_type": self.query_type,
            "confidence": round(self.confidence, 3),
            "sources": self.sources,
            "kg_facts": self.kg_facts,
            "follow_up_questions": self.follow_up_questions,
            "total_time_ms": round(self.total_time_ms, 2),
        }


class QAEngine:
    def __init__(
        self,
        pipeline=None,
        adaptive_retriever=None,
        knowledge_graph=None,
        llm_func: Optional[Callable] = None,
        learning_context_builder=None,
    ):
        self.pipeline = pipeline
        self.adaptive_retriever = adaptive_retriever
        self.knowledge_graph = knowledge_graph
        self.llm_func = llm_func
        self.learning_context_builder = learning_context_builder

    def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        use_adaptive: bool = True,
    ) -> AnswerResult:
        start = time.time()

        if use_adaptive and self.adaptive_retriever:
            result = self.adaptive_retriever.adaptive_query(
                question, session_id=session_id, doc_id=doc_id
            )
        elif self.pipeline:
            result = self.pipeline.query(
                question, session_id=session_id, doc_id=doc_id
            )
        else:
            return AnswerResult(question=question, answer="[未配置检索引擎]")

        contexts = result.get("contexts", [])
        kg_context = result.get("kg_context", [])
        query_analysis = result.get("query_analysis", {})
        context_chains = result.get("context_chains", [])

        answer = self._generate_answer(
            question, contexts, kg_context, query_analysis, session_id
        )

        confidence = self._estimate_confidence(answer, contexts, query_analysis)

        sources = self._extract_sources(context_chains)

        kg_facts = self._extract_kg_facts(kg_context)

        follow_ups = self._generate_follow_ups(question, answer, query_analysis)

        if session_id and self.pipeline:
            self.pipeline.conversation_memory.add_message(
                session_id, "assistant", answer[:500],
                metadata={"query_type": query_analysis.get("query_type", "unknown")},
            )

        total_time = (time.time() - start) * 1000

        return AnswerResult(
            question=question,
            answer=answer,
            query_type=query_analysis.get("query_type", "factual"),
            confidence=confidence,
            sources=sources,
            kg_facts=kg_facts,
            follow_up_questions=follow_ups,
            total_time_ms=total_time,
        )

    def _generate_answer(
        self,
        question: str,
        contexts: List[str],
        kg_context: List[str],
        query_analysis: Dict,
        session_id: Optional[str],
    ) -> str:
        if not self.llm_func:
            if contexts:
                return contexts[0][:500]
            return "[未配置LLM，无法生成回答]"

        context_blocks = []
        for i, ctx in enumerate(contexts[:5]):
            context_blocks.append(f"[参考 {i+1}]:\n{ctx}")
        context_text = "\n\n---\n\n".join(context_blocks)

        kg_text = ""
        if kg_context:
            kg_text = "\n\n[知识图谱信息]:\n" + "\n".join(kg_context)

        conv_text = ""
        if session_id and self.pipeline:
            conv_context = self.pipeline.conversation_memory.get_full_context(
                session_id, query=question, max_tokens=3000,
                include_relevant_history=False,
                include_user_preferences=False,
            )
            if conv_context:
                conv_parts = []
                for msg in conv_context[-6:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    conv_parts.append(f"[{role}]: {content[:150]}")
                conv_text = "\n[对话历史]:\n" + "\n".join(conv_parts) + "\n"

        query_type = query_analysis.get("query_type", "factual")

        learning_context = ""
        if self.learning_context_builder and self.pipeline:
            session = self.pipeline.conversation_memory.get_session(session_id) if session_id else None
            uid = session.user_id if session else "default"
            learning_context = self.learning_context_builder.build_system_context(uid, session_id)

        if self.pipeline and hasattr(self.pipeline, 'render_prompt'):
            prompt = self.pipeline.render_prompt("qa",
                system_prompt=self.pipeline.render_prompt("system", learning_context=learning_context),
                conv_text=conv_text,
                context_text=context_text,
                kg_text=kg_text,
                question=question,
            )
        else:
            prompt = f"""{conv_text}[参考资料]:
{context_text}
{kg_text}

问题: {question}"""

        try:
            return self.llm_func(prompt)
        except Exception as e:
            return f"[回答生成失败: {e}]"

    def _estimate_confidence(
        self,
        answer: str,
        contexts: List[str],
        query_analysis: Dict,
    ) -> float:
        if not answer or answer.startswith("["):
            return 0.0

        confidence = 0.5

        if contexts:
            confidence += min(len(contexts) * 0.1, 0.3)

        if any(kw in answer for kw in ["根据", "基于", "资料表明", "根据上下文"]):
            confidence += 0.1

        if any(kw in answer for kw in ["不确定", "可能", "似乎", "也许", "大概"]):
            confidence -= 0.1

        if any(kw in answer for kw in ["没有相关信息", "无法回答", "未找到"]):
            confidence -= 0.3

        confidence += query_analysis.get("confidence", 0) * 0.1

        return max(0.0, min(1.0, confidence))

    def _extract_sources(self, context_chains: List[Dict]) -> List[Dict[str, Any]]:
        sources = []
        seen = set()
        for chain in context_chains:
            key = f"{chain.get('l1_title', '')}_{chain.get('l2_title', '')}_{chain.get('l3_title', '')}"
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "chapter": chain.get("l1_title", ""),
                "section": chain.get("l2_title", ""),
                "paragraph": chain.get("l3_title", ""),
                "score": chain.get("l3_score", 0),
            })
        return sources[:5]

    def _extract_kg_facts(self, kg_context: List[str]) -> List[str]:
        facts = []
        for ctx in kg_context:
            lines = ctx.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("- "):
                    facts.append(line[2:])
        return facts[:5]

    def _generate_follow_ups(
        self,
        question: str,
        answer: str,
        query_analysis: Dict,
    ) -> List[str]:
        if not self.llm_func:
            return self._generate_rule_based_followups(question, query_analysis)

        prompt = f"""基于以下问题和回答，生成2-3个相关的后续问题，帮助用户深入学习。

问题: {question}
回答: {answer[:500]}

请直接输出后续问题，每行一个，不要编号和其他内容:"""

        try:
            response = self.llm_func(prompt)
            follow_ups = [
                line.strip().lstrip("0123456789.-) ")
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            return follow_ups[:3]
        except Exception:
            return self._generate_rule_based_followups(question, query_analysis)

    def _generate_rule_based_followups(
        self,
        question: str,
        query_analysis: Dict,
    ) -> List[str]:
        keywords = query_analysis.get("keywords", [])
        query_type = query_analysis.get("query_type", "factual")

        follow_ups = []

        if keywords:
            kw = keywords[0]
            if query_type == "factual":
                follow_ups.append(f"{kw}的应用场景有哪些？")
                follow_ups.append(f"{kw}和其他相关概念有什么区别？")
            elif query_type == "reasoning":
                follow_ups.append(f"{kw}的实际案例有哪些？")
                follow_ups.append(f"如何在实际中应用{kw}的原理？")
            elif query_type == "exploratory":
                follow_ups.append(f"还有其他实现{kw}的方法吗？")
                follow_ups.append(f"{kw}的最佳实践是什么？")
            else:
                follow_ups.append(f"能详细解释一下{kw}吗？")

        if not follow_ups:
            follow_ups.append("能否提供更多相关的详细信息？")

        return follow_ups[:3]
