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
    ):
        self.pipeline = pipeline
        self.adaptive_retriever = adaptive_retriever
        self.knowledge_graph = knowledge_graph
        self.llm_func = llm_func

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
            conv_context = self.pipeline.conversation_memory.get_context_window(
                session_id, window_size=4
            )
            if conv_context:
                conv_parts = []
                for msg in conv_context[-4:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    conv_parts.append(f"[{role}]: {content[:150]}")
                conv_text = "\n[对话历史]:\n" + "\n".join(conv_parts) + "\n"

        query_type = query_analysis.get("query_type", "factual")
        answer_guidance = self._get_answer_guidance(query_type)

        prompt = f"""{answer_guidance}
{conv_text}[参考资料]:
{context_text}
{kg_text}

问题: {question}

请给出详细、准确的回答:"""

        try:
            return self.llm_func(prompt)
        except Exception as e:
            return f"[回答生成失败: {e}]"

    def _get_answer_guidance(self, query_type: str) -> str:
        prefix = (
            "你是一个专业的深度学习知识助手。请严格基于参考资料回答，不要添加参考资料中没有的信息。"
            "如果参考资料不足以完整回答问题，请明确指出哪些部分没有找到相关信息。"
            "在回答中引用具体的章节标题作为来源参考。\n\n"
        )
        guidance_map = {
            "factual": (
                f"{prefix}"
                "请基于参考资料给出精确的定义或描述。\n"
                "- 如果资料中有明确定义，直接引用并解释\n"
                "- 如果资料中没有相关信息，明确回答'参考资料中未找到相关说明'\n"
                "- 使用编号列表或段落形式组织内容"
            ),
            "reasoning": (
                f"{prefix}"
                "请基于参考资料深入分析问题和原理。\n"
                "- 使用'因为...所以...'的逻辑结构解释因果关系\n"
                "- 如果有多个因素，逐一分析并说明它们之间的关系\n"
                "- 引用具体的章节标题作为分析依据"
            ),
            "exploratory": (
                f"{prefix}"
                "请提供多种可行的方法或方案。\n"
                "- 列出每种方法的核心思路和适用场景\n"
                "- 简要对比各方法的优缺点\n"
                "- 如果资料中有推荐，给出推荐建议"
            ),
            "comparison": (
                f"{prefix}"
                "请对比分析不同选项。\n"
                "- 优先使用表格展示差异点\n"
                "- 列出各选项的核心特征、优势和局限\n"
                "- 如果资料中有明确的选择建议，请引用"
            ),
            "procedural": (
                f"{prefix}"
                "请给出详细的操作步骤。\n"
                "- 按顺序编号，确保每一步清晰可执行\n"
                "- 对关键步骤补充说明其目的\n"
                "- 如果有注意事项，单独列出"
            ),
        }
        return guidance_map.get(query_type, guidance_map["factual"])

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
