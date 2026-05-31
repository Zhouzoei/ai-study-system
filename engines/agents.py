import re
import json
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Generator
from dataclasses import dataclass, field

from engines.resilience import (
    HealthChecker, DegradedResult, ErrorCode, ERROR_USER_MESSAGES,
    build_degraded_answer,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    answer: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    agent_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_code: ErrorCode = ErrorCode.OK
    degraded: bool = False
    degradation_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "agent_type": self.agent_type,
            "metadata": self.metadata,
            "error_code": self.error_code.value,
            "degraded": self.degraded,
            "degradation_note": self.degradation_note,
        }


class BaseAgent:
    def __init__(
        self,
        pipeline=None,
        llm_func: Optional[Callable] = None,
    ):
        self.pipeline = pipeline
        self.llm_func = llm_func

    def _get_health(self) -> HealthChecker:
        if self.pipeline and hasattr(self.pipeline, 'health_checker'):
            return self.pipeline.health_checker
        return HealthChecker(None)

    def _get_learning_context(self) -> str:
        if self.pipeline and hasattr(self.pipeline, 'learning_context'):
            return self.pipeline.learning_context.build_markdown_context()
        return ""

    def _extract_knowledge_node_ids(self, contexts: List[Dict]) -> List[str]:
        seen = set()
        ids = []
        for ctx in contexts:
            nid = ctx.get("node_id", "")
            if nid and nid not in seen:
                seen.add(nid)
                ids.append(nid)
        return ids

    def _retrieve_knowledge(self, message: str, topic: str, doc_id: Optional[str], top_k: int = 5) -> List[Dict]:
        if not self.pipeline:
            return []
        query = topic if topic else message
        try:
            result = self.pipeline.query(query, use_hybrid=True, use_reranker=False, top_k=top_k, doc_id=doc_id)
            self._get_health().record_retrieval_success()
            return result.get("context_sources", [])
        except Exception as e:
            logger.warning(f"{self.__class__.__name__} retrieval failed: {e}")
            self._get_health().record_retrieval_failure()
            try:
                result = self.pipeline.query(query, use_hybrid=False, top_k=top_k, doc_id=doc_id)
                return result.get("context_sources", [])
            except Exception:
                return []

    @staticmethod
    def _format_contexts(contexts: List[Dict], top_n: int = 5) -> str:
        parts = []
        for i, ctx in enumerate(contexts[:top_n]):
            excerpt = ctx.get("excerpt", ctx.get("content", ""))
            level_title = ctx.get("level_title", ctx.get("title", ""))
            parts.append(f"[来源 {i+1}] {level_title}\n{excerpt}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_sources(contexts: List[Dict], top_n: int = 5) -> List[Dict]:
        sources = []
        for ctx in contexts[:top_n]:
            sources.append({
                "level_title": ctx.get("level_title", ctx.get("title", "")),
                "excerpt": ctx.get("excerpt", ctx.get("content", ""))[:200],
            })
        return sources


class QuizAgent(BaseAgent):
    def __init__(
        self,
        pipeline=None,
        llm_func: Optional[Callable] = None,
    ):
        super().__init__(pipeline=pipeline, llm_func=llm_func)

    def generate(
        self,
        message: str,
        topic: str = "",
        sub_type: str = "choice",
        doc_id: Optional[str] = None,
    ) -> AgentResult:
        health = self._get_health()
        contexts = self._retrieve_knowledge(message, topic, doc_id)
        knowledge_node_ids = self._extract_knowledge_node_ids(contexts)

        if not contexts:
            return AgentResult(
                answer=ERROR_USER_MESSAGES[ErrorCode.RETRIEVAL_EMPTY],
                agent_type="quiz",
                error_code=ErrorCode.RETRIEVAL_EMPTY,
                metadata={"knowledge_node_ids": []},
            )

        if not health.check_llm():
            degraded = build_degraded_answer("quiz", contexts, ErrorCode.LLM_UNAVAILABLE, topic)
            return AgentResult(
                answer=degraded.answer,
                sources=degraded.sources,
                agent_type="quiz",
                error_code=degraded.error_code,
                degraded=True,
                degradation_note=degraded.degradation_note,
                metadata={"topic": topic, "sub_type": sub_type, "num_contexts": len(contexts), "knowledge_node_ids": knowledge_node_ids},
            )

        context_text = self._format_contexts(contexts)
        sources = self._extract_sources(contexts)
        prompt = self._build_quiz_prompt(context_text, topic or message, sub_type)

        try:
            answer = self.llm_func(prompt)
            health.record_llm_success()
            return AgentResult(
                answer=answer,
                sources=sources,
                agent_type="quiz",
                metadata={"topic": topic, "sub_type": sub_type, "num_contexts": len(contexts), "knowledge_node_ids": knowledge_node_ids},
            )
        except Exception as e:
            error_code = health.classify_error(e)
            health.record_llm_failure(e)
            logger.warning(f"QuizAgent LLM failed: {e}")
            degraded = build_degraded_answer("quiz", contexts, error_code, topic)
            return AgentResult(
                answer=degraded.answer,
                sources=degraded.sources,
                agent_type="quiz",
                error_code=error_code,
                degraded=True,
                degradation_note=degraded.degradation_note,
                metadata={"topic": topic, "sub_type": sub_type, "num_contexts": len(contexts), "knowledge_node_ids": knowledge_node_ids},
            )

    def generate_stream(
        self,
        message: str,
        topic: str = "",
        sub_type: str = "choice",
        doc_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        health = self._get_health()
        contexts = self._retrieve_knowledge(message, topic, doc_id)
        knowledge_node_ids = self._extract_knowledge_node_ids(contexts)

        if not contexts:
            yield {"type": "error", "content": ERROR_USER_MESSAGES[ErrorCode.RETRIEVAL_EMPTY], "error_code": ErrorCode.RETRIEVAL_EMPTY.value}
            return

        if not health.check_llm():
            degraded = build_degraded_answer("quiz", contexts, ErrorCode.LLM_UNAVAILABLE, topic)
            yield {"type": "degraded", "content": degraded.answer, "sources": degraded.sources, "degradation_note": degraded.degradation_note}
            return

        yield {"type": "progress", "content": "正在检索知识库..."}
        context_text = self._format_contexts(contexts)
        sources = self._extract_sources(contexts)
        yield {"type": "progress", "content": "正在生成题目..."}

        prompt = self._build_quiz_prompt(context_text, topic or message, sub_type)
        done_metadata = {"topic": topic, "sub_type": sub_type, "num_contexts": len(contexts), "knowledge_node_ids": knowledge_node_ids}

        if health.check_llm_stream():
            try:
                partial = ""
                for chunk in self.pipeline.llm_service.invoke_stream(prompt):
                    partial += chunk
                    yield {"type": "token", "content": chunk}
                health.record_llm_success()
                yield {"type": "done", "sources": sources, "metadata": done_metadata}
                return
            except Exception as e:
                logger.warning(f"QuizAgent stream failed, falling back to sync: {e}")

        try:
            answer = self.llm_func(prompt)
            health.record_llm_success()
            yield {"type": "full", "content": answer, "sources": sources, "metadata": done_metadata}
        except Exception as e:
            error_code = health.classify_error(e)
            health.record_llm_failure(e)
            degraded = build_degraded_answer("quiz", contexts, error_code, topic)
            yield {"type": "degraded", "content": degraded.answer, "sources": degraded.sources, "degradation_note": degraded.degradation_note}

    def evaluate_answer(
        self,
        question: str,
        user_answer: str,
        correct_answer: str,
        knowledge_node_ids: Optional[List[str]] = None,
        sub_type: str = "choice",
        user_id: str = "default",
        session_id: Optional[str] = None,
        options: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        knowledge_node_ids = knowledge_node_ids or []
        is_correct = False
        score = 0.0
        essay_comment = ""

        recorded_user_answer = user_answer
        recorded_correct_answer = correct_answer

        if sub_type == "choice":
            user_choice = user_answer.strip().upper().rstrip(".")
            correct_choice = correct_answer.strip().upper().rstrip(".")
            is_correct = user_choice == correct_choice
            score = 1.0 if is_correct else 0.0
            if options:
                opt_map = {o.get("label", "").upper(): o.get("text", o.get("label", "")) for o in options}
                recorded_user_answer = opt_map.get(user_choice, user_answer)
                recorded_correct_answer = opt_map.get(correct_choice, correct_answer)
        elif sub_type in ("judgment", "fill"):
            is_correct = user_answer.strip().lower() == correct_answer.strip().lower()
            score = 1.0 if is_correct else 0.0
        elif sub_type == "essay" and self.llm_func:
            essay_comment = ""
            try:
                safe_question = question.replace("{", "{{").replace("}", "}}")
                safe_correct = correct_answer.replace("{", "{{").replace("}", "}}")
                safe_user = user_answer.replace("{", "{{").replace("}", "}}")
                prompt = f"""请评估以下简答回答的得分。

题目: {safe_question}
参考答案: {safe_correct}
用户回答: {safe_user}

请给出0-1之间的分数（0=完全不正确，1=完全正确）和简要评语。
只输出JSON格式: {{"score": 0.85, "comment": "..."}}"""
                resp = self.llm_func(prompt).strip()
                if resp.startswith("```"):
                    resp = resp.split("\n", 1)[-1].rsplit("```", 1)[0]
                result = json.loads(resp)
                score = max(0.0, min(1.0, float(result.get("score", 0))))
                essay_comment = result.get("comment", "")
                is_correct = score >= 0.6
            except Exception as e:
                logger.warning(f"Essay grading LLM failed, using fallback score: {e}")
                score = 0.5
                essay_comment = "自动评分暂时不可用，已给予默认分数"
                is_correct = False

        quality = 5 if is_correct and score >= 0.9 else (4 if is_correct else (2 if score >= 0.4 else 1))

        if self.pipeline and self.pipeline.progress_tracker:
            tracker = self.pipeline.progress_tracker
            for node_id in knowledge_node_ids:
                tracker.record_review(
                    knowledge_node_id=node_id,
                    quality=quality,
                    user_id=user_id,
                )

            if not is_correct:
                tracker.record_wrong_answer(
                    question=question,
                    user_answer=recorded_user_answer,
                    correct_answer=recorded_correct_answer,
                    knowledge_node_ids=knowledge_node_ids,
                    user_id=user_id,
                )

        if self.pipeline and hasattr(self.pipeline, 'event_bus'):
            self.pipeline.event_bus.publish(
                "quiz:evaluated", "quiz_agent",
                payload={
                    "is_correct": is_correct,
                    "score": score,
                    "quality": quality,
                    "knowledge_node_ids": knowledge_node_ids,
                    "question": question[:200],
                    "user_id": user_id,
                    "session_id": session_id or "",
                },
            )

        result = {
            "is_correct": is_correct,
            "score": round(score, 2),
            "quality": quality,
            "knowledge_node_ids": knowledge_node_ids,
            "sub_type": sub_type,
        }
        if sub_type == "essay":
            result["comment"] = essay_comment
        return result

    def _retrieve_knowledge(self, message: str, topic: str, doc_id: Optional[str]) -> List[Dict]:
        contexts = super()._retrieve_knowledge(message, topic, doc_id, top_k=5)
        if self.pipeline:
            self._concept_sources = getattr(self.pipeline, '_last_concept_sources', [])
        return contexts

    def _format_concepts(self) -> str:
        cs = getattr(self, "_concept_sources", [])
        if not cs:
            return ""
        parts = []
        for c in cs[:3]:
            name = c.get("concept", "")
            definition = c.get("definition", "")
            bloom = c.get("bloom_level", "")
            prereqs = c.get("prerequisites", [])
            if name and definition:
                prereq_str = f"，前置: {'、'.join(prereqs[:3])}" if prereqs else ""
                parts.append(f"- **{name}** (Bloom: {bloom}{prereq_str}): {definition[:200]}")
        if not parts:
            return ""
        return "\n[相关概念]\n" + "\n".join(parts)

    def _build_quiz_prompt(self, context_text: str, topic: str, sub_type: str) -> str:
        type_instructions = {
            "choice": (
                "请出一道四选一的选择题。\n"
                "格式要求：\n## 题目\n[题目内容]\n\n## 选项\nA. [选项]\nB. [选项]\nC. [选项]\nD. [选项]\n\n"
                "## 正确答案\n[正确选项字母]\n\n## 解析\n[基于参考资料的详细解析，标注来源]"
            ),
            "judgment": (
                "请出一道判断题。\n格式要求：\n## 题目\n[判断陈述]\n\n## 答案\n✅ 正确 / ❌ 错误\n\n"
                "## 解析\n[基于参考资料的详细解析，标注来源]"
            ),
            "fill": (
                "请出一道填空题。\n格式要求：\n## 题目\n[包含____的题目]\n\n## 答案\n[填空答案]\n\n"
                "## 解析\n[基于参考资料的详细解析，标注来源]"
            ),
            "essay": (
                "请出一道简答题。\n格式要求：\n## 题目\n[简答题目]\n\n## 参考答案\n[基于参考资料的参考答案，标注来源]\n\n"
                "## 评分要点\n[列出3-5个得分点]"
            ),
        }
        instruction = type_instructions.get(sub_type, type_instructions["choice"])

        learning_context = ""
        difficulty_instruction = ""
        if self.pipeline and hasattr(self.pipeline, 'learning_context'):
            learning_context = self.pipeline.learning_context.build_markdown_context()
        if self.pipeline and hasattr(self.pipeline, 'analytics'):
            try:
                summary = self.pipeline.analytics._get_progress_data("default")
                mastery_dist = summary.get("mastery_distribution", {})
                total = summary.get("total_knowledge_nodes", 0)
                if total > 0:
                    mastered = mastery_dist.get("mastered", 0) + mastery_dist.get("proficient", 0)
                    pct = round(mastered / total * 100) if total > 0 else 0
                    if pct < 20:
                        difficulty_instruction = "\n注意：该用户是初学者，请出基础难度的题目，重点考察概念理解和记忆。"
                    elif pct < 50:
                        difficulty_instruction = "\n注意：该用户处于中级水平，请出中等难度的题目，重点考察概念应用和分析。"
                    else:
                        difficulty_instruction = "\n注意：该用户处于高级水平，请出较高难度的题目，重点考察综合评价和创造能力。"
            except Exception as e:
                logger.debug(f"Failed to get learner level for quiz: {e}")

        return f"""你是一个专业的学习测试助手。请严格基于以下参考资料出题，不要使用参考资料以外的知识。
{learning_context}
参考资料:
{context_text}
{self._format_concepts()}

出题主题: {topic}

{instruction}
{difficulty_instruction}
重要：题目和解析必须基于参考资料内容，如果参考资料不足以出题，请说明。"""

    def generate_adaptive_quiz(
        self,
        user_id: str = "default",
        sub_type: str = "choice",
        limit: int = 5,
    ) -> AgentResult:
        if not self.pipeline:
            return AgentResult(answer="系统未初始化", agent_type="quiz", error_code=ErrorCode.AGENT_ERROR)

        weak_nodes = self.pipeline.progress_tracker.get_weak_nodes(user_id, threshold=1)
        if not weak_nodes:
            return AgentResult(
                answer="🎉 没有发现薄弱知识点！你可以尝试「出题」功能来随机测试。",
                agent_type="quiz",
                metadata={"adaptive": True, "weak_count": 0},
            )

        weak_items = weak_nodes[:limit]
        topics_text = "\n".join(f"- {w['title']}(答错{w['wrong_count']}次, 掌握度:{w['mastery']})" for w in weak_items)
        topic_names = [w['title'] for w in weak_items]

        contexts = []
        for name in topic_names:
            try:
                result = self.pipeline.query(name, use_hybrid=True, top_k=3)
                contexts.extend(result.get("context_sources", []))
            except Exception as e:
                logger.debug(f"Query failed for topic '{name}': {e}")

        context_text = self._format_contexts(contexts) if contexts else "（无直接参考资料，请基于通用知识出题）"
        type_instruction = self._build_quiz_prompt(context_text, "", sub_type).split("出题主题:")[0]

        prompt = f"""你是一个自适应出题助手。用户有以下薄弱知识点需要针对性练习：

{topics_text}

请针对这些薄弱知识点出一道{sub_type}题，帮助用户巩固理解。

{type_instruction}

出题主题: {', '.join(topic_names[:3])}

请确保题目直接针对薄弱知识点，难度适中："""

        health = self._get_health()
        try:
            answer = self.llm_func(prompt)
            health.record_llm_success()
            return AgentResult(
                answer=answer,
                sources=self._extract_sources(contexts),
                agent_type="quiz",
                metadata={"adaptive": True, "weak_count": len(weak_items), "weak_topics": topic_names},
            )
        except Exception as e:
            error_code = health.classify_error(e)
            health.record_llm_failure(e)
            return AgentResult(
                answer="生成自适应题目失败，请稍后重试",
                agent_type="quiz",
                error_code=error_code,
            )

    def generate_styled_quiz(
        self,
        topic: str = "",
        sub_type: str = "choice",
        doc_id: Optional[str] = None,
    ) -> AgentResult:
        contexts = self._retrieve_knowledge("", topic, doc_id)
        if not contexts:
            return AgentResult(
                answer=ERROR_USER_MESSAGES[ErrorCode.RETRIEVAL_EMPTY],
                agent_type="quiz",
                error_code=ErrorCode.RETRIEVAL_EMPTY,
            )

        doc_examples = []
        for ctx in contexts[:3]:
            content = ctx.get("excerpt", ctx.get("content", ""))
            title = ctx.get("level_title", ctx.get("title", ""))
            if content:
                doc_examples.append(f"[示例段落: {title}]\n{content[:600]}")

        examples_text = "\n\n---\n\n".join(doc_examples) if doc_examples else ""

        type_instructions = {
            "choice": "选择题（四个选项）",
            "judgment": "判断题",
            "fill": "填空题",
            "essay": "简答题",
        }
        type_name = type_instructions.get(sub_type, "选择题")

        prompt = f"""你是一个擅长模仿出题风格的AI助手。请分析以下文档段落的表达风格、术语使用和难度水平，然后模仿这种风格出一道{type_name}。

[文档样本]
{examples_text}

请分析以上文档的风格特征（包括：语言风格、术语密度、句子结构、难度水平），然后模仿这个风格出一道关于「{topic}」的{type_name}。

要求：
1. 题目风格、术语使用、难度要与文档样本保持一致
2. 题目内容必须基于文档中的知识点
3. 解析要详细，标注知识点来源

格式要求：
## 题目
[题目内容]

## 选项（如果是选择题）
A. [选项] B. [选项] C. [选项] D. [选项]

## 正确答案
[答案]

## 风格说明
[简要说明本题模仿了文档的哪些风格特征]

## 解析
[基于参考资料的详细解析]"""

        health = self._get_health()
        context_text = self._format_contexts(contexts)
        prompt_with_context = f"""{prompt}

参考资料:
{context_text}"""

        try:
            answer = self.llm_func(prompt_with_context)
            health.record_llm_success()
            return AgentResult(
                answer=answer,
                sources=self._extract_sources(contexts),
                agent_type="quiz",
                metadata={"styled": True, "topic": topic, "sub_type": sub_type},
            )
        except Exception as e:
            error_code = health.classify_error(e)
            health.record_llm_failure(e)
            return AgentResult(
                answer="生成风格化题目失败",
                agent_type="quiz",
                error_code=error_code,
            )


class SummaryAgent(BaseAgent):
    def __init__(self, pipeline=None, llm_func: Optional[Callable] = None):
        super().__init__(pipeline=pipeline, llm_func=llm_func)

    def generate(
        self,
        message: str,
        topic: str = "",
        sub_type: str = "topic",
        doc_id: Optional[str] = None,
    ) -> AgentResult:
        health = self._get_health()
        contexts = self._retrieve_knowledge(message, topic, doc_id)

        if not contexts:
            return AgentResult(
                answer=ERROR_USER_MESSAGES[ErrorCode.RETRIEVAL_EMPTY],
                agent_type="summary",
                error_code=ErrorCode.RETRIEVAL_EMPTY,
            )

        if not health.check_llm():
            degraded = build_degraded_answer("summary", contexts, ErrorCode.LLM_UNAVAILABLE, topic)
            return AgentResult(
                answer=degraded.answer,
                sources=degraded.sources,
                agent_type="summary",
                error_code=degraded.error_code,
                degraded=True,
                degradation_note=degraded.degradation_note,
                metadata={"topic": topic, "sub_type": sub_type, "num_contexts": len(contexts)},
            )

        context_text = self._format_contexts(contexts, top_n=8)
        sources = self._extract_sources(contexts, top_n=8)
        prompt = self._build_summary_prompt(context_text, topic or message, sub_type)

        try:
            answer = self.llm_func(prompt)
            health.record_llm_success()
            return AgentResult(
                answer=answer,
                sources=sources,
                agent_type="summary",
                metadata={"topic": topic, "sub_type": sub_type, "num_contexts": len(contexts)},
            )
        except Exception as e:
            error_code = health.classify_error(e)
            health.record_llm_failure(e)
            logger.warning(f"SummaryAgent LLM failed: {e}")
            degraded = build_degraded_answer("summary", contexts, error_code, topic)
            return AgentResult(
                answer=degraded.answer,
                sources=degraded.sources,
                agent_type="summary",
                error_code=error_code,
                degraded=True,
                degradation_note=degraded.degradation_note,
                metadata={"topic": topic, "sub_type": sub_type, "num_contexts": len(contexts)},
            )

    def generate_stream(
        self,
        message: str,
        topic: str = "",
        sub_type: str = "topic",
        doc_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        health = self._get_health()
        contexts = self._retrieve_knowledge(message, topic, doc_id)

        if not contexts:
            yield {"type": "error", "content": ERROR_USER_MESSAGES[ErrorCode.RETRIEVAL_EMPTY], "error_code": ErrorCode.RETRIEVAL_EMPTY.value}
            return

        if not health.check_llm():
            degraded = build_degraded_answer("summary", contexts, ErrorCode.LLM_UNAVAILABLE, topic)
            yield {"type": "degraded", "content": degraded.answer, "sources": degraded.sources, "degradation_note": degraded.degradation_note}
            return

        yield {"type": "progress", "content": "正在检索知识库..."}
        context_text = self._format_contexts(contexts, top_n=8)
        sources = self._extract_sources(contexts, top_n=8)
        yield {"type": "progress", "content": "正在生成总结..."}

        prompt = self._build_summary_prompt(context_text, topic or message, sub_type)

        if health.check_llm_stream():
            try:
                for chunk in self.pipeline.llm_service.invoke_stream(prompt):
                    yield {"type": "token", "content": chunk}
                health.record_llm_success()
                yield {"type": "done", "sources": sources, "metadata": {"topic": topic, "sub_type": sub_type, "num_contexts": len(contexts)}}
                return
            except Exception as e:
                logger.warning(f"SummaryAgent stream failed, falling back to sync: {e}")

        try:
            answer = self.llm_func(prompt)
            health.record_llm_success()
            yield {"type": "full", "content": answer, "sources": sources, "metadata": {"topic": topic, "sub_type": sub_type, "num_contexts": len(contexts)}}
        except Exception as e:
            error_code = health.classify_error(e)
            health.record_llm_failure(e)
            degraded = build_degraded_answer("summary", contexts, error_code, topic)
            yield {"type": "degraded", "content": degraded.answer, "sources": degraded.sources, "degradation_note": degraded.degradation_note}

    def _retrieve_knowledge(self, message: str, topic: str, doc_id: Optional[str]) -> List[Dict]:
        return super()._retrieve_knowledge(message, topic, doc_id, top_k=8)

    def _build_summary_prompt(self, context_text: str, topic: str, sub_type: str) -> str:
        type_instructions = {
            "document": "请对以下文档内容进行全面总结。",
            "chapter": "请对以下章节内容进行重点总结。",
            "topic": "请对以下关于指定主题的内容进行总结归纳。",
        }
        instruction = type_instructions.get(sub_type, type_instructions["topic"])

        learning_context = ""
        if self.pipeline and hasattr(self.pipeline, 'learning_context'):
            learning_context = self.pipeline.learning_context.build_markdown_context()

        return f"""你是一个专业的知识总结助手。请严格基于以下参考资料进行总结，不要添加参考资料中没有的信息。
{learning_context}
{instruction}

参考资料:
{context_text}

总结主题: {topic}

请按以下格式输出总结：

## 核心要点
- [3-5个核心观点，每个观点标注来源]

## 关键术语
- [列出关键术语及其简短定义]

## 知识结构
[用层级列表展示知识点之间的逻辑关系]

## 一句话总结
[用一句话概括核心内容]"""


class ReviewAgent(BaseAgent):
    def __init__(self, pipeline=None, llm_func: Optional[Callable] = None):
        super().__init__(pipeline=pipeline, llm_func=llm_func)

    def generate(
        self,
        message: str,
        topic: str = "",
        sub_type: str = "scheduled",
        user_id: str = "default",
        doc_id: Optional[str] = None,
    ) -> AgentResult:
        health = self._get_health()

        if not health.check_llm():
            return self._degraded_review(user_id, topic, doc_id, ErrorCode.LLM_UNAVAILABLE)

        if sub_type == "weak_point":
            result = self._weak_point_review(user_id, topic, doc_id)
        elif sub_type == "associated":
            result = self._associated_review(topic, doc_id)
        else:
            result = self._scheduled_review(user_id, topic, doc_id)

        if not result.degraded and self.pipeline and hasattr(self.pipeline, 'event_bus'):
            self.pipeline.event_bus.publish(
                "review:completed", "review_agent",
                payload={
                    "sub_type": sub_type,
                    "topic": topic,
                    "user_id": user_id,
                    "doc_id": doc_id or "",
                    "metadata": result.metadata,
                },
            )

        return result

    def generate_stream(
        self,
        message: str,
        topic: str = "",
        sub_type: str = "scheduled",
        user_id: str = "default",
        doc_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        health = self._get_health()

        if not health.check_llm():
            degraded = self._degraded_review(user_id, topic, doc_id, ErrorCode.LLM_UNAVAILABLE)
            yield {"type": "degraded", "content": degraded.answer, "sources": degraded.sources, "degradation_note": degraded.degradation_note}
            return

        yield {"type": "progress", "content": "正在获取复习数据..."}
        result = self.generate(message, topic=topic, sub_type=sub_type, user_id=user_id, doc_id=doc_id)

        if result.degraded or result.error_code != ErrorCode.OK:
            yield {"type": "degraded", "content": result.answer, "sources": result.sources, "degradation_note": result.degradation_note}
            return

        yield {"type": "progress", "content": "正在生成复习内容..."}

        if health.check_llm_stream() and self.pipeline and self.pipeline.llm_service:
            prompt = self._build_review_prompt_from_result(result)
            if prompt:
                try:
                    for chunk in self.pipeline.llm_service.invoke_stream(prompt):
                        yield {"type": "token", "content": chunk}
                    health.record_llm_success()
                    yield {"type": "done", "sources": result.sources, "metadata": result.metadata}
                    return
                except Exception as e:
                    logger.warning(f"ReviewAgent stream failed: {e}")

        yield {"type": "full", "content": result.answer, "sources": result.sources, "metadata": result.metadata}

    def _degraded_review(
        self,
        user_id: str,
        topic: str,
        doc_id: Optional[str],
        error_code: ErrorCode,
    ) -> AgentResult:
        if not self.pipeline:
            return AgentResult(
                answer=ERROR_USER_MESSAGES[ErrorCode.AGENT_ERROR],
                agent_type="review",
                error_code=ErrorCode.AGENT_ERROR,
                degraded=True,
                degradation_note=ERROR_USER_MESSAGES[error_code],
            )

        try:
            due_reviews = self.pipeline.get_due_reviews(user_id, limit=5)
        except Exception:
            due_reviews = []

        if not due_reviews:
            return AgentResult(
                answer="🎉 当前没有需要复习的知识点！",
                agent_type="review",
                error_code=error_code,
                degraded=True,
                degradation_note=ERROR_USER_MESSAGES[error_code],
                metadata={"review_type": "scheduled", "due_count": 0},
            )

        items = []
        for r in due_reviews[:5]:
            title = r.get("title", "")
            mastery = r.get("mastery", "")
            overdue = r.get("overdue_days", 0)
            items.append(f"- **{title}** (掌握度: {mastery}, 逾期: {overdue}天)")

        note = ERROR_USER_MESSAGES[error_code]
        answer = f"⚠️ {note}\n\n以下是需要复习的知识点：\n\n" + "\n".join(items)
        return AgentResult(
            answer=answer,
            agent_type="review",
            error_code=error_code,
            degraded=True,
            degradation_note=note,
            metadata={"review_type": "scheduled", "due_count": len(due_reviews)},
        )

    def _scheduled_review(self, user_id: str, topic: str, doc_id: Optional[str]) -> AgentResult:
        if not self.pipeline:
            return AgentResult(answer=ERROR_USER_MESSAGES[ErrorCode.AGENT_ERROR], agent_type="review", error_code=ErrorCode.AGENT_ERROR)

        health = self._get_health()

        try:
            due_reviews = self.pipeline.get_due_reviews(user_id, limit=5)
        except Exception as e:
            logger.warning(f"ReviewAgent get_due_reviews failed: {e}")
            return self._degraded_review(user_id, topic, doc_id, ErrorCode.AGENT_ERROR)

        if not due_reviews:
            return AgentResult(
                answer="🎉 当前没有需要复习的知识点！继续保持学习节奏。",
                agent_type="review",
                metadata={"review_type": "scheduled", "due_count": 0},
            )

        review_items = []
        sources = []
        mastery_context = ""
        for review in due_reviews[:3]:
            title = review.get("title", "")
            mastery = review.get("mastery", "")
            overdue = review.get("overdue_days", 0)
            exposures = review.get("exposure_count", 0)
            review_items.append(f"- **{title}** (掌握度: {mastery}, 逾期: {overdue}天, 复习次数: {exposures})")
            mastery_context += f"\n知识点: {title}, 当前掌握度: {mastery}, 逾期天数: {overdue}, 历史复习次数: {exposures}"

            if title and self.pipeline:
                try:
                    kg_result = self.pipeline.query_knowledge_graph(title)
                    if kg_result:
                        neighbors = kg_result.get("neighbors", [])
                        for n in neighbors[:2]:
                            nname = n.get("name", "")
                            if nname:
                                review_items.append(f"  - 关联: {nname}")
                    health.record_kg_success()
                except Exception as e:
                    logger.debug(f"KG query failed for '{title}': {e}")
                    health.record_kg_failure()

        review_text = "\n".join(review_items)

        contexts = []
        for review in due_reviews[:3]:
            title = review.get("title", "")
            if title:
                try:
                    result = self.pipeline.query(title, use_hybrid=True, top_k=2, doc_id=doc_id)
                    for src in result.get("context_sources", [])[:1]:
                        contexts.append(src)
                        sources.append({
                            "level_title": src.get("level_title", ""),
                            "excerpt": src.get("excerpt", "")[:200],
                        })
                except Exception as e:
                    logger.debug(f"Retrieval failed for review item '{title}': {e}")

        context_text = ""
        if contexts:
            parts = []
            for i, ctx in enumerate(contexts):
                excerpt = ctx.get("excerpt", "")
                level_title = ctx.get("level_title", "")
                parts.append(f"[参考资料 {i+1}] {level_title}\n{excerpt}")
            context_text = "\n\n---\n\n".join(parts)

        prompt = f"""你是一个个性化的学习复习助手。以下是根据间隔重复算法(SM-2)计算出的待复习知识点，请帮助用户进行高效复习。

用户对各知识点的掌握情况：
{mastery_context}

待复习知识点:
{review_text}

{f"参考资料:" + chr(10) + context_text if context_text else ""}

请按以下格式生成复习内容：

## 📋 复习概览
[简要说明当前复习状态和重点]

## 🔍 知识点回顾
[对每个待复习知识点，用1-2段话回顾核心内容，帮助用户重新激活记忆]

## 💡 记忆技巧
[针对容易遗忘的知识点，提供记忆口诀或关联记忆方法]

## ✅ 自测问题
[针对待复习知识点，提出2-3个自测问题。**重要：只出问题，不要给出答案。** 每个问题以问号结尾。格式为每行一个"问题: [问题内容]"]"""

        try:
            answer = self.llm_func(prompt)
            health.record_llm_success()
            return AgentResult(
                answer=answer,
                sources=sources,
                agent_type="review",
                metadata={
                    "review_type": "scheduled",
                    "due_count": len(due_reviews),
                    "reviewed_topics": [r.get("title", "") for r in due_reviews[:3]],
                },
            )
        except Exception as e:
            error_code = health.classify_error(e)
            health.record_llm_failure(e)
            logger.warning(f"ReviewAgent LLM failed: {e}")
            return self._degraded_review(user_id, topic, doc_id, error_code)

    def _weak_point_review(self, user_id: str, topic: str, doc_id: Optional[str]) -> AgentResult:
        if not self.pipeline:
            return AgentResult(answer=ERROR_USER_MESSAGES[ErrorCode.AGENT_ERROR], agent_type="review", error_code=ErrorCode.AGENT_ERROR)

        health = self._get_health()

        try:
            recommendations = self.pipeline.get_study_recommendations(user_id)
        except Exception as e:
            logger.warning(f"ReviewAgent get_study_recommendations failed: {e}")
            return self._degraded_review(user_id, topic, doc_id, ErrorCode.AGENT_ERROR)

        if not recommendations:
            return AgentResult(
                answer="🎉 暂无薄弱知识点，学习状态良好！",
                agent_type="review",
                metadata={"review_type": "weak_point"},
            )

        weak_items = []
        for rec in recommendations[:5]:
            title = rec.get("title", "")
            reason = rec.get("reason", "")
            action = rec.get("action", "")
            priority = rec.get("priority", "medium")
            weak_items.append(f"- **{title}** (优先级: {priority})\n  原因: {reason}\n  建议: {action}")

        weak_text = "\n".join(weak_items)

        prompt = f"""你是一个个性化学习助手。以下是系统检测到的薄弱知识点，请帮助用户有针对性地加强学习。

薄弱知识点:
{weak_text}

请按以下格式生成针对性复习内容：

## ⚠️ 薄弱知识点分析
[分析薄弱环节的共同特征]

## 📖 针对性讲解
[对每个薄弱知识点进行深入浅出的讲解，确保理解到位]

## 🎯 练习建议
[针对薄弱环节的具体练习建议]"""

        try:
            answer = self.llm_func(prompt)
            health.record_llm_success()
            return AgentResult(
                answer=answer,
                agent_type="review",
                metadata={"review_type": "weak_point", "weak_count": len(recommendations)},
            )
        except Exception as e:
            error_code = health.classify_error(e)
            health.record_llm_failure(e)
            logger.warning(f"ReviewAgent weak_point LLM failed: {e}")
            return self._degraded_review(user_id, topic, doc_id, error_code)

    def _associated_review(self, topic: str, doc_id: Optional[str]) -> AgentResult:
        if not self.pipeline:
            return AgentResult(answer=ERROR_USER_MESSAGES[ErrorCode.AGENT_ERROR], agent_type="review", error_code=ErrorCode.AGENT_ERROR)

        health = self._get_health()

        if not topic:
            return AgentResult(answer="请指定要关联复习的知识点主题。", agent_type="review")

        try:
            kg_result = self.pipeline.query_knowledge_graph(topic)
            health.record_kg_success()
        except Exception as e:
            logger.warning(f"ReviewAgent KG query failed: {e}")
            health.record_kg_failure()
            return AgentResult(
                answer=f"⚠️ 知识图谱查询异常，无法获取与「{topic}」的关联知识点。" + ERROR_USER_MESSAGES[ErrorCode.KG_ERROR],
                agent_type="review",
                error_code=ErrorCode.KG_ERROR,
                degraded=True,
                degradation_note=ERROR_USER_MESSAGES[ErrorCode.KG_ERROR],
            )

        if not kg_result or not kg_result.get("neighbors"):
            return AgentResult(
                answer=f"未找到与「{topic}」关联的知识点。请尝试其他关键词。",
                agent_type="review",
            )

        neighbors = kg_result.get("neighbors", [])
        relations = kg_result.get("relations", [])

        assoc_items = []
        for n in neighbors[:5]:
            name = n.get("name", "")
            ntype = n.get("entity_type", "")
            assoc_items.append(f"- {name} ({ntype})")

        rel_items = []
        for r in relations[:5]:
            rtype = r.get("relation_type", "")
            desc = r.get("description", "")
            rel_items.append(f"- {rtype}: {desc}")

        assoc_text = "\n".join(assoc_items)
        rel_text = "\n".join(rel_items)

        prompt = f"""你是一个个性化学习助手。以下是与「{topic}」关联的知识点，请帮助用户进行关联复习。
{self._get_learning_context()}
关联知识点:
{assoc_text}

关联关系:
{rel_text}

请按以下格式生成关联复习内容：

## 🔗 知识关联图
[说明这些知识点之间的逻辑关系和依赖]

## 📝 关联复习要点
[对每个关联知识点，说明其与「{topic}」的关系，并回顾核心内容]

## 🧠 知识网络构建
[帮助用户建立这些知识点之间的心智模型]"""

        try:
            answer = self.llm_func(prompt)
            health.record_llm_success()
            return AgentResult(
                answer=answer,
                agent_type="review",
                metadata={"review_type": "associated", "topic": topic, "neighbor_count": len(neighbors)},
            )
        except Exception as e:
            error_code = health.classify_error(e)
            health.record_llm_failure(e)
            logger.warning(f"ReviewAgent associated LLM failed: {e}")
            return AgentResult(
                answer=f"⚠️ {ERROR_USER_MESSAGES[error_code]}\n\n与「{topic}」关联的知识点：\n\n" + assoc_text,
                agent_type="review",
                error_code=error_code,
                degraded=True,
                degradation_note=ERROR_USER_MESSAGES[error_code],
            )

    def _build_review_prompt_from_result(self, result: AgentResult) -> Optional[str]:
        if not result or not result.answer:
            return None
        return f"请优化以下复习内容，使其更清晰易读：\n\n{result.answer}"
