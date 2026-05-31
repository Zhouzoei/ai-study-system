import json
import re
import time
import logging
from typing import Dict, Any, Optional, Generator, List

from engines.intent_router import IntentType, IntentResult

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = """你是一个 AI 学习助手的决策核心。请分析用户消息，决定下一步行动。

可用工具:
{tools_description}

输出格式（严格JSON，不要其他内容）:

1. 如果需要调用工具:
{{"action": "工具名称", "params": {{"参数名": "参数值"}} }}

2. 如果已有足够信息回答:
{{"answer": "你的回答"}}

每次只调用一个工具。观察结果后继续。最多 {max_iterations} 步。"""

OBSERVATION_PROMPT = """用户原始消息: {message}

{history}

第 {step} 步:
之前调用: {last_action}
观察结果: {observation}

接下来要做什么？只输出JSON（action或answer）："""


class OrchestratorAgent:
    def __init__(self, pipeline, llm_func):
        self.pipeline = pipeline
        self.llm_func = llm_func
        self.max_iterations = 5

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

        is_chat_intent = intent_result.intent == IntentType.CHAT
        if is_chat_intent:
            if self.llm_func:
                try:
                    answer = self.llm_func(
                        self.pipeline.render_prompt("chat_response", message=message)
                    )
                    yield {"type": "token", "content": answer}
                except Exception:
                    yield {"type": "token", "content": f"{message}？可以详细说说。"}
            else:
                yield {"type": "token", "content": f"{message}？可以详细说说。"}
            yield {"type": "done", "sources": []}
            return

        is_tutor_intent = intent_result.intent == IntentType.TUTOR
        if is_tutor_intent:
            yield {"type": "progress", "content": "正在生成学习建议..."}
            for event in self._handle_tutor_intent(message, intent_result, user_id):
                yield event
            return

        is_simple_chat = (
            intent_result.intent == IntentType.QA
            and len(message.strip()) < 10
            and not self._has_knowledge_intent(message)
            and "?" not in message and "？" not in message
        )
        if is_simple_chat:
            if self.llm_func:
                try:
                    answer = self.llm_func(
                        self.pipeline.render_prompt("chat_response", message=message)
                    )
                    yield {"type": "token", "content": answer}
                except Exception:
                    yield {"type": "token", "content": f"{message}？可以详细说说。"}
            else:
                yield {"type": "token", "content": f"{message}？可以详细说说。"}
            yield {"type": "done", "sources": []}
            return

        if intent_result.confidence >= 0.8 and intent_result.intent != IntentType.QA:
            yield {"type": "progress", "content": f"正在处理{self._intent_label(intent_result.intent)}请求..."}
            for event in self._direct_delegate(intent_result, message, session_id, doc_id, user_id):
                yield event
            return

        yield {"type": "progress", "content": "正在分析你的问题..."}
        session_events = ""
        if hasattr(self.pipeline, 'event_bus'):
            recent = self.pipeline.event_bus.format_recent_context(limit=3)
            if recent:
                session_events = f"\n近期事件:\n{recent}\n"

        for event in self._react_loop(message, intent_result, session_id, doc_id, user_id, extra_context=session_events):
            yield event

    def _direct_delegate(
        self,
        intent_result: IntentResult,
        message: str,
        session_id: Optional[str],
        doc_id: Optional[str],
        user_id: str,
    ) -> Generator[Dict[str, Any], None, None]:
        if intent_result.intent == IntentType.QUIZ:
            for event in self.pipeline.quiz_agent.generate_stream(
                message, topic=intent_result.topic,
                sub_type=intent_result.sub_type or "choice", doc_id=doc_id,
            ):
                yield event

        elif intent_result.intent == IntentType.SUMMARY:
            for event in self.pipeline.summary_agent.generate_stream(
                message, topic=intent_result.topic,
                sub_type=intent_result.sub_type or "topic", doc_id=doc_id,
            ):
                yield event

        elif intent_result.intent == IntentType.REVIEW:
            for event in self.pipeline.review_agent.generate_stream(
                message, topic=intent_result.topic,
                sub_type=intent_result.sub_type or "scheduled",
                user_id=user_id, doc_id=doc_id,
            ):
                yield event

        elif intent_result.intent == IntentType.EXPLAIN:
            for event in self._handle_explain_intent(message, intent_result, session_id, doc_id):
                yield event

        elif intent_result.intent == IntentType.COMPARE:
            for event in self._handle_compare_intent(message, intent_result, session_id, doc_id):
                yield event

        else:
            yield {"type": "progress", "content": "正在处理请求..."}
            for event in self._fallback_qa(message, intent_result, session_id, doc_id):
                yield event

    def _react_loop(
        self,
        message: str,
        intent_result: IntentResult,
        session_id: Optional[str],
        doc_id: Optional[str],
        user_id: str,
        extra_context: str = "",
    ) -> Generator[Dict[str, Any], None, None]:
        tools = self.pipeline.tool_registry.list_tools()
        tool_lines = "\n".join(f"{t['name']}({', '.join(t['parameters'].keys())}): {t['description']}" for t in tools)
        system_prompt = REACT_SYSTEM_PROMPT.format(tools_description=tool_lines, max_iterations=self.max_iterations)

        history = []
        last_observation = ""
        for step in range(self.max_iterations):
            if step == 0:
                prompt = f"{system_prompt}{extra_context}\n\n用户消息: {message}\n\n请分析问题，决定第一步行动。"
            else:
                history_text = "\n".join(f"第{i+1}步: {h}" for i, h in enumerate(history[-3:]))
                prompt = OBSERVATION_PROMPT.format(
                    message=message,
                    history=history_text,
                    step=step + 1,
                    last_action=history[-1] if history else "无",
                    observation=last_observation if history else "无",
                )

            try:
                response = self.llm_func(prompt).strip()
            except Exception as e:
                logger.warning(f"ReAct loop LLM failed at step {step}: {e}")
                break

            parsed = self._parse_json_response(response)
            if not parsed:
                yield {"type": "token", "content": response}
                yield {"type": "done", "sources": []}
                return

            if "answer" in parsed:
                yield {"type": "token", "content": parsed["answer"]}
                yield {"type": "done", "sources": []}
                return

            if "action" not in parsed:
                yield {"type": "token", "content": response}
                yield {"type": "done", "sources": []}
                return

            tool_name = parsed["action"]
            params = parsed.get("params", {})

            if session_id and "session_id" not in params:
                params["session_id"] = session_id
            if user_id and "user_id" not in params:
                params["user_id"] = user_id

            try:
                result = self.pipeline.tool_registry.call(tool_name, **params)
                last_observation = str(result)[:500] if result else "执行成功（无返回数据）"
            except ValueError as e:
                last_observation = f"工具不存在: {tool_name}，可用工具: {tool_lines[:200]}"
            except Exception as e:
                last_observation = f"工具执行失败: {e}"

            history.append(f"调用 {tool_name}({json.dumps(params, ensure_ascii=False)}) → {last_observation[:100]}")

        fallback = self._fallback_qa(message, intent_result, session_id, doc_id)
        for event in fallback:
            yield event

    def _fallback_qa(
        self,
        message: str,
        intent_result: IntentResult,
        session_id: Optional[str],
        doc_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        yield {"type": "progress", "content": "正在检索知识库..."}
        try:
            pipeline_result = self.pipeline.query(message, session_id=session_id, mode="hybrid", doc_id=doc_id)
        except Exception as e:
            logger.warning(f"Fallback query failed: {e}")
            pipeline_result = {"contexts": [], "context_sources": []}

        if pipeline_result.get("contexts"):
            yield {"type": "progress", "content": "正在生成回答..."}
            try:
                partial = ""
                for chunk in self.pipeline.generate_answer_stream(message, pipeline_result):
                    partial += chunk
                    yield {"type": "token", "content": chunk}
                sources = pipeline_result.get("context_sources", [])
                yield {"type": "done", "sources": sources}
                return
            except Exception as e:
                try:
                    answer = self.pipeline.generate_answer(message, pipeline_result)
                    yield {"type": "full", "content": answer, "sources": pipeline_result.get("context_sources", [])}
                    return
                except Exception as e2:
                    logger.warning(f"Answer generation failed: {e2}")

        yield {"type": "progress", "content": "知识库中暂无相关内容，正在通过网络搜索获取信息..."}
        try:
            from utils.web_search import web_search, format_search_results
            results = web_search(message, max_results=5)
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            results = []

        if not results:
            yield {"type": "error", "content": "知识库中暂无相关内容，且网络搜索未获取到有效信息。请上传相关文档或换个问题试试。"}
            return

        web_context = format_search_results(results)
        yield {"type": "progress", "content": "正在根据网络搜索结果生成回答..."}
        try:
            prompt = f"""你是一个AI学习助手。请基于以下网络搜索结果回答用户的问题。

注意：
1. 严格基于搜索结果提供信息，不编造内容
2. 如果搜索结果信息不足，明确指出
3. 标注信息来源

[网络搜索结果]:
{web_context}

用户问题: {message}

请给出详细、准确的回答："""
            partial = ""
            for chunk in self.pipeline.llm_service.invoke_stream(prompt):
                partial += chunk
                yield {"type": "token", "content": chunk}
            yield {"type": "done", "sources": [{"level_title": f"网络: {r.get('title', '')}", "excerpt": r.get('content', '')[:200]} for r in results[:3]]}
        except Exception as e:
            try:
                answer = self.llm_func(prompt)
                yield {"type": "full", "content": answer, "sources": []}
            except Exception as e2:
                yield {"type": "error", "content": f"回答生成失败: {e2}"}

    def _handle_tutor_intent(
        self,
        message: str,
        intent_result: IntentResult,
        user_id: str,
    ) -> Generator[Dict[str, Any], None, None]:
        try:
            path = self.pipeline.tutor_agent.generate_learning_path(
                goal=intent_result.topic or message, user_id=user_id
            )
            if path:
                answer = "## 学习路径建议\n\n"
                for i, step in enumerate(path[:5]):
                    title = step.get("title", step.get("concept", f"步骤{i+1}"))
                    desc = step.get("description", step.get("definition", ""))
                    answer += f"### {i+1}. {title}\n{desc}\n\n"
                yield {"type": "full", "content": answer, "sources": []}
            else:
                yield {"type": "progress", "content": "正在为你制定学习计划..."}
                prompt = f"""用户希望获得关于「{intent_result.topic or message}」的学习建议。
请给出一个结构化的学习计划，包括学习路线、推荐资源和关键里程碑。"""
                partial = ""
                for chunk in self.pipeline.llm_service.invoke_stream(prompt):
                    partial += chunk
                    yield {"type": "token", "content": chunk}
                yield {"type": "done", "sources": []}
        except Exception as e:
            logger.warning(f"Tutor intent handler failed: {e}")
            yield {"type": "error", "content": "无法生成学习建议，请稍后重试"}

    def _handle_explain_intent(
        self,
        message: str,
        intent_result: IntentResult,
        session_id: Optional[str],
        doc_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        yield {"type": "progress", "content": "正在检索知识库，准备通俗解释..."}
        try:
            pipeline_result = self.pipeline.query(
                intent_result.topic or message, session_id=session_id, mode="hybrid", doc_id=doc_id
            )
        except Exception as e:
            logger.warning(f"Explain intent retrieval failed: {e}")
            pipeline_result = {"contexts": [], "context_sources": []}

        context_text = ""
        if pipeline_result.get("contexts"):
            sources = pipeline_result.get("context_sources", [])
            context_parts = []
            for i, ctx in enumerate(sources[:5]):
                content = ctx.get("excerpt", ctx.get("content", ""))
                title = ctx.get("level_title", ctx.get("title", ""))
                context_parts.append(f"[来源 {i+1}] {title}\n{content}")
            context_text = "\n\n---\n\n".join(context_parts)

        tone = ""
        if intent_result.sub_type == "analogy":
            tone = "请使用生动的比喻或类比来解释，让抽象的概念变得具体易懂。"
        elif intent_result.sub_type == "simplify":
            tone = "请用最简单直白的语言解释，避免使用专业术语，就像在向一个完全不懂的人解释。"
        else:
            tone = "请结合具体例子进行解释，让概念更易于理解。"

        prompt = f"""你是一个擅长通俗解释复杂概念的AI老师。请用以下风格解释用户的疑问。

{tone}

{chr(10) + '[参考资料]:' + chr(10) + context_text if context_text else '如果没有参考资料，请基于你自己的知识进行解释。'}

用户问题: {message}

请给出清晰、易懂的解释："""

        try:
            partial = ""
            for chunk in self.pipeline.llm_service.invoke_stream(prompt):
                partial += chunk
                yield {"type": "token", "content": chunk}
            yield {"type": "done", "sources": pipeline_result.get("context_sources", [])}
        except Exception as e:
            try:
                answer = self.llm_func(prompt)
                yield {"type": "full", "content": answer, "sources": pipeline_result.get("context_sources", [])}
            except Exception as e2:
                yield {"type": "error", "content": "解释生成失败"}

    def _handle_compare_intent(
        self,
        message: str,
        intent_result: IntentResult,
        session_id: Optional[str],
        doc_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        yield {"type": "progress", "content": "正在检索相关信息，准备对比分析..."}
        try:
            pipeline_result = self.pipeline.query(
                intent_result.topic or message, session_id=session_id, mode="hybrid", doc_id=doc_id
            )
        except Exception as e:
            logger.warning(f"Compare intent retrieval failed: {e}")
            pipeline_result = {"contexts": [], "context_sources": []}

        context_text = ""
        if pipeline_result.get("contexts"):
            sources = pipeline_result.get("context_sources", [])
            context_parts = []
            for i, ctx in enumerate(sources[:5]):
                content = ctx.get("excerpt", ctx.get("content", ""))
                title = ctx.get("level_title", ctx.get("title", ""))
                context_parts.append(f"[来源 {i+1}] {title}\n{content}")
            context_text = "\n\n---\n\n".join(context_parts)

        prompt = f"""请对用户提出的问题进行全面的对比分析。

对比分析应包括：
1. 各自的核心定义和原理
2. 主要区别和相似之处
3. 各自的优缺点
4. 适用场景建议

{chr(10) + '[参考资料]:' + chr(10) + context_text if context_text else '如果没有参考资料，请基于你自己的知识进行对比。'}

用户问题: {message}

请用结构化的方式呈现对比结果："""

        try:
            partial = ""
            for chunk in self.pipeline.llm_service.invoke_stream(prompt):
                partial += chunk
                yield {"type": "token", "content": chunk}
            yield {"type": "done", "sources": pipeline_result.get("context_sources", [])}
        except Exception as e:
            try:
                answer = self.llm_func(prompt)
                yield {"type": "full", "content": answer, "sources": pipeline_result.get("context_sources", [])}
            except Exception as e2:
                yield {"type": "error", "content": "对比分析生成失败"}

    @staticmethod
    def _has_knowledge_intent(msg: str) -> bool:
        keywords = {"什么", "怎么", "为什么", "如何", "区别",
                    "对比", "原理", "定义", "概念", "是啥",
                    "含义", "作用", "好处", "缺点", "应用",
                    "实现", "推导", "证明", "公式", "例子",
                    "what", "how", "why", "difference",
                    "principle", "definition", "example"}
        return any(kw in msg for kw in keywords)

    @staticmethod
    def _intent_label(intent) -> str:
        labels = {
            "qa": "问答", "quiz": "出题", "summary": "总结", "review": "复习",
            "chat": "聊天", "tutor": "学习建议", "explain": "通俗解释", "compare": "对比",
        }
        return labels.get(intent.label, "处理")

    @staticmethod
    def _parse_json_response(response: str) -> Optional[Dict]:
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass
        try:
            start = clean.index("{")
            end = clean.rindex("}") + 1
            return json.loads(clean[start:end])
        except (ValueError, json.JSONDecodeError):
            return None
