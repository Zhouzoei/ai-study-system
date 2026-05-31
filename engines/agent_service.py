import logging
from typing import List, Dict, Any, Optional, Callable, Generator

from engines.intent_router import IntentRouter, IntentType, IntentResult
from engines.orchestrator import OrchestratorAgent
from engines.agents import QuizAgent, SummaryAgent, ReviewAgent
from engines.tutor_agent import TutorAgent
from engines.tool_registry import ToolRegistry, PromptManager
from engines.message_bus import MessageBus
from engines.resilience import HealthChecker
from utils.web_search import web_search

logger = logging.getLogger(__name__)


def _search_web_handler(query: str) -> Dict:
    results = web_search(query)
    if not results:
        return {"found": False, "message": "网络搜索无结果", "results": []}
    return {"found": True, "results": results}


class AgentService:
    """Encapsulates all agent, orchestrator, intent routing, and tool concerns."""

    def __init__(self, llm_func=None):
        self.llm_func = llm_func

        self.intent_router = IntentRouter(llm_func=llm_func)
        self.tool_registry = ToolRegistry()
        self.prompt_manager = PromptManager()
        self.event_bus = MessageBus()
        self.health_checker = HealthChecker(None)

        # Agents are wired after pipeline and services are available
        self.orchestrator = None
        self.quiz_agent = None
        self.summary_agent = None
        self.review_agent = None
        self.tutor_agent = None

    def wire(self, pipeline, progress_service, knowledge_service):
        self.health_checker.pipeline = pipeline

        self.quiz_agent = QuizAgent(pipeline=pipeline, llm_func=self.llm_func)
        self.summary_agent = SummaryAgent(pipeline=pipeline, llm_func=self.llm_func)
        self.review_agent = ReviewAgent(pipeline=pipeline, llm_func=self.llm_func)
        self.tutor_agent = TutorAgent(
            progress_tracker=progress_service.progress_tracker,
            knowledge_graph=knowledge_service.knowledge_graph,
            llm_func=self.llm_func,
        )
        self.orchestrator = OrchestratorAgent(pipeline=pipeline, llm_func=self.llm_func)

    def register_core_tools(self, pipeline):
        from engines.tool_registry import Tool

        self.tool_registry.register(Tool(
            name="query_knowledge", category="retrieval",
            description="从知识库中检索与问题相关的文档内容",
            parameters={"question": "用户的问题"},
            handler=lambda question=None, query=None, **kwargs: pipeline.query(
                question or query or kwargs.get("message", ""),
                use_hybrid=True, top_k=5,
            ),
        ))
        self.tool_registry.register(Tool(
            name="web_search", category="retrieval",
            description="当知识库中没有相关内容时，通过网络搜索获取实时信息",
            parameters={"query": "搜索关键词"},
            handler=lambda query, **kwargs: _search_web_handler(query),
        ))
        self.tool_registry.register(Tool(
            name="get_progress_summary", category="learning",
            description="获取用户的学习进度摘要",
            parameters={"user_id": "用户ID"},
            handler=lambda user_id="default", **kwargs: pipeline.progress.progress_tracker.get_progress_summary(user_id),
        ))
        self.tool_registry.register(Tool(
            name="get_weak_nodes", category="learning",
            description="获取用户的薄弱知识点列表",
            parameters={"user_id": "用户ID", "threshold": "答错次数阈值"},
            handler=lambda user_id="default", threshold=2, **kwargs: pipeline.progress.progress_tracker.get_weak_nodes(user_id, threshold),
        ))
        self.tool_registry.register(Tool(
            name="get_due_reviews", category="learning",
            description="获取待复习的知识点列表",
            parameters={"user_id": "用户ID", "limit": "最大数量"},
            handler=lambda user_id="default", limit=5, **kwargs: pipeline.progress.progress_tracker.get_due_reviews(user_id, limit),
        ))
        self.tool_registry.register(Tool(
            name="record_review", category="writing",
            description="记录对某个知识点的复习，更新掌握度",
            parameters={"knowledge_node_id": "知识点ID", "quality": "0-5的掌握评分", "user_id": "用户ID"},
            handler=lambda knowledge_node_id, quality=3, user_id="default", **kwargs: pipeline.progress.progress_tracker.record_review(knowledge_node_id, quality, user_id).to_dict(),
        ))
        self.tool_registry.register(Tool(
            name="create_review_reminder", category="writing",
            description="为某个知识点创建复习提醒",
            parameters={"knowledge_node_id": "知识点ID", "title": "知识点标题", "user_id": "用户ID"},
            handler=lambda knowledge_node_id, title="", user_id="default", **kwargs: pipeline.progress.learning_reminder.create_review_reminder(knowledge_node_id, title, user_id).to_dict() if pipeline.progress.learning_reminder else {},
        ))
        self.tool_registry.register(Tool(
            name="get_upcoming_reviews", category="learning",
            description="获取未来几天待复习的知识点",
            parameters={"user_id": "用户ID", "days_ahead": "未来天数", "limit": "最大数量"},
            handler=lambda user_id="default", days_ahead=7, limit=10, **kwargs: pipeline.progress.progress_tracker.get_upcoming_reviews(user_id, days_ahead, limit),
        ))
        self.tool_registry.register(Tool(
            name="write_note", category="writing",
            description="将重要内容写入用户的笔记本。当用户表达了值得记录的知识点、学习心得、关键理解时使用",
            parameters={"title": "笔记标题", "content": "笔记正文（markdown格式）", "summary": "一句话摘要（可选）", "tags": "标签列表，逗号分隔（可选）"},
            handler=lambda title, content="", summary="", tags="", **kwargs: pipeline.learner_model.write_note(
                title=title, content=content, summary=summary,
                tags=[t.strip() for t in tags.split(",") if t.strip()] if tags else [],
                source="conversation",
            ),
        ))
        self.tool_registry.register(Tool(
            name="list_notes", category="learning",
            description="列出用户的笔记本记录",
            parameters={"limit": "最大数量"},
            handler=lambda limit=10, **kwargs: pipeline.learner_model.list_notes(limit=limit),
        ))
        self.tool_registry.register(Tool(
            name="get_scope", category="learning",
            description="获取用户的学习范围文档（scope.md），了解用户掌握了什么、什么还薄弱",
            parameters={},
            handler=lambda **kwargs: pipeline.learner_model.get_l3_scope(),
        ))

        self._register_event_subscribers()

    def _register_event_subscribers(self):
        self.event_bus.subscribe("quiz:evaluated", self._on_quiz_evaluated)
        self.event_bus.subscribe("review:completed", self._on_review_completed)
        self.event_bus.subscribe("document:ingested", self._on_document_ingested)

    def _on_quiz_evaluated(self, event):
        payload = event.payload
        if not payload.get("is_correct") and payload.get("knowledge_node_ids"):
            logger.info(f"Quiz wrong answer recorded for: {payload.get('knowledge_node_ids')}")

    def _on_review_completed(self, event):
        payload = event.payload
        node_ids = payload.get("knowledge_node_ids", [])
        if node_ids:
            logger.debug(f"Review completed for nodes: {node_ids}")

    def _on_document_ingested(self, event):
        payload = event.payload
        doc_id = payload.get("doc_id", "")
        count = payload.get("node_count", 0)
        logger.info(f"Document ingested: {doc_id} ({count} nodes)")

    def render_prompt(self, name: str, **kwargs) -> str:
        return self.prompt_manager.render(name, **kwargs)

    def route_intent(self, message: str) -> IntentResult:
        return self.intent_router.route(message)

    def process_message(
        self, message: str, session_id=None, doc_id=None, user_id="default", mode="hybrid"
    ) -> Generator[Dict[str, Any], None, None]:
        for event in self.orchestrator.process(
            message, session_id=session_id, doc_id=doc_id,
            user_id=user_id, mode=mode,
        ):
            yield event

    def get_next_learning_steps(self, user_id="default", limit=5) -> List[Dict]:
        return self.tutor_agent.get_next_steps(user_id, limit) if self.tutor_agent else []

    def generate_learning_path(self, goal: str, user_id="default") -> List[Dict]:
        return self.tutor_agent.generate_learning_path(goal, user_id) if self.tutor_agent else []
