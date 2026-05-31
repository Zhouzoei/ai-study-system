import time
import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    OK = "ok"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_ERROR = "llm_error"
    RETRIEVAL_EMPTY = "retrieval_empty"
    RETRIEVAL_ERROR = "retrieval_error"
    KG_ERROR = "kg_error"
    KG_EMPTY = "kg_empty"
    NO_DOCUMENT = "no_document"
    SESSION_ERROR = "session_error"
    AGENT_ERROR = "agent_error"
    INTENT_LOW_CONFIDENCE = "intent_low_confidence"
    DEGRADED_MODE = "degraded_mode"


ERROR_USER_MESSAGES = {
    ErrorCode.LLM_UNAVAILABLE: "AI 服务暂不可用，已切换到纯检索模式",
    ErrorCode.LLM_TIMEOUT: "AI 响应超时，已切换到纯检索模式",
    ErrorCode.LLM_RATE_LIMIT: "AI 请求过于频繁，请稍后再试",
    ErrorCode.LLM_ERROR: "AI 服务出现异常，已切换到纯检索模式",
    ErrorCode.RETRIEVAL_EMPTY: "知识库中暂无相关内容，请先上传文档",
    ErrorCode.RETRIEVAL_ERROR: "检索服务异常，已降级到关键词搜索",
    ErrorCode.KG_ERROR: "知识图谱查询异常，已跳过图谱增强",
    ErrorCode.KG_EMPTY: "知识图谱暂无数据",
    ErrorCode.NO_DOCUMENT: "尚未导入任何文档，请先上传学习资料",
    ErrorCode.SESSION_ERROR: "会话异常，已自动恢复",
    ErrorCode.AGENT_ERROR: "智能助手处理异常，已降级处理",
    ErrorCode.INTENT_LOW_CONFIDENCE: "意图识别不确定，已按问答模式处理",
    ErrorCode.DEGRADED_MODE: "系统运行在降级模式下，部分功能受限",
}


@dataclass
class ServiceHealth:
    llm_available: bool = True
    llm_stream_available: bool = True
    retrieval_available: bool = True
    vector_available: bool = True
    kg_available: bool = True

    def get_degradation_level(self) -> str:
        if not self.llm_available:
            return "no_llm"
        if not self.vector_available:
            return "bm25_only"
        if not self.kg_available:
            return "no_kg"
        return "full"


@dataclass
class DegradedResult:
    answer: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    agent_type: str = ""
    error_code: ErrorCode = ErrorCode.OK
    degraded: bool = False
    degradation_note: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "agent_type": self.agent_type,
            "error_code": self.error_code.value,
            "degraded": self.degraded,
            "degradation_note": self.degradation_note,
            "metadata": self.metadata,
        }


class HealthChecker:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._health = ServiceHealth()
        self._llm_fail_count = 0
        self._llm_last_check = 0
        self._retrieval_fail_count = 0
        self._kg_fail_count = 0
        self._LLM_RETRY_THRESHOLD = 3
        self._RECOVERY_INTERVAL = 60
        self._llm_half_open = False
        self._retrieval_half_open = False
        self._kg_half_open = False

    @property
    def health(self) -> ServiceHealth:
        return self._health

    def check_llm(self) -> bool:
        if not self.pipeline.llm_func:
            self._health.llm_available = False
            self._health.llm_stream_available = False
            return False
        if self._llm_fail_count >= self._LLM_RETRY_THRESHOLD:
            now = time.time()
            if now - self._llm_last_check >= self._RECOVERY_INTERVAL:
                self._llm_half_open = True
                self._health.llm_available = True
                self._health.llm_stream_available = bool(self.pipeline.llm_service)
                logger.info("LLM entering half-open state, allowing probe request")
                return True
            self._health.llm_available = False
            self._health.llm_stream_available = False
            return False
        return self._health.llm_available

    def check_llm_stream(self) -> bool:
        if not self.pipeline.llm_service:
            self._health.llm_stream_available = False
            return False
        return self._health.llm_stream_available and self._health.llm_available

    def check_retrieval(self) -> bool:
        try:
            stats = self.pipeline.storage.get_stats()
            if stats.get("total_nodes", 0) == 0:
                self._health.retrieval_available = True
                self._health.vector_available = True
                return True
            return self._health.retrieval_available
        except Exception:
            self._health.retrieval_available = False
            return False

    def check_kg(self) -> bool:
        return self._health.kg_available

    def record_llm_success(self):
        self._llm_fail_count = 0
        self._llm_half_open = False
        self._health.llm_available = True
        self._health.llm_stream_available = bool(self.pipeline.llm_service)

    def record_llm_failure(self, error: Optional[Exception] = None):
        self._llm_fail_count += 1
        self._llm_last_check = time.time()
        if self._llm_half_open:
            self._llm_half_open = False
            logger.warning(f"LLM probe failed in half-open state, returning to closed: {error}")
        if self._llm_fail_count >= self._LLM_RETRY_THRESHOLD:
            self._health.llm_available = False
            self._health.llm_stream_available = False
            logger.warning(f"LLM marked unavailable after {self._llm_fail_count} failures: {error}")

    def record_retrieval_failure(self):
        self._retrieval_fail_count += 1
        if self._retrieval_fail_count >= 2:
            self._health.vector_available = False
            logger.warning("Vector retrieval marked unavailable, falling back to BM25")

    def record_retrieval_success(self):
        self._retrieval_fail_count = 0
        self._health.vector_available = True
        self._health.retrieval_available = True

    def record_kg_failure(self):
        self._kg_fail_count += 1
        if self._kg_fail_count >= 2:
            self._health.kg_available = False
            logger.warning("Knowledge graph marked unavailable")

    def record_kg_success(self):
        self._kg_fail_count = 0
        self._health.kg_available = True

    def classify_error(self, error: Exception) -> ErrorCode:
        error_msg = str(error).lower()
        if "timeout" in error_msg or "timed out" in error_msg:
            return ErrorCode.LLM_TIMEOUT
        if "rate" in error_msg or "429" in error_msg or "too many" in error_msg:
            return ErrorCode.LLM_RATE_LIMIT
        if "connection" in error_msg or "connect" in error_msg or "unavailable" in error_msg:
            return ErrorCode.LLM_UNAVAILABLE
        return ErrorCode.LLM_ERROR


def build_degraded_answer(
    agent_type: str,
    contexts: List[Dict],
    error_code: ErrorCode,
    topic: str = "",
) -> DegradedResult:
    note = ERROR_USER_MESSAGES.get(error_code, "服务异常，已降级处理")

    if not contexts:
        no_doc_note = ERROR_USER_MESSAGES.get(ErrorCode.NO_DOCUMENT, "")
        return DegradedResult(
            answer=no_doc_note,
            agent_type=agent_type,
            error_code=error_code if error_code != ErrorCode.LLM_ERROR else ErrorCode.NO_DOCUMENT,
            degraded=True,
            degradation_note=note,
        )

    sources = []
    for ctx in contexts[:5]:
        sources.append({
            "level_title": ctx.get("level_title", ctx.get("title", "")),
            "excerpt": ctx.get("excerpt", ctx.get("content", ""))[:300],
        })

    if agent_type == "qa":
        parts = []
        for i, ctx in enumerate(contexts[:5]):
            title = ctx.get("level_title", ctx.get("title", ""))
            content = ctx.get("excerpt", ctx.get("content", ""))
            parts.append(f"**{title}**\n\n{content}")
        answer = f"⚠️ {note}\n\n以下是与您问题最相关的原文内容：\n\n" + "\n\n---\n\n".join(parts)

    elif agent_type == "quiz":
        template = _build_template_quiz(contexts, topic)
        answer = f"⚠️ {note}\n\n以下是基于检索内容的简化题目：\n\n{template}"

    elif agent_type == "summary":
        parts = []
        for i, ctx in enumerate(contexts[:8]):
            title = ctx.get("level_title", ctx.get("title", ""))
            content = ctx.get("excerpt", ctx.get("content", ""))[:200]
            parts.append(f"- **{title}**: {content}")
        answer = f"⚠️ {note}\n\n以下是相关内容的要点列表：\n\n" + "\n".join(parts)

    elif agent_type == "review":
        items = []
        for ctx in contexts[:5]:
            title = ctx.get("level_title", ctx.get("title", ""))
            items.append(f"- {title}")
        answer = f"⚠️ {note}\n\n以下是需要复习的知识点：\n\n" + "\n".join(items)

    else:
        parts = []
        for i, ctx in enumerate(contexts[:3]):
            title = ctx.get("level_title", ctx.get("title", ""))
            content = ctx.get("excerpt", ctx.get("content", ""))
            parts.append(f"**{title}**\n\n{content}")
        answer = f"⚠️ {note}\n\n相关内容：\n\n" + "\n\n---\n\n".join(parts)

    return DegradedResult(
        answer=answer,
        sources=sources,
        agent_type=agent_type,
        error_code=error_code,
        degraded=True,
        degradation_note=note,
    )


def _build_template_quiz(contexts: List[Dict], topic: str) -> str:
    if not contexts:
        return "暂无足够内容生成题目"

    content = contexts[0].get("excerpt", contexts[0].get("content", ""))
    title = contexts[0].get("level_title", contexts[0].get("title", ""))

    sentences = [s.strip() for s in content.replace("。", "。\n").split("\n") if len(s.strip()) > 10]
    if not sentences:
        return f"关于「{topic or title}」的内容不足以生成题目"

    target = sentences[0]
    words = [w for w in target if '\u4e00' <= w <= '\u9fff']
    if len(words) >= 4:
        blank_word = words[len(words) // 2]
        question = target.replace(blank_word, "____")
        return f"## 填空题\n\n{question}\n\n## 答案\n\n{blank_word}\n\n## 来源\n\n{title}"

    return f"## 简答题\n\n请简述：{target}\n\n## 来源\n\n{title}"
