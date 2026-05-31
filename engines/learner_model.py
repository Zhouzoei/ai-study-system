import json
import os
import re
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SURFACES = {"chat", "quiz", "kb", "notebook", "book", "tutorbot", "cowriter"}

USER_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_data")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class L1Event:
    event_id: str = ""
    ts: str = ""
    surface: str = ""
    kind: str = ""
    session_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps({
            "id": self.event_id,
            "ts": self.ts,
            "surface": self.surface,
            "kind": self.kind,
            "session_id": self.session_id,
            **self.data,
        }, ensure_ascii=False)


@dataclass
class NoteRecord:
    record_id: str = ""
    ts: str = ""
    title: str = ""
    content: str = ""
    summary: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = ""  # conversation / guided_learning / manual

    def to_dict(self) -> Dict:
        return {
            "record_id": self.record_id,
            "ts": self.ts,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "tags": self.tags,
            "source": self.source,
        }


class LearnerModel:
    def __init__(
        self,
        user_id: str = "default",
        llm_func: Optional[Callable] = None,
        base_dir: Optional[str] = None,
    ):
        self.user_id = user_id
        self.llm_func = llm_func
        self.base_dir = base_dir or os.path.join(USER_DATA_DIR, user_id)
        self._init_dirs()

        # In-memory scope cache
        self._scope_cache: Optional[str] = None
        self._scope_ts: float = 0
        self._scope_ttl: float = 300  # 5 minutes

    def _init_dirs(self):
        for s in SURFACES:
            _ensure_dir(os.path.join(self.base_dir, "l1", s))
        _ensure_dir(os.path.join(self.base_dir, "l2"))
        _ensure_dir(os.path.join(self.base_dir, "notebook"))
        _ensure_dir(os.path.join(self.base_dir, "l3"))

    # ═══════════════════════════════════
    # L1 — 原始事件记录
    # ═══════════════════════════════════

    def emit(
        self,
        surface: str,
        kind: str,
        data: Dict[str, Any],
        session_id: str = "",
    ) -> str:
        if surface not in SURFACES:
            logger.warning(f"Unknown surface: {surface}")
            return ""

        event = L1Event(
            event_id=f"{surface}:{uuid.uuid4().hex[:16]}",
            ts=_now_iso(),
            surface=surface,
            kind=kind,
            session_id=session_id,
            data=data,
        )

        filepath = os.path.join(self.base_dir, "l1", surface, f"{surface}.jsonl")
        _ensure_dir(os.path.dirname(filepath))

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(event.to_jsonl() + "\n")

        logger.debug(f"L1 event recorded: {event.event_id} [{surface}/{kind}]")
        return event.event_id

    def get_l1_events(
        self,
        surface: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        events = []
        surfaces = [surface] if surface else SURFACES
        for s in surfaces:
            filepath = os.path.join(self.base_dir, "l1", s, f"{s}.jsonl")
            if not os.path.exists(filepath):
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        if kind and ev.get("kind") != kind:
                            continue
                        ev["surface"] = s
                        events.append(ev)
                    except json.JSONDecodeError:
                        continue
        events.sort(key=lambda x: x.get("ts", ""), reverse=True)
        return events[:limit]

    # ═══════════════════════════════════
    # Notebook — 笔记系统
    # ═══════════════════════════════════

    def write_note(
        self,
        title: str = "",
        content: str = "",
        summary: str = "",
        tags: Optional[List[str]] = None,
        source: str = "manual",
        mode: str = "append",
        record_id: Optional[str] = None,
    ) -> Dict:
        if mode == "edit" and record_id:
            existing = self._load_note(record_id)
            if existing:
                if content:
                    existing["content"] = content
                if title:
                    existing["title"] = title
                if summary:
                    existing["summary"] = summary
                existing["ts"] = _now_iso()
                self._save_note(existing)
                self.emit("notebook", "edit", {"record_id": record_id, "title": title})
                return existing

        note = NoteRecord(
            record_id=f"note_{uuid.uuid4().hex[:12]}",
            ts=_now_iso(),
            title=title or f"笔记 {datetime.now().strftime('%m-%d %H:%M')}",
            content=content,
            summary=summary,
            tags=tags or [],
            source=source,
        )
        self._save_note(note.to_dict())
        self.emit("notebook", "create", {"record_id": note.record_id, "title": note.title, "source": source})
        return note.to_dict()

    def list_notes(self, limit: int = 20) -> List[Dict]:
        dirpath = os.path.join(self.base_dir, "notebook")
        if not os.path.exists(dirpath):
            return []
        notes = []
        for fname in sorted(os.listdir(dirpath), reverse=True)[:limit]:
            if fname.endswith(".json"):
                with open(os.path.join(dirpath, fname), "r", encoding="utf-8") as f:
                    notes.append(json.load(f))
        return notes

    def get_note(self, record_id: str) -> Optional[Dict]:
        return self._load_note(record_id)

    def _load_note(self, record_id: str) -> Optional[Dict]:
        path = os.path.join(self.base_dir, "notebook", f"{record_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_note(self, note: Dict):
        path = os.path.join(self.base_dir, "notebook", f"{note['record_id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(note, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════
    # L2 — 每个界面的 Markdown 摘要
    # ═══════════════════════════════════

    def generate_l2_summary(self, surface: str) -> str:
        if not self.llm_func:
            return ""
        events = self.get_l1_events(surface=surface, limit=50)
        if not events:
            return ""

        events_text = "\n".join(
            f"[{e.get('ts', '')[:19]}] {e.get('kind', '')}: {json.dumps({k: v for k, v in e.items() if k not in ('id','ts','surface','kind')}, ensure_ascii=False)}"
            for e in events[:30]
        )

        prompt = f"""你是一个学习分析助手。以下是用户在过去一段时间内「{surface}」界面的交互记录。请写一份 Markdown 摘要，分析用户的学习状态。

每次事件格式: [时间戳] 事件类型: 事件数据

事件记录:
{events_text}

请按以下格式输出：
## {surface} 摘要

### 学习活动
[列出用户在这个界面做了哪些主要活动]

### 掌握情况
[根据交互记录推断用户掌握了什么、什么还薄弱]

### 信号强度
[用 1-5 星评估该界面提供的信息置信度]

### 关键观察
[列出 2-3 条最有价值的发现]"""

        try:
            summary = self.llm_func(prompt)
            filepath = os.path.join(self.base_dir, "l2", f"{surface}.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(summary)
            return summary
        except Exception as e:
            logger.warning(f"L2 summary generation failed for {surface}: {e}")
            return ""

    def get_l2_summary(self, surface: str) -> str:
        filepath = os.path.join(self.base_dir, "l2", f"{surface}.md")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    # ═══════════════════════════════════
    # L3 — scope.md（用户知识范围综合文档）
    # ═══════════════════════════════════

    def generate_l3_scope(self, force: bool = False) -> str:
        if not self.llm_func:
            # Fallback: return SM-2 based summary
            return self._build_scope_fallback()

        # Collect L2 summaries for all surfaces that have data
        l2_sections = []
        for s in SURFACES:
            summary = self.get_l2_summary(s)
            if summary:
                l2_sections.append(summary)

        recent_events_text = ""
        recent = self.get_l1_events(limit=30)
        if recent:
            recent_events_text = "\n".join(
                f"[{e.get('surface', '?')}] [{e.get('ts', '')[:19]}] {e.get('kind', '')}: {json.dumps({k: v for k, v in e.items() if k not in ('id','ts','surface','kind','session_id')}, ensure_ascii=False)}"
                for e in recent[:20]
            )

        # Also get SM-2 data as reference
        mastery_data = self._get_mastery_fallback_data()

        prompt = f"""你是一个学习分析助手。请根据以下信息，生成用户的学习范围文档（scope.md）。

这份文档是系统判断用户知识掌握情况的核心依据。

"""

        separator = "\n\n"
        if l2_sections:
            prompt += f"""## 各界面摘要
{separator.join(l2_sections[:5])}

"""

        if recent_events_text:
            prompt += f"""## 近期事件记录（最近 30 条）
{recent_events_text}

"""

        if mastery_data:
            prompt += f"""## SM-2 掌握度参考数据
{mastery_data}

"""

        prompt += """请按以下格式输出 scope.md：

## 已掌握
[列出用户已经掌握的知识点。每条格式：**概念名**（置信度：高/中/低）— 简要证据说明]
每行末尾标注来源，如 [^quiz]、[^chat]、[^kb]。

## 学习中
[列出用户正在学习、尚未完全掌握的知识点。同上格式]

## 薄弱点
[列出用户反复出错、检索多次仍然不明白的知识点]

## 待探索
[基于文档内容，用户尚未接触但值得学习的知识点]

## 学习模式
[用户的学习习惯分析：偏好什么方式、在什么地方卡住、什么时间学习等]

---

*由 LearnerModel 于 """ + _now_iso()[:19] + """ 自动生成*
"""

        try:
            scope = self.llm_func(prompt)
        except Exception as e:
            logger.warning(f"L3 scope generation failed: {e}")
            scope = self._build_scope_fallback()

        filepath = os.path.join(self.base_dir, "l3", "scope.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(scope)

        self._scope_cache = scope
        self._scope_ts = time.time()
        return scope

    def get_l3_scope(self) -> str:
        if self._scope_cache and time.time() - self._scope_ts < self._scope_ttl:
            return self._scope_cache
        filepath = os.path.join(self.base_dir, "l3", "scope.md")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self._scope_cache = f.read()
                self._scope_ts = time.time()
                return self._scope_cache
        return ""

    def get_mastery(self, concept_name: str) -> Dict:
        """查询某个概念的掌握情况。先查 scope.md，查不到再看 SM-2."""
        scope = self.get_l3_scope()
        if scope and len(scope.strip()) > 50 and self.llm_func:
            try:
                resp = self.llm_func(
                    f"""根据以下学习范围文档，查询「{concept_name}」的掌握情况。

{scope}

请回答：
掌握状态: [已掌握/学习中/薄弱点/未涉及]
置信度: [高/中/低]
证据: [一句话说明判断依据]
来源: [主要来源 surface]"""
                )
                result = {"concept": concept_name, "raw": resp}
                for line in resp.split("\n"):
                    if "掌握状态" in line or "掌握状态" in line:
                        result["status"] = line.split(":")[-1].strip()
                    elif "置信度" in line:
                        result["confidence"] = line.split(":")[-1].strip()
                    elif "来源" in line:
                        result["source"] = line.split(":")[-1].strip()
                    elif "证据" in line:
                        result["evidence"] = line.split(":")[-1].strip()
                return result
            except Exception as e:
                logger.debug(f"Parse concept status failed for '{concept_name}': {e}")
        return {"concept": concept_name, "status": "unknown", "confidence": "低", "source": ""}

    # ═══════════════════════════════════
    # Fallback：当 LLM 不可用时，退回到 SM-2 规则
    # ═══════════════════════════════════

    def _build_scope_fallback(self) -> str:
        return f"""# 学习范围（系统生成）

## 已掌握
（LLM 不可用，无法综合分析）

## 学习中
（LLM 不可用，无法综合分析）

## 薄弱点
（LLM 不可用，无法综合分析）

*由 LearnerModel 于 {_now_iso()[:19]} 自动生成（降级模式）*
"""

    def _get_mastery_fallback_data(self) -> str:
        return "（SM-2 数据接口待接入）"

    # ═══════════════════════════════════
    # 便捷记录方法 — 各 Surface 专用
    # ═══════════════════════════════════

    def record_chat(self, message: str, role: str, session_id: str = ""):
        self.emit("chat", role, {"content": message[:500]}, session_id=session_id)

    def record_quiz_result(
        self,
        question: str,
        user_answer: str,
        correct_answer: str,
        is_correct: bool,
        concept: str = "",
        knowledge_node_ids: Optional[List[str]] = None,
        session_id: str = "",
    ):
        self.emit("quiz", "evaluate", {
            "question": question[:100],
            "user_answer": user_answer[:100],
            "correct_answer": correct_answer[:100],
            "is_correct": is_correct,
            "concept": concept or "",
            "knowledge_node_ids": knowledge_node_ids or [],
        }, session_id=session_id)

    def record_retrieval(self, query: str, num_results: int, session_id: str = ""):
        self.emit("kb", "retrieve", {
            "query": query[:200],
            "num_results": num_results,
        }, session_id=session_id)

    def record_reading(self, doc_title: str, node_title: str, duration_sec: float = 0):
        self.emit("book", "read", {
            "doc_title": doc_title[:100],
            "node_title": node_title[:100],
            "duration_sec": round(duration_sec, 1),
        })

    def record_tutorbot(self, concept: str, action: str, detail: str = ""):
        self.emit("tutorbot", action, {
            "concept": concept,
            "detail": detail[:300],
        })

    def record_notebook(self, record_id: str, title: str, action: str = "create"):
        self.emit("notebook", action, {
            "record_id": record_id,
            "title": title[:100],
        })
