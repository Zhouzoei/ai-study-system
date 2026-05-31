import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Event:
    event_type: str
    source: str
    payload: Dict[str, Any]
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "source": self.source,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


class MessageBus:
    def __init__(self, max_history: int = 100):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._history: List[Event] = []
        self._max_history = max_history

    def publish(self, event_type: str, source: str, payload: Optional[Dict[str, Any]] = None):
        event = Event(
            event_type=event_type,
            source=source,
            payload=payload or {},
        )
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        callbacks = self._subscribers.get(event_type, [])
        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.warning(f"Subscriber callback failed for {event_type}: {e}")

        wildcard_callbacks = self._subscribers.get("*", [])
        for cb in wildcard_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.warning(f"Wildcard subscriber callback failed: {e}")

        logger.debug(f"Event published: {event_type} from {source}")

    def subscribe(self, event_type: str, callback: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        if event_type in self._subscribers:
            self._subscribers[event_type] = [cb for cb in self._subscribers[event_type] if cb != callback]

    def get_recent_events(self, event_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return [e.to_dict() for e in events[-limit:]]

    def get_events_by_source(self, source: str, limit: int = 10) -> List[Dict[str, Any]]:
        events = [e for e in self._history if e.source == source]
        return [e.to_dict() for e in events[-limit:]]

    def get_session_events(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        events = [
            e for e in self._history
            if e.payload.get("session_id") == session_id
        ]
        return [e.to_dict() for e in events[-limit:]]

    def format_recent_context(self, limit: int = 5) -> str:
        events = self._history[-limit:]
        if not events:
            return ""
        lines = []
        for e in events:
            payload_summary = ""
            p = e.payload
            if "concept" in p:
                payload_summary = f"概念: {p['concept']}"
            elif "knowledge_node_id" in p:
                payload_summary = f"知识点: {p.get('knowledge_node_id', '')}"
            elif "is_correct" in p:
                payload_summary = f"{'正确' if p['is_correct'] else '错误'}"
            lines.append(f"[{e.source}] {e.event_type} {payload_summary}")
        return "\n".join(lines)

    def clear(self):
        self._history.clear()


# ── 预定义事件类型 ──

class EventTypes:
    QUIZ_EVALUATED = "quiz:evaluated"
    REVIEW_COMPLETED = "review:completed"
    DOCUMENT_INGESTED = "document:ingested"
    KNOWLEDGE_EXPOSED = "knowledge:exposed"
    WEAK_NODE_UPDATED = "weak_node:updated"
    PROGRESS_CHANGED = "progress:changed"
    SESSION_STARTED = "session:started"
    AGENT_ACTION = "agent:action"
    ERROR_OCCURRED = "error:occurred"
