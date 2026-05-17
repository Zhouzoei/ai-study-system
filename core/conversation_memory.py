import json
import sqlite3
import time
import uuid
import math
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from config import config


@dataclass
class Message:
    role: str = "user"
    content: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationSession:
    session_id: str = ""
    user_id: str = "default"
    title: str = ""
    messages: List[Message] = field(default_factory=list)
    summary: str = ""
    medium_summary: str = ""
    topics: List[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"sess_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()


@dataclass
class UserProfile:
    user_id: str = "default"
    interests: List[str] = field(default_factory=list)
    expertise: Dict[str, float] = field(default_factory=dict)
    interaction_count: int = 0
    total_sessions: int = 0
    last_active: float = 0.0
    preferred_topics: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "interests": self.interests,
            "expertise": self.expertise,
            "interaction_count": self.interaction_count,
            "total_sessions": self.total_sessions,
            "last_active": self.last_active,
            "preferred_topics": self.preferred_topics,
            "summary": self.summary,
        }


class ConversationMemory:
    def __init__(
        self,
        db_path: Optional[str] = None,
        max_window_size: int = 20,
        summary_threshold: int = 10,
        llm_func: Optional[Callable] = None,
        embed_func: Optional[Callable] = None,
        time_decay_factor: float = 0.5,
    ):
        self.db_path = db_path or config.HIERARCHICAL_TREE_DB.replace("tree_store", "conversation")
        self.max_window_size = max_window_size
        self.summary_threshold = summary_threshold
        self.llm_func = llm_func
        self.embed_func = embed_func
        self.time_decay_factor = time_decay_factor
        self._active_sessions: Dict[str, ConversationSession] = {}
        self._active_profiles: Dict[str, UserProfile] = {}
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                title TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                medium_summary TEXT NOT NULL DEFAULT '',
                topics TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding TEXT DEFAULT '',
                FOREIGN KEY (session_id) REFERENCES conversations(session_id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                interests TEXT NOT NULL DEFAULT '[]',
                expertise TEXT NOT NULL DEFAULT '{}',
                interaction_count INTEGER NOT NULL DEFAULT 0,
                total_sessions INTEGER NOT NULL DEFAULT 0,
                last_active REAL NOT NULL DEFAULT 0,
                preferred_topics TEXT NOT NULL DEFAULT '[]',
                summary TEXT NOT NULL DEFAULT ''
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS session_summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'medium',
                summary TEXT NOT NULL DEFAULT '',
                msg_range_start INTEGER NOT NULL DEFAULT 0,
                msg_range_end INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES conversations(session_id)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_summ_session ON session_summaries(session_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_msg_timestamp ON messages(timestamp)
        """)
        self.conn.commit()

    # ── Session Management ──────────────────────────────────────────

    def create_session(self, user_id: str = "default", title: str = "") -> ConversationSession:
        session = ConversationSession(
            user_id=user_id,
            title=title,
        )
        self._save_session(session)
        self._active_sessions[session.session_id] = session

        profile = self._get_or_create_profile(user_id)
        profile.total_sessions += 1
        self._save_profile(profile)

        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        session = self._load_session(session_id)
        if session:
            self._active_sessions[session_id] = session
        return session

    def delete_session(self, session_id: str):
        self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
        self.conn.commit()
        self._active_sessions.pop(session_id, None)

    # ── Message Operations ─────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Message:
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        msg = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        session.messages.append(msg)
        session.updated_at = time.time()

        embedding_json = self._compute_embedding(content)
        self._save_message(session_id, msg, embedding_json)
        self._update_session_timestamp(session_id)

        profile = self._get_or_create_profile(session.user_id)
        profile.interaction_count += 1
        profile.last_active = time.time()
        self._save_profile(profile)

        if len(session.messages) >= self.summary_threshold * 2:
            self._hierarchical_compress(session)

        return msg

    def _compute_embedding(self, text: str) -> str:
        if not self.embed_func or not text.strip():
            return ""
        try:
            vec = self.embed_func([text[:512]])[0]
            return json.dumps(vec)
        except Exception:
            return ""

    # ── Context Window ─────────────────────────────────────────────

    def get_context_window(
        self,
        session_id: str,
        window_size: Optional[int] = None,
        include_summary: bool = True,
        include_medium_summary: bool = False,
    ) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session:
            return []

        window_size = window_size or self.max_window_size
        recent = session.messages[-window_size:]

        context = []
        if include_summary and session.summary:
            context.append({
                "role": "system",
                "content": f"[长期摘要]: {session.summary}",
            })
        if include_medium_summary and session.medium_summary:
            context.append({
                "role": "system",
                "content": f"[近期摘要]: {session.medium_summary}",
            })

        for msg in recent:
            context.append(msg.to_dict())

        return context

    # ── Embedding-based Relevant History ────────────────────────────

    def get_relevant_history(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        use_time_decay: bool = True,
    ) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session or not session.messages:
            return []

        if self.embed_func:
            return self._embedding_based_retrieval(session, query, top_k, use_time_decay)

        return self._char_overlap_retrieval(session, query, top_k, use_time_decay)

    def _embedding_based_retrieval(
        self,
        session: ConversationSession,
        query: str,
        top_k: int,
        use_time_decay: bool,
    ) -> List[Dict[str, Any]]:
        candidates = [m for m in session.messages if m.role != "system"]

        query_vec = self.embed_func([query[:512]])[0]

        scored = []
        now = time.time()
        for msg in candidates:
            stored_emb = self._load_embedding(msg)
            if stored_emb:
                sim = self._cosine_similarity(query_vec, stored_emb)
            else:
                sim = 0.0

            if use_time_decay:
                age_hours = (now - msg.timestamp) / 3600
                decay = math.exp(-self.time_decay_factor * age_hours)
                sim *= decay

            scored.append((sim, msg))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [msg.to_dict() for _, msg in scored[:top_k]]

    def _char_overlap_retrieval(
        self,
        session: ConversationSession,
        query: str,
        top_k: int,
        use_time_decay: bool,
    ) -> List[Dict[str, Any]]:
        query_chars = set(query)
        scored = []
        now = time.time()
        for msg in session.messages:
            if msg.role == "system":
                continue
            overlap = len(query_chars & set(msg.content))
            score = overlap / max(len(query_chars), 1)

            if use_time_decay:
                age_hours = (now - msg.timestamp) / 3600
                decay = math.exp(-self.time_decay_factor * age_hours)
                score *= decay

            scored.append((score, msg))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [msg.to_dict() for _, msg in scored[:top_k]]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(av * bv for av, bv in zip(a, b))
        na = math.sqrt(sum(av * av for av in a))
        nb = math.sqrt(sum(bv * bv for bv in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _load_embedding(self, msg: Message) -> Optional[List[float]]:
        if not hasattr(msg, "_embedding_cache"):
            cursor = self.conn.execute(
                "SELECT embedding FROM messages WHERE timestamp = ? AND content = ?",
                (msg.timestamp, msg.content[:100]),
            )
            row = cursor.fetchone()
            if row and row[0]:
                msg._embedding_cache = json.loads(row[0])
            else:
                msg._embedding_cache = None
        return getattr(msg, "_embedding_cache", None)

    # ── User Profile ────────────────────────────────────────────────

    def get_user_profile(self, user_id: str = "default") -> Optional[Dict[str, Any]]:
        profile = self._get_or_create_profile(user_id)
        return profile.to_dict()

    def update_user_profile_interests(self, user_id: str, interests: List[str]):
        profile = self._get_or_create_profile(user_id)
        existing = set(profile.interests)
        for t in interests:
            existing.add(t)
        profile.interests = sorted(existing)[:20]
        self._save_profile(profile)

    def get_cross_session_context(
        self,
        user_id: str = "default",
        query: str = "",
        max_sessions: int = 3,
    ) -> List[Dict[str, Any]]:
        profile = self._get_or_create_profile(user_id)
        cursor = self.conn.execute(
            "SELECT session_id, title, summary, topics, updated_at "
            "FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, max_sessions + 1),
        )
        rows = cursor.fetchall()

        sessions = []
        for row in rows:
            sid = row[0]
            if sid in self._active_sessions:
                s = self._active_sessions[sid]
                sessions.append({
                    "session_id": sid,
                    "title": row[1],
                    "summary": s.summary or row[2],
                    "topics": s.topics or (json.loads(row[3]) if row[3] else []),
                    "updated_at": row[4],
                    "message_count": len(s.messages),
                })
            else:
                sessions.append({
                    "session_id": sid,
                    "title": row[1],
                    "summary": row[2],
                    "topics": json.loads(row[3]) if row[3] else [],
                    "updated_at": row[4],
                    "message_count": 0,
                })

        return {
            "user_id": user_id,
            "profile_summary": profile.summary,
            "interests": profile.interests,
            "expertise": profile.expertise,
            "recent_sessions": sessions,
        }

    # ── Topic Extraction ───────────────────────────────────────────

    def get_session_topics(self, session_id: str) -> List[str]:
        session = self.get_session(session_id)
        if not session:
            return []
        if session.topics:
            return session.topics

        if self.llm_func and session.messages:
            self._extract_topics(session)

        return session.topics

    # ── Hierarchical Summarization ─────────────────────────────────

    def _hierarchical_compress(self, session: ConversationSession):
        if len(session.messages) < self.summary_threshold * 2:
            return

        old_messages = session.messages[:self.summary_threshold]
        new_messages = session.messages[self.summary_threshold:]

        if self.llm_func:
            old_text = "\n".join(f"[{m.role}]: {m.content[:200]}" for m in old_messages)

            medium_prompt = f"""请用2-3句话总结以下对话的核心内容，突出关键知识点和结论：

{old_text}

近期摘要:"""
            try:
                new_medium = self.llm_func(medium_prompt)

                session.medium_summary = new_medium
                self._save_session_summary(
                    session.session_id, "medium", new_medium,
                    len(new_messages), len(new_messages) + len(old_messages),
                )

                all_summaries = self._get_all_medium_summaries(session.session_id)
                if len(all_summaries) >= 3:
                    combined = "\n".join(all_summaries[-3:])
                    long_prompt = f"""以下是一段长对话的多段摘要，请用3-5句话生成一个统一的长期摘要，保留所有重要知识点：

{combined}

长期摘要:"""
                    try:
                        new_long = self.llm_func(long_prompt)
                        session.summary = new_long
                        self._update_session_summary(session.session_id, session.summary)
                    except Exception:
                        pass
            except Exception:
                pass

        session.messages = new_messages

    def _get_all_medium_summaries(self, session_id: str) -> List[str]:
        cursor = self.conn.execute(
            "SELECT summary FROM session_summaries "
            "WHERE session_id = ? AND level = 'medium' ORDER BY created_at ASC",
            (session_id,),
        )
        return [row[0] for row in cursor.fetchall() if row[0]]

    # ── Listing ────────────────────────────────────────────────────

    def list_sessions(
        self,
        user_id: str = "default",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT session_id, title, summary, medium_summary, topics, created_at, updated_at "
            "FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "session_id": row[0],
                "title": row[1],
                "summary": row[2][:200] if row[2] else "",
                "medium_summary": row[3][:200] if row[3] else "",
                "topics": json.loads(row[4]) if row[4] else [],
                "created_at": row[4],
                "updated_at": row[5],
            }
            for row in rows
        ]

    # ── Stats ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        cursor = self.conn.execute("SELECT COUNT(*) FROM conversations")
        session_count = cursor.fetchone()[0]
        cursor = self.conn.execute("SELECT COUNT(*) FROM messages")
        msg_count = cursor.fetchone()[0]
        cursor = self.conn.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
        user_count = cursor.fetchone()[0]
        cursor = self.conn.execute("SELECT COUNT(*) FROM user_profiles")
        profile_count = cursor.fetchone()[0]
        cursor = self.conn.execute("SELECT COUNT(*) FROM session_summaries")
        summary_count = cursor.fetchone()[0]
        return {
            "total_sessions": session_count,
            "total_messages": msg_count,
            "total_users": user_count,
            "total_profiles": profile_count,
            "total_summaries": summary_count,
            "active_sessions": len(self._active_sessions),
        }

    # ── Internal Helpers ───────────────────────────────────────────

    def _get_or_create_profile(self, user_id: str) -> UserProfile:
        if user_id in self._active_profiles:
            return self._active_profiles[user_id]

        cursor = self.conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
        )
        row = cursor.fetchone()
        if row:
            profile = UserProfile(
                user_id=row[0],
                interests=json.loads(row[1]) if row[1] else [],
                expertise=json.loads(row[2]) if row[2] else {},
                interaction_count=row[3],
                total_sessions=row[4],
                last_active=row[5],
                preferred_topics=json.loads(row[6]) if row[6] else [],
                summary=row[7],
            )
        else:
            profile = UserProfile(user_id=user_id)
            self._save_profile(profile)

        self._active_profiles[user_id] = profile
        return profile

    def _save_profile(self, profile: UserProfile):
        self.conn.execute(
            """INSERT OR REPLACE INTO user_profiles
            (user_id, interests, expertise, interaction_count, total_sessions, last_active, preferred_topics, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                profile.user_id,
                json.dumps(profile.interests, ensure_ascii=False),
                json.dumps(profile.expertise, ensure_ascii=False),
                profile.interaction_count,
                profile.total_sessions,
                profile.last_active,
                json.dumps(profile.preferred_topics, ensure_ascii=False),
                profile.summary,
            ),
        )
        self.conn.commit()

    def _extract_topics(self, session: ConversationSession):
        if not self.llm_func or not session.messages:
            return

        recent = session.messages[-10:]
        text = "\n".join(f"[{m.role}]: {m.content[:100]}" for m in recent)
        prompt = f"""请从以下对话中提取3-5个关键主题词，用逗号分隔，只输出主题词：

{text}

主题词:"""

        try:
            response = self.llm_func(prompt)
            topics = [t.strip() for t in response.split(",") if t.strip()]
            session.topics = topics[:5]
            self._update_session_topics(session.session_id, session.topics)

            profile = self._get_or_create_profile(session.user_id)
            existing = set(profile.interests)
            for t in topics:
                existing.add(t)
            profile.interests = sorted(existing)[:20]
            self._save_profile(profile)
        except Exception:
            pass

    def _save_session_summary(
        self, session_id: str, level: str, summary: str,
        msg_start: int, msg_end: int,
    ):
        self.conn.execute(
            """INSERT INTO session_summaries
            (session_id, level, summary, msg_range_start, msg_range_end, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, level, summary, msg_start, msg_end, time.time()),
        )
        self.conn.commit()

    def _save_session(self, session: ConversationSession):
        self.conn.execute(
            """INSERT OR REPLACE INTO conversations
            (session_id, user_id, title, summary, medium_summary, topics, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session.session_id,
                session.user_id,
                session.title,
                session.summary,
                session.medium_summary,
                json.dumps(session.topics, ensure_ascii=False),
                session.created_at,
                session.updated_at,
            ),
        )
        self.conn.commit()

    def _load_session(self, session_id: str) -> Optional[ConversationSession]:
        cursor = self.conn.execute(
            "SELECT session_id, user_id, title, summary, medium_summary, topics, created_at, updated_at "
            "FROM conversations WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        session = ConversationSession(
            session_id=row[0],
            user_id=row[1],
            title=row[2],
            summary=row[3],
            medium_summary=row[4],
            topics=json.loads(row[5]) if row[5] else [],
            created_at=row[6],
            updated_at=row[7],
        )

        msg_cursor = self.conn.execute(
            "SELECT role, content, timestamp, metadata FROM messages "
            "WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        )
        session.messages = [
            Message(
                role=m[0],
                content=m[1],
                timestamp=m[2],
                metadata=json.loads(m[3]) if m[3] else {},
            )
            for m in msg_cursor.fetchall()
        ]

        return session

    def _save_message(self, session_id: str, msg: Message, embedding_json: str = ""):
        self.conn.execute(
            """INSERT INTO messages (session_id, role, content, timestamp, metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                msg.role,
                msg.content,
                msg.timestamp,
                json.dumps(msg.metadata, ensure_ascii=False),
                embedding_json,
            ),
        )
        self.conn.commit()

    def _update_session_timestamp(self, session_id: str):
        self.conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE session_id = ?",
            (time.time(), session_id),
        )
        self.conn.commit()

    def _update_session_summary(self, session_id: str, summary: str):
        self.conn.execute(
            "UPDATE conversations SET summary = ?, medium_summary = ? WHERE session_id = ?",
            (summary, summary, session_id),
        )
        self.conn.commit()

    def _update_session_topics(self, session_id: str, topics: List[str]):
        self.conn.execute(
            "UPDATE conversations SET topics = ? WHERE session_id = ?",
            (json.dumps(topics, ensure_ascii=False), session_id),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
