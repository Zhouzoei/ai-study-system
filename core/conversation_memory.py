import json
import time
import uuid
import math
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from core.database import DatabaseManager, get_database


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

    def token_estimate(self) -> int:
        cn = sum(1 for c in self.content if '\u4e00' <= c <= '\u9fff')
        en_words = len(self.content.split())
        return cn * 2 + en_words


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
    compressed_count: int = 0

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


@dataclass
class UserPreference:
    pref_id: str = ""
    user_id: str = "default"
    category: str = ""
    content: str = ""
    embedding: List[float] = field(default_factory=list)
    weight: float = 1.0
    source: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    access_count: int = 0

    def __post_init__(self):
        if not self.pref_id:
            self.pref_id = f"pref_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()


class LayeredMemory:
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        short_window_tokens: int = 4000,
        short_window_messages: int = 20,
        medium_compress_threshold: int = 10,
        medium_summary_tokens: int = 500,
        long_summary_tokens: int = 800,
        llm_func: Optional[Callable] = None,
        embed_func: Optional[Callable] = None,
        time_decay_factor: float = 0.3,
        db_path: Optional[str] = None,
    ):
        if db is not None:
            self.db = db
        elif db_path is not None:
            self.db = get_database(db_path)
        else:
            self.db = get_database()
        self.short_window_tokens = short_window_tokens
        self.short_window_messages = short_window_messages
        self.medium_compress_threshold = medium_compress_threshold
        self.medium_summary_tokens = medium_summary_tokens
        self.long_summary_tokens = long_summary_tokens
        self.llm_func = llm_func
        self.embed_func = embed_func
        self.time_decay_factor = time_decay_factor
        self._active_sessions: Dict[str, ConversationSession] = {}
        self._active_profiles: Dict[str, UserProfile] = {}
        self._pref_cache: Dict[str, List[UserPreference]] = {}
        self._init_db()

    def _init_db(self):
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                title TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                medium_summary TEXT NOT NULL DEFAULT '',
                topics TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                compressed_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        self.db.execute("""
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
        self.db.execute("""
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
        self.db.execute("""
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
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                pref_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                category TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                embedding TEXT DEFAULT '',
                weight REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0,
                access_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_summ_session ON session_summaries(session_id)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_msg_timestamp ON messages(timestamp)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_pref_user ON user_preferences(user_id)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_pref_category ON user_preferences(user_id, category)
        """)
        self.db.commit()

    def set_llm_func(self, llm_func: Callable):
        self.llm_func = llm_func

    def set_embed_func(self, embed_func: Callable):
        self.embed_func = embed_func

    # ── Session Management ──────────────────────────────────────────

    def create_session(self, user_id: str = "default", title: str = "") -> ConversationSession:
        session = ConversationSession(user_id=user_id, title=title)
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
            self._active_sessions[session.session_id] = session
        return session

    def delete_session(self, session_id: str):
        self.db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self.db.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        self.db.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
        self.db.commit()
        self._active_sessions.pop(session_id, None)

    # ── Short-term: Sliding Window ─────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Message:
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(user_id="default", title=f"Session {session_id[:12]}")
            if session.session_id != session_id:
                self._active_sessions.pop(session.session_id, None)
                session.session_id = session_id
                self._active_sessions[session_id] = session
                self._save_session(session)

        msg = Message(role=role, content=content, timestamp=time.time(), metadata=metadata or {})
        session.messages.append(msg)
        session.updated_at = time.time()

        embedding_json = self._compute_embedding(content)
        self._save_message(session_id, msg, embedding_json)
        self._update_session_timestamp(session_id)

        profile = self._get_or_create_profile(session.user_id)
        profile.interaction_count += 1
        profile.last_active = time.time()
        self._save_profile(profile)

        self._extract_preference_from_message(session.user_id, msg)

        if len(session.messages) >= self.medium_compress_threshold * 2:
            self._compress_short_to_medium(session)

        return msg

    def get_short_term_context(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session:
            return []

        max_tokens = max_tokens or self.short_window_tokens
        max_messages = max_messages or self.short_window_messages

        recent = session.messages[-max_messages:]

        selected = []
        total_tokens = 0
        for msg in reversed(recent):
            est = msg.token_estimate()
            if total_tokens + est > max_tokens and selected:
                break
            selected.insert(0, msg)
            total_tokens += est

        return [msg.to_dict() for msg in selected]

    # ── Medium-term: Summary Compression ───────────────────────────

    def get_medium_term_context(
        self,
        session_id: str,
    ) -> Optional[str]:
        session = self.get_session(session_id)
        if not session:
            return None
        return session.medium_summary or None

    def _compress_short_to_medium(self, session: ConversationSession):
        if len(session.messages) < self.medium_compress_threshold * 2:
            return

        old_messages = session.messages[:self.medium_compress_threshold]
        new_messages = session.messages[self.medium_compress_threshold:]

        if self.llm_func:
            old_text = "\n".join(f"[{m.role}]: {m.content[:300]}" for m in old_messages)

            existing_medium = session.medium_summary or ""
            medium_prompt = f"""请将以下对话内容压缩为简洁的摘要，保留关键知识点、用户问题和系统回答的核心信息。

{f"已有近期摘要：{existing_medium}" if existing_medium else ""}

新增对话内容：
{old_text}

请输出更新后的近期摘要（不超过{self.medium_summary_tokens}字）："""

            try:
                new_medium = self.llm_func(medium_prompt)
                session.medium_summary = new_medium
                self._save_session_summary(
                    session.session_id, "medium", new_medium,
                    session.compressed_count * self.medium_compress_threshold,
                    (session.compressed_count + 1) * self.medium_compress_threshold,
                )

                all_summaries = self._get_all_medium_summaries(session.session_id)
                if len(all_summaries) >= 3:
                    self._compress_medium_to_long(session, all_summaries)
            except Exception as e:
                logger.warning(f"Medium compression failed for session {session.session_id}: {e}")

        session.messages = new_messages
        session.compressed_count += 1
        self._save_session(session)

    def _compress_medium_to_long(self, session: ConversationSession, medium_summaries: List[str]):
        if not self.llm_func:
            return

        combined = "\n\n".join(f"[摘要{i+1}]: {s}" for i, s in enumerate(medium_summaries[-5:]))
        existing_long = session.summary or ""

        long_prompt = f"""请将以下多段对话摘要整合为一个统一的长期摘要，保留所有重要知识点和结论。

{f"已有长期摘要：{existing_long}" if existing_long else ""}

近期摘要：
{combined}

请输出更新后的长期摘要（不超过{self.long_summary_tokens}字）："""

        try:
            new_long = self.llm_func(long_prompt)
            session.summary = new_long
            self._update_session_summary(session.session_id, session.summary)
        except Exception as e:
            logger.warning(f"Long compression failed for session {session.session_id}: {e}")

    # ── Long-term: Full Context with Layered Memory ────────────────

    def get_full_context(
        self,
        session_id: str,
        query: str = "",
        max_tokens: int = 6000,
        include_relevant_history: bool = True,
        include_user_preferences: bool = True,
    ) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session:
            return []

        context = []
        total_tokens = 0

        if session.summary:
            summary_dict = {
                "role": "system",
                "content": f"[长期记忆]: {session.summary}",
            }
            est = len(session.summary) * 2
            context.append(summary_dict)
            total_tokens += est

        if session.medium_summary:
            medium_dict = {
                "role": "system",
                "content": f"[近期记忆]: {session.medium_summary}",
            }
            est = len(session.medium_summary) * 2
            context.append(medium_dict)
            total_tokens += est

        if include_user_preferences and session.user_id:
            pref_context = self._get_preference_context(session.user_id, query)
            if pref_context:
                pref_dict = {
                    "role": "system",
                    "content": f"[用户偏好]: {pref_context}",
                }
                est = len(pref_context) * 2
                context.append(pref_dict)
                total_tokens += est

        if include_relevant_history and query and self.embed_func:
            relevant = self.get_relevant_history(session_id, query, top_k=3)
            for rh in relevant:
                est = len(rh.get("content", "")) * 2
                if total_tokens + est > max_tokens * 0.7:
                    break
                context.append(rh)
                total_tokens += est

        remaining_tokens = max_tokens - total_tokens
        if remaining_tokens > 200:
            short_ctx = self.get_short_term_context(
                session_id,
                max_tokens=remaining_tokens,
            )
            for msg_dict in short_ctx:
                context.append(msg_dict)

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
        if not candidates:
            return []

        vecs = self.embed_func([query[:512]])
        if not vecs or len(vecs) == 0:
            return []
        query_vec = vecs[0]

        scored = []
        now = time.time()
        for msg in candidates:
            stored_emb = self._load_embedding(msg)
            sim = self._cosine_similarity(query_vec, stored_emb) if stored_emb else 0.0

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

    # ── Long-term: User Preference Storage ─────────────────────────

    def add_preference(
        self,
        user_id: str,
        category: str,
        content: str,
        weight: float = 1.0,
        source: str = "conversation",
    ) -> UserPreference:
        existing = self._find_similar_preference(user_id, category, content)
        if existing:
            existing.weight = min(existing.weight + 0.2, 3.0)
            existing.access_count += 1
            existing.updated_at = time.time()
            self._save_preference(existing)
            return existing

        pref = UserPreference(
            user_id=user_id,
            category=category,
            content=content,
            weight=weight,
            source=source,
        )

        if self.embed_func:
            try:
                vecs = self.embed_func([content[:512]])
                if vecs and len(vecs) > 0:
                    pref.embedding = vecs[0]
            except Exception as e:
                logger.warning(f"Embedding preference failed: {e}")

        self._save_preference(pref)

        if user_id not in self._pref_cache:
            self._pref_cache[user_id] = []
        self._pref_cache[user_id].append(pref)

        return pref

    def get_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        prefs = self._load_preferences(user_id, category)

        prefs.sort(key=lambda p: p.weight * p.access_count, reverse=True)

        return [
            {
                "pref_id": p.pref_id,
                "category": p.category,
                "content": p.content,
                "weight": p.weight,
                "source": p.source,
                "access_count": p.access_count,
                "updated_at": p.updated_at,
            }
            for p in prefs[:limit]
        ]

    def get_personalized_recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        prefs = self._load_preferences(user_id)
        if not prefs:
            return []

        if self.embed_func:
            return self._embedding_preference_recall(prefs, query, top_k)
        return self._text_preference_recall(prefs, query, top_k)

    def _embedding_preference_recall(
        self,
        prefs: List[UserPreference],
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        try:
            vecs = self.embed_func([query[:512]])
            if not vecs or len(vecs) == 0:
                return self._text_preference_recall(prefs, query, top_k)
            query_vec = vecs[0]
        except Exception:
            return self._text_preference_recall(prefs, query, top_k)

        scored = []
        for pref in prefs:
            if pref.embedding:
                sim = self._cosine_similarity(query_vec, pref.embedding)
            else:
                sim = 0.0

            score = sim * pref.weight
            scored.append((score, pref))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, pref in scored[:top_k]:
            if score > 0.3:
                results.append({
                    "content": pref.content,
                    "category": pref.category,
                    "relevance_score": round(score, 3),
                    "weight": pref.weight,
                })

        return results

    def _text_preference_recall(
        self,
        prefs: List[UserPreference],
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        query_chars = set(query)
        scored = []
        for pref in prefs:
            overlap = len(query_chars & set(pref.content))
            score = (overlap / max(len(query_chars), 1)) * pref.weight
            scored.append((score, pref))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, pref in scored[:top_k]:
            if score > 0.1:
                results.append({
                    "content": pref.content,
                    "category": pref.category,
                    "relevance_score": round(score, 3),
                    "weight": pref.weight,
                })

        return results

    def _extract_preference_from_message(self, user_id: str, msg: Message):
        if msg.role != "user":
            return

        content = msg.content.strip()
        if len(content) < 4:
            return

        metadata = msg.metadata or {}
        intent = metadata.get("intent", "")

        if intent in ("quiz", "summary", "review"):
            topic = metadata.get("topic", "")
            if topic:
                self.add_preference(
                    user_id, "learning_topic", topic,
                    weight=1.5, source=f"intent:{intent}",
                )

        if any(kw in content for kw in ["我喜欢", "我偏好", "我更倾向", "重点看", "重点关注"]):
            self.add_preference(
                user_id, "explicit_preference", content[:200],
                weight=2.0, source="explicit",
            )

        if any(kw in content for kw in ["不懂", "不理解", "没搞懂", "困惑", "搞不清楚"]):
            self.add_preference(
                user_id, "weak_point", content[:200],
                weight=1.5, source="weakness_signal",
            )

    def _get_preference_context(self, user_id: str, query: str = "") -> str:
        if query:
            relevant = self.get_personalized_recall(user_id, query, top_k=3)
            if relevant:
                parts = [f"- {r['content']} ({r['category']}, 相关度: {r['relevance_score']})" for r in relevant]
                return "用户相关偏好:\n" + "\n".join(parts)

        prefs = self.get_preferences(user_id, limit=5)
        if not prefs:
            return ""

        categories = defaultdict(list)
        for p in prefs:
            categories[p["category"]].append(p["content"])

        parts = []
        for cat, items in categories.items():
            cat_name = {
                "learning_topic": "学习主题",
                "explicit_preference": "明确偏好",
                "weak_point": "薄弱环节",
            }.get(cat, cat)
            parts.append(f"{cat_name}: {', '.join(items[:3])}")

        return "用户偏好概览: " + "; ".join(parts)

    def _find_similar_preference(
        self, user_id: str, category: str, content: str,
    ) -> Optional[UserPreference]:
        prefs = self._load_preferences(user_id, category)
        content_lower = content.lower()

        for pref in prefs:
            if pref.content.lower() == content_lower:
                return pref

            if self.embed_func and pref.embedding:
                try:
                    content_vecs = self.embed_func([content[:512]])
                    if not content_vecs or len(content_vecs) == 0:
                        continue
                    content_vec = content_vecs[0]
                    sim = self._cosine_similarity(content_vec, pref.embedding)
                    if sim > 0.9:
                        return pref
                except Exception as e:
                    logger.debug(f"Preference similarity check failed: {e}")

        return None

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
    ) -> Dict[str, Any]:
        profile = self._get_or_create_profile(user_id)
        cursor = self.db.execute(
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

        pref_recall = []
        if query:
            pref_recall = self.get_personalized_recall(user_id, query, top_k=3)

        return {
            "user_id": user_id,
            "profile_summary": profile.summary,
            "interests": profile.interests,
            "expertise": profile.expertise,
            "recent_sessions": sessions,
            "relevant_preferences": pref_recall,
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

    # ── Listing ────────────────────────────────────────────────────

    def list_sessions(
        self,
        user_id: str = "default",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        cursor = self.db.execute(
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
                "created_at": row[5],
                "updated_at": row[6],
            }
            for row in rows
        ]

    # ── Stats ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        cursor = self.db.execute("SELECT COUNT(*) FROM conversations")
        session_count = cursor.fetchone()[0]
        cursor = self.db.execute("SELECT COUNT(*) FROM messages")
        msg_count = cursor.fetchone()[0]
        cursor = self.db.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
        user_count = cursor.fetchone()[0]
        cursor = self.db.execute("SELECT COUNT(*) FROM user_profiles")
        profile_count = cursor.fetchone()[0]
        cursor = self.db.execute("SELECT COUNT(*) FROM session_summaries")
        summary_count = cursor.fetchone()[0]
        cursor = self.db.execute("SELECT COUNT(*) FROM user_preferences")
        pref_count = cursor.fetchone()[0]
        return {
            "total_sessions": session_count,
            "total_messages": msg_count,
            "total_users": user_count,
            "total_profiles": profile_count,
            "total_summaries": summary_count,
            "total_preferences": pref_count,
            "active_sessions": len(self._active_sessions),
        }

    # ── Internal Helpers ───────────────────────────────────────────

    def _compute_embedding(self, text: str) -> str:
        if not self.embed_func or not text.strip():
            return ""
        try:
            vecs = self.embed_func([text[:512]])
            if not vecs or len(vecs) == 0:
                return ""
            return json.dumps(vecs[0])
        except Exception:
            return ""

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
            cursor = self.db.execute(
                "SELECT embedding FROM messages WHERE session_id = ? AND timestamp = ? AND content = ? LIMIT 1",
                (getattr(msg, '_session_id', ''), msg.timestamp, msg.content),
            )
            row = cursor.fetchone()
            if not row or not row[0]:
                cursor = self.db.execute(
                    "SELECT embedding FROM messages WHERE timestamp = ? AND content = ? LIMIT 1",
                    (msg.timestamp, msg.content),
                )
                row = cursor.fetchone()
            if row and row[0]:
                msg._embedding_cache = json.loads(row[0])
            else:
                msg._embedding_cache = None
        return getattr(msg, "_embedding_cache", None)

    def _get_or_create_profile(self, user_id: str) -> UserProfile:
        if user_id in self._active_profiles:
            return self._active_profiles[user_id]

        cursor = self.db.execute(
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
        self.db.execute(
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
        self.db.commit()

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
        except Exception as e:
            logger.warning(f"Update interests failed for user {session.user_id}: {e}")

    def _save_session_summary(
        self, session_id: str, level: str, summary: str,
        msg_start: int, msg_end: int,
    ):
        self.db.execute(
            """INSERT INTO session_summaries
            (session_id, level, summary, msg_range_start, msg_range_end, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, level, summary, msg_start, msg_end, time.time()),
        )
        self.db.commit()

    def _save_session(self, session: ConversationSession):
        self.db.execute(
            """INSERT OR REPLACE INTO conversations
            (session_id, user_id, title, summary, medium_summary, topics, created_at, updated_at, compressed_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session.session_id,
                session.user_id,
                session.title,
                session.summary,
                session.medium_summary,
                json.dumps(session.topics, ensure_ascii=False),
                session.created_at,
                session.updated_at,
                session.compressed_count,
            ),
        )
        self.db.commit()

    def _load_session(self, session_id: str) -> Optional[ConversationSession]:
        cursor = self.db.execute(
            "SELECT session_id, user_id, title, summary, medium_summary, topics, created_at, updated_at, compressed_count "
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
            compressed_count=row[8] if len(row) > 8 else 0,
        )

        msg_cursor = self.db.execute(
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
        msg._session_id = session_id
        self.db.execute(
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
        self.db.commit()

    def _update_session_timestamp(self, session_id: str):
        self.db.execute(
            "UPDATE conversations SET updated_at = ? WHERE session_id = ?",
            (time.time(), session_id),
        )
        self.db.commit()

    def _update_session_summary(self, session_id: str, summary: str):
        self.db.execute(
            "UPDATE conversations SET summary = ? WHERE session_id = ?",
            (summary, session_id),
        )
        self.db.commit()

    def _update_session_topics(self, session_id: str, topics: List[str]):
        self.db.execute(
            "UPDATE conversations SET topics = ? WHERE session_id = ?",
            (json.dumps(topics, ensure_ascii=False), session_id),
        )
        self.db.commit()

    def _get_all_medium_summaries(self, session_id: str) -> List[str]:
        cursor = self.db.execute(
            "SELECT summary FROM session_summaries "
            "WHERE session_id = ? AND level = 'medium' ORDER BY created_at ASC",
            (session_id,),
        )
        return [row[0] for row in cursor.fetchall() if row[0]]

    def _save_preference(self, pref: UserPreference):
        embedding_json = json.dumps(pref.embedding) if pref.embedding else ""
        self.db.execute(
            """INSERT OR REPLACE INTO user_preferences
            (pref_id, user_id, category, content, embedding, weight, source, created_at, updated_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pref.pref_id,
                pref.user_id,
                pref.category,
                pref.content,
                embedding_json,
                pref.weight,
                pref.source,
                pref.created_at,
                pref.updated_at,
                pref.access_count,
            ),
        )
        self.db.commit()

    def _load_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[UserPreference]:
        if category:
            cursor = self.db.execute(
                "SELECT pref_id, user_id, category, content, embedding, weight, source, created_at, updated_at, access_count "
                "FROM user_preferences WHERE user_id = ? AND category = ? ORDER BY weight DESC",
                (user_id, category),
            )
        else:
            cursor = self.db.execute(
                "SELECT pref_id, user_id, category, content, embedding, weight, source, created_at, updated_at, access_count "
                "FROM user_preferences WHERE user_id = ? ORDER BY weight DESC",
                (user_id,),
            )

        prefs = []
        for row in cursor.fetchall():
            embedding = json.loads(row[4]) if row[4] else []
            pref = UserPreference(
                pref_id=row[0],
                user_id=row[1],
                category=row[2],
                content=row[3],
                embedding=embedding,
                weight=row[5],
                source=row[6],
                created_at=row[7],
                updated_at=row[8],
                access_count=row[9],
            )
            prefs.append(pref)

        return prefs

    def close(self):
        pass  # DatabaseManager is shared, closed at app shutdown
