import json
import sqlite3
import time
import math
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from config import config


class MasteryLevel(str, Enum):
    UNKNOWN = "unknown"
    EXPOSED = "exposed"
    FAMILIAR = "familiar"
    PROFICIENT = "proficient"
    MASTERED = "mastered"


@dataclass
class KnowledgeRecord:
    record_id: str = ""
    user_id: str = "default"
    knowledge_node_id: str = ""
    title: str = ""
    mastery: MasteryLevel = MasteryLevel.UNKNOWN
    exposure_count: int = 0
    last_reviewed_at: float = 0.0
    next_review_at: float = 0.0
    review_interval_days: float = 1.0
    ease_factor: float = 2.5
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.record_id:
            self.record_id = f"kr_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_reviewed_at:
            self.last_reviewed_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "knowledge_node_id": self.knowledge_node_id,
            "title": self.title,
            "mastery": self.mastery.value if isinstance(self.mastery, MasteryLevel) else self.mastery,
            "exposure_count": self.exposure_count,
            "last_reviewed_at": self.last_reviewed_at,
            "next_review_at": self.next_review_at,
            "review_interval_days": self.review_interval_days,
            "ease_factor": self.ease_factor,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class ReviewEvent:
    event_id: str = ""
    record_id: str = ""
    user_id: str = "default"
    quality: int = 3
    old_mastery: str = "unknown"
    new_mastery: str = "unknown"
    old_interval: float = 1.0
    new_interval: float = 1.0
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"rev_{uuid.uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = time.time()


class ProgressTracker:
    def __init__(
        self,
        db_path: Optional[str] = None,
    ):
        self.db_path = db_path or config.HIERARCHICAL_TREE_DB.replace("tree_store", "progress")
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_records (
                record_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                knowledge_node_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                mastery TEXT NOT NULL DEFAULT 'unknown',
                exposure_count INTEGER NOT NULL DEFAULT 0,
                last_reviewed_at REAL NOT NULL DEFAULT 0,
                next_review_at REAL NOT NULL DEFAULT 0,
                review_interval_days REAL NOT NULL DEFAULT 1.0,
                ease_factor REAL NOT NULL DEFAULT 2.5,
                created_at REAL NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS review_events (
                event_id TEXT PRIMARY KEY,
                record_id TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT 'default',
                quality INTEGER NOT NULL DEFAULT 3,
                old_mastery TEXT NOT NULL DEFAULT 'unknown',
                new_mastery TEXT NOT NULL DEFAULT 'unknown',
                old_interval REAL NOT NULL DEFAULT 1.0,
                new_interval REAL NOT NULL DEFAULT 1.0,
                timestamp REAL NOT NULL DEFAULT 0
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kr_user ON knowledge_records(user_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kr_node ON knowledge_records(knowledge_node_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kr_next_review ON knowledge_records(next_review_at)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rev_record ON review_events(record_id)
        """)
        self.conn.commit()

    def record_exposure(
        self,
        knowledge_node_id: str,
        title: str = "",
        user_id: str = "default",
        metadata: Optional[Dict] = None,
    ) -> KnowledgeRecord:
        existing = self._find_record(knowledge_node_id, user_id)
        if existing:
            existing.exposure_count += 1
            existing.last_reviewed_at = time.time()
            if existing.mastery == MasteryLevel.UNKNOWN:
                existing.mastery = MasteryLevel.EXPOSED
            if metadata:
                existing.metadata.update(metadata)
            self._save_record(existing)
            return existing

        record = KnowledgeRecord(
            user_id=user_id,
            knowledge_node_id=knowledge_node_id,
            title=title,
            mastery=MasteryLevel.EXPOSED,
            exposure_count=1,
            next_review_at=time.time() + 86400,
            review_interval_days=1.0,
            metadata=metadata or {},
        )
        self._save_record(record)
        return record

    def record_review(
        self,
        knowledge_node_id: str,
        quality: int,
        user_id: str = "default",
    ) -> KnowledgeRecord:
        record = self._find_record(knowledge_node_id, user_id)
        if not record:
            record = KnowledgeRecord(
                user_id=user_id,
                knowledge_node_id=knowledge_node_id,
                mastery=MasteryLevel.EXPOSED,
                exposure_count=1,
            )

        old_mastery = record.mastery
        old_interval = record.review_interval_days

        record.exposure_count += 1
        record.last_reviewed_at = time.time()

        self._update_sm2(record, quality)
        self._update_mastery(record, quality)

        self._save_record(record)

        event = ReviewEvent(
            record_id=record.record_id,
            user_id=user_id,
            quality=quality,
            old_mastery=old_mastery.value if isinstance(old_mastery, MasteryLevel) else old_mastery,
            new_mastery=record.mastery.value if isinstance(record.mastery, MasteryLevel) else record.mastery,
            old_interval=old_interval,
            new_interval=record.review_interval_days,
        )
        self._save_event(event)

        return record

    def get_due_reviews(
        self,
        user_id: str = "default",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        now = time.time()
        cursor = self.conn.execute(
            "SELECT record_id, knowledge_node_id, title, mastery, "
            "last_reviewed_at, next_review_at, review_interval_days, exposure_count "
            "FROM knowledge_records "
            "WHERE user_id = ? AND next_review_at <= ? AND mastery NOT IN ('mastered') "
            "ORDER BY next_review_at ASC LIMIT ?",
            (user_id, now, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "record_id": row[0],
                "knowledge_node_id": row[1],
                "title": row[2],
                "mastery": row[3],
                "last_reviewed_at": row[4],
                "next_review_at": row[5],
                "review_interval_days": row[6],
                "exposure_count": row[7],
                "overdue_days": round((now - row[5]) / 86400, 1) if row[5] > 0 else 0,
            }
            for row in rows
        ]

    def get_upcoming_reviews(
        self,
        user_id: str = "default",
        days_ahead: int = 7,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        now = time.time()
        future = now + days_ahead * 86400
        cursor = self.conn.execute(
            "SELECT record_id, knowledge_node_id, title, mastery, next_review_at, review_interval_days "
            "FROM knowledge_records "
            "WHERE user_id = ? AND next_review_at > ? AND next_review_at <= ? "
            "ORDER BY next_review_at ASC LIMIT ?",
            (user_id, now, future, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "record_id": row[0],
                "knowledge_node_id": row[1],
                "title": row[2],
                "mastery": row[3],
                "next_review_at": row[4],
                "days_until": round((row[4] - now) / 86400, 1),
                "review_interval_days": row[5],
            }
            for row in rows
        ]

    def get_progress_summary(self, user_id: str = "default") -> Dict[str, Any]:
        cursor = self.conn.execute(
            "SELECT mastery, COUNT(*) FROM knowledge_records WHERE user_id = ? GROUP BY mastery",
            (user_id,),
        )
        mastery_counts = {row[0]: row[1] for row in cursor.fetchall()}

        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM knowledge_records WHERE user_id = ? AND next_review_at <= ?",
            (user_id, time.time()),
        )
        due_count = cursor.fetchone()[0]

        cursor = self.conn.execute(
            "SELECT COUNT(*), SUM(exposure_count), AVG(ease_factor) "
            "FROM knowledge_records WHERE user_id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        total = row[0]
        total_exposures = row[1] or 0
        avg_ease = row[2] or 2.5

        mastered_or_proficient = mastery_counts.get("mastered", 0) + mastery_counts.get("proficient", 0)
        progress_pct = round(mastered_or_proficient / total * 100, 1) if total > 0 else 0

        return {
            "user_id": user_id,
            "total_knowledge_nodes": total,
            "mastery_distribution": mastery_counts,
            "due_for_review": due_count,
            "total_exposures": total_exposures,
            "avg_ease_factor": round(avg_ease, 2),
            "progress_pct": progress_pct,
        }

    def get_knowledge_record(self, knowledge_node_id: str, user_id: str = "default") -> Optional[Dict]:
        record = self._find_record(knowledge_node_id, user_id)
        return record.to_dict() if record else None

    def get_review_history(
        self,
        knowledge_node_id: str,
        user_id: str = "default",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        record = self._find_record(knowledge_node_id, user_id)
        if not record:
            return []

        cursor = self.conn.execute(
            "SELECT quality, old_mastery, new_mastery, old_interval, new_interval, timestamp "
            "FROM review_events WHERE record_id = ? ORDER BY timestamp DESC LIMIT ?",
            (record.record_id, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "quality": row[0],
                "old_mastery": row[1],
                "new_mastery": row[2],
                "old_interval_days": row[3],
                "new_interval_days": row[4],
                "timestamp": row[5],
            }
            for row in rows
        ]

    def batch_record_exposure(
        self,
        node_ids: List[str],
        titles: Optional[Dict[str, str]] = None,
        user_id: str = "default",
    ) -> List[KnowledgeRecord]:
        titles = titles or {}
        records = []
        for nid in node_ids:
            r = self.record_exposure(nid, titles.get(nid, ""), user_id)
            records.append(r)
        return records

    def _update_sm2(self, record: KnowledgeRecord, quality: int):
        quality = max(0, min(5, quality))

        if quality >= 3:
            if record.exposure_count == 1:
                record.review_interval_days = 1.0
            elif record.exposure_count == 2:
                record.review_interval_days = 6.0
            else:
                record.review_interval_days = record.review_interval_days * record.ease_factor
        else:
            record.review_interval_days = 1.0
            record.exposure_count = 0

        record.ease_factor = max(1.3, record.ease_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))

        interval_seconds = record.review_interval_days * 86400
        record.next_review_at = time.time() + interval_seconds

    def _update_mastery(self, record: KnowledgeRecord, quality: int):
        mastery_order = [
            MasteryLevel.UNKNOWN,
            MasteryLevel.EXPOSED,
            MasteryLevel.FAMILIAR,
            MasteryLevel.PROFICIENT,
            MasteryLevel.MASTERED,
        ]
        current_idx = mastery_order.index(record.mastery) if record.mastery in mastery_order else 0

        if quality >= 4 and current_idx < len(mastery_order) - 1:
            record.mastery = mastery_order[current_idx + 1]
        elif quality == 3 and current_idx < len(mastery_order) - 2:
            record.mastery = mastery_order[current_idx + 1]
        elif quality <= 1 and current_idx > 0:
            record.mastery = mastery_order[current_idx - 1]

    def _find_record(self, knowledge_node_id: str, user_id: str) -> Optional[KnowledgeRecord]:
        cursor = self.conn.execute(
            "SELECT record_id, user_id, knowledge_node_id, title, mastery, "
            "exposure_count, last_reviewed_at, next_review_at, review_interval_days, "
            "ease_factor, created_at, metadata "
            "FROM knowledge_records WHERE knowledge_node_id = ? AND user_id = ?",
            (knowledge_node_id, user_id),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return KnowledgeRecord(
            record_id=row[0],
            user_id=row[1],
            knowledge_node_id=row[2],
            title=row[3],
            mastery=MasteryLevel(row[4]),
            exposure_count=row[5],
            last_reviewed_at=row[6],
            next_review_at=row[7],
            review_interval_days=row[8],
            ease_factor=row[9],
            created_at=row[10],
            metadata=json.loads(row[11]) if row[11] else {},
        )

    def _save_record(self, record: KnowledgeRecord):
        self.conn.execute(
            """INSERT OR REPLACE INTO knowledge_records
            (record_id, user_id, knowledge_node_id, title, mastery, exposure_count,
             last_reviewed_at, next_review_at, review_interval_days, ease_factor, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.record_id,
                record.user_id,
                record.knowledge_node_id,
                record.title,
                record.mastery.value if isinstance(record.mastery, MasteryLevel) else record.mastery,
                record.exposure_count,
                record.last_reviewed_at,
                record.next_review_at,
                record.review_interval_days,
                record.ease_factor,
                record.created_at,
                json.dumps(record.metadata, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def _save_event(self, event: ReviewEvent):
        self.conn.execute(
            """INSERT INTO review_events
            (event_id, record_id, user_id, quality, old_mastery, new_mastery, old_interval, new_interval, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.record_id,
                event.user_id,
                event.quality,
                event.old_mastery,
                event.new_mastery,
                event.old_interval,
                event.new_interval,
                event.timestamp,
            ),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
