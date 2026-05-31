import json
import time
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from core.database import DatabaseManager, get_database


class ReminderType(str, Enum):
    REVIEW = "review"
    PLAN_TASK = "plan_task"
    DAILY_GOAL = "daily_goal"
    STREAK = "streak"
    CUSTOM = "custom"


class ReminderStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    SNOOZED = "snoozed"
    DISMISSED = "dismissed"


@dataclass
class Reminder:
    reminder_id: str = ""
    user_id: str = "default"
    reminder_type: ReminderType = ReminderType.CUSTOM
    title: str = ""
    message: str = ""
    status: ReminderStatus = ReminderStatus.PENDING
    trigger_at: float = 0.0
    snooze_until: float = 0.0
    reference_id: str = ""
    reference_type: str = ""
    priority: int = 5
    created_at: float = 0.0
    sent_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.reminder_id:
            self.reminder_id = f"rem_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = time.time()
        if not self.trigger_at:
            self.trigger_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reminder_id": self.reminder_id,
            "user_id": self.user_id,
            "reminder_type": self.reminder_type.value if isinstance(self.reminder_type, ReminderType) else self.reminder_type,
            "title": self.title,
            "message": self.message,
            "status": self.status.value if isinstance(self.status, ReminderStatus) else self.status,
            "trigger_at": self.trigger_at,
            "snooze_until": self.snooze_until,
            "reference_id": self.reference_id,
            "reference_type": self.reference_type,
            "priority": self.priority,
            "created_at": self.created_at,
            "sent_at": self.sent_at,
            "metadata": self.metadata,
        }


class LearningReminder:
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        progress_tracker=None,
        learning_planner=None,
        llm_func: Optional[Callable] = None,
        db_path: Optional[str] = None,
    ):
        if db is not None:
            self.db = db
        elif db_path is not None:
            self.db = get_database(db_path)
        else:
            self.db = get_database()
        self.progress_tracker = progress_tracker
        self.learning_planner = learning_planner
        self.llm_func = llm_func
        self._init_db()

    def _init_db(self):
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                reminder_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                reminder_type TEXT NOT NULL DEFAULT 'custom',
                title TEXT NOT NULL DEFAULT '',
                message TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                trigger_at REAL NOT NULL DEFAULT 0,
                snooze_until REAL NOT NULL DEFAULT 0,
                reference_id TEXT NOT NULL DEFAULT '',
                reference_type TEXT NOT NULL DEFAULT '',
                priority INTEGER NOT NULL DEFAULT 5,
                created_at REAL NOT NULL DEFAULT 0,
                sent_at REAL NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS reminder_preferences (
                user_id TEXT PRIMARY KEY,
                daily_study_time TEXT NOT NULL DEFAULT '09:00',
                review_time TEXT NOT NULL DEFAULT '20:00',
                reminder_enabled INTEGER NOT NULL DEFAULT 1,
                snooze_duration_minutes INTEGER NOT NULL DEFAULT 30,
                max_daily_reminders INTEGER NOT NULL DEFAULT 5,
                streak_threshold INTEGER NOT NULL DEFAULT 3,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_rem_user ON reminders(user_id)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_rem_status ON reminders(status)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_rem_trigger ON reminders(trigger_at)
        """)
        self.db.commit()

    def create_review_reminder(
        self,
        knowledge_node_id: str,
        title: str = "",
        user_id: str = "default",
        trigger_at: Optional[float] = None,
    ) -> Reminder:
        if not trigger_at and self.progress_tracker:
            record = self.progress_tracker._find_record(knowledge_node_id, user_id)
            if record:
                trigger_at = record.next_review_at
                title = title or f"复习: {record.title}"

        if not trigger_at:
            trigger_at = time.time() + 86400

        reminder = Reminder(
            user_id=user_id,
            reminder_type=ReminderType.REVIEW,
            title=title or f"复习知识点",
            message=self._generate_review_message(knowledge_node_id, title, user_id),
            trigger_at=trigger_at,
            reference_id=knowledge_node_id,
            reference_type="knowledge_node",
            priority=7,
        )
        self._save_reminder(reminder)
        return reminder

    def create_plan_reminder(
        self,
        plan_id: str,
        task_id: str = "",
        user_id: str = "default",
        trigger_at: Optional[float] = None,
    ) -> Reminder:
        task_title = ""
        if self.learning_planner:
            plan = self.learning_planner.get_plan(plan_id)
            if plan:
                for task in plan.tasks:
                    if not task_id or task.task_id == task_id:
                        task_title = task.title
                        break

        if not trigger_at:
            trigger_at = time.time() + 3600

        reminder = Reminder(
            user_id=user_id,
            reminder_type=ReminderType.PLAN_TASK,
            title=f"学习任务: {task_title}" if task_title else "学习计划提醒",
            message=self._generate_plan_message(plan_id, task_title, user_id),
            trigger_at=trigger_at,
            reference_id=plan_id,
            reference_type="learning_plan",
            priority=6,
            metadata={"task_id": task_id},
        )
        self._save_reminder(reminder)
        return reminder

    def create_daily_goal_reminder(
        self,
        user_id: str = "default",
        target_minutes: int = 60,
        trigger_at: Optional[float] = None,
    ) -> Reminder:
        if not trigger_at:
            prefs = self.get_preferences(user_id)
            trigger_at = self._next_occurrence(prefs.get("daily_study_time", "09:00"))

        reminder = Reminder(
            user_id=user_id,
            reminder_type=ReminderType.DAILY_GOAL,
            title="每日学习目标",
            message=f"今天的目标是学习 {target_minutes} 分钟，加油！",
            trigger_at=trigger_at,
            priority=4,
            metadata={"target_minutes": target_minutes},
        )
        self._save_reminder(reminder)
        return reminder

    def create_streak_reminder(
        self,
        user_id: str = "default",
        current_streak: int = 0,
    ) -> Reminder:
        prefs = self.get_preferences(user_id)
        threshold = prefs.get("streak_threshold", 3)

        if current_streak >= threshold:
            message = f"太棒了！你已经连续学习 {current_streak} 天，继续保持！"
        else:
            days_left = threshold - current_streak
            message = f"你已经学习了 {current_streak} 天，再坚持 {days_left} 天就能达成连续学习目标！"

        reminder = Reminder(
            user_id=user_id,
            reminder_type=ReminderType.STREAK,
            title="学习连续天数提醒",
            message=message,
            trigger_at=time.time(),
            priority=3,
            metadata={"current_streak": current_streak, "threshold": threshold},
        )
        self._save_reminder(reminder)
        return reminder

    def get_due_reminders(
        self,
        user_id: str = "default",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        now = time.time()
        cursor = self.db.execute(
            "SELECT reminder_id, user_id, reminder_type, title, message, status, "
            "trigger_at, snooze_until, reference_id, reference_type, priority, "
            "created_at, sent_at, metadata "
            "FROM reminders "
            "WHERE user_id = ? AND status = 'pending' AND trigger_at <= ? "
            "ORDER BY priority DESC, trigger_at ASC LIMIT ?",
            (user_id, now, limit),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_upcoming_reminders(
        self,
        user_id: str = "default",
        hours_ahead: int = 24,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        now = time.time()
        future = now + hours_ahead * 3600
        cursor = self.db.execute(
            "SELECT reminder_id, user_id, reminder_type, title, message, status, "
            "trigger_at, snooze_until, reference_id, reference_type, priority, "
            "created_at, sent_at, metadata "
            "FROM reminders "
            "WHERE user_id = ? AND status = 'pending' AND trigger_at > ? AND trigger_at <= ? "
            "ORDER BY trigger_at ASC LIMIT ?",
            (user_id, now, future, limit),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def mark_sent(self, reminder_id: str) -> bool:
        self.db.execute(
            "UPDATE reminders SET status = 'sent', sent_at = ? WHERE reminder_id = ?",
            (time.time(), reminder_id),
        )
        self.db.commit()
        return True

    def snooze(self, reminder_id: str, duration_minutes: int = 30) -> bool:
        snooze_until = time.time() + duration_minutes * 60
        self.db.execute(
            "UPDATE reminders SET status = 'snoozed', snooze_until = ? WHERE reminder_id = ?",
            (snooze_until, reminder_id),
        )
        self.db.commit()
        return True

    def dismiss(self, reminder_id: str) -> bool:
        self.db.execute(
            "UPDATE reminders SET status = 'dismissed' WHERE reminder_id = ?",
            (reminder_id,),
        )
        self.db.commit()
        return True

    def process_snoozed(self, user_id: str = "default") -> int:
        now = time.time()
        cursor = self.db.execute(
            "UPDATE reminders SET status = 'pending' "
            "WHERE user_id = ? AND status = 'snoozed' AND snooze_until <= ?",
            (user_id, now),
        )
        self.db.commit()
        return cursor.rowcount

    def auto_generate_reminders(self, user_id: str = "default") -> List[Reminder]:
        reminders = []

        if self.progress_tracker:
            due_reviews = self.progress_tracker.get_due_reviews(user_id, limit=5)
            for review in due_reviews:
                existing = self._find_reminder_by_reference(
                    review["knowledge_node_id"], "knowledge_node", user_id
                )
                if not existing:
                    reminder = self.create_review_reminder(
                        knowledge_node_id=review["knowledge_node_id"],
                        title=review.get("title", ""),
                        user_id=user_id,
                    )
                    reminders.append(reminder)

        if self.learning_planner:
            plans = self.learning_planner.list_plans(user_id, status=None)
            for plan_info in plans:
                if plan_info["status"] == "active":
                    plan = self.learning_planner.get_plan(plan_info["plan_id"])
                    if plan:
                        next_task = self.learning_planner.get_next_task(plan_info["plan_id"])
                        if next_task:
                            existing = self._find_reminder_by_reference(
                                plan_info["plan_id"], "learning_plan", user_id
                            )
                            if not existing:
                                reminder = self.create_plan_reminder(
                                    plan_id=plan_info["plan_id"],
                                    task_id=next_task.task_id,
                                    user_id=user_id,
                                )
                                reminders.append(reminder)

        return reminders

    def get_preferences(self, user_id: str = "default") -> Dict[str, Any]:
        cursor = self.db.execute(
            "SELECT daily_study_time, review_time, reminder_enabled, "
            "snooze_duration_minutes, max_daily_reminders, streak_threshold, metadata "
            "FROM reminder_preferences WHERE user_id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        if not row:
            return self._default_preferences()

        return {
            "daily_study_time": row[0],
            "review_time": row[1],
            "reminder_enabled": bool(row[2]),
            "snooze_duration_minutes": row[3],
            "max_daily_reminders": row[4],
            "streak_threshold": row[5],
            "metadata": json.loads(row[6]) if row[6] else {},
        }

    def update_preferences(self, user_id: str = "default", **kwargs) -> Dict[str, Any]:
        current = self.get_preferences(user_id)
        current.update(kwargs)
        self.db.execute(
            """INSERT OR REPLACE INTO reminder_preferences
            (user_id, daily_study_time, review_time, reminder_enabled,
             snooze_duration_minutes, max_daily_reminders, streak_threshold, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                current.get("daily_study_time", "09:00"),
                current.get("review_time", "20:00"),
                int(current.get("reminder_enabled", True)),
                current.get("snooze_duration_minutes", 30),
                current.get("max_daily_reminders", 5),
                current.get("streak_threshold", 3),
                json.dumps(current.get("metadata", {}), ensure_ascii=False),
            ),
        )
        self.db.commit()
        return current

    def get_reminder_stats(self, user_id: str = "default") -> Dict[str, Any]:
        cursor = self.db.execute(
            "SELECT status, COUNT(*) FROM reminders WHERE user_id = ? GROUP BY status",
            (user_id,),
        )
        status_counts = {row[0]: row[1] for row in cursor.fetchall()}

        cursor = self.db.execute(
            "SELECT reminder_type, COUNT(*) FROM reminders WHERE user_id = ? AND status = 'pending' GROUP BY reminder_type",
            (user_id,),
        )
        pending_by_type = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "status_distribution": status_counts,
            "pending_by_type": pending_by_type,
            "total_pending": status_counts.get("pending", 0),
            "total_sent": status_counts.get("sent", 0),
        }

    def _generate_review_message(self, knowledge_node_id: str, title: str, user_id: str) -> str:
        base_msg = f"是时候复习了: {title}" if title else "你有知识点需要复习"

        if self.progress_tracker:
            record = self.progress_tracker._find_record(knowledge_node_id, user_id)
            if record:
                mastery = record.mastery.value if hasattr(record.mastery, "value") else record.mastery
                base_msg += f"\n当前掌握程度: {mastery}"
                base_msg += f"\n已复习次数: {record.exposure_count}"
                if record.review_interval_days > 0:
                    base_msg += f"\n建议复习间隔: {record.review_interval_days:.1f}天"

        return base_msg

    def _generate_plan_message(self, plan_id: str, task_title: str, user_id: str) -> str:
        base_msg = "学习计划提醒"
        if self.learning_planner:
            progress = self.learning_planner.get_plan_progress(plan_id)
            if progress:
                base_msg = f"学习计划: {progress.get('title', '')}"
                base_msg += f"\n总进度: {progress.get('progress_pct', 0)}%"
                base_msg += f"\n已完成: {progress.get('completed', 0)}/{progress.get('total_tasks', 0)}"

        if task_title:
            base_msg += f"\n下一个任务: {task_title}"

        return base_msg

    def _next_occurrence(self, time_str: str) -> float:
        import datetime
        now = datetime.datetime.now()
        try:
            hour, minute = map(int, time_str.split(":"))
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += datetime.timedelta(days=1)
            return target.timestamp()
        except (ValueError, AttributeError):
            return time.time() + 86400

    def _find_reminder_by_reference(
        self, reference_id: str, reference_type: str, user_id: str
    ) -> Optional[Dict]:
        cursor = self.db.execute(
            "SELECT reminder_id FROM reminders "
            "WHERE reference_id = ? AND reference_type = ? AND user_id = ? AND status = 'pending'",
            (reference_id, reference_type, user_id),
        )
        row = cursor.fetchone()
        return {"reminder_id": row[0]} if row else None

    def _default_preferences(self) -> Dict[str, Any]:
        return {
            "daily_study_time": "09:00",
            "review_time": "20:00",
            "reminder_enabled": True,
            "snooze_duration_minutes": 30,
            "max_daily_reminders": 5,
            "streak_threshold": 3,
            "metadata": {},
        }

    def _save_reminder(self, reminder: Reminder):
        self.db.execute(
            """INSERT OR REPLACE INTO reminders
            (reminder_id, user_id, reminder_type, title, message, status,
             trigger_at, snooze_until, reference_id, reference_type, priority,
             created_at, sent_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                reminder.reminder_id,
                reminder.user_id,
                reminder.reminder_type.value if isinstance(reminder.reminder_type, ReminderType) else reminder.reminder_type,
                reminder.title,
                reminder.message,
                reminder.status.value if isinstance(reminder.status, ReminderStatus) else reminder.status,
                reminder.trigger_at,
                reminder.snooze_until,
                reminder.reference_id,
                reminder.reference_type,
                reminder.priority,
                reminder.created_at,
                reminder.sent_at,
                json.dumps(reminder.metadata, ensure_ascii=False),
            ),
        )
        self.db.commit()

    def _row_to_dict(self, row) -> Dict[str, Any]:
        return {
            "reminder_id": row[0],
            "user_id": row[1],
            "reminder_type": row[2],
            "title": row[3],
            "message": row[4],
            "status": row[5],
            "trigger_at": row[6],
            "snooze_until": row[7],
            "reference_id": row[8],
            "reference_type": row[9],
            "priority": row[10],
            "created_at": row[11],
            "sent_at": row[12],
            "metadata": json.loads(row[13]) if row[13] else {},
        }

    def close(self):
        pass  # DatabaseManager is shared, closed at app shutdown
