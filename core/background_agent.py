import threading
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable, List

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 120
INACTIVE_THRESHOLD = 86400 * 2


class BackgroundAgent:
    def __init__(self, pipeline_getter: Callable):
        self.pipeline_getter = pipeline_getter
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_check = 0.0
        self._notifications: list = []
        self._max_notifications = 50
        self._user_ids: List[str] = ["default"]

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("BackgroundAgent started")

    def register_user(self, user_id: str):
        if user_id not in self._user_ids:
            self._user_ids.append(user_id)
            logger.info(f"BackgroundAgent registered user: {user_id}")

    def stop(self):
        self._running = False
        logger.info("BackgroundAgent stopped")

    def get_notifications(self, clear: bool = True) -> list:
        result = list(self._notifications)
        if clear:
            self._notifications.clear()
        return result

    def _add_notification(self, notification: dict):
        self._notifications.append(notification)
        if len(self._notifications) > self._max_notifications:
            self._notifications.pop(0)

    def _run_loop(self):
        while self._running:
            try:
                now = time.time()
                if now - self._last_check >= CHECK_INTERVAL:
                    self._tick()
                    self._last_check = now
            except Exception as e:
                logger.warning(f"BackgroundAgent tick failed: {e}")
            time.sleep(30)

    def _tick(self):
        pipeline = self.pipeline_getter()
        if not pipeline:
            return

        for user_id in self._user_ids:
            self._check_due_reviews(pipeline, user_id)
            self._check_inactivity(pipeline, user_id)
            self._auto_generate_reminders(pipeline, user_id)
            self._generate_daily_report(pipeline, user_id)

    def _check_due_reviews(self, pipeline, user_id: str):
        try:
            due = pipeline.get_due_reviews(user_id, limit=5)
            if not due:
                return
            titles = [d.get("title", "") for d in due[:3] if d.get("title")]
            if titles:
                self._add_notification({
                    "type": "review_due",
                    "severity": "warning",
                    "title": "待复习知识点",
                    "message": f"你有 {len(due)} 个知识点需要复习：{'、'.join(titles)}",
                    "count": len(due),
                })
        except Exception as e:
            logger.debug(f"BackgroundAgent check_due_reviews failed: {e}")

    def _check_inactivity(self, pipeline, user_id: str):
        try:
            profile = pipeline.conversation_memory.get_user_profile(user_id)
            if not profile:
                return
            last_active = profile.get("last_active", 0)
            if last_active <= 0:
                return
            elapsed = time.time() - last_active
            if elapsed > INACTIVE_THRESHOLD:
                days = int(elapsed / 86400)
                self._add_notification({
                    "type": "inactivity",
                    "severity": "info",
                    "title": "学习中断提醒",
                    "message": f"你已经 {days} 天没有学习了，回来继续吧",
                    "days_inactive": days,
                })
        except Exception as e:
            logger.debug(f"BackgroundAgent check_inactivity failed: {e}")

    def _auto_generate_reminders(self, pipeline, user_id: str):
        try:
            created = pipeline.auto_generate_reminders(user_id)
            if created:
                logger.info(f"Auto-generated {len(created)} reminders for {user_id}")
        except Exception as e:
            logger.debug(f"BackgroundAgent auto_generate_reminders failed: {e}")

    def _generate_daily_report(self, pipeline, user_id: str):
        try:
            now = time.time()
            local_today = datetime.now().astimezone()
            today_start = datetime(local_today.year, local_today.month, local_today.day, tzinfo=local_today.tzinfo).timestamp()
            if abs(now - today_start) > 120:
                return
            summary = pipeline.progress_tracker.get_progress_summary(user_id)
            total = summary.get("total_knowledge_nodes", 0)
            if total == 0:
                return
            mastered = summary.get("mastery_distribution", {}).get("mastered", 0)
            due = summary.get("due_for_review", 0)
            pct = summary.get("progress_pct", 0)
            self._add_notification({
                "type": "daily_report",
                "severity": "info",
                "title": "今日学习报告",
                "message": f"进度 {pct}% · 已掌握 {mastered}/{total} · 待复习 {due}",
            })
        except Exception as e:
            logger.debug(f"BackgroundAgent daily_report failed: {e}")
