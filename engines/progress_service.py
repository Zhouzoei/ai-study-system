import logging
from typing import List, Dict, Any, Optional, Callable

from core.progress_tracker import ProgressTracker
from core.learning_planner import LearningPlanner
from core.learning_reminder import LearningReminder
from engines.learning_analytics import LearningAnalytics

logger = logging.getLogger(__name__)


class ProgressService:
    """Encapsulates progress tracking, learning plans, reminders, and analytics."""

    def __init__(self, llm_func=None):
        self.llm_func = llm_func

        self.progress_tracker = ProgressTracker()
        self.learning_planner = LearningPlanner(llm_func=llm_func)
        self.learning_reminder = None  # wired later
        self.analytics = None  # wired later

    def wire(self, tree_storage, document_manager, knowledge_graph):
        self.learning_planner.knowledge_graph = knowledge_graph
        self.learning_reminder = LearningReminder(
            progress_tracker=self.progress_tracker,
            learning_planner=self.learning_planner,
            llm_func=self.llm_func,
        )
        self.analytics = LearningAnalytics(
            progress_tracker=self.progress_tracker,
            learning_planner=self.learning_planner,
            knowledge_graph=knowledge_graph,
            tree_storage=tree_storage,
            document_manager=document_manager,
        )

    def on_document_ingested(self, l2_nodes, knowledge_units, kg_result, l3_nodes, doc_id):
        l2_titles = {n.node_id: n.title for n in l2_nodes}
        self.progress_tracker.batch_record_exposure(
            [n.node_id for n in l2_nodes], titles=l2_titles,
        )
        for unit in knowledge_units:
            self.progress_tracker.record_exposure(
                knowledge_node_id=unit.concept,
                title=unit.concept,
                metadata={
                    "bloom_level": unit.bloom_level,
                    "difficulty": unit.difficulty,
                    "source_nodes": unit.source_node_ids,
                    "keywords": unit.keywords,
                    "is_distilled": True,
                },
            )

    # -- Convenience wrappers --

    def record_review(self, knowledge_node_id: str, quality: int, user_id: str = "default") -> Dict:
        return self.progress_tracker.record_review(knowledge_node_id, quality, user_id).to_dict()

    def get_due_reviews(self, user_id: str = "default", limit: int = 10) -> List[Dict]:
        return self.progress_tracker.get_due_reviews(user_id, limit)

    def get_progress_summary(self, user_id: str = "default") -> Dict[str, Any]:
        return self.progress_tracker.get_progress_summary(user_id)

    def get_weak_nodes(self, user_id: str = "default", threshold: int = 2) -> List[Dict]:
        return self.progress_tracker.get_weak_nodes(user_id, threshold)

    def record_wrong_answer(self, question, user_answer, correct_answer, node_ids, user_id="default"):
        return self.progress_tracker.record_wrong_answer(question, user_answer, correct_answer, node_ids, user_id)

    def create_learning_plan(self, doc_id, l2_nodes, user_id="default", title="", daily_minutes=60) -> Dict:
        l2_dicts = [{"node_id": n.node_id, "title": n.title, "content": n.content} for n in l2_nodes]
        plan = self.learning_planner.create_plan_from_doc(doc_id, l2_dicts, user_id, title, daily_minutes)
        return plan.to_dict()

    def create_plan_from_goal(self, goal, user_id="default", daily_minutes=60, total_days=7) -> Dict:
        plan = self.learning_planner.create_plan_from_goal(goal, user_id, daily_minutes, total_days)
        return plan.to_dict()

    def get_due_reminders(self, user_id="default", limit=10) -> List[Dict]:
        return self.learning_reminder.get_due_reminders(user_id, limit) if self.learning_reminder else []

    def auto_generate_reminders(self, user_id="default") -> List[Dict]:
        if self.learning_reminder:
            return [r.to_dict() for r in self.learning_reminder.auto_generate_reminders(user_id)]
        return []

    def get_learning_dashboard(self, user_id="default") -> Dict:
        return self.analytics.get_learning_dashboard(user_id) if self.analytics else {}

    def get_study_recommendations(self, user_id="default") -> List[Dict]:
        return self.analytics.get_study_recommendations(user_id) if self.analytics else []

    def get_knowledge_graph_insights(self, user_id="default") -> Dict:
        return self.analytics.get_knowledge_graph_insights(user_id) if self.analytics else {}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "learning_progress": self.progress_tracker.get_progress_summary(),
            "reminders": self.learning_reminder.get_reminder_stats() if self.learning_reminder else {},
        }

    def close(self):
        self.progress_tracker.close()
        self.learning_planner.close()
        if self.learning_reminder:
            self.learning_reminder.close()
