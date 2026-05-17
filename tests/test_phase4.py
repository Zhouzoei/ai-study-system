import os
import sys
import time
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.learning_reminder import LearningReminder, Reminder, ReminderType, ReminderStatus
from core.document_manager import DocumentManager, Document, DocStatus
from engines.learning_analytics import LearningAnalytics


def test_learning_reminder():
    print("\n=== LearningReminder ===")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        reminder = LearningReminder(db_path=db_path)

        r1 = reminder.create_review_reminder(
            knowledge_node_id="node_001",
            title="Python Basics",
            user_id="test_user",
            trigger_at=time.time() - 100,
        )
        assert r1.reminder_id
        assert r1.reminder_type == ReminderType.REVIEW
        assert r1.status == ReminderStatus.PENDING
        print(f"[PASS] create_review_reminder: {r1.title}")

        r2 = reminder.create_plan_reminder(
            plan_id="plan_001",
            task_id="task_001",
            user_id="test_user",
            trigger_at=time.time() - 50,
        )
        assert r2.reminder_type == ReminderType.PLAN_TASK
        print(f"[PASS] create_plan_reminder: {r2.title}")

        r3 = reminder.create_daily_goal_reminder(
            user_id="test_user",
            target_minutes=60,
            trigger_at=time.time() - 10,
        )
        assert r3.reminder_type == ReminderType.DAILY_GOAL
        print(f"[PASS] create_daily_goal_reminder: {r3.title}")

        r4 = reminder.create_streak_reminder(
            user_id="test_user",
            current_streak=5,
        )
        assert r4.reminder_type == ReminderType.STREAK
        print(f"[PASS] create_streak_reminder: {r4.title}")

        due = reminder.get_due_reminders("test_user", limit=10)
        assert len(due) >= 3
        print(f"[PASS] get_due_reminders: {len(due)} due")

        first_due = due[0]
        reminder.mark_sent(first_due["reminder_id"])
        updated_due = reminder.get_due_reminders("test_user", limit=10)
        assert len(updated_due) == len(due) - 1
        print("[PASS] mark_sent")

        if updated_due:
            reminder.snooze(updated_due[0]["reminder_id"], duration_minutes=60)
            print("[PASS] snooze")

        if updated_due:
            reminder.dismiss(updated_due[0]["reminder_id"])
            print("[PASS] dismiss")

        prefs = reminder.get_preferences("test_user")
        assert "daily_study_time" in prefs
        assert "review_time" in prefs
        print(f"[PASS] get_preferences: {prefs}")

        updated_prefs = reminder.update_preferences("test_user", daily_study_time="10:00", streak_threshold=5)
        assert updated_prefs["daily_study_time"] == "10:00"
        assert updated_prefs["streak_threshold"] == 5
        print("[PASS] update_preferences")

        stats = reminder.get_reminder_stats("test_user")
        assert "status_distribution" in stats
        print(f"[PASS] get_reminder_stats: {stats}")

        reminder.close()
        print("[PASS] LearningReminder all tests passed")
    finally:
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            pass


def test_document_manager():
    print("\n=== DocumentManager ===")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        dm = DocumentManager(db_path=db_path)

        doc = dm.register_document(
            doc_id="doc_ml_101",
            title="Machine Learning 101",
            source="upload",
            content="Machine learning is a subset of AI...",
            file_type="markdown",
            tags=["ML", "AI", "basics"],
            description="Introduction to ML",
        )
        assert doc.doc_id == "doc_ml_101"
        assert doc.title == "Machine Learning 101"
        assert doc.version == 1
        assert doc.content_hash != ""
        print(f"[PASS] register_document: {doc.title}, version={doc.version}")

        same_doc = dm.register_document(
            doc_id="doc_ml_101",
            title="Machine Learning 101",
            content="Machine learning is a subset of AI...",
        )
        assert same_doc.version == 1
        print("[PASS] register same content returns same doc")

        updated_doc = dm.register_document(
            doc_id="doc_ml_101",
            title="Machine Learning 101 v2",
            content="Machine learning is a subset of AI. Deep learning is...",
        )
        assert updated_doc.version == 2
        print(f"[PASS] register updated content bumps version: v{updated_doc.version}")

        dm.update_document_stats("doc_ml_101", node_count=15, entity_count=8, relation_count=5)
        fetched = dm.get_document("doc_ml_101")
        assert fetched.node_count == 15
        assert fetched.entity_count == 8
        print("[PASS] update_document_stats")

        doc2 = dm.register_document(
            doc_id="doc_python_basics",
            title="Python Basics",
            content="Python is a programming language...",
            tags=["Python", "programming"],
        )

        all_docs = dm.list_documents()
        assert len(all_docs) >= 2
        print(f"[PASS] list_documents: {len(all_docs)} docs")

        ml_docs = dm.search_documents("Machine Learning")
        assert len(ml_docs) >= 1
        print(f"[PASS] search_documents: {len(ml_docs)} results")

        tagged = dm.list_documents(tag="ML")
        assert len(tagged) >= 1
        print(f"[PASS] list_documents by tag: {len(tagged)} docs")

        versions = dm.get_document_versions("doc_ml_101")
        assert len(versions) >= 2
        print(f"[PASS] get_document_versions: {len(versions)} versions")

        stats = dm.get_stats()
        assert stats["total_documents"] >= 2
        assert "ML" in stats["all_tags"]
        print(f"[PASS] get_stats: {stats}")

        dm.delete_document("doc_python_basics")
        deleted = dm.get_document("doc_python_basics")
        assert deleted is None
        print("[PASS] delete_document")

        dm.close()
        print("[PASS] DocumentManager all tests passed")
    finally:
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            pass


def test_learning_analytics():
    print("\n=== LearningAnalytics ===")

    from core.progress_tracker import ProgressTracker
    from core.learning_planner import LearningPlanner

    with tempfile.TemporaryDirectory() as tmpdir:
        from config import config as app_config
        old_tree_db = app_config.HIERARCHICAL_TREE_DB
        app_config.HIERARCHICAL_TREE_DB = os.path.join(tmpdir, "tree_store.db")

        try:
            from engines.pipeline import EnhancedRAGPipeline

            pipeline = EnhancedRAGPipeline()

            doc_text = """# Data Science Guide

## Statistics
Statistics is the science of collecting and analyzing data. Descriptive statistics summarizes data. Inferential statistics makes predictions.

## Data Processing
Data processing involves cleaning and transforming data. Pandas is popular for data manipulation. Data cleaning handles missing values.

## Machine Learning
Machine learning algorithms learn from data. Supervised learning uses labeled data. Common algorithms include regression and trees.
"""
            pipeline.ingest(doc_text, doc_id="ds_guide", title="Data Science Guide", tags=["DS", "ML"])

            tracker = pipeline.progress_tracker
            tracker.record_review("L3_nonexistent1", quality=4, user_id="test_user")
            tracker.record_review("L3_nonexistent2", quality=2, user_id="test_user")

            analytics = LearningAnalytics(
                progress_tracker=tracker,
                learning_planner=pipeline.learning_planner,
                knowledge_graph=pipeline.knowledge_graph,
                tree_storage=pipeline.storage,
                document_manager=pipeline.document_manager,
            )

            dashboard = analytics.get_learning_dashboard("test_user")
            assert "progress" in dashboard
            assert "knowledge_coverage" in dashboard
            assert "weaknesses" in dashboard
            assert "study_pattern" in dashboard
            assert "plan_summary" in dashboard
            print(f"[PASS] get_learning_dashboard")

            coverage = dashboard["knowledge_coverage"]
            assert "coverage_pct" in coverage
            assert "total_knowledge_nodes" in coverage
            print(f"[PASS] knowledge_coverage: {coverage['coverage_pct']}%")

            weaknesses = dashboard["weaknesses"]
            assert "weak_areas" in weaknesses
            assert "forgotten_areas" in weaknesses
            assert "never_reviewed" in weaknesses
            print(f"[PASS] weaknesses: weak={weaknesses['total_weak']}, forgotten={weaknesses['total_forgotten']}")

            pattern = dashboard["study_pattern"]
            assert "pattern" in pattern
            assert pattern["pattern"] in ["deep_learner", "balanced_learner", "broad_learner", "no_data"]
            print(f"[PASS] study_pattern: {pattern['pattern']}")

            recommendations = analytics.get_study_recommendations("test_user")
            assert isinstance(recommendations, list)
            print(f"[PASS] get_study_recommendations: {len(recommendations)} recommendations")

            kg_insights = analytics.get_knowledge_graph_insights("test_user")
            assert "total_entities" in kg_insights
            assert "hub_entities" in kg_insights
            print(f"[PASS] get_knowledge_graph_insights: {kg_insights['total_entities']} entities")

            pipeline.close()
            print("[PASS] LearningAnalytics all tests passed")
        finally:
            app_config.HIERARCHICAL_TREE_DB = old_tree_db


def test_pipeline_phase4_integration():
    print("\n=== Pipeline Phase 4 Integration ===")

    from engines.pipeline import EnhancedRAGPipeline
    from config import config as app_config

    tmpdir = tempfile.mkdtemp()
    old_tree_db = app_config.HIERARCHICAL_TREE_DB
    app_config.HIERARCHICAL_TREE_DB = os.path.join(tmpdir, "tree_store.db")

    try:
        pipeline = EnhancedRAGPipeline()

        doc_text = """# AI Handbook

## Chapter 1: AI Overview
Artificial intelligence is the simulation of human intelligence by machines. AI includes machine learning, natural language processing, and computer vision.

## Chapter 2: Neural Networks
Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes. Backpropagation is used for training.

## Chapter 3: Applications
AI applications include speech recognition, image classification, autonomous vehicles, and recommendation systems.
"""
        result = pipeline.ingest(doc_text, doc_id="ai_handbook", title="AI Handbook", tags=["AI", "ML"])
        assert result["total_nodes"] > 0
        print(f"[PASS] ingest with doc management: {result['total_nodes']} nodes")

        doc_info = pipeline.get_document("ai_handbook")
        assert doc_info is not None
        assert doc_info["title"] == "AI Handbook"
        print(f"[PASS] get_document: {doc_info['title']}")

        docs = pipeline.list_documents()
        assert len(docs) >= 1
        print(f"[PASS] list_documents: {len(docs)} docs")

        reminder = pipeline.create_review_reminder("L3_test_node", title="Test Review", user_id="test_user")
        assert "reminder_id" in reminder
        print(f"[PASS] create_review_reminder: {reminder['reminder_id']}")

        due_reminders = pipeline.get_due_reminders("test_user")
        print(f"[PASS] get_due_reminders: {len(due_reminders)} due")

        dashboard = pipeline.get_learning_dashboard("test_user")
        assert "progress" in dashboard
        assert "knowledge_coverage" in dashboard
        print(f"[PASS] get_learning_dashboard")

        recommendations = pipeline.get_study_recommendations("test_user")
        assert isinstance(recommendations, list)
        print(f"[PASS] get_study_recommendations: {len(recommendations)} items")

        kg_insights = pipeline.get_knowledge_graph_insights("test_user")
        assert "total_entities" in kg_insights
        print(f"[PASS] get_knowledge_graph_insights")

        stats = pipeline.get_stats()
        assert "documents" in stats
        assert "reminders" in stats
        print(f"[PASS] get_stats with Phase 4: docs={stats['documents']}, reminders={stats['reminders']}")

        pipeline.close()
        print("[PASS] Pipeline Phase 4 integration all tests passed")
    finally:
        app_config.HIERARCHICAL_TREE_DB = old_tree_db
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_learning_reminder()
    test_document_manager()
    test_learning_analytics()
    test_pipeline_phase4_integration()
    print("\n" + "=" * 50)
    print("ALL PHASE 4 TESTS PASSED!")
    print("=" * 50)
