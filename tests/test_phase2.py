import os
import sys
import time
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.conversation_memory import ConversationMemory
from core.learning_planner import LearningPlanner, PlanStatus, TaskPriority, LearningTask, LearningPlan
from core.progress_tracker import ProgressTracker, MasteryLevel


def test_conversation_memory():
    print("\n=== ConversationMemory ===")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        mem = ConversationMemory(db_path=db_path, max_window_size=5, summary_threshold=3)

        session = mem.create_session(user_id="test_user", title="Test Session")
        assert session.session_id
        assert session.title == "Test Session"
        print("[PASS] create_session")

        sid = session.session_id

        mem.add_message(sid, "user", "Hello, I want to learn Python")
        mem.add_message(sid, "assistant", "Great! Let's start with basics.")
        mem.add_message(sid, "user", "What are variables?")
        mem.add_message(sid, "assistant", "Variables are containers for data.")
        print("[PASS] add_message x4")

        context = mem.get_context_window(sid)
        assert len(context) >= 4
        print(f"[PASS] get_context_window: {len(context)} messages")

        s = mem.get_session(sid)
        assert s is not None
        assert len(s.messages) == 4
        print("[PASS] get_session")

        mem.add_message(sid, "user", "How about loops?")
        mem.add_message(sid, "assistant", "Loops repeat code blocks.")
        mem.add_message(sid, "user", "And functions?")
        mem.add_message(sid, "assistant", "Functions are reusable code.")
        mem.add_message(sid, "user", "What about classes?")
        mem.add_message(sid, "assistant", "Classes are blueprints for objects.")

        s = mem.get_session(sid)
        assert len(s.messages) <= 5
        print(f"[PASS] auto compression: {len(s.messages)} messages after compression")

        stats = mem.get_stats()
        assert "total_sessions" in stats
        print(f"[PASS] get_stats: {stats}")

        mem.close()
        print("[PASS] ConversationMemory all tests passed")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_learning_planner():
    print("\n=== LearningPlanner ===")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        planner = LearningPlanner(db_path=db_path)

        l2_nodes = [
            {"node_id": "l2_1", "title": "Chapter 1: Intro", "content": "Introduction to the topic " * 20},
            {"node_id": "l2_2", "title": "Chapter 2: Basics", "content": "Basic concepts and definitions " * 30},
            {"node_id": "l2_3", "title": "Chapter 3: Advanced", "content": "Advanced topics and deep dive " * 40},
        ]

        plan = planner.create_plan_from_doc(
            doc_id="doc_test",
            l2_nodes=l2_nodes,
            user_id="test_user",
            title="Test Learning Plan",
        )
        assert plan.plan_id
        assert len(plan.tasks) == 3
        assert plan.status == PlanStatus.DRAFT
        print(f"[PASS] create_plan_from_doc: {len(plan.tasks)} tasks")

        plan_id = plan.plan_id

        retrieved = planner.get_plan(plan_id)
        assert retrieved is not None
        assert len(retrieved.tasks) == 3
        print("[PASS] get_plan")

        planner.update_plan_status(plan_id, PlanStatus.ACTIVE)
        active_plan = planner.get_plan(plan_id)
        assert active_plan.status == PlanStatus.ACTIVE
        print("[PASS] update_plan_status -> ACTIVE")

        first_task = active_plan.tasks[0]
        planner.update_task_status(plan_id, first_task.task_id, "completed")
        updated = planner.get_plan(plan_id)
        assert updated.tasks[0].status == "completed"
        print("[PASS] update_task_status -> completed")

        next_task = planner.get_next_task(plan_id)
        assert next_task is not None
        assert next_task.status == "pending"
        print(f"[PASS] get_next_task: {next_task.title}")

        schedule = planner.get_daily_schedule(plan_id, available_minutes=120)
        assert len(schedule) > 0
        print(f"[PASS] get_daily_schedule: {len(schedule)} tasks scheduled")

        progress = planner.get_plan_progress(plan_id)
        assert progress["total_tasks"] == 3
        assert progress["completed"] == 1
        print(f"[PASS] get_plan_progress: {progress['progress_pct']}%")

        plans = planner.list_plans(user_id="test_user")
        assert len(plans) >= 1
        print(f"[PASS] list_plans: {len(plans)} plans")

        goal_plan = planner.create_plan_from_goal(
            goal="Learn Machine Learning",
            user_id="test_user",
            daily_minutes=60,
            total_days=7,
        )
        assert goal_plan.plan_id
        assert len(goal_plan.tasks) >= 1
        print(f"[PASS] create_plan_from_goal: {len(goal_plan.tasks)} tasks")

        planner.delete_plan(goal_plan.plan_id)
        assert planner.get_plan(goal_plan.plan_id) is None
        print("[PASS] delete_plan")

        planner.close()
        print("[PASS] LearningPlanner all tests passed")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_progress_tracker():
    print("\n=== ProgressTracker ===")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        tracker = ProgressTracker(db_path=db_path)

        r1 = tracker.record_exposure("node_1", "Variables in Python")
        assert r1.mastery == MasteryLevel.EXPOSED
        assert r1.exposure_count == 1
        print("[PASS] record_exposure: first exposure")

        r1_again = tracker.record_exposure("node_1", "Variables in Python")
        assert r1_again.exposure_count == 2
        print("[PASS] record_exposure: second exposure increments count")

        r2 = tracker.record_exposure("node_2", "Loops in Python")
        r3 = tracker.record_exposure("node_3", "Functions in Python")
        print("[PASS] record_exposure: multiple nodes")

        reviewed = tracker.record_review("node_1", quality=4)
        assert reviewed.exposure_count == 3
        assert reviewed.mastery in [MasteryLevel.FAMILIAR, MasteryLevel.PROFICIENT]
        assert reviewed.next_review_at > time.time()
        print(f"[PASS] record_review: mastery={reviewed.mastery.value}, interval={reviewed.review_interval_days}d")

        poor_review = tracker.record_review("node_2", quality=1)
        assert poor_review.mastery == MasteryLevel.UNKNOWN
        assert poor_review.review_interval_days == 1.0
        print(f"[PASS] record_review (poor): mastery={poor_review.mastery.value}, interval reset")

        good_review = tracker.record_review("node_3", quality=5)
        assert good_review.mastery in [MasteryLevel.FAMILIAR, MasteryLevel.PROFICIENT]
        print(f"[PASS] record_review (excellent): mastery={good_review.mastery.value}")

        record = tracker.get_knowledge_record("node_1")
        assert record is not None
        assert record["knowledge_node_id"] == "node_1"
        print("[PASS] get_knowledge_record")

        summary = tracker.get_progress_summary()
        assert summary["total_knowledge_nodes"] == 3
        assert "mastery_distribution" in summary
        print(f"[PASS] get_progress_summary: {summary['progress_pct']}% progress")

        history = tracker.get_review_history("node_1")
        assert len(history) >= 1
        print(f"[PASS] get_review_history: {len(history)} events")

        tracker.record_review("node_3", quality=5)
        tracker.record_review("node_3", quality=5)
        mastered = tracker.record_review("node_3", quality=5)
        print(f"[PASS] mastery progression: {mastered.mastery.value}")

        batch = tracker.batch_record_exposure(
            ["batch_1", "batch_2", "batch_3"],
            titles={"batch_1": "Topic A", "batch_2": "Topic B", "batch_3": "Topic C"},
        )
        assert len(batch) == 3
        print("[PASS] batch_record_exposure")

        tracker.close()
        print("[PASS] ProgressTracker all tests passed")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_pipeline_integration():
    print("\n=== Pipeline Phase 2 Integration ===")

    from engines.pipeline import EnhancedRAGPipeline
    from config import config as app_config

    tmpdir = tempfile.mkdtemp()
    old_tree_db = app_config.HIERARCHICAL_TREE_DB
    app_config.HIERARCHICAL_TREE_DB = os.path.join(tmpdir, "tree_store.db")

    try:
        pipeline = EnhancedRAGPipeline()

        doc_text = """# Python Learning Guide

## Chapter 1: Introduction
Python is a high-level programming language. It is widely used in web development, data science, and artificial intelligence. Python's syntax is clean and readable, making it an excellent choice for beginners.

## Chapter 2: Variables and Data Types
Variables in Python are containers for storing data values. Python has several built-in data types including strings, integers, floats, and booleans. Type conversion is done using built-in functions like int(), float(), and str().

## Chapter 3: Control Flow
Control flow statements in Python include if-elif-else for conditional execution, and for/while loops for iteration. Python uses indentation to define code blocks, which makes the code more readable.

## Chapter 4: Functions
Functions in Python are defined using the def keyword. They can accept parameters and return values. Python supports default arguments, keyword arguments, and variable-length arguments.

## Chapter 5: Classes and Objects
Python is an object-oriented programming language. Classes are defined using the class keyword. Objects are instances of classes. Python supports inheritance, encapsulation, and polymorphism.
"""
        result = pipeline.ingest(doc_text, doc_id="python_guide")
        assert result["total_nodes"] > 0
        assert result["exposed_knowledge_nodes"] > 0
        print(f"[PASS] ingest: {result['total_nodes']} nodes, {result['exposed_knowledge_nodes']} knowledge nodes")

        session_info = pipeline.create_session(user_id="test_user", title="Python Learning")
        assert "session_id" in session_info
        print(f"[PASS] create_session: {session_info['session_id']}")

        sid = session_info["session_id"]

        q1 = pipeline.query("What are variables in Python?", session_id=sid)
        assert q1["num_contexts"] > 0
        print(f"[PASS] first query with session: {q1['num_contexts']} contexts (no history yet)")

        q2 = pipeline.query("How do functions work?", session_id=sid)
        assert q2["num_contexts"] > 0
        conv_ctx = q2.get("conversation_context", [])
        print(f"[PASS] second query with session: {q2['num_contexts']} contexts, conv_history={len(conv_ctx)}")

        plan_dict = pipeline.create_learning_plan(
            doc_id="python_guide",
            user_id="test_user",
            title="Python Study Plan",
        )
        assert "plan_id" in plan_dict
        assert len(plan_dict["tasks"]) > 0
        print(f"[PASS] create_learning_plan: {len(plan_dict['tasks'])} tasks")

        progress = pipeline.get_learning_progress(user_id="test_user")
        assert "total_knowledge_nodes" in progress
        print(f"[PASS] get_learning_progress: {progress['progress_pct']}%")

        stats = pipeline.get_stats()
        assert "conversation" in stats
        assert "learning_progress" in stats
        print(f"[PASS] get_stats with Phase 2 data")

        pipeline.close()
        print("[PASS] Pipeline Phase 2 integration all tests passed")
    finally:
        app_config.HIERARCHICAL_TREE_DB = old_tree_db
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_conversation_memory()
    test_learning_planner()
    test_progress_tracker()
    test_pipeline_integration()
    print("\n" + "=" * 50)
    print("ALL PHASE 2 TESTS PASSED!")
    print("=" * 50)
