import json
import re
import time
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from core.database import DatabaseManager, get_database


class PlanStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class TaskPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class LearningTask:
    task_id: str = ""
    title: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_minutes: int = 30
    prerequisites: List[str] = field(default_factory=list)
    knowledge_nodes: List[str] = field(default_factory=list)
    status: str = "pending"
    order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"task_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value if isinstance(self.priority, TaskPriority) else self.priority,
            "estimated_minutes": self.estimated_minutes,
            "prerequisites": self.prerequisites,
            "knowledge_nodes": self.knowledge_nodes,
            "status": self.status,
            "order": self.order,
            "metadata": self.metadata,
        }


@dataclass
class LearningPlan:
    plan_id: str = ""
    user_id: str = "default"
    title: str = ""
    description: str = ""
    doc_id: str = ""
    status: PlanStatus = PlanStatus.DRAFT
    tasks: List[LearningTask] = field(default_factory=list)
    total_estimated_minutes: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()
        self._recalculate()

    def _recalculate(self):
        self.total_estimated_minutes = sum(t.estimated_minutes for t in self.tasks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "user_id": self.user_id,
            "title": self.title,
            "description": self.description,
            "doc_id": self.doc_id,
            "status": self.status.value if isinstance(self.status, PlanStatus) else self.status,
            "tasks": [t.to_dict() for t in self.tasks],
            "total_estimated_minutes": self.total_estimated_minutes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class LearningPlanner:
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        llm_func: Optional[Callable] = None,
        db_path: Optional[str] = None,
        knowledge_graph=None,
    ):
        if db is not None:
            self.db = db
        elif db_path is not None:
            self.db = get_database(db_path)
        else:
            self.db = get_database()
        self.llm_func = llm_func
        self.knowledge_graph = knowledge_graph
        self._init_db()

    def _init_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS learning_plans (
                plan_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                title TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                doc_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'draft',
                total_estimated_minutes INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS learning_tasks (
                task_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                priority TEXT NOT NULL DEFAULT 'medium',
                estimated_minutes INTEGER NOT NULL DEFAULT 30,
                prerequisites TEXT NOT NULL DEFAULT '[]',
                knowledge_nodes TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'pending',
                task_order INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (plan_id) REFERENCES learning_plans(plan_id)
            )
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_plan ON learning_tasks(plan_id)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_plan_user ON learning_plans(user_id)
        """)
        self.db.commit()

    def create_plan_from_doc(
        self,
        doc_id: str,
        l2_nodes: List[Dict],
        user_id: str = "default",
        title: str = "",
        daily_minutes: int = 60,
    ) -> LearningPlan:
        tasks = self._generate_tasks_from_nodes(l2_nodes, daily_minutes)

        plan = LearningPlan(
            user_id=user_id,
            title=title or f"学习计划 - {doc_id}",
            doc_id=doc_id,
            status=PlanStatus.DRAFT,
            tasks=tasks,
        )
        self._save_plan(plan)
        return plan

    def create_plan_from_goal(
        self,
        goal: str,
        user_id: str = "default",
        daily_minutes: int = 60,
        total_days: int = 7,
    ) -> LearningPlan:
        tasks = []
        if self.llm_func:
            tasks = self._generate_tasks_from_llm(goal, daily_minutes, total_days)
        else:
            tasks = self._generate_tasks_from_template(goal, daily_minutes, total_days)

        plan = LearningPlan(
            user_id=user_id,
            title=f"学习计划 - {goal}",
            status=PlanStatus.DRAFT,
            tasks=tasks,
        )
        self._save_plan(plan)
        return plan

    def get_plan(self, plan_id: str) -> Optional[LearningPlan]:
        cursor = self.db.execute(
            "SELECT plan_id, user_id, title, description, doc_id, status, "
            "total_estimated_minutes, created_at, updated_at "
            "FROM learning_plans WHERE plan_id = ?",
            (plan_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        plan = LearningPlan(
            plan_id=row[0],
            user_id=row[1],
            title=row[2],
            description=row[3],
            doc_id=row[4],
            status=PlanStatus(row[5]),
            total_estimated_minutes=row[6],
            created_at=row[7],
            updated_at=row[8],
        )
        plan.tasks = self._load_tasks(plan_id)
        return plan

    def update_plan_status(self, plan_id: str, status: PlanStatus) -> Optional[LearningPlan]:
        plan = self.get_plan(plan_id)
        if not plan:
            return None

        plan.status = status
        plan.updated_at = time.time()
        self.db.execute(
            "UPDATE learning_plans SET status = ?, updated_at = ? WHERE plan_id = ?",
            (status.value, time.time(), plan_id),
        )
        self.db.commit()
        return plan

    def update_task_status(self, plan_id: str, task_id: str, status: str) -> Optional[LearningPlan]:
        plan = self.get_plan(plan_id)
        if not plan:
            return None

        for task in plan.tasks:
            if task.task_id == task_id:
                task.status = status
                break

        self.db.execute(
            "UPDATE learning_tasks SET status = ? WHERE task_id = ?",
            (status, task_id),
        )
        self.db.commit()

        all_done = all(t.status in ("completed", "skipped") for t in plan.tasks)
        if all_done:
            self.update_plan_status(plan_id, PlanStatus.COMPLETED)

        return plan

    def get_next_task(self, plan_id: str) -> Optional[LearningTask]:
        plan = self.get_plan(plan_id)
        if not plan or plan.status != PlanStatus.ACTIVE:
            return None

        completed_ids = {t.task_id for t in plan.tasks if t.status == "completed"}
        for task in sorted(plan.tasks, key=lambda t: t.order):
            if task.status != "pending":
                continue
            if all(pre in completed_ids for pre in task.prerequisites):
                return task
        return None

    def get_daily_schedule(
        self,
        plan_id: str,
        available_minutes: int = 60,
    ) -> List[Dict[str, Any]]:
        plan = self.get_plan(plan_id)
        if not plan:
            return []

        completed_ids = {t.task_id for t in plan.tasks if t.status == "completed"}
        schedule = []
        remaining = available_minutes

        for task in sorted(plan.tasks, key=lambda t: (t.order, -ord(t.priority[0]) if t.priority else 0)):
            if task.status != "pending":
                continue
            if not all(pre in completed_ids for pre in task.prerequisites):
                continue
            if task.estimated_minutes <= remaining:
                schedule.append(task.to_dict())
                remaining -= task.estimated_minutes
            if remaining <= 0:
                break

        return schedule

    def list_plans(
        self,
        user_id: str = "default",
        status: Optional[PlanStatus] = None,
    ) -> List[Dict[str, Any]]:
        if status:
            cursor = self.db.execute(
                "SELECT plan_id, title, status, total_estimated_minutes, created_at, updated_at "
                "FROM learning_plans WHERE user_id = ? AND status = ? ORDER BY updated_at DESC",
                (user_id, status.value),
            )
        else:
            cursor = self.db.execute(
                "SELECT plan_id, title, status, total_estimated_minutes, created_at, updated_at "
                "FROM learning_plans WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            )
        rows = cursor.fetchall()
        return [
            {
                "plan_id": row[0],
                "title": row[1],
                "status": row[2],
                "total_estimated_minutes": row[3],
                "created_at": row[4],
                "updated_at": row[5],
            }
            for row in rows
        ]

    def get_plan_progress(self, plan_id: str) -> Dict[str, Any]:
        plan = self.get_plan(plan_id)
        if not plan:
            return {}

        total = len(plan.tasks)
        completed = sum(1 for t in plan.tasks if t.status == "completed")
        in_progress = sum(1 for t in plan.tasks if t.status == "in_progress")
        pending = sum(1 for t in plan.tasks if t.status == "pending")
        skipped = sum(1 for t in plan.tasks if t.status == "skipped")

        completed_minutes = sum(t.estimated_minutes for t in plan.tasks if t.status == "completed")
        remaining_minutes = sum(t.estimated_minutes for t in plan.tasks if t.status != "completed")

        return {
            "plan_id": plan_id,
            "title": plan.title,
            "status": plan.status.value if isinstance(plan.status, PlanStatus) else plan.status,
            "total_tasks": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "skipped": skipped,
            "progress_pct": round(completed / total * 100, 1) if total > 0 else 0,
            "completed_minutes": completed_minutes,
            "remaining_minutes": remaining_minutes,
        }

    def delete_plan(self, plan_id: str):
        self.db.execute("DELETE FROM learning_tasks WHERE plan_id = ?", (plan_id,))
        self.db.execute("DELETE FROM learning_plans WHERE plan_id = ?", (plan_id,))
        self.db.commit()

    @staticmethod
    def _strip_section(title: str) -> str:
        return re.sub(r"^\d+(?:[.\-]\d+)*\s*", "", title).strip()

    def _find_task_index_by_name(self, name: str, node_titles: List[tuple]) -> int:
        norm = self._strip_section(name).lower()
        for i, t in node_titles:
            if self._strip_section(t).lower() == norm:
                return i
        for i, t in node_titles:
            if norm in self._strip_section(t).lower() or self._strip_section(t).lower() in norm:
                return i
        return -1

    def _generate_tasks_from_nodes(
        self,
        l2_nodes: List[Dict],
        daily_minutes: int,
    ) -> List[LearningTask]:
        tasks = []

        node_titles = [(i, n.get("title", "")) for i, n in enumerate(l2_nodes)]

        kg_deps = {}
        if self.knowledge_graph:
            for i, title in node_titles:
                prereq_ids = set()
                stripped = self._strip_section(title)
                entity = self.knowledge_graph.query_entity(title)
                if not entity and stripped != title:
                    entity = self.knowledge_graph.query_entity(stripped)
                if entity:
                    related = self.knowledge_graph.get_related_entities(
                        entity.get("name", stripped), relation_type="prerequisite_of", limit=20
                    )
                    for r in related:
                        pname = r.get("name", "")
                        j = self._find_task_index_by_name(pname, node_titles)
                        if j >= 0:
                            prereq_ids.add(j)
                kg_deps[i] = prereq_ids

        task_map = {}
        for i, node in enumerate(l2_nodes):
            content_len = len(node.get("content", ""))
            est_minutes = max(15, min(90, content_len // 50))

            prereq_ids = kg_deps.get(i, set()) if kg_deps else set()
            prerequisites = [task_map[pi] for pi in sorted(prereq_ids) if pi in task_map]

            task = LearningTask(
                title=node.get("title", f"学习任务 {i+1}"),
                description=node.get("content", "")[:300],
                priority=TaskPriority.HIGH if i < 2 else TaskPriority.MEDIUM,
                estimated_minutes=est_minutes,
                prerequisites=prerequisites,
                knowledge_nodes=[node.get("node_id", "")],
                status="pending",
                order=i,
            )
            tasks.append(task)
            task_map[i] = task.task_id

        return tasks

    def _generate_tasks_from_llm(
        self,
        goal: str,
        daily_minutes: int,
        total_days: int,
    ) -> List[LearningTask]:
        prompt = f"""请为以下学习目标制定一个{total_days}天的学习计划，每天约{daily_minutes}分钟。

学习目标: {goal}

请按以下JSON格式输出学习任务列表:
[
  {{"title": "任务标题", "description": "任务描述", "estimated_minutes": 30, "priority": "high"}},
  ...
]

只输出JSON数组，不要其他内容:"""

        try:
            response = self.llm_func(prompt)
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]
            task_defs = json.loads(clean)

            tasks = []
            for i, td in enumerate(task_defs):
                task_title = td.get("title", f"任务{i+1}")
                prereq_titles = td.get("prerequisites", [])
                prereq_ids = []
                if self.knowledge_graph and prereq_titles:
                    for pt in prereq_titles:
                        for t in tasks:
                            if t.title == pt:
                                prereq_ids.append(t.task_id)
                task = LearningTask(
                    title=task_title,
                    description=td.get("description", ""),
                    priority=TaskPriority(td.get("priority", "medium")),
                    estimated_minutes=td.get("estimated_minutes", 30),
                    prerequisites=prereq_ids,
                    status="pending",
                    order=i,
                )
                tasks.append(task)
            return tasks
        except Exception:
            return self._generate_tasks_from_template(goal, daily_minutes, total_days)

    def _generate_tasks_from_template(
        self,
        goal: str,
        daily_minutes: int,
        total_days: int,
    ) -> List[LearningTask]:
        phases = [
            ("基础概念学习", "了解核心概念和术语", "high"),
            ("深入理解原理", "掌握关键原理和机制", "high"),
            ("实践练习", "通过练习巩固理解", "medium"),
            ("综合应用", "将所学知识应用到实际问题", "medium"),
            ("总结复习", "回顾和巩固所有知识点", "low"),
        ]

        tasks = []
        prev_id = None
        minutes_per_task = (daily_minutes * total_days) // len(phases)

        for i, (title, desc, priority) in enumerate(phases):
            task = LearningTask(
                title=f"{title} - {goal}",
                description=desc,
                priority=TaskPriority(priority),
                estimated_minutes=minutes_per_task,
                prerequisites=[prev_id] if prev_id else [],
                status="pending",
                order=i,
            )
            tasks.append(task)
            prev_id = task.task_id

        return tasks

    def _save_plan(self, plan: LearningPlan):
        self.db.execute(
            """INSERT OR REPLACE INTO learning_plans
            (plan_id, user_id, title, description, doc_id, status, total_estimated_minutes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                plan.plan_id,
                plan.user_id,
                plan.title,
                plan.description,
                plan.doc_id,
                plan.status.value if isinstance(plan.status, PlanStatus) else plan.status,
                plan.total_estimated_minutes,
                plan.created_at,
                plan.updated_at,
            ),
        )
        self.db.commit()

        for task in plan.tasks:
            self._save_task(plan.plan_id, task)

    def _save_task(self, plan_id: str, task: LearningTask):
        self.db.execute(
            """INSERT OR REPLACE INTO learning_tasks
            (task_id, plan_id, title, description, priority, estimated_minutes,
             prerequisites, knowledge_nodes, status, task_order, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task.task_id,
                plan_id,
                task.title,
                task.description,
                task.priority.value if isinstance(task.priority, TaskPriority) else task.priority,
                task.estimated_minutes,
                json.dumps(task.prerequisites),
                json.dumps(task.knowledge_nodes),
                task.status,
                task.order,
                json.dumps(task.metadata, ensure_ascii=False),
            ),
        )
        self.db.commit()

    def _load_tasks(self, plan_id: str) -> List[LearningTask]:
        cursor = self.db.execute(
            "SELECT task_id, title, description, priority, estimated_minutes, "
            "prerequisites, knowledge_nodes, status, task_order, metadata "
            "FROM learning_tasks WHERE plan_id = ? ORDER BY task_order",
            (plan_id,),
        )
        rows = cursor.fetchall()
        return [
            LearningTask(
                task_id=row[0],
                title=row[1],
                description=row[2],
                priority=TaskPriority(row[3]),
                estimated_minutes=row[4],
                prerequisites=json.loads(row[5]) if row[5] else [],
                knowledge_nodes=json.loads(row[6]) if row[6] else [],
                status=row[7],
                order=row[8],
                metadata=json.loads(row[9]) if row[9] else {},
            )
            for row in rows
        ]

    def close(self):
        pass  # DatabaseManager is shared, closed at app shutdown
