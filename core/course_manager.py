import json
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from core.database import DatabaseManager, get_database

logger = logging.getLogger(__name__)


@dataclass
class Course:
    course_id: str = ""
    name: str = ""
    description: str = ""
    doc_count: int = 0
    node_count: int = 0
    entity_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        if not self.course_id:
            self.course_id = f"course_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "course_id": self.course_id,
            "name": self.name,
            "description": self.description,
            "doc_count": self.doc_count,
            "node_count": self.node_count,
            "entity_count": self.entity_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class CourseManager:
    def __init__(self, db: Optional[DatabaseManager] = None, db_path: Optional[str] = None):
        if db is not None:
            self.db = db
        elif db_path is not None:
            self.db = get_database(db_path)
        else:
            self.db = get_database()
        self._init_db()

    def _init_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                course_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                doc_count INTEGER NOT NULL DEFAULT 0,
                node_count INTEGER NOT NULL DEFAULT 0,
                entity_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0
            )
        """)
        self.db.commit()

        self._ensure_default_course()

    def _ensure_default_course(self):
        cursor = self.db.execute("SELECT COUNT(*) FROM courses")
        if cursor.fetchone()[0] == 0:
            self.create_course("通用学习", "默认课程，未分类的文档归入此处")

    def create_course(self, name: str, description: str = "") -> Course:
        course = Course(name=name, description=description)
        self.db.execute(
            "INSERT INTO courses (course_id, name, description, doc_count, node_count, entity_count, created_at, updated_at) "
            "VALUES (?, ?, ?, 0, 0, 0, ?, ?)",
            (course.course_id, course.name, course.description, course.created_at, course.updated_at),
        )
        self.db.commit()
        return course

    def list_courses(self) -> List[Dict[str, Any]]:
        cursor = self.db.execute(
            "SELECT course_id, name, description, doc_count, node_count, entity_count, created_at, updated_at "
            "FROM courses ORDER BY updated_at DESC"
        )
        return [
            {
                "course_id": row[0],
                "name": row[1],
                "description": row[2],
                "doc_count": row[3],
                "node_count": row[4],
                "entity_count": row[5],
                "created_at": row[6],
                "updated_at": row[7],
            }
            for row in cursor.fetchall()
        ]

    def get_course(self, course_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.db.execute(
            "SELECT course_id, name, description, doc_count, node_count, entity_count, created_at, updated_at "
            "FROM courses WHERE course_id = ?",
            (course_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "course_id": row[0],
            "name": row[1],
            "description": row[2],
            "doc_count": row[3],
            "node_count": row[4],
            "entity_count": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }

    def update_course_stats(self, course_id: str, doc_count: int = 0, node_count: int = 0, entity_count: int = 0):
        course = self.get_course(course_id)
        if not course:
            return
        self.db.execute(
            "UPDATE courses SET doc_count = ?, node_count = ?, entity_count = ?, updated_at = ? WHERE course_id = ?",
            (doc_count, node_count, entity_count, time.time(), course_id),
        )
        self.db.commit()

    def delete_course(self, course_id: str) -> bool:
        self.db.execute("DELETE FROM courses WHERE course_id = ?", (course_id,))
        self.db.commit()
        return True
