import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LearningStep:
    concept: str = ""
    definition: str = ""
    bloom_level: str = "理解"
    prerequisites: List[str] = field(default_factory=list)
    difficulty: float = 0.5
    mastery: str = "unknown"
    wrong_count: int = 0
    reason: str = ""
    action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept,
            "definition": self.definition,
            "bloom_level": self.bloom_level,
            "prerequisites": self.prerequisites,
            "difficulty": self.difficulty,
            "mastery": self.mastery,
            "wrong_count": self.wrong_count,
            "reason": self.reason,
            "action": self.action,
        }


class TutorAgent:
    PREREQUISITE_TYPES = {"依赖", "包含", "属于", "基于", "前提", "前置",
                          "prerequisite_of", "requires", "depends_on"}

    def __init__(
        self,
        progress_tracker=None,
        knowledge_graph=None,
        llm_func: Optional[Callable] = None,
    ):
        self.progress_tracker = progress_tracker
        self.knowledge_graph = knowledge_graph
        self.llm_func = llm_func

    def get_next_steps(
        self,
        user_id: str = "default",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.progress_tracker:
            return []

        weak_nodes = self.progress_tracker.get_weak_nodes(user_id, threshold=1)
        if weak_nodes:
            steps = []
            for w in weak_nodes[:limit]:
                step = LearningStep(
                    concept=w["title"],
                    mastery=w["mastery"],
                    wrong_count=w["wrong_count"],
                    reason=f"连续答错 {w['wrong_count']} 次，需要巩固",
                    action="建议重新学习相关文档段落，然后做练习题",
                )
                steps.append(step)
            return [s.to_dict() for s in steps]

        summary = self.progress_tracker.get_progress_summary(user_id)
        mastery_dist = summary.get("mastery_distribution", {})
        total = summary.get("total_knowledge_nodes", 0)
        if total == 0:
            return [{
                "concept": "开始学习",
                "reason": "还没有学习记录",
                "action": "请先上传一份学习文档开始你的学习之旅",
            }]

        due_reviews = self.progress_tracker.get_due_reviews(user_id, limit=limit)
        if due_reviews:
            steps = []
            for d in due_reviews[:limit]:
                overdue_days = d.get("overdue_days", 0)
                reason = f"已逾期 {overdue_days} 天" if overdue_days > 0 else "即将到复习时间"
                steps.append(LearningStep(
                    concept=d.get("title", d.get("knowledge_node_id", "知识点")),
                    mastery=d["mastery"],
                    reason=reason,
                    action="建议立即复习，使用间隔重复对抗遗忘曲线",
                ))
            return [s.to_dict() for s in steps]

        unknown_count = mastery_dist.get("unknown", 0) + mastery_dist.get("exposed", 0)
        if unknown_count > 0:
            return [{
                "concept": "探索新知识",
                "reason": f"还有 {unknown_count} 个知识点未学习",
                "action": "可以在对话中输入'帮我总结文档内容'来概览全貌",
            }]

        return [{
            "concept": "复习巩固",
            "reason": "所有知识点已学习完毕",
            "action": "建议进行综合测验或探索新的学习领域",
        }]

    def generate_learning_path(
        self,
        goal_concept: str,
        user_id: str = "default",
        max_steps: int = 10,
    ) -> List[Dict[str, Any]]:
        if not self.knowledge_graph or not self.progress_tracker:
            return [{"concept": goal_concept, "action": "直接学习"}]

        goal_entity = self.knowledge_graph.query_entity(goal_concept)
        if not goal_entity:
            return [{"concept": goal_concept, "action": "开始学习"}]

        all_prereqs = self._collect_prerequisites(goal_entity["entity_id"], max_depth=3)
        mastered = set()
        for node_id, _ in all_prereqs:
            record = self.progress_tracker._find_record(node_id, user_id)
            if record and record.mastery.value in ("proficient", "mastered"):
                mastered.add(node_id)

        path = []
        for node_id, name in all_prereqs:
            is_mastered = node_id in mastered
            path.append({
                "concept": name,
                "mastery": "已掌握" if is_mastered else "待学习",
                "action": "可跳过" if is_mastered else "建议优先学习",
            })

        path.append({
            "concept": goal_concept,
            "mastery": "目标",
            "action": "完成前置知识后开始学习",
        })

        return path[:max_steps]

    def _collect_prerequisites(
        self,
        entity_id: str,
        max_depth: int = 3,
        depth: int = 0,
    ) -> List[tuple]:
        if depth >= max_depth:
            return []
        prereqs = []
        for rel_type in self.PREREQUISITE_TYPES:
            related = self.knowledge_graph.get_related_entities_by_id(entity_id, relation_type=rel_type)
            for rel in related:
                prereqs.append((rel["entity_id"], rel["name"]))
        deduped = list(dict.fromkeys(prereqs))
        result = list(deduped)
        for eid, _ in deduped:
            result.extend(self._collect_prerequisites(eid, max_depth, depth + 1))
        return result
