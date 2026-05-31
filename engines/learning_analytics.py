import json
import sqlite3
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict
from config import config


class LearningAnalytics:
    def __init__(
        self,
        progress_tracker=None,
        learning_planner=None,
        knowledge_graph=None,
        tree_storage=None,
        document_manager=None,
    ):
        self.progress_tracker = progress_tracker
        self.learning_planner = learning_planner
        self.knowledge_graph = knowledge_graph
        self.tree_storage = tree_storage
        self.document_manager = document_manager

    def get_learning_dashboard(self, user_id: str = "default") -> Dict[str, Any]:
        progress = self._get_progress_data(user_id)
        coverage = self._get_knowledge_coverage(user_id)
        weaknesses = self._identify_weaknesses(user_id)
        study_pattern = self._analyze_study_pattern(user_id)
        plan_summary = self._get_plan_summary(user_id)

        return {
            "user_id": user_id,
            "progress": progress,
            "knowledge_coverage": coverage,
            "weaknesses": weaknesses,
            "study_pattern": study_pattern,
            "plan_summary": plan_summary,
            "generated_at": time.time(),
        }

    def _get_progress_data(self, user_id: str) -> Dict[str, Any]:
        if not self.progress_tracker:
            return {}

        summary = self.progress_tracker.get_progress_summary(user_id)
        due_reviews = self.progress_tracker.get_due_reviews(user_id, limit=50)
        upcoming = self.progress_tracker.get_upcoming_reviews(user_id, days_ahead=7)

        mastery_dist = summary.get("mastery_distribution", {})
        total = summary.get("total_knowledge_nodes", 0)

        mastery_timeline = self._build_mastery_timeline(user_id)

        return {
            "total_knowledge_nodes": total,
            "progress_pct": summary.get("progress_pct", 0),
            "mastery_distribution": mastery_dist,
            "due_reviews_count": len(due_reviews),
            "upcoming_reviews_count": len(upcoming),
            "total_exposures": summary.get("total_exposures", 0),
            "avg_ease_factor": summary.get("avg_ease_factor", 2.5),
            "mastery_timeline": mastery_timeline,
        }

    def _get_knowledge_coverage(self, user_id: str) -> Dict[str, Any]:
        if not self.progress_tracker or not self.tree_storage:
            return {"coverage_pct": 0, "covered_docs": 0, "total_docs": 0}

        storage_stats = self.tree_storage.get_stats()
        total_docs = storage_stats.get("doc_count", 0)
        total_l3 = storage_stats.get("level_counts", {}).get(3, 0)

        progress_summary = self.progress_tracker.get_progress_summary(user_id)
        exposed = progress_summary.get("total_knowledge_nodes", 0)

        coverage_pct = round(exposed / total_l3 * 100, 1) if total_l3 > 0 else 0

        doc_coverage = self._get_per_doc_coverage(user_id)

        return {
            "coverage_pct": coverage_pct,
            "total_knowledge_nodes": total_l3,
            "covered_nodes": exposed,
            "total_docs": total_docs,
            "covered_docs": len([d for d in doc_coverage if d["coverage_pct"] > 0]),
            "doc_coverage": doc_coverage,
        }

    def _get_per_doc_coverage(self, user_id: str) -> List[Dict[str, Any]]:
        if not self.tree_storage or not self.progress_tracker:
            return []

        docs = []
        if self.document_manager:
            doc_list = self.document_manager.list_documents(limit=100)
            docs = [d["doc_id"] for d in doc_list]

        if not docs:
            storage_stats = self.tree_storage.get_stats()
            if storage_stats.get("doc_count", 0) > 0:
                cursor = self.tree_storage.db.execute(
                    "SELECT DISTINCT doc_id FROM tree_nodes WHERE level = 3"
                )
                docs = [row[0] for row in cursor.fetchall()]

        # Batch query all exposed node IDs for the user to avoid N+1 queries
        exposed_node_ids = set()
        cursor = self.progress_tracker.db.execute(
            "SELECT knowledge_node_id FROM knowledge_records "
            "WHERE user_id = ? AND exposure_count > 0",
            (user_id,),
        )
        for row in cursor.fetchall():
            exposed_node_ids.add(row[0])

        result = []
        for doc_id in docs:
            l3_nodes = self.tree_storage.get_nodes_by_level(3, doc_id)
            total = len(l3_nodes)
            covered = sum(1 for node in l3_nodes if node.node_id in exposed_node_ids)

            result.append({
                "doc_id": doc_id,
                "total_nodes": total,
                "covered_nodes": covered,
                "coverage_pct": round(covered / total * 100, 1) if total > 0 else 0,
            })

        return result

    def _identify_weaknesses(self, user_id: str) -> Dict[str, Any]:
        if not self.progress_tracker:
            return {"weak_areas": [], "forgotten_areas": [], "never_reviewed": []}

        weak_areas = []
        forgotten_areas = []
        never_reviewed = []

        cursor = self.progress_tracker.db.execute(
            "SELECT knowledge_node_id, title, mastery, exposure_count, "
            "last_reviewed_at, next_review_at, review_interval_days, ease_factor "
            "FROM knowledge_records WHERE user_id = ?",
            (user_id,),
        )
        rows = cursor.fetchall()

        now = time.time()
        for row in rows:
            node_id, title, mastery, exposure_count, last_reviewed, next_review, interval, ease = row

            if mastery in ("unknown", "exposed") and exposure_count >= 2:
                weak_areas.append({
                    "node_id": node_id,
                    "title": title,
                    "mastery": mastery,
                    "exposure_count": exposure_count,
                    "reason": f"多次接触但仍未掌握 (掌握度: {mastery})",
                })

            if next_review > 0 and next_review < now - 86400 * 3:
                overdue_days = round((now - next_review) / 86400, 1)
                forgotten_areas.append({
                    "node_id": node_id,
                    "title": title,
                    "mastery": mastery,
                    "overdue_days": overdue_days,
                    "reason": f"复习已逾期 {overdue_days} 天，可能已遗忘",
                })

            if exposure_count == 1 and last_reviewed > 0 and (now - last_reviewed) > 86400 * 7:
                never_reviewed.append({
                    "node_id": node_id,
                    "title": title,
                    "days_since_exposure": round((now - last_reviewed) / 86400, 1),
                    "reason": "仅接触一次，从未复习",
                })

        weak_areas.sort(key=lambda x: x["exposure_count"], reverse=True)
        forgotten_areas.sort(key=lambda x: x["overdue_days"], reverse=True)
        never_reviewed.sort(key=lambda x: x["days_since_exposure"], reverse=True)

        return {
            "weak_areas": weak_areas[:5],
            "forgotten_areas": forgotten_areas[:5],
            "never_reviewed": never_reviewed[:5],
            "total_weak": len(weak_areas),
            "total_forgotten": len(forgotten_areas),
            "total_never_reviewed": len(never_reviewed),
        }

    def _analyze_study_pattern(self, user_id: str) -> Dict[str, Any]:
        if not self.progress_tracker:
            return {}

        cursor = self.progress_tracker.db.execute(
            "SELECT knowledge_node_id, exposure_count, last_reviewed_at, created_at "
            "FROM knowledge_records WHERE user_id = ? AND exposure_count > 0",
            (user_id,),
        )
        rows = cursor.fetchall()

        if not rows:
            return {"pattern": "no_data", "total_interactions": 0}

        total_interactions = sum(row[1] for row in rows)

        hour_distribution = defaultdict(int)
        day_distribution = defaultdict(int)
        for row in rows:
            created = row[3]
            if created > 0:
                import datetime
                dt = datetime.datetime.fromtimestamp(created)
                hour_distribution[dt.hour] += 1
                day_distribution[dt.strftime("%A")] += 1

        peak_hour = max(hour_distribution, key=hour_distribution.get) if hour_distribution else 0
        peak_day = max(day_distribution, key=day_distribution.get) if day_distribution else "N/A"

        exposure_counts = [row[1] for row in rows]
        avg_exposure = sum(exposure_counts) / len(exposure_counts) if exposure_counts else 0

        high_exposure = sum(1 for c in exposure_counts if c >= 5)
        low_exposure = sum(1 for c in exposure_counts if c <= 1)

        if avg_exposure >= 3:
            pattern = "deep_learner"
            pattern_desc = "深度学习型 - 倾向于对少量知识点进行反复学习"
        elif avg_exposure >= 1.5:
            pattern = "balanced_learner"
            pattern_desc = "均衡学习型 - 学习广度和深度较为均衡"
        else:
            pattern = "broad_learner"
            pattern_desc = "广度学习型 - 倾向于广泛涉猎，但复习深度不足"

        streak = self._calculate_streak(user_id)

        return {
            "pattern": pattern,
            "pattern_description": pattern_desc,
            "total_interactions": total_interactions,
            "avg_exposure_per_node": round(avg_exposure, 2),
            "high_exposure_nodes": high_exposure,
            "low_exposure_nodes": low_exposure,
            "peak_study_hour": peak_hour,
            "peak_study_day": peak_day,
            "hour_distribution": dict(hour_distribution),
            "current_streak_days": streak,
        }

    def _calculate_streak(self, user_id: str) -> int:
        if not self.progress_tracker:
            return 0

        cursor = self.progress_tracker.db.execute(
            "SELECT DISTINCT date(last_reviewed_at, 'unixepoch') as study_date "
            "FROM knowledge_records WHERE user_id = ? AND last_reviewed_at > 0 "
            "ORDER BY study_date DESC",
            (user_id,),
        )
        dates = [row[0] for row in cursor.fetchall()]

        if not dates:
            return 0

        import datetime
        streak = 0
        today = datetime.date.today()

        for i, date_str in enumerate(dates):
            try:
                study_date = datetime.date.fromisoformat(date_str)
                expected = today - datetime.timedelta(days=i)
                if study_date == expected:
                    streak += 1
                else:
                    break
            except (ValueError, TypeError):
                break

        return streak

    def _get_plan_summary(self, user_id: str) -> Dict[str, Any]:
        if not self.learning_planner:
            return {}

        plans = self.learning_planner.list_plans(user_id)
        active_plans = [p for p in plans if p["status"] == "active"]
        completed_plans = [p for p in plans if p["status"] == "completed"]

        plan_details = []
        for p in active_plans[:5]:
            progress = self.learning_planner.get_plan_progress(p["plan_id"])
            if progress:
                plan_details.append(progress)

        return {
            "total_plans": len(plans),
            "active_plans": len(active_plans),
            "completed_plans": len(completed_plans),
            "active_plan_details": plan_details,
        }

    def _build_mastery_timeline(self, user_id: str) -> List[Dict[str, Any]]:
        if not self.progress_tracker:
            return []

        cursor = self.progress_tracker.db.execute(
            "SELECT date(timestamp, 'unixepoch') as review_date, "
            "old_mastery, new_mastery, quality "
            "FROM review_events WHERE user_id = ? "
            "ORDER BY timestamp ASC",
            (user_id,),
        )
        rows = cursor.fetchall()

        timeline = []
        daily_changes = defaultdict(lambda: {"upgrades": 0, "downgrades": 0, "reviews": 0})

        for row in rows:
            date_str, old_m, new_m, quality = row
            daily_changes[date_str]["reviews"] += 1
            mastery_order = {"unknown": 0, "exposed": 1, "familiar": 2, "proficient": 3, "mastered": 4}
            old_level = mastery_order.get(old_m, 0)
            new_level = mastery_order.get(new_m, 0)
            if new_level > old_level:
                daily_changes[date_str]["upgrades"] += 1
            elif new_level < old_level:
                daily_changes[date_str]["downgrades"] += 1

        for date_str, changes in sorted(daily_changes.items()):
            timeline.append({
                "date": date_str,
                "reviews": changes["reviews"],
                "upgrades": changes["upgrades"],
                "downgrades": changes["downgrades"],
            })

        return timeline[-30:]

    def get_confusion_patterns(self, user_id: str = "default") -> List[Dict[str, Any]]:
        if not self.progress_tracker:
            return []
        wrong_items = self.progress_tracker.get_wrong_answers(user_id, limit=100)
        if len(wrong_items) < 2:
            return []

        for item in wrong_items:
            q = item.get("question", "").lower()
            ca = item.get("correct_answer", "").lower()
            ua = item.get("user_answer", "").lower()
            item["_q_norm"] = q
            item["_ca_norm"] = ca
            item["_ua_norm"] = ua

        pairs = []
        seen_pairs = set()
        for i in range(len(wrong_items)):
            for j in range(i + 1, len(wrong_items)):
                a = wrong_items[i]
                b = wrong_items[j]
                if a["_ca_norm"] == b["_ua_norm"] and b["_ca_norm"] == a["_ua_norm"]:
                    pair_key = tuple(sorted([a["_ca_norm"], b["_ca_norm"]]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        pairs.append({
                            "concept_a": a.get("correct_answer", ""),
                            "concept_b": b.get("correct_answer", ""),
                            "user_wrong_answer_a": a.get("user_answer", ""),
                            "user_wrong_answer_b": b.get("user_answer", ""),
                            "count": 2,
                            "node_ids_a": a.get("knowledge_node_ids", []),
                            "node_ids_b": b.get("knowledge_node_ids", []),
                        })
        return pairs

    def get_study_recommendations(self, user_id: str = "default") -> List[Dict[str, Any]]:
        recommendations = []

        weaknesses = self._identify_weaknesses(user_id)

        for area in weaknesses.get("forgotten_areas", [])[:3]:
            recommendations.append({
                "type": "review",
                "priority": "high",
                "node_id": area["node_id"],
                "title": area["title"],
                "reason": area["reason"],
                "action": "立即复习已遗忘的知识点",
            })

        for area in weaknesses.get("weak_areas", [])[:3]:
            recommendations.append({
                "type": "reinforce",
                "priority": "medium",
                "node_id": area["node_id"],
                "title": area["title"],
                "reason": area["reason"],
                "action": "加强薄弱知识点的学习",
            })

        for area in weaknesses.get("never_reviewed", [])[:2]:
            recommendations.append({
                "type": "first_review",
                "priority": "medium",
                "node_id": area["node_id"],
                "title": area["title"],
                "reason": area["reason"],
                "action": "进行首次复习巩固",
            })

        if self.progress_tracker:
            due = self.progress_tracker.get_due_reviews(user_id, limit=3)
            for review in due:
                recommendations.append({
                    "type": "spaced_review",
                    "priority": "high",
                    "node_id": review["knowledge_node_id"],
                    "title": review.get("title", ""),
                    "reason": f"间隔复习到期 (逾期 {review.get('overdue_days', 0)} 天)",
                    "action": "按SM-2算法进行复习",
                })

        recommendations.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["priority"], 3))

        return recommendations[:8]

    def get_knowledge_graph_insights(self, user_id: str = "default") -> Dict[str, Any]:
        if not self.knowledge_graph or not self.progress_tracker:
            return {}

        kg_stats = self.knowledge_graph.get_graph_stats()

        cursor = self.knowledge_graph.db.execute(
            "SELECT e.name, e.entity_type, COUNT(r.relation_id) as rel_count "
            "FROM entities e LEFT JOIN relations r ON "
            "(r.source_entity_id = e.entity_id OR r.target_entity_id = e.entity_id) "
            "GROUP BY e.entity_id ORDER BY rel_count DESC LIMIT 10"
        )
        hub_entities = [
            {"name": row[0], "type": row[1], "relation_count": row[2]}
            for row in cursor.fetchall()
        ]

        cursor = self.knowledge_graph.db.execute(
            "SELECT e.name, e.entity_type FROM entities e "
            "LEFT JOIN relations r ON "
            "(r.source_entity_id = e.entity_id OR r.target_entity_id = e.entity_id) "
            "WHERE r.relation_id IS NULL"
        )
        isolated = [{"name": row[0], "type": row[1]} for row in cursor.fetchall()]

        return {
            "total_entities": kg_stats.get("total_entities", 0),
            "total_relations": kg_stats.get("total_relations", 0),
            "hub_entities": hub_entities,
            "isolated_entities": isolated[:5],
            "entity_types": kg_stats.get("entity_types", {}),
            "relation_types": kg_stats.get("relation_types", {}),
        }
