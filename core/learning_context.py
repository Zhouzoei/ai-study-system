import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LearningContextBuilder:
    def __init__(
        self,
        progress_tracker=None,
        document_manager=None,
        conversation_memory=None,
        knowledge_graph=None,
        analytics=None,
    ):
        self.progress_tracker = progress_tracker
        self.document_manager = document_manager
        self.conversation_memory = conversation_memory
        self.knowledge_graph = knowledge_graph
        self.analytics = analytics

    def build_system_context(
        self,
        user_id: str = "default",
        session_id: Optional[str] = None,
    ) -> str:
        sections = []

        master_section = self._build_mastery_section(user_id)
        if master_section:
            sections.append(master_section)

        doc_section = self._build_doc_manifest(user_id)
        if doc_section:
            sections.append(doc_section)

        weak_section = self._build_weakness_section(user_id)
        if weak_section:
            sections.append(weak_section)

        confusion_section = self._build_confusion_section(user_id)
        if confusion_section:
            sections.append(confusion_section)

        profile_section = self._build_profile_section(user_id)
        if profile_section:
            sections.append(profile_section)

        memory_section = self._build_memory_section(user_id, session_id)
        if memory_section:
            sections.append(memory_section)

        if not sections:
            return ""

        return "\n\n".join(sections)

    def build_markdown_context(
        self,
        user_id: str = "default",
        session_id: Optional[str] = None,
    ) -> str:
        raw = self.build_system_context(user_id, session_id)
        if not raw:
            return ""
        return f"\n\n[学习上下文]\n{raw}\n[/学习上下文]\n"

    def _build_mastery_section(self, user_id: str) -> Optional[str]:
        if not self.progress_tracker:
            return None
        try:
            summary = self.progress_tracker.get_progress_summary(user_id)
            total = summary.get("total_knowledge_nodes", 0)
            if total == 0:
                return None
            mastery_dist = summary.get("mastery_distribution", {})
            mastered = mastery_dist.get("mastered", 0) + mastery_dist.get("proficient", 0)
            learning = mastery_dist.get("familiar", 0) + mastery_dist.get("exposed", 0)
            unknown = mastery_dist.get("unknown", 0)
            due = summary.get("due_for_review", 0)
            pct = summary.get("progress_pct", 0)
            return (
                f"[学习进度] 共 {total} 个知识点 · "
                f"已掌握 {mastered} 个({pct}%) · "
                f"学习中 {learning} 个 · "
                f"未学习 {unknown} 个 · "
                f"待复习 {due} 个"
            )
        except Exception as e:
            logger.warning(f"Build mastery section failed: {e}")
            return None

    def _build_doc_manifest(self, user_id: str) -> Optional[str]:
        if not self.document_manager:
            return None
        try:
            docs = self.document_manager.list_documents(limit=20)
            if not docs:
                return None
            titles = [d.get("title", d.get("doc_id", "?")) for d in docs[:10]]
            tags_set = set()
            for d in docs:
                tags_set.update(d.get("tags", []))
            tag_str = f"，标签: {'、'.join(list(tags_set)[:5])}" if tags_set else ""
            return f"[可用知识库] {'、'.join(titles)}{tag_str}"
        except Exception as e:
            logger.warning(f"Build doc manifest failed: {e}")
            return None

    def _build_weakness_section(self, user_id: str) -> Optional[str]:
        if not self.progress_tracker:
            return None
        try:
            weak = self.progress_tracker.get_weak_nodes(user_id, threshold=1)
            if not weak:
                return None
            items = [f"「{w['title']}」(答错{w['wrong_count']}次)" for w in weak[:5]]
            return f"[薄弱知识点] {'、'.join(items)}"
        except Exception as e:
            logger.warning(f"Build weakness section failed: {e}")
            return None

    def _build_confusion_section(self, user_id: str) -> Optional[str]:
        if not self.analytics:
            return None
        try:
            patterns = self.analytics.get_confusion_patterns(user_id)
            if not patterns:
                return None
            items = []
            for p in patterns[:3]:
                a = p.get("concept_a", "")
                b = p.get("concept_b", "")
                if a and b:
                    items.append(f"「{a}」与「{b}」易混淆")
            if not items:
                return None
            return f"[常见混淆] {'、'.join(items)}"
        except Exception as e:
            logger.warning(f"Build confusion section failed: {e}")
            return None

    def _build_profile_section(self, user_id: str) -> Optional[str]:
        if not self.conversation_memory:
            return None
        try:
            profile = self.conversation_memory.get_user_profile(user_id)
            if not profile:
                return None
            parts = []
            interests = profile.get("interests", [])
            if interests:
                parts.append(f"兴趣领域: {'、'.join(interests[:5])}")
            expertise = profile.get("expertise", {})
            if expertise:
                top = sorted(expertise.items(), key=lambda x: -x[1])[:3]
                parts.append(f"专长: {'、'.join(f'{k}' for k, v in top)}")
            interaction = profile.get("interaction_count", 0)
            if interaction > 0:
                parts.append(f"总交互: {interaction}次")
            if not parts:
                return None
            return f"[用户画像] {' · '.join(parts)}"
        except Exception as e:
            logger.warning(f"Build profile section failed: {e}")
            return None

    def _build_memory_section(self, user_id: str, session_id: Optional[str]) -> Optional[str]:
        if not self.conversation_memory or not session_id:
            return None
        try:
            parts = []
            context = self.conversation_memory.get_full_context(
                session_id, max_tokens=2000,
                include_relevant_history=False,
                include_user_preferences=False,
            )
            if context:
                long_term = [m["content"] for m in context if m.get("role") == "system" and "长期记忆" in m.get("content", "")]
                medium_term = [m["content"] for m in context if m.get("role") == "system" and "近期记忆" in m.get("content", "")]
                if long_term:
                    parts.append(long_term[0][:200])
                if medium_term:
                    parts.append(medium_term[0][:200])
            if not parts:
                return None
            return "[记忆快照] " + " | ".join(parts)
        except Exception as e:
            logger.warning(f"Build memory section failed: {e}")
            return None
