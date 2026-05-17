from enum import Enum
from typing import List, Dict, Any, Optional
from core.tree_storage import TreeStorage
from core.hierarchical_chunker import ChunkNode


class ContextStrategy(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    FULL = "full"
    AUTO_MERGE = "auto_merge"


class HierarchicalRetriever:
    def __init__(
        self,
        tree_storage: TreeStorage,
        strategy: ContextStrategy = ContextStrategy.BALANCED,
        max_context_tokens: int = 4000,
        auto_merge_threshold: float = 0.3,
    ):
        self.storage = tree_storage
        self.strategy = strategy
        self.max_context_tokens = max_context_tokens
        self.auto_merge_threshold = auto_merge_threshold
        self._chars_per_token = 1.5

    def retrieve_with_context(
        self,
        l3_results: List[Dict],
        strategy: Optional[ContextStrategy] = None,
    ) -> List[Dict[str, Any]]:
        strategy = strategy or self.strategy
        enriched_results = []

        if strategy == ContextStrategy.AUTO_MERGE:
            return self._retrieve_with_auto_merge(l3_results)

        for result in l3_results:
            node_id = result.get("node_id", "")
            context_chain = self.storage.get_l3_context_chain(node_id)

            context = self._assemble_context(context_chain, strategy)
            enriched_results.append(
                {
                    "l3_node_id": node_id,
                    "l3_content": result.get("content", ""),
                    "l3_title": result.get("title", ""),
                    "l3_score": result.get("score", 0.0),
                    "assembled_context": context,
                    "context_chain": {
                        "l1_title": context_chain["l1"].title if context_chain["l1"] else None,
                        "l2_title": context_chain["l2"].title if context_chain["l2"] else None,
                        "l3_title": context_chain["l3"].title if context_chain["l3"] else None,
                    },
                }
            )

        return enriched_results

    def _retrieve_with_auto_merge(
        self, l3_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        l2_groups: Dict[str, Dict] = {}

        for result in l3_results:
            node_id = result.get("node_id", "")
            score = result.get("score", 0.0) or result.get("rrf_score", 0.0) or result.get("rerank_score", 0.0)
            chain = self.storage.get_l3_context_chain(node_id)
            l3 = chain["l3"]
            l2 = chain["l2"]
            l1 = chain["l1"]

            l2_key = l2.node_id if l2 else ""

            if l2_key not in l2_groups:
                l2_groups[l2_key] = {
                    "l2": l2,
                    "l1": l1,
                    "l3_nodes": [],
                    "max_score": score,
                    "merged": False,
                }

            l2_groups[l2_key]["l3_nodes"].append({
                "node_id": node_id,
                "content": result.get("content", ""),
                "title": result.get("title", ""),
                "score": score,
            })
            l2_groups[l2_key]["max_score"] = max(l2_groups[l2_key]["max_score"], score)

        enriched = []
        for key, group in l2_groups.items():
            l2 = group["l2"]
            l1 = group["l1"]
            l3_nodes = group["l3_nodes"]
            max_score = group["max_score"]

            if max_score >= self.auto_merge_threshold and l2:
                context = f"[章节: {l2.title}]\n{l2.content}"
                if l1:
                    context = f"[背景: {l1.title}]\n{l1.content[:300]}\n\n---\n\n{context}"
                group["merged"] = True
            else:
                best = max(l3_nodes, key=lambda x: x["score"])
                context = f"[具体内容: {best['title']}]\n{best['content']}"

            for ln in l3_nodes:
                enriched.append({
                    "l3_node_id": ln["node_id"],
                    "l3_content": ln["content"],
                    "l3_title": ln["title"],
                    "l3_score": ln["score"],
                    "assembled_context": context,
                    "auto_merged": group["merged"],
                    "context_chain": {
                        "l1_title": l1.title if l1 else None,
                        "l2_title": l2.title if l2 else None,
                        "l3_title": ln["title"],
                    },
                })

        enriched.sort(key=lambda x: x["l3_score"], reverse=True)
        return enriched

    def _assemble_context(
        self, context_chain: Dict[str, Any], strategy: ContextStrategy
    ) -> str:
        l3 = context_chain.get("l3")
        l2 = context_chain.get("l2")
        l1 = context_chain.get("l1")

        if not l3:
            return ""

        parts = []
        remaining_chars = int(self.max_context_tokens * self._chars_per_token)

        if strategy == ContextStrategy.CONSERVATIVE:
            if l2:
                parts.append(f"[章节上下文: {l2.title}]\n{l2.content}")
                remaining_chars -= len(l2.content) + len(l2.title) + 20
            parts.append(f"[具体内容: {l3.title}]\n{l3.content}")

        elif strategy == ContextStrategy.BALANCED:
            if l1:
                l1_summary = self._summarize_for_context(l1, max_chars=int(remaining_chars * 0.3))
                parts.append(f"[章节背景: {l1.title}]\n{l1_summary}")
                remaining_chars -= len(l1_summary) + len(l1.title) + 20
            if l2:
                l2_content = l2.content[:int(remaining_chars * 0.5)]
                parts.append(f"[小节上下文: {l2.title}]\n{l2_content}")
                remaining_chars -= len(l2_content) + len(l2.title) + 20
            parts.append(f"[具体内容: {l3.title}]\n{l3.content}")

        elif strategy == ContextStrategy.FULL:
            if l1:
                parts.append(f"[章节背景: {l1.title}]\n{l1.content}")
                remaining_chars -= len(l1.content) + len(l1.title) + 20
            if l2:
                parts.append(f"[小节上下文: {l2.title}]\n{l2.content}")
                remaining_chars -= len(l2.content) + len(l2.title) + 20
            parts.append(f"[具体内容: {l3.title}]\n{l3.content}")

        context = "\n\n---\n\n".join(parts)

        max_chars = int(self.max_context_tokens * self._chars_per_token)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[...上下文已截断]"

        return context

    def _summarize_for_context(self, node: ChunkNode, max_chars: int = 600) -> str:
        if len(node.content) <= max_chars:
            return node.content
        return node.content[:max_chars] + "..."

    def retrieve_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        context_chain = self.storage.get_l3_context_chain(node_id)
        if not context_chain.get("l3"):
            return None

        context = self._assemble_context(context_chain, self.strategy)
        l3 = context_chain["l3"]

        return {
            "l3_node_id": l3.node_id,
            "l3_content": l3.content,
            "l3_title": l3.title,
            "assembled_context": context,
            "context_chain": {
                "l1_title": context_chain["l1"].title if context_chain["l1"] else None,
                "l2_title": context_chain["l2"].title if context_chain["l2"] else None,
                "l3_title": l3.title,
            },
        }

    def get_sibling_context(self, node_id: str) -> List[Dict[str, Any]]:
        node = self.storage.get_node(node_id)
        if not node or not node.parent_id:
            return []

        parent = self.storage.get_node(node.parent_id)
        if not parent:
            return []

        siblings = []
        for cid in parent.children_ids:
            if cid != node_id:
                child = self.storage.get_node(cid)
                if child:
                    siblings.append(
                        {
                            "node_id": child.node_id,
                            "title": child.title,
                            "content": child.content[:200],
                        }
                    )
        return siblings
