import json
import logging
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

RELEVANCE_SCORE_TEMPLATE = """判断以下【参考段落】是否对回答问题有帮助。

问题: {question}

参考段落: {passage}

如果该段落包含与问题相关的信息，回答"相关"；否则回答"不相关"。
只回答"相关"或"不相关"，不要其他内容。"""

QUALITY_CHECK_TEMPLATE = """评价以下 AI 回答的质量，按三项标准分别打分（0-10分）。

问题: {question}
回答: {answer}

【完整性】回答是否全面覆盖了问题的各个方面？
【依据性】回答是否严格基于提供的参考资料，没有编造信息？
【有用性】回答是否对用户有帮助、清晰易懂？

只输出 JSON 格式，不要其他内容:
{{
  "completeness": 7,
  "groundedness": 8,
  "usefulness": 9,
  "overall": 8.0,
  "issues": ["如果质量低，列出具体问题"],
  "suggested_fix": "如何改进这个回答"
}}"""


class SelfRAGCritic:
    """Self-RAG: relevance filtering before generation + quality check after.

    Before generation:
      - Score each retrieved context for relevance to the query
      - Drop passages below threshold

    After generation:
      - Check answer quality (completeness, groundedness, usefulness)
      - Optionally trigger regeneration with quality improvement hints
    """

    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm_func = llm_func
        self._relevance_threshold = 0.5
        self._min_overall_quality = 6.0

    # ── Pre-generation: relevance filter ──

    def filter_contexts(
        self,
        question: str,
        contexts: List[Dict],
    ) -> List[Dict]:
        if not contexts or not self.llm_func:
            return contexts

        scored = []
        for ctx in contexts:
            score = self._score_relevance(question, ctx)
            if score is not None:
                scored.append((score, ctx))
            else:
                scored.append((self._relevance_threshold, ctx))

        scored.sort(key=lambda x: x[0], reverse=True)

        kept = [ctx for score, ctx in scored if score >= self._relevance_threshold]
        dropped = len(scored) - len(kept)
        if dropped > 0:
            logger.info(f"SelfRAG: filtered {dropped}/{len(scored)} low-relevance passages")
        return kept

    def _score_relevance(self, question: str, ctx: Dict) -> Optional[float]:
        passage = ctx.get("excerpt") or ctx.get("content") or ctx.get("text", "")
        if not passage or len(passage) < 20:
            return 0.0

        try:
            short_passage = passage[:400]
            response = self.llm_func(
                RELEVANCE_SCORE_TEMPLATE.format(question=question, passage=short_passage)
            ).strip().lower()
            return 1.0 if "相关" in response else 0.0
        except Exception as e:
            logger.debug(f"SelfRAG relevance scoring failed: {e}")
            return None

    # ── Post-generation: quality check ──

    def check_quality(
        self,
        question: str,
        answer: str,
    ) -> Dict[str, Any]:
        if not self.llm_func or not answer or len(answer) < 20:
            return {"overall": 10.0, "needs_regenerate": False}

        try:
            response = self.llm_func(
                QUALITY_CHECK_TEMPLATE.format(question=question, answer=answer[:1500])
            ).strip()

            if response.startswith("```"):
                response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            result = json.loads(response)
            result["overall"] = float(result.get("overall", 7.0))
            result["needs_regenerate"] = result["overall"] < self._min_overall_quality
            return result
        except Exception as e:
            logger.debug(f"SelfRAG quality check failed: {e}")
            return {"overall": 7.0, "needs_regenerate": False}

    def build_quality_prompt_suffix(self, quality_result: Dict[str, Any]) -> str:
        if not quality_result.get("needs_regenerate"):
            return ""
        issues = quality_result.get("issues", [])
        fix = quality_result.get("suggested_fix", "")
        suffix = "\n\n[质量改进提示]\n"
        if issues:
            suffix += "之前回答的问题:\n" + "\n".join(f"- {i}" for i in issues[:3]) + "\n"
        if fix:
            suffix += f"改进方向: {fix}"
        return suffix
