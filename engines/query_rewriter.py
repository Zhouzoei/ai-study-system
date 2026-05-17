import re
from typing import List, Optional, Callable, Dict


class QueryRewriter:
    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm_func = llm_func

    def rewrite(
        self,
        query: str,
        strategies: Optional[List[str]] = None,
        num_queries: int = 3,
    ) -> List[str]:
        if strategies is None:
            strategies = ["expand"]

        rewritten = [query]

        for strategy in strategies:
            if strategy == "expand":
                expanded = self.expand_query(query, num_queries)
                rewritten.extend(expanded)
            elif strategy == "decompose":
                decomposed = self.decompose_query(query)
                rewritten.extend(decomposed)
            elif strategy == "hyde":
                hyde = self.hyde_query(query)
                if hyde and hyde.strip() and hyde.strip() != query.strip():
                    rewritten.append(hyde.strip())

        seen = set()
        unique = []
        for q in rewritten:
            normalized = re.sub(r"\s+", "", q.strip().lower())
            if normalized not in seen:
                seen.add(normalized)
                unique.append(q.strip())

        return unique[:num_queries * 2 + 1]

    def expand_query(self, query: str, num_queries: int = 3) -> List[str]:
        if self.llm_func:
            return self._llm_expand(query, num_queries)
        return self._rule_based_expand(query)

    def decompose_query(self, query: str) -> List[str]:
        if self.llm_func:
            return self._llm_decompose(query)

        parts = re.split(r"[，,。.？?！!；;]", query)
        parts = [p.strip() for p in parts if len(p.strip()) > 3]
        return parts if len(parts) > 1 else []

    def hyde_query(self, query: str) -> str:
        if not self.llm_func:
            return query

        prompt = (
            f"请针对以下问题生成一段假设性的详细回答，仿佛你正在回答这个问题。\n"
            f"这段回答将被用作检索的查询语句，请确保覆盖问题的关键概念和细节。\n\n"
            f"问题: {query}\n\n"
            f"假设性回答:"
        )
        try:
            response = self.llm_func(prompt)
            response = response.strip()
            if response:
                return response[:500]
            return query
        except Exception:
            return query

    def _llm_expand(self, query: str, num_queries: int = 3) -> List[str]:
        prompt = (
            f"请将以下用户查询改写成{num_queries}个不同的检索查询，每个查询从不同角度扩展原意。\n"
            f"要求：\n"
            f"1. 保持核心意图不变\n"
            f"2. 添加同义词、相关术语或下位概念\n"
            f"3. 中英文术语可以混用\n"
            f"4. 每行一个查询，不要编号\n\n"
            f"原始查询: {query}\n\n"
            f"改写后的查询:"
        )
        try:
            response = self.llm_func(prompt)
            lines = [
                line.strip().lstrip("0123456789.-) ")
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 3
            ]
            return lines[:num_queries]
        except Exception:
            return self._rule_based_expand(query)

    def _llm_decompose(self, query: str) -> List[str]:
        prompt = (
            f"以下是一个复杂查询，请将其分解成2-3个独立的子查询，每个子查询聚焦于一个方面。\n"
            f"每行一个子查询，不要编号。\n\n"
            f"原始查询: {query}\n\n"
            f"子查询:"
        )
        try:
            response = self.llm_func(prompt)
            lines = [
                line.strip().lstrip("0123456789.-) ")
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 3
            ]
            return lines[:3]
        except Exception:
            return []

    def _rule_based_expand(self, query: str) -> List[str]:
        expansions = []

        terms = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        en_terms = re.findall(r"[a-zA-Z]{2,}", query)

        if terms:
            expanded = " ".join(terms)
            expansions.append(expanded)

        if en_terms:
            expansions.append(query + " " + " ".join(en_terms))

        return expansions[:2]
