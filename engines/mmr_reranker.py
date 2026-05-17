import math
from typing import List, Dict, Any, Optional, Callable


class MMRReranker:
    def __init__(
        self,
        lambda_param: float = 0.7,
        top_k: int = 10,
        embed_func: Optional[Callable] = None,
    ):
        self.lambda_param = lambda_param
        self.top_k = top_k
        self.embed_func = embed_func

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k] if top_k else candidates

        top_k = top_k or self.top_k
        k = min(top_k, len(candidates))

        if self.embed_func:
            return self._mmr_with_embeddings(query, candidates, k)
        return self._mmr_with_text_overlap(candidates, k)

    def _mmr_with_embeddings(
        self, query: str, candidates: List[Dict], k: int
    ) -> List[Dict]:
        query_vec = self.embed_func([query])[0]

        cand_vecs = []
        for c in candidates:
            vec = self.embed_func([c["content"][:500]])[0]
            cand_vecs.append(vec)

        query_norm = self._l2_norm(query_vec)
        cand_norms = [self._l2_norm(v) for v in cand_vecs]

        query_sims = []
        for v in cand_norms:
            sim = sum(a * b for a, b in zip(query_norm, v))
            query_sims.append(sim)

        selected = []
        remaining = list(range(len(candidates)))

        first = max(remaining, key=lambda i: query_sims[i])
        selected.append(first)
        remaining.remove(first)

        while len(selected) < k and remaining:
            mmr_scores = []
            for i in remaining:
                rel = query_sims[i]
                sim_to_sel = max(
                    sum(a * b for a, b in zip(cand_norms[i], cand_norms[j]))
                    for j in selected
                )
                mmr_score = self.lambda_param * rel - (1 - self.lambda_param) * sim_to_sel
                mmr_scores.append((i, mmr_score))

            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_idx)
            remaining.remove(best_idx)

        result = []
        for idx in selected:
            c = dict(candidates[idx])
            c["mmr_score"] = query_sims[idx]
            c["mmr_rank"] = len(result)
            result.append(c)

        return result

    def _mmr_with_text_overlap(
        self, candidates: List[Dict], k: int
    ) -> List[Dict]:
        texts = [c["content"] for c in candidates]

        def jaccard_sim(a: str, b: str) -> float:
            set_a = set(a)
            set_b = set(b)
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)

        selected = []
        remaining = list(range(len(candidates)))
        ref_scores = [c.get("rrf_score", c.get("score", 0)) for c in candidates]

        if not any(ref_scores):
            ref_scores = [1.0 / (i + 1) for i in range(len(candidates))]

        max_score = max(ref_scores) if ref_scores else 1.0
        ref_scores = [s / max_score for s in ref_scores]

        first = max(remaining, key=lambda i: ref_scores[i])
        selected.append(first)
        remaining.remove(first)

        while len(selected) < k and remaining:
            mmr_scores = []
            for i in remaining:
                rel = ref_scores[i]
                sim_to_sel = max(jaccard_sim(texts[i], texts[j]) for j in selected)
                mmr_score = self.lambda_param * rel - (1 - self.lambda_param) * sim_to_sel
                mmr_scores.append((i, mmr_score))

            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_idx)
            remaining.remove(best_idx)

        result = []
        for idx in selected:
            c = dict(candidates[idx])
            c["mmr_score"] = ref_scores[idx]
            c["mmr_rank"] = len(result)
            result.append(c)

        return result

    @staticmethod
    def _l2_norm(vec):
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]