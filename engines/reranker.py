from typing import List, Dict, Any, Optional

from config import config


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_k: int = 5,
        use_llm_fallback: bool = True,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.use_llm_fallback = use_llm_fallback
        self.model = None
        self._load_attempted = False

    def _ensure_model(self):
        if self.model is not None or self._load_attempted:
            return
        self._load_attempted = True
        try:
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["CURL_CA_BUNDLE"] = ""
            os.environ["REQUESTS_CA_BUNDLE"] = ""
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            print(f"[Reranker] Cross-Encoder model loaded: {self.model_name}")
        except ImportError:
            print("[Reranker] sentence-transformers not installed, using LLM fallback")
            self.model = None
        except Exception as e:
            print(f"[Reranker] Model load failed: {type(e).__name__}, using LLM fallback")
            self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        top_k = top_k or self.top_k
        self._ensure_model()

        if self.model:
            return self._rerank_with_model(query, candidates, top_k)
        elif self.use_llm_fallback:
            return self._rerank_with_llm(query, candidates, top_k)
        else:
            return candidates[:top_k]

    def _rerank_with_model(
        self, query: str, candidates: List[Dict], top_k: int
    ) -> List[Dict]:
        pairs = [(query, c["content"]) for c in candidates]
        scores = self.model.predict(pairs)

        scored_candidates = []
        for i, candidate in enumerate(candidates):
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = float(scores[i])
            candidate_copy["original_score"] = candidate.get("score", 0)
            candidate_copy["source"] = "reranked"
            scored_candidates.append(candidate_copy)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]

    def _rerank_with_llm(
        self, query: str, candidates: List[Dict], top_k: int
    ) -> List[Dict]:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_BASE_URL,
            )
        except (ImportError, Exception) as e:
            print(f"[Reranker] LLM fallback unavailable: {e}, using keyword fallback")
            return self._rerank_with_keyword(query, candidates, top_k)

        scored_candidates = []
        batch_size = 5

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            prompt = self._build_rerank_prompt(query, batch)

            try:
                response = client.chat.completions.create(
                    model=config.LLM_MODEL_ID,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100,
                )
                text = response.choices[0].message.content
                scores = self._parse_rerank_response(text, len(batch))
            except Exception:
                scores = [0.5] * len(batch)

            for j, candidate in enumerate(batch):
                candidate_copy = candidate.copy()
                candidate_copy["rerank_score"] = scores[j] if j < len(scores) else 0.5
                candidate_copy["original_score"] = candidate.get("score", 0)
                candidate_copy["source"] = "reranked_llm"
                scored_candidates.append(candidate_copy)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]

    def _build_rerank_prompt(self, query: str, candidates: List[Dict]) -> str:
        candidates_text = ""
        for i, c in enumerate(candidates):
            content = c["content"][:300]
            candidates_text += f"\n[文档{i+1}]: {content}\n"

        return f"""请对以下文档与查询的相关性进行打分，分数范围0-10。

查询: {query}
{candidates_text}

请只输出分数，用逗号分隔，不要其他内容。例如: 8,5,3,7,6"""

    def _parse_rerank_response(self, response: str, expected_count: int) -> List[float]:
        try:
            clean = response.strip().strip("[]")
            parts = clean.split(",")
            scores = []
            for p in parts:
                p = p.strip()
                try:
                    score = float(p)
                    scores.append(min(max(score / 10.0, 0), 1.0))
                except ValueError:
                    scores.append(0.5)
            while len(scores) < expected_count:
                scores.append(0.5)
            return scores[:expected_count]
        except Exception:
            return [0.5] * expected_count

    def _rerank_with_keyword(
        self, query: str, candidates: List[Dict], top_k: int
    ) -> List[Dict]:
        query_chars = set(query)
        scored_candidates = []
        for candidate in candidates:
            content = candidate.get("content", "")
            overlap = len(query_chars & set(content))
            score = min(overlap / max(len(query_chars), 1), 1.0)
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = score
            candidate_copy["original_score"] = candidate.get("score", 0)
            candidate_copy["source"] = "reranked_keyword"
            scored_candidates.append(candidate_copy)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]
