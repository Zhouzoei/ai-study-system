import math
import re
import threading
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Callable

from core.tree_storage import TreeStorage
from config import config

_jieba_lock = threading.Lock()


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        self.idf: Dict[str, float] = {}
        self.token_freqs_list: List[Counter] = []
        self._domain_dict_loaded = False

    def build(self, documents: List[Dict]):
        self.corpus = documents
        self.n_docs = len(documents)
        self.doc_lens = []
        self.doc_freqs = defaultdict(int)
        self.token_freqs_list = []

        if not self._domain_dict_loaded:
            self._build_domain_dict(documents)
            self._domain_dict_loaded = True

        for doc in documents:
            tokens = self.tokenize(doc["content"])
            self.doc_lens.append(len(tokens))
            tf_counter = Counter(tokens)
            self.token_freqs_list.append(tf_counter)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        self.avgdl = sum(self.doc_lens) / self.n_docs if self.n_docs > 0 else 0

        for token, df in self.doc_freqs.items():
            self.idf[token] = math.log(
                (self.n_docs - df + 0.5) / (df + 0.5) + 1
            )

    def _build_domain_dict(self, documents: List[Dict]):
        import jieba
        text = " ".join(d.get("content", "") for d in documents)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]+", text)

        bigram_counts = defaultdict(int)
        trigram_counts = defaultdict(int)
        char_counts = defaultdict(int)

        for segment in chinese_chars:
            for i in range(len(segment)):
                char_counts[segment[i]] += 1
                if i + 1 < len(segment):
                    bigram_counts[segment[i:i+2]] += 1
                if i + 2 < len(segment):
                    trigram_counts[segment[i:i+3]] += 1

        total_bigrams = sum(bigram_counts.values()) or 1
        min_freq = max(3, int(len(documents) * 0.05))

        candidates = []
        for ngram, freq in bigram_counts.items():
            if freq < min_freq:
                continue
            c1, c2 = ngram[0], ngram[1]
            p_both = freq / total_bigrams
            p_c1 = char_counts.get(c1, 1) / total_bigrams
            p_c2 = char_counts.get(c2, 1) / total_bigrams
            if p_c1 * p_c2 > 0:
                pmi = math.log2(p_both / (p_c1 * p_c2))
                if pmi > 1.5:
                    candidates.append((ngram, freq, pmi))

        with _jieba_lock:
            for ngram, freq, _ in candidates:
                jieba.add_word(ngram, freq=freq, tag="n")

            trigram_candidates = []
            for ngram, freq in trigram_counts.items():
                if freq < min_freq:
                    continue
                bi1 = bigram_counts.get(ngram[:2], 0)
                bi2 = bigram_counts.get(ngram[1:], 0)
                if bi1 > 0 and bi2 > 0 and freq >= min(bi1, bi2) * 0.8:
                    trigram_candidates.append((ngram, freq))

            for ngram, freq in trigram_candidates:
                jieba.add_word(ngram, freq=freq, tag="n")

    def tokenize(self, text: str) -> List[str]:
        import jieba
        text = text.lower()
        tokens = []
        for token in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text):
            if re.match(r"^[\u4e00-\u9fff]+$", token):
                with _jieba_lock:
                    tokens.extend(jieba.lcut(token))
            else:
                tokens.append(token)
        return tokens

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        query_tokens = self.tokenize(query)
        scores = []

        for idx, doc in enumerate(self.corpus):
            token_freqs = self.token_freqs_list[idx] if idx < len(self.token_freqs_list) else Counter()
            score = 0.0

            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = token_freqs.get(token, 0)
                idf = self.idf[token]
                dl = self.doc_lens[idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * numerator / denominator

            if score > 0:
                scores.append(
                    {
                        "node_id": doc["node_id"],
                        "content": doc["content"],
                        "title": doc.get("title", ""),
                        "parent_id": doc.get("parent_id", ""),
                        "score": score,
                        "source": "bm25",
                    }
                )

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    def __init__(
        self,
        tree_storage: TreeStorage,
        embed_func: Optional[Callable] = None,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        rrf_k: int = 60,
        final_top_k: int = 5,
    ):
        self.storage = tree_storage
        self.embed_func = embed_func
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k
        self.bm25_index = BM25Index()
        self._bm25_built = False

    def build_bm25_index(self, doc_id: Optional[str] = None):
        documents = self.storage.get_all_l3_content(doc_id)
        if documents:
            self.bm25_index.build(documents)
            self._bm25_built = True

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_id: Optional[str] = None,
        use_hybrid: bool = True,
    ) -> List[Dict]:
        top_k = top_k or self.final_top_k

        if not use_hybrid or not self._bm25_built:
            return self._vector_search(query, top_k, doc_id)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self.bm25_index.search, query, self.bm25_top_k)
            vector_future = executor.submit(self._vector_search, query, self.vector_top_k, doc_id)
            bm25_results = bm25_future.result()
            vector_results = vector_future.result()

        fused = self._rrf_fuse(bm25_results, vector_results)
        return fused[:top_k]

    def _vector_search(
        self, query: str, top_k: int, doc_id: Optional[str] = None
    ) -> List[Dict]:
        if not self.embed_func:
            return []

        query_vectors = self.embed_func([query])
        if not query_vectors or len(query_vectors) == 0:
            return []
        query_vector = query_vectors[0]
        results = self.storage.search_l3_by_vector(query_vector, top_k, doc_id)

        return [
            {
                "node_id": r["node_id"],
                "content": r["content"],
                "title": r.get("title", ""),
                "parent_id": r.get("parent_id", ""),
                "score": r["score"],
                "source": "vector",
            }
            for r in results
        ]

    def _rrf_fuse(
        self, bm25_results: List[Dict], vector_results: List[Dict]
    ) -> List[Dict]:
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_info: Dict[str, Dict] = {}

        for rank, result in enumerate(bm25_results):
            node_id = result["node_id"]
            rrf_scores[node_id] += 1.0 / (rank + 1 + self.rrf_k)
            if node_id not in doc_info:
                doc_info[node_id] = result

        for rank, result in enumerate(vector_results):
            node_id = result["node_id"]
            rrf_scores[node_id] += 1.0 / (rank + 1 + self.rrf_k)
            if node_id not in doc_info:
                doc_info[node_id] = result

        fused = []
        for node_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            info = doc_info[node_id].copy()
            info["rrf_score"] = score
            info["source"] = "hybrid"
            fused.append(info)

        return fused
