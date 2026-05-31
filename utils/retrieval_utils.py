from typing import List, Dict


def merge_retrieval_results(all_results: List[List[Dict]], top_k: int) -> List[Dict]:
    seen = {}
    for results in all_results:
        for r in results:
            node_id = r["node_id"]
            score_key = "rrf_score" if "rrf_score" in r else "score"
            score = r.get(score_key, 0)
            if node_id not in seen or score > seen[node_id].get(score_key, 0):
                if node_id not in seen:
                    seen[node_id] = dict(r)
                else:
                    seen[node_id].update(r)
                    seen[node_id][score_key] = score

    sorted_results = sorted(
        seen.values(),
        key=lambda x: x.get("rrf_score", x.get("score", 0)),
        reverse=True,
    )
    return sorted_results[:top_k]
