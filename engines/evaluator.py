import json
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class EvalSample:
    question: str = ""
    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    ground_truth: str = ""
    doc_id: str = ""


@dataclass
class EvalResult:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    num_contexts: int = 0
    context_total_chars: int = 0


class RAGASEvaluator:
    def __init__(self, llm_func: Optional[Callable] = None, embed_func: Optional[Callable] = None):
        self.llm_func = llm_func
        self.embed_func = embed_func

    def evaluate_single(self, sample: EvalSample) -> EvalResult:
        result = EvalResult(
            num_contexts=len(sample.contexts),
            context_total_chars=sum(len(c) for c in sample.contexts),
        )

        if self.llm_func:
            result.faithfulness = self._compute_faithfulness(sample)
            result.answer_relevancy = self._compute_answer_relevancy(sample)
            result.context_precision = self._compute_context_precision(sample)

        if self.llm_func and sample.ground_truth:
            result.context_recall = self._compute_context_recall(sample)

        return result

    def evaluate_batch(self, samples: List[EvalSample]) -> Dict[str, Any]:
        results = []
        for sample in samples:
            result = self.evaluate_single(sample)
            results.append(result)

        if not results:
            return {"error": "no results"}

        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        avg_precision = sum(r.context_precision for r in results) / len(results)
        avg_recall = sum(r.context_recall for r in results if r.context_recall > 0) or 0
        recall_count = sum(1 for r in results if r.context_recall > 0)
        if recall_count > 0:
            avg_recall = sum(r.context_recall for r in results if r.context_recall > 0) / recall_count

        return {
            "num_samples": len(results),
            "faithfulness": round(avg_faithfulness, 4),
            "answer_relevancy": round(avg_relevancy, 4),
            "context_precision": round(avg_precision, 4),
            "context_recall": round(avg_recall, 4),
            "avg_num_contexts": round(sum(r.num_contexts for r in results) / len(results), 1),
            "avg_context_chars": round(sum(r.context_total_chars for r in results) / len(results), 0),
        }

    def _compute_faithfulness(self, sample: EvalSample) -> float:
        if not sample.contexts or not sample.answer:
            return 0.0

        context_text = "\n".join(sample.contexts)
        prompt = f"""请判断以下答案是否忠实于给定的上下文信息。只根据上下文判断，不要使用外部知识。

上下文:
{context_text}

答案:
{sample.answer}

请按以下格式输出:
1. 提取答案中的所有声明(claim)
2. 对每个声明，判断是否能从上下文中推导出来
3. 计算忠实度 = 可推导的声明数 / 总声明数

只输出一个0到1之间的数字表示忠实度:"""

        try:
            response = self.llm_func(prompt)
            score = self._extract_score(response)
            return score
        except Exception:
            return 0.0

    def _compute_answer_relevancy(self, sample: EvalSample) -> float:
        if not sample.answer or not sample.question:
            return 0.0

        prompt = f"""请评估以下答案与问题的相关性。答案应该直接回答问题，不包含无关信息。

问题: {sample.question}
答案: {sample.answer}

请输出0到1之间的相关性分数(1=完全相关,0=完全不相关):
只输出一个数字:"""

        try:
            response = self.llm_func(prompt)
            score = self._extract_score(response)
            return score
        except Exception:
            return 0.0

    def _compute_context_precision(self, sample: EvalSample) -> float:
        if not sample.contexts or not sample.question:
            return 0.0

        contexts_with_idx = []
        for i, ctx in enumerate(sample.contexts):
            contexts_with_idx.append(f"[上下文{i+1}]: {ctx[:500]}")

        prompt = f"""请评估以下每个上下文片段对回答问题的有用程度。

问题: {sample.question}

{chr(10).join(contexts_with_idx)}

对每个上下文，输出1(有用)或0(无用)，用逗号分隔。
只输出数字和逗号，例如: 1,0,1,1,0"""

        try:
            response = self.llm_func(prompt)
            relevance_labels = self._parse_binary_labels(response, len(sample.contexts))

            if not relevance_labels:
                return 0.0

            precision_at_k = []
            relevant_count = 0
            for k, label in enumerate(relevance_labels):
                if label == 1:
                    relevant_count += 1
                    precision_at_k.append(relevant_count / (k + 1))

            if relevant_count == 0:
                return 0.0

            return sum(precision_at_k) / relevant_count
        except Exception:
            return 0.0

    def _compute_context_recall(self, sample: EvalSample) -> float:
        if not sample.contexts or not sample.ground_truth:
            return 0.0

        context_text = "\n".join(sample.contexts)
        prompt = f"""请判断标准答案中的每个关键信息点是否能在上下文中找到。

上下文:
{context_text}

标准答案:
{sample.ground_truth}

请输出0到1之间的召回率分数(能找到的关键信息点比例):
只输出一个数字:"""

        try:
            response = self.llm_func(prompt)
            score = self._extract_score(response)
            return score
        except Exception:
            return 0.0

    def _extract_score(self, response: str) -> float:
        response = response.strip()
        if not response:
            return 0.5

        import re
        numbers = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", response)
        if numbers:
            try:
                score = float(numbers[0])
                return min(max(score, 0.0), 1.0)
            except ValueError:
                pass

        numbers = re.findall(r"\d+\.\d+", response)
        if numbers:
            try:
                score = float(numbers[0])
                return min(max(score, 0.0), 1.0)
            except ValueError:
                pass

        return 0.5

    def _parse_binary_labels(self, response: str, expected_count: int) -> List[int]:
        import re
        numbers = re.findall(r"[01]", response)
        labels = [int(n) for n in numbers]
        while len(labels) < expected_count:
            labels.append(0)
        return labels[:expected_count]


class RetrievalTracer:
    def __init__(self):
        self.traces: List[Dict] = []

    def trace_retrieval(
        self,
        query: str,
        strategy: str,
        results: List[Dict],
        latency_ms: float,
        extra: Optional[Dict] = None,
    ):
        trace = {
            "query": query,
            "strategy": strategy,
            "num_results": len(results),
            "top_scores": [r.get("score", r.get("rrf_score", r.get("rerank_score", 0))) for r in results[:3]],
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        }
        if extra:
            trace.update(extra)
        self.traces.append(trace)

    def get_summary(self) -> Dict[str, Any]:
        if not self.traces:
            return {}

        by_strategy = {}
        for trace in self.traces:
            strategy = trace["strategy"]
            if strategy not in by_strategy:
                by_strategy[strategy] = {
                    "count": 0,
                    "total_latency": 0,
                    "scores": [],
                }
            by_strategy[strategy]["count"] += 1
            by_strategy[strategy]["total_latency"] += trace["latency_ms"]
            by_strategy[strategy]["scores"].extend(trace["top_scores"])

        summary = {}
        for strategy, data in by_strategy.items():
            avg_latency = data["total_latency"] / data["count"]
            avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
            summary[strategy] = {
                "count": data["count"],
                "avg_latency_ms": round(avg_latency, 2),
                "avg_top_score": round(avg_score, 4),
            }

        return summary

    def export_traces(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.traces, f, ensure_ascii=False, indent=2)
