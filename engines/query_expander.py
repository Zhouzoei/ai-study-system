import re
from typing import List, Optional, Dict, Any


ABBREVIATION_MAP = {
    "sgd": "随机梯度下降 stochastic gradient descent",
    "adam": "adam优化器 adaptive moment estimation",
    "rmsprop": "均方根传播 root mean square propagation",
    "relu": "线性整流单元 rectified linear unit",
    "cnn": "卷积神经网络 convolutional neural network",
    "rnn": "循环神经网络 recurrent neural network",
    "lstm": "长短期记忆网络 long short-term memory",
    "gru": "门控循环单元 gated recurrent unit",
    "gan": "生成对抗网络 generative adversarial network",
    "vae": "变分自编码器 variational autoencoder",
    "svm": "支持向量机 support vector machine",
    "knn": "k近邻 k-nearest neighbors",
    "pca": "主成分分析 principal component analysis",
    "nlp": "自然语言处理 natural language processing",
    "cv": "计算机视觉 computer vision",
    "ml": "机器学习 machine learning",
    "dl": "深度学习 deep learning",
    "ai": "人工智能 artificial intelligence",
    "bptt": "随时间反向传播 backpropagation through time",
    "bp": "反向传播 backpropagation",
    "bn": "批归一化 batch normalization",
    "ln": "层归一化 layer normalization",
    "mmr": "最大边际相关性 maximum marginal relevance",
    "hyde": "假设文档嵌入 hypothetical document embeddings",
    "bm25": "bm25检索算法 okapi bm25",
    "rrf": "倒数排序融合 reciprocal rank fusion",
    "resnet": "残差网络 residual network",
    "vit": "视觉transformer vision transformer",
    "bert": "bert预训练模型 bidirectional encoder representations from transformers",
    "gpt": "gpt生成式预训练transformer generative pre-trained transformer",
    "ner": "命名实体识别 named entity recognition",
    "pos": "词性标注 part-of-speech tagging",
}

INFIX_ABBR_PATTERN = re.compile(r"\b([A-Za-z]{2,6})\b", re.IGNORECASE)


class QueryExpander:
    def __init__(self, knowledge_graph=None):
        self.knowledge_graph = knowledge_graph
        self.abbreviation_map = ABBREVIATION_MAP

    def expand(self, query: str, strategies: List[str] = None) -> List[str]:
        if strategies is None:
            strategies = ["abbr"]

        expanded = []

        if "abbr" in strategies:
            abbr_expanded = self.expand_abbr(query)
            expanded.extend(abbr_expanded)

        if "kg" in strategies and self.knowledge_graph:
            kg_expanded = self.expand_with_kg(query)
            expanded.extend(kg_expanded)

        seen = set()
        unique = []
        for q in [query] + expanded:
            normalized = re.sub(r"\s+", "", q.strip().lower())
            if normalized not in seen:
                seen.add(normalized)
                unique.append(q.strip())

        return unique[1:]

    def expand_abbr(self, query: str) -> List[str]:
        matches = INFIX_ABBR_PATTERN.findall(query)
        if not matches:
            return []

        found = []
        for abbr in matches:
            key = abbr.lower()
            if key in self.abbreviation_map:
                found.append((abbr, self.abbreviation_map[key]))

        if not found:
            return []

        expansions = []
        full_expansion = query
        for abbr, meaning in found:
            full_expansion = full_expansion.replace(abbr, meaning, 1)
        if full_expansion != query:
            expansions.append(full_expansion)

            abbrs_only = " ".join(meaning for _, meaning in found)
            expansions.append(f"{query} ({abbrs_only})")

        return expansions

    def expand_with_kg(self, query: str) -> List[str]:
        if not self.knowledge_graph:
            return []

        terms = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        expanded_queries = []

        for term in terms[:2]:
            entity = self.knowledge_graph.query_entity(term)
            if not entity:
                continue

            relations = self.knowledge_graph.get_entity_relations(
                entity["entity_id"], depth=1
            )
            related_names = []
            for rel in relations.get("relations", [])[:3]:
                name = rel.get("name", "")
                if name and name != term and len(name) <= 20:
                    related_names.append(name)

            if related_names:
                expanded_queries.append(f"{query} {' '.join(related_names)}")

        return expanded_queries
