import sys
import os
import time
import json
import pytest

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["QDRANT_URL"] = ""
os.environ["QDRANT_API_KEY"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hierarchical_chunker import HierarchicalChunker, ChunkNode
from core.tree_storage import TreeStorage
from engines.hierarchical_retriever import HierarchicalRetriever, ContextStrategy
from engines.hybrid_retriever import HybridRetriever, BM25Index
from engines.reranker import CrossEncoderReranker
from engines.evaluator import RAGASEvaluator, EvalSample, RetrievalTracer


def _safe_remove_db(db_path: str):
    """Remove a SQLite database file and its WAL/checkpoint siblings."""
    for suffix in ("", "-shm", "-wal"):
        p = db_path + suffix
        if os.path.exists(p):
            try:
                os.remove(p)
            except PermissionError:
                time.sleep(0.1)
                try:
                    os.remove(p)
                except Exception:
                    pass


SAMPLE_DOCUMENT = """# 机器学习基础

## 2.1 监督学习

监督学习是机器学习中最常见的范式。它从带标签的训练数据中学习输入到输出的映射关系。

### 2.1.1 线性回归

线性回归是最基础的回归算法。它假设输入特征与输出之间存在线性关系，通过最小化均方误差来求解最优参数。

损失函数定义为：L(w) = (1/n) * sum(yi - wTxi)^2

其中 w 是权重向量，xi 是输入特征，yi 是真实标签。通过梯度下降法可以求解最优权重。

### 2.1.2 逻辑回归

逻辑回归虽然名字中有回归，但实际上是分类算法。它通过sigmoid函数将线性组合映射到0到1区间，用于二分类问题。

sigmoid函数定义：sigma(z) = 1 / (1 + e^(-z))

逻辑回归的决策边界是线性的。对于非线性可分问题，可以通过特征变换或使用核技巧来扩展。

### 2.1.3 支持向量机

支持向量机SVM寻找最大化间隔的超平面。它只依赖少数支持向量，具有良好的泛化能力。

对于线性不可分问题，SVM通过核技巧将数据映射到高维空间。常用核函数包括RBF核、多项式核和sigmoid核。

SVM的正则化参数C控制间隔与分类错误的权衡。C越大模型越倾向于正确分类所有训练样本，可能导致过拟合。

## 2.2 无监督学习

无监督学习处理没有标签的数据，目标是发现数据中的内在结构和模式。

### 2.2.1 K-Means聚类

K-Means是最经典的聚类算法。它将数据划分为K个簇，每个簇由其质心表示。

算法步骤：1.随机初始化K个质心 2.将每个样本分配到最近的质心 3.重新计算每个簇的质心 4.重复步骤2-3直到收敛

K-Means的缺点是需要预先指定K值，且对初始化敏感。K-Means++改进了初始化策略。

### 2.2.2 主成分分析

主成分分析PCA是最常用的降维方法。它通过正交变换将高维数据投影到方差最大的方向上。

PCA的计算步骤：1.对数据进行中心化处理 2.计算协方差矩阵 3.对协方差矩阵进行特征值分解 4.选择前k个最大特征值对应的特征向量作为主成分

PCA的本质是找到数据方差最大的投影方向，保留最多的信息量。

### 2.2.3 自编码器

自编码器是神经网络架构的无监督学习方法。它由编码器和解码器组成，目标是重建输入数据。

编码器将输入压缩为低维表示即瓶颈层，解码器从低维表示重建原始输入。瓶颈层迫使网络学习数据的最重要特征。

变分自编码器VAE在瓶颈层引入概率分布，可以生成新样本。

## 2.3 深度学习

深度学习是机器学习的子领域，使用多层神经网络学习数据的层次化表示。

### 2.3.1 反向传播算法

反向传播是训练神经网络的核心算法。它通过链式法则计算损失函数对每个参数的梯度。

计算图将复合函数分解为基本操作的有向无环图。链式法则允许我们逐层计算梯度。前向传播按拓扑序计算每个节点的值，反向传播按逆拓扑序计算每个节点的梯度。

梯度计算步骤：1.前向传播计算预测值 2.计算损失函数 3.反向传播逐层计算梯度 4.参数更新

常见问题与解决方案：梯度消失使用ReLU激活函数和残差连接和批量归一化。梯度爆炸使用梯度裁剪和权重初始化。过拟合使用Dropout和正则化和数据增强和早停。

### 2.3.2 卷积神经网络

卷积神经网络CNN专门处理具有网格结构的数据如图像。

CNN的核心组件包括卷积层提取局部特征，池化层降低空间维度增强平移不变性，全连接层将特征映射到输出空间。

经典架构包括LeNet和AlexNet和VGG和ResNet等。ResNet通过残差连接解决了深层网络的退化问题。

### 2.3.3 循环神经网络

循环神经网络RNN处理序列数据，通过隐藏状态传递历史信息。

标准RNN存在梯度消失问题，难以捕捉长距离依赖。LSTM通过门控机制即遗忘门和输入门和输出门解决了这个问题。

GRU是LSTM的简化版本，只有重置门和更新门，计算效率更高但表达能力相近。

Transformer架构完全基于注意力机制，摒弃了循环结构，支持并行计算，已成为NLP领域的主流架构。
"""

EVAL_QUESTIONS = [
    {
        "question": "逻辑回归和线性回归有什么区别？",
        "ground_truth": "线性回归是回归算法，假设输入与输出有线性关系，最小化均方误差。逻辑回归是分类算法，通过sigmoid函数映射到[0,1]区间用于二分类。",
    },
    {
        "question": "反向传播中梯度消失怎么解决？",
        "ground_truth": "梯度消失的解决方案包括：使用ReLU激活函数、残差连接、批量归一化。",
    },
    {
        "question": "K-Means聚类算法的步骤是什么？",
        "ground_truth": "K-Means步骤：1.随机初始化K个质心 2.将每个样本分配到最近的质心 3.重新计算每个簇的质心 4.重复步骤2-3直到收敛。",
    },
    {
        "question": "PCA降维的计算步骤是什么？",
        "ground_truth": "PCA步骤：1.数据中心化 2.计算协方差矩阵 3.特征值分解 4.选择前k个最大特征值对应的特征向量作为主成分。",
    },
    {
        "question": "ResNet解决了什么问题？",
        "ground_truth": "ResNet通过残差连接解决了深层网络的退化问题。",
    },
    {
        "question": "LSTM和GRU的区别是什么？",
        "ground_truth": "LSTM有三个门（遗忘门、输入门、输出门），GRU只有两个门（重置门和更新门）。GRU计算效率更高但表达能力相近。",
    },
]

def _build_nodes():
    print("=" * 60)
    print("Test 1: Hierarchical Chunker")
    print("=" * 60)

    chunker = HierarchicalChunker(
        l1_max_size=2000,
        l2_max_size=500,
        l3_max_size=150,
        overlap=30,
    )

    nodes = chunker.chunk(SAMPLE_DOCUMENT, doc_id="ml_basics")

    level_counts = {}
    for n in nodes:
        level_counts[n.level] = level_counts.get(n.level, 0) + 1

    print(f"\nTotal nodes: {len(nodes)}")
    print(f"Level distribution: L1={level_counts.get(1,0)}, L2={level_counts.get(2,0)}, L3={level_counts.get(3,0)}")

    l1 = [n for n in nodes if n.level == 1]
    l2 = [n for n in nodes if n.level == 2]
    l3 = [n for n in nodes if n.level == 3]

    print("\n--- Tree Structure ---")
    for n1 in l1:
        print(f"\n[L1] {n1.title} (children: {len(n1.children_ids)})")
        for cid in n1.children_ids:
            n2 = next((n for n in l2 if n.node_id == cid), None)
            if n2:
                print(f"  [L2] {n2.title} (children: {len(n2.children_ids)})")
                for cid2 in n2.children_ids:
                    n3 = next((n for n in l3 if n.node_id == cid2), None)
                    if n3:
                        print(f"    [L3] {n3.title[:50]}... ({len(n3.content)}chars)")

    orphan = 0
    for n in nodes:
        if n.parent_id:
            parent = next((p for p in nodes if p.node_id == n.parent_id), None)
            if not parent:
                orphan += 1

    assert orphan == 0, f"Found {orphan} orphan nodes!"
    assert all(n.parent_id for n in l3), "Some L3 nodes have no parent!"
    assert all(n.parent_id for n in l2), "Some L2 nodes have no parent!"
    print(f"\n[PASS] Orphan nodes: {orphan}")
    print(f"[PASS] All L3 have parent: True")
    print(f"[PASS] All L2 have parent: True")

    return nodes


@pytest.fixture
def nodes():
    return _build_nodes()


def test_storage(nodes):
    print("\n" + "=" * 60)
    print("Test 2: Tree Storage (SQLite)")
    print("=" * 60)

    db_path = "test_tree_store.db"
    if os.path.exists(db_path):
            _safe_remove_db(db_path)

    storage = TreeStorage(
        sqlite_path=db_path,
        qdrant_url="",
        qdrant_api_key="",
        collection_name="test_collection",
    )

    for node in nodes:
        storage._store_sqlite_node(node)

    l1_nodes = storage.get_nodes_by_level(1)
    l2_nodes = storage.get_nodes_by_level(2)
    l3_nodes = storage.get_nodes_by_level(3)

    print(f"\nStored nodes: L1={len(l1_nodes)}, L2={len(l2_nodes)}, L3={len(l3_nodes)}")
    assert len(l1_nodes) + len(l2_nodes) + len(l3_nodes) == len(nodes), "Node count mismatch!"

    print("\n--- Context Chain Test ---")
    if l3_nodes:
        test_l3 = l3_nodes[0]
        chain = storage.get_l3_context_chain(test_l3.node_id)
        print(f"  L3: {test_l3.title[:40]}...")
        if chain["l2"]:
            print(f"  L2 parent: {chain['l2'].title}")
        if chain["l1"]:
            print(f"  L1 grandparent: {chain['l1'].title}")
        assert chain["l2"] is not None, "L3 should have L2 parent!"
        assert chain["l1"] is not None, "L2 should have L1 parent!"

    print("\n--- Ancestor Traversal Test ---")
    if l3_nodes:
        ancestors = storage.get_ancestors(l3_nodes[0].node_id)
        print(f"  Ancestors count: {len(ancestors)}")
        for a in ancestors:
            print(f"    L{a.level}: {a.title}")
        assert len(ancestors) >= 2, "L3 should have at least 2 ancestors (L2 + L1)"

    stats = storage.get_stats()
    print(f"\nStorage stats: {json.dumps(stats, ensure_ascii=False)}")

    storage.db.close()
    _safe_remove_db(db_path)

    print("\n[PASS] Tree Storage test passed")
    return True


def test_retriever(nodes):
    print("\n" + "=" * 60)
    print("Test 3: Hierarchical Retriever")
    print("=" * 60)

    db_path = "test_retriever.db"
    if os.path.exists(db_path):
            _safe_remove_db(db_path)

    storage = TreeStorage(
        sqlite_path=db_path,
        qdrant_url="",
        qdrant_api_key="",
        collection_name="test_collection",
    )

    for node in nodes:
        storage._store_sqlite_node(node)

    for strategy in [ContextStrategy.CONSERVATIVE, ContextStrategy.BALANCED, ContextStrategy.FULL]:
        print(f"\n--- Strategy: {strategy.value} ---")
        retriever = HierarchicalRetriever(
            tree_storage=storage,
            strategy=strategy,
        )

        l3_nodes = storage.get_nodes_by_level(3)
        if l3_nodes:
            test_l3 = l3_nodes[0]
            mock_results = [{
                "node_id": test_l3.node_id,
                "content": test_l3.content,
                "title": test_l3.title,
                "score": 0.92,
            }]
            enriched = retriever.retrieve_with_context(mock_results)
            if enriched:
                ctx = enriched[0]["assembled_context"]
                chain = enriched[0]["context_chain"]
                print(f"  Context length: {len(ctx)} chars")
                print(f"  Chain: L1={chain['l1_title']} > L2={chain['l2_title']} > L3={chain['l3_title'][:30]}...")
                assert len(ctx) > 0, "Assembled context should not be empty!"

    storage.db.close()
    _safe_remove_db(db_path)

    print("\n[PASS] Hierarchical Retriever test passed")
    return True


def test_bm25(nodes):
    print("\n" + "=" * 60)
    print("Test 4: BM25 Search")
    print("=" * 60)

    bm25 = BM25Index()
    documents = [
        {"node_id": n.node_id, "content": n.content, "title": n.title}
        for n in nodes if n.level == 3
    ]
    bm25.build(documents)

    queries = [
        "梯度消失怎么解决",
        "PCA降维步骤",
        "LSTM和GRU区别",
        "K-Means聚类",
        "线性回归损失函数",
    ]

    for query in queries:
        results = bm25.search(query, top_k=3)
        print(f"\n  Query: {query}")
        for r in results:
            print(f"    score={r['score']:.4f} | {r['content'][:60]}...")
        assert len(results) > 0, f"BM25 should return results for query: {query}"

    print("\n[PASS] BM25 Search test passed")
    return True


def test_hybrid_retriever(nodes):
    print("\n" + "=" * 60)
    print("Test 5: Hybrid Retriever (BM25 only, no vector)")
    print("=" * 60)

    db_path = "test_hybrid.db"
    if os.path.exists(db_path):
            _safe_remove_db(db_path)

    storage = TreeStorage(
        sqlite_path=db_path,
        qdrant_url="",
        qdrant_api_key="",
        collection_name="test_collection",
    )

    for node in nodes:
        storage._store_sqlite_node(node)

    hybrid = HybridRetriever(
        tree_storage=storage,
        embed_func=None,
        bm25_top_k=10,
        vector_top_k=10,
        rrf_k=60,
        final_top_k=5,
    )
    hybrid.build_bm25_index()

    queries = ["梯度消失怎么解决", "PCA降维步骤", "ResNet残差连接"]
    for query in queries:
        results = hybrid.search(query, use_hybrid=True)
        print(f"\n  Query: {query}")
        print(f"  Results: {len(results)}")
        for r in results:
            source = r.get("source", "unknown")
            score = r.get("rrf_score", r.get("score", 0))
            print(f"    [{source}] score={score:.4f} | {r['content'][:60]}...")

    storage.db.close()
    _safe_remove_db(db_path)

    print("\n[PASS] Hybrid Retriever test passed")
    return True


def test_reranker():
    print("\n" + "=" * 60)
    print("Test 6: Cross-Encoder Reranker")
    print("=" * 60)

    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        top_k=3,
        use_llm_fallback=True,
    )

    query = "梯度消失怎么解决"
    candidates = [
        {"node_id": "1", "content": "梯度消失使用ReLU激活函数和残差连接和批量归一化", "title": "梯度消失", "score": 0.8},
        {"node_id": "2", "content": "K-Means是最经典的聚类算法。它将数据划分为K个簇", "title": "K-Means", "score": 0.7},
        {"node_id": "3", "content": "PCA是最常用的降维方法。它通过正交变换将高维数据投影", "title": "PCA", "score": 0.6},
        {"node_id": "4", "content": "梯度爆炸使用梯度裁剪和权重初始化Xavier和He初始化", "title": "梯度爆炸", "score": 0.75},
        {"node_id": "5", "content": "线性回归是最基础的回归算法，通过最小化均方误差求解参数", "title": "线性回归", "score": 0.5},
    ]

    reranked = reranker.rerank(query, candidates, top_k=3)
    print(f"\n  Query: {query}")
    print(f"  Reranked results: {len(reranked)}")
    for r in reranked:
        rerank_score = r.get("rerank_score", 0)
        original_score = r.get("original_score", 0)
        source = r.get("source", "unknown")
        print(f"    [{source}] rerank={rerank_score:.4f} original={original_score:.4f} | {r['content'][:50]}...")

    assert len(reranked) > 0, "Reranker should return results!"

    print("\n[PASS] Reranker test passed")
    return True


def test_evaluator():
    print("\n" + "=" * 60)
    print("Test 7: RAGAS Evaluator")
    print("=" * 60)

    evaluator = RAGASEvaluator()

    samples = [
        EvalSample(
            question="梯度消失怎么解决？",
            answer="梯度消失可以通过使用ReLU激活函数、残差连接和批量归一化来解决。",
            contexts=["常见问题与解决方案：梯度消失使用ReLU激活函数和残差连接和批量归一化。梯度爆炸使用梯度裁剪和权重初始化。"],
            ground_truth="梯度消失的解决方案包括：使用ReLU激活函数、残差连接、批量归一化。",
        ),
        EvalSample(
            question="K-Means的步骤是什么？",
            answer="K-Means步骤：1.随机初始化K个质心 2.分配样本到最近质心 3.重新计算质心 4.重复直到收敛",
            contexts=["K-Means是最经典的聚类算法。算法步骤：1.随机初始化K个质心 2.将每个样本分配到最近的质心 3.重新计算每个簇的质心 4.重复步骤2-3直到收敛"],
            ground_truth="K-Means步骤：1.随机初始化K个质心 2.将每个样本分配到最近的质心 3.重新计算每个簇的质心 4.重复步骤2-3直到收敛。",
        ),
    ]

    results = evaluator.evaluate_batch(samples)
    print(f"\n  Batch evaluation results:")
    print(f"  {json.dumps(results, ensure_ascii=False, indent=2)}")

    assert "num_samples" in results, "Evaluation should return num_samples!"

    tracer = RetrievalTracer()
    tracer.trace_retrieval("test query", "hybrid", [{"score": 0.9}, {"score": 0.8}], 150.5)
    tracer.trace_retrieval("test query 2", "vector_only", [{"score": 0.85}], 80.2)
    summary = tracer.get_summary()
    print(f"\n  Tracer summary: {json.dumps(summary, ensure_ascii=False, indent=2)}")

    print("\n[PASS] Evaluator test passed")
    return True


def test_pipeline():
    print("\n" + "=" * 60)
    print("Test 8: Full Pipeline (without Qdrant/LLM)")
    print("=" * 60)

    from engines.pipeline import EnhancedRAGPipeline

    pipeline = EnhancedRAGPipeline(
        embed_func=None,
        llm_func=None,
    )

    print("\n--- Document Ingestion ---")
    ingest_result = pipeline.ingest(SAMPLE_DOCUMENT, doc_id="ml_pipeline_test")
    print(f"  Total nodes: {ingest_result['total_nodes']}")
    print(f"  Level counts: {ingest_result['level_counts']}")
    print(f"  Chunk time: {ingest_result['chunk_time_ms']}ms")
    print(f"  Store time: {ingest_result['store_time_ms']}ms")

    assert ingest_result["total_nodes"] > 0, "Should have ingested nodes!"

    print("\n--- Query (BM25 only, no vector) ---")
    queries = ["梯度消失怎么解决", "PCA降维步骤", "LSTM和GRU区别"]
    for query in queries:
        result = pipeline.query(query, use_hybrid=True, use_reranker=False, top_k=3)
        print(f"\n  Query: {query}")
        print(f"  Strategy: {result['strategy']}")
        print(f"  Contexts: {result['num_contexts']}")
        print(f"  Total time: {result['total_time_ms']}ms")
        for i, chain in enumerate(result["context_chains"]):
            l1 = chain.get("l1_title") or "N/A"
            l2 = chain.get("l2_title") or "N/A"
            l3 = chain.get("l3_title", "")[:30] or "N/A"
            print(f"    Context {i+1}: L1={l1} > L2={l2} > L3={l3}...")

    print("\n--- Pipeline Stats ---")
    stats = pipeline.get_stats()
    print(f"  {json.dumps(stats, ensure_ascii=False, indent=2)}")

    pipeline.close()

    cleanup_files = ["test_tree_store.db", "tree_store.db"]
    for f in cleanup_files:
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", f)
        fp = os.path.normpath(fp)
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except OSError:
                pass
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    print("\n[PASS] Full Pipeline test passed")
    return True


def main():
    print("=" * 60)
    print("  Phase 1 Complete Test Suite")
    print("=" * 60)

    results = {}

    try:
        nodes = _build_nodes()
        results["chunker"] = "PASS"
    except Exception as e:
        print(f"\n[FAIL] Chunker: {e}")
        results["chunker"] = "FAIL"
        import traceback
        traceback.print_exc()
        return

    try:
        test_storage(nodes)
        results["storage"] = "PASS"
    except Exception as e:
        print(f"\n[FAIL] Storage: {e}")
        results["storage"] = "FAIL"
        import traceback
        traceback.print_exc()

    try:
        test_retriever(nodes)
        results["retriever"] = "PASS"
    except Exception as e:
        print(f"\n[FAIL] Retriever: {e}")
        results["retriever"] = "FAIL"
        import traceback
        traceback.print_exc()

    try:
        test_bm25(nodes)
        results["bm25"] = "PASS"
    except Exception as e:
        print(f"\n[FAIL] BM25: {e}")
        results["bm25"] = "FAIL"
        import traceback
        traceback.print_exc()

    try:
        test_hybrid_retriever(nodes)
        results["hybrid"] = "PASS"
    except Exception as e:
        print(f"\n[FAIL] Hybrid Retriever: {e}")
        results["hybrid"] = "FAIL"
        import traceback
        traceback.print_exc()

    try:
        test_reranker()
        results["reranker"] = "PASS"
    except Exception as e:
        print(f"\n[FAIL] Reranker: {e}")
        results["reranker"] = "FAIL"
        import traceback
        traceback.print_exc()

    try:
        test_evaluator()
        results["evaluator"] = "PASS"
    except Exception as e:
        print(f"\n[FAIL] Evaluator: {e}")
        results["evaluator"] = "FAIL"
        import traceback
        traceback.print_exc()

    try:
        test_pipeline()
        results["pipeline"] = "PASS"
    except Exception as e:
        print(f"\n[FAIL] Pipeline: {e}")
        results["pipeline"] = "FAIL"
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    all_pass = True
    for name, status in results.items():
        icon = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {icon} {name}")
        if status != "PASS":
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("  ALL TESTS PASSED!")
    else:
        print("  SOME TESTS FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
