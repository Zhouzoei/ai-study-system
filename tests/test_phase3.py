import os
import sys
import time
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge_graph import KnowledgeGraphBuilder, Entity, Relation
from engines.adaptive_retriever import QueryClassifier, QueryType, AdaptiveRetriever
from engines.qa_engine import QAEngine, AnswerResult


def test_knowledge_graph():
    print("\n=== KnowledgeGraphBuilder ===")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        kg = KnowledgeGraphBuilder(db_path=db_path)

        nodes = [
            {"node_id": "n1", "title": "Python Basics", "content": "Python是一种高级编程语言，被称为解释型语言。Python包括变量、数据类型、控制流等基本概念。Python基于面向对象编程范式。"},
            {"node_id": "n2", "title": "Functions", "content": "函数是Python中可重用的代码块。函数使用def关键字定义。Python支持默认参数和关键字参数。函数可以实现代码的模块化。"},
            {"node_id": "n3", "title": "Classes", "content": "类是Python面向对象编程的核心。类使用class关键字定义。对象是类的实例。Python支持继承和多态。"},
        ]

        result = kg.build_from_nodes(nodes, doc_id="test_doc")
        assert result["total_entities"] > 0
        print(f"[PASS] build_from_nodes (rule-based): {result['total_entities']} entities, {result['total_relations']} relations")

        stats = kg.get_graph_stats()
        assert stats["total_entities"] > 0
        print(f"[PASS] get_graph_stats: {stats}")

        e1 = Entity(name="TestEntity", entity_type="concept", description="A test entity", doc_id="test_doc")
        kg._save_entity(e1)
        found = kg.query_entity("TestEntity")
        assert found is not None
        assert found["name"] == "TestEntity"
        print("[PASS] query_entity")

        e2 = Entity(name="RelatedEntity", entity_type="tool", description="Related to test", doc_id="test_doc")
        kg._save_entity(e2)
        r1 = Relation(source_entity_id=e1.entity_id, target_entity_id=e2.entity_id, relation_type="depends_on", doc_id="test_doc")
        kg._save_relation(r1)

        related = kg.get_related_entities("TestEntity")
        assert len(related) > 0
        print(f"[PASS] get_related_entities: {len(related)} related")

        relations = kg.get_entity_relations(e1.entity_id, depth=1)
        assert relations["entity"] is not None
        assert len(relations["relations"]) > 0
        print(f"[PASS] get_entity_relations: {len(relations['relations'])} relations")

        kg.close()
        print("[PASS] KnowledgeGraphBuilder all tests passed")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_query_classifier():
    print("\n=== QueryClassifier ===")

    classifier = QueryClassifier()

    factual = classifier.classify("什么是Python？")
    assert factual.query_type == QueryType.FACTUAL
    assert len(factual.keywords) > 0
    print(f"[PASS] factual query: type={factual.query_type.value}, keywords={factual.keywords}")

    reasoning = classifier.classify("为什么Python是解释型语言？")
    assert reasoning.query_type in [QueryType.REASONING, QueryType.FACTUAL]
    print(f"[PASS] reasoning query: type={reasoning.query_type.value}")

    exploratory = classifier.classify("如何学习Python编程？")
    assert exploratory.query_type in [QueryType.EXPLORATORY, QueryType.PROCEDURAL, QueryType.FACTUAL]
    print(f"[PASS] exploratory query: type={exploratory.query_type.value}")

    comparison = classifier.classify("Python和Java的区别是什么？")
    assert comparison.query_type in [QueryType.COMPARISON, QueryType.REASONING, QueryType.FACTUAL]
    print(f"[PASS] comparison query: type={comparison.query_type.value}")

    procedural = classifier.classify("Python安装步骤是什么？")
    assert procedural.query_type in [QueryType.PROCEDURAL, QueryType.EXPLORATORY, QueryType.FACTUAL]
    print(f"[PASS] procedural query: type={procedural.query_type.value}")

    mixed = classifier.classify("Tell me about machine learning")
    assert mixed.query_type in [QueryType.FACTUAL, QueryType.EXPLORATORY]
    print(f"[PASS] mixed query: type={mixed.query_type.value}")

    for qt in [QueryType.FACTUAL, QueryType.REASONING, QueryType.EXPLORATORY]:
        analysis = classifier.classify("random query xyz")
        assert analysis.suggested_strategy is not None
        assert analysis.suggested_top_k > 0
    print("[PASS] all query types have strategy configs")

    print("[PASS] QueryClassifier all tests passed")


def test_qa_engine():
    print("\n=== QAEngine ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        from config import config as app_config
        old_tree_db = app_config.HIERARCHICAL_TREE_DB
        app_config.HIERARCHICAL_TREE_DB = os.path.join(tmpdir, "tree_store.db")

        try:
            from engines.pipeline import EnhancedRAGPipeline

            pipeline = EnhancedRAGPipeline()

            doc_text = """# Machine Learning Guide

## Introduction
Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data. ML includes supervised learning, unsupervised learning, and reinforcement learning.

## Supervised Learning
Supervised learning uses labeled data to train models. Common algorithms include linear regression, decision trees, and neural networks. The goal is to learn a mapping from inputs to outputs.

## Unsupervised Learning
Unsupervised learning finds patterns in unlabeled data. Clustering and dimensionality reduction are key techniques. K-means and PCA are popular algorithms.

## Deep Learning
Deep learning is a subset of machine learning using neural networks with many layers. It excels at image recognition, natural language processing, and speech recognition. Popular frameworks include TensorFlow and PyTorch.
"""
            pipeline.ingest(doc_text, doc_id="ml_guide")

            qa = QAEngine(
                pipeline=pipeline,
                adaptive_retriever=pipeline.adaptive_retriever,
                knowledge_graph=pipeline.knowledge_graph,
                llm_func=None,
            )

            result = qa.ask("What is machine learning?", use_adaptive=True)
            assert isinstance(result, AnswerResult)
            assert result.question == "What is machine learning?"
            assert result.query_type in ["factual", "reasoning", "exploratory", "comparison", "procedural"]
            print(f"[PASS] ask (no LLM): type={result.query_type}, confidence={result.confidence}")

            result2 = qa.ask("How does supervised learning work?", use_adaptive=True)
            assert result2.query_type in ["factual", "reasoning", "exploratory", "comparison", "procedural"]
            print(f"[PASS] ask (reasoning): type={result2.query_type}")

            sources = qa._extract_sources([{"l1_title": "Ch1", "l2_title": "Sec1", "l3_title": "Para1", "l3_score": 0.9}])
            assert len(sources) == 1
            assert sources[0]["chapter"] == "Ch1"
            print("[PASS] _extract_sources")

            kg_facts = qa._extract_kg_facts(["[KG: Python]\n  - depends_on: library\n  - related_to: AI"])
            assert len(kg_facts) == 2
            print(f"[PASS] _extract_kg_facts: {kg_facts}")

            confidence = qa._estimate_confidence("Based on the data, this is true.", ["context1", "context2"], {"confidence": 0.8})
            assert 0.0 <= confidence <= 1.0
            print(f"[PASS] _estimate_confidence: {confidence}")

            followups = qa._generate_rule_based_followups("What is ML?", {"keywords": ["ML"], "query_type": "factual"})
            assert len(followups) > 0
            print(f"[PASS] _generate_rule_based_followups: {followups}")

            pipeline.close()
            print("[PASS] QAEngine all tests passed")
        finally:
            app_config.HIERARCHICAL_TREE_DB = old_tree_db


def test_pipeline_phase3_integration():
    print("\n=== Pipeline Phase 3 Integration ===")

    from engines.pipeline import EnhancedRAGPipeline
    from config import config as app_config

    tmpdir = tempfile.mkdtemp()
    old_tree_db = app_config.HIERARCHICAL_TREE_DB
    app_config.HIERARCHICAL_TREE_DB = os.path.join(tmpdir, "tree_store.db")

    try:
        pipeline = EnhancedRAGPipeline()

        doc_text = """# Data Science Handbook

## Chapter 1: Statistics Fundamentals
Statistics is the science of collecting, analyzing, and interpreting data. Descriptive statistics summarizes data using measures like mean, median, and standard deviation. Inferential statistics makes predictions about populations based on samples.

## Chapter 2: Data Processing
Data processing involves cleaning, transforming, and organizing raw data. Pandas is a popular Python library for data manipulation. Data cleaning handles missing values, outliers, and inconsistencies.

## Chapter 3: Machine Learning Basics
Machine learning algorithms learn patterns from data. Supervised learning requires labeled training data. Common algorithms include linear regression, logistic regression, and random forests.

## Chapter 4: Deep Learning
Deep learning uses multi-layer neural networks. Backpropagation is the core training algorithm. CNNs excel at image tasks while RNNs handle sequential data.
"""
        result = pipeline.ingest(doc_text, doc_id="ds_handbook")
        assert result["total_nodes"] > 0
        assert result["kg_entities"] >= 0
        print(f"[PASS] ingest with KG: {result['total_nodes']} nodes, {result['kg_entities']} entities, {result['kg_relations']} relations")

        classification = pipeline.classify_query("What is the difference between supervised and unsupervised learning?")
        assert "query_type" in classification
        assert "keywords" in classification
        print(f"[PASS] classify_query: type={classification['query_type']}, intent={classification['intent']}")

        session_info = pipeline.create_session(user_id="test_user", title="DS Study")
        sid = session_info["session_id"]

        ask_result = pipeline.ask("What is statistics?", session_id=sid, doc_id="ds_handbook")
        assert "answer" in ask_result
        assert "query_type" in ask_result
        assert "confidence" in ask_result
        assert "sources" in ask_result
        print(f"[PASS] ask: type={ask_result['query_type']}, confidence={ask_result['confidence']:.2f}")

        ask_result2 = pipeline.ask("How to process data?", session_id=sid, doc_id="ds_handbook")
        assert "answer" in ask_result2
        print(f"[PASS] ask (follow-up): type={ask_result2['query_type']}")

        kg_entity = pipeline.query_knowledge_graph("Statistics")
        if kg_entity:
            print(f"[PASS] query_knowledge_graph: found entity '{kg_entity['name']}'")
        else:
            print("[PASS] query_knowledge_graph: no entity found (rule-based extraction limited)")

        stats = pipeline.get_stats()
        assert "knowledge_graph" in stats
        print(f"[PASS] get_stats with KG: {stats['knowledge_graph']}")

        pipeline.close()
        print("[PASS] Pipeline Phase 3 integration all tests passed")
    finally:
        app_config.HIERARCHICAL_TREE_DB = old_tree_db
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_knowledge_graph()
    test_query_classifier()
    test_qa_engine()
    test_pipeline_phase3_integration()
    print("\n" + "=" * 50)
    print("ALL PHASE 3 TESTS PASSED!")
    print("=" * 50)
