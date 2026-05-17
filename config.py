import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "deepseek-chat")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "qa_system_vectors")
    QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
    QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "cosine")

    NEO4J_URI = os.getenv("NEO4J_URI", "")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

    EMBED_MODEL_TYPE = os.getenv("EMBED_MODEL_TYPE", "dashscope")
    EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "text-embedding-v3")
    EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
    EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "")

    HIERARCHICAL_COLLECTION = "hierarchical_chunks"
    HIERARCHICAL_TREE_DB = os.path.join(os.path.dirname(__file__), "tree_store.db")

    CHUNK_L1_MAX_SIZE = 2000
    CHUNK_L2_MAX_SIZE = 500
    CHUNK_L3_MAX_SIZE = 200
    CHUNK_L3_MIN_SIZE = 60
    CHUNK_OVERLAP = 30

    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    RERANKER_TOP_K = 5
    RERANKER_AUTO_DISABLE = True
    RERANKER_MIN_CONFIDENCE = 0.3

    BM25_TOP_K = 30
    VECTOR_TOP_K = 30
    RRF_K = 30

    RETRIEVAL_TOP_K = 8
    CONTEXT_STRATEGY = "auto_merge"

    AUTO_MERGE_THRESHOLD = 0.3

    QUERY_REWRITING_ENABLED = False
    QUERY_EXPANSION_STRATEGIES = ["expand", "hyde", "terms"]
    QUERY_EXPANSION_NUM_QUERIES = 3
    HYDE_TOP_K = 5

    MMR_ENABLED = True
    MMR_LAMBDA = 0.7
    MMR_TOP_K = 10


config = Config()
