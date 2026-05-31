from typing import List, Optional
import logging
import time
import dashscope
from dashscope import TextEmbedding
from config import config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 1.0


class EmbeddingService:
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name or config.EMBED_MODEL_NAME
        self._api_key = api_key or config.EMBED_API_KEY
        dashscope.api_key = self._api_key

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_vectors = []
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            last_error = None
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = TextEmbedding.call(
                        model=self.model_name,
                        input=batch,
                        dimensions=config.QDRANT_VECTOR_SIZE,
                    )
                    if resp.status_code == 200:
                        for item in resp.output["embeddings"]:
                            vec = item["embedding"]
                            if len(vec) != config.QDRANT_VECTOR_SIZE:
                                logger.warning(
                                    f"Embedding dimension mismatch: got {len(vec)}, "
                                    f"config expects {config.QDRANT_VECTOR_SIZE}"
                                )
                            all_vectors.append(vec)
                        last_error = None
                        break
                    else:
                        last_error = RuntimeError(f"Embedding API error: {resp.message}")
                except Exception as e:
                    last_error = e
                    logger.warning(f"Embedding batch {i} attempt {attempt + 1} failed: {e}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY * (attempt + 1))

            if last_error is not None:
                raise RuntimeError(
                    f"Embedding 批次 {i} 失败 (model={self.model_name}, retries={_MAX_RETRIES}): {last_error}"
                ) from last_error

        return all_vectors

    def embed_single(self, text: str) -> List[float]:
        results = self.embed([text])
        if results:
            return results[0]
        logger.error("Embedding returned empty result, returning zero vector as fallback")
        return [0.0] * config.QDRANT_VECTOR_SIZE
