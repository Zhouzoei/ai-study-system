from typing import List, Optional
import dashscope
from dashscope import TextEmbedding
from config import config


class EmbeddingService:
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name or config.EMBED_MODEL_NAME
        dashscope.api_key = api_key or config.EMBED_API_KEY

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_vectors = []
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                resp = TextEmbedding.call(
                    model=self.model_name,
                    input=batch,
                    dimensions=config.QDRANT_VECTOR_SIZE,
                )
                if resp.status_code == 200:
                    for item in resp.output["embeddings"]:
                        all_vectors.append(item["embedding"])
                else:
                    raise RuntimeError(f"Embedding API error: {resp.message}")
            except Exception as e:
                raise RuntimeError(
                    f"Embedding 批次 {i} 失败 (model={self.model_name}): {e}"
                ) from e

        return all_vectors

    def embed_single(self, text: str) -> List[float]:
        results = self.embed([text])
        return results[0] if results else [0.0] * config.QDRANT_VECTOR_SIZE
