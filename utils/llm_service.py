from typing import Optional, Generator
import logging
import httpx
from config import config, resolve_llm_config

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(
        self,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        cfg = resolve_llm_config()
        self.provider = provider or cfg["provider"]
        self.model_id = model_id or cfg["model_id"]
        self.api_key = api_key or cfg["api_key"]
        self.base_url = base_url or cfg["base_url"]
        self.timeout = cfg["timeout"]
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=5.0),
                max_retries=1,
            )
        return self._client

    def invoke(self, prompt: str) -> str:
        try:
            response = self._get_client().chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM invoke failed (model={self.model_id}): {e}", exc_info=True)
            raise

    def invoke_stream(self, prompt: str) -> Generator[str, None, None]:
        try:
            client = self._get_client()
            stream_timeout = httpx.Timeout(self.timeout, connect=5.0)
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                stream=True,
                timeout=stream_timeout,
            )
            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            logger.error(f"LLM invoke_stream failed (model={self.model_id}): {e}", exc_info=True)
            raise
