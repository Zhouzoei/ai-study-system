from typing import Optional, Generator
from config import config


class LLMService:
    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_id = model_id or config.LLM_MODEL_ID
        self.api_key = api_key or config.LLM_API_KEY
        self.base_url = base_url or config.LLM_BASE_URL
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
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
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM调用失败: {e}]"

    def invoke_stream(self, prompt: str) -> Generator[str, None, None]:
        try:
            response = self._get_client().chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            yield f"[LLM流式调用失败: {e}]"
