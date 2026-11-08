"""
llm_client.py - 다중 LLM 프로바이더 지원 클라이언트

지원 프로바이더:
- OpenAI (gpt-6-luna)
- Anthropic (claude-sonnet-4-20250514, claude-haiku-4-20250514)
- Google (gemini-2.0-flash, gemini-2.5-flash) - google-genai SDK 사용
- OpenRouter (다양한 모델)
- Ollama (로컬 모델)
- Custom (OpenAI 호환 엔드포인트)
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class LLMError(ValueError):
    """Log-safe application-authored configuration or response diagnostics only.

    Messages must never contain API payloads, prompts, keys, or arbitrary config
    values. SDK exceptions are not log-safe and must not be wrapped in this type.
    """


class LLMClient(ABC):
    """LLM 클라이언트 추상 베이스 클래스"""

    @abstractmethod
    def generate(
        self, prompt: str, system_prompt: Optional[str] = None,
        response_schema: Optional[dict] = None,
    ) -> str:
        """프롬프트에 대한 응답 생성"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API 클라이언트 (OpenRouter, Custom 엔드포인트 포함)"""

    def __init__(self, config: dict):
        from openai import OpenAI

        self.provider = config.get("provider", "openai").lower()
        key_name = "OPENROUTER_API_KEY" if self.provider == "openrouter" else "OPENAI_API_KEY"
        api_key = os.environ.get(key_name) or os.environ.get("LLM_API_KEY")

        if not api_key:
            raise LLMError(f"{key_name} 또는 LLM_API_KEY가 설정되지 않았습니다.")

        # 재시도는 아래 루프에서만 수행해 SDK와 중복되지 않게 한다.
        self.client = OpenAI(api_key=api_key, base_url=config.get("base_url"), max_retries=0)
        self.model = config.get("model", "gpt-6-luna" if self.provider == "openai" else "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 800)
        self.max_completion_tokens = config.get("max_completion_tokens", self.max_tokens)
        self.reasoning_effort = config.get(
            "reasoning_effort",
            "none" if self.provider == "openai" and self.model == "gpt-6-luna" else None,
        )
        self.timeout = config.get("timeout", 30)
        self.retry_count = config.get("retry_count", 3)
        self.retry_delay = config.get("retry_delay", 2)

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None,
        response_schema: Optional[dict] = None,
    ) -> str:
        from openai import APIConnectionError, APIStatusError

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        params = {"model": self.model, "messages": messages, "timeout": self.timeout}
        if self.provider == "openai":
            params["max_completion_tokens"] = self.max_completion_tokens
            if self.reasoning_effort is not None:
                params["reasoning_effort"] = self.reasoning_effort
            if response_schema is not None:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "article_translations", "strict": True, "schema": response_schema,
                    },
                }
        else:
            params["max_tokens"] = self.max_tokens
        if self.provider != "openai" or self.reasoning_effort in (None, "none"):
            params["temperature"] = self.temperature

        for attempt in range(max(1, self.retry_count)):
            try:
                response = self.client.chat.completions.create(**params)
            except (APIConnectionError, APIStatusError) as error:
                status = getattr(error, "status_code", None)
                retryable = isinstance(error, APIConnectionError) or (
                    status in (408, 409, 429) or (status is not None and 500 <= status < 600)
                )
                # API 오류 본문은 키나 입력을 포함할 수 있으므로 출력하지 않는다.
                logger.warning(
                    "OpenAI API 호출 실패 (시도 %s/%s, 유형=%s, status=%s)",
                    attempt + 1, self.retry_count, type(error).__name__, status,
                )
                if not retryable or attempt + 1 >= self.retry_count:
                    raise
                time.sleep(self.retry_delay)
                continue

            if not response.choices:
                raise LLMError("OpenAI 응답에 선택지가 없습니다.")
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                raise LLMError("OpenAI 응답이 정상적으로 완료되지 않았습니다.")
            if choice.message.refusal:
                raise LLMError("OpenAI가 요청을 거절했습니다.")
            content = choice.message.content
            if not isinstance(content, str) or not content.strip():
                raise LLMError("OpenAI 응답 내용이 비어 있거나 문자열이 아닙니다.")
            if response.usage is not None:
                logger.info(
                    "OpenAI token usage: prompt=%s completion=%s total=%s",
                    response.usage.prompt_tokens, response.usage.completion_tokens,
                    response.usage.total_tokens,
                )
            return content


class AnthropicClient(LLMClient):
    """Anthropic Claude API 클라이언트"""

    def __init__(self, config: dict):
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY")

        if not api_key:
            raise ValueError("Anthropic API 키가 설정되지 않았습니다.")

        self.client = Anthropic(api_key=api_key)
        self.model = config.get("model", "claude-haiku-4-20250514")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 800)
        self.retry_count = config.get("retry_count", 3)
        self.retry_delay = config.get("retry_delay", 2)

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None,
        response_schema: Optional[dict] = None,
    ) -> str:
        last_error = None
        for attempt in range(self.retry_count):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                last_error = e
                logger.warning(f"Anthropic API 호출 실패 (시도 {attempt + 1}/{self.retry_count}): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)

        raise last_error


class GoogleClient(LLMClient):
    """Google Gemini API 클라이언트 (google-genai SDK 사용)"""

    def __init__(self, config: dict):
        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("LLM_API_KEY")

        if not api_key:
            raise ValueError("Google API 키가 설정되지 않았습니다.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = config.get("model", "gemini-2.0-flash")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 800)
        self.retry_count = config.get("retry_count", 3)
        self.retry_delay = config.get("retry_delay", 2)

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None,
        response_schema: Optional[dict] = None,
    ) -> str:
        from google.genai import types

        last_error = None
        for attempt in range(self.retry_count):
            try:
                # GenerateContentConfig로 시스템 프롬프트와 설정 전달
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt if system_prompt else None,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text
            except Exception as e:
                last_error = e
                logger.warning(f"Google API 호출 실패 (시도 {attempt + 1}/{self.retry_count}): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)

        raise last_error


class OllamaClient(LLMClient):
    """Ollama 로컬 LLM 클라이언트"""

    def __init__(self, config: dict):
        import requests

        self.requests = requests
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "llama3")
        self.temperature = config.get("temperature", 0.3)
        self.retry_count = config.get("retry_count", 3)
        self.retry_delay = config.get("retry_delay", 2)

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None,
        response_schema: Optional[dict] = None,
    ) -> str:
        last_error = None
        for attempt in range(self.retry_count):
            try:
                response = self.requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": system_prompt or "",
                        "stream": False,
                        "options": {"temperature": self.temperature},
                    },
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()["response"]
            except Exception as e:
                last_error = e
                logger.warning(f"Ollama API 호출 실패 (시도 {attempt + 1}/{self.retry_count}): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)

        raise last_error


def create_llm_client(config: dict) -> Optional[LLMClient]:
    """설정에 따라 적절한 LLM 클라이언트 생성

    Args:
        config: LLM 설정 딕셔너리

    Returns:
        LLMClient 인스턴스 또는 None (비활성화 시)
    """
    if not config.get("enabled", False):
        logger.info("LLM 기능이 비활성화되어 있습니다.")
        return None

    provider = config.get("provider", "openai").lower()

    # OpenRouter는 OpenAI 호환 엔드포인트 사용
    if provider == "openrouter":
        config = config.copy()
        config["base_url"] = "https://openrouter.ai/api/v1"

    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GoogleClient,
        "ollama": OllamaClient,
        "custom": OpenAIClient,
        "openrouter": OpenAIClient,
    }

    client_class = clients.get(provider)
    if not client_class:
        raise LLMError("지원하지 않는 LLM 프로바이더입니다.")

    logger.info(f"LLM 클라이언트 생성: {provider} ({config.get('model', 'default')})")
    return client_class(config)
