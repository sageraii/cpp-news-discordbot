"""
llm_client.py - 다중 LLM 프로바이더 지원 클라이언트

지원 프로바이더:
- OpenAI (gpt-4o, gpt-4o-mini)
- Anthropic (claude-sonnet-4-20250514, claude-haiku-4-20250514)
- Google (gemini-1.5-flash, gemini-1.5-pro)
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


class LLMClient(ABC):
    """LLM 클라이언트 추상 베이스 클래스"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """프롬프트에 대한 응답 생성"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API 클라이언트 (OpenRouter, Custom 엔드포인트 포함)"""

    def __init__(self, config: dict):
        from openai import OpenAI

        api_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )

        if not api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=api_key, base_url=config.get("base_url"))
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 800)
        self.timeout = config.get("timeout", 30)
        self.retry_count = config.get("retry_count", 3)
        self.retry_delay = config.get("retry_delay", 2)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self.retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                logger.warning(f"OpenAI API 호출 실패 (시도 {attempt + 1}/{self.retry_count}): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)

        raise last_error


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

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
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
    """Google Gemini API 클라이언트"""

    def __init__(self, config: dict):
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("LLM_API_KEY")

        if not api_key:
            raise ValueError("Google API 키가 설정되지 않았습니다.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.get("model", "gemini-1.5-flash"))
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 800)
        self.retry_count = config.get("retry_count", 3)
        self.retry_delay = config.get("retry_delay", 2)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        last_error = None
        for attempt in range(self.retry_count):
            try:
                response = self.model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    },
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

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
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
        raise ValueError(f"지원하지 않는 LLM 프로바이더: {provider}")

    logger.info(f"LLM 클라이언트 생성: {provider} ({config.get('model', 'default')})")
    return client_class(config)
