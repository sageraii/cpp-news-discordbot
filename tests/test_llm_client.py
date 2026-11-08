import json
import os
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

import llm_client
from llm_client import create_llm_client


SCHEMA = {
    "type": "object",
    "properties": {"articles": {"type": "array", "items": {"type": "string"}}},
    "required": ["articles"],
    "additionalProperties": False,
}


def completion(content='{"articles": []}', finish_reason="stop", refusal=None):
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
        "model": "gpt-6-luna",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content, "refusal": refusal}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    }


class OpenAITransportTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.sleep = patch("llm_client.time.sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def client(self, config=None, handler=None):
        def transport(request):
            self.requests.append(request)
            if handler:
                return handler(request)
            return httpx.Response(200, json=completion())

        http_client = httpx.Client(transport=httpx.MockTransport(transport))
        self.addCleanup(http_client.close)

        def sdk(**kwargs):
            return OpenAI(http_client=http_client, **kwargs)

        with patch("openai.OpenAI", side_effect=sdk):
            return create_llm_client({"enabled": True, "retry_count": 3,
                                      "retry_delay": 0, **(config or {})})

    def test_default_luna_disables_reasoning(self):
        for config in ({}, {"model": "gpt-6-luna"}):
            with self.subTest(config=config):
                self.client(config).generate("translate")
                body = json.loads(self.requests[-1].content)
                self.assertEqual(body["model"], "gpt-6-luna")
                self.assertEqual(body.get("reasoning_effort"), "none")

    def test_other_models_preserve_unspecified_reasoning(self):
        for config in ({"model": "gpt-4o-mini"},
                       {"provider": "custom", "model": "gpt-6-luna"}):
            with self.subTest(config=config):
                self.client(config).generate("translate")
                body = json.loads(self.requests[-1].content)
                self.assertNotIn("reasoning_effort", body)
                self.assertEqual(body["temperature"], 0.3)

    def test_unsupported_provider_has_safe_diagnostic(self):
        with self.assertRaises(ValueError) as caught:
            create_llm_client({"enabled": True, "provider": "private-api-key"})
        self.assertNotIn("private-api-key", str(caught.exception))
        self.assertEqual(type(caught.exception).__name__, "LLMError")

    def test_sdk_errors_are_not_log_safe_diagnostics(self):
        self.assertTrue(hasattr(llm_client, "LLMError"))
        client = self.client(handler=lambda request: httpx.Response(401, json={
            "error": {"message": "private-api-key", "type": "error"}}))
        with self.assertRaises(Exception) as caught:
            client.generate("translate")
        self.assertNotIsInstance(caught.exception, llm_client.LLMError)

    def test_openai_request_uses_strict_schema_and_completion_budget(self):
        client = self.client({"reasoning_effort": "none", "max_completion_tokens": 1200})
        self.assertEqual(client.generate("private prompt", "translate", response_schema=SCHEMA), '{"articles": []}')
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["model"], "gpt-6-luna")
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["max_completion_tokens"], 1200)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["response_format"], {
            "type": "json_schema", "json_schema": {
                "name": "article_translations", "strict": True, "schema": SCHEMA}})
        self.assertEqual(body["messages"], [{"role": "system", "content": "translate"},
                                            {"role": "user", "content": "private prompt"}])

    def test_reasoning_omits_temperature_and_old_budget_is_supported(self):
        client = self.client({"reasoning_effort": "low", "max_tokens": 456})
        client.generate("translate")
        body = json.loads(self.requests[0].content)
        self.assertNotIn("temperature", body)
        self.assertEqual(body["max_completion_tokens"], 456)

    def test_provider_keys_and_legacy_request_parameters(self):
        cases = [("openai", "test-openai"), ("openrouter", "test-router"), ("custom", "test-openai")]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai", "OPENROUTER_API_KEY": "test-router", "LLM_API_KEY": "test-generic"}, clear=True):
            for provider, key in cases:
                with self.subTest(provider=provider):
                    client = self.client({"provider": provider, "max_tokens": 123})
                    client.generate("translate")
                    request = self.requests[-1]
                    self.assertEqual(request.headers["authorization"], "Bearer " + key)
                    if provider != "openai":
                        body = json.loads(request.content)
                        self.assertEqual(body["max_tokens"], 123)
                        self.assertNotIn("max_completion_tokens", body)
                        self.assertNotIn("reasoning_effort", body)
                    if provider == "openrouter":
                        self.assertEqual(str(request.url), "https://openrouter.ai/api/v1/chat/completions")

    def test_unrelated_provider_key_is_not_used(self):
        for provider, environment in [("openai", {"OPENROUTER_API_KEY": "wrong"}),
                                      ("openrouter", {"OPENAI_API_KEY": "wrong"})]:
            with self.subTest(provider=provider), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(ValueError) as caught:
                    self.client({"provider": provider})
                self.assertEqual(type(caught.exception).__name__, "LLMError")

    def test_generic_key_fallback(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-generic"}, clear=True):
            for provider in ("openai", "openrouter", "custom"):
                with self.subTest(provider=provider):
                    self.client({"provider": provider}).generate("translate")
                    self.assertEqual(self.requests[-1].headers["authorization"], "Bearer test-generic")

    def test_permanent_errors_fail_once_without_logging_response_secrets(self):
        for status in (400, 401, 403, 404, 422, 600):
            with self.subTest(status=status):
                self.requests.clear()
                client = self.client(handler=lambda request: httpx.Response(status, json={"error": {"message": "secret-api-key private prompt", "type": "error"}}))
                with self.assertLogs("llm_client", level="WARNING") as logs:
                    with self.assertRaises(Exception):
                        client.generate("private prompt")
                self.assertEqual(len(self.requests), 1)
                self.assertNotIn("secret-api-key", " ".join(logs.output))
                self.assertNotIn("private prompt", " ".join(logs.output))

    def test_transient_errors_have_only_three_total_attempts(self):
        for status in (408, 409, 429, 500, 503):
            with self.subTest(status=status):
                self.requests.clear()
                client = self.client(handler=lambda request: httpx.Response(status, json={"error": {"message": "transient", "type": "error"}}))
                with self.assertRaises(Exception):
                    client.generate("translate")
                self.assertEqual(len(self.requests), 3)

    def test_connection_and_timeout_are_retried_then_succeed(self):
        for error_type in (httpx.ConnectError, httpx.ReadTimeout):
            with self.subTest(error_type=error_type):
                self.requests.clear()
                def handler(request):
                    if len(self.requests) < 3:
                        raise error_type("offline test", request=request)
                    return httpx.Response(200, json=completion())
                self.assertEqual(self.client(handler=handler).generate("translate"), '{"articles": []}')
                self.assertEqual(len(self.requests), 3)

    def test_invalid_output_fails_once(self):
        for payload in (completion(finish_reason="length"), completion(finish_reason="content_filter"),
                        completion(refusal="cannot comply"), completion(content=None),
                        completion(content=""), completion(content="  "),
                        {**completion(), "choices": []}):
            with self.subTest(payload=payload):
                self.requests.clear()
                client = self.client(handler=lambda request: httpx.Response(200, json=payload))
                with self.assertRaises(ValueError) as caught:
                    client.generate("translate")
                self.assertEqual(type(caught.exception).__name__, "LLMError")
                self.assertNotIn("cannot comply", str(caught.exception))
                self.assertEqual(len(self.requests), 1)

    def test_usage_logs_counts_not_prompt_or_output(self):
        client = self.client(handler=lambda request: httpx.Response(200, json=completion(content="private output")))
        with self.assertLogs("llm_client", level="INFO") as logs:
            self.assertEqual(client.generate("private prompt"), "private output")
        logged = " ".join(logs.output)
        for number in ("12", "7", "19"):
            self.assertIn(number, logged)
        for secret in ("private output", "private prompt", "test-openai"):
            self.assertNotIn(secret, logged)


if __name__ == "__main__":
    unittest.main()
