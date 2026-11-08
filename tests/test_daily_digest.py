import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import httpx
import yaml
from openai import OpenAI

# Keep the pre-change import safe while the import-side-effect regression is red.
_exists = Path.exists
with patch.object(Path, "exists", lambda path: False if path.name == ".env" else _exists(path)):
    import daily_digest


ROOT = Path(__file__).resolve().parents[1]


def article(index=0):
    return {
        "id": f"article-{index}",
        "title": f"C++ news {index}",
        "description": "<p>Use <b>std::vector</b> safely.</p>",
        "content": "FULL CONTENT MUST NOT BE SENT TO THE MODEL",
        "source": "Test RSS",
        "link": f"https://example.invalid/{index}",
        "published": "2026-11-08T00:00:00+00:00",
    }


def translated(index=0):
    return {
        "article_index": index,
        "translated_title": f"C++ 소식 {index}",
        "translated_description": "std::vector를 안전하게 사용하세요.",
    }


class DigestTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch("requests.post", side_effect=AssertionError("No network in tests")).start()
        patch("feedparser.parse", side_effect=AssertionError("No feed fetching in tests")).start()
        patch("daily_digest.time.sleep").start()
        patch.object(daily_digest.logger, "info").start()
        patch.object(daily_digest.logger, "warning").start()
        patch.object(daily_digest.logger, "error").start()
        self.llm = Mock()
        with patch("daily_digest.create_llm_client", return_value=self.llm):
            self.bot = daily_digest.CPPDailyDigest("https://example.invalid/webhook")

    def respond(self, value):
        self.llm.generate.return_value = json.dumps(value, ensure_ascii=False)

    def test_default_config_is_luna_without_reasoning(self):
        config = self.bot.config["llm"]
        self.assertEqual(config["provider"], "openai")
        self.assertEqual(config["model"], "gpt-6-luna")
        self.assertEqual(config.get("reasoning_effort"), "none")
        self.assertIn("max_completion_tokens", config)

    def test_input_contains_only_title_description_source(self):
        prepared = self.bot._prepare_article_for_batch(article())
        self.assertEqual(set(prepared), {"title", "description", "source"})
        self.assertEqual(prepared["description"], "Use std::vector safely.")
        self.assertEqual(len(self.bot._prepare_article_for_batch({**article(), "description": "x" * 700})["description"]), 500)

    def test_object_response_is_validated_and_sorted(self):
        self.respond({"articles": [translated(1), translated(0)]})
        result = self.bot.translate_and_summarize_batch([article(0), article(1)])
        self.assertEqual(result, [translated(0), translated(1)])
        schema = self.llm.generate.call_args.kwargs.get("response_schema")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["properties"]), {"articles"})

    def test_invalid_json_does_not_fall_back_to_english(self):
        self.llm.generate.return_value = "not JSON"
        with self.assertRaises(ValueError):
            self.bot.translate_and_summarize_batch([article()])

    def test_api_error_propagates(self):
        self.llm.generate.side_effect = RuntimeError("offline failure")
        with self.assertRaises(RuntimeError):
            self.bot.translate_and_summarize_batch([article()])

    def test_invalid_response_shapes_are_rejected(self):
        values = [
            [translated()],
            {"articles": []},
            {"articles": [None]},
            {"articles": [{**translated(), "article_index": True}]},
            {"articles": [{**translated(), "article_index": "0"}]},
            {"articles": [{**translated(), "article_index": 1}]},
            {"articles": [{**translated(), "translated_title": None}]},
            {"articles": [{**translated(), "translated_title": "  "}]},
            {"articles": [{**translated(), "translated_description": []}]},
            {"articles": [{**translated(), "translated_description": ""}]},
            {"articles": [{**translated(), "unexpected": "field"}]},
            {"articles": [translated()], "unexpected": True},
        ]
        for value in values:
            with self.subTest(value=value):
                self.respond(value)
                with self.assertRaises(ValueError):
                    self.bot.translate_and_summarize_batch([article()])

    def test_duplicate_indices_are_rejected(self):
        self.respond({"articles": [translated(0), translated(0)]})
        with self.assertRaises(ValueError):
            self.bot.translate_and_summarize_batch([article(0), article(1)])

    def test_empty_source_description_may_stay_empty(self):
        value = {**translated(), "translated_description": ""}
        self.respond({"articles": [value]})
        self.assertEqual(self.bot.translate_and_summarize_batch([{**article(), "description": ""}]), [value])

    def test_empty_source_description_cannot_gain_generated_content(self):
        self.respond({"articles": [translated()]})
        with self.assertRaises(ValueError):
            self.bot.translate_and_summarize_batch([{**article(), "description": ""}])

    def test_single_article_uses_same_contract(self):
        self.respond({"articles": [translated()]})
        self.assertEqual(self.bot.translate_and_summarize(article()), translated())

    def test_disabled_llm_is_explicit_original_text_mode(self):
        self.bot.llm_client = None
        result = self.bot.translate_and_summarize_batch([article()])[0]
        self.assertEqual(result["translated_title"], article()["title"])
        self.assertEqual(result.get("translated_description"), "Use std::vector safely.")

    def test_embed_uses_translated_description_without_analysis(self):
        value = {**translated(), "code_analysis": {"purpose": "DO NOT SHOW"}, "cpp_version": "C++26"}
        embed = self.bot.create_discord_embed(article(), value)
        self.assertEqual(embed["description"], value["translated_description"])
        self.assertEqual(embed["title"], value["translated_title"])

    def configure_run(self, articles):
        self.bot.fetch_feeds = Mock(return_value=articles)
        self.bot._load_state = Mock(return_value={"date": "2026-11-08", "sent_today": ["already-sent"]})
        self.bot._save_state = Mock()
        self.bot.send_to_discord = Mock(return_value=True)
        self.bot.send_categorized_to_discord = Mock(return_value=True)
        self.bot.translate_and_summarize_batch = Mock(side_effect=lambda items: [translated(i) for i in range(len(items))])

    def test_limit_applies_before_translation_and_only_selected_ids_are_saved(self):
        self.configure_run([article(i) for i in range(40)])
        self.bot.run()
        calls = self.bot.translate_and_summarize_batch.call_args_list
        self.assertEqual(sum(len(call.args[0]) for call in calls), 35)
        self.assertEqual(len(self.bot.send_to_discord.call_args.args[0]), 35)
        saved = self.bot._save_state.call_args.args[0]["sent_today"]
        self.assertEqual(set(saved), {"already-sent"} | {f"article-{i}" for i in range(35)})

    def test_category_limit_applies_before_translation(self):
        self.configure_run([article(i) for i in range(8)])
        self.bot.categorization_enabled = True
        self.bot.max_articles_per_category = 2
        self.bot.run()
        self.assertEqual(sum(len(c.args[0]) for c in self.bot.translate_and_summarize_batch.call_args_list), 2)
        self.assertEqual(len(self.bot.send_categorized_to_discord.call_args.args[0]), 2)

    def test_later_translation_failure_does_not_post_or_save_anything(self):
        self.configure_run([article(i) for i in range(11)])
        self.bot.translate_and_summarize_batch.side_effect = [[translated(i) for i in range(10)], RuntimeError("failed")]
        with self.assertRaises(RuntimeError):
            self.bot.run()
        self.bot.send_to_discord.assert_not_called()
        self.bot._save_state.assert_not_called()

    def test_discord_failure_propagates_without_saving(self):
        self.configure_run([article()])
        self.bot.send_to_discord.return_value = False
        with self.assertRaises(RuntimeError):
            self.bot.run()
        self.bot._save_state.assert_not_called()

    def test_webhook_error_does_not_log_secret_url(self):
        secret = "webhook-secret-must-not-appear"
        with patch("requests.post", side_effect=daily_digest.requests.RequestException(secret)):
            self.assertFalse(self.bot._send_webhook({"embeds": []}))
        self.assertNotIn(secret, str(daily_digest.logger.error.call_args))

    def test_long_embeds_are_split_by_total_text_budget(self):
        embeds = [{"title": "t" * 100, "description": "d" * 1400,
                   "footer": {"text": "f" * 25}} for _ in range(4)]
        self.bot._send_webhook = Mock(return_value=True)
        self.assertTrue(self.bot.send_to_discord(embeds))
        payloads = [call.args[0] for call in self.bot._send_webhook.call_args_list]
        self.assertEqual([len(p["embeds"]) for p in payloads], [3, 1])
        self.assertTrue(payloads[0]["content"])
        self.assertIsNone(payloads[1]["content"])

    def test_short_embeds_still_obey_ten_embed_limit(self):
        self.bot._send_webhook = Mock(return_value=True)
        self.assertTrue(self.bot.send_to_discord([{"title": "short"}] * 11))
        self.assertEqual([len(c.args[0]["embeds"]) for c in self.bot._send_webhook.call_args_list], [10, 1])

    def test_categorized_sends_split_long_embeds_and_include_custom_categories(self):
        for category in ("general", "custom"):
            with self.subTest(category=category):
                self.bot._send_webhook = Mock(return_value=True)
                items = [{"category": category, "embed": {"description": "d" * 2001}} for _ in range(4)]
                self.assertTrue(self.bot.send_categorized_to_discord(items))
                payloads = [c.args[0] for c in self.bot._send_webhook.call_args_list]
                self.assertEqual([len(p["embeds"]) for p in payloads if "embeds" in p], [2, 2])

    def test_state_save_failure_propagates(self):
        with patch("builtins.open", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.bot._save_state({"sent_today": []})

    def test_real_sdk_translation_to_delivery_and_state_with_mock_http(self):
        requests_seen = []

        def respond(request):
            requests_seen.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": "gpt-6-luna",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": json.dumps({
                        "articles": [translated(1), translated(0)]
                    }, ensure_ascii=False),
                }}],
            })

        with TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(respond)) as http:
            sdk = OpenAI(api_key="test-openai", max_retries=0, http_client=http)
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-openai"}, clear=True), patch("openai.OpenAI", return_value=sdk):
                self.bot.llm_client = daily_digest.create_llm_client(self.bot.config["llm"])
            self.bot.state_file = Path(directory) / "sent_articles.json"
            self.bot.fetch_feeds = Mock(return_value=[article(0), article(1)])
            self.bot._send_webhook = Mock(return_value=True)
            self.bot.run()
            saved = json.loads(self.bot.state_file.read_text())

        self.assertEqual(saved["sent_today"], ["article-0", "article-1"])
        self.assertEqual(len(requests_seen), 1)
        body = requests_seen[0]
        self.assertEqual(body["model"], "gpt-6-luna")
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["response_format"]["json_schema"]["schema"], daily_digest.TRANSLATION_RESPONSE_SCHEMA)
        self.assertNotIn("FULL CONTENT MUST NOT BE SENT", body["messages"][-1]["content"])
        embeds = self.bot._send_webhook.call_args.args[0]["embeds"]
        self.assertEqual([embed["title"] for embed in embeds], ["C++ 소식 0", "C++ 소식 1"])

    def test_cli_failure_exits_without_printing_raw_api_error(self):
        for error_type in (RuntimeError, ValueError):
            with self.subTest(error_type=error_type), patch("daily_digest.load_dotenv"), patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": "https://example.invalid/test"}), patch("daily_digest.CPPDailyDigest", side_effect=error_type("secret-api-key")):
                with self.assertRaises(SystemExit) as raised:
                    daily_digest.main()
            self.assertEqual(raised.exception.code, 1)
        self.assertNotIn("secret-api-key", str(daily_digest.logger.error.call_args))

    def test_cli_missing_key_reports_required_environment_name(self):
        with patch("daily_digest.load_dotenv"), patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": "https://example.invalid/test"}, clear=True):
            with self.assertRaises(SystemExit) as raised:
                daily_digest.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("OPENAI_API_KEY", str(daily_digest.logger.error.call_args))

    def test_cli_validation_failure_is_explained_without_echoing_response(self):
        self.respond({"articles": [{**translated(), "translated_title": {"secret": "private-response-data"}}]})
        self.bot.run = Mock(side_effect=lambda: self.bot.translate_and_summarize_batch([article()]))
        with patch("daily_digest.load_dotenv"), patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": "https://example.invalid/test"}), patch("daily_digest.CPPDailyDigest", return_value=self.bot):
            with self.assertRaises(SystemExit):
                daily_digest.main()
        diagnostic = str(daily_digest.logger.error.call_args)
        self.assertIn("번역 제목", diagnostic)
        self.assertNotIn("private-response-data", diagnostic)

    def test_no_articles_does_not_call_llm_or_post(self):
        self.configure_run([])
        self.bot.run()
        self.bot.translate_and_summarize_batch.assert_not_called()
        self.bot.send_to_discord.assert_not_called()

    def test_import_does_not_probe_dotenv(self):
        seen = []
        def record(path):
            if path.name == ".env":
                seen.append(path)
                return False
            return _exists(path)
        spec = importlib.util.spec_from_file_location("digest_import_probe", ROOT / "daily_digest.py")
        module = importlib.util.module_from_spec(spec)
        with patch.object(Path, "exists", record):
            spec.loader.exec_module(module)
        self.assertEqual(seen, [])

    def test_workflow_preserves_schedule_and_openai_secret(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/daily-digest.yml").read_text())
        triggers = workflow.get("on", workflow.get(True))  # PyYAML YAML 1.1 boolean key.
        self.assertEqual(triggers["schedule"][0]["cron"], "0 0 * * *")
        steps = workflow["jobs"]["send-digest"]["steps"]
        digest_step = next(s for s in steps if s.get("run") == "python daily_digest.py")
        self.assertEqual(digest_step["env"]["OPENAI_API_KEY"], "${{ secrets.OPENAI_API_KEY }}")
        self.assertTrue(any("unittest discover" in s.get("run", "") for s in steps))


if __name__ == "__main__":
    unittest.main()
