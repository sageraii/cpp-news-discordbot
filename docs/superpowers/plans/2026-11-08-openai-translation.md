# OpenAI Translation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for the assigned slice;
> tests must fail for the missing behavior before production changes.

**Goal:** Run the digest with low-cost, no-reasoning OpenAI translation.
**Architecture:** Preserve the provider interface and webhook pipeline; use a
strict translation response contract and explicit failures instead of English
fallback on API errors.
**Tech Stack:** Python 3.11+, existing OpenAI SDK 2.14.0, unittest, GitHub Actions.
**Spec:** ../specs/2026-11-08-openai-translation.md

## Global constraints

- No live API/Discord calls, key reads, secret writes, commits, pushes or dispatch.
- Preserve 09:00 KST schedule and unrelated provider support.
- No new runtime/test dependencies. Pin the locally exercised OpenAI SDK.
- Scope excludes full-body scraping and durable delivery/cache redesign.

## Review focus

- Missing or wrong-provider key must fail without leaking key values.
- JSON can be valid while fields, indices, counts or titles are invalid.
- Refusal/truncation must not be converted into success or marked sent.
- Unposted/capped articles must not incur translation calls or be marked sent.
- Mock tests must not read .env or contact external endpoints.

## Task 1: OpenAI transport (delegated executor)

Files: llm_client.py, requirements.txt, tests/test_llm_client.py.
Interface: `generate(prompt, system_prompt=None, response_schema=None) -> str`.

- [x] Add red tests for request parameters, provider-key selection, structured
  output, refusal/length/empty output, and transient versus permanent retries.
- [x] Implement compatible request handling and safe usage logging.
- [x] Run isolated unittest tests using mocked HTTP; pin tested openai==2.14.0.

## Task 2: Translation pipeline and deployment docs (main)

Files: daily_digest.py, config.yaml, prompts/system.txt,
prompts/translate_summarize*.txt, tests/test_daily_digest.py,
.github/workflows/daily-digest.yml, .env.example, README.md.

- [x] Add red tests for translation-only contract, invalid/missing/duplicate
  results, no implicit error fallback, selected limits/IDs, posting failure.
- [x] Implement minimal schema validation and pipeline changes; move dotenv load
  to CLI entry so test imports do not read local credentials.
- [x] Set Luna/none defaults and simplify prompts/config/example documentation.
- [x] Add offline unittest workflow step; preserve Secret mapping and cron.

## Task 3: Integration and review

- [x] Run `.venv/bin/python -B -m unittest discover -s tests -v`.
- [x] Run AST syntax checks, `git diff --check`, workflow/config contract checks.
- [x] Fresh read-only reviewer verifies diff, scope, tests and no secret exposure.
- [x] Report local verification separately from pending Secret/model/live checks.

## Execution evidence

- Transport: RED request/retry/key/response tests before implementation, then
  10 passing SDK + MockTransport tests. No live requests.
- Digest: RED translation/validation/selection/error tests before implementation.
- Delivery follow-up: 24 tests produced 4 expected failures before fixing the
  Discord text budget, custom-category delivery and state-write error handling.
- Integrated suite: 42 tests passed on Python 3.14.7, including a real-SDK mock
  request through response parsing, webhook payload creation and temporary state.
- Review fixes: rejected content invented for an empty source description and
  introduced log-safe application diagnostics, with 3 digest RED tests observed
  before fixes. Added Luna-only default reasoning and SDK error isolation tests.
- Python 3.11 AST syntax check and git diff --check passed. Python 3.11 runtime
  is not installed at the checked local paths; CI execution remains unverified.
- No existing formatter/linter configuration or installed lint tool was used.
- Repository Secret metadata lacks OPENAI_API_KEY; no secret values were read.
- No API billing, external posting, installation, commit, push or deployment.
- Discord budget follows https://docs.discord.com/developers/resources/message#embed-limits.

- Final focused independent review: APPROVE, both initial medium findings fixed;
  42 tests and diff check revalidated. No remaining code review blockers.
- Handoff: local uncommitted changes remain on codex/openai-translation. Add
  OPENAI_API_KEY in GitHub repository Secrets before deployment. Live model,
  translation quality and actual Discord delivery have not been exercised.

## Deployment authorization update — 2026-11-09

The user registered OPENAI_API_KEY; GitHub Secret name metadata was verified.
The user then explicitly approved applying changes to main and running Actions
once, including API billing and actual Discord delivery. This supersedes the
implementation-only restrictions above for this deployment. Local pre-deploy
verification: 42 tests passed; no active workflow run; origin/main unchanged.
Secret values remain unread and must not appear in source or logs.
