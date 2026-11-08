# OpenAI translation-only digest

## Approved outcome

The user approved GPT-6 Luna with reasoning disabled for Korean translation of
article titles and RSS descriptions. Keep GitHub Actions at 09:00 KST and reuse
its existing `OPENAI_API_KEY` Secret mapping. No live API request, Discord post,
push, workflow dispatch, or secret value access is authorized by this local edit.

## Design

- Keep the existing Python/LLMClient/Discord Webhook architecture and optional
  non-OpenAI providers. Default to `openai`, `gpt-6-luna`, reasoning `none`.
- Keep `generate()` returning text; add optional `response_schema` to the
  provider-neutral call. OpenAI enforces strict JSON Schema; other providers
  retain prompt-based JSON followed by the same application validation.
- Translation output is `{articles: [{article_index, translated_title,
  translated_description}]}`. Validate count, unique integer indices, ranges,
  nonempty titles and string descriptions before any posting. No model code
  analysis, category inference, summarization, or invented information.
- RSS description text remains bounded to 500 characters, explicitly documented;
  this is not full-page translation. Strip HTML. Existing terminology rules stay.
- Translation failure aborts before posting and must not mark original English
  text as successful translation. Existing explicitly disabled LLM mode may
  continue to use original text.
- Limit selected articles before paid translation, then record only selected,
  successfully posted articles. Propagate Discord failure to Actions.
- SDK retry handling has one owner, with no retries on authentication/invalid
  requests. Validate refusal, truncation and empty content, and log token counts
  without prompt/key values. Keep compatibility with other provider parameters.
- Keep optional keyword categorization, but remove LLM code-analysis fields from
  the active translation path. No durable outbox/cache-history redesign here.
- Tests use unittest and mocked HTTP; importing the app must not read `.env`.

## Scope and evidence boundaries

2026-11-08: GitHub repository Secret metadata lists only DISCORD_WEBHOOK_URL and
GOOGLE_API_KEY. OPENAI_API_KEY must be added by the user through GitHub's secret UI
before deployment. Local mock tests cannot prove model access or Korean quality.
Changes are made on `codex/openai-translation` in the existing clean checkout;
no new worktree, merge, commit, or push is needed for this authorized local edit.

## Acceptance

Offline request tests verify Luna/none/max_completion_tokens/strict schema and
OpenAI key selection. Digest tests verify translation-only inputs, output
validation, fail-before-post behavior, selected-ID persistence and Discord error
propagation. Workflow runs tests before the daily digest. README documents setup,
Secret requirements, test command, and translation/input limits.

## Deployment authorization update — 2026-11-09

The user registered OPENAI_API_KEY; GitHub Secret name metadata was verified.
The user then explicitly approved applying changes to main and running Actions
once, including API billing and actual Discord delivery. This supersedes the
implementation-only restrictions above for this deployment. Local pre-deploy
verification: 42 tests passed; no active workflow run; origin/main unchanged.
Secret values remain unread and must not appear in source or logs.
