# C++ Daily Digest Bot

[![C++ Daily Digest](https://github.com/sageraii/cpp-news-discordbot/actions/workflows/daily-digest.yml/badge.svg)](https://github.com/sageraii/cpp-news-discordbot/actions/workflows/daily-digest.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

C++ RSS 피드의 **제목과 설명을 한국어로 번역**하여 Discord에 전송합니다.
기본 설정은 OpenAI `gpt-6-luna`, `reasoning_effort: none`입니다.
모델에 요약·코드 분석·카테고리 추론을 요청하지 않습니다.

## 처리 범위

```text
RSS 수집 → 중복/게시 상한 적용 → 제목·설명 번역 → 응답 검증 → Discord 게시 → 상태 저장
```

- 16개 피드에서 최근 24시간 기사를 수집합니다. 발행일이 없는 기사는 제외하지 않습니다.
- 번역 입력은 제목과 **HTML을 제거한 RSS 설명의 앞 500자**입니다. 원문 페이지 전체를 가져오거나 번역하지 않습니다.
- 기본적으로 기사 10개를 한 API 요청에 넣습니다. 이는 OpenAI Batch API가 아닌 일반 Chat Completions 호출입니다.
- 기본 게시 상한은 35개입니다. 선택적 키워드 분류를 켜면 카테고리당 5개입니다. 상한은 유료 번역 전에 적용합니다.
- 번역 결과의 JSON 구조, 기사 번호, 개수, 필수 문자열을 검증합니다. 실패하면 영어 원문으로 대신 게시하지 않고 실행을 실패 처리합니다.
- Discord 메시지는 임베드 개수와 전체 텍스트 길이에 따라 나눕니다. [Discord 제한](https://docs.discord.com/developers/resources/message#embed-limits)
- GitHub Actions 실행 시각은 매일 **09:00 KST**로 예약되어 있습니다. 실제 시작 시각은 GitHub 실행 대기 상황에 따라 늦어질 수 있습니다.

## GitHub Actions 설정

저장소의 [Settings → Secrets and variables → Actions](https://github.com/sageraii/cpp-news-discordbot/settings/secrets/actions)에 다음 **Repository secrets**를 등록합니다.

| Secret 이름 | 값 |
|---|---|
| `DISCORD_WEBHOOK_URL` | 게시할 Discord 채널의 웹훅 URL |
| `OPENAI_API_KEY` | 해당 모델을 사용할 수 있는 OpenAI API 키 |

워크플로우에는 `${{ secrets.OPENAI_API_KEY }}` → `OPENAI_API_KEY` 환경 변수 연결이 이미 있습니다.
기존 `GOOGLE_API_KEY`만으로는 OpenAI를 호출할 수 없습니다. 키는 코드·커밋·대화창에 넣지 마세요.
`LLM_API_KEY`도 기존 호환용으로 지원하지만, 기본 설정에는 `OPENAI_API_KEY`를 권장합니다.

변경사항과 Secret을 준비한 다음 기본 브랜치에 반영하면 예약 실행에 적용됩니다.
수동 실행은 **Actions → C++ Daily Digest → Run workflow**이며, 실제 API 비용과 Discord 게시가 발생합니다.
예약은 `.github/workflows/daily-digest.yml`의 `0 0 * * *`(UTC)로 관리합니다.
워크플로우는 오프라인 테스트가 통과한 뒤에만 봇을 실행합니다.

## 로컬 설치와 실행

Python 3.11 이상에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 처음 설정할 때만 복사합니다. 기존 .env를 덮어쓰지 마세요.
test -f .env || cp .env.example .env
```

`.env`에 `DISCORD_WEBHOOK_URL`과 `OPENAI_API_KEY`를 입력합니다.
CLI 실행 시에만 `.env`를 읽으며, 이미 설정된 환경 변수는 덮어쓰지 않습니다.

```bash
# 오프라인 회귀 테스트: 키 불필요, API/RSS/Discord 네트워크 호출 없음
python -B -m unittest discover -s tests -v

# 실제 RSS 수집, 유료 번역 및 Discord 게시
python daily_digest.py
```

## 모델 및 출력 설정

```yaml
llm:
  enabled: true
  provider: openai
  model: gpt-6-luna
  reasoning_effort: none
  temperature: 0.3
  max_completion_tokens: 8000
  batch_size: 10
  timeout: 120
  retry_count: 3
  retry_delay: 5
  rate_limit_delay: 1
```

OpenAI에는 strict JSON Schema를 전달합니다. reasoning을 켜면 `temperature`는 전송하지 않습니다.
OpenAI SDK 자체 재시도는 끄고, 연결/타임아웃 및 HTTP 408·409·429·5xx 오류만 설정 횟수 내에서 재시도합니다.
`retry_count: 3`은 최초 요청을 포함한 총 3회입니다. 인증 오류나 잘못된 응답은 즉시 실패합니다.
출력 토큰 한도 초과도 실패하므로, 필요하면 `batch_size`를 줄이거나 출력 상한을 조정하세요.
모델 사양은 [공식 OpenAI 문서](https://developers.openai.com/api/docs/models/gpt-6-luna)를 참고하세요.

기존 Anthropic, Google, OpenRouter, Ollama, custom 클라이언트는 유지합니다.
전환 시 해당 제공자의 키·모델·`max_tokens`를 따로 설정해야 합니다.
다른 제공자는 프롬프트 기반 JSON을 반환하고 동일한 애플리케이션 검증을 거칩니다.
`llm.enabled: false`는 명시적으로 번역 없이 원문을 게시하는 모드입니다.

## 피드 및 분류

`config.yaml`의 `feeds`에서 피드를 추가·삭제할 수 있습니다.
ISO C++, Herb Sutter, Microsoft C++ Team, Barry Revzin, C++ Stories,
Modernes C++, Fluent C++, Arthur O'Dwyer, Andrzej Krzemienski, Sandor Dargo,
Shafik Yaghmour, Reddit r/cpp, JetBrains CLion, Easyperf, KDAB, Hacking C++가 설정되어 있습니다.

`categorization.enabled: true`이면 **원문 제목·RSS 설명의 키워드**로 그룹화합니다.
기본 카테고리는 표준, 모던 C++, 성능, 동시성, 도구, 안전성, 일반입니다.
LLM 분류나 코드 분석은 사용하지 않습니다.

## 오류와 중복 처리의 한계

- 모든 선택 기사의 번역이 검증된 뒤에만 게시합니다. 번역 실패 시 게시·완료 상태 저장을 하지 않습니다.
- 전송 실패나 상태 저장 실패는 CLI 종료 코드 1로 Actions에 전달합니다.
- 성공하면 게시한 기사 ID만 `sent_articles.json`에 저장하고 Actions cache로 다음 실행에 전달합니다.
- 상태는 실행 환경의 날짜가 바뀌면 초기화하는 기존 방식입니다. 영구 중복 방지는 아닙니다.
- 일부 메시지 게시 후 실패, 동시 실행 또는 cache 유실 시 재실행에서 중복 게시될 수 있습니다. 메시지별 영구 전송 기록은 구현하지 않았습니다.
- 오프라인 테스트는 요청 형식과 오류 처리를 검증합니다. 계정의 모델 접근 권한, 실제 번역 품질과 Discord 전달은 별도 실행으로 확인해야 합니다.

## 프로젝트 구조

```text
├── daily_digest.py                    # 수집·선택·번역 검증·게시
├── llm_client.py                      # 제공자별 API 클라이언트
├── config.yaml                       # 피드·모델·출력 설정
├── prompts/
│   ├── system.txt                    # 번역 및 용어 규칙
│   └── translate_summarize_batch.txt  # 파일명 유지, 번역 전용 계약
├── tests/                            # unittest, 모의 HTTP
└── .github/workflows/daily-digest.yml # 예약 및 수동 실행
```

`code_analyzer.py`와 `prompts/code_analysis.txt`는 기존 파일로 남아 있지만 현재 봇 경로에서는 사용하지 않습니다.

## 라이선스

MIT License
