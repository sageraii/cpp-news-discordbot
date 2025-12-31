# C++ Daily Digest Bot

[![C++ Daily Digest](https://github.com/sageraii/cpp-news-discordbot/actions/workflows/daily-digest.yml/badge.svg)](https://github.com/sageraii/cpp-news-discordbot/actions/workflows/daily-digest.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

C++ 관련 RSS 피드를 매일 수집하여 Discord 채널로 한국어 요약을 전송하는 봇입니다.

## 주요 기능

- **RSS 피드 수집**: 16개의 C++ 관련 블로그 및 커뮤니티 피드
- **AI 번역/요약**: LLM을 활용한 한국어 번역 및 요약 (C++ 전문 용어 처리)
- **코드 분석**: 기사 내 C++ 코드 자동 추출 및 분석
- **카테고리 분류**: 기사를 주제별로 자동 분류 (표준, 성능, 동시성 등)
- **자동 실행**: GitHub Actions로 매일 오전 9시(KST) 자동 전송
- **중복 방지**: 이미 전송한 기사는 다시 전송하지 않음

## 지원 LLM 프로바이더

| 프로바이더 | 모델 예시 | 비고 |
|------------|-----------|------|
| OpenAI | `gpt-4o`, `gpt-4o-mini` | 안정적, 범용 |
| Anthropic | `claude-sonnet-4-20250514` | C++ 이해도 높음 |
| Google | `gemini-2.0-flash-exp` | 무료 티어 있음 |
| OpenRouter | 다양한 모델 | 여러 모델 통합 |
| Ollama | `llama3`, `mistral` | 무료, 로컬 전용 |

## 설치 및 실행

### 1. 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env.example`을 `.env`로 복사하고 값을 설정합니다:

```bash
cp .env.example .env
```

```env
# 필수
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxxxx/yyyyy

# LLM API 키 (프로바이더에 따라 선택)
GOOGLE_API_KEY=your_api_key
# 또는
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. LLM 프로바이더 설정

`config.yaml`에서 사용할 LLM 프로바이더를 설정합니다:

```yaml
llm:
  enabled: true
  provider: google  # openai | anthropic | google | openrouter | ollama
  model: gemini-2.0-flash-exp
```

### 4. 로컬 실행

```bash
python daily_digest.py
```

## GitHub Actions 설정

### 1. Repository Secrets 설정

GitHub 저장소 → Settings → Secrets and variables → Actions에서 설정:

| Secret 이름 | 설명 |
|-------------|------|
| `DISCORD_WEBHOOK_URL` | Discord 웹훅 URL |
| `GOOGLE_API_KEY` | Google Gemini API 키 (또는 다른 LLM 키) |

### 2. 수동 실행

Actions 탭 → C++ Daily Digest → Run workflow

### 3. 자동 실행

매일 오전 9시(KST)에 자동 실행됩니다.

## 프로젝트 구조

```
├── daily_digest.py          # 메인 실행 파일
├── llm_client.py            # LLM API 클라이언트
├── code_analyzer.py         # C++ 코드 추출/분석
├── config.yaml              # 설정 파일 (피드, LLM, Discord)
├── requirements.txt         # Python 의존성
├── prompts/
│   ├── system.txt           # 시스템 프롬프트
│   ├── translate_summarize.txt  # 번역/요약 프롬프트
│   └── code_analysis.txt    # 코드 분석 프롬프트
└── .github/workflows/
    └── daily-digest.yml     # GitHub Actions 워크플로우
```

## RSS 피드 목록 (16개)

| 카테고리 | 피드 |
|----------|------|
| 공식 | ISO C++, Herb Sutter, Microsoft C++ Team, Barry Revzin |
| 블로그 | C++ Stories, Modernes C++, Fluent C++, Arthur O'Dwyer, Andrzej Krzemienski, Sandor Dargo, Shafik Yaghmour |
| 커뮤니티 | Reddit r/cpp, JetBrains CLion, Easyperf, KDAB, Hacking C++ |

피드를 추가/삭제하려면 `config.yaml`의 `feeds` 섹션을 수정하세요.

## 카테고리 분류

기사를 주제별로 자동 분류하여 Discord에 그룹화하여 전송할 수 있습니다.

### 설정

`config.yaml`에서 활성화:

```yaml
categorization:
  enabled: true  # false로 설정하면 분류 없이 전송
```

### 카테고리 목록

| 카테고리 | 설명 | 키워드 예시 |
|----------|------|-------------|
| 📋 표준 및 제안 | C++ 표준, WG21 제안 | c++23, c++26, proposal, wg21 |
| ✨ 모던 C++ | 최신 C++ 기능 | ranges, concepts, modules, constexpr |
| ⚡ 성능 최적화 | 성능 관련 | performance, optimization, simd, cache |
| 🔄 동시성 | 멀티스레딩, 코루틴 | coroutine, thread, async, atomic |
| 🛠️ 도구 및 빌드 | 컴파일러, 빌드 시스템 | cmake, clang, gcc, sanitizer |
| 🛡️ 안전성 | 메모리 안전성 | safety, memory, undefined behavior |
| 📰 일반 | 기타 | - |

분류는 LLM의 category_hint 또는 키워드 매칭으로 자동 수행됩니다.

## Discord 출력 예시

```
📰 C++ Daily Digest - 2025년 01월 01일

[C++20 코루틴을 활용한 비동기 파일 I/O 구현]
이 기사는 C++20 코루틴을 사용하여 비동기 파일 읽기/쓰기를
구현하는 방법을 설명합니다.

코드: 비동기 파일 읽기를 위한 awaitable 타입 구현
사용 기능: co_await, co_return, std::coroutine_handle
표준: C++20
```

## 라이선스

MIT License
