# CLAUDE.md - C++ Daily Digest Bot

## 프로젝트 개요

- **목적**: C++ 관련 RSS 피드를 매일 수집하여 Discord 채널로 일일 요약 전송
- **아키텍처**: GitHub Actions 기반 서버리스 아키텍처 (별도 서버 불필요)
- **언어**: Python 3.11+
- **핵심 라이브러리**: `feedparser`, `requests`, `pyyaml`, `python-dotenv`, `openai` (또는 선택한 LLM SDK), `beautifulsoup4`
- **실행 방식**: Discord Webhook (봇 토큰 불필요)

## 빌드 및 실행 명령어

```bash
# 의존성 설치
pip install -r requirements.txt

# 로컬 실행 (테스트용)
python daily_digest.py

# 린트/포맷
flake8 . && black .
```

## 환경 설정

환경 변수는 두 가지 방식으로 설정할 수 있습니다:

### 방식 1: 셸 export (일시적)

터미널 세션에서 직접 설정:

```bash
# 필수
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/xxxxx/yyyyy"

# LLM 프로바이더에 따라 하나 선택
export LLM_API_KEY="your_api_key"           # 범용 키 (config.yaml의 provider에 따라 사용)
export OPENAI_API_KEY="sk-..."              # OpenAI 전용
export ANTHROPIC_API_KEY="sk-ant-..."       # Anthropic 전용
export GOOGLE_API_KEY="..."                 # Google 전용
export OPENROUTER_API_KEY="sk-or-..."       # OpenRouter 전용

# 실행
python daily_digest.py
```

### 방식 2: .env 파일 (권장, 영구적)

프로젝트 루트에 `.env` 파일 생성:

```env
# .env - 환경 변수 설정 파일
# 이 파일은 .gitignore에 추가하여 버전 관리에서 제외할 것

# Discord 웹훅 (필수)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxxxx/yyyyy

# LLM API 키 (프로바이더에 따라 하나 이상 설정)
# 범용 키 - config.yaml의 provider 설정에 따라 자동 사용
LLM_API_KEY=your_api_key_here

# 또는 프로바이더별 개별 키 설정
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-...

# 디버그 모드 (선택)
DEBUG=false
```

### .env 파일 로드 코드

`daily_digest.py` 상단에 dotenv 로드 추가:

```python
# daily_digest.py
import os
from pathlib import Path

# .env 파일 로드 (있는 경우)
def load_dotenv():
    """프로젝트 루트의 .env 파일에서 환경 변수 로드"""
    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        return
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 빈 줄, 주석 무시
            if not line or line.startswith('#'):
                continue
            # KEY=VALUE 파싱
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                # 따옴표 제거
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                # 이미 설정된 환경 변수는 덮어쓰지 않음
                if key not in os.environ:
                    os.environ[key] = value

# 앱 시작 시 로드
load_dotenv()
```

또는 `python-dotenv` 패키지 사용:

```python
# python-dotenv 사용 시
from dotenv import load_dotenv
load_dotenv()  # .env 파일 자동 로드
```

### GitHub Secrets (GitHub Actions용)

GitHub 저장소 Settings > Secrets and variables > Actions에서 설정:

| Secret 이름 | 설명 | 필수 |
|-------------|------|------|
| `DISCORD_WEBHOOK_URL` | Discord 웹훅 URL | O |
| `LLM_API_KEY` | LLM API 키 | O |

### 환경 변수 우선순위

1. 셸 export (가장 높음)
2. GitHub Secrets (Actions 실행 시)
3. .env 파일 (가장 낮음)

### .gitignore 설정

```gitignore
# 환경 변수 파일 (API 키 보호)
.env
.env.local
.env.*.local

# 상태 파일
sent_articles.json

# Python
__pycache__/
*.pyc
.venv/
```

## 코드 구조

```
.
├── daily_digest.py              # 메인 실행 파일
├── config.yaml                  # 통합 설정 파일 (피드 + AI + 기타)
├── llm_client.py                # LLM API 추상화 클라이언트
├── code_analyzer.py             # C++ 코드 추출 및 분석 모듈
├── prompts/
│   ├── system.txt               # 시스템 프롬프트
│   ├── translate_summarize.txt  # 번역/요약 프롬프트 템플릿
│   └── code_analysis.txt        # C++ 코드 분석 프롬프트
├── sent_articles.json           # 상태 관리 파일 (GitHub Actions 캐시)
├── requirements.txt             # Python 의존성
├── .env                         # 환경 변수 파일 (git 제외)
├── .env.example                 # 환경 변수 예시 파일
├── .gitignore                   # Git 제외 파일 목록
├── .github/workflows/
│   └── daily-digest.yml         # GitHub Actions 워크플로우
└── CLAUDE.md                    # 이 파일
```

---

## GitHub Actions 워크플로우

### .github/workflows/daily-digest.yml

```yaml
name: C++ Daily Digest

on:
  schedule:
    # 매일 한국시간 오전 9시 (UTC 0시)
    - cron: '0 0 * * *'
  
  # 수동 실행 (테스트용)
  workflow_dispatch:

jobs:
  send-digest:
    runs-on: ubuntu-latest
    
    steps:
      - name: 코드 체크아웃
        uses: actions/checkout@v4
      
      - name: Python 환경 설정
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: 의존성 설치
        run: pip install -r requirements.txt
      
      - name: 이전 상태 복원
        uses: actions/cache/restore@v4
        with:
          path: sent_articles.json
          key: digest-state-${{ github.run_id }}
          restore-keys: |
            digest-state-
      
      - name: 일일 요약 실행
        run: python daily_digest.py
        env:
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
          LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
      
      - name: 상태 저장
        uses: actions/cache/save@v4
        if: always()
        with:
          path: sent_articles.json
          key: digest-state-${{ github.run_id }}
```

### requirements.txt

```
feedparser>=6.0.10
requests>=2.31.0
pyyaml>=6.0.1
beautifulsoup4>=4.12.0
python-dotenv>=1.0.0
openai>=1.0.0
anthropic>=0.18.0
google-generativeai>=0.4.0
```

---

## C++ 코드 분석 기능

### 개요

RSS 피드의 기사에 C++ 코드가 포함된 경우, 코드를 추출하여 분석하고 요약에 포함합니다.

### 코드 추출 방식

1. RSS description/content에서 `<code>`, `<pre>` 태그 추출
2. Markdown 코드 블록 (```cpp, ```c++) 추출
3. 추출된 코드가 C++ 코드인지 휴리스틱 판단

### code_analyzer.py

```python
# code_analyzer.py
import re
from bs4 import BeautifulSoup
from typing import List, Optional, Dict

class CppCodeAnalyzer:
    """C++ 코드 추출 및 분석"""
    
    # C++ 코드 식별 패턴
    CPP_INDICATORS = [
        r'#include\s*<',           # 헤더 인클루드
        r'std::',                   # std 네임스페이스
        r'template\s*<',            # 템플릿
        r'constexpr\s+',            # constexpr
        r'auto\s+\w+\s*=',          # auto 변수
        r'->\s*\w+',                # trailing return type
        r'nullptr',                 # nullptr
        r'::',                      # 스코프 연산자
        r'\bclass\s+\w+',           # 클래스 선언
        r'\bstruct\s+\w+',          # 구조체 선언
        r'\bnamespace\s+\w+',       # 네임스페이스
        r'unique_ptr|shared_ptr',   # 스마트 포인터
        r'vector<|map<|string>',    # STL 컨테이너
        r'co_await|co_yield',       # 코루틴
        r'concept\s+\w+',           # 컨셉
        r'requires\s*\(',           # requires 절
    ]
    
    def __init__(self, min_lines: int = 3, max_length: int = 2000):
        self.min_lines = min_lines
        self.max_length = max_length
    
    def extract_code_blocks(self, html_content: str) -> List[Dict]:
        """HTML/Markdown에서 코드 블록 추출"""
        
        code_blocks = []
        
        # 1. HTML <pre><code> 태그에서 추출
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for pre in soup.find_all('pre'):
            code = pre.get_text(strip=True)
            if self._is_cpp_code(code):
                code_blocks.append({
                    'code': code[:self.max_length],
                    'source': 'html_pre'
                })
        
        for code_tag in soup.find_all('code'):
            if code_tag.parent.name != 'pre':  # pre 내부가 아닌 경우만
                code = code_tag.get_text(strip=True)
                if len(code.split('\n')) >= self.min_lines and self._is_cpp_code(code):
                    code_blocks.append({
                        'code': code[:self.max_length],
                        'source': 'html_code'
                    })
        
        # 2. Markdown 코드 블록 추출
        md_pattern = r'```(?:cpp|c\+\+|cxx)?\s*\n(.*?)```'
        for match in re.finditer(md_pattern, html_content, re.DOTALL | re.IGNORECASE):
            code = match.group(1).strip()
            if self._is_cpp_code(code):
                code_blocks.append({
                    'code': code[:self.max_length],
                    'source': 'markdown'
                })
        
        return code_blocks
    
    def _is_cpp_code(self, code: str) -> bool:
        """C++ 코드인지 휴리스틱 판단"""
        
        if len(code.split('\n')) < self.min_lines:
            return False
        
        # C++ 특징 패턴 매칭
        matches = sum(1 for pattern in self.CPP_INDICATORS 
                      if re.search(pattern, code))
        
        return matches >= 2  # 2개 이상 매칭 시 C++ 코드로 판단
    
    def get_code_summary_prompt(self, code_blocks: List[Dict]) -> Optional[str]:
        """코드 분석을 위한 프롬프트 생성"""
        
        if not code_blocks:
            return None
        
        # 가장 긴 코드 블록 선택 (최대 2개)
        sorted_blocks = sorted(code_blocks, 
                               key=lambda x: len(x['code']), 
                               reverse=True)[:2]
        
        code_text = "\n\n---\n\n".join(
            f"```cpp\n{block['code']}\n```" 
            for block in sorted_blocks
        )
        
        return code_text
```

### 코드 분석 프롬프트 (prompts/code_analysis.txt)

```text
다음 C++ 코드를 분석하고 요약해주세요.

## 코드
{code}

## 분석 요청사항
1. 코드의 핵심 목적 (1문장)
2. 사용된 주요 C++ 기능/기법 (키워드 나열)
3. 주목할 만한 패턴이나 기법 (있는 경우)

## 출력 형식 (JSON)
{{
  "purpose": "이 코드는 ~을 수행합니다",
  "cpp_features": ["constexpr", "ranges", "concepts"],
  "notable_patterns": "CRTP 패턴을 사용하여 정적 다형성 구현" 또는 null
}}

## 주의사항
- C++ 키워드(constexpr, auto 등)는 원문 유지
- 간결하게 핵심만 설명
- 코드가 불완전해도 보이는 부분만 분석
```

---

## AI 번역 및 요약 기능

### 개요

이 봇은 LLM API를 사용하여:
1. 영어 기사를 한국어로 번역
2. C++ 전문 지식을 반영한 요약 생성
3. 기사 내 C++ 코드가 있으면 코드 분석 포함

### 지원 LLM 프로바이더

| 프로바이더 | 모델 예시 | 환경 변수 | 비고 |
|------------|-----------|-----------|------|
| OpenAI | `gpt-4o`, `gpt-4o-mini` | `OPENAI_API_KEY` | 안정적, 범용 |
| Anthropic | `claude-sonnet-4-20250514`, `claude-haiku-4-20250514` | `ANTHROPIC_API_KEY` | C++ 이해도 높음 |
| Google | `gemini-1.5-flash`, `gemini-1.5-pro` | `GOOGLE_API_KEY` | 무료 티어 있음 |
| OpenRouter | 다양한 모델 | `OPENROUTER_API_KEY` | 여러 모델 통합 |
| Ollama | `llama3`, `mistral` | (로컬) | 무료, 로컬 전용 |
| Custom | 사용자 정의 | `LLM_API_KEY` | OpenAI 호환 엔드포인트 |

---

## 통합 설정 파일 (config.yaml)

```yaml
# config.yaml - C++ Daily Digest 통합 설정

# =============================================================================
# LLM API 설정
# =============================================================================
llm:
  enabled: true
  provider: openai                # openai | anthropic | google | openrouter | ollama | custom
  model: gpt-4o-mini
  # base_url: https://api.openai.com/v1    # custom 엔드포인트 사용 시
  temperature: 0.3
  max_tokens: 800                 # 코드 분석 포함 시 늘림
  timeout: 30
  retry_count: 3
  retry_delay: 2
  rate_limit_delay: 1

# =============================================================================
# 번역 및 요약 설정
# =============================================================================
translation:
  target_language: ko
  summary_style: concise
  summary_length: 2-3
  translate_title: true
  preserve_technical_terms: true

# =============================================================================
# C++ 코드 분석 설정
# =============================================================================
code_analysis:
  enabled: true                   # 코드 분석 활성화
  min_code_lines: 3               # 최소 코드 줄 수
  max_code_length: 2000           # 최대 코드 길이 (토큰 절약)
  include_in_summary: true        # 요약에 코드 분석 포함

# =============================================================================
# RSS 피드 목록
# =============================================================================
feeds:
  # 공식 및 표준
  ISO C++: https://isocpp.org/blog/rss
  Herb Sutter: https://herbsutter.com/feed/
  Microsoft C++ Team: https://devblogs.microsoft.com/cppblog/feed/
  Barry Revzin: https://brevzin.github.io/feed.xml
  
  # 고품질 기술 블로그
  C++ Stories: https://www.cppstories.com/index.xml
  Modernes C++: https://www.modernescpp.com/index.php/feed
  Fluent C++: https://www.fluentcpp.com/feed/
  Arthur O'Dwyer: https://quuxplusone.github.io/blog/feed.xml
  Andrzej Krzemienski: https://akrzemi1.wordpress.com/feed/
  
  # 커뮤니티 및 도구
  Reddit r/cpp: https://www.reddit.com/r/cpp/top/.rss?t=day
  JetBrains CLion: https://blog.jetbrains.com/clion/feed/
  Easyperf: https://easyperf.net/feed.xml

# =============================================================================
# Discord 설정
# =============================================================================
discord:
  embed_color: 0x0052CC
  show_summary: true
  show_code_analysis: true        # 코드 분석 결과 표시
  show_original_title: false
  max_articles_per_category: 5

# =============================================================================
# 스케줄 설정
# =============================================================================
schedule:
  timezone: Asia/Seoul
  hour: 9
  lookback_hours: 24
```

---

## 프롬프트 템플릿

### 시스템 프롬프트 (prompts/system.txt)

```text
당신은 C++ 프로그래밍 언어 전문가이자 기술 번역가입니다.

## 역할
- C++ 관련 영어 기사를 한국어로 번역하고 요약합니다.
- 기사에 C++ 코드가 포함된 경우, 코드의 핵심을 분석하여 요약에 포함합니다.
- C++ 표준(C++11 ~ C++26), 컴파일러, 라이브러리에 대한 깊은 이해를 바탕으로 정확한 번역을 제공합니다.

## C++ 전문 용어 처리 규칙

영어 원문 유지:
- 키워드: constexpr, consteval, constinit, concept, requires, co_await, co_yield, co_return, auto, decltype
- 약어: RAII, SFINAE, CTAD, NRVO, RVO, ABI, ODR, ADL, POD, UB, CRTP, EBO
- 타입 특성: trivially copyable, standard layout, trivially destructible

한국어 번역:
- coroutine -> 코루틴
- template -> 템플릿  
- lambda -> 람다
- move semantics -> 이동 시맨틱스
- copy elision -> 복사 생략
- undefined behavior -> 미정의 동작 (UB)
- structured bindings -> 구조적 바인딩
- type erasure -> 타입 소거
- value category -> 값 카테고리

## 코드 분석 시 포함할 정보
1. 코드의 핵심 목적
2. 사용된 주요 C++ 기능 (C++11/14/17/20/23/26)
3. 주목할 만한 패턴이나 기법 (CRTP, SFINAE, tag dispatch 등)
```

### 번역/요약 프롬프트 (prompts/translate_summarize.txt)

```text
다음 C++ 관련 기사를 한국어로 번역하고 요약해주세요.

## 입력
- 제목: {title}
- 원문 요약: {description}
- 출처: {source}
{code_section}

## 출력 형식 (JSON)
{{
  "translated_title": "한국어 번역 제목",
  "summary": "2-3문장의 한국어 요약",
  "code_analysis": {{
    "purpose": "코드의 핵심 목적 (코드가 있는 경우)",
    "cpp_features": ["사용된 C++ 기능들"],
    "notable_patterns": "주목할 패턴 또는 null"
  }} 또는 null,
  "category_hint": "standard|performance|concurrency|modern|tools|safety|general",
  "cpp_version": "C++20" 또는 null
}}

## 주의사항
- 기술적 정확성을 유지하세요.
- C++ 키워드와 약어는 원문 유지 규칙을 따르세요.
- 코드가 있으면 코드 분석을 포함하고, 없으면 code_analysis는 null로 설정하세요.
- 요약에 코드의 핵심 내용도 반영하세요.
```

---

## 통합 처리 코드 예시

```python
# daily_digest.py 내 통합 처리 부분

import json
import yaml
import time
from llm_client import create_llm_client
from code_analyzer import CppCodeAnalyzer

class CPPDailyDigest:
    def __init__(self, webhook_url, config_path='config.yaml'):
        self.webhook_url = webhook_url
        self.config = self._load_config(config_path)
        self.feeds = self.config.get('feeds', {})
        self.llm_client = create_llm_client(self.config.get('llm', {}))
        
        # 코드 분석기 초기화
        code_config = self.config.get('code_analysis', {})
        self.code_analyzer = CppCodeAnalyzer(
            min_lines=code_config.get('min_code_lines', 3),
            max_length=code_config.get('max_code_length', 2000)
        ) if code_config.get('enabled', True) else None
        
        self.system_prompt = self._load_prompt('prompts/system.txt')
        self.translate_prompt_template = self._load_prompt('prompts/translate_summarize.txt')
    
    def _load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_prompt(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None
    
    def translate_and_summarize(self, article: dict) -> dict:
        """기사 번역, 요약, 코드 분석"""
        
        # LLM 비활성화 시 원문 반환
        if not self.llm_client:
            return {
                'translated_title': article['title'],
                'summary': article.get('description', '')[:200],
                'code_analysis': None,
                'category_hint': None,
                'cpp_version': None
            }
        
        # 코드 추출
        code_section = ""
        if self.code_analyzer:
            content = article.get('description', '') + article.get('content', '')
            code_blocks = self.code_analyzer.extract_code_blocks(content)
            if code_blocks:
                code_text = self.code_analyzer.get_code_summary_prompt(code_blocks)
                code_section = f"\n- 포함된 C++ 코드:\n{code_text}"
        
        # 프롬프트 생성
        prompt = self.translate_prompt_template.format(
            title=article['title'],
            description=article.get('description', ''),
            source=article['source'],
            code_section=code_section
        )
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt
            )
            
            # JSON 추출 (마크다운 코드블록 제거)
            json_str = response.strip()
            if json_str.startswith('```'):
                json_str = json_str.split('\n', 1)[1].rsplit('```', 1)[0]
            
            result = json.loads(json_str)
            return result
            
        except Exception as e:
            print(f"처리 실패: {e}")
            return {
                'translated_title': article['title'],
                'summary': article.get('description', '')[:200],
                'code_analysis': None,
                'category_hint': None,
                'cpp_version': None
            }
    
    def create_discord_embed(self, article: dict) -> dict:
        """Discord Embed 생성"""
        
        description = article['summary']
        
        # 코드 분석 결과 추가
        if article.get('code_analysis') and self.config.get('discord', {}).get('show_code_analysis', True):
            ca = article['code_analysis']
            if ca.get('purpose'):
                description += f"\n\n**코드**: {ca['purpose']}"
            if ca.get('cpp_features'):
                description += f"\n**사용 기능**: {', '.join(ca['cpp_features'])}"
        
        # C++ 버전 표시
        if article.get('cpp_version'):
            description += f"\n**표준**: {article['cpp_version']}"
        
        return {
            'title': article['translated_title'],
            'url': article['link'],
            'description': description,
            'footer': {'text': f"{article['source']}"}
        }
```

---

## LLM 클라이언트 구현 (llm_client.py)

```python
# llm_client.py
import os
from abc import ABC, abstractmethod
from typing import Optional

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, config: dict):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY') or os.environ.get('LLM_API_KEY'),
            base_url=config.get('base_url')
        )
        self.model = config.get('model', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 800)
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

class AnthropicClient(LLMClient):
    def __init__(self, config: dict):
        from anthropic import Anthropic
        self.client = Anthropic(
            api_key=os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('LLM_API_KEY')
        )
        self.model = config.get('model', 'claude-haiku-4-20250514')
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 800)
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

class GoogleClient(LLMClient):
    def __init__(self, config: dict):
        import google.generativeai as genai
        genai.configure(
            api_key=os.environ.get('GOOGLE_API_KEY') or os.environ.get('LLM_API_KEY')
        )
        self.model = genai.GenerativeModel(config.get('model', 'gemini-1.5-flash'))
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 800)
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self.model.generate_content(
            full_prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens
            }
        )
        return response.text

class OllamaClient(LLMClient):
    def __init__(self, config: dict):
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model', 'llama3')
        self.temperature = config.get('temperature', 0.3)
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        import requests
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt or "",
                "stream": False,
                "options": {"temperature": self.temperature}
            }
        )
        return response.json()['response']

def create_llm_client(config: dict) -> Optional[LLMClient]:
    if not config.get('enabled', False):
        return None
    
    provider = config.get('provider', 'openai').lower()
    
    clients = {
        'openai': OpenAIClient,
        'anthropic': AnthropicClient,
        'google': GoogleClient,
        'ollama': OllamaClient,
        'custom': OpenAIClient,
        'openrouter': OpenAIClient,
    }
    
    if provider == 'openrouter':
        config['base_url'] = 'https://openrouter.ai/api/v1'
    
    client_class = clients.get(provider)
    if not client_class:
        raise ValueError(f"지원하지 않는 프로바이더: {provider}")
    
    return client_class(config)
```

---

## Discord 출력 예시

코드 분석이 포함된 경우:

```
[C++20 코루틴을 활용한 비동기 파일 I/O 구현]
  C++ Stories / 09:30

  이 기사는 C++20 코루틴을 사용하여 비동기 파일 읽기/쓰기를 
  구현하는 방법을 설명합니다. io_uring과 코루틴을 결합하여 
  높은 성능을 달성합니다.

  **코드**: 비동기 파일 읽기를 위한 awaitable 타입 구현
  **사용 기능**: co_await, co_return, std::coroutine_handle
  **표준**: C++20
```

---

## 비용 고려사항

### 예상 API 비용 (기사 20개/일, 코드 분석 포함)

| 프로바이더 | 모델 | 일일 비용 | 월간 비용 |
|------------|------|-----------|-----------|
| OpenAI | gpt-4o-mini | ~$0.02 | ~$0.60 |
| OpenAI | gpt-4o | ~$0.15 | ~$4.50 |
| Anthropic | claude-haiku-4-20250514 | ~$0.03 | ~$0.90 |
| Anthropic | claude-sonnet-4-20250514 | ~$0.20 | ~$6.00 |
| Google | gemini-1.5-flash | 무료* | 무료* |
| Ollama | llama3 | 무료 | 무료 |

*코드 분석으로 인해 토큰 사용량 증가 (기존 대비 약 1.5배)

---

## C++ 전문 용어 한국어 표기 가이드

| 영어 용어 | 한국어 표기 | 비고 |
|-----------|-------------|------|
| constexpr | constexpr | 키워드 원문 유지 |
| consteval | consteval | 키워드 원문 유지 |
| constinit | constinit | 키워드 원문 유지 |
| concept | concept (컨셉) | 병기 가능 |
| ranges | ranges (레인지) | 병기 가능 |
| coroutine | 코루틴 | 한국어화 정착 |
| template | 템플릿 | 한국어화 정착 |
| lambda | 람다 | 한국어화 정착 |
| RAII | RAII | 약어 원문 유지 |
| SFINAE | SFINAE | 약어 원문 유지 |
| CTAD | CTAD | 약어 원문 유지 |
| CRTP | CRTP | 약어 원문 유지 |
| EBO | EBO | 약어 원문 유지 |
| NRVO / RVO | NRVO / RVO | 약어 원문 유지 |
| move semantics | 이동 시맨틱스 | |
| copy elision | 복사 생략 | |
| undefined behavior | 미정의 동작 (UB) | |
| structured bindings | 구조적 바인딩 | |
| fold expression | 폴드 표현식 | |
| parameter pack | 파라미터 팩 | |
| variadic template | 가변 템플릿 | |
| type trait | 타입 특성 | |
| type erasure | 타입 소거 | |
| value category | 값 카테고리 | |
| tag dispatch | 태그 디스패치 | |
| expression template | 표현식 템플릿 | |

---

## C++ 특화 콘텐츠 이해 가이드

### C++ 표준 버전 인식

| 표준 | 키워드 | 주요 기능 |
|------|--------|-----------|
| C++11 | c++11, c++0x | auto, lambda, move semantics, variadic templates, smart pointers |
| C++14 | c++14, c++1y | generic lambda, relaxed constexpr, variable templates |
| C++17 | c++17, c++1z | structured bindings, if constexpr, std::optional, std::variant, fold expressions |
| C++20 | c++20, c++2a | concepts, ranges, coroutines, modules, three-way comparison |
| C++23 | c++23, c++2b | std::expected, deducing this, std::print, std::mdspan |
| C++26 | c++26, c++2c | reflection, contracts, std::execution (예정) |

### 컴파일러별 키워드 인식

```
GCC     : gcc, g++, libstdc++, __attribute__, __builtin_*
Clang   : clang, clang++, libc++, __has_feature, __has_builtin
MSVC    : msvc, cl.exe, /std:c++, __declspec, _MSC_VER
Intel   : icc, icpc, icx, Intel C++
```

### 주요 C++ 토픽별 컨텍스트

**메모리 관리**
- unique_ptr, shared_ptr, weak_ptr: 스마트 포인터
- new, delete, allocator: 메모리 할당
- memory_resource, pmr: 다형성 메모리 리소스
- placement new: 배치 new

**동시성/병렬성**
- std::thread, std::jthread: 스레드 관리
- std::mutex, std::shared_mutex, std::lock_guard: 동기화
- std::atomic, memory_order: 원자적 연산
- co_await, co_yield, co_return: 코루틴
- std::latch, std::barrier: 동기화 프리미티브 (C++20)

**메타프로그래밍**
- template, typename, class: 템플릿 기초
- enable_if, void_t, decltype: SFINAE 기법
- if constexpr: 컴파일 타임 분기 (C++17)
- requires, concept: C++20 컨셉
- constexpr, consteval, constinit: 컴파일 타임 프로그래밍

**성능 최적화**
- inline, [[likely]], [[unlikely]]: 컴파일러 힌트
- SIMD, vectorization, AVX, SSE, NEON: 벡터화
- cache, prefetch, alignment, alignas: 캐시/메모리 정렬 최적화

**빌드 시스템 및 도구**
- CMake, Meson, Bazel: 빌드 시스템
- vcpkg, Conan: 패키지 매니저
- ASan, TSan, UBSan, MSan: Sanitizer

### 코드 분석 시 인식할 패턴

| 패턴 | 설명 | 인식 키워드 |
|------|------|-------------|
| CRTP | 정적 다형성 | `class Derived : public Base<Derived>` |
| SFINAE | 치환 실패 활용 | `enable_if`, `void_t`, `decltype` |
| Tag Dispatch | 태그 기반 오버로딩 | `std::true_type`, `std::false_type` |
| Type Erasure | 타입 소거 | `std::any`, `std::function`, virtual |
| Expression Templates | 지연 평가 | 연산자 오버로딩 체인 |
| Pimpl | 컴파일 방화벽 | `unique_ptr<Impl>` |
| Policy-based Design | 정책 기반 설계 | 다중 템플릿 파라미터 |

---

## 자동 카테고리 분류 규칙

| 카테고리 | 키워드 |
|----------|--------|
| 표준 및 제안 | c++26, c++23, c++20, standard, proposal, wg21, iso, p0000, n0000, committee |
| 성능 최적화 | performance, optimization, benchmark, simd, vectorization, cache, profiling, latency |
| 동시성 | coroutine, async, thread, concurrency, parallel, atomic, mutex, future, execution |
| 모던 C++ | ranges, concepts, modules, constexpr, consteval, template, lambda, auto, decltype |
| 도구 및 컴파일러 | compiler, cmake, clang, gcc, msvc, vcpkg, conan, sanitizer, debugger, linter |
| 안전성 | safety, memory safety, undefined behavior, ub, lifetime, dangling, bounds, hardening |
| 일반 | 위 카테고리에 해당하지 않는 모든 기사 |

---

## 핵심 기능 및 로직

### 중복 방지
- 기사 ID: MD5(소스명 + URL)
- 상태 파일: sent_articles.json (GitHub Actions 캐시로 유지)

### 시간 필터링
- 기준: 지난 24시간 이내 발행 (KST 기준)
- 내부 로직: UTC 사용, 표시용만 KST 변환

### 에러 처리
- 피드별 독립적 처리 (하나 실패해도 다른 피드는 계속 진행)
- LLM API 실패 시 원문으로 폴백
- 코드 추출 실패 시 코드 분석 없이 진행

## GitHub Actions 스케줄링

- 실행 시간: 매일 한국시간 오전 9시 (UTC 00:00)
- Cron 표현식: `0 0 * * *`
- 수동 실행: workflow_dispatch 지원
- 상태 캐싱: actions/cache로 sent_articles.json 유지

## 상태 관리

```json
{
  "date": "2025-01-01",
  "sent_today": ["article_id_1", "article_id_2"]
}
```

---

## 개발 시 주의사항

1. 피드 추가/삭제 시: config.yaml의 feeds 섹션만 수정
2. LLM 변경 시: config.yaml의 llm.provider와 llm.model 수정
3. API 키 관리: .env 파일 또는 GitHub Secrets 사용 (코드에 하드코딩 금지)
4. .env 파일: 반드시 .gitignore에 추가하여 저장소에 커밋되지 않도록 할 것
5. 새 환경 설정 시: .env.example 파일을 복사하여 .env 생성
6. 코드 분석 비활성화: config.yaml의 code_analysis.enabled를 false로 설정
7. 비용 모니터링: 코드 분석 시 토큰 사용량 증가 (약 1.5배)
8. 프롬프트 수정 시: prompts/ 디렉토리 내 파일 수정

## 확장 가능성

- [x] AI 기반 기사 요약 (LLM API 연동)
- [x] C++ 코드 분석 및 요약
- [ ] 중요도 점수 시스템 추가
- [ ] 이메일 알림 기능
- [ ] 주간/월간 요약 기능
- [ ] Slack/Teams 웹훅 지원

---

업데이트: 2025년 1월 기준
라이선스: MIT