#!/usr/bin/env python3
"""
daily_digest.py - C++ Daily Digest 메인 실행 파일

C++ RSS 제목과 설명을 한국어로 번역하여 Discord에 전송합니다.
"""

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup


def strip_html(html_content: str) -> str:
    """HTML 태그를 제거하고 순수 텍스트만 반환"""
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        # 스크립트, 스타일 태그 제거
        for tag in soup(["script", "style"]):
            tag.decompose()
        # 텍스트 추출
        text = soup.get_text(separator=" ", strip=True)
        # 여러 공백을 하나로
        text = " ".join(text.split())
        return text
    except Exception:
        # 파싱 실패 시 원문 반환
        return html_content


from llm_client import LLMError, create_llm_client

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


TRANSLATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "articles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "article_index": {"type": "integer"},
                    "translated_title": {"type": "string"},
                    "translated_description": {"type": "string"},
                },
                "required": ["article_index", "translated_title", "translated_description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["articles"],
    "additionalProperties": False,
}


def load_dotenv():
    """프로젝트 루트의 .env 파일에서 환경 변수 로드"""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 빈 줄, 주석 무시
            if not line or line.startswith("#"):
                continue
            # KEY=VALUE 파싱
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # 따옴표 제거
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                # 이미 설정된 환경 변수는 덮어쓰지 않음
                if key not in os.environ:
                    os.environ[key] = value


class CPPDailyDigest:
    """C++ Daily Digest 봇 메인 클래스"""

    def __init__(self, webhook_url: str, config_path: str = "config.yaml"):
        """
        Args:
            webhook_url: Discord 웹훅 URL
            config_path: 설정 파일 경로
        """
        self.webhook_url = webhook_url
        self.config = self._load_config(config_path)
        self.feeds = self.config.get("feeds", {})
        self.llm_client = create_llm_client(self.config.get("llm", {}))

        # 프롬프트 로드
        self.system_prompt = self._load_prompt("prompts/system.txt")
        self.batch_prompt_template = self._load_prompt("prompts/translate_summarize_batch.txt")

        # 배치 처리 설정
        llm_config = self.config.get("llm", {})
        self.batch_size = llm_config.get("batch_size", 10)

        # 상태 파일 경로
        self.state_file = Path(__file__).parent / "sent_articles.json"

        # Discord 설정
        discord_config = self.config.get("discord", {})
        self.embed_color = discord_config.get("embed_color", 0x0052CC)
        self.max_articles_per_category = discord_config.get("max_articles_per_category", 5)

        # 스케줄 설정
        schedule_config = self.config.get("schedule", {})
        self.lookback_hours = schedule_config.get("lookback_hours", 24)

        # 카테고리 분류 설정
        cat_config = self.config.get("categorization", {})
        self.categorization_enabled = cat_config.get("enabled", False)
        self.categories = cat_config.get("categories", {})

    def _load_config(self, path: str) -> Dict:
        """YAML 설정 파일 로드"""
        config_path = Path(__file__).parent / path
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"YAML 파싱 오류: {e}")
            return {}

    def _load_prompt(self, path: str) -> Optional[str]:
        """프롬프트 파일 로드"""
        prompt_path = Path(__file__).parent / path
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
            return None

    def _categorize_article(self, article: Dict) -> str:
        """기사를 카테고리로 분류

        Args:
            article: 원본 기사 정보

        Returns:
            카테고리 키 (예: 'standard', 'performance', 'general')
        """
        if not self.categorization_enabled:
            return "general"

        # 번역 결과에 의존하지 않아 전송 대상을 API 호출 전에 확정할 수 있다.
        search_text = " ".join([
            article.get("title", ""),
            strip_html(article.get("description", "") or ""),
        ]).lower()

        for cat_key, cat_info in self.categories.items():
            if cat_key == "general":
                continue
            keywords = cat_info.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in search_text:
                    return cat_key

        return "general"

    def _group_by_category(self, articles: List[Dict]) -> Dict[str, List[Dict]]:
        """기사들을 카테고리별로 그룹화

        Args:
            articles: (article, processed, embed) 튜플 리스트

        Returns:
            카테고리별로 그룹화된 딕셔너리
        """
        grouped = {}
        for item in articles:
            category = item.get("category", "general")
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(item)

        return grouped

    def _load_state(self) -> Dict:
        """이전 상태 로드"""
        try:
            if self.state_file.exists():
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    # 날짜가 다르면 초기화
                    today = datetime.now().strftime("%Y-%m-%d")
                    if state.get("date") != today:
                        return {"date": today, "sent_today": []}
                    return state
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"상태 파일 로드 실패: {e}")

        return {"date": datetime.now().strftime("%Y-%m-%d"), "sent_today": []}

    def _save_state(self, state: Dict):
        """상태 저장"""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error("상태 파일 저장 실패 (%s)", type(e).__name__)
            raise

    def _generate_article_id(self, source: str, url: str) -> str:
        """기사 고유 ID 생성"""
        return hashlib.md5(f"{source}:{url}".encode()).hexdigest()

    def _parse_published_date(self, entry: Any) -> Optional[datetime]:
        """RSS 엔트리에서 발행일 추출"""
        # 다양한 발행일 필드 시도
        date_fields = ["published_parsed", "updated_parsed", "created_parsed"]

        for field in date_fields:
            parsed = getattr(entry, field, None)
            if parsed:
                try:
                    return datetime(*parsed[:6], tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    continue

        return None

    def fetch_feeds(self) -> List[Dict]:
        """모든 RSS 피드에서 기사 수집"""
        articles = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        state = self._load_state()
        sent_ids = set(state.get("sent_today", []))

        for source_name, feed_url in self.feeds.items():
            logger.info(f"피드 수집 중: {source_name}")
            try:
                feed = feedparser.parse(feed_url)

                if feed.bozo and feed.bozo_exception:
                    logger.warning(f"피드 파싱 경고 ({source_name}): {feed.bozo_exception}")

                for entry in feed.entries[:10]:  # 피드당 최대 10개
                    try:
                        url = entry.get("link", "")
                        article_id = self._generate_article_id(source_name, url)

                        # 이미 전송된 기사 건너뛰기
                        if article_id in sent_ids:
                            continue

                        # 발행일 확인
                        pub_date = self._parse_published_date(entry)
                        if pub_date and pub_date < cutoff_time:
                            continue

                        # 콘텐츠 추출
                        description = ""
                        if hasattr(entry, "summary"):
                            description = entry.summary
                        elif hasattr(entry, "description"):
                            description = entry.description

                        content = ""
                        if hasattr(entry, "content") and entry.content:
                            content = entry.content[0].get("value", "")

                        articles.append(
                            {
                                "id": article_id,
                                "title": entry.get("title", "제목 없음"),
                                "link": url,
                                "source": source_name,
                                "description": description,
                                "content": content,
                                "published": pub_date.isoformat() if pub_date else None,
                            }
                        )

                    except Exception as e:
                        logger.warning(f"엔트리 처리 오류 ({source_name}): {e}")
                        continue

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"피드 수집 실패 ({source_name}): {e}")
                continue

        logger.info(f"총 {len(articles)}개 기사 수집됨")
        return articles

    def _prepare_article_for_batch(self, article: Dict) -> Dict:
        """본문을 수집하지 않고 RSS 설명의 앞 500자만 번역한다."""
        return {
            "title": article["title"],
            "description": strip_html(article.get("description", "") or "")[:500],
            "source": article["source"],
        }

    def _create_fallback_result(self, article: Dict) -> Dict:
        """사용자가 LLM을 명시적으로 끈 경우에만 원문을 사용한다."""
        return {
            "translated_title": article["title"],
            "translated_description": strip_html(article.get("description", "") or "")[:500],
        }

    def translate_and_summarize_batch(self, articles: List[Dict]) -> List[Dict]:
        """기존 메서드명을 유지하되 요약/분석 없이 제목과 RSS 설명만 번역한다."""
        if not articles:
            return []
        if not self.llm_client:
            return [self._create_fallback_result(article) for article in articles]
        if not self.batch_prompt_template:
            raise LLMError("번역 프롬프트를 찾을 수 없습니다.")

        inputs = [dict(self._prepare_article_for_batch(article), index=index)
                  for index, article in enumerate(articles)]
        prompt = self.batch_prompt_template.format(
            articles_json=json.dumps(inputs, ensure_ascii=False)
        )
        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            response_schema=TRANSLATION_RESPONSE_SCHEMA,
        )
        if not isinstance(response, str):
            raise LLMError("번역 응답은 문자열이어야 합니다.")
        text = response.strip()
        # 다른 제공자의 기존 Markdown JSON 응답도 같은 검증을 거친다.
        if text.startswith("```") and text.endswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            raise LLMError("번역 응답이 올바른 JSON이 아닙니다.") from None
        if not isinstance(payload, dict) or set(payload) != {"articles"}:
            raise LLMError("번역 응답에 articles 객체가 필요합니다.")
        results = payload["articles"]
        if not isinstance(results, list) or len(results) != len(articles):
            raise LLMError("번역 결과 개수가 입력 기사 수와 다릅니다.")

        by_index = {}
        for result in results:
            if not isinstance(result, dict) or set(result) != {
                "article_index", "translated_title", "translated_description"
            }:
                raise LLMError("번역 결과 필드가 잘못되었습니다.")
            index = result["article_index"]
            if type(index) is not int or not 0 <= index < len(articles) or index in by_index:
                raise LLMError("기사 번호가 중복되거나 범위를 벗어났습니다.")
            title = result["translated_title"]
            description = result["translated_description"]
            if not isinstance(title, str) or not title.strip():
                raise LLMError("번역 제목이 비어 있거나 문자열이 아닙니다.")
            if not isinstance(description, str):
                raise LLMError("번역 설명은 문자열이어야 합니다.")
            if inputs[index]["description"].strip() and not description.strip():
                raise LLMError("원문 설명의 번역이 누락되었습니다.")
            if not inputs[index]["description"].strip() and description.strip():
                raise LLMError("빈 원문 설명에 번역 내용이 추가되었습니다.")
            by_index[index] = result
        return [by_index[index] for index in range(len(articles))]

    def translate_and_summarize(self, article: Dict) -> Dict:
        """단일 기사도 같은 번역 계약과 검증 경로를 사용한다."""
        return self.translate_and_summarize_batch([article])[0]

    def create_discord_embed(self, article: Dict, processed: Dict) -> Dict:
        """Discord Embed 생성"""
        description = processed["translated_description"]

        # 설명이 너무 길면 자르기
        if len(description) > 4000:
            description = description[:3997] + "..."

        return {
            "title": processed.get("translated_title", article["title"])[:256],
            "url": article["link"],
            "description": description,
            "color": self.embed_color,
            "footer": {"text": article["source"]},
            "timestamp": article.get("published") or datetime.now(timezone.utc).isoformat(),
        }

    def _send_webhook(self, payload: Dict) -> bool:
        """Discord 웹훅으로 페이로드 전송"""
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 429:  # Rate limited
                retry_after = response.json().get("retry_after", 5)
                logger.warning(f"Rate limited. {retry_after}초 후 재시도...")
                time.sleep(retry_after)
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )

            response.raise_for_status()
            return True

        except requests.RequestException as e:
            logger.error("Discord 전송 실패 (%s)", type(e).__name__)
            return False

    def _batch_embeds(self, embeds: List[Dict]):
        """임베드 개수뿐 아니라 메시지 전체 텍스트의 6000자 상한을 지킨다."""
        batch = []
        total = 0
        for embed in embeds:
            size = len(embed.get("title", "")) + len(embed.get("description", ""))
            size += len(embed.get("footer", {}).get("text", ""))
            size += len(embed.get("author", {}).get("name", ""))
            size += sum(len(field.get("name", "")) + len(field.get("value", ""))
                        for field in embed.get("fields", []))
            if size > 6000:
                raise ValueError("단일 임베드가 Discord 텍스트 상한을 초과했습니다.")
            if batch and (len(batch) == 10 or total + size > 6000):
                yield batch
                batch = []
                total = 0
            batch.append(embed)
            total += size
        if batch:
            yield batch

    def send_to_discord(self, embeds: List[Dict]) -> bool:
        """Discord로 메시지 전송 (카테고리 미분류)"""
        if not embeds:
            logger.info("전송할 기사가 없습니다.")
            return True

        batches = list(self._batch_embeds(embeds))
        for i, batch in enumerate(batches):

            # 첫 번째 배치에만 헤더 추가
            content = None
            if i == 0:
                today = datetime.now().strftime("%Y년 %m월 %d일")
                content = f"📰 **C++ Daily Digest** - {today}"

            payload = {"content": content, "embeds": batch}

            if not self._send_webhook(payload):
                return False

            logger.info(f"Discord 전송 성공: {len(batch)}개 기사")

            # Rate limiting
            if i + 1 < len(batches):
                time.sleep(1)

        return True

    def send_categorized_to_discord(self, categorized_articles: List[Dict]) -> bool:
        """카테고리별로 그룹화하여 Discord로 전송

        Args:
            categorized_articles: category 키가 포함된 기사 딕셔너리 리스트

        Returns:
            성공 여부
        """
        if not categorized_articles:
            logger.info("전송할 기사가 없습니다.")
            return True

        # 카테고리별로 그룹화
        grouped = self._group_by_category(categorized_articles)

        # 헤더 전송
        today = datetime.now().strftime("%Y년 %m월 %d일")
        header_payload = {"content": f"📰 **C++ Daily Digest** - {today}"}
        if not self._send_webhook(header_payload):
            return False
        time.sleep(0.5)

        # 카테고리 순서 정의
        category_order = ["standard", "modern", "performance", "concurrency", "tools", "safety", "general"]
        category_order.extend(key for key in grouped if key not in category_order)

        total_sent = 0
        for cat_key in category_order:
            if cat_key not in grouped:
                continue

            articles = grouped[cat_key]
            if not articles:
                continue

            # 카테고리당 최대 기사 수 제한
            articles = articles[:self.max_articles_per_category]

            # 카테고리 이름 가져오기
            cat_info = self.categories.get(cat_key, {})
            cat_name = cat_info.get("name", f"📰 {cat_key}")

            # 카테고리 헤더와 기사들 전송
            embeds = [article["embed"] for article in articles]

            for i, batch in enumerate(self._batch_embeds(embeds)):

                # 첫 번째 배치에만 카테고리 헤더 추가
                content = f"\n**{cat_name}** ({len(articles)}개)" if i == 0 else None

                payload = {"content": content, "embeds": batch}

                if not self._send_webhook(payload):
                    return False

                total_sent += len(batch)
                time.sleep(1)  # Rate limiting

        logger.info(f"Discord 전송 성공: {total_sent}개 기사 ({len(grouped)}개 카테고리)")
        return True

    def _select_articles(self, articles: List[Dict]) -> List[Dict]:
        """게시 상한을 먼저 적용하여 버릴 기사를 유료 번역하지 않는다."""
        if not self.categorization_enabled:
            return articles[:self.max_articles_per_category * 7]
        selected = []
        counts = {}
        for article in articles:
            category = self._categorize_article(article)
            if counts.get(category, 0) < self.max_articles_per_category:
                selected.append(article)
                counts[category] = counts.get(category, 0) + 1
        return selected

    def run(self):
        """번역을 모두 검증한 후 게시한다. 실패는 호출자에게 전달한다."""
        logger.info("C++ Daily Digest 시작")
        articles = self._select_articles(self.fetch_feeds())
        if not articles:
            logger.info("새로운 기사가 없습니다.")
            return

        state = self._load_state()
        processed_articles = []
        rate_limit_delay = self.config.get("llm", {}).get("rate_limit_delay", 1)
        for start in range(0, len(articles), self.batch_size):
            batch = articles[start:start + self.batch_size]
            logger.info("번역 처리 중: %s-%s/%s", start + 1, start + len(batch), len(articles))
            results = self.translate_and_summarize_batch(batch)
            for article, result in zip(batch, results):
                processed_articles.append({
                    "article": article,
                    "processed": result,
                    "embed": self.create_discord_embed(article, result),
                    "category": self._categorize_article(article),
                })
            if start + len(batch) < len(articles) and self.llm_client:
                time.sleep(rate_limit_delay)

        if self.categorization_enabled:
            success = self.send_categorized_to_discord(processed_articles)
        else:
            success = self.send_to_discord([item["embed"] for item in processed_articles])
        if not success:
            raise RuntimeError("Discord 전송 실패; 완료 상태를 저장하지 않았습니다.")

        sent_ids = set(state.get("sent_today", []))
        sent_ids.update(item["article"]["id"] for item in processed_articles)
        state["sent_today"] = sorted(sent_ids)
        self._save_state(state)
        logger.info("총 %s개 기사 전송 완료", len(processed_articles))


def main():
    """메인 함수"""
    load_dotenv()
    # 환경 변수 확인
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.error("DISCORD_WEBHOOK_URL 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    # 봇 실행
    try:
        bot = CPPDailyDigest(webhook_url=webhook_url)
        bot.run()
    except LLMError as error:
        # 직접 작성한 비밀값 없는 진단만 출력한다. 외부 오류는 아래에서 숨긴다.
        logger.error("실행 실패: %s", error)
        sys.exit(1)
    except Exception as error:
        # SDK 오류 본문과 HTTP URL에는 API 키/웹훅 토큰이 포함될 수 있다.
        logger.error("실행 실패 (%s). 설정, API 키와 앞선 상태 로그를 확인하세요.", type(error).__name__)
        sys.exit(1)


if __name__ == "__main__":
    main()
