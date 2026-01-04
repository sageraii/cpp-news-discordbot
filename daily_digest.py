#!/usr/bin/env python3
"""
daily_digest.py - C++ Daily Digest 메인 실행 파일

C++ 관련 RSS 피드를 수집하여 Discord 채널로 일일 요약을 전송합니다.
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

from code_analyzer import CppCodeAnalyzer


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


from llm_client import create_llm_client

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


# 앱 시작 시 로드
load_dotenv()


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

        # 코드 분석기 초기화
        code_config = self.config.get("code_analysis", {})
        self.code_analyzer = (
            CppCodeAnalyzer(
                min_lines=code_config.get("min_code_lines", 3),
                max_length=code_config.get("max_code_length", 2000),
            )
            if code_config.get("enabled", True)
            else None
        )

        # 프롬프트 로드
        self.system_prompt = self._load_prompt("prompts/system.txt")
        self.translate_prompt_template = self._load_prompt("prompts/translate_summarize.txt")
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

    def _categorize_article(self, article: Dict, processed: Dict) -> str:
        """기사를 카테고리로 분류

        Args:
            article: 원본 기사 정보
            processed: LLM 처리된 기사 정보

        Returns:
            카테고리 키 (예: 'standard', 'performance', 'general')
        """
        if not self.categorization_enabled:
            return "general"

        # LLM이 제안한 카테고리 힌트 확인
        category_hint = processed.get("category_hint", "")
        if category_hint and category_hint in self.categories:
            return category_hint

        # 키워드 기반 분류
        search_text = " ".join([
            article.get("title", ""),
            article.get("description", ""),
            processed.get("translated_title", ""),
            processed.get("summary", ""),
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
            logger.error(f"상태 파일 저장 실패: {e}")

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
        """배치 처리를 위해 기사 정보를 준비"""
        # 코드 추출
        code_section = ""
        if self.code_analyzer:
            content = (article.get("description", "") or "") + (article.get("content", "") or "")
            code_blocks = self.code_analyzer.extract_code_blocks(content)
            if code_blocks:
                code_text = self.code_analyzer.get_code_summary_prompt(code_blocks)
                if code_text:
                    code_section = code_text

        return {
            "title": article["title"],
            "description": strip_html(article.get("description", "") or "")[:500],  # 토큰 절약
            "source": article["source"],
            "code_section": code_section[:1000] if code_section else "",  # 토큰 절약
        }

    def _create_fallback_result(self, article: Dict) -> Dict:
        """LLM 실패 시 폴백 결과 생성"""
        clean_desc = strip_html(article.get("description", "") or "")
        return {
            "translated_title": article["title"],
            "summary": clean_desc[:300],
            "code_analysis": None,
            "category_hint": None,
            "cpp_version": None,
        }

    def translate_and_summarize_batch(self, articles: List[Dict]) -> List[Dict]:
        """여러 기사를 배치로 번역 및 요약 (API 호출 최적화)

        Args:
            articles: 기사 딕셔너리 리스트

        Returns:
            처리된 결과 리스트 (입력과 동일한 순서)
        """
        if not articles:
            return []

        # LLM 비활성화 시 폴백
        if not self.llm_client:
            return [self._create_fallback_result(article) for article in articles]

        # 배치 프롬프트가 없으면 개별 처리로 폴백
        if not self.batch_prompt_template:
            logger.warning("배치 프롬프트 템플릿이 없어 개별 처리로 전환합니다.")
            return [self.translate_and_summarize(article) for article in articles]

        # 배치용 기사 정보 준비
        articles_data = []
        for idx, article in enumerate(articles):
            prepared = self._prepare_article_for_batch(article)
            prepared["index"] = idx
            articles_data.append(prepared)

        # JSON으로 직렬화
        articles_json = json.dumps(articles_data, ensure_ascii=False, indent=2)

        # 프롬프트 생성
        prompt = self.batch_prompt_template.format(articles_json=articles_json)

        try:
            logger.info(f"배치 처리 중: {len(articles)}개 기사")
            response = self.llm_client.generate(prompt=prompt, system_prompt=self.system_prompt)

            # JSON 추출 (마크다운 코드블록 제거)
            json_str = response.strip()
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:])
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                json_str = json_str.strip()

            results = json.loads(json_str)

            # 결과가 리스트가 아닌 경우 처리
            if not isinstance(results, list):
                logger.warning("배치 결과가 리스트가 아닙니다. 폴백 처리합니다.")
                return [self._create_fallback_result(article) for article in articles]

            # article_index 기준으로 정렬하여 원래 순서 보장
            sorted_results = [None] * len(articles)
            for result in results:
                idx = result.get("article_index", -1)
                if 0 <= idx < len(articles):
                    sorted_results[idx] = result

            # 누락된 결과는 폴백으로 채움
            for idx, result in enumerate(sorted_results):
                if result is None:
                    logger.warning(f"기사 {idx} 결과 누락, 폴백 처리")
                    sorted_results[idx] = self._create_fallback_result(articles[idx])

            logger.info(f"배치 처리 완료: {len(articles)}개 기사")
            return sorted_results

        except json.JSONDecodeError as e:
            logger.warning(f"배치 JSON 파싱 실패: {e}")
        except Exception as e:
            logger.warning(f"배치 처리 실패: {e}")

        # 배치 실패 시 폴백
        return [self._create_fallback_result(article) for article in articles]

    def translate_and_summarize(self, article: Dict) -> Dict:
        """단일 기사 번역, 요약, 코드 분석 (하위 호환성 유지)"""

        # LLM 비활성화 시 원문 반환 (HTML 태그 제거)
        if not self.llm_client:
            return self._create_fallback_result(article)

        # 코드 추출
        code_section = ""
        if self.code_analyzer:
            content = (article.get("description", "") or "") + (article.get("content", "") or "")
            code_blocks = self.code_analyzer.extract_code_blocks(content)
            if code_blocks:
                code_text = self.code_analyzer.get_code_summary_prompt(code_blocks)
                if code_text:
                    code_section = f"\n- 포함된 C++ 코드:\n{code_text}"

        # 프롬프트 생성
        if not self.translate_prompt_template:
            return self._create_fallback_result(article)

        prompt = self.translate_prompt_template.format(
            title=article["title"],
            description=article.get("description", "") or "",
            source=article["source"],
            code_section=code_section,
        )

        try:
            response = self.llm_client.generate(prompt=prompt, system_prompt=self.system_prompt)

            # JSON 추출 (마크다운 코드블록 제거)
            json_str = response.strip()
            if json_str.startswith("```"):
                # ```json 또는 ``` 제거
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:])
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                json_str = json_str.strip()

            result = json.loads(json_str)
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {e}")
        except Exception as e:
            logger.warning(f"번역/요약 실패: {e}")

        # 폴백 (HTML 태그 제거)
        return self._create_fallback_result(article)

    def create_discord_embed(self, article: Dict, processed: Dict) -> Dict:
        """Discord Embed 생성"""
        description = processed.get("summary", "") or ""

        # 코드 분석 결과 추가
        discord_config = self.config.get("discord", {})
        if processed.get("code_analysis") and discord_config.get("show_code_analysis", True):
            ca = processed["code_analysis"]
            if ca.get("purpose"):
                description += f"\n\n**코드**: {ca['purpose']}"
            if ca.get("cpp_features"):
                features = ca["cpp_features"]
                if isinstance(features, list):
                    description += f"\n**사용 기능**: {', '.join(features)}"

        # C++ 버전 표시
        if processed.get("cpp_version"):
            description += f"\n**표준**: {processed['cpp_version']}"

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
            logger.error(f"Discord 전송 실패: {e}")
            return False

    def send_to_discord(self, embeds: List[Dict]) -> bool:
        """Discord로 메시지 전송 (카테고리 미분류)"""
        if not embeds:
            logger.info("전송할 기사가 없습니다.")
            return True

        # Discord는 한 번에 최대 10개 임베드 허용
        for i in range(0, len(embeds), 10):
            batch = embeds[i : i + 10]

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
            if i + 10 < len(embeds):
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

            # Discord는 한 번에 최대 10개 임베드 허용
            for i in range(0, len(embeds), 10):
                batch = embeds[i : i + 10]

                # 첫 번째 배치에만 카테고리 헤더 추가
                content = f"\n**{cat_name}** ({len(articles)}개)" if i == 0 else None

                payload = {"content": content, "embeds": batch}

                if not self._send_webhook(payload):
                    return False

                total_sent += len(batch)
                time.sleep(1)  # Rate limiting

        logger.info(f"Discord 전송 성공: {total_sent}개 기사 ({len(grouped)}개 카테고리)")
        return True

    def run(self):
        """메인 실행 로직"""
        logger.info("C++ Daily Digest 시작")

        # 1. 피드 수집
        articles = self.fetch_feeds()
        if not articles:
            logger.info("새로운 기사가 없습니다.")
            return

        # 2. 번역 및 요약 (배치 처리)
        processed_articles = []
        state = self._load_state()
        sent_ids = set(state.get("sent_today", []))

        rate_limit_delay = self.config.get("llm", {}).get("rate_limit_delay", 1)

        # 배치 단위로 처리 (API 호출 최적화)
        for batch_start in range(0, len(articles), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(articles))
            batch_articles = articles[batch_start:batch_end]

            logger.info(f"배치 처리 중: {batch_start + 1}-{batch_end}/{len(articles)}")

            try:
                # 배치로 번역/요약 처리 (API 1회 호출)
                batch_results = self.translate_and_summarize_batch(batch_articles)

                # 결과 처리
                for article, processed in zip(batch_articles, batch_results):
                    embed = self.create_discord_embed(article, processed)
                    category = self._categorize_article(article, processed)

                    processed_articles.append({
                        "article": article,
                        "processed": processed,
                        "embed": embed,
                        "category": category,
                    })
                    sent_ids.add(article["id"])

                # 다음 배치 전 rate limiting (Gemini 무료 플랜: RPM=10)
                if batch_end < len(articles) and self.llm_client:
                    logger.info(f"Rate limit 대기: {rate_limit_delay}초")
                    time.sleep(rate_limit_delay)

            except Exception as e:
                logger.error(f"배치 처리 실패: {e}")
                # 배치 실패 시 해당 배치의 기사들은 폴백 처리
                for article in batch_articles:
                    processed = self._create_fallback_result(article)
                    embed = self.create_discord_embed(article, processed)
                    category = self._categorize_article(article, processed)
                    processed_articles.append({
                        "article": article,
                        "processed": processed,
                        "embed": embed,
                        "category": category,
                    })
                    sent_ids.add(article["id"])

        # 3. Discord 전송
        if processed_articles:
            if self.categorization_enabled:
                # 카테고리별로 그룹화하여 전송
                success = self.send_categorized_to_discord(processed_articles)
            else:
                # 기존 방식 (카테고리 미분류)
                embeds = [item["embed"] for item in processed_articles]
                embeds = embeds[: self.max_articles_per_category * 7]
                success = self.send_to_discord(embeds)

            if success:
                # 상태 업데이트
                state["sent_today"] = list(sent_ids)
                self._save_state(state)
                logger.info(f"총 {len(processed_articles)}개 기사 전송 완료")
            else:
                logger.error("Discord 전송 실패")
        else:
            logger.info("처리된 기사가 없습니다.")

        logger.info("C++ Daily Digest 완료")


def main():
    """메인 함수"""
    # 환경 변수 확인
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.error("DISCORD_WEBHOOK_URL 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    # 봇 실행
    bot = CPPDailyDigest(webhook_url=webhook_url)
    bot.run()


if __name__ == "__main__":
    main()
