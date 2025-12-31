"""
code_analyzer.py - C++ 코드 추출 및 분석 모듈

RSS 피드 콘텐츠에서 C++ 코드를 추출하고 분석하는 기능을 제공합니다.
"""

import re
import logging
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class CppCodeAnalyzer:
    """C++ 코드 추출 및 분석 클래스"""

    # C++ 코드 식별 패턴
    CPP_INDICATORS = [
        r"#include\s*<",  # 헤더 인클루드
        r"std::",  # std 네임스페이스
        r"template\s*<",  # 템플릿
        r"constexpr\s+",  # constexpr
        r"auto\s+\w+\s*=",  # auto 변수
        r"->\s*\w+",  # trailing return type
        r"nullptr",  # nullptr
        r"::",  # 스코프 연산자
        r"\bclass\s+\w+",  # 클래스 선언
        r"\bstruct\s+\w+",  # 구조체 선언
        r"\bnamespace\s+\w+",  # 네임스페이스
        r"unique_ptr|shared_ptr",  # 스마트 포인터
        r"vector<|map<|string>",  # STL 컨테이너
        r"co_await|co_yield",  # 코루틴
        r"concept\s+\w+",  # 컨셉
        r"requires\s*\(",  # requires 절
        r"consteval\s+",  # consteval
        r"constinit\s+",  # constinit
        r"std::ranges::",  # ranges
        r"std::views::",  # views
        r"\[\[.*\]\]",  # 속성
    ]

    def __init__(self, min_lines: int = 3, max_length: int = 2000):
        """
        Args:
            min_lines: 코드로 인식할 최소 줄 수
            max_length: 최대 코드 길이 (토큰 절약)
        """
        self.min_lines = min_lines
        self.max_length = max_length

    def extract_code_blocks(self, html_content: str) -> List[Dict]:
        """HTML/Markdown에서 코드 블록 추출

        Args:
            html_content: HTML 또는 Markdown 콘텐츠

        Returns:
            추출된 코드 블록 목록 [{'code': str, 'source': str}, ...]
        """
        if not html_content:
            return []

        code_blocks = []

        # 1. HTML <pre><code> 태그에서 추출
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # <pre> 태그 처리
            for pre in soup.find_all("pre"):
                code = pre.get_text(strip=True)
                if self._is_cpp_code(code):
                    code_blocks.append({"code": code[: self.max_length], "source": "html_pre"})

            # <code> 태그 처리 (pre 내부가 아닌 경우만)
            for code_tag in soup.find_all("code"):
                if code_tag.parent and code_tag.parent.name != "pre":
                    code = code_tag.get_text(strip=True)
                    if len(code.split("\n")) >= self.min_lines and self._is_cpp_code(code):
                        code_blocks.append({"code": code[: self.max_length], "source": "html_code"})
        except Exception as e:
            logger.warning(f"HTML 파싱 오류: {e}")

        # 2. Markdown 코드 블록 추출
        md_patterns = [
            r"```(?:cpp|c\+\+|cxx|c)\s*\n(.*?)```",  # 명시적 언어 태그
            r"```\s*\n(.*?)```",  # 언어 태그 없는 코드 블록
        ]

        for pattern in md_patterns:
            for match in re.finditer(pattern, html_content, re.DOTALL | re.IGNORECASE):
                code = match.group(1).strip()
                if self._is_cpp_code(code):
                    code_blocks.append({"code": code[: self.max_length], "source": "markdown"})

        # 중복 제거 (코드 내용 기준)
        seen = set()
        unique_blocks = []
        for block in code_blocks:
            code_hash = hash(block["code"][:100])  # 앞 100자로 중복 체크
            if code_hash not in seen:
                seen.add(code_hash)
                unique_blocks.append(block)

        logger.debug(f"추출된 코드 블록: {len(unique_blocks)}개")
        return unique_blocks

    def _is_cpp_code(self, code: str) -> bool:
        """C++ 코드인지 휴리스틱 판단

        Args:
            code: 코드 문자열

        Returns:
            C++ 코드 여부
        """
        if not code:
            return False

        lines = code.split("\n")
        if len(lines) < self.min_lines:
            return False

        # C++ 특징 패턴 매칭
        matches = sum(1 for pattern in self.CPP_INDICATORS if re.search(pattern, code))

        # 2개 이상 매칭 시 C++ 코드로 판단
        return matches >= 2

    def get_code_summary_prompt(self, code_blocks: List[Dict]) -> Optional[str]:
        """코드 분석을 위한 프롬프트 생성

        Args:
            code_blocks: 추출된 코드 블록 목록

        Returns:
            코드 분석 프롬프트 또는 None
        """
        if not code_blocks:
            return None

        # 가장 긴 코드 블록 선택 (최대 2개)
        sorted_blocks = sorted(code_blocks, key=lambda x: len(x["code"]), reverse=True)[:2]

        code_text = "\n\n---\n\n".join(f"```cpp\n{block['code']}\n```" for block in sorted_blocks)

        return code_text

    def detect_cpp_version(self, code: str) -> Optional[str]:
        """코드에서 C++ 버전 감지

        Args:
            code: 코드 문자열

        Returns:
            감지된 C++ 버전 또는 None
        """
        if not code:
            return None

        # C++ 버전별 특징 패턴
        version_patterns = {
            "C++26": [r"std::execution", r"reflection", r"contracts"],
            "C++23": [r"std::expected", r"std::mdspan", r"std::print", r"deducing\s+this"],
            "C++20": [
                r"concept\s+\w+",
                r"requires\s*\(",
                r"co_await|co_yield|co_return",
                r"std::ranges::",
                r"std::views::",
                r"<=>",
                r"consteval",
                r"constinit",
                r"\[\[likely\]\]|\[\[unlikely\]\]",
            ],
            "C++17": [
                r"if\s+constexpr",
                r"std::optional",
                r"std::variant",
                r"std::any",
                r"std::string_view",
                r"structured\s+bindings",
                r"\[\[nodiscard\]\]",
                r"\[\[maybe_unused\]\]",
                r"std::filesystem",
            ],
            "C++14": [r"generic\s+lambda", r"variable\s+template", r"std::make_unique"],
            "C++11": [
                r"auto\s+\w+\s*=",
                r"nullptr",
                r"std::unique_ptr|std::shared_ptr",
                r"std::thread",
                r"std::mutex",
                r"std::async",
                r"range-based\s+for",
                r"lambda",
            ],
        }

        for version, patterns in version_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return version

        return None

    def categorize_code(self, code: str) -> Optional[str]:
        """코드 카테고리 분류

        Args:
            code: 코드 문자열

        Returns:
            카테고리 힌트 또는 None
        """
        if not code:
            return None

        category_patterns = {
            "concurrency": [r"std::thread", r"std::mutex", r"std::async", r"co_await", r"std::atomic"],
            "performance": [r"SIMD", r"vectoriz", r"cache", r"benchmark", r"inline", r"constexpr"],
            "modern": [r"ranges", r"concepts", r"modules", r"consteval", r"constinit"],
            "standard": [r"proposal", r"P\d{4}", r"N\d{4}", r"wg21"],
            "tools": [r"cmake", r"clang", r"gcc", r"msvc", r"sanitizer"],
            "safety": [r"memory\s*safety", r"undefined\s*behavior", r"lifetime", r"bounds"],
        }

        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return category

        return "general"
