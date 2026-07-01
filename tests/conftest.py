"""
tests/conftest.py

단위 테스트 환경 설정.
모듈 import 전에 더미 API 키를 설정해 ChatOpenAI 생성자 검증을 통과시킨다.
실제 API 호출은 각 테스트에서 mock 처리한다.
"""

import os

# 테스트 실행 시 OPENAI_API_KEY 가 없으면 더미 값으로 채운다.
# ChatOpenAI 생성자는 키 형식을 검증하지 않고 존재 여부만 확인한다.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-unit-tests")
