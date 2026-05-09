"""
app/services/pdf_parser.py

차량 매뉴얼 PDF를 LangChain Document 리스트로 변환한다.

흐름:
    PDF → PyMuPDFLoader(텍스트 추출) → _clean_text(노이즈 제거) → _detect_section(섹션 감지) → list[Document]

메타데이터 4종 (계획서 기준):
    source, page_num, section, content_type
"""

from __future__ import annotations  # Python 3.9 이하에서도 타입 힌트가 동작하도록 함

import re           # 정규 표현식 모듈 (노이즈 패턴 감지 및 제거에 사용)
from pathlib import Path  # 파일 경로를 문자열이 아닌 객체로 다루기 위한 모듈

import fitz  # PyMuPDF 라이브러리. PDF 내부 폰트 크기 등 상세 구조 정보 추출에 사용
from langchain_core.documents import Document  # LangChain의 기본 문서 단위. page_content + metadata 구조
from langchain_community.document_loaders import PyMuPDFLoader  # LangChain 제공 PDF 로더. 페이지 단위로 텍스트 추출


def _clean_text(raw: str) -> str:
    """
    PDF에서 추출한 원본 텍스트의 노이즈를 제거하고 공백을 정규화한다.
    줄 단위로 순회하며 불필요한 패턴을 필터링한다.
    """
    cleaned = []  # 정제된 줄을 담을 리스트

    for line in raw.splitlines():  # 텍스트를 줄 단위로 분리하여 순회

        # 목차의 점선 제거 (예: "차량 외부 .............. 7")
        # 점(.)이 5개 이상 연속되면 목차 점선으로 판단(문자열 어디서든 패턴 찾기)
        if re.search(r'\.{5,}', line):
            continue

        # 단독 페이지 번호 제거 (예: "123")
        # 줄 전체가 숫자로만 이루어진 경우 페이지 번호로 판단
        if re.match(r'^\s*\d+\s*$', line):
            continue

        # 구분선 제거 (예: "-----", "=====", "_____")
        # 동일한 기호가 5개 이상 반복되는 줄을 구분선으로 판단
        if re.match(r'^\s*[-_=]{5,}\s*$', line):
            continue

        # 이미지 파일명 패턴 제거 (예: 5페이진가?의 "OutsideVehicleFrontOverview")
        # 대문자로 시작하는 영문+숫자 조합의 단독 단어를 이미지 참조로 판단
        if re.match(r'^[A-Z0-9][A-Za-z0-9_]+$', line.strip()):
            continue

        # 번호만 있는 라인 제거 (예: "1:", "2:", "3:")
        # 이미지 설명의 번호 목록으로 내용 없이 번호만 있는 경우
        if re.match(r'^\s*\d+:\s*$', line):
            continue
        
        # 깨진 문자 전부 필터링하려 했는데, 그러면 ℃ ㎞ → 이런것들도 같이 날아가서 일단 수동필터링함
        # PyMuPDF가 PDF 내부 폰트를 못 읽으면 해당 문자를 다른 유니코드로 매핑하기 때문에 일단 최대한 필터링
        # 특히 표가 많이 깨짐. 표를 OCR해서 인식하는 방법도 있지만 표 내부 데이터는 지금 정상적으로 읽히기도 하고 복잡도가 너무 올라가서 제외.
        # 깨진 문자의 경우 RAG가 그냥 자동으로 뛰어넘어서 인식하기 때문에 큰 영향 없다.
        line = re.sub(r'\ufffd+', '', line)                      # 유니코드 대체 문자
        line = re.sub(r'\(cid:\d+\)', '', line)                  # PDF 인코딩 오류
        line = re.sub(r'[□■▣]{2,}', '', line)                   # 연속 박스 문자
        line = re.sub(r'[Ƅ-ƿ]+', '', line)                      # 라틴 확장 깨진 문자
        line = re.sub(r'[\u0a80-\u0aff]+', '', line)             # 구자라트 깨진 문자
        line = re.sub(r'[\u0980-\u09ff]+', '', line)             # 벵골 깨진 문자
        line = re.sub(r'[\u0600-\u06ff]+', '', line)             # 아랍 깨진 문자
        line = re.sub(r'[\u0400-\u04ff]+', '', line)             # 키릴 깨진 문자
        line = re.sub(r'[\u0e00-\u0e7f]+', '', line)             # 태국 깨진 문자

        # 정제 후 내용이 남아있는 줄만 추가
        if line.strip():
            cleaned.append(line)

    text = "\n".join(cleaned)  # 정제된 줄들을 다시 하나의 텍스트로 합치기
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3개 이상 연속 줄바꿈을 2개로 축소
    text = re.sub(r" {2,}", " ", text)       # 연속된 공백을 1개로 축소
    return text.strip()  # 텍스트 양 끝의 공백 제거 후 반환


def _estimate_body_font_size(fitz_doc: fitz.Document) -> float:
    """
    문서 전체를 순회하여 가장 많이 사용된 폰트 크기를 본문 크기로 추정한다.
    섹션 헤더는 본문보다 폰트가 크므로, 이 값을 기준으로 헤더를 감지한다.
    """
    size_counts: dict[float, int] = {}  # {폰트 크기: 등장 횟수} 딕셔너리

    for page in fitz_doc:  # 문서의 모든 페이지 순회
        # pdf의 계층 구조를 순회하는것. span내부에 size(폰트 크기)가 있음
        for block in page.get_text("dict")["blocks"]:  # 페이지를 딕셔너리 구조로 파싱하여 블록 순회
            if block.get("type") != 0:  # type 0이 텍스트 블록. 이미지 등 다른 블록은 스킵
                continue
            for line in block.get("lines", []):  # 텍스트 블록의 줄 순회
                for span in line.get("spans", []):  # 줄의 span(동일 폰트 묶음) 순회
                    size = round(span.get("size", 0), 1)  # 폰트 크기를 소수점 1자리로 반올림
                    size_counts[size] = size_counts.get(size, 0) + 1  # 등장 횟수 카운트

    # 가장 많이 등장한 폰트 크기 반환. 데이터 없으면 기본값 10.0 반환
    return max(size_counts, key=size_counts.__getitem__) if size_counts else 10.0


def _detect_section(fitz_page: fitz.Page, body_size: float) -> str:
    """
    페이지에서 본문보다 큰 폰트의 텍스트를 섹션 헤더로 감지한다.
    첫 번째로 감지된 헤더만 반환하며, 감지 실패 시 빈 문자열을 반환한다.
    """
    for block in fitz_page.get_text("dict")["blocks"]:  # 페이지의 모든 블록 순회
        if block.get("type") != 0:  # 텍스트 블록이 아니면 스킵
            continue
        for line in block.get("lines", []):  # 블록의 줄 순회
            for span in line.get("spans", []):  # 줄의 span 순회
                text = span.get("text", "").strip()  # span의 텍스트 추출 및 공백 제거
                size = span.get("size", 0)           # span의 폰트 크기 추출

                # 본문보다 폰트가 크고 3자 이상인 텍스트를 섹션 헤더로 판단
                if size > body_size:
                    # 깨진문자가 섹션으로 인식되는걸 막기위해 영어나 한글이 들어가 있는 경우에만 섹션으로 인식하도록 함
                    if re.search(r'[가-힣a-zA-Z]', text):
                        return text

    return ""  # 헤더를 찾지 못한 경우 빈 문자열 반환


class VehiclePDFParser:
    """
    차량 매뉴얼 PDF 파서.
    LangChain Document 리스트를 반환하여 이후 LangChain 파이프라인과 자연스럽게 연결된다.

    Args:
        min_text_length: 이 길이 미만 블록은 노이즈로 제거 (기본 20자)
    """

    def __init__(self, min_text_length: int = 20) -> None:
        self.min_text_length = min_text_length  # 최소 텍스트 길이 저장

    def parse(self, pdf_path: str | Path) -> list[Document]:
        """
        PDF를 파싱하여 LangChain Document 리스트를 반환한다.
        각 Document는 한 페이지에 해당하며 메타데이터 4종을 포함한다.

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            list[Document] (메타데이터: source, page_num, section, content_type)

        Raises:
            FileNotFoundError: 파일이 없을 때
        """
        pdf_path = Path(pdf_path)  # 문자열 경로를 Path 객체로 변환

        if not pdf_path.exists():  # 파일 존재 여부 확인
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        documents = []  # 결과를 담을 Document 리스트

        # PyMuPDFLoader로 페이지 단위 텍스트 추출
        # 각 Document는 한 페이지에 해당하며 metadata에 page 번호 포함
        lc_pages = PyMuPDFLoader(str(pdf_path)).load()

        # fitz로 섹션 감지를 위해 동일 파일을 다시 열기
        # PyMuPDFLoader는 텍스트만 주고 폰트 크기 정보를 주지 않으므로 fitz 병행 사용
        with fitz.open(str(pdf_path)) as fitz_doc:
            body_size = _estimate_body_font_size(fitz_doc)  # 본문 폰트 크기 추정
            current_section = ""  # 현재 섹션 추적 (페이지를 넘어서도 유지)

            # PyMuPDFLoader 결과와 fitz 페이지를 동시에 순회
            for lc_page, fitz_page in zip(lc_pages, fitz_doc):

                # LangChain의 page는 0-indexed → 1-indexed로 변환
                page_num = lc_page.metadata.get("page", 0) + 1

                # 섹션 헤더 감지 시도. 감지 실패 시 이전 섹션 유지
                detected = _detect_section(fitz_page, body_size)
                if detected:
                    current_section = detected  # 새 섹션 발견 시 업데이트

                # 노이즈 제거 및 텍스트 정규화
                content = _clean_text(lc_page.page_content)

                # 정제 후 내용이 너무 짧으면 노이즈로 간주하여 제외
                if not content or len(content) < self.min_text_length:
                    continue

                # 메타데이터 4종을 포함한 Document 생성 후 리스트에 추가
                documents.append(Document(
                    page_content=content,       # 정제된 텍스트
                    metadata={
                        "source": pdf_path.name,         # 원본 파일명
                        "page_num": page_num,             # 페이지 번호
                        "section": current_section,       # 섹션 헤더
                        "content_type": "text",           # 콘텐츠 유형
                    }
                ))

        return documents  # 완성된 Document 리스트 반환


## 이건 파서 잘 작동하는지 시험용 코드임. 

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:  # 파일 경로 인자가 없으면 사용법 출력
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    docs = VehiclePDFParser().parse(sys.argv[1])  # 파서 실행

    # 결과 요약을 JSON으로 출력 (확인용)
    print(json.dumps({
        "total_documents": len(docs),  # 전체 Document 수
        "sample": [
            {
                "page": d.metadata["page_num"],    # 페이지 번호
                "section": d.metadata["section"],  # 섹션 헤더
                "preview": d.page_content[:120],   # 본문 앞 120자 미리보기
            }
            for d in docs[:5]  # 처음 5개만 샘플로 출력
        ],
    }, ensure_ascii=False, indent=2))