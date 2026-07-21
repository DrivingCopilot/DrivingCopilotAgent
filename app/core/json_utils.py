# app/core/json_utils.py
#
# 로컬 LLM(Qwen2-VL 7B 등)이 JSON 객체를 반복 출력하거나 reasoning 텍스트에
# 중괄호를 포함시켜도 안전하게 파싱하기 위한 공유 유틸리티.
# perception.py, app/model_server/server.py(tool_call 블록 파싱)에서 사용한다.


def extract_first_json_object(text: str) -> str:
    """
    텍스트에서 첫 번째로 완성되는 최상위 JSON 객체만 잘라서 반환한다.
    모델이 같은(혹은 다른) 객체를 계속 이어붙여도 첫 블록만 사용하고 나머지는 버린다.
    문자열 리터럴 내부의 '{'/'}' 는 깊이 계산에서 제외해 reasoning/description 내용에
    중괄호가 등장해도 오작동하지 않는다.
    """
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:]  # 못 닫혔으면 원본 그대로 반환 (json.loads 에서 에러로 처리됨)
