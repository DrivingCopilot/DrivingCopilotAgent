"""
evaluation/_gold.py

RAG/SQL/multi/tool Gold Set 공용 로더.

run_knowledge_eval.py 의 load_gold_set() 을 일반화한 버전 — 두 가지 소스를 지원한다:
    1) 워킹트리에 GOLD_FILES 경로의 파일이 실제로 존재하면 그것을 우선 읽는다.
    2) 없으면 git show {gold_ref}:{path} 로 지정 브랜치의 blob 을 읽는다
       (run_knowledge_eval.py 의 기존 방식과 동일한 폴백).

호출자가 어느 소스에서 읽혔는지 추적할 수 있도록 provenance(출처/해시/건수)를
함께 반환해, 평가 리포트에 그대로 실을 수 있게 한다.

run_knowledge_eval.py / run_exception_sweep.py 는 이 모듈을 쓰지 않는다(기존 동작 유지).
새로 추가되는 run_tool_eval.py 가 이 모듈을 사용한다.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

# gold-set 파일명 ↔ route 대응.
GOLD_FILES: Dict[str, str] = {
    "rag": "evaluation/gold_set/RAG_gold_set.json",
    "sql": "evaluation/gold_set/SQL_gold_set.json",
    "multi": "evaluation/gold_set/multi_gold_set.json",
    "tool": "evaluation/gold_set/tool_gold_set.json",
}


def load_gold_set(
    set_name: str,
    gold_ref: str = "origin/feature/gold-set",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """지정된 gold set 을 로드한다.

    워킹트리에 GOLD_FILES[set_name] 경로의 파일이 있으면 그것을 우선 읽고,
    없으면 git show 로 gold_ref 브랜치의 blob 을 읽는다(폴백).

    Args:
        set_name: GOLD_FILES 의 키("rag"/"sql"/"multi"/"tool").
        gold_ref: 워킹트리에 파일이 없을 때 읽어올 git ref.

    Returns:
        (items, provenance) 튜플.
        provenance 는 {"set", "source", "sha256", "count"} 를 담는다.
        - source 는 "worktree:<path>" 또는 "git:<gold_ref>:<path>" 형식.
        - sha256 은 읽은 원본 바이트 기준 해시(파싱 전).

    Raises:
        KeyError: set_name 이 GOLD_FILES 에 없는 경우.
        SystemExit: 워킹트리에도 없고 git show 도 실패한 경우.
    """
    path = GOLD_FILES[set_name]
    worktree_path = Path(path)

    if worktree_path.is_file():
        raw = worktree_path.read_bytes()
        source = f"worktree:{path}"
    else:
        try:
            raw = subprocess.check_output(
                ["git", "show", f"{gold_ref}:{path}"],
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", "replace").strip()
            raise SystemExit(
                f"[gold-set 로드 실패] 워킹트리에도 없고 git show {gold_ref}:{path} 도 실패\n{stderr}\n"
                f"→ 브랜치가 있는지 확인: git fetch origin {gold_ref.split('/')[-1]}"
            )
        source = f"git:{gold_ref}:{path}"

    items: List[Dict[str, Any]] = json.loads(raw)
    provenance: Dict[str, Any] = {
        "set": set_name,
        "source": source,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "count": len(items),
    }
    return items, provenance
