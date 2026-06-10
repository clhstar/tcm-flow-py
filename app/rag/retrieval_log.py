# 用于记录每次 RAG 检索，后面论文实验和调试都很有用。
import json
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_DIR = Path("data/logs")
LOG_FILE = LOG_DIR / "retrieval.jsonl"


def write_retrieval_log(payload: dict[str, Any]):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "time": datetime.utcnow().isoformat(),
        **payload,
    }

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_recent_logs(limit: int = 20) -> list[dict[str, Any]]:
    if not LOG_FILE.exists():
        return []

    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    recent_lines = lines[-limit:]

    results = []
    for line in recent_lines:
        try:
            results.append(json.loads(line))
        except Exception:
            continue

    return results