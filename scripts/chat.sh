#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${MINI_DEERFLOW_URL:-http://localhost:2027}"
MESSAGE="${1:?Usage: chat.sh <message> [thread_id]}"
THREAD_ID="${2:-}"
STREAM_MODE="${MINI_DEERFLOW_STREAM_MODE:-messages}"

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health" || echo "000")
if [ "$HTTP_CODE" = "000" ] || [ "$HTTP_CODE" -ge 400 ]; then
  echo "Mini DeerFlow is not reachable at $BASE_URL"
  exit 1
fi

if [ -z "$THREAD_ID" ]; then
  RESP=$(curl -s -X POST "$BASE_URL/api/threads")
  THREAD_ID=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['thread_id'])")
  echo "Thread: $THREAD_ID" >&2
fi

ESCAPED_MSG=$(python -c "import json,sys; print(json.dumps(sys.argv[1], ensure_ascii=False))" "$MESSAGE")

BODY=$(cat <<ENDJSON
{
  "assistant_id": "lead_agent",
  "input": {
    "messages": [
      {
        "type": "human",
        "content": [{"type": "text", "text": ${ESCAPED_MSG}}]
      }
    ]
  },
  "stream_mode": ["${STREAM_MODE}"],
  "context": {
    "thinking_enabled": true,
    "is_plan_mode": true,
    "subagent_enabled": false
  }
}
ENDJSON
)

curl -s -N -X POST "$BASE_URL/api/threads/$THREAD_ID/runs/stream" \
  -H "Content-Type: application/json" \
  -d "$BODY" | python - << 'PYEOF'
import json
import sys

current_event = None
current_data = []
last_response = None
streamed_any = False

def handle_event(event, data):
    global last_response, streamed_any
    try:
        payload = json.loads(data)
    except Exception:
        return

    if event == "messages" and isinstance(payload, list) and payload:
        chunk = payload[0]
        if isinstance(chunk, dict):
            content = chunk.get("content") or ""
            if content:
                print(content, end="", flush=True)
                streamed_any = True
        return

    if event in {"final", "clarification"}:
        last_response = payload

for raw_line in sys.stdin:
    line = raw_line.rstrip("\n")
    if line.startswith("event:"):
        if current_event and current_data:
            handle_event(current_event, "\n".join(current_data))
        current_event = line[len("event:"):].strip()
        current_data = []
    elif line.startswith("data:"):
        current_data.append(line[len("data:"):].strip())
    elif line == "" and current_event:
        if current_data:
            handle_event(current_event, "\n".join(current_data))
        current_event = None
        current_data = []

if current_event and current_data:
    handle_event(current_event, "\n".join(current_data))

assistant_message = (last_response or {}).get("assistant_message")
if assistant_message and not streamed_any:
    print(assistant_message)
    sys.exit(0)

if streamed_any:
    print()
    sys.exit(0)

print("No AI message")
PYEOF

echo ""
echo "---"
echo "Thread ID: $THREAD_ID" >&2
