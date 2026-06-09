#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${MINI_DEERFLOW_URL:-http://localhost:2027}"
MESSAGE="${1:?Usage: chat.sh <message> [thread_id]}"
THREAD_ID="${2:-}"

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
  "stream_mode": ["values"],
  "context": {
    "thinking_enabled": true,
    "is_plan_mode": true,
    "subagent_enabled": false
  }
}
ENDJSON
)

TMPFILE=$(mktemp)
trap "rm -f '$TMPFILE'" EXIT

curl -s -N -X POST "$BASE_URL/api/threads/$THREAD_ID/runs/stream" \
  -H "Content-Type: application/json" \
  -d "$BODY" > "$TMPFILE"

python - "$TMPFILE" << 'PYEOF'
import json
import sys

raw = open(sys.argv[1], encoding="utf-8").read()

events = []
current_event = None
current_data = []

for line in raw.splitlines():
    if line.startswith("event:"):
        if current_event and current_data:
            events.append((current_event, "\n".join(current_data)))
        current_event = line[len("event:"):].strip()
        current_data = []
    elif line.startswith("data:"):
        current_data.append(line[len("data:"):].strip())
    elif line == "" and current_event:
        if current_data:
            events.append((current_event, "\n".join(current_data)))
        current_event = None
        current_data = []

if current_event and current_data:
    events.append((current_event, "\n".join(current_data)))

last_values = None
for event, data in reversed(events):
    if event == "values":
        try:
            last_values = json.loads(data)
            break
        except Exception:
            pass

if not last_values:
    print("No response")
    sys.exit(1)

messages = last_values.get("messages", [])
for m in reversed(messages):
    if m.get("type") == "ai" and m.get("content"):
        print(m["content"])
        sys.exit(0)

print("No AI message")
PYEOF

echo ""
echo "---"
echo "Thread ID: $THREAD_ID" >&2