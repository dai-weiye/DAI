#!/usr/bin/env bash
# Cross-model non-termination probe via pure curl (Python stdout is swallowed in this
# shell, so everything goes through curl+files instead). Reads results/xmodel_prompts.jsonl,
# calls the relay per item at cap=8192, appends one JSON line per item to the output.
# Idempotent: skips items already present in the output file, so re-running resumes.
#
# Usage:
#   OPENAI_BASE_URL=https://api.v36.cm/v1 OPENAI_API_KEY=sk-... \
#     bash scripts/xmodel_curl.sh claude-opus-4-8
set -u
MODEL="${1:-claude-opus-4-8}"
BASE="${OPENAI_BASE_URL:?set OPENAI_BASE_URL}"
KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"
CAP=8192
# Resolve paths relative to this script's location, so it works regardless of CWD.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN="$HERE/results/xmodel_prompts.jsonl"
OUT="$HERE/results/xmodel_${MODEL//\//_}.jsonl"
LOG="$HERE/results/xmodel_${MODEL//\//_}.log"
: > "$LOG"
touch "$OUT"
echo "START model=$MODEL cwd=$(pwd) HERE=$HERE" | tee -a "$LOG"

total=$(wc -l < "$IN" | tr -d ' ')
echo "model=$MODEL base=$BASE cap=$CAP total=$total out=$OUT" | tee -a "$LOG"

i=0
while IFS= read -r line; do
  i=$((i+1))
  id=$(printf '%s' "$line"    | jq -r '.id')
  cond=$(printf '%s' "$line"  | jq -r '.condition')
  gold=$(printf '%s' "$line"  | jq -r '.gold')
  prompt=$(printf '%s' "$line"| jq -r '.prompt')
  tag="${id}|${cond}"

  # resume: skip if already done
  if grep -q "\"tag\":\"${tag}\"" "$OUT" 2>/dev/null; then
    echo "[$i/$total] skip $tag (cached)" | tee -a "$LOG"; continue
  fi

  body=$(jq -n --arg m "$MODEL" --arg c "$prompt" --argjson mt "$CAP" \
    '{model:$m, messages:[{role:"user",content:$c}], temperature:0, max_tokens:$mt}')

  resp=$(curl -sS --max-time 180 -X POST "$BASE/chat/completions" \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "$body" -w $'\n__HTTP__%{http_code}')
  http=$(printf '%s' "$resp" | sed -n 's/.*__HTTP__\([0-9]*\)$/\1/p')
  jsonpart=$(printf '%s' "$resp" | sed 's/__HTTP__[0-9]*$//')

  if [ "$http" != "200" ]; then
    echo "[$i/$total] $tag HTTP=$http FAIL: $(printf '%s' "$jsonpart" | head -c 200)" | tee -a "$LOG"
    continue
  fi

  # extract fields; content may be null -> treat as empty
  content=$(printf '%s' "$jsonpart" | jq -r '.choices[0].message.content // ""')
  finish=$( printf '%s' "$jsonpart" | jq -r '.choices[0].finish_reason // "null"')
  ctok=$(   printf '%s' "$jsonpart" | jq -r '.usage.completion_tokens // 0')

  # non-termination (gold-independent): empty answer-ish AND budget exhausted.
  # We approximate "no committed answer" as content lacking an "Answer:" tag.
  has_ans=$(printf '%s' "$content" | grep -ci 'answer:' || true)
  nonterm=false
  if [ "$ctok" -ge $((CAP-8)) ] && [ "$has_ans" -eq 0 ]; then nonterm=true; fi

  # append result line (store content length + whether it had an Answer tag, not full text)
  jq -cn --arg tag "$tag" --arg id "$id" --arg cond "$cond" --arg gold "$gold" \
     --arg finish "$finish" --argjson ctok "$ctok" --argjson has_ans "$has_ans" \
     --argjson nonterm "$nonterm" \
     '{tag:$tag,id:$id,condition:$cond,gold:$gold,finish_reason:$finish,completion_tokens:$ctok,has_answer_tag:($has_ans>0),nonterm:$nonterm}' >> "$OUT"

  echo "[$i/$total] $tag ct=$ctok finish=$finish nonterm=$nonterm" | tee -a "$LOG"
done < "$IN"

echo "DONE. wrote $OUT" | tee -a "$LOG"
