#!/bin/bash
# Poll the Vast pod (shards 06-09) every 5 minutes; when the run finishes,
# download the shards to vast2/Corpus_TeenyTinyStories/, md5-verify, and mark done.
# Runs under the user account (no sudo). Stop by killing the pid in vast2/.watcher.pid.
set -u
cd "$(dirname "$0")/.."
PROJ="$(pwd)"
STAGE="$PROJ/vast2/Corpus_TeenyTinyStories"
MARK="$PROJ/vast2/.download_complete"
PIDF="$PROJ/vast2/.watcher.pid"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -p 5370 root@62.65.214.167"
REMOTE="/workspace/vast2/Corpus_TeenyTinyStories"
FILES=(stories_shard06.jsonl stories_shard07.jsonl stories_shard08.jsonl stories_shard09.jsonl
       summary_shard06.json summary_shard07.json summary_shard08.json summary_shard09.json
       run_config_shard06.json run_config_shard07.json run_config_shard08.json run_config_shard09.json)

log() { echo "[$(date '+%F %T')] $*"; }

if [ -f "$MARK" ]; then log "marker exists; nothing to do"; exit 0; fi
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
  log "another watcher is running (pid $(cat "$PIDF")); exiting"; exit 0
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

mkdir -p "$STAGE"
log "watcher started (pid $$); polling every 300s"

while true; do
  if [ -f "$MARK" ]; then log "marker appeared; exiting"; exit 0; fi

  FIN="$($SSH 'awk "/^all shards finished/{f=1} END{print f+0}" /workspace/vast2/logs/run_all.log 2>/dev/null' 2>/dev/null)"
  FIN="$(printf '%s' "$FIN" | tr -dc '0-9' | head -c 1)"
  [ -z "$FIN" ] && FIN=0
  if [ "$FIN" -lt 1 ]; then
    PROCS="$($SSH 'pgrep -c -f "[g]enerate.py" 2>/dev/null || echo 0' 2>/dev/null)"
    PROG="$($SSH 'for f in /workspace/vast2/logs/shard0*.log; do tail -1 "$f" | cut -c1-72; done' 2>/dev/null)"
    log "still running (procs=$PROCS):"
    echo "$PROG" | while IFS= read -r l; do [ -n "$l" ] && log "  $l"; done
    sleep 300
    continue
  fi

  log "pod reports all shards finished; starting download"
  rm -f "$STAGE"/*.jsonl "$STAGE"/*.json
  pids=()
  for f in "${FILES[@]}"; do
    scp -q -C -P 5370 "root@62.65.214.167:$REMOTE/$f" "$STAGE/" &
    pids+=($!)
  done
  ok=1
  for p in "${pids[@]}"; do wait "$p" || ok=0; done
  if [ "$ok" -ne 1 ]; then log "scp failed; retrying next cycle"; sleep 300; continue; fi

  log "download done; verifying md5"
  RHASH="$($SSH 'cd /workspace/vast2/Corpus_TeenyTinyStories && md5sum stories_shard0{6,7,8,9}.jsonl' 2>/dev/null | awk '{print $1, $2}' | sort)"
  LHASH="$(cd "$STAGE" && md5 -r stories_shard0{6,7,8,9}.jsonl | awk '{print $1, $2}' | sort)"
  if [ "$RHASH" != "$LHASH" ]; then
    log "MD5 MISMATCH; retrying next cycle"
    log "remote: $RHASH"
    log "local:  $LHASH"
    sleep 300
    continue
  fi

  TOT="$(python3 - <<'EOF'
import json, glob
tot = 0
for f in sorted(glob.glob("vast2/Corpus_TeenyTinyStories/summary_shard0[6-9].json")):
    d = json.load(open(f))
    tot += d["story_tokens"]
    print(f"{f.split('/')[-1]}: {d['stories']} stories, {d['story_tokens']} tokens")
print("total tokens:", tot)
EOF
)"
  log "md5 verified for 4 shards"
  log "$TOT"
  date '+%F %T' > "$MARK"
  log "wrote $MARK; watcher exiting"
  exit 0
done
