#!/bin/bash
# Scheduled wrapper for the frontier-LLM forward test: predict, then commit and push ONLY the
# day's tamper-evidence digests (research/forward/hashes). The git timestamp is the external record.
cd /Users/yeswanthrajmajji/atlas || exit 1
echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
/Users/yeswanthrajmajji/.local/bin/uv run python -u -W ignore scripts/llm_forward_run.py
status=$?
[ -d research/forward/hashes ] && git add research/forward/hashes
if [ -d research/forward/hashes ] && ! git diff --cached --quiet -- research/forward/hashes; then
  git commit -q -m "Forward test: digests $(date -u +%Y-%m-%d)" -- research/forward/hashes \
    && git push -q && echo "digests pushed" || echo "PUSH FAILED: push the digests by hand"
fi
exit $status
