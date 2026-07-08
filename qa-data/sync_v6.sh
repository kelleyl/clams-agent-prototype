#!/bin/bash
# Sync v6 run outputs from aristotle to local (mirror of sync_v5_in_progress.sh).
set -e
cd "$(dirname "$0")/.."
REMOTE=aristotle:clams_apps/clams-agent-prototype

rsync -a "$REMOTE/data/qa_needdown/" data/qa_needdown/
rsync -a "$REMOTE/data/qa_exploration/" data/qa_exploration/ 2>/dev/null || true
rsync -a "$REMOTE/data/corpus_catalog.json" data/ 2>/dev/null || true

echo "--- local state after sync ---"
python3 - <<'EOF'
import json, glob
n = k = 0
for f in glob.glob('data/qa_needdown/*.json'):
    if f.endswith('run_stats.json'):
        continue
    d = json.load(open(f))
    n += sum(1 for r in d.get('rows', []) if r.get('qa', {}).get('question'))
en = ek = 0
for f in glob.glob('data/qa_exploration/*.json'):
    d = json.load(open(f))
    en += len(d.get('rows', []))
    ek += sum(1 for r in d.get('rows', []) if r.get('keep'))
print(f'need-down questions: {n} | exploration: {ek}/{en} kept')
EOF
