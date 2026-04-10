#!/bin/bash
set -e

WORK_DIR=$HOME/newshour_pipeline
QWEN_DIR=$HOME/clams_apps/app-qwen3vl-captioner
VIDEO_DIR=/mnt/llc/llc_data/clams/wgbh/NewsHour

GUIDS=(
  cpb-aacip-507-154dn40c26 cpb-aacip-507-1v5bc3tf81
  cpb-aacip-507-4t6f18t178 cpb-aacip-507-6w96689725
  cpb-aacip-507-7659c6sk7z cpb-aacip-507-9882j68s35
  cpb-aacip-507-cf9j38m509 cpb-aacip-507-n29p26qt59
  cpb-aacip-507-nk3610wp6s cpb-aacip-507-pc2t43js98
)
total=${#GUIDS[@]}

export CUDA_VISIBLE_DEVICES=1
cd $QWEN_DIR
source .venv/bin/activate

# === Pass 1: Qwen on SWT timeframes (input: SWT MMIF only) ===
echo "=== Qwen Pass 1: SWT Timeframes ==="
mkdir -p $WORK_DIR/qwen_swt_out
i=0
for g in "${GUIDS[@]}"; do
  i=$((i+1))
  INPUT="$WORK_DIR/step1_swt/${g}.mmif"
  OUTPUT="$WORK_DIR/qwen_swt_out/${g}.mmif"
  if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
    echo "[$i/$total] SKIP: $g"
    continue
  fi
  echo "[$i/$total] Qwen(SWT): $g"
  t_start=$(date +%s)
  python3 cli.py --config config/swt_transcription.yaml --batchSize 64 "$INPUT" "$OUTPUT"
  t_end=$(date +%s)
  sz=$(du -h "$OUTPUT" | cut -f1)
  echo "[$i/$total] Done ${sz} in $((t_end - t_start))s"
done

# === Pass 2: Qwen on TransNet shots (input: TransNet MMIF only) ===
echo ""
echo "=== Qwen Pass 2: TransNet Shots ==="
mkdir -p $WORK_DIR/qwen_shots_out
i=0
for g in "${GUIDS[@]}"; do
  i=$((i+1))
  INPUT="$WORK_DIR/step3_transnet_raw/${g}_transnet.mmif"
  OUTPUT="$WORK_DIR/qwen_shots_out/${g}.mmif"
  if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
    echo "[$i/$total] SKIP: $g"
    continue
  fi
  echo "[$i/$total] Qwen(shots): $g"
  t_start=$(date +%s)
  python3 cli.py --config config/transnet_shots.yaml --batchSize 64 "$INPUT" "$OUTPUT"
  t_end=$(date +%s)
  sz=$(du -h "$OUTPUT" | cut -f1)
  echo "[$i/$total] Done ${sz} in $((t_end - t_start))s"
done

# === Merge all views into final MMIFs ===
echo ""
echo "=== Merging all views ==="
mkdir -p $WORK_DIR/final
python3 << 'PYEOF'
import json, os

work_dir = os.path.expanduser("~/newshour_pipeline")
guids = [
    "cpb-aacip-507-154dn40c26", "cpb-aacip-507-1v5bc3tf81",
    "cpb-aacip-507-4t6f18t178", "cpb-aacip-507-6w96689725",
    "cpb-aacip-507-7659c6sk7z", "cpb-aacip-507-9882j68s35",
    "cpb-aacip-507-cf9j38m509", "cpb-aacip-507-n29p26qt59",
    "cpb-aacip-507-nk3610wp6s", "cpb-aacip-507-pc2t43js98",
]

for i, g in enumerate(guids, 1):
    base = json.load(open(f"{work_dir}/step2_whisper/{g}.mmif"))

    # Add TransNet views
    tn_path = f"{work_dir}/step3_transnet_raw/{g}_transnet.mmif"
    if os.path.exists(tn_path):
        tn = json.load(open(tn_path))
        for v in tn.get("views", []):
            if v.get("annotations"):
                base["views"].append(v)

    # Add Qwen SWT views
    qswt_path = f"{work_dir}/qwen_swt_out/{g}.mmif"
    if os.path.exists(qswt_path):
        qswt = json.load(open(qswt_path))
        for v in qswt.get("views", []):
            app = v.get("metadata", {}).get("app", "")
            if "qwen" in app.lower() or "captioner" in app.lower():
                base["views"].append(v)

    # Add Qwen shots views
    qshots_path = f"{work_dir}/qwen_shots_out/{g}.mmif"
    if os.path.exists(qshots_path):
        qshots = json.load(open(qshots_path))
        for v in qshots.get("views", []):
            app = v.get("metadata", {}).get("app", "")
            if "qwen" in app.lower() or "captioner" in app.lower():
                base["views"].append(v)

    out_path = f"{work_dir}/final/{g}.mmif"
    json.dump(base, open(out_path, "w"))
    print(f"  [{i}/{len(guids)}] {g}: {len(base['views'])} views merged")
PYEOF

echo ""
echo "=== PIPELINE COMPLETE ==="
ls -lh $WORK_DIR/final/
