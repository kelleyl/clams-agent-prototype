# Pipeline Instructions: TransNet Shot Detection + Qwen Shot Captioning

After Whisper ASR finishes, run these two remaining steps on Aristotle to complete the video indexes. The goal is to add shot boundaries (TransNet) and then caption every shot with visual descriptions to supplement the OCR text, ASR transcripts, and other information already in the index. Some shots will overlap with SWT text-containing regions and that's fine.

## Prerequisites

- **Aristotle access** with GPU available (RTX A6000)
- **TransNet container already built**: `app-transnet-wrapper:local` (Podman image on Aristotle)
- **Qwen captioner**: `~/app-qwen3vl-captioner/cli.py` (runs directly, no container)
- **Whisper output MMIFs**: should contain views from SWT + Qwen OCR + Qwen caption + Whisper ASR
- **10 test videos** at `/home/kmlynch/chicago_tv/FuzzyMemoriesTV/`:
  1. `A_CBS_Family_Presentation_Bumper_1973.mp4`
  2. `ABC_Network_-_Barbary_Coast_-_Funny_Money_-_WCVB_Channel_5_Complete_Broadcast_9_8_1975.mp4`
  3. `ABC_Network_-_Future_Cop_-_The_Kansas_City_Kid_-_WCVB-TV_Complete_Broadcast_4_30_1977.mp4`
  4. `ABC_Network_-_General_Hospital_Last_14_Minutes_11_20_1979.mp4`
  5. `ABC_Network_-_Let_s_Make_a_Deal_-_WLS_Channel_7_Pre-Show_Break_Opening_Moments_2_5_1976.mp4`
  6. `ABC_News_Nightline_-_Bitter_Cold_-_WLS_Channel_7_1st_16_Minutes_1_11_1982.mp4`
  7. `ABC_News_Special_Report_-_The_President_at_the_United_Nations_-_WLS-TV_1985.mp4`
  8. `ABC_News_Weekend_Report_-_WLS_Channel_7_Complete_Broadcast_2_18_1979.mp4`
  9. `ABC_News_Weekend_Report_-_WLS_Channel_7_Complete_Broadcast_3_9_1980.mp4`
  10. `Alexander_s_Star_by_Ideal_-_Don_t_Pick_It_Up_Commercial_1982.mp4`

## Step 1: Run TransNet Shot Detection (Container)

TransNet uses TensorFlow on GPU and runs inside a Podman container. Feed each Whisper output MMIF as input so all views accumulate in a single MMIF.

### 1a. Start the TransNet server

```bash
# Pick a free GPU (check with nvidia-smi). Avoid GPUs 2-3 (krim's training).
# GPU 0 or GPU 1 are safe.
clamspod app-transnet-wrapper:local 5555 --device nvidia.com/gpu=1
```

Verify it's running:
```bash
curl http://localhost:5555
# Should return app metadata JSON
```

### 1b. Create output directory

```bash
mkdir -p ~/clams-agent-prototype/data/combined_outputs
```

### 1c. Run TransNet on all 10 videos

For each video, POST the Whisper output MMIF to the TransNet server. The output will contain all previous views plus a new TransNet view with shot boundary TimeFrames.

```bash
# Adjust WHISPER_OUTPUT_DIR to wherever the Whisper output MMIFs are stored
WHISPER_OUTPUT_DIR=~/clams-agent-prototype/data/whisper_outputs
COMBINED_DIR=~/clams-agent-prototype/data/combined_outputs

for vid in \
  "A_CBS_Family_Presentation_Bumper_1973" \
  "ABC_Network_-_Barbary_Coast_-_Funny_Money_-_WCVB_Channel_5_Complete_Broadcast_9_8_1975" \
  "ABC_Network_-_Future_Cop_-_The_Kansas_City_Kid_-_WCVB-TV_Complete_Broadcast_4_30_1977" \
  "ABC_Network_-_General_Hospital_Last_14_Minutes_11_20_1979" \
  "ABC_Network_-_Let_s_Make_a_Deal_-_WLS_Channel_7_Pre-Show_Break_Opening_Moments_2_5_1976" \
  "ABC_News_Nightline_-_Bitter_Cold_-_WLS_Channel_7_1st_16_Minutes_1_11_1982" \
  "ABC_News_Special_Report_-_The_President_at_the_United_Nations_-_WLS-TV_1985" \
  "ABC_News_Weekend_Report_-_WLS_Channel_7_Complete_Broadcast_2_18_1979" \
  "ABC_News_Weekend_Report_-_WLS_Channel_7_Complete_Broadcast_3_9_1980" \
  "Alexander_s_Star_by_Ideal_-_Don_t_Pick_It_Up_Commercial_1982"
do
  echo "Processing: $vid"
  # Find the Whisper output MMIF (adjust filename pattern as needed)
  INPUT_MMIF="$WHISPER_OUTPUT_DIR/${vid}.mmif"
  OUTPUT_MMIF="$COMBINED_DIR/${vid}.mmif"

  if [ ! -f "$INPUT_MMIF" ]; then
    echo "  SKIP: input MMIF not found at $INPUT_MMIF"
    continue
  fi

  curl -s -X POST http://localhost:5555/annotate \
    -H 'Content-Type: application/json' \
    -d @"$INPUT_MMIF" \
    -o "$OUTPUT_MMIF"

  echo "  Done -> $OUTPUT_MMIF"
done
```

TransNet is fast (~100x realtime on GPU), so even the longest video (Barbary Coast, ~60 min) should finish in under a minute.

### 1d. Verify TransNet output

```bash
# Quick check: count TimeFrames in the TransNet view
python3 -c "
import json, sys
for f in sorted(__import__('pathlib').Path('$COMBINED_DIR').glob('*.mmif')):
    m = json.load(open(f))
    for v in m.get('views', []):
        if 'transnet' in v.get('metadata', {}).get('app', '').lower():
            tfs = sum(1 for a in v.get('annotations', []) if 'TimeFrame' in a.get('@type', ''))
            print(f'{f.stem[:55]:55s} {tfs:4d} shots')
            break
    else:
        print(f'{f.stem[:55]:55s}  NO TRANSNET VIEW')
"
```

### 1e. Stop the TransNet server

```bash
podman stop $(podman ps -q --filter ancestor=app-transnet-wrapper:local)
```

## Step 2: Create Qwen Config for Shot Captioning

Create a YAML config file that tells the Qwen captioner to process TransNet's shot boundaries (not SWT's text frames).

### 2a. Create the config file

```bash
cat > ~/app-qwen3vl-captioner/config/transnet_caption.yaml << 'EOF'
default_prompt: |
  Describe the visual content of this video frame. Include details about:
  - People visible and what they are doing
  - The setting or location
  - Any text, graphics, or on-screen elements
  - The general mood or tone of the scene
  Provide a concise but informative description in 2-3 sentences.

context_config:
  input_context: "timeframe"

  timeframe:
    app_uri: "http://apps.clams.ai/transnet-wrapper/"
    label_mapping:
      "shot": "shot"
    ignore_other_labels: false
EOF
```

Key points about this config:
- `app_uri` matches TransNet's identifier so it finds the right view
- `label_mapping` maps TransNet's "shot" label (you can verify the exact label by checking the TransNet output MMIF)
- `ignore_other_labels: false` ensures all TimeFrames are processed even if the label doesn't match the mapping (they'll use the `default_prompt`)
- No OCR-style prompts needed -- this is purely visual description

### 2b. Verify the TransNet label name

Before running, check what label TransNet actually assigns to its TimeFrames:

```bash
python3 -c "
import json
m = json.load(open('$COMBINED_DIR/A_CBS_Family_Presentation_Bumper_1973.mmif'))
for v in m.get('views', []):
    if 'transnet' in v.get('metadata', {}).get('app', '').lower():
        labels = set()
        for a in v.get('annotations', []):
            if 'TimeFrame' in a.get('@type', ''):
                labels.add(a.get('properties', {}).get('label', 'NO_LABEL'))
        print('TransNet labels:', labels)
        break
"
```

If the label is something other than `"shot"` (e.g., `"Shot"` or blank), update the `label_mapping` in the config accordingly. If there's no label at all, just remove the `label_mapping` section entirely and rely on `default_prompt`.

## Step 3: Run Qwen Captioning on TransNet Shots

The Qwen captioner runs via `cli.py` directly (not containerized). It uses the GPU and takes significantly longer than TransNet.

### 3a. Pick a GPU

```bash
# Check GPU availability
nvidia-smi
# Use a free GPU. Set CUDA_VISIBLE_DEVICES to isolate it.
export CUDA_VISIBLE_DEVICES=1  # or 0, whichever is free
```

### 3b. Run Qwen on each combined MMIF

```bash
COMBINED_DIR=~/clams-agent-prototype/data/combined_outputs
QWEN_DIR=~/app-qwen3vl-captioner

for vid in \
  "A_CBS_Family_Presentation_Bumper_1973" \
  "ABC_Network_-_Barbary_Coast_-_Funny_Money_-_WCVB_Channel_5_Complete_Broadcast_9_8_1975" \
  "ABC_Network_-_Future_Cop_-_The_Kansas_City_Kid_-_WCVB-TV_Complete_Broadcast_4_30_1977" \
  "ABC_Network_-_General_Hospital_Last_14_Minutes_11_20_1979" \
  "ABC_Network_-_Let_s_Make_a_Deal_-_WLS_Channel_7_Pre-Show_Break_Opening_Moments_2_5_1976" \
  "ABC_News_Nightline_-_Bitter_Cold_-_WLS_Channel_7_1st_16_Minutes_1_11_1982" \
  "ABC_News_Special_Report_-_The_President_at_the_United_Nations_-_WLS-TV_1985" \
  "ABC_News_Weekend_Report_-_WLS_Channel_7_Complete_Broadcast_2_18_1979" \
  "ABC_News_Weekend_Report_-_WLS_Channel_7_Complete_Broadcast_3_9_1980" \
  "Alexander_s_Star_by_Ideal_-_Don_t_Pick_It_Up_Commercial_1982"
do
  INPUT="$COMBINED_DIR/${vid}.mmif"
  OUTPUT="$COMBINED_DIR/${vid}_final.mmif"

  if [ ! -f "$INPUT" ]; then
    echo "SKIP: $vid (no combined MMIF)"
    continue
  fi

  echo "[$(date +%H:%M)] Captioning shots: $vid"
  python3 "$QWEN_DIR/cli.py" \
    --config config/transnet_caption.yaml \
    --batchSize 12 \
    "$INPUT" "$OUTPUT"
  echo "[$(date +%H:%M)] Done: $vid"
done
```

**Timing estimates** (based on prior Qwen runs on this dataset):
- Short videos (CBS Bumper, Let's Make a Deal, Alexander's Star): 2-10 minutes each
- Medium videos (General Hospital, Nightline, Special Report, Weekend Reports): 15-25 minutes each
- Long videos (Barbary Coast, Future Cop): 40-60 minutes each
- **Total: ~3-5 hours for all 10 videos**

### 3c. Rename final outputs

Once all captioning completes, replace the intermediate files:

```bash
cd $COMBINED_DIR
for f in *_final.mmif; do
  base="${f%_final.mmif}.mmif"
  mv "$f" "$base"
  echo "Renamed: $f -> $base"
done
```

### 3d. Verify final MMIFs

```bash
python3 -c "
import json
from pathlib import Path
combined = Path('$COMBINED_DIR')
for f in sorted(combined.glob('*.mmif')):
    m = json.load(open(f))
    views = []
    for v in m.get('views', []):
        app = v.get('metadata', {}).get('app', '')
        n_ann = len(v.get('annotations', []))
        if 'swt' in app: views.append(f'SWT({n_ann})')
        elif 'qwen' in app: views.append(f'Qwen({n_ann})')
        elif 'whisper' in app: views.append(f'Whisper({n_ann})')
        elif 'transnet' in app: views.append(f'TransNet({n_ann})')
        else: views.append(f'?({n_ann})')
    print(f'{f.stem[:50]:50s} {\" + \".join(views)}')
"
```

Each video should show: `SWT + SWT + Qwen(OCR) + Qwen(caption) + Whisper + TransNet + Qwen(shot_caption)`

## Step 4: Rebuild Video Indexes + SpaCy NER + Entity Grounding

After all MMIFs are finalized, copy them to the local machine and rebuild indexes with full NER and entity grounding.

### 4a. Copy combined outputs from Aristotle

```bash
# Run this on local machine
mkdir -p ~/clams/clams-agent-prototype/data/combined_outputs
scp aristotle:~/clams-agent-prototype/data/combined_outputs/*.mmif \
    ~/clams/clams-agent-prototype/data/combined_outputs/
```

### 4b. Rebuild indexes with NER + grounding

This script does three things per video:
1. **Rebuild index** from combined MMIF (extracts segments from all views)
2. **SpaCy NER** on ALL text fields — ASR transcripts, OCR text, AND visual captions
3. **Entity grounding** via Wikidata (structured facts) + Wikipedia (prose descriptions)

```bash
cd ~/clams/clams-agent-prototype
python scripts/rebuild_indexes.py
```

Options:
```bash
# Custom MMIF directory
python scripts/rebuild_indexes.py --mmif-dir data/combined_outputs

# Just rebuild indexes, skip NER/grounding (fast)
python scripts/rebuild_indexes.py --skip-ner

# NER but no grounding (faster, no network calls)
python scripts/rebuild_indexes.py --skip-grounding

# Only specific videos
python scripts/rebuild_indexes.py --videos A_CBS_Family_Presentation_Bumper_1973 ABC_News_Nightline_-_Bitter_Cold_-_WLS_Channel_7_1st_16_Minutes_1_11_1982
```

### 4c. Verify the rebuilt indexes

```bash
python3 -c "
from utils.video_index import get_video_index
idx = get_video_index()
for v in sorted(idx.get_indexed_videos()):
    s = idx.get_video_summary(v)
    segs = s.get('total_segments', 0)
    ents = s.get('total_entities', 0)
    ocr = s.get('total_ocr_chars', 0)
    asr = s.get('total_asr_chars', 0)
    print(f'{v[:55]:55s} {segs:4d} segs  {ents:4d} ents  OCR:{ocr:6d}  ASR:{asr:6d}')
"
```

Each video should now have entities from ALL text sources (ASR + OCR + VLM captions) with Wikidata/Wikipedia descriptions for disambiguation.

## Troubleshooting

### TransNet container not found
If `clamspod app-transnet-wrapper:local 5555` fails, verify the image exists:
```bash
podman images | grep transnet
```
If missing, rebuild from `~/clams_apps/app-transnet-wrapper/`:
```bash
cd ~/clams_apps/app-transnet-wrapper
podman build -t app-transnet-wrapper:local .
```

### Qwen out of GPU memory
Reduce batch size: `--batchSize 4` instead of 12. The 3B model uses ~8GB VRAM; batch_size=12 needs ~20GB.

### TransNet produces no TimeFrames
Check that the video file paths in the MMIF are accessible from inside the container. The `clamspod` command should mount the video directory. If not, you may need:
```bash
clamspod app-transnet-wrapper:local 5555 --device nvidia.com/gpu=1 -v /home/kmlynch/chicago_tv:/home/kmlynch/chicago_tv
```

### Weekend Report OCR merge issue
The `ABC_News_Weekend_Report` caption MMIFs from `data/qwen_outputs/` are missing OCR views due to a race condition during the original pipeline run. The OCR data exists separately in `*_ocr.mmif` files. If the Whisper worker started from the `_caption.mmif` files, the OCR views may be missing. To merge them, load both the caption and OCR MMIFs and copy the OCR view into the combined MMIF before running TransNet.

```python
import json

def merge_ocr_view(caption_mmif_path, ocr_mmif_path, output_path):
    caption = json.load(open(caption_mmif_path))
    ocr = json.load(open(ocr_mmif_path))

    # Find the OCR view (has "qwen" app and promptMap with "transcribe")
    for v in ocr.get("views", []):
        app_config = v.get("metadata", {}).get("appConfiguration", {})
        prompt_map = app_config.get("promptMap", {})
        prompt_text = str(prompt_map).lower()
        if "transcribe" in prompt_text or "key-value" in prompt_text:
            caption["views"].append(v)
            break

    with open(output_path, "w") as f:
        json.dump(caption, f)
```

Apply this to the two Weekend Report videos before running TransNet if their OCR views are missing.
