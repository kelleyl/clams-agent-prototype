# V3 Dataset Manifest

Generated: April 7-8, 2026
Generator: `qa-data/generate_qa_v3.py` using Gemini 2.5 Flash
Source: 107 v2 layered video indexes with Qwen 3.5 captions

## Canonical Files

### Raw Questions (qa-data/raw/)
| File | Count | Description |
|------|-------|-------------|
| `qa_v3_combined_valid.jsonl` | 925 | All valid V3 questions (source of truth) |
| `qa_v3_train.jsonl` | 465 | Train split (53 videos) |
| `qa_v3_val.jsonl` | 84 | Val split (10 videos) |
| `qa_v3_test.jsonl` | 376 | Test split (43 videos) |
| `video_split.json` | -- | Split metadata (video lists, seed=42, 50/10/40) |

### Benchmark with Distractors (qa-data/benchmark/v3/)
| File | Count | MC | FT |
|------|-------|-----|-----|
| `benchmark_combined.jsonl` | 917 | 488 | 429 |
| `train_benchmark.jsonl` | 464 | 242 | 222 |
| `val_benchmark.jsonl` | 80 | 34 | 46 |
| `test_benchmark.jsonl` | 373 | 212 | 161 |

## Split Details

- **Split level**: Video (no video appears in multiple splits)
- **Seed**: 42
- **Ratio**: 50% train / 10% val / 40% test
- Train: 53 videos (NH:28, NJN:8, FM:8, Other:9)
- Val: 10 videos (NH:4, FM:3, Other:2, NJN:1)
- Test: 43 videos (NJN:14, NH:13, Other:9, FM:7)

## Rules

- **Train videos**: trajectory generation, SFT, GRPO, all training
- **Val videos**: hyperparameter tuning, early stopping, debugging
- **Test videos**: headline numbers only, never train on these
- No trajectory generation, prompt tuning, or analysis on test videos

## Build Artifacts (do not use for training or eval)

Located in `qa-data/raw/v3_build/` and `qa-data/benchmark/v3_build/`:
- `qa_v3.jsonl` (560) -- first Gemini batch
- `qa_v3_missing.jsonl` (510) -- second batch for failed videos
- `qa_v3_missing_valid.jsonl` (428)
- `qa_v3_valid.jsonl` (497)
- `benchmark.jsonl`, `mc_questions.jsonl`, `freetext_questions.jsonl` -- first batch only
- `v3_missing/` -- second batch benchmark

## Generation Pipeline

1. Full video index -> Gemini 2.5 Flash (full-context, ~33k tokens/video)
2. Validation: answer in index, cross-modal check, interstitial filter
3. Distractor generation: qwen3:8b via Ollama with adversarial filter
4. Video-level split: random shuffle with seed=42
