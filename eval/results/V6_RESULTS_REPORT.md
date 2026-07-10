# v6.0 Benchmark: Model Testing and Training Results

Status: LIVE REPORT (updated as overnight runs complete). Started 2026-07-10.
Benchmark: qa-data/benchmark/v6/ v6.0 final (585 questions, 92 videos; splits
train 266 / val 65 / test 254). Provenance: 858 generated -> 585 after gates,
two-stage machine review, three attribution audits, 132-vote human ratification,
and a calibrated machine pass transferring the human standard to unreviewed rows
(details in qa-data/benchmark/v6/REVIEW_REPORT.md).

## 1. Four-level decomposition (Qwen3.5-9B, full benchmark, pre-finalization run)

Scored by llama3.3 judge; acc = correctness >= 4/5; robust stratum = questions
no blind panel model answered (n=539 of 744 at run time).

| Config   | Acc    | Robust-stratum acc | Avg tool calls |
|----------|--------|--------------------|----------------|
| No-tools | 15.1%  | 10.0%              | 0              |
| RAG      | 57.3%  | 57.0%              | 0              |
| Agent    | 47.9%  | 48.6%              | 9.1            |
| Oracle   | 81.4%  | 85.5%              | 0              |

Findings:
- The benchmark resists parametric answering by construction: the no-tools
  floor on the robust stratum is 10% (v5-line floor was 56%). Any score above
  it is earned by evidence.
- The oracle ceiling (85.5%) shows the questions are answerable when the right
  evidence is provided; the answerer is not the bottleneck.
- RAG > agent by ~9 points, holding on the robust stratum: the diagnostic
  finding from the v5-era evaluation replicates on clean data. Evidence
  acquisition, not comprehension, is the binding constraint.

## 2. Qualitative review of agent outputs (Qwen3.5-9B warm-index, 584 rows)

Success mode: locate-then-inspect works when vocabulary aligns
(search_transcript hit -> run_asr read -> correct answer).

Dominant failure mode: RETRIEVAL VOCABULARY MISMATCH. The policy searches with
the question's phrasing; when the ASR words differ (garbled names, paraphrase),
searches return nothing, the policy retries near-identical searches (up to 6x),
then answers "the provided evidence does not contain...". 98 of 200 sampled
failures share the get_video_info -> search_transcript opening followed by
repeated fruitless searches. The agent is honest but scoreless where RAG's
overlap-scored whole-index retrieval still lands nearby passages. This mirrors
the round-trip vocabulary confound found during benchmark construction
(2026-07-01), now appearing as the agent's primary deficit: the fix direction
is semantic/fuzzy search tools rather than more capable answering.

## 3. Model comparison sweep (D3) - RUNNING

Same protocol on v6.0-test (254 rows): full agent stack + no-tools floor.
- gemma3:27b-it-qat: in progress
- llama3.3-70B: queued (sequential to avoid model thrashing)
- Qwen3.5-9B test-split numbers to be recut from section 1 outputs
(RESULTS PENDING)

## 4. SFT on v6.0 trajectories - TRAINING

976 warm-index trajectories from 260 train questions (2 standard + retry +
interleaved recovery per question), LoRA r=16 on Qwen3.5-9B, 2 epochs.
Head-to-head planned: SFT policy vs base policy on v6.0-test - the direct test
of whether training on reviewed data reverses the v4.1 negative result
(base 71% > SFT 62% on v4.1 MC).
(RESULTS PENDING)

## 5. Earlier judge-filter experiment (negative result, kept for the record)

One-shot local-LLM judges (llama3.3) rate machine-excluded rows 4.86/5 vs 4.97
for kept rows: too lenient to function as a quality filter. Evidence-grounded
agent review with adversarial verification succeeded where rubric scoring
failed; this motivated the calibrated transfer pass used for v6.0 finalization.
