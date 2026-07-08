# V6 Dataset Plan

Status: drafted 2026-06-29. Supersedes the v5.x line. Companion to
`qa-data/QUESTION_TYPE_SPEC.md` (the three-axis question design).

## Why v6, not v5.3

Point releases (5.0 -> 5.1 -> 5.2) kept the same generation method and only changed
filters/triage. v6 changes the generation method itself, so it is a major bump:

- evidence-up window sampling -> need-down, salience-first generation
- flat `task_family` -> three-axis taxonomy (information need x cognitive level x 5W role)
- LLM-opinion gates (gamed `modality_fit`) -> empirical gates (blind solvers, entailment)
- new question types absent from 5.x: exploration/retrieval and two-hop grounding-document
- new infrastructure: salience maps, context-aware entity grounding

## Design overview

v6 = need-down, salience-grounded generation with empirical, citation-aware curation.
Generate questions FROM what matters in a video, then curate FOR difficulty and grounding
(the inverse of Source2Synth's answerability curation, which targets training utility).

## A. Generation (built this iteration)

- **Salience map** (`scripts/build_salience_map.py`): chapters -> salient segments;
  speakers -> main participants (with variant dedup); entities -> recurring named PERSON/ORG;
  scenes -> production structure. Validated on all 108 videos.
- **Entity grounding** (`scripts/ground_entities_llm.py`): top-k Wikipedia candidates +
  transcript context + broadcast date -> LLM disambiguation on aristotle vLLM, with abstention.
  Drops disambiguation pages; era reasoning rejects anachronisms. Running over all 108.
- **Need-down generator** (`scripts/generate_qa_needdown.py`): three-axis targets with a
  why/how skew; types = subject, interpretive (why/how/comparison), factual, cataloging,
  two-hop grounding-document (era + uniqueness + necessity gates). Validated: why/how share
  46.7% on the pilot vs 6.5% v5.2 baseline; 15/15 LLM questions parsed.
- **TODO this iteration:** the exploration/retrieval type ("which segments discuss X",
  "find interviews with scientists on climate"), the highest-value type and the one that
  exercises the agent's tool-chaining. Needs cross-segment retrieval over salience + grounding.

## B. Curation / QC gates (to build)

Three gates, each combining motivations from several QG benchmarks. They map to the three
ways a question fails: it does not require the video, the answer is not grounded, or the
answer set is unfair. (Gates 1 and 2 can be collapsed into a single "evidence gate" if a
two-gate framing reads better.)

**Gate 1 - Necessity: graded parametric-answerability annotation (NOT a hard single-model filter).**
- World-knowledge "leakage" is a property of (question x model), not of the question alone, and
  it scales with model size. A single-model hard gate over-prunes: on the pilot video, llama3.3
  alone flagged 65% of self-contained questions as leaky and left only 5 survivors, while a
  multi-model panel found 10 questions that NO model answers blind plus 5 leaky only to a few.
- So run a PANEL of models (sizes x families) blind (free text, no options), score each with the
  shared matcher, and record a per-question `blind_score` (fraction of the panel that answers it)
  plus the per-model profile. This is dataset METADATA / stratification, not a delete.
- Strata: robust (no model answers blind), leaky-to-few (large models only), leaky-to-some,
  trivial (even small models answer). Only `trivial` is a removal candidate; "leaky-to-large"
  questions are VALUABLE -- they are where grounded retrieval beats a large model's parametric
  recall (the thesis claim).
- Observed scaling (pilot, 8-model panel): small ~8-12B 10%, medium ~26-30B 13%, large 70B 38%.
- Note: still measure per-modality necessity for the (small) cross-modal slice (transcript-only /
  OCR-only must fail). And caution: thinking models (Qwen) need a high token budget or they
  under-answer and understate leakage.
- Motivations combined: VBenchComp / EgoSchema / Neptune (blind baseline) but used as a graded
  baseline distribution rather than a binary filter; the parametric-baseline framing of the thesis.

**Gate 2 - Groundedness: the answer must be entailed by a localized clue span.**
- Every kept item carries a marked evidence span; the answer must be entailed by that span
  alone (NLI or span-only LLM). Report acc.@IoU and grounded accuracy beside raw accuracy.
- Motivations combined: CG-Bench (clue-grounded QAC triplets); Source2Synth's curation
  operation, inverted for a benchmark (keep only if grounded and hard, not merely answerable).

**Gate 3 - Answer-set integrity: one defensible answer, fair distractors.**
- Exactly one defensible answer (for two-hop, exclude the `also_true` values). Distractors are
  type-consistent and length-matched; a lightweight surface classifier on (question, options)
  must NOT recover the gold above chance.
- Motivations combined: SWAG / AFLite (adversarial filtering of stylistic artifacts);
  answer-only / length / word-overlap diagnostics.

**Cross-cutting practices (applied across the gates, not gates themselves):**
- Generate-then-curate framing (Source2Synth), curation objective inverted for evaluation.
- Judge debiasing: separate generator and verifier model families (no self-preference);
  randomize MC option order.

### Paraphrasing (cross-cutting, not a gate)

Paraphrasing threads through generation and all three gates with four distinct jobs. It is
NOT a separate gate; each use plugs into an existing stage.

1. **Enrich -> deeper questions (generation).** Dense-paraphrase / enrich the index layers
   (transcript + OCR + scene captions) to make implicit roles, subevents, and cross-layer
   links explicit, so the generator can ask why/how and multi-hop questions instead of surface
   lookups. This is our group's own method: Dense Paraphrasing (Tu et al., IWCS 2023) +
   Competence-based Question Generation (Tu et al., COLING 2022).
2. **Decontaminate -> less leakage (generation, feeds Gate 1).** Rewrite the question to
   minimize surface-word overlap with its evidence span (QCLO-style synonym replacement), so
   the word-matching shortcut cannot solve it. Pairs with Gate 1's blind filter.
3. **Gate via consistency (Gate 2).** Round-trip validation (Alberti et al., ACL 2019; the
   same operation as Source2Synth's imputation) plus a paraphrase-consistency check: drop
   questions whose answer flips across paraphrases as ill-posed.
4. **Match -> robust answers (Gate 3 / scoring).** Normalize answers and accept a small
   paraphrase set of valid surface forms (ANLS* / VQA normalization), so variant names, titles,
   and dates from archival footage are not scored wrong.

Summary: enrich to go deeper, paraphrase the question to decontaminate, round-trip/consistency
to gate, paraphrase the answer to match. Note that among cited benchmarks only Source2Synth
(imputation) and Neptune (decoy rewriting) do paraphrase-adjacent work; CG-Bench, EgoSchema,
VBenchComp, SWAG/AFLite, and MMStar do not.

## C. Issues targeted for v6

Each row: the observed problem, its evidence, root cause, and the v6 mechanism that addresses
it. This is the explicit punch-list the iteration is designed to fix.

### C.1 Question value / interestingness
- **Factoid dominance.** Evidence: v5.2 is 92-95% who/what/when/where, only 6.5% why/how
  (v5.1 4.2%); "what" alone ~54-59%. Root cause: evidence-up generation extracts surface
  facts; no information-need model. v6: need-down generation, three-axis taxonomy, why/how
  skew (pilot 46.7%).
- **Surface-detail trivia** (clothing colors, cymbal brands). Root cause: all extractable
  facts treated as equally question-worthy. v6: salience gate (questions only about salient
  segments/participants/claims).
- **Verification-artifact questions** (yes/no "does the visual confirm X"). Root cause:
  scene-reconciliation prompt leaking into the question set. v6: those are not question types
  in the spec; generation is from typed cells only.
- **Interstitial/commercial-as-substance** (e.g. "New Beefresh"). Root cause: no segment-role
  filter. v6: salience excludes promo/credits roles from content cells.

### C.2 Leakage / shortcuts
- **Parametric answerability.** Evidence: no-tools blind floor 56.3% (random 25%); 73% of an
  agent's "correct" answers are ungrounded lucky guesses (grounded MC 17.6% vs raw 65.1%).
  v6: Gate 1 (necessity, blind ensemble); Gate 2 (groundedness, clue-span entailment).
- **Length/position bias.** Evidence: longest-option heuristic 47.3% on v5. v6: Gate 3
  (answer-set integrity, adversarial + length balance); option-order randomization (debiasing).
- **ASR-mangled proper nouns as gold** (Jokoman/Giacomin). Root cause: verifier accepts ASR
  spelling. v6: entity grounding canonicalizes names; clue-span entailment.

### C.3 Cross-modal integrity
- **Cross-modal faking.** Evidence: ~50% of [speech, visual] rows answerable from speech
  alone; verifier rationale accepts modality even when one suffices. v6: Gate 1 (per-modality
  necessity) replaces the gamed LLM `modality_fit`; cross-modal is a typed minority with a
  necessity test, not a tax on every question.
- **Modality contrivance** (visual descriptor bolted onto a speech fact). Same fix.

### C.4 Evidence / grounding integrity
- **Visual hallucination from single-frame captions** (~28% VLM mismatch). v6: scene-summary
  reconciliation (kept from v5.2) + clue-span entailment; fine-grained single-frame claims
  gated.
- **Long-window narrow questions** (99% single-moment in large windows). Root cause: window
  scale not tied to question scope. v6: anchor scale is set per question type; multi-moment
  is required only where the type demands it.
- **Grounding precision** (NEW). Evidence: naive top-1 grounder ~50% precision on news, ~0 on
  fiction, with false facts. v6: context-aware LLM disambiguation with abstention (built);
  ~43% grounded / 51% abstain on the full run, 0 throttling.
- **Two-hop fact hazards** (NEW). Evidence: facts can be non-unique (Krim multiple employers)
  or anachronistic (Woodruff -> Duke for a 1985 broadcast). v6: prefer time-invariant facts,
  take primary value with `also_true` exclusion, entity era-consistency, necessity name-hiding.

### C.5 Coverage / data
- **v5.2 coverage shrinkage.** Evidence: v5.2 = 95 videos vs v5.1 = 103; 10 real-content
  videos dropped (including the pilot cpb-507-154dn40c26). Root cause: strict triage dropped
  videos with no passing questions. v6: generate over the full 108-video corpus; investigate
  the 10 drops before finalizing.
- **Stub / thin indexes.** Evidence: 5 videos with ~0 entities, near-empty ASR. v6: flag and
  either re-process or exclude explicitly (no silent loss).
- **OCR under-producing entities.** Evidence: OCR contributed only 1,089 entity mentions vs
  ASR 20,435. Root cause: chyron/slate text barely entering the entity layer. v6: revisit the
  chyron/OCR -> entity path (chyrons are the richest name source).

### C.6 Process / methodology
- **Self-preference bias.** Evidence: v5.x used the same model (Qwen3.5-27B) for generation
  and verification. v6: cross-cutting debiasing decouples generator and verifier families.
- **Taxonomy not used in generation.** Evidence: `task_family` is 85% "canonical_field", a
  generation artifact, not the two-axis taxonomy. v6: the three-axis label is the generation
  target and the row schema.
- **Scaling before validation.** v6 rule: do not scale generation until a human-rated pilot
  shows interestingness up AND blind floor down.

## D. Acceptance criteria

- Per-cell coverage hits a target distribution (skewed to subject/interpretive + why/how;
  real allocation of exploration; cataloging present but not dominant).
- Blind no-video floor near chance; grounded accuracy reported alongside raw.
- Length/position heuristics at chance.
- Human interest rating on a sample (reviewer app) above a threshold.
- Decontamination note: AAPB obscurity resists pretraining leakage.

## E. Coverage and corpus

Full 108 indexed videos (grounding running on all 108). The ~3 promos/bumpers yield ~0
content and are excluded explicitly; the other ~105 are in scope.

## F. Pilot-before-scale

The recurring failure across v3-v5.2 was scaling generation before validating quality. v6
gates a full run behind a human-rated pilot on a handful of NewsHour episodes that must clear
both the interest rating and the blind-floor drop.

## Related-work mapping (citations)

- CG-Bench (ICLR 2025): clue-grounded evaluation, text-only filter, difficulty filter.
- EgoSchema (NeurIPS 2023): blind filter, temporal certificate.
- VBenchComp (2025): LLM-answerable taxonomy / blind diagnosis.
- Neptune (2024): caption -> QAD -> blind-vote pipeline (closest architectural analog).
- MVP / Minimal Video Pairs (2025): contrastive leakage-proof items.
- SWAG (EMNLP 2018) / AFLite (AAAI 2020): adversarial distractor / artifact removal.
- Source2Synth (2025): generate-then-curate grounded in real sources (inverted for eval).
- VideoZeroBench (2026): answer + evidence verification collapses accuracy.
- TVQA modality-bias study (2020): per-modality blind baselines.
- Duff & Johnson (2002), Webb DOK, Tu et al. CB-QG (COLING 2022): the three axes.
- Tu et al. Dense Paraphrasing (IWCS 2023) + CB-QG (COLING 2022): enrichment-paraphrase for
  deeper why/how and multi-hop questions (our own method).
- QCLO (arXiv 2109.11256): question rewriting to reduce question-evidence lexical overlap
  (decontamination).
- Alberti et al. Roundtrip Consistency (ACL 2019): generate-then-re-answer answerability gate.
- mQG (EMNLP 2023): diversity-by-recursion QG with an answerability filter.
- ANLS* / VQA answer normalization: robust answer matching for surface variants.

## Open questions / risks

- Exploration/retrieval answers are sets of segments, harder to verify than spans.
- Grounding skews to famous (leaky) entities; obscure local figures abstain. Two-hop must
  force a non-trivial video hop.
- Need-down generation depends on chapter quality (97/108 have chapters).
- Why/how cells sometimes blur (LLM produces why-flavored "how"); needs prompt tightening.
