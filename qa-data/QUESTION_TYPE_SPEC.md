# Question-Type Specification (v6 design)

Purpose: a generation blueprint that produces *interesting and useful* questions, not
cleanly-grounded trivia. Replaces the flat `task_family` field (which collapsed to ~85%
"canonical_field") with a principled three-axis design, and flips generation from
evidence-up (extract whatever fits a window) to need-down (ask what matters, then verify).

This is both the generator's target and a thesis taxonomy table.

## The three axes

1. **Information need** (Duff & Johnson 2002, archival reference practice):
   Cataloging, Factual, Subject, Interpretive.
2. **Cognitive level** (Webb DOK / DramaQA):
   L1 Identify, L2 Retrieve, L3 Integrate, L4 Analyze.
3. **Fact role / 5W1H** (event argument structure; Competence-based QG, Tu et al. COLING 2022):
   who, what, when, where, why, how. This is the *generatable* axis: it instantiates the
   abstract cells as concrete question stems by interrogating the argument slots of an event.

Most VQA taxonomies use only axis 2. Three grounded axes is the contribution; axis 3 is what
makes generation operational.

## The salience gate (applies to every question)

A question must target a *salient* element of the program, not an incidental extractable
fact. Salience is computed from existing index layers, NOT chosen at random:

- segment salience: chapter boundaries / topic segments (a segment that matters)
- participant salience: appears in chyron AND speaks substantially AND grounds to an entity
- claim/event salience: is the topic of a segment (topic model + scene/chapter summary),
  not a passing mention
- production salience: slate / credits metadata

Rule: generate 5W1H over the arguments of a *salient* event. 5W over a random detail
reproduces the current trivia.

## 5W1H weighting (the interestingness lever)

- why / how: PRIORITIZE. Cause, reason, motivation, mechanism, process. L3-L4,
  video-dependent, shortcut-resistant. These are the interesting questions.
- who / what: keep mainly for the Cataloging and Factual columns; useful but factoid-prone.
- when / where: GATE against parametric leakage. Keep when they are cataloging metadata
  ("when did this air"); reject when the answer is world knowledge ("when did the war start").

L4 questions are *compositions* of W's across salient events (e.g., "how do guests A and B
differ on why X happened"). 5W1H is the atomic vocabulary, not a straitjacket.

## Question-type catalog

| Type | Need | Level | 5W focus | Anchor scale | Source layers | Good example | Boring version to avoid |
|---|---|---|---|---|---|---|---|
| Program ID & production metadata | Cataloging | L1 | what/when/who | whole-video / slate | slate + credits KIE | "Who is credited as producer of this program?" | (n/a, inherently useful) |
| Participant inventory & roles | Cataloging | L1-L2 | who | chapter / program | chyron KIE + speaker ID + entity grounding | "Which public officials are interviewed in this episode?" | "What is the woman in the tan jacket wearing?" |
| Segment inventory | Cataloging/Subject | L2 | what | whole-video | chapter titles + summaries | "What stories does this newscast cover?" | "What is shown at 03:12?" |
| Key claim / fact stated | Factual | L2 | what/who | segment | ASR, salient claims | "What does the senator say the Financial Institutions Act will do?" | "What number does the speaker cite?" (random) |
| Subject of a segment | Subject | L2-L3 | what/why | chapter | scene/chapter summary + topic | "What is the central argument of this report on beach erosion?" | "What is shown on screen during the report?" |
| Exploration / retrieval | Subject | L3-L4 | what/who | program / corpus | multi-tool chain | "Which segments discuss education policy?" / "Find interviews with scientists on climate." | (currently absent; this is the highest-value missing type) |
| Framing / stance | Interpretive | L3-L4 | how/why | segment | ASR + structure | "How does the anchor frame the budget debate?" | "Is the visual focus on the speaker?" (yes/no artifact) |
| Comparison / relationship | Interpretive | L4 | how/why | multi-segment | ASR across segments | "How do the two analysts differ on why support is lower among Catholic voters?" | "What colors are the two people wearing?" |
| Context via external knowledge | Interpretive | L3 | who/what/why | participant | Wikidata grounding | "What is the professional background of the CDC guest?" | (n/a if grounded to entity) |
| Genuine cross-modal | Factual | L2-L3 | who | moment | chyron/OCR + ASR, with necessity test | "Who, identified only by the chyron, makes the claim about driver error?" | "Who is the blonde woman discussing branching laws?" (visual is decoration) |

## Target distribution (starting point, tune on the pilot)

- Skew toward why/how and toward Subject/Interpretive (the gap in current data).
- Include a real allocation of Exploration/retrieval (currently 0%).
- Keep Cataloging present but not dominant (it is useful but easy).
- Cross-modal is a *type with a necessity test*, not a tax on every question.

## Generation procedure (need-down)

1. Build the salience map for the video from existing layers (chapters, participants,
   claims/topics, production metadata).
2. For each target cell (need × level × 5W role), pick a salient element at the right anchor
   scale and compose the question (CB-QG: interrogate the event's argument slots).
3. Verify TWO things: grounding (answer entailed by a localized clue span) AND salience
   (the targeted element is actually central, not incidental).
4. Generate distractors type-consistent and length-matched; ensure exactly one defensible
   answer (NLI check against the clue span).

## What changes vs the current pipeline

- Window sampling is no longer the unit of generation; the salience map is. (Current data is
  96% local/medium windows, which structurally forces factoids.)
- `task_family` is replaced by the three-axis label (need, level, 5W role).
- The verifier's self-reported `modality_fit` / `parametric_risk` opinions are replaced by
  empirical checks (blind solver, per-modality necessity, clue-span entailment).
- Salience becomes a first-class gate; cross-modal becomes a typed minority, not the default.
