# v6.0 release candidate - machine review report

Full-coverage two-stage review: 13 independent reviewer agents over all
preview rows (evidence-grounded rubric), then adversarial verification of
every complaint against the full evidence window. Human sample review
(fast_review, port 8782) ratifies before release.

- kept: 796 | excluded: 62
- disposition: {'pass': 758, 'pass_after_verification': 35, 'excluded': 62, 'unreviewed': 3}
- families: {'speech': 603, 'visual': 160, 'exploration': 33}
- blind strata (kept): {'robust': 584, 'leaky-to-few': 119, 'leaky-to-some': 60, 'n/a': 33}

## Excluded rows

- `v6-301701ac33-008` [GARBLED] 'Creole' as traded item looks ASR-garbled
- `v6-6a7b4b3ea1-003` [LEAK] Question names Gleidman then asks who is responsible; answer is Gleidman.
- `v6-9b95482ca4-001` [GARBLED] Name/title garbled ASR, likely conflated with Shevardnadze.
- `v6-9b95482ca4-005` [MISMATCH] answer explains what to look for, not why encouraged
- `v6-a1378bcd5c-004` [GARBLED] ASR-garbled name 'Shama Lan' vs excerpt 'Shaman Ann' baked into answer
- `v6-ce9a297126-007` [UNSUPPORTED] excerpt lacks any mention of 'religion'
- `v6-1279b012b8-003` [MISMATCH] question says four animals but evidence names five
- `v6-0169b45d48-001` [UNSUPPORTED] names James/Ernie Branch absent from excerpt
- `v6-da2a0fe492-001` [CONFLATION] question ties scene to Maya Angelou, not evident in excerpt
- `v6-e708482fb1-000` [CONFLATION] 'Three weeks' modifies Goulart's trip, not Gordon's absence.
- `v6-b95295228e-003` [UNSUPPORTED] speaker name Henry Camp not shown in excerpt
- `v6-b95295228e-007` [GARBLED] Speaker name 'Henry Cam' not shown/supported in evidence, likely ASR mangle.
- `v6-b95295228e-009` [GARBLED] ASR mangled Henry Kamm as 'Henry Cam' in question/gold
- `v6-d2204afef6-000` [UNSUPPORTED] Evidence excerpt is about Quayle/Clintons; no lawyer quote from Barbara Bush.
- `v6-d2204afef6-002` [DEIXIS] "this action" unresolved; not self-contained without prior context.
- `v6-d2204afef6-010` [UNSUPPORTED] Excerpt shown never mentions Barbara Bush or quote.
- `v6-30c466bae7-003` [TRIVIA] sponsor underwriting ad copy, not editorial content
- `v6-30c466bae7-010` [CONFLATION] water pressure line not clearly attributed to Blanco
- `v6-b494468686-000` [UNSUPPORTED] 5 named speakers not attributed to quote in evidence
- `v6-fab8913b45-006` [CONFLATION,GARBLED] Wizard of Oz dialogue misattributed to Marshall Goldman; mangled name
- `v6-a02f8a4d03-008` [UNSUPPORTED,CONFLATION] Excerpt is about a visa-fraud indictment, not Sullivan/South Africa/Iran-Contra.
- `v6-e910571ac6-000` [GARBLED] 'broadcast on 9725' is id fragment, not a date
- `v6-c2df6d1e89-002` [TRIVIA] answer sourced from Smith Barney ad testimonial, not news
- `v6-8fda9d1ec4-000` [DEIXIS] "The speaker" is unidentified/unresolved referent.
- `v6-8fda9d1ec4-003` [DEGENERATE] answer 'Pacific Fleet regs' is circular, near content-free
- `v6-8fda9d1ec4-008` [CONFLATION] Evidence shows Serb checkpoint, not 'American soldiers at Terminator'.
- `v6-512e6593b0-004` [TRIVIA] sponsor underwriting ad copy, not editorial content
- `v6-8caebf6903-011` [GARBLED] Nazhov likely ASR-garbled, distinct Najaf named nearby
- `v6-b21d9905dc-000` [DEGENERATE] Answer is a vague, near-circular phrase, not a concrete situation.
- `v6-3ad921fcd6-002` [UNSUPPORTED] Excerpt never names the speaker as Senator Dole.
- `v6-843cd04590-001` [DEGENERATE] "Other parts of Iraq" is a vague non-answer to "what region."
- `v6-a65dc7edcf-003` [MISMATCH] question lists five segment titles, malformed/template leak
- `v6-a65dc7edcf-006` [DEIXIS] "This Wednesday evening" is unresolved relative time.
- `v6-167673f29f-000` [GARBLED] ASR name varies Duset/Doucet/Dusette across excerpt
- `v6-432196bd5b-000` [DEIXIS] 'Speaker 5' is an unresolved, non-self-contained label.
- `v6-432196bd5b-007` [DEIXIS] "Speaker 5" is an anonymized diarization label, not self-contained.
- `v6-432196bd5b-010` [DEIXIS] "Speaker 5" label is not self-contained/resolvable.
- `v6-097a8ff5e3-004` [GARBLED] George Schulte is ASR garble of conductor Georg Solti
- `v6-097a8ff5e3-007` [DEGENERATE] Answer restates claim, doesn't explain why
- `v6-62138b60c9-000` [GARBLED] name inconsistent: Prince ICL vs Prince Asiel in same excerpt
- `v6-62138b60c9-008` [UNSUPPORTED] Excerpt lacks speaker name and quoted location entirely.
- `v6-7261698190-000` [GARBLED] 'broadcast from 7399' is id fragment, not a date
- `v6-7261698190-008` [GARBLED] Question references garbled internal ID 'broadcast of 7399 from 7399'.
- `v6-8f80c3c2ea-001` [UNSUPPORTED] Feb 28 1980 date not present anywhere in evidence
- `v6-3782846bc7-003` [DEIXIS] 'This country' is unresolved; no country named in evidence shown.
- `v6-ea79b1a91c-003` [TRIVIA] generic filler address term, not archivable fact
- `v6-0587671f14-002` [DEGENERATE] answer 'These things' is a near-empty circular lyric fragment
- `v6-0587671f14-010` [GARBLED] Speaker name 'Darliwa' absent from evidence
- `v6-bfb42eb6d8-006` [DEGENERATE] answer is a dangling fragment, not a standalone reason
- `v6-aa46dfce16-004` [UNSUPPORTED] named panelists in question absent from excerpt
- `v6v-e708482fb1-002` [UNSUPPORTED] 'Folha' is isolated OCR fragment, not clearly named newspaper
- `v6v-272c8b8e13-002` [CONFLATION] 'James Bond novel' text wrongly tied to Smiley's People
- `v6v-c2df6d1e89-001` [TRIVIA] unrelated software UI id no researcher would seek
- `v6v-e52df4d408-000` [GARBLED] OCR text garbled, answer arbitrarily excludes Coalition partners
- `v6v-b21d9905dc-002` [GARBLED] Gold answer 'TAMMY FAIE' is OCR-mangled Tammy Faye
- `v6v-5d2bc841e6-001` [GARBLED] byline 'Michael Dorni' likely OCR-garbled reporter name
- `v6v-26d9ef8db9-001` [GARBLED] on-screen text nonsensical, answer mismatches framing
- `v6v-8e9e456904-004` [DEIXIS] 'the chyron' unresolved, no broadcast context given
- `v6v-097a8ff5e3-000` [GARBLED] 'Card Pub of 1916 Opera' is garbled OCR baked into gold answer.
- `v6v-2673bfa56d-000` [GARBLED] price '3-1' is OCR-mangled, baked verbatim into gold answer
- `v6x-046a36efb6-004` [MISMATCH] Retrieved speaker is a commentator, not a "political adviser."
- `v6x-f1cc51dac0-000` [MISMATCH] credit reports segment doesn't mention William Kennedy Smith

## Attribution audit addendum (2026-07-09)

Deterministic diarization/chyron audit + evidence-reading adjudication over 97 flagged attribution pairs: 57 OK, 5 garbled-name matches (kept), 3 non-person artifacts, 32 MISATTRIBUTED rows excluded (32 unique). Validated against 4 independent human catches (all 4 confirmed by the audit). Benchmark now 764 rows.
