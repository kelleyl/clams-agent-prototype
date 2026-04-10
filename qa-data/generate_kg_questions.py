#!/usr/bin/env python3
"""Generate KG-grounded QA pairs inspired by INQUIRER/DramaQA cognitive hierarchy.

Replaces timestamp-grounded retrieval questions with understanding questions
that require reasoning over the knowledge graph (entities, relations, speakers,
temporal structure).

Difficulty levels (adapted from DramaQA):
  L1: Simple recall — single fact from the KG
  L2: Multi-cue inference — combining multiple facts
  L3: Temporal reasoning — tracking changes across segments
  L4: Causal/analytical — why/how questions requiring understanding

No questions are grounded to specific timestamps. All questions should be
answerable by reasoning over the indexed content, not by looking at a clock.
"""
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

random.seed(42)


@dataclass
class QA:
    id: str
    question: str
    answer: dict
    category: str
    difficulty: str  # L1, L2, L3, L4
    video_id: str
    tags: list = field(default_factory=list)
    evidence_segments: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d = {k: v for k, v in d.items() if v}
        return d


class KGQuestionGenerator:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self._counter = 0

    def _id(self, cat: str) -> str:
        self._counter += 1
        return f"qa-{cat}-{self._counter:05d}"

    def generate_all(self) -> list:
        entries = []
        for f in sorted(self.index_dir.glob("*.json")):
            d = json.load(open(f))
            entries.extend(self._generate_for_video(d))
        return entries

    def _generate_for_video(self, index: dict) -> list:
        segs = index.get("segments", [])
        if len(segs) < 2:
            return []
        vid = index["video_id"]
        speaker_names = index.get("speaker_names", {})
        broadcast_date = index.get("broadcast_date", "")

        entries = []
        entries.extend(self._level1_factual(vid, segs, speaker_names, broadcast_date, index))
        entries.extend(self._level2_multi_cue(vid, segs, speaker_names))
        entries.extend(self._level3_temporal(vid, segs, speaker_names))
        entries.extend(self._level4_causal(vid, segs, speaker_names, broadcast_date))
        return entries

    # ================================================================
    # L1: Simple recall — single fact retrieval from KG
    # ================================================================
    def _level1_factual(self, vid, segs, speaker_names, broadcast_date, index=None):
        entries = []

        # --- Who appears in this video? ---
        all_persons = set()
        for s in segs:
            for e in s.get("named_entities", []):
                if e["type"] == "PERSON" and len(e["text"]) > 3:
                    all_persons.add(e["text"])
        if len(all_persons) >= 3:
            entries.append(QA(
                id=self._id("L1"), category="entity_recall",
                question="Who are the people mentioned in this broadcast?",
                answer={"people": sorted(all_persons)[:15],
                        "total": len(all_persons)},
                difficulty="L1", video_id=vid, tags=["entity", "person"],
            ))

        # --- What organizations are discussed? ---
        all_orgs = set()
        for s in segs:
            for e in s.get("named_entities", []):
                if e["type"] == "ORG" and len(e["text"]) > 3:
                    all_orgs.add(e["text"])
        if len(all_orgs) >= 3:
            entries.append(QA(
                id=self._id("L1"), category="entity_recall",
                question="What organizations are mentioned in this broadcast?",
                answer={"organizations": sorted(all_orgs)[:15],
                        "total": len(all_orgs)},
                difficulty="L1", video_id=vid, tags=["entity", "org"],
            ))

        # --- What locations are referenced? ---
        all_locs = set()
        for s in segs:
            for e in s.get("named_entities", []):
                if e["type"] in ("GPE", "LOC") and len(e["text"]) > 3:
                    all_locs.add(e["text"])
        if len(all_locs) >= 3:
            entries.append(QA(
                id=self._id("L1"), category="entity_recall",
                question="What places or locations are discussed in this broadcast?",
                answer={"locations": sorted(all_locs)[:15]},
                difficulty="L1", video_id=vid, tags=["entity", "location"],
            ))

        # --- What is [identified speaker]'s role? ---
        for spk, name in speaker_names.items():
            idx_details = (index or {}).get("speaker_details", {}).get(spk, {})
            title = idx_details.get("title", "")
            if title:
                entries.append(QA(
                    id=self._id("L1"), category="speaker_identity",
                    question=f"Who is {name} and what is their role in this broadcast?",
                    answer={"name": name, "title": title,
                            "source": idx_details.get("source", "")},
                    difficulty="L1", video_id=vid, tags=["speaker", "identity"],
                ))

        # --- What on-screen text appears? (OCR recall) ---
        ocr_texts = [s.get("ocr_text", "").strip() for s in segs if s.get("ocr_text", "").strip()]
        if ocr_texts:
            # Deduplicate and pick interesting ones
            unique_ocr = list(set(t[:100] for t in ocr_texts if len(t) > 10 and not t.startswith("The image")))[:5]
            if unique_ocr:
                entries.append(QA(
                    id=self._id("L1"), category="text_recall",
                    question="What on-screen text or graphics appear in this broadcast?",
                    answer={"texts": unique_ocr},
                    difficulty="L1", video_id=vid, tags=["ocr", "text"],
                ))

        return entries

    # ================================================================
    # L2: Multi-cue inference — combining facts from multiple segments
    # ================================================================
    def _level2_multi_cue(self, vid, segs, speaker_names):
        entries = []

        # --- Which speakers discussed [entity]? ---
        entity_speakers = defaultdict(set)
        for s in segs:
            spk = s.get("primary_speaker", "")
            display = speaker_names.get(spk, spk) if spk else ""
            if not display:
                continue
            for e in s.get("named_entities", []):
                if e["type"] in ("PERSON", "ORG", "GPE", "EVENT") and len(e["text"]) > 3:
                    entity_speakers[e["text"]].add(display)

        multi = [(ent, spks) for ent, spks in entity_speakers.items() if len(spks) >= 2]
        for ent, spks in random.sample(multi, min(5, len(multi))):
            entries.append(QA(
                id=self._id("L2"), category="cross_speaker",
                question=f'Which speakers discussed "{ent}" in this broadcast?',
                answer={"entity": ent, "speakers": sorted(spks)},
                difficulty="L2", video_id=vid, tags=["speaker", "entity", "multi-cue"],
            ))

        # --- What topics did [speaker] cover? ---
        speaker_topics = defaultdict(lambda: defaultdict(int))
        for s in segs:
            spk = s.get("primary_speaker", "")
            name = speaker_names.get(spk)
            if not name:
                continue
            for e in s.get("named_entities", []):
                if e["type"] in ("PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART") and len(e["text"]) > 3:
                    speaker_topics[name][e["text"]] += 1

        for name, topics in speaker_topics.items():
            top = sorted(topics.items(), key=lambda x: -x[1])[:10]
            if len(top) >= 3:
                entries.append(QA(
                    id=self._id("L2"), category="speaker_topics",
                    question=f"What topics and entities did {name} discuss?",
                    answer={"speaker": name,
                            "topics": [{"entity": e, "mentions": c} for e, c in top]},
                    difficulty="L2", video_id=vid, tags=["speaker", "topics", "multi-cue"],
                ))

        # --- What actions are described involving [entity]? ---
        entity_actions = defaultdict(list)
        for s in segs:
            for r in s.get("relations", []):
                subj, pred, obj = r
                for e in s.get("named_entities", []):
                    if e["text"] in subj or e["text"] in obj:
                        entity_actions[e["text"]].append(r)

        interesting = [(e, acts) for e, acts in entity_actions.items()
                       if len(acts) >= 2 and len(e) > 4]
        for ent, actions in random.sample(interesting, min(5, len(interesting))):
            unique_acts = list({tuple(a) for a in actions})[:5]
            entries.append(QA(
                id=self._id("L2"), category="entity_actions",
                question=f'What actions or events involve "{ent}" in this broadcast?',
                answer={"entity": ent,
                        "actions": [{"subject": s, "predicate": p, "object": o}
                                    for s, p, o in unique_acts]},
                difficulty="L2", video_id=vid, tags=["relation", "entity", "multi-cue"],
            ))

        # --- What entities co-occur across segments? ---
        cooccur = defaultdict(int)
        for s in segs:
            ents = list({e["text"] for e in s.get("named_entities", [])
                         if e["type"] in ("PERSON", "ORG", "GPE") and len(e["text"]) > 3})
            for i, e1 in enumerate(ents):
                for e2 in ents[i + 1:]:
                    cooccur[(min(e1, e2), max(e1, e2))] += 1

        frequent_pairs = [(pair, c) for pair, c in cooccur.items() if c >= 2]
        frequent_pairs.sort(key=lambda x: -x[1])
        for (e1, e2), count in frequent_pairs[:5]:
            entries.append(QA(
                id=self._id("L2"), category="entity_cooccurrence",
                question=f'How are "{e1}" and "{e2}" connected in this broadcast?',
                answer={"entity_1": e1, "entity_2": e2,
                        "co_occurrences": count,
                        "explanation": f'"{e1}" and "{e2}" appear together in {count} segments'},
                difficulty="L2", video_id=vid, tags=["entity", "cooccurrence", "multi-cue"],
            ))

        return entries

    # ================================================================
    # L3: Temporal reasoning — tracking across the video
    # ================================================================
    def _level3_temporal(self, vid, segs, speaker_names):
        entries = []

        # --- How does the topic shift across the broadcast? ---
        # Track dominant entities per segment chunk
        chunk_size = max(1, len(segs) // 4)
        chunks = [segs[i:i + chunk_size] for i in range(0, len(segs), chunk_size)]
        if len(chunks) >= 3:
            chunk_topics = []
            for chunk in chunks:
                topic_counts = defaultdict(int)
                for s in chunk:
                    for e in s.get("named_entities", []):
                        if e["type"] in ("PERSON", "ORG", "GPE", "EVENT") and len(e["text"]) > 4:
                            topic_counts[e["text"]] += 1
                top = sorted(topic_counts.items(), key=lambda x: -x[1])[:5]
                chunk_topics.append([e for e, _ in top])

            entries.append(QA(
                id=self._id("L3"), category="topic_progression",
                question="How do the topics discussed change over the course of this broadcast?",
                answer={"progression": [
                    {"section": f"Part {i + 1}", "topics": topics}
                    for i, topics in enumerate(chunk_topics) if topics
                ]},
                difficulty="L3", video_id=vid, tags=["temporal", "topic-shift"],
            ))

        # --- Speaker transitions: who speaks and in what order? ---
        speaker_order = []
        prev_spk = None
        for s in segs:
            spk = s.get("primary_speaker")
            if spk and spk != prev_spk:
                name = speaker_names.get(spk, spk)
                speaker_order.append(name)
                prev_spk = spk

        if len(speaker_order) >= 4:
            # Deduplicate consecutive repeats
            deduped = [speaker_order[0]]
            for s in speaker_order[1:]:
                if s != deduped[-1]:
                    deduped.append(s)

            entries.append(QA(
                id=self._id("L3"), category="speaker_flow",
                question="What is the sequence of speakers in this broadcast?",
                answer={"speaker_sequence": deduped[:20],
                        "total_transitions": len(deduped)},
                difficulty="L3", video_id=vid, tags=["speaker", "temporal", "sequence"],
            ))

        # --- Entity tracking: when does [entity] first/last appear? ---
        entity_spans = defaultdict(lambda: {"first": float("inf"), "last": 0, "count": 0})
        for s in segs:
            for e in s.get("named_entities", []):
                if e["type"] == "PERSON" and len(e["text"]) > 4:
                    es = entity_spans[e["text"]]
                    es["first"] = min(es["first"], s["start_ms"])
                    es["last"] = max(es["last"], s["end_ms"])
                    es["count"] += 1

        recurring = [(e, sp) for e, sp in entity_spans.items()
                     if sp["count"] >= 3 and sp["last"] - sp["first"] > 60000]
        for ent, span in random.sample(recurring, min(3, len(recurring))):
            entries.append(QA(
                id=self._id("L3"), category="entity_tracking",
                question=f'When and how often is "{ent}" discussed throughout this broadcast?',
                answer={"entity": ent, "mentions": span["count"],
                        "span_seconds": round((span["last"] - span["first"]) / 1000),
                        "explanation": f'"{ent}" is mentioned {span["count"]} times across {round((span["last"] - span["first"]) / 1000)}s of the broadcast'},
                difficulty="L3", video_id=vid, tags=["entity", "temporal", "tracking"],
            ))

        # --- What visual content accompanies the discussion of [topic]? ---
        for s in segs:
            ents = [e["text"] for e in s.get("named_entities", [])
                    if e["type"] in ("PERSON", "ORG", "GPE") and len(e["text"]) > 4]
            caption = s.get("visual_caption", "")
            asr = s.get("asr_transcript", "")
            if ents and caption and asr and len(caption) > 30:
                main_ent = ents[0]
                entries.append(QA(
                    id=self._id("L3"), category="cross_modal",
                    question=f'What is shown on screen while "{main_ent}" is being discussed?',
                    answer={"entity": main_ent,
                            "visual": caption[:200],
                            "speech": asr[:200]},
                    difficulty="L3", video_id=vid, tags=["cross-modal", "visual", "temporal"],
                ))
                break  # One per video

        return entries

    # ================================================================
    # L4: Causal/analytical — why, how, comparative reasoning
    # ================================================================
    def _level4_causal(self, vid, segs, speaker_names, broadcast_date):
        entries = []

        # --- What is the main story/topic of this broadcast? ---
        all_ents = defaultdict(int)
        for s in segs:
            for e in s.get("named_entities", []):
                if e["type"] in ("PERSON", "ORG", "GPE", "EVENT") and len(e["text"]) > 4:
                    all_ents[e["text"]] += 1
        top_ents = sorted(all_ents.items(), key=lambda x: -x[1])[:10]
        all_rels = []
        for s in segs:
            for r in s.get("relations", []):
                all_rels.append(r)
        top_rels = list({tuple(r) for r in all_rels})[:10]

        if top_ents and top_rels:
            entries.append(QA(
                id=self._id("L4"), category="narrative_summary",
                question="What are the main stories covered in this broadcast and how are they connected?",
                answer={
                    "key_entities": [{"entity": e, "mentions": c} for e, c in top_ents],
                    "key_relations": [{"subject": s, "predicate": p, "object": o}
                                      for s, p, o in top_rels[:8]],
                    "broadcast_date": broadcast_date,
                },
                difficulty="L4", video_id=vid, tags=["narrative", "summary", "analytical"],
            ))

        # --- Do different speakers agree or disagree? ---
        if len(speaker_names) >= 2:
            spk_items = list(speaker_names.items())
            for i in range(min(3, len(spk_items))):
                for j in range(i + 1, min(4, len(spk_items))):
                    spk_a, name_a = spk_items[i]
                    spk_b, name_b = spk_items[j]

                    # Get their relation triples
                    rels_a = set()
                    rels_b = set()
                    for s in segs:
                        sp = s.get("primary_speaker", "")
                        for r in s.get("relations", []):
                            if sp == spk_a:
                                rels_a.add(tuple(r))
                            elif sp == spk_b:
                                rels_b.add(tuple(r))

                    # Find shared subjects/objects
                    subjs_a = {r[0] for r in rels_a}
                    subjs_b = {r[0] for r in rels_b}
                    shared = subjs_a & subjs_b
                    if shared:
                        entries.append(QA(
                            id=self._id("L4"), category="speaker_comparison",
                            question=f"Do {name_a} and {name_b} discuss the same subjects? How do their perspectives differ?",
                            answer={
                                "speaker_a": name_a,
                                "speaker_b": name_b,
                                "shared_subjects": sorted(shared)[:5],
                                "a_relations": [list(r) for r in list(rels_a)[:5]],
                                "b_relations": [list(r) for r in list(rels_b)[:5]],
                            },
                            difficulty="L4", video_id=vid,
                            tags=["speaker", "comparison", "analytical"],
                        ))

        # --- What is the relationship between [entity] and [entity]? ---
        # Find entity pairs connected by relations
        entity_rels = defaultdict(list)
        for s in segs:
            for r in s.get("relations", []):
                subj, pred, obj = r
                for e1 in s.get("named_entities", []):
                    for e2 in s.get("named_entities", []):
                        if e1["text"] != e2["text"] and e1["text"] in subj and e2["text"] in obj:
                            entity_rels[(e1["text"], e2["text"])].append(pred)

        connected = [(pair, preds) for pair, preds in entity_rels.items() if len(preds) >= 1]
        for (e1, e2), preds in random.sample(connected, min(3, len(connected))):
            entries.append(QA(
                id=self._id("L4"), category="entity_relationship",
                question=f'What is the relationship between "{e1}" and "{e2}" in this broadcast?',
                answer={"entity_1": e1, "entity_2": e2,
                        "relationships": list(set(preds))[:5]},
                difficulty="L4", video_id=vid,
                tags=["entity", "relationship", "analytical"],
            ))

        # --- Context-dependent: What was happening in [broadcast_date]? ---
        if broadcast_date and len(all_ents) >= 5:
            entries.append(QA(
                id=self._id("L4"), category="historical_context",
                question=f"Based on this {broadcast_date} broadcast, what were the major events or issues being reported?",
                answer={
                    "date": broadcast_date,
                    "key_topics": [e for e, _ in top_ents[:8]],
                    "key_events": [list(r) for r in top_rels[:5]],
                },
                difficulty="L4", video_id=vid,
                tags=["historical", "context", "analytical", "date"],
            ))

        return entries


def main():
    gen = KGQuestionGenerator("data/video_indexes")
    entries = gen.generate_all()

    # Stats
    by_diff = defaultdict(int)
    by_cat = defaultdict(int)
    for e in entries:
        by_diff[e.difficulty] += 1
        by_cat[e.category] += 1

    print(f"Total: {len(entries)} questions across {len(set(e.video_id for e in entries))} videos")
    print(f"\nBy difficulty:")
    for d in ["L1", "L2", "L3", "L4"]:
        print(f"  {d}: {by_diff[d]}")
    print(f"\nBy category:")
    for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Write
    out = Path("qa-data/raw/kg_questions.jsonl")
    with open(out, "w") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
    print(f"\nWritten to {out}")

    # Samples per level
    for level in ["L1", "L2", "L3", "L4"]:
        qs = [e for e in entries if e.difficulty == level]
        print(f"\n--- {level} samples ---")
        for q in random.sample(qs, min(3, len(qs))):
            print(f"  [{q.category}] Q: {q.question[:100]}")
            val = q.answer.get("value", q.answer.get("people", q.answer.get("topics", str(q.answer))))
            print(f"    A: {str(val)[:100]}")


if __name__ == "__main__":
    main()
