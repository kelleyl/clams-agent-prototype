#!/usr/bin/env python3
"""Generate comprehension questions that require understanding video content.

Questions are grounded in actual transcript/visual content and require reading
comprehension to answer — not database queries over entity counts.

Uses the KG (entities, relations, speakers) to identify WHAT to ask about,
but answers require understanding the actual speech and visual content.

Inspired by DramaQA cognitive hierarchy and INQUIRER's KG-guided generation.
"""
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

random.seed(42)

# Phrases that signal substantive speech content
CLAIM_MARKERS = re.compile(
    r"\b(I think|I believe|we need to|we should|we must|the problem is|"
    r"the issue is|the question is|what we need|in my view|my position|"
    r"I would argue|the fact is|the reality is|it's clear that|"
    r"we have to|I disagree|I agree|that's wrong|that's right|"
    r"the evidence shows|studies show|according to|the data shows|"
    r"the report found|experts say)\b",
    re.IGNORECASE,
)

EVENT_MARKERS = re.compile(
    r"\b(announced|signed|passed|voted|rejected|approved|killed|"
    r"arrested|attacked|invaded|withdrew|resigned|appointed|fired|"
    r"collapsed|exploded|crashed|discovered|launched|banned|"
    r"declared|denied|confirmed|revealed|warned|threatened)\b",
    re.IGNORECASE,
)

INTRO_MARKERS = re.compile(
    r"\b(tonight|this evening|our top story|we begin with|"
    r"turning now to|our next story|we turn to|joining us|"
    r"here's|now to|first up|the big story)\b",
    re.IGNORECASE,
)


@dataclass
class QA:
    id: str
    question: str
    answer: str
    answer_evidence: str  # the actual transcript passage
    category: str
    difficulty: str
    video_id: str
    tags: list = field(default_factory=list)

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v}


class ComprehensionQAGenerator:
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
        if len(segs) < 3:
            return []
        vid = index["video_id"]
        speaker_names = index.get("speaker_names", {})
        broadcast_date = index.get("broadcast_date", "")

        entries = []
        entries.extend(self._position_questions(vid, segs, speaker_names))
        entries.extend(self._event_questions(vid, segs, speaker_names))
        entries.extend(self._evidence_questions(vid, segs, speaker_names))
        entries.extend(self._comparison_questions(vid, segs, speaker_names))
        entries.extend(self._narrative_questions(vid, segs, speaker_names, broadcast_date))
        entries.extend(self._visual_context_questions(vid, segs, speaker_names))
        return entries

    def _get_speaker(self, seg, speaker_names):
        """Get display name for segment's speaker."""
        spk = seg.get("primary_speaker", "")
        return speaker_names.get(spk, "") if spk else ""

    def _find_substantive_segments(self, segs, min_words=30):
        """Find segments with enough speech content to ask about."""
        return [s for s in segs if len(s.get("asr_transcript", "").split()) >= min_words]

    # ================================================================
    # Position/stance questions: "What position did X take on Y?"
    # ================================================================
    def _position_questions(self, vid, segs, speaker_names):
        entries = []

        for seg in self._find_substantive_segments(segs, 40):
            asr = seg.get("asr_transcript", "")
            speaker = self._get_speaker(seg, speaker_names)
            if not speaker:
                continue

            # Find claims/positions in speech
            claims = list(CLAIM_MARKERS.finditer(asr))
            if not claims:
                continue

            # Extract the claim context (sentence around the marker)
            for claim in claims[:1]:  # one per segment
                start = max(0, asr.rfind(".", 0, claim.start()) + 1)
                end = asr.find(".", claim.end())
                if end == -1:
                    end = min(len(asr), claim.end() + 200)
                else:
                    end = min(end + 1, claim.end() + 300)
                passage = asr[start:end].strip()
                if len(passage) < 30:
                    continue

                # Find the topic (most prominent entity in this segment)
                ents = [e["text"] for e in seg.get("named_entities", [])
                        if e["type"] in ("ORG", "GPE", "EVENT", "WORK_OF_ART")
                        and len(e["text"]) > 3]
                topic = ents[0] if ents else None

                if topic:
                    q = f"What position did {speaker} take regarding {topic}?"
                else:
                    # Use the claim marker itself to frame the question
                    q = f"What did {speaker} argue in their remarks?"

                entries.append(QA(
                    id=self._id("position"), category="position_stance",
                    question=q,
                    answer=passage[:300],
                    answer_evidence=asr[:500],
                    difficulty="L2", video_id=vid,
                    tags=["speaker", "position", "comprehension"],
                ))
                break  # one per speaker per video

        return entries[:5]  # cap per video

    # ================================================================
    # Event questions: "What happened with X?"
    # ================================================================
    def _event_questions(self, vid, segs, speaker_names):
        entries = []

        for seg in self._find_substantive_segments(segs, 25):
            asr = seg.get("asr_transcript", "")

            events = list(EVENT_MARKERS.finditer(asr))
            if not events:
                continue

            # Get the event context
            for event in events[:1]:
                start = max(0, asr.rfind(".", 0, event.start()) + 1)
                end = asr.find(".", event.end())
                if end == -1:
                    end = min(len(asr), event.end() + 200)
                else:
                    end = min(end + 1, event.end() + 250)
                passage = asr[start:end].strip()
                if len(passage) < 25:
                    continue

                # Find involved entities
                persons = [e["text"] for e in seg.get("named_entities", [])
                           if e["type"] == "PERSON" and len(e["text"]) > 3]
                orgs = [e["text"] for e in seg.get("named_entities", [])
                        if e["type"] in ("ORG", "GPE") and len(e["text"]) > 3]

                verb = event.group(0)
                if persons and orgs:
                    q = f"What did {persons[0]} {verb} regarding {orgs[0]}?"
                elif persons:
                    q = f"What {verb} involving {persons[0]} is reported in this broadcast?"
                elif orgs:
                    q = f"What event involving {orgs[0]} is described in this broadcast?"
                else:
                    continue

                entries.append(QA(
                    id=self._id("event"), category="event_comprehension",
                    question=q,
                    answer=passage[:300],
                    answer_evidence=asr[:500],
                    difficulty="L2", video_id=vid,
                    tags=["event", "comprehension"],
                ))

        return entries[:6]

    # ================================================================
    # Evidence/reasoning questions: "What evidence did X cite?"
    # ================================================================
    def _evidence_questions(self, vid, segs, speaker_names):
        entries = []

        evidence_re = re.compile(
            r"\b(because|the reason|due to|as a result|therefore|"
            r"consequently|evidence|studies show|according to|"
            r"data shows|percent|billion|million|statistics|"
            r"report found|research shows|numbers show)\b",
            re.IGNORECASE,
        )

        for seg in self._find_substantive_segments(segs, 50):
            asr = seg.get("asr_transcript", "")
            speaker = self._get_speaker(seg, speaker_names)

            matches = list(evidence_re.finditer(asr))
            if not matches:
                continue

            # Get context around evidence
            m = matches[0]
            start = max(0, asr.rfind(".", 0, m.start()) + 1)
            end = asr.find(".", m.end())
            if end == -1:
                end = min(len(asr), m.end() + 250)
            else:
                end = min(end + 1, m.end() + 300)
            passage = asr[start:end].strip()
            if len(passage) < 30:
                continue

            topic_ents = [e["text"] for e in seg.get("named_entities", [])
                          if e["type"] in ("ORG", "GPE", "EVENT") and len(e["text"]) > 3]
            topic = topic_ents[0] if topic_ents else "the issue being discussed"

            if speaker:
                q = f"What evidence or reasoning did {speaker} provide about {topic}?"
            else:
                q = f"What evidence is cited in the discussion about {topic}?"

            entries.append(QA(
                id=self._id("evidence"), category="evidence_reasoning",
                question=q,
                answer=passage[:300],
                answer_evidence=asr[:500],
                difficulty="L3", video_id=vid,
                tags=["evidence", "reasoning", "comprehension"],
            ))

        return entries[:4]

    # ================================================================
    # Comparison questions: "How did X's view differ from Y's?"
    # ================================================================
    def _comparison_questions(self, vid, segs, speaker_names):
        entries = []

        # Group segments by speaker
        speaker_content = defaultdict(list)
        for seg in self._find_substantive_segments(segs, 30):
            spk = seg.get("primary_speaker", "")
            name = speaker_names.get(spk)
            if name:
                speaker_content[name].append(seg)

        if len(speaker_content) < 2:
            return entries

        # Find speakers who discuss overlapping topics
        speaker_topics = {}
        for name, content_segs in speaker_content.items():
            topics = defaultdict(list)
            for seg in content_segs:
                for e in seg.get("named_entities", []):
                    if e["type"] in ("ORG", "GPE", "EVENT", "PERSON") and len(e["text"]) > 3:
                        topics[e["text"]].append(seg.get("asr_transcript", "")[:200])
            speaker_topics[name] = topics

        names = list(speaker_topics.keys())
        for i in range(min(3, len(names))):
            for j in range(i + 1, min(4, len(names))):
                name_a, name_b = names[i], names[j]
                shared = set(speaker_topics[name_a].keys()) & set(speaker_topics[name_b].keys())
                shared = {t for t in shared if len(t) > 4}
                if not shared:
                    continue

                topic = random.choice(sorted(shared))
                passage_a = speaker_topics[name_a][topic][0] if speaker_topics[name_a][topic] else ""
                passage_b = speaker_topics[name_b][topic][0] if speaker_topics[name_b][topic] else ""

                if passage_a and passage_b:
                    entries.append(QA(
                        id=self._id("compare"), category="speaker_comparison",
                        question=f"How did {name_a}'s remarks about {topic} compare to {name_b}'s?",
                        answer=f"{name_a}: {passage_a[:200]}... | {name_b}: {passage_b[:200]}...",
                        answer_evidence=f"[{name_a}] {passage_a[:300]}\n[{name_b}] {passage_b[:300]}",
                        difficulty="L4", video_id=vid,
                        tags=["comparison", "speaker", "analytical"],
                    ))
                    break  # one per pair

        return entries[:3]

    # ================================================================
    # Narrative questions: "What was the story about? What happened?"
    # ================================================================
    def _narrative_questions(self, vid, segs, speaker_names, broadcast_date):
        entries = []

        # Find story introductions
        for seg in segs:
            asr = seg.get("asr_transcript", "")
            if len(asr) < 50:
                continue

            intros = list(INTRO_MARKERS.finditer(asr))
            if not intros:
                continue

            # Get the story introduction passage
            intro = intros[0]
            # Get everything after the intro marker until the next major break
            passage = asr[intro.start():min(len(asr), intro.start() + 400)].strip()
            if len(passage) < 40:
                continue

            # What story is being introduced?
            ents = [e["text"] for e in seg.get("named_entities", [])
                    if e["type"] in ("PERSON", "ORG", "GPE", "EVENT") and len(e["text"]) > 3]

            if ents:
                q = f"What story about {ents[0]} is reported in this broadcast?"
            else:
                q = "What is the main story introduced in this broadcast?"

            entries.append(QA(
                id=self._id("narrative"), category="narrative_comprehension",
                question=q,
                answer=passage[:300],
                answer_evidence=asr[:500],
                difficulty="L3", video_id=vid,
                tags=["narrative", "story", "comprehension"],
            ))

        # Overall broadcast summary question
        # Collect all substantive ASR content
        all_asr = " ".join(s.get("asr_transcript", "")[:100]
                           for s in segs if s.get("asr_transcript"))
        if len(all_asr) > 200:
            all_ents = defaultdict(int)
            for s in segs:
                for e in s.get("named_entities", []):
                    if e["type"] in ("PERSON", "ORG", "GPE") and len(e["text"]) > 3:
                        all_ents[e["text"]] += 1
            top = sorted(all_ents.items(), key=lambda x: -x[1])[:5]

            if broadcast_date:
                q = f"What were the major news stories covered in the {broadcast_date} broadcast?"
            else:
                q = "What are the main topics covered in this broadcast?"

            # Build answer from story intros and top entities
            story_passages = []
            for seg in segs[:5]:
                asr = seg.get("asr_transcript", "")
                if len(asr) > 50:
                    story_passages.append(asr[:150])

            entries.append(QA(
                id=self._id("narrative"), category="broadcast_summary",
                question=q,
                answer=" | ".join(story_passages[:3]),
                answer_evidence=" ".join(story_passages[:5]),
                difficulty="L4", video_id=vid,
                tags=["narrative", "summary", "analytical"],
            ))

        return entries[:5]

    # ================================================================
    # Visual context: "What is shown while X is discussed?"
    # ================================================================
    def _visual_context_questions(self, vid, segs, speaker_names):
        entries = []

        for seg in segs:
            asr = seg.get("asr_transcript", "")
            caption = seg.get("visual_caption", "")
            ocr = seg.get("ocr_text", "")

            if not asr or not caption or len(asr) < 30 or len(caption) < 30:
                continue
            # Skip hallucinated captions
            if caption.startswith(("The image is completely black", "The video clip is currently empty")):
                continue

            ents = [e["text"] for e in seg.get("named_entities", [])
                    if e["type"] in ("PERSON", "ORG", "GPE") and len(e["text"]) > 3]
            speaker = self._get_speaker(seg, speaker_names)

            if ents:
                topic = ents[0]
                q = f"What visual content accompanies the discussion of {topic}?"
            elif speaker:
                q = f"What is shown on screen while {speaker} is speaking?"
            else:
                continue

            answer_parts = [f"Visual: {caption[:200]}"]
            if ocr and len(ocr) > 5 and not ocr.startswith("The"):
                answer_parts.append(f"On-screen text: {ocr[:100]}")
            answer_parts.append(f"Speech: {asr[:200]}")

            entries.append(QA(
                id=self._id("visual"), category="visual_context",
                question=q,
                answer=" | ".join(answer_parts),
                answer_evidence=f"Caption: {caption[:300]}\nASR: {asr[:300]}",
                difficulty="L3", video_id=vid,
                tags=["visual", "cross-modal", "comprehension"],
            ))

        return entries[:3]


def main():
    gen = ComprehensionQAGenerator("data/video_indexes")
    entries = gen.generate_all()

    by_diff = defaultdict(int)
    by_cat = defaultdict(int)
    for e in entries:
        by_diff[e.difficulty] += 1
        by_cat[e.category] += 1

    print(f"Total: {len(entries)} questions across {len(set(e.video_id for e in entries))} videos")
    print(f"\nBy difficulty:")
    for d in ["L2", "L3", "L4"]:
        print(f"  {d}: {by_diff[d]}")
    print(f"\nBy category:")
    for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    out = Path("qa-data/raw/comprehension_qa.jsonl")
    with open(out, "w") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
    print(f"\nWritten to {out}")

    # Samples
    for cat in sorted(by_cat.keys()):
        qs = [e for e in entries if e.category == cat]
        print(f"\n--- {cat} ---")
        for q in random.sample(qs, min(2, len(qs))):
            print(f"  Q: {q.question}")
            print(f"  A: {q.answer[:200]}")
            print()


if __name__ == "__main__":
    main()
