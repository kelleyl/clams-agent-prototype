"""
LLM-augmented QA generation for video indexes.

Generates semantically rich, reasoning-required questions by feeding dense
segment descriptions to an LLM. Inspired by CinePile (LLM generation from
text descriptions), VideoEspresso (chain-of-thought), and AutoConverter
(challenging MC distractors).

Three-stage pipeline:
  1. Dense context assembly — fuse SWT labels + SmolVLM2 captions + ASR + NER
     into rich natural language descriptions per segment and per video
  2. LLM question generation — prompt an LLM with segment descriptions and ask
     for questions requiring cross-modal reasoning, temporal understanding,
     and multi-hop inference
  3. Adversarial filtering — test each question against a text-only LLM with
     no index access; discard if answerable without the video context

Question categories:
  - visual_description: What is shown? Requires visual caption understanding
  - cross_modal: Connect spoken words to visual content
  - temporal_reasoning: Ordering, causation, before/after relationships
  - entity_identification: Who/what appears, based on visual + text evidence
  - narrative_comprehension: What is happening, storyline, topic of segment
  - factual_extraction: Specific facts from OCR text (dates, names, titles)

Usage:
    python generate_llm_qa.py                           # All indexed videos
    python generate_llm_qa.py --video <video_id>        # Single video
    python generate_llm_qa.py --provider ollama          # Use local Ollama
    python generate_llm_qa.py --provider openai           # Use OpenAI
    python generate_llm_qa.py --no-filter                 # Skip adversarial filtering
    python generate_llm_qa.py --mc                        # Generate MC choices
"""

import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class GeneratedQA:
    """A single generated QA pair."""
    id: str
    video_id: str
    category: str
    question: str
    answer: str
    reasoning: str  # Chain-of-thought explaining why this answer is correct
    difficulty: str  # easy, medium, hard
    evidence_segments: list[int]  # segment indices used to derive the answer
    modalities: list[str]  # which modalities are needed: visual, audio, text, temporal
    mc_choices: Optional[dict] = None  # {A: ..., B: ..., C: ..., D: ...}
    mc_correct: Optional[str] = None  # "A", "B", etc.
    filtered: bool = False  # True if failed adversarial filter
    filter_reason: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# ============================================================================
# Stage 1: Dense context assembly
# ============================================================================

def assemble_segment_description(seg: dict, seg_idx: int) -> str:
    """Convert a raw index segment into a rich natural-language description."""
    parts = []

    # Time range
    start_s = seg.get("start_ms", 0) / 1000
    end_s = seg.get("end_ms", 0) / 1000
    parts.append(f"[Segment {seg_idx}: {start_s:.1f}s - {end_s:.1f}s]")

    # Scene label
    label = seg.get("scene_label", "")
    if label:
        parts.append(f"Scene type: {label}")

    # Visual caption (from SmolVLM2)
    caption = seg.get("visual_caption", "")
    if caption:
        parts.append(f"Visual description: {caption}")

    # OCR text
    ocr = seg.get("ocr_text", "")
    if ocr:
        parts.append(f"On-screen text (OCR): {ocr}")

    # ASR transcript
    asr = seg.get("asr_transcript", "")
    if asr and len(asr.strip()) > 10:
        parts.append(f"Spoken dialogue: {asr.strip()}")

    # Named entities
    entities = seg.get("named_entities", [])
    if entities:
        ent_strs = [f"{e['text']} ({e.get('type', '?')})" for e in entities]
        parts.append(f"Entities mentioned: {', '.join(ent_strs)}")

    # Grounded entities
    grounded = seg.get("grounded_entities", {})
    if grounded and isinstance(grounded, dict):
        for ent_text, desc in list(grounded.items())[:3]:
            if desc:
                parts.append(f"  - {ent_text}: {desc[:150]}")

    return "\n".join(parts)


def assemble_video_context(index: dict) -> str:
    """Build a full video context string from all segments."""
    video_id = index.get("video_id", "unknown")
    segments = index.get("segments", [])

    header = f"Video: {video_id}\n"
    header += f"Total segments: {len(segments)}\n"

    # Compute stats
    labels = defaultdict(int)
    total_asr_words = 0
    total_entities = 0
    has_ocr = 0
    has_caption = 0

    for seg in segments:
        lbl = seg.get("scene_label", "")
        if lbl:
            labels[lbl] += 1
        asr = seg.get("asr_transcript", "")
        if asr:
            total_asr_words += len(asr.split())
        total_entities += len(seg.get("named_entities", []))
        if seg.get("ocr_text", ""):
            has_ocr += 1
        if seg.get("visual_caption", ""):
            has_caption += 1

    header += f"Scene types: {dict(labels)}\n"
    header += f"ASR words: {total_asr_words}, OCR segments: {has_ocr}, Captioned segments: {has_caption}\n"
    header += f"Named entities: {total_entities}\n"
    header += "---\n\n"

    # Segment descriptions
    seg_descs = []
    for i, seg in enumerate(segments):
        desc = assemble_segment_description(seg, i)
        seg_descs.append(desc)

    return header + "\n\n".join(seg_descs)


# ============================================================================
# Stage 2: LLM question generation
# ============================================================================

GENERATION_PROMPT = """You are generating evaluation questions for a video question-answering benchmark.

Below is a structured description of a video, derived from automated analysis (scene detection, speech recognition, OCR, visual captioning, and named entity recognition). Your task is to generate challenging, diverse questions that test a model's ability to understand and reason about this video.

{video_context}

---

Generate {n_questions} question-answer pairs following these guidelines:

QUESTION TYPES (generate a mix):
1. **visual_description**: Questions about what is visually shown (requires the visual caption)
   - "What is displayed on the slate at the beginning of the broadcast?"
   - "Describe the visual setting of the news segment."

2. **cross_modal**: Questions requiring information from BOTH visual AND audio/text
   - "What topic is the anchor discussing while the chyron shows [X]?"
   - "How does the on-screen text relate to what is being said?"

3. **temporal_reasoning**: Questions about ordering, duration, or temporal relationships
   - "What happens immediately after the commercial break?"
   - "Which segment is longer: the news report or the interview?"

4. **entity_identification**: Questions about who/what appears, requiring evidence fusion
   - "Who is the reporter covering this story, based on the chyron and spoken introduction?"
   - "What organization is associated with [person] in this broadcast?"

5. **narrative_comprehension**: Questions about the overall content or storyline
   - "What is the main topic of this news broadcast?"
   - "Summarize the key events shown in this video."

6. **factual_extraction**: Specific facts from OCR/ASR that require precision
   - "What date appears on the production slate?"
   - "What phone number or address is shown in the commercial?"

REQUIREMENTS:
- Questions MUST require information from the video — they should NOT be answerable from world knowledge alone
- Prefer questions that need multiple segments or modalities to answer
- Include the specific evidence (segment numbers) supporting each answer
- Vary difficulty: some easy (single segment lookup), some hard (multi-segment reasoning)
- Answers should be specific and verifiable, not vague

OUTPUT FORMAT (JSON array):
[
  {{
    "category": "visual_description",
    "question": "What is shown on screen during the opening of the broadcast?",
    "answer": "A CBS Family Presentation title card with white text on a black background.",
    "reasoning": "Segment 0 has scene type 'slate' and visual caption describes the CBS title card.",
    "difficulty": "easy",
    "evidence_segments": [0],
    "modalities": ["visual"]
  }},
  ...
]

Generate exactly {n_questions} questions. Return ONLY the JSON array, no other text."""


MC_CONVERSION_PROMPT = """Convert the following question-answer pair into a multiple-choice question with 4 options (A-D). The correct answer should be included, and the 3 distractors should be plausible but wrong.

Video context (abbreviated):
{abbreviated_context}

Question: {question}
Correct answer: {answer}

Generate 3 plausible but INCORRECT distractor answers. Distractors should:
- Be similar in length and style to the correct answer
- Be plausible given the video topic but factually wrong
- NOT be obviously absurd

OUTPUT FORMAT (JSON):
{{
  "choices": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "correct": "B"
}}

Return ONLY the JSON, no other text."""


# ============================================================================
# Stage 3: Adversarial filtering
# ============================================================================

ADVERSARIAL_PROMPT = """Answer the following question about a video. You have NO access to the video itself — only the question text. Try your best to guess the answer based on common sense and world knowledge.

Question: {question}

If you can answer this confidently without seeing the video, provide your answer.
If you genuinely cannot answer without the video, respond with "CANNOT_ANSWER".

Your response (just the answer or CANNOT_ANSWER):"""


# ============================================================================
# LLM interface
# ============================================================================

def call_llm(prompt: str, provider: str = "ollama", model: str = "qwen2.5:7b",
             temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """Call an LLM and return the response text."""
    if provider == "ollama":
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    elif provider == "openai":
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider: {provider}")


def parse_json_from_llm(text: str) -> list | dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    # Find JSON array or object
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


# ============================================================================
# Pipeline
# ============================================================================

class LLMQAGenerator:
    """Generate QA pairs using LLM augmentation."""

    def __init__(
        self,
        index_dir: str | Path,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        questions_per_video: int = 20,
        do_mc: bool = False,
        do_filter: bool = True,
    ):
        self.index_dir = Path(index_dir)
        self.provider = provider
        self.model = model
        self.questions_per_video = questions_per_video
        self.do_mc = do_mc
        self.do_filter = do_filter
        self._id_counter = 0

    def generate_id(self) -> str:
        self._id_counter += 1
        return f"qa-llm-{self._id_counter:06d}"

    def list_videos(self) -> list[str]:
        return [p.stem for p in sorted(self.index_dir.glob("*.json"))]

    def load_index(self, video_id: str) -> dict:
        with open(self.index_dir / f"{video_id}.json") as f:
            return json.load(f)

    def generate_for_video(self, video_id: str) -> list[GeneratedQA]:
        """Full pipeline: assemble context → generate → filter → MC convert."""
        index = self.load_index(video_id)
        segments = index.get("segments", [])
        if not segments:
            logger.warning(f"No segments for {video_id}")
            return []

        # Stage 1: Assemble dense context
        context = assemble_video_context(index)
        logger.info(f"  Context: {len(context)} chars, {len(segments)} segments")

        # Truncate context if too long (keep first and last segments)
        max_context = 6000
        if len(context) > max_context:
            # Keep header + first 60% + last 40% of segments
            lines = context.split("\n\n")
            header = lines[0] if lines else ""
            seg_parts = lines[1:] if len(lines) > 1 else []
            n = len(seg_parts)
            keep_first = int(n * 0.6)
            keep_last = n - keep_first
            truncated = [header] + seg_parts[:keep_first] + ["[... middle segments omitted ...]"] + seg_parts[-keep_last:]
            context = "\n\n".join(truncated)
            if len(context) > max_context:
                context = context[:max_context] + "\n[... truncated ...]"

        # Stage 2: LLM question generation
        prompt = GENERATION_PROMPT.format(
            video_context=context,
            n_questions=self.questions_per_video,
        )

        logger.info(f"  Generating {self.questions_per_video} questions via {self.provider}/{self.model}...")
        try:
            response = call_llm(prompt, self.provider, self.model)
            raw_questions = parse_json_from_llm(response)
        except Exception as e:
            logger.error(f"  LLM generation failed: {e}")
            return []

        if not isinstance(raw_questions, list):
            raw_questions = [raw_questions]

        # Convert to GeneratedQA objects
        qa_pairs = []
        for raw in raw_questions:
            try:
                qa = GeneratedQA(
                    id=self.generate_id(),
                    video_id=video_id,
                    category=raw.get("category", "unknown"),
                    question=raw["question"],
                    answer=raw["answer"],
                    reasoning=raw.get("reasoning", ""),
                    difficulty=raw.get("difficulty", "medium"),
                    evidence_segments=raw.get("evidence_segments", []),
                    modalities=raw.get("modalities", []),
                )
                qa_pairs.append(qa)
            except (KeyError, TypeError) as e:
                logger.debug(f"  Skipping malformed QA: {e}")

        logger.info(f"  Generated {len(qa_pairs)} questions")

        # Stage 3: Adversarial filtering
        if self.do_filter and qa_pairs:
            qa_pairs = self._adversarial_filter(qa_pairs)

        # Optional: MC conversion
        if self.do_mc and qa_pairs:
            abbreviated = context[:3000]
            for qa in qa_pairs:
                if not qa.filtered:
                    self._convert_to_mc(qa, abbreviated)

        return qa_pairs

    def _adversarial_filter(self, qa_pairs: list[GeneratedQA]) -> list[GeneratedQA]:
        """Filter out questions answerable without the video."""
        logger.info(f"  Adversarial filtering {len(qa_pairs)} questions...")
        filtered_count = 0

        for qa in qa_pairs:
            try:
                prompt = ADVERSARIAL_PROMPT.format(question=qa.question)
                response = call_llm(prompt, self.provider, self.model, temperature=0.1, max_tokens=200)
                response = response.strip()

                if "CANNOT_ANSWER" in response.upper():
                    # Good — question requires the video
                    pass
                else:
                    # Check if the LLM's guess is close to the actual answer
                    guess = response.lower().strip()
                    answer = qa.answer.lower().strip()

                    # Simple overlap check — if >50% of answer words appear in guess
                    answer_words = set(answer.split())
                    guess_words = set(guess.split())
                    if answer_words and len(answer_words & guess_words) / len(answer_words) > 0.5:
                        qa.filtered = True
                        qa.filter_reason = f"Answerable without video. LLM guessed: {response[:100]}"
                        filtered_count += 1

            except Exception as e:
                logger.debug(f"  Filter error: {e}")

        logger.info(f"  Filtered out {filtered_count}/{len(qa_pairs)} (answerable without video)")
        return qa_pairs

    def _convert_to_mc(self, qa: GeneratedQA, abbreviated_context: str):
        """Convert a QA pair to multiple choice using LLM-generated distractors."""
        try:
            prompt = MC_CONVERSION_PROMPT.format(
                abbreviated_context=abbreviated_context,
                question=qa.question,
                answer=qa.answer,
            )
            response = call_llm(prompt, self.provider, self.model, temperature=0.8, max_tokens=500)
            mc_data = parse_json_from_llm(response)

            if isinstance(mc_data, dict) and "choices" in mc_data:
                qa.mc_choices = mc_data["choices"]
                qa.mc_correct = mc_data.get("correct", "A")

        except Exception as e:
            logger.debug(f"  MC conversion failed: {e}")

    def generate_all(self, video_id: Optional[str] = None) -> list[GeneratedQA]:
        """Generate QA pairs for one or all indexed videos."""
        videos = [video_id] if video_id else self.list_videos()
        all_qa = []

        for i, vid in enumerate(videos, 1):
            logger.info(f"\n[{i}/{len(videos)}] {vid}")
            qa_pairs = self.generate_for_video(vid)
            all_qa.extend(qa_pairs)

        return all_qa


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-augmented QA generation from video indexes")
    parser.add_argument("--index-dir", type=Path,
                        default=Path(__file__).parent.parent / "data" / "video_indexes")
    parser.add_argument("--video", type=str, help="Single video ID to process")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--questions", type=int, default=20, help="Questions per video")
    parser.add_argument("--mc", action="store_true", help="Generate multiple-choice")
    parser.add_argument("--no-filter", action="store_true", help="Skip adversarial filtering")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stats", action="store_true", help="Print stats only")
    args = parser.parse_args()

    generator = LLMQAGenerator(
        index_dir=args.index_dir,
        provider=args.provider,
        model=args.model,
        questions_per_video=args.questions,
        do_mc=args.mc,
        do_filter=not args.no_filter,
    )

    if args.stats:
        videos = generator.list_videos()
        print(f"Indexed videos: {len(videos)}")
        for vid in videos:
            idx = generator.load_index(vid)
            segs = idx.get("segments", [])
            n_asr = sum(1 for s in segs if s.get("asr_transcript", "").strip())
            n_ocr = sum(1 for s in segs if s.get("ocr_text", "").strip())
            n_cap = sum(1 for s in segs if s.get("visual_caption", "").strip())
            n_ent = sum(len(s.get("named_entities", [])) for s in segs)
            print(f"  {vid}: {len(segs)} segs, {n_asr} ASR, {n_ocr} OCR, {n_cap} captions, {n_ent} entities")
        return

    qa_pairs = generator.generate_all(args.video)

    # Output
    suffix = "_mc" if args.mc else ""
    out_path = args.output or (Path(__file__).parent / "raw" / f"llm_qa{suffix}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = [qa for qa in qa_pairs if not qa.filtered]
    filtered = [qa for qa in qa_pairs if qa.filtered]

    with open(out_path, "w") as f:
        for qa in kept:
            f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + "\n")

    logger.info(f"\nResults: {len(kept)} kept, {len(filtered)} filtered → {out_path}")

    # Stats
    by_cat = defaultdict(int)
    by_diff = defaultdict(int)
    by_mod = defaultdict(int)
    for qa in kept:
        by_cat[qa.category] += 1
        by_diff[qa.difficulty] += 1
        for m in qa.modalities:
            by_mod[m] += 1

    logger.info(f"Categories: {dict(by_cat)}")
    logger.info(f"Difficulty: {dict(by_diff)}")
    logger.info(f"Modalities: {dict(by_mod)}")


if __name__ == "__main__":
    main()
