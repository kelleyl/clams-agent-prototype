"""Evaluation harness for the video understanding QA benchmark.

Loads QA pairs from all generators (index, LLM, KG), presents them to an LLM
with video index context, and scores the responses.

Scoring methods:
  - Exact match (MC questions)
  - Fuzzy text match (free-form text answers)
  - Entity overlap (entity listing questions)
  - Reasoning chain evaluation (multi-hop questions)

Usage:
    python evaluate.py                              # Evaluate all QA
    python evaluate.py --qa-file raw/kg_qa.jsonl    # Specific QA file
    python evaluate.py --category scene_classification  # Single category
    python evaluate.py --difficulty hard             # Filter by difficulty
    python evaluate.py --provider ollama --model qwen2.5:7b
    python evaluate.py --dry-run                     # Show questions without running LLM
    python evaluate.py --max-questions 50            # Limit for quick testing
"""

import json
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single QA pair."""
    qa_id: str
    category: str
    difficulty: str
    question: str
    gold_answer: str
    predicted_answer: str
    score: float  # 0.0 to 1.0
    scoring_method: str
    is_correct: bool
    latency_ms: float
    video_id: str = ""
    tags: list[str] = field(default_factory=list)
    error: Optional[str] = None


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


BOOLEAN_TRUE = {"true", "yes", "correct", "right", "1"}
BOOLEAN_FALSE = {"false", "no", "incorrect", "wrong", "0"}


def normalize_boolean(text: str) -> Optional[str]:
    """Normalize boolean-like answers. Returns 'true', 'false', or None."""
    t = text.lower().strip().rstrip(".")
    # Strip common preambles like "Yes, ..." or "No, ..."
    first_word = t.split(",")[0].split()[0] if t else ""
    if first_word in BOOLEAN_TRUE or t in BOOLEAN_TRUE:
        return "true"
    if first_word in BOOLEAN_FALSE or t in BOOLEAN_FALSE:
        return "false"
    return None


def fuzzy_match_score(gold: str, predicted: str) -> float:
    """Compute fuzzy match score between gold and predicted text."""
    gold_norm = normalize_text(gold)
    pred_norm = normalize_text(predicted)

    if not gold_norm or not pred_norm:
        return 0.0

    # Exact match
    if gold_norm == pred_norm:
        return 1.0

    # Boolean matching — "True" should match "Yes, ...", etc.
    gold_bool = normalize_boolean(gold)
    pred_bool = normalize_boolean(predicted)
    if gold_bool is not None and pred_bool is not None:
        return 1.0 if gold_bool == pred_bool else 0.0

    # Containment — gold in predicted or vice versa
    # Only count if the gold answer is reasonably specific (>3 words or >15 chars)
    if gold_norm in pred_norm and (len(gold_norm.split()) > 3 or len(gold_norm) > 15):
        return 1.0
    if pred_norm in gold_norm and (len(pred_norm.split()) > 3 or len(pred_norm) > 15):
        return 0.8

    # Word overlap (Jaccard)
    gold_words = set(gold_norm.split())
    pred_words = set(pred_norm.split())
    if not gold_words or not pred_words:
        return 0.0
    intersection = gold_words & pred_words
    union = gold_words | pred_words
    return len(intersection) / len(union)


def entity_overlap_score(gold_entities: list[str], predicted_text: str) -> float:
    """Score based on how many gold entities appear in the predicted text."""
    if not gold_entities:
        return 0.0
    pred_lower = predicted_text.lower()
    found = sum(1 for e in gold_entities if e.lower() in pred_lower)
    return found / len(gold_entities)


def mc_score(gold_letter: str, predicted_text: str) -> float:
    """Score a multiple-choice answer."""
    pred = predicted_text.strip().upper()
    gold = gold_letter.strip().upper()

    # Direct letter match
    if pred == gold:
        return 1.0

    # Look for letter at start of response
    match = re.match(r'^([A-D])', pred)
    if match and match.group(1) == gold:
        return 1.0

    # Look for "Answer: X" pattern
    match = re.search(r'(?:answer|choice)[\s:]*([A-D])', pred, re.IGNORECASE)
    if match and match.group(1).upper() == gold:
        return 1.0

    return 0.0


def extract_gold_answer_text(qa: dict) -> str:
    """Extract a comparable text answer from the QA entry's answer field."""
    answer = qa.get("answer", {})

    if isinstance(answer, str):
        return answer

    # LLM QA format
    if "answer" in qa and isinstance(qa["answer"], str):
        return qa["answer"]

    # Multi-hop QA format
    if "answer" in qa and isinstance(qa["answer"], str):
        return qa["answer"]

    # Index QA formats
    if isinstance(answer, dict):
        if "text" in answer:
            return answer["text"]
        if "value" in answer:
            return str(answer["value"])
        if "scene_label" in answer:
            return answer["scene_label"]
        if "count" in answer:
            return str(answer["count"])
        if "people" in answer:
            return ", ".join(answer["people"])
        if "organizations" in answer:
            return ", ".join(answer["organizations"])
        if "locations" in answer:
            return ", ".join(answer["locations"])
        if "entities" in answer:
            return ", ".join(e.get("text", "") for e in answer["entities"])
        if "explanation" in answer:
            return answer["explanation"]

    return json.dumps(answer)


def extract_gold_entities(qa: dict) -> list[str]:
    """Extract entity names from answer for entity overlap scoring."""
    answer = qa.get("answer", {})
    if not isinstance(answer, dict):
        return []

    entities = []
    for key in ["people", "organizations", "locations"]:
        entities.extend(answer.get(key, []))
    for ent in answer.get("entities", []):
        if isinstance(ent, dict) and "text" in ent:
            entities.append(ent["text"])
    return entities


def is_mc_question(qa: dict) -> bool:
    """Check if this is a multiple-choice question."""
    answer = qa.get("answer", {})
    if isinstance(answer, dict):
        if "mc" in answer:
            return True
        mc = answer.get("mc", {})
        if mc and "correct" in mc:
            return True
    if qa.get("mc_choices"):
        return True
    return False


def get_mc_correct(qa: dict) -> str:
    """Get the correct MC letter."""
    answer = qa.get("answer", {})
    if isinstance(answer, dict):
        mc = answer.get("mc", {})
        if mc:
            return mc.get("correct", "")
    return qa.get("mc_correct", "")


def get_mc_choices(qa: dict) -> dict:
    """Get MC choices dict."""
    answer = qa.get("answer", {})
    if isinstance(answer, dict):
        mc = answer.get("mc", {})
        if mc:
            return mc.get("choices", {})
    return qa.get("mc_choices", {})


def build_eval_prompt(qa: dict, video_context: Optional[str] = None) -> str:
    """Build the evaluation prompt for the LLM."""
    parts = []

    if video_context:
        parts.append(f"You have access to the following video index information:\n\n{video_context}\n\n---\n")

    parts.append(f"Question: {qa['question']}")

    if is_mc_question(qa):
        choices = get_mc_choices(qa)
        if choices:
            parts.append("\nChoices:")
            for letter, text in sorted(choices.items()):
                parts.append(f"  {letter}) {text}")
            parts.append("\nAnswer with ONLY the letter (A, B, C, or D).")
        else:
            parts.append("\nProvide a concise answer.")
    else:
        parts.append("\nProvide a concise, specific answer. Be brief.")

    return "\n".join(parts)


def call_llm(prompt: str, provider: str = "ollama", model: str = "qwen2.5:7b",
             temperature: float = 0.1, max_tokens: int = 500) -> str:
    """Call LLM and return response."""
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
            timeout=120,
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

    raise ValueError(f"Unknown provider: {provider}")


# Ablation layer configurations
ABLATION_CONFIGS = {
    "A_asr_only":       {"asr"},
    "B_asr_captions":   {"asr", "captions"},
    "C_asr_entities":   {"asr", "entities"},
    "D_asr_ocr":        {"asr", "ocr"},
    "E_full_index":     {"asr", "captions", "entities", "ocr", "labels"},
    "F_no_context":     set(),  # Single-model baseline: no structured index
}


def load_video_context(video_id: str, index_dir: Path,
                       layers: Optional[set[str]] = None) -> Optional[str]:
    """Load video index as context string for the LLM.

    Args:
        video_id: Video identifier
        index_dir: Directory containing video index JSON files
        layers: Set of layer names to include. If None, include all.
            Valid layers: "asr", "ocr", "captions", "entities", "labels"
    """
    index_path = index_dir / f"{video_id}.json"
    if not index_path.exists():
        return None

    # No-context baseline
    if layers is not None and len(layers) == 0:
        return None

    with open(index_path) as f:
        index = json.load(f)

    include_all = layers is None
    segments = index.get("segments", [])
    lines = [f"Video: {video_id}", f"Segments: {len(segments)}", ""]

    for i, seg in enumerate(segments):
        parts = [f"[Seg {i}: {seg['start_ms'] / 1000:.1f}s-{seg['end_ms'] / 1000:.1f}s]"]
        if (include_all or "labels" in layers) and seg.get("scene_label"):
            parts.append(f"type={seg['scene_label']}")
        if (include_all or "ocr" in layers) and seg.get("ocr_text", "").strip():
            parts.append(f"OCR: {seg['ocr_text'][:100]}")
        if (include_all or "asr" in layers) and seg.get("asr_transcript", "").strip():
            parts.append(f"ASR: {seg['asr_transcript'][:100]}")
        if (include_all or "captions" in layers) and seg.get("visual_caption", "").strip():
            parts.append(f"Visual: {seg['visual_caption'][:100]}")
        if (include_all or "entities" in layers):
            ents = seg.get("named_entities", [])
            if ents:
                parts.append(f"Entities: {', '.join(e['text'] for e in ents[:5])}")
            descs = seg.get("entity_descriptions", {})
            if descs:
                for name, desc in list(descs.items())[:3]:
                    parts.append(f"  {name}: {desc[:80]}")

        # Only add segment if it has content beyond the timestamp
        if len(parts) > 1:
            lines.append(" | ".join(parts))

    context = "\n".join(lines)
    if len(context) > 8000:
        context = context[:8000] + "\n[... truncated ...]"
    return context


def load_qa_entries(qa_files: list[Path]) -> list[dict]:
    """Load QA entries from JSONL files."""
    entries = []
    for path in qa_files:
        if not path.exists():
            logger.warning(f"QA file not found: {path}")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def llm_judge_score(question: str, gold: str, predicted: str,
                     provider: str = "ollama", model: str = "qwen2.5:7b") -> float:
    """Use an LLM to judge if the predicted answer is correct.

    Returns 0.0, 0.5, or 1.0.
    """
    judge_prompt = f"""You are evaluating a question-answering system. Judge whether the predicted answer is correct given the reference answer.

Question: {question}
Reference answer: {gold}
Predicted answer: {predicted}

Score the prediction:
- 1.0 if the predicted answer is correct (matches the reference, possibly paraphrased)
- 0.5 if partially correct (contains some correct information but is incomplete or has errors)
- 0.0 if incorrect or irrelevant

Respond with ONLY a number: 0.0, 0.5, or 1.0"""

    try:
        response = call_llm(judge_prompt, provider, model, temperature=0.0, max_tokens=10)
        # Parse the score
        for token in response.strip().split():
            try:
                score = float(token)
                if score in (0.0, 0.5, 1.0):
                    return score
            except ValueError:
                continue
        return 0.0
    except Exception:
        return 0.0


class Evaluator:
    """Run evaluation and collect results."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        index_dir: Optional[Path] = None,
        include_context: bool = True,
        layers: Optional[set[str]] = None,
        use_llm_judge: bool = False,
        judge_provider: Optional[str] = None,
        judge_model: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.index_dir = index_dir or Path(__file__).parent.parent / "data" / "video_indexes"
        self.include_context = include_context
        self.layers = layers
        self.use_llm_judge = use_llm_judge
        self.judge_provider = judge_provider or provider
        self.judge_model = judge_model or model
        self._context_cache: dict[str, Optional[str]] = {}

    def _get_context(self, video_id: str) -> Optional[str]:
        if not self.include_context:
            return None
        cache_key = f"{video_id}:{','.join(sorted(self.layers)) if self.layers else 'all'}"
        if cache_key not in self._context_cache:
            self._context_cache[cache_key] = load_video_context(
                video_id, self.index_dir, layers=self.layers
            )
        return self._context_cache[cache_key]

    def evaluate_one(self, qa: dict) -> EvalResult:
        """Evaluate a single QA entry."""
        video_id = qa.get("video_id") or qa.get("source_guid", "")
        category = qa.get("category", "unknown")
        difficulty = qa.get("difficulty", "unknown")
        question = qa.get("question", "")
        qa_id = qa.get("id", "unknown")
        tags = qa.get("tags", [])

        context = self._get_context(video_id)
        prompt = build_eval_prompt(qa, context)

        gold_text = extract_gold_answer_text(qa)
        gold_entities = extract_gold_entities(qa)

        try:
            t0 = time.perf_counter()
            predicted = call_llm(prompt, self.provider, self.model)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            return EvalResult(
                qa_id=qa_id, category=category, difficulty=difficulty,
                question=question, gold_answer=gold_text, predicted_answer="",
                score=0.0, scoring_method="error", is_correct=False,
                latency_ms=0, video_id=video_id, tags=tags,
                error=str(e),
            )

        # Score
        if is_mc_question(qa):
            correct_letter = get_mc_correct(qa)
            score = mc_score(correct_letter, predicted)
            method = "mc_exact"
        elif gold_entities:
            score = entity_overlap_score(gold_entities, predicted)
            method = "entity_overlap"
        elif self.use_llm_judge:
            score = llm_judge_score(
                question, gold_text, predicted,
                self.judge_provider, self.judge_model,
            )
            method = "llm_judge"
        else:
            score = fuzzy_match_score(gold_text, predicted)
            method = "fuzzy_text"

        return EvalResult(
            qa_id=qa_id, category=category, difficulty=difficulty,
            question=question, gold_answer=gold_text[:200],
            predicted_answer=predicted[:200],
            score=score, scoring_method=method,
            is_correct=score >= 0.5,
            latency_ms=latency, video_id=video_id, tags=tags,
        )

    def evaluate_all(self, qa_entries: list[dict]) -> list[EvalResult]:
        """Evaluate all QA entries."""
        results = []
        for i, qa in enumerate(qa_entries, 1):
            if i % 10 == 0 or i == 1:
                logger.info(f"  [{i}/{len(qa_entries)}] evaluating...")
            result = self.evaluate_one(qa)
            results.append(result)
            if result.error:
                logger.warning(f"  Error on {result.qa_id}: {result.error}")
        return results


def print_report(results: list[EvalResult]):
    """Print evaluation report."""
    if not results:
        print("No results to report.")
        return

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    errors = sum(1 for r in results if r.error)
    avg_score = sum(r.score for r in results) / total
    avg_latency = sum(r.latency_ms for r in results if not r.error) / max(1, total - errors)

    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Total questions: {total}")
    print(f"Correct (score >= 0.5): {correct} ({correct/total*100:.1f}%)")
    print(f"Average score: {avg_score:.3f}")
    print(f"Errors: {errors}")
    print(f"Average latency: {avg_latency:.0f}ms")

    # By category
    print(f"\n{'─'*60}")
    print(f"By Category:")
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        acc = sum(1 for r in rs if r.is_correct) / len(rs)
        avg = sum(r.score for r in rs) / len(rs)
        print(f"  {cat:30s}  n={len(rs):4d}  acc={acc:.1%}  avg_score={avg:.3f}")

    # By difficulty
    print(f"\n{'─'*60}")
    print(f"By Difficulty:")
    by_diff = defaultdict(list)
    for r in results:
        by_diff[r.difficulty].append(r)
    for diff in ["easy", "medium", "hard", "very_hard", "expert", "unknown"]:
        if diff not in by_diff:
            continue
        rs = by_diff[diff]
        acc = sum(1 for r in rs if r.is_correct) / len(rs)
        avg = sum(r.score for r in rs) / len(rs)
        print(f"  {diff:15s}  n={len(rs):4d}  acc={acc:.1%}  avg_score={avg:.3f}")

    # By scoring method
    print(f"\n{'─'*60}")
    print(f"By Scoring Method:")
    by_method = defaultdict(list)
    for r in results:
        by_method[r.scoring_method].append(r)
    for method in sorted(by_method):
        rs = by_method[method]
        acc = sum(1 for r in rs if r.is_correct) / len(rs)
        avg = sum(r.score for r in rs) / len(rs)
        print(f"  {method:20s}  n={len(rs):4d}  acc={acc:.1%}  avg_score={avg:.3f}")

    # By video
    print(f"\n{'─'*60}")
    print(f"By Video:")
    by_vid = defaultdict(list)
    for r in results:
        by_vid[r.video_id].append(r)
    for vid in sorted(by_vid):
        rs = by_vid[vid]
        acc = sum(1 for r in rs if r.is_correct) / len(rs)
        name = vid[:50] if vid else "(no video)"
        print(f"  {name:50s}  n={len(rs):4d}  acc={acc:.1%}")

    # Tag analysis
    print(f"\n{'─'*60}")
    print(f"By Tag (top 15):")
    tag_results = defaultdict(list)
    for r in results:
        for tag in r.tags:
            tag_results[tag].append(r)
    tag_stats = []
    for tag, rs in tag_results.items():
        acc = sum(1 for r in rs if r.is_correct) / len(rs)
        tag_stats.append((tag, len(rs), acc))
    for tag, n, acc in sorted(tag_stats, key=lambda x: -x[1])[:15]:
        print(f"  {tag:30s}  n={n:4d}  acc={acc:.1%}")

    print(f"\n{'='*60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate video understanding QA benchmark")
    parser.add_argument("--qa-file", type=Path, action="append", default=None,
                        help="QA JSONL file(s) to evaluate (can repeat). Default: all raw/*.jsonl")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to specific category")
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Filter to specific difficulty")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--no-context", action="store_true",
                        help="Don't provide video index context (baseline)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers to include: asr,ocr,captions,entities,labels")
    parser.add_argument("--ablation", action="store_true",
                        help="Run full ablation study across all layer configurations")
    parser.add_argument("--use-llm-judge", action="store_true",
                        help="Use LLM-as-judge for free-form answer scoring")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Model for LLM-as-judge (default: same as --model)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions evaluated")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show questions without running LLM")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save results to JSONL")
    parser.add_argument("--sample", action="store_true",
                        help="Stratified sample across categories")
    args = parser.parse_args()

    # Load QA entries
    raw_dir = Path(__file__).parent / "raw"
    if args.qa_file:
        qa_files = args.qa_file
    else:
        qa_files = sorted(raw_dir.glob("*.jsonl"))

    logger.info(f"Loading QA from: {[f.name for f in qa_files]}")
    entries = load_qa_entries(qa_files)
    logger.info(f"Loaded {len(entries)} QA entries")

    # Filter
    if args.category:
        entries = [e for e in entries if e.get("category") == args.category]
        logger.info(f"Filtered to category '{args.category}': {len(entries)} entries")

    if args.difficulty:
        entries = [e for e in entries if e.get("difficulty") == args.difficulty]
        logger.info(f"Filtered to difficulty '{args.difficulty}': {len(entries)} entries")

    # Sample
    if args.sample and args.max_questions:
        import random
        by_cat = defaultdict(list)
        for e in entries:
            by_cat[e.get("category", "unknown")].append(e)
        per_cat = max(1, args.max_questions // len(by_cat))
        sampled = []
        for cat, cat_entries in by_cat.items():
            sampled.extend(random.sample(cat_entries, min(per_cat, len(cat_entries))))
        entries = sampled[:args.max_questions]
        logger.info(f"Stratified sample: {len(entries)} entries across {len(by_cat)} categories")
    elif args.max_questions:
        import random
        entries = random.sample(entries, min(args.max_questions, len(entries)))
        logger.info(f"Random sample: {len(entries)} entries")

    if not entries:
        print("No QA entries to evaluate.")
        return

    # Dry run
    if args.dry_run:
        print(f"\n--- Dry run: {len(entries)} questions ---\n")
        for i, qa in enumerate(entries[:20], 1):
            cat = qa.get("category", "?")
            diff = qa.get("difficulty", "?")
            vid = (qa.get("video_id") or qa.get("source_guid", ""))[:40]
            print(f"{i:3d}. [{cat}] ({diff}) {vid}")
            print(f"     Q: {qa['question'][:100]}")
            gold = extract_gold_answer_text(qa)
            print(f"     A: {gold[:100]}")
            print()
        if len(entries) > 20:
            print(f"... and {len(entries) - 20} more")
        return

    # Parse layers
    layers = None
    if args.layers:
        layers = set(args.layers.split(","))
    elif args.no_context:
        layers = set()

    # Ablation mode — run all layer configurations
    if args.ablation:
        logger.info(f"Running ablation study with {len(entries)} questions × {len(ABLATION_CONFIGS)} conditions")
        all_ablation_results = {}
        for config_name, config_layers in ABLATION_CONFIGS.items():
            logger.info(f"\n{'='*40}\nAblation condition: {config_name} (layers: {config_layers or 'none'})\n{'='*40}")
            evaluator = Evaluator(
                provider=args.provider,
                model=args.model,
                include_context=bool(config_layers),
                layers=config_layers if config_layers else None,
                use_llm_judge=args.use_llm_judge,
                judge_model=args.judge_model,
            )
            results = evaluator.evaluate_all(entries)
            all_ablation_results[config_name] = results
            print_report(results)

        # Summary comparison
        print(f"\n{'='*60}")
        print("ABLATION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Condition':<25s} {'N':>5s} {'Accuracy':>10s} {'Avg Score':>10s}")
        print(f"{'─'*55}")
        for config_name, results in all_ablation_results.items():
            n = len(results)
            acc = sum(1 for r in results if r.is_correct) / n if n else 0
            avg = sum(r.score for r in results) / n if n else 0
            print(f"{config_name:<25s} {n:>5d} {acc:>9.1%} {avg:>10.3f}")
        print(f"{'='*60}")

        # Save all results
        out_dir = Path(__file__).parent / "eval_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for config_name, results in all_ablation_results.items():
            out_path = out_dir / f"ablation_{config_name}_{args.model.replace('/', '_')}_{timestamp}.jsonl"
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        logger.info(f"Saved ablation results to {out_dir}")
        return

    # Single evaluation run
    logger.info(f"Evaluating {len(entries)} questions with {args.provider}/{args.model}")
    evaluator = Evaluator(
        provider=args.provider,
        model=args.model,
        include_context=not args.no_context,
        layers=layers,
        use_llm_judge=args.use_llm_judge,
        judge_model=args.judge_model,
    )

    results = evaluator.evaluate_all(entries)

    # Report
    print_report(results)

    # Save
    layer_tag = f"_layers-{'-'.join(sorted(layers))}" if layers else ""
    out_path = args.output or (Path(__file__).parent / "eval_results" /
                                f"eval_{args.model.replace('/', '_')}{layer_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
