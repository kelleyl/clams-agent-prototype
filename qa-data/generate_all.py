"""Master script to generate all QA data for the video understanding benchmark.

Runs all three generators in sequence and produces combined statistics.

Usage:
    python generate_all.py                   # Generate everything
    python generate_all.py --skip-llm        # Skip LLM generation (fast)
    python generate_all.py --skip-kg         # Skip KG generation
    python generate_all.py --mc              # Include multiple-choice variants
    python generate_all.py --stats           # Print stats for existing data only
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def print_combined_stats(raw_dir: Path):
    """Print statistics across all generated QA files."""
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        print("No QA files found.")
        return

    all_entries = []
    print(f"\n{'='*60}")
    print("QA BENCHMARK STATISTICS")
    print(f"{'='*60}")

    for f in files:
        entries = load_jsonl(f)
        all_entries.extend(entries)
        print(f"  {f.name:30s}  {len(entries):5d} entries  ({f.stat().st_size / 1024:.0f} KB)")

    print(f"  {'TOTAL':30s}  {len(all_entries):5d} entries")

    # By category
    by_cat = defaultdict(int)
    by_diff = defaultdict(int)
    by_tag = defaultdict(int)
    by_video = defaultdict(int)
    mc_count = 0
    has_reasoning = 0

    for e in all_entries:
        cat = e.get("category", "unknown")
        by_cat[cat] += 1

        diff = e.get("difficulty", "unknown")
        by_diff[diff] += 1

        for tag in e.get("tags", []):
            by_tag[tag] += 1

        vid = e.get("video_id") or e.get("source_guid", "")
        if vid:
            by_video[vid] += 1

        # MC detection
        answer = e.get("answer", {})
        if isinstance(answer, dict) and ("mc" in answer or "mc_choices" in e):
            mc_count += 1

        if e.get("reasoning") or e.get("reasoning_chain"):
            has_reasoning += 1

    print(f"\n{'─'*60}")
    print("By Category:")
    for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat:30s}  {count:5d}")

    print(f"\nBy Difficulty:")
    for diff in ["easy", "medium", "hard", "very_hard", "expert", "unknown"]:
        if diff in by_diff:
            print(f"  {diff:15s}  {by_diff[diff]:5d}")

    print(f"\nBy Video ({len(by_video)} videos):")
    for vid, count in sorted(by_video.items(), key=lambda x: -x[1]):
        print(f"  {vid[:50]:50s}  {count:5d}")

    print(f"\nMultiple choice: {mc_count}")
    print(f"Has reasoning chain: {has_reasoning}")

    # Modality analysis
    vis_only = sum(1 for e in all_entries if "visual_only" in e.get("tags", []))
    req_video = sum(1 for e in all_entries if "requires_video" in e.get("tags", []))
    req_audio = sum(1 for e in all_entries if "requires_audio" in e.get("tags", []))
    cross = sum(1 for e in all_entries if "cross-layer" in e.get("tags", []))

    print(f"\nModality Requirements:")
    print(f"  Visual only: {vis_only}")
    print(f"  Requires video: {req_video}")
    print(f"  Requires audio: {req_audio}")
    print(f"  Cross-layer: {cross}")

    print(f"\nTop Tags:")
    for tag, count in sorted(by_tag.items(), key=lambda x: -x[1])[:20]:
        print(f"  {tag:30s}  {count:5d}")

    print(f"\n{'='*60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate all QA benchmark data")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM-generated QA")
    parser.add_argument("--skip-kg", action="store_true", help="Skip KG multi-hop QA")
    parser.add_argument("--skip-index", action="store_true", help="Skip index-based QA")
    parser.add_argument("--mc", action="store_true", help="Generate multiple-choice variants")
    parser.add_argument("--stats", action="store_true", help="Print stats for existing data only")
    parser.add_argument("--llm-provider", default="ollama")
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument("--llm-questions", type=int, default=20, help="Questions per video for LLM gen")
    args = parser.parse_args()

    raw_dir = Path(__file__).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.stats:
        print_combined_stats(raw_dir)
        return

    project_root = Path(__file__).parent.parent
    index_dir = project_root / "data" / "video_indexes"
    kg_db = project_root / "data" / "knowledge_graph.db"

    timings = {}

    # 1. Index-based QA
    if not args.skip_index:
        print("\n" + "="*60)
        print("STEP 1: Index-based QA generation")
        print("="*60)
        t0 = time.time()

        run_index_generation(index_dir, raw_dir, mc=args.mc)

        timings["index"] = time.time() - t0
        print(f"  Completed in {timings['index']:.1f}s")

    # 2. KG multi-hop QA
    if not args.skip_kg:
        print("\n" + "="*60)
        print("STEP 2: KG multi-hop QA generation")
        print("="*60)
        t0 = time.time()

        if kg_db.exists():
            from kg.qgen.question_generator import generate_questions
            questions = generate_questions(str(kg_db), min_hops=2, max_hops=4, max_paths_per_hop=200)
            out = raw_dir / "kg_qa.jsonl"
            with open(out, "w") as f:
                for q in questions:
                    f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
            print(f"  Generated {len(questions)} multi-hop questions → {out}")
            timings["kg"] = time.time() - t0
        else:
            print(f"  Skipping: KG database not found at {kg_db}")

    # 3. LLM-augmented QA
    if not args.skip_llm:
        print("\n" + "="*60)
        print("STEP 3: LLM-augmented QA generation")
        print("="*60)
        t0 = time.time()

        from generate_llm_qa import LLMQAGenerator
        generator = LLMQAGenerator(
            index_dir=index_dir,
            provider=args.llm_provider,
            model=args.llm_model,
            questions_per_video=args.llm_questions,
            do_mc=args.mc,
            do_filter=True,
        )
        qa_pairs = generator.generate_all()
        kept = [qa for qa in qa_pairs if not qa.filtered]
        out = raw_dir / "llm_qa.jsonl"
        with open(out, "w") as f:
            for qa in kept:
                f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + "\n")
        filtered = len(qa_pairs) - len(kept)
        print(f"  Generated {len(kept)} questions ({filtered} filtered) → {out}")
        timings["llm"] = time.time() - t0

    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    for step, t in timings.items():
        print(f"  {step}: {t:.1f}s")

    print_combined_stats(raw_dir)


def run_index_generation(index_dir: Path, raw_dir: Path, mc: bool = False):
    """Run index QA generation (importable helper)."""
    from generate_index_qa import IndexQAGenerator, print_stats

    generator = IndexQAGenerator(index_dir, multiple_choice=False)
    entries = generator.generate_all()
    out = raw_dir / "index_qa.jsonl"
    with open(out, "w") as f:
        for entry in entries:
            f.write(entry.to_json() + "\n")
    print(f"  Generated {len(entries)} index QA → {out}")
    print_stats(entries)

    if mc:
        generator_mc = IndexQAGenerator(index_dir, multiple_choice=True)
        entries_mc = generator_mc.generate_all()
        out_mc = raw_dir / "index_qa_mc.jsonl"
        with open(out_mc, "w") as f:
            for entry in entries_mc:
                f.write(entry.to_json() + "\n")
        print(f"  Generated {len(entries_mc)} MC index QA → {out_mc}")


if __name__ == "__main__":
    main()
