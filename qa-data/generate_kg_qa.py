"""Generate multi-hop QA pairs from the knowledge graph.

Enumerates reasoning paths through the KG (built from chyron extraction,
NER, entity resolution, and Wikidata linking) and generates templated
questions with full reasoning chains.

Usage:
    python generate_kg_qa.py                          # Default settings
    python generate_kg_qa.py --max-hops 5             # Include 5-hop paths
    python generate_kg_qa.py --max-paths 500          # More paths per hop level
    python generate_kg_qa.py --stats                  # Print stats only
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate multi-hop QA from knowledge graph")
    parser.add_argument("--db", type=Path,
                        default=Path(__file__).parent.parent / "data" / "knowledge_graph.db",
                        help="Path to KG SQLite database")
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--max-paths", type=int, default=200,
                        help="Max paths per hop level")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stats", action="store_true", help="Print stats only")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: KG database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    from kg.qgen.question_generator import generate_questions

    logger.info(f"Generating multi-hop questions from {args.db}")
    logger.info(f"  Hop range: {args.min_hops}-{args.max_hops}, max paths/hop: {args.max_paths}")

    questions = generate_questions(
        db_path=str(args.db),
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        max_paths_per_hop=args.max_paths,
    )

    # Stats
    by_pattern = defaultdict(int)
    by_difficulty = defaultdict(int)
    by_hops = defaultdict(int)
    external_count = 0
    cross_seg = 0
    cross_vid = 0

    for q in questions:
        by_pattern[q.pattern] += 1
        by_difficulty[q.difficulty] += 1
        by_hops[q.hop_count] += 1
        if q.uses_external_knowledge:
            external_count += 1
        if q.crosses_segments:
            cross_seg += 1
        if q.crosses_videos:
            cross_vid += 1

    print(f"\nTotal multi-hop questions: {len(questions)}")
    print(f"\nBy hop count:")
    for h, c in sorted(by_hops.items()):
        print(f"  {h}-hop: {c}")
    print(f"\nBy pattern:")
    for p, c in sorted(by_pattern.items(), key=lambda x: -x[1]):
        print(f"  {p}: {c}")
    print(f"\nBy difficulty:")
    for d, c in sorted(by_difficulty.items()):
        print(f"  {d}: {c}")
    print(f"\nUses external knowledge: {external_count}")
    print(f"Crosses segments: {cross_seg}")
    print(f"Crosses videos: {cross_vid}")

    if args.stats:
        return

    # Write output
    out_path = args.output or (Path(__file__).parent / "raw" / "kg_qa.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(questions)} questions to {out_path}")

    # Print samples
    import random
    print("\n--- Sample questions ---")
    for q in random.sample(questions, min(5, len(questions))):
        print(f"\n[{q.pattern}] ({q.difficulty}, {q.hop_count}-hop)")
        print(f"  Q: {q.question}")
        print(f"  A: {q.answer}")
        chain = " → ".join(f"{h.operation}({h.output})" for h in q.reasoning_chain)
        print(f"  Chain: {chain}")
        print(f"  Sources: {q.evidence_sources}")


if __name__ == "__main__":
    main()
