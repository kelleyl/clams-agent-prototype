#!/usr/bin/env python3
"""Run GEPA optimization for cross-modal QA generation.

Follows the pattern from slate_dataset_refactored/experiments/working_gepa_experiment.py:
1. Establish baseline (un-optimized prompt)
2. Run GEPA optimization
3. Compare baseline vs optimized quality scores
4. Save optimized prompts and reports

Usage:
    python run_optimization.py                              # Full run
    python run_optimization.py --num-instances 5            # Test mode
    python run_optimization.py --model qwen3:8b             # Specific model
    python run_optimization.py --judge-model qwen3:30b      # Specific judge
    python run_optimization.py --skip-gepa                  # Baseline only
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent))
from ollama_lm import make_ollama_lm, make_vllm_lm
from qa_module import QAGeneratorModule, QAOutput, QuestionOutput
from qa_metric import (
    qa_quality_metric_cached,
    judge_question,
    compute_question_score,
)
from prepare_examples import (
    load_qa_data,
    group_by_segment,
    segments_to_dspy_examples,
    split_train_dev,
)

logger = logging.getLogger(__name__)

QA_DATA_DIR = Path(__file__).resolve().parent.parent
RAW_QA_FILE = QA_DATA_DIR / "raw" / "llm_comprehension_qa.jsonl"


def setup_logging(log_path: Path):
    """Configure logging to console and file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    root.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%dT%H:%M:%S"
    ))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%dT%H:%M:%S"
    ))

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    logging.info(f"Logging to {log_path}")


def evaluate_predictions(program: QAGeneratorModule,
                         devset: list[dspy.Example],
                         judge_model: str,
                         judge_api_base: str,
                         backend: str = "vllm") -> dict:
    """Run the program on the devset and score with the LLM judge.

    Returns a dict with overall stats and per-question scores.
    """
    all_scores = []
    all_questions = []
    all_feedbacks = []
    failures = 0

    for i, example in enumerate(devset):
        try:
            prediction = program(
                segment_evidence=example.segment_evidence,
                broadcast_context=example.broadcast_context,
            )
            questions = getattr(prediction, "questions", [])

            for q in questions:
                if hasattr(q, "model_dump"):
                    q_dict = q.model_dump()
                elif isinstance(q, dict):
                    q_dict = q
                else:
                    continue

                scores = judge_question(
                    q_dict, example.segment_evidence,
                    model=judge_model, api_base=judge_api_base,
                    backend=backend,
                )
                if scores:
                    composite = compute_question_score(scores)
                    all_scores.append(composite)
                    all_questions.append({
                        "question": q_dict.get("question", ""),
                        "answer": q_dict.get("answer", ""),
                        "modalities_required": q_dict.get("modalities_required", []),
                        "reasoning_type": q_dict.get("reasoning_type", ""),
                        "scores": scores,
                        "composite": composite,
                    })

            if (i + 1) % 5 == 0:
                avg_so_far = sum(all_scores) / len(all_scores) if all_scores else 0
                logging.info(f"  Evaluated {i+1}/{len(devset)} segments, "
                             f"avg quality: {avg_so_far:.3f}")

        except Exception as e:
            failures += 1
            logger.warning(f"  Failed on example {i}: {e}")

    # Compute aggregate stats
    result = {
        "num_segments": len(devset),
        "num_questions": len(all_questions),
        "num_failures": failures,
        "avg_composite": sum(all_scores) / len(all_scores) if all_scores else 0,
    }

    # Per-dimension averages
    dims = ["cognitive_level", "naturalness", "groundedness", "information_need"]
    for dim in dims:
        vals = [q["scores"].get(dim, 0) for q in all_questions if dim in q.get("scores", {})]
        result[f"avg_{dim}"] = sum(vals) / len(vals) if vals else 0

    # Reasoning type distribution
    type_counts = defaultdict(int)
    for q in all_questions:
        type_counts[q.get("reasoning_type", "unknown")] += 1
    result["reasoning_types"] = dict(type_counts)

    # Modality distribution
    mod_counts = defaultdict(int)
    for q in all_questions:
        for m in q.get("modalities_required", []):
            mod_counts[m] += 1
    result["modality_distribution"] = dict(mod_counts)

    result["questions"] = all_questions
    return result


def save_report(report: dict, stage: str, output_dir: Path):
    """Save evaluation report to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"report_{stage}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved {stage} report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GEPA optimization for cross-modal QA generation"
    )
    parser.add_argument("--model", type=str,
                        default=os.environ.get("MODEL", "qwen3:8b"),
                        help="Generation model")
    parser.add_argument("--judge-model", type=str,
                        default=os.environ.get("JUDGE_MODEL", "qwen3:30b"),
                        help="Judge model (should be stronger than generation model)")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["ollama", "vllm"],
                        help="LLM backend (default: vllm)")
    parser.add_argument("--api-base", type=str,
                        default=os.environ.get("API_BASE", "http://localhost:8100/v1"),
                        help="API base URL")
    parser.add_argument("--judge-api-base", type=str, default=None,
                        help="Judge API base URL (defaults to --api-base)")
    parser.add_argument("--num-instances", type=int, default=None,
                        help="Limit to N instances (test mode)")
    parser.add_argument("--train-ratio", type=float, default=0.3,
                        help="Fraction for training")
    parser.add_argument("--skip-gepa", action="store_true",
                        help="Skip GEPA optimization (baseline only)")
    parser.add_argument("--qa-file", type=str, default=str(RAW_QA_FILE),
                        help="Path to QA JSONL file")
    parser.add_argument("--use-cot", action="store_true", default=True,
                        help="Use ChainOfThought (default: True)")
    parser.add_argument("--gepa-auto", type=str, default="medium",
                        choices=["light", "medium", "heavy"],
                        help="GEPA auto setting")
    args = parser.parse_args()

    # Setup output directory
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_safe = args.model.replace("/", "_").replace(":", "_")
    experiment_name = f"{run_timestamp}_qa-gepa_{model_safe}"
    output_dir = Path(__file__).resolve().parent / "experiment_outputs" / experiment_name

    setup_logging(output_dir / "experiment.log")

    judge_api_base = args.judge_api_base or args.api_base

    logging.info("=" * 60)
    logging.info("GEPA QA OPTIMIZATION EXPERIMENT")
    logging.info("=" * 60)
    logging.info(f"  Backend:          {args.backend}")
    logging.info(f"  Generation model: {args.model}")
    logging.info(f"  Judge model:      {args.judge_model}")
    logging.info(f"  API base:         {args.api_base}")
    logging.info(f"  Judge API base:   {judge_api_base}")
    logging.info(f"  Output:           {output_dir}")

    # --- Step 1: Configure DSPy ---
    logging.info("STEP 1: Configuring DSPy with generation LM...")
    make_lm = make_vllm_lm if args.backend == "vllm" else make_ollama_lm
    gen_lm = make_lm(model=args.model, api_base=args.api_base)
    dspy.configure(lm=gen_lm)

    # --- Step 2: Load and prepare data ---
    logging.info("STEP 2: Loading and preparing data...")
    items = load_qa_data(Path(args.qa_file))
    segments = group_by_segment(items)
    examples = segments_to_dspy_examples(segments)

    if args.num_instances:
        examples = examples[:args.num_instances]
        logging.info(f"  Test mode: limited to {len(examples)} examples")

    trainset, devset = split_train_dev(examples, train_ratio=args.train_ratio)

    # --- Step 3: Baseline evaluation ---
    logging.info("STEP 3: Evaluating baseline (un-optimized prompt)...")
    student = QAGeneratorModule(use_cot=args.use_cot)

    baseline_report = evaluate_predictions(
        student, devset,
        judge_model=args.judge_model,
        judge_api_base=judge_api_base,
        backend=args.backend,
    )
    baseline_report["stage"] = "baseline"
    baseline_report["model"] = args.model
    baseline_report["judge_model"] = args.judge_model
    baseline_report["timestamp"] = datetime.now().isoformat()

    save_report(baseline_report, "baseline", output_dir)

    logging.info(f"  Baseline results:")
    logging.info(f"    Avg composite quality: {baseline_report['avg_composite']:.3f}")
    logging.info(f"    Cognitive level:       {baseline_report.get('avg_cognitive_level', 0):.2f}/5")
    logging.info(f"    Naturalness:           {baseline_report.get('avg_naturalness', 0):.2f}/5")
    logging.info(f"    Groundedness:          {baseline_report.get('avg_groundedness', 0):.2f}/5")
    logging.info(f"    Information need:      {baseline_report.get('avg_information_need', 0):.2f}/5")
    logging.info(f"    Questions generated:   {baseline_report['num_questions']}")

    if args.skip_gepa:
        logging.info("Skipping GEPA optimization (--skip-gepa)")
        logging.info("EXPERIMENT COMPLETE (baseline only)")
        return 0

    # --- Step 4: GEPA optimization ---
    logging.info("STEP 4: Running GEPA optimization...")

    metric = qa_quality_metric_cached(
        judge_model=args.judge_model,
        judge_api_base=judge_api_base,
        backend=args.backend,
    )

    try:
        # Use the judge model for GEPA reflection (stronger model = better
        # prompt mutations).
        reflection_lm = make_lm(
            model=args.judge_model,
            api_base=judge_api_base,
            temperature=1.0,  # GEPA recommends higher temp for reflection
            max_tokens=8000,
        )

        gepa = dspy.GEPA(
            metric=metric,
            reflection_lm=reflection_lm,
            auto=args.gepa_auto,
        )

        optimized_program = gepa.compile(
            student,
            trainset=trainset,
            valset=devset,
        )
        logging.info("  GEPA optimization completed!")

    except Exception as e:
        logging.error(f"  GEPA optimization failed: {e}", exc_info=True)
        logging.info("EXPERIMENT COMPLETE (GEPA failed, baseline only)")
        return 1

    # --- Step 5: Evaluate optimized program ---
    logging.info("STEP 5: Evaluating optimized program...")

    optimized_report = evaluate_predictions(
        optimized_program, devset,
        judge_model=args.judge_model,
        judge_api_base=judge_api_base,
        backend=args.backend,
    )
    optimized_report["stage"] = "optimized"
    optimized_report["model"] = args.model
    optimized_report["judge_model"] = args.judge_model
    optimized_report["timestamp"] = datetime.now().isoformat()
    optimized_report["gepa_auto"] = args.gepa_auto

    save_report(optimized_report, "optimized", output_dir)

    # --- Step 6: Comparison ---
    logging.info("=" * 60)
    logging.info("RESULTS COMPARISON")
    logging.info("=" * 60)

    b = baseline_report
    o = optimized_report

    logging.info(f"  {'Metric':<25} {'Baseline':>10} {'Optimized':>10} {'Delta':>10}")
    logging.info(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    metrics = [
        ("Composite quality", "avg_composite"),
        ("Cognitive level", "avg_cognitive_level"),
        ("Naturalness", "avg_naturalness"),
        ("Groundedness", "avg_groundedness"),
        ("Information need", "avg_information_need"),
        ("Questions generated", "num_questions"),
    ]
    for label, key in metrics:
        bv = b.get(key, 0)
        ov = o.get(key, 0)
        delta = ov - bv
        if isinstance(bv, int):
            logging.info(f"  {label:<25} {bv:>10d} {ov:>10d} {delta:>+10d}")
        else:
            logging.info(f"  {label:<25} {bv:>10.3f} {ov:>10.3f} {delta:>+10.3f}")

    # Save comparison summary
    comparison = {
        "baseline": {k: b.get(k) for _, k in metrics},
        "optimized": {k: o.get(k) for _, k in metrics},
        "improvement": {
            k: o.get(k, 0) - b.get(k, 0)
            for _, k in metrics
            if isinstance(b.get(k, 0), (int, float))
        },
    }
    save_report(comparison, "comparison", output_dir)

    logging.info("EXPERIMENT COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
