#!/usr/bin/env python3
"""
Main pipeline orchestrator for CLAMS training data generation.
Based on NVIDIA ToolOrchestra data synthesis approach.

This script coordinates the full data preparation pipeline:
1. Generate synthetic tasks
2. Create tool-calling trajectories
3. Format data for SFT training
4. Optionally generate data for RLHF/DPO
"""

import json
import os
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from generate_tasks import CLAMSTaskGenerator, SyntheticTask
from prepare_sft_data import SFTDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipelineConfig:
    """Configuration for the data generation pipeline."""

    def __init__(
        self,
        num_tasks: int = 500,
        output_dir: str = "output",
        tools_file: str = "tools.json",
        seed: int = 42,
        balanced: bool = True,
        incremental: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ):
        self.num_tasks = num_tasks
        self.output_dir = Path(output_dir)
        self.tools_file = tools_file
        self.seed = seed
        self.balanced = balanced
        self.incremental = incremental
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 0.01, \
            "Splits must sum to 1.0"


class DataPipeline:
    """Main data generation pipeline."""

    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.task_generator = CLAMSTaskGenerator(config.tools_file)
        self.sft_generator = SFTDataGenerator(config.tools_file)

        # Set random seed
        random.seed(config.seed)

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_tasks(self) -> list:
        """Step 1: Generate synthetic tasks."""
        logger.info(f"Generating {self.config.num_tasks} synthetic tasks...")

        tasks = self.task_generator.generate_batch(
            self.config.num_tasks,
            balanced=self.config.balanced
        )

        # Save tasks
        tasks_file = self.config.output_dir / "tasks.jsonl"
        self.task_generator.save_tasks(tasks, str(tasks_file))

        logger.info(f"Generated {len(tasks)} tasks")
        return tasks

    def generate_trajectories(self, tasks: list) -> list:
        """Step 2: Generate tool-calling trajectories."""
        logger.info("Generating training trajectories...")

        trajectories = []
        for i, task in enumerate(tasks):
            if i % 50 == 0:
                logger.info(f"Processing task {i + 1}/{len(tasks)}...")

            trajectory = self.sft_generator.generate_trajectory(task)
            trajectories.append(trajectory)

        logger.info(f"Generated {len(trajectories)} trajectories")
        return trajectories

    def format_sft_data(self, trajectories: list) -> list:
        """Step 3: Format data for SFT training."""
        logger.info("Formatting SFT training data...")

        all_examples = []
        for trajectory in trajectories:
            if self.config.incremental:
                examples = self.sft_generator.generate_incremental_examples(trajectory)
                all_examples.extend(examples)
            else:
                all_examples.append(trajectory.to_dict())

        logger.info(f"Created {len(all_examples)} training examples")
        return all_examples

    def split_data(self, examples: list) -> Dict[str, list]:
        """Step 4: Split into train/val/test sets."""
        logger.info("Splitting data into train/val/test sets...")

        # Shuffle examples
        random.shuffle(examples)

        n = len(examples)
        train_end = int(n * self.config.train_split)
        val_end = train_end + int(n * self.config.val_split)

        splits = {
            "train": examples[:train_end],
            "val": examples[train_end:val_end],
            "test": examples[val_end:]
        }

        for split_name, split_data in splits.items():
            logger.info(f"  {split_name}: {len(split_data)} examples")

        return splits

    def save_splits(self, splits: Dict[str, list]):
        """Save data splits to files."""
        for split_name, split_data in splits.items():
            output_file = self.config.output_dir / f"{split_name}.jsonl"
            with open(output_file, 'w') as f:
                for example in split_data:
                    f.write(json.dumps(example) + '\n')
            logger.info(f"Saved {split_name} split to {output_file}")

    def generate_metadata(self, tasks: list, examples: list, splits: Dict[str, list]) -> Dict:
        """Generate metadata about the dataset."""
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "num_tasks": self.config.num_tasks,
                "seed": self.config.seed,
                "balanced": self.config.balanced,
                "incremental": self.config.incremental,
            },
            "statistics": {
                "total_tasks": len(tasks),
                "total_examples": len(examples),
                "splits": {k: len(v) for k, v in splits.items()},
            },
            "task_categories": {},
            "difficulty_distribution": {},
            "tools_usage": {},
        }

        # Calculate distributions
        for task in tasks:
            cat = task.category
            metadata["task_categories"][cat] = metadata["task_categories"].get(cat, 0) + 1

            diff = task.difficulty
            metadata["difficulty_distribution"][diff] = \
                metadata["difficulty_distribution"].get(diff, 0) + 1

            for tool in task.tools_sequence:
                metadata["tools_usage"][tool] = metadata["tools_usage"].get(tool, 0) + 1

        return metadata

    def run(self) -> Dict[str, Any]:
        """Run the complete data generation pipeline."""
        logger.info("=" * 60)
        logger.info("CLAMS Training Data Generation Pipeline")
        logger.info("=" * 60)

        # Step 1: Generate tasks
        tasks = self.generate_tasks()

        # Step 2: Generate trajectories
        trajectories = self.generate_trajectories(tasks)

        # Step 3: Format SFT data
        examples = self.format_sft_data(trajectories)

        # Step 4: Split data
        splits = self.split_data(examples)

        # Step 5: Save splits
        self.save_splits(splits)

        # Step 6: Generate and save metadata
        metadata = self.generate_metadata(tasks, examples, splits)
        metadata_file = self.config.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")

        # Print summary
        self._print_summary(metadata)

        return metadata

    def _print_summary(self, metadata: Dict):
        """Print pipeline summary."""
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)

        print(f"\nOutput directory: {self.config.output_dir}")

        print("\nData Statistics:")
        print(f"  Total tasks: {metadata['statistics']['total_tasks']}")
        print(f"  Total examples: {metadata['statistics']['total_examples']}")

        print("\nSplits:")
        for split, count in metadata['statistics']['splits'].items():
            print(f"  {split}: {count}")

        print("\nTask Categories:")
        for cat, count in sorted(metadata['task_categories'].items()):
            print(f"  {cat}: {count}")

        print("\nDifficulty Distribution:")
        for diff, count in sorted(metadata['difficulty_distribution'].items()):
            print(f"  {diff}: {count}")

        print("\nTop Tools Used:")
        sorted_tools = sorted(metadata['tools_usage'].items(), key=lambda x: -x[1])
        for tool, count in sorted_tools[:5]:
            print(f"  {tool}: {count}")


def generate_dpo_pairs(sft_data_path: str, output_path: str, num_pairs: int = 100):
    """
    Generate DPO (Direct Preference Optimization) training pairs.

    Creates pairs of (chosen, rejected) responses for preference learning.
    """
    logger.info(f"Generating {num_pairs} DPO pairs...")

    # Load SFT data
    examples = []
    with open(sft_data_path) as f:
        for line in f:
            examples.append(json.loads(line))

    dpo_pairs = []

    for _ in range(num_pairs):
        # Select a random example
        example = random.choice(examples)
        messages = example["messages"]

        # Find assistant messages with tool calls
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant" and "<tool_call>" in msg["content"]:
                # Create a rejected version with wrong tool or parameters
                chosen = msg["content"]

                # Generate rejected response (wrong approach)
                rejected_variants = [
                    # Wrong tool order
                    chosen.replace("swt_detection", "whisper_transcription"),
                    # Missing thinking
                    chosen.split("</thinking>")[-1] if "</thinking>" in chosen else chosen,
                    # Wrong parameters
                    chosen.replace('"use_classifier": true', '"use_classifier": false'),
                    # Incomplete response
                    chosen[:len(chosen) // 2] + "...",
                ]

                rejected = random.choice(rejected_variants)

                if chosen != rejected:
                    dpo_pairs.append({
                        "prompt": messages[:i],
                        "chosen": chosen,
                        "rejected": rejected,
                        "task_id": example.get("task_id", "unknown")
                    })
                break

    # Save DPO pairs
    with open(output_path, 'w') as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + '\n')

    logger.info(f"Saved {len(dpo_pairs)} DPO pairs to {output_path}")
    return dpo_pairs


def main():
    parser = argparse.ArgumentParser(
        description="CLAMS Training Data Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default dataset (500 tasks)
  python run_pipeline.py

  # Generate larger dataset
  python run_pipeline.py --num-tasks 1000 --output-dir large_dataset

  # Generate with specific seed for reproducibility
  python run_pipeline.py --seed 123 --num-tasks 200

  # Generate DPO pairs from existing SFT data
  python run_pipeline.py --dpo --sft-input output/train.jsonl
        """
    )

    parser.add_argument("--num-tasks", type=int, default=500,
                        help="Number of synthetic tasks to generate")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory for generated data")
    parser.add_argument("--tools", type=str, default="tools.json",
                        help="Path to tools definition file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--balanced", action="store_true", default=True,
                        help="Balance tasks across categories")
    parser.add_argument("--no-incremental", action="store_true",
                        help="Don't generate incremental examples")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Fraction for training set")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction for validation set")
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="Fraction for test set")

    # DPO generation
    parser.add_argument("--dpo", action="store_true",
                        help="Generate DPO training pairs")
    parser.add_argument("--sft-input", type=str,
                        help="Input SFT data for DPO pair generation")
    parser.add_argument("--dpo-pairs", type=int, default=100,
                        help="Number of DPO pairs to generate")

    args = parser.parse_args()

    if args.dpo:
        # Generate DPO pairs from existing SFT data
        if not args.sft_input:
            args.sft_input = os.path.join(args.output_dir, "train.jsonl")

        output_path = os.path.join(args.output_dir, "dpo_pairs.jsonl")
        generate_dpo_pairs(args.sft_input, output_path, args.dpo_pairs)
    else:
        # Run full pipeline
        config = DataPipelineConfig(
            num_tasks=args.num_tasks,
            output_dir=args.output_dir,
            tools_file=args.tools,
            seed=args.seed,
            balanced=args.balanced,
            incremental=not args.no_incremental,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
        )

        pipeline = DataPipeline(config)
        pipeline.run()


if __name__ == "__main__":
    main()
