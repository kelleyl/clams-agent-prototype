#!/usr/bin/env python3
"""
Prepare Supervised Fine-Tuning (SFT) data for CLAMS orchestrator training.
Based on NVIDIA ToolOrchestra data preparation approach.

This script converts synthetic tasks into conversation-format training examples
suitable for instruction-tuning language models.
"""

import json
import os
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from generate_tasks import CLAMSTaskGenerator, ToolResultSimulator, SyntheticTask

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# System prompt for the orchestrator
SYSTEM_PROMPT = """You are a CLAMS (Computational Linguistics Applications for Multimedia Services) pipeline orchestrator. Your role is to analyze multimedia content by coordinating specialized tools for video/audio analysis.

# Available Tools

You have access to the following tools:

<tools>
{tools_xml}
</tools>

# How to Use Tools

To call a tool, use this format:
<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

# Guidelines

1. Analyze the user's request to understand what information they need
2. Plan the sequence of tools needed to accomplish the task
3. Execute tools in order, using outputs from previous tools as inputs
4. Handle partial results or errors gracefully
5. Provide a clear summary of extracted information

Always explain your reasoning before making tool calls."""


def format_tools_xml(tools: Dict[str, Any]) -> str:
    """Format tools as XML for system prompt."""
    xml_parts = []
    for tool_name, tool_def in tools.items():
        params_json = json.dumps(tool_def.get("parameters", {}), indent=2)
        xml_parts.append(f'''<tool>
  <name>{tool_name}</name>
  <description>{tool_def.get("description", "")}</description>
  <parameters>{params_json}</parameters>
</tool>''')
    return "\n".join(xml_parts)


@dataclass
class TrainingMessage:
    """A single message in a training conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class TrainingExample:
    """A complete training example with conversation history."""
    task_id: str
    messages: List[TrainingMessage]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "metadata": self.metadata
        }

    def to_chatml(self) -> str:
        """Convert to ChatML format for training."""
        parts = []
        for msg in self.messages:
            parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
        return "\n".join(parts)


class SFTDataGenerator:
    """Generates SFT training data from synthetic tasks."""

    def __init__(self, tools_path: str = "tools.json"):
        with open(tools_path) as f:
            data = json.load(f)
        self.tools = {tool["name"]: tool for tool in data["tools"]}
        self.simulator = ToolResultSimulator(tools_path)
        self.tools_xml = format_tools_xml(self.tools)

    def _format_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Format a tool call."""
        return f'''<tool_call>
{json.dumps({"name": tool_name, "arguments": arguments}, indent=2)}
</tool_call>'''

    def _format_tool_result(self, result: Dict[str, Any]) -> str:
        """Format tool execution result."""
        return f"[Tool Result]\n{json.dumps(result, indent=2)}"

    def _generate_thinking(self, task: SyntheticTask, step: int, total_steps: int) -> str:
        """Generate thinking/reasoning text."""
        if step == 0:
            # Initial planning
            tools_list = ", ".join(task.tools_sequence)
            constraints_text = "\n".join(f"- {c}" for c in task.constraints)
            return f"""Let me analyze this request. The user wants to {task.category.replace("_", " ")}.

To accomplish this, I'll need to use the following tools in sequence:
{tools_list}

Key constraints to consider:
{constraints_text}

Let me start with {task.tools_sequence[0]}."""
        elif step >= total_steps:
            # Final synthesis (after all tools have run)
            return "Now I have all the information needed. Let me compile the final results."
        elif step == total_steps - 1:
            # Last tool step
            current_tool = task.tools_sequence[step]
            prev_tool = task.tools_sequence[step - 1] if step > 0 else None
            if prev_tool:
                return f"The {prev_tool} results look good. Now I'll use {current_tool} to complete the analysis."
            else:
                return f"Let me use {current_tool} to complete the analysis."
        else:
            # Intermediate step
            current_tool = task.tools_sequence[step]
            prev_tool = task.tools_sequence[step - 1] if step > 0 else None
            if prev_tool:
                return f"The {prev_tool} results look good. Now I'll use {current_tool} to process this further."
            else:
                return f"Let me start with {current_tool}."

    def _generate_final_response(self, task: SyntheticTask, results: List[Dict]) -> str:
        """Generate final assistant response summarizing results."""
        category_responses = {
            "slate_extraction": """Based on my analysis of the video slate, here is the extracted metadata:

| Field | Value |
|-------|-------|
| Title | {title} |
| Date | {date} |
| Episode | {episode} |
| Producer | {producer} |

The slate was detected with {confidence}% confidence in the first few seconds of the video.""",

            "chyron_extraction": """I've identified the following individuals from the on-screen chyrons:

{persons_list}

A total of {count} unique individuals were identified with their names and affiliations.""",

            "credits_analysis": """Here are the extracted credits from the video:

{credits_list}

I identified {count} production roles from the credits sequence.""",

            "transcription": """I've completed the transcription. Here's a summary:

- Duration: {duration}
- Language: English
- Segments: {num_segments}

The full transcript has been saved with timestamps for each segment.""",

            "content_indexing": """I've created a comprehensive index of the video content:

- Scenes detected: {num_scenes}
- Text segments found: {num_text}
- Speech segments: {num_speech}
- Named entities: {num_entities}

The content is now fully indexed and searchable.""",

            "entity_extraction": """Here are the key entities extracted from the video:

**People:**
{persons}

**Organizations:**
{orgs}

**Locations:**
{locations}

Total entities found: {total}""",

            "quality_control": """Quality control check complete:

| Field | Catalog | Video | Status |
|-------|---------|-------|--------|
| Title | {cat_title} | {vid_title} | {title_status} |
| Date | {cat_date} | {vid_date} | {date_status} |

Overall match confidence: {confidence}%"""
        }

        template = category_responses.get(task.category, "Analysis complete. Results have been extracted successfully.")

        # Fill in template with simulated values
        filled = template.format(
            title="EVENING NEWS",
            date="1985-03-15",
            episode="247",
            producer="WGBH Boston",
            confidence=random.randint(88, 97),
            persons_list="1. John Smith - Professor, Harvard University\n2. Sarah Johnson - Director, MIT",
            count=random.randint(3, 8),
            credits_list="- Director: John Smith\n- Producer: Sarah Johnson\n- Writer: Michael Williams",
            duration=f"{random.randint(10, 60)}:{random.randint(0,59):02d}",
            num_segments=random.randint(20, 100),
            num_scenes=random.randint(10, 50),
            num_text=random.randint(5, 20),
            num_speech=random.randint(20, 100),
            num_entities=random.randint(10, 50),
            persons="- John Smith\n- Sarah Johnson",
            orgs="- Harvard University\n- MIT",
            locations="- Boston\n- Cambridge",
            total=random.randint(15, 40),
            cat_title="Evening News",
            vid_title="EVENING NEWS",
            title_status="✓ Match",
            cat_date="1985-03-15",
            vid_date="March 15, 1985",
            date_status="✓ Match",
        )

        return filled

    def generate_trajectory(self, task: SyntheticTask) -> TrainingExample:
        """Generate a complete training trajectory for a task."""
        messages = []

        # System message with tools
        system_content = SYSTEM_PROMPT.format(tools_xml=self.tools_xml)
        messages.append(TrainingMessage(role="system", content=system_content))

        # User request
        user_content = task.user_request
        if task.video_context:
            user_content += f"\n\nVideo file: {task.video_context.get('video_path', 'input.mp4')}"
        messages.append(TrainingMessage(role="user", content=user_content))

        # Generate tool calling trajectory
        previous_results = []
        assistant_parts = []

        for i, tool_name in enumerate(task.tools_sequence):
            # Add thinking
            thinking = self._generate_thinking(task, i, len(task.tools_sequence))
            assistant_parts.append(f"<thinking>\n{thinking}\n</thinking>")

            # Simulate tool execution
            result = self.simulator.simulate_tool(
                tool_name,
                task.video_context,
                task.category,
                previous_results
            )
            previous_results.append(result)

            # Build tool arguments
            tool_args = {"input_path": task.video_context.get("video_path", "input.mp4")}

            # Add relevant arguments based on tool
            if tool_name == "doctr_ocr" and previous_results:
                for pr in previous_results:
                    if pr.tool_name == "swt_detection":
                        tool_args["timeframes"] = pr.output.get("timeframes", [])[:2]

            if tool_name in ["slate_parser", "chyron_parser", "credits_parser"]:
                for pr in previous_results:
                    if pr.tool_name == "doctr_ocr":
                        tool_args = {"ocr_text": pr.output.get("full_text", "")}

            # Add tool call
            tool_call = self._format_tool_call(tool_name, tool_args)
            assistant_parts.append(tool_call)

            # Create assistant message with tool call
            assistant_content = "\n\n".join(assistant_parts)
            messages.append(TrainingMessage(role="assistant", content=assistant_content))

            # Add tool result as user message
            tool_result = self._format_tool_result(result.output)
            messages.append(TrainingMessage(role="user", content=tool_result))

            # Reset for next iteration
            assistant_parts = []

        # Final response
        final_thinking = self._generate_thinking(task, len(task.tools_sequence), len(task.tools_sequence))
        final_response = self._generate_final_response(task, [r.output for r in previous_results])
        final_content = f"<thinking>\n{final_thinking}\n</thinking>\n\n{final_response}"
        messages.append(TrainingMessage(role="assistant", content=final_content))

        return TrainingExample(
            task_id=task.task_id,
            messages=messages,
            metadata={
                "category": task.category,
                "difficulty": task.difficulty,
                "tools_used": task.tools_sequence,
                "num_turns": len(task.tools_sequence) + 1
            }
        )

    def generate_incremental_examples(self, trajectory: TrainingExample) -> List[Dict]:
        """
        Generate incremental training examples from a trajectory.
        Each example ends at a different point, training the model to generate
        the next response given history.
        """
        examples = []

        # Skip system message, start from first user message
        for i in range(2, len(trajectory.messages), 2):
            # Include messages up to and including the assistant response
            if i + 1 <= len(trajectory.messages):
                partial_messages = trajectory.messages[:i + 1]
                example = {
                    "task_id": f"{trajectory.task_id}_turn_{i // 2}",
                    "messages": [{"role": m.role, "content": m.content} for m in partial_messages],
                    "metadata": {
                        **trajectory.metadata,
                        "turn": i // 2,
                        "is_final": i + 1 >= len(trajectory.messages)
                    }
                }
                examples.append(example)

        return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT training data for CLAMS orchestrator")
    parser.add_argument("--input", type=str, default="tasks.jsonl", help="Input tasks file")
    parser.add_argument("--output", type=str, default="sft_data.jsonl", help="Output training data file")
    parser.add_argument("--output-dir", type=str, default="sft_examples", help="Directory for individual examples")
    parser.add_argument("--tools", type=str, default="tools.json", help="Tools definition file")
    parser.add_argument("--format", type=str, choices=["jsonl", "chatml", "both"], default="jsonl",
                        help="Output format")
    parser.add_argument("--incremental", action="store_true",
                        help="Generate incremental training examples")
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="Number of tasks to process (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    # Initialize generators
    sft_generator = SFTDataGenerator(args.tools)

    # Load or generate tasks
    if os.path.exists(args.input):
        logger.info(f"Loading tasks from {args.input}")
        tasks = []
        with open(args.input) as f:
            for line in f:
                data = json.loads(line)
                task = SyntheticTask(**data)
                tasks.append(task)
    else:
        logger.info("No input file found, generating tasks...")
        task_generator = CLAMSTaskGenerator(args.tools)
        num_tasks = args.num_tasks or 100
        tasks = task_generator.generate_batch(num_tasks, balanced=True)
        task_generator.save_tasks(tasks, args.input)

    if args.num_tasks:
        tasks = tasks[:args.num_tasks]

    logger.info(f"Processing {len(tasks)} tasks...")

    # Generate training data
    all_examples = []
    trajectories = []

    for task in tasks:
        trajectory = sft_generator.generate_trajectory(task)
        trajectories.append(trajectory)

        if args.incremental:
            examples = sft_generator.generate_incremental_examples(trajectory)
            all_examples.extend(examples)
        else:
            all_examples.append(trajectory.to_dict())

    # Save outputs
    if args.format in ["jsonl", "both"]:
        with open(args.output, 'w') as f:
            for example in all_examples:
                f.write(json.dumps(example) + '\n')
        logger.info(f"Saved {len(all_examples)} examples to {args.output}")

    if args.format in ["chatml", "both"]:
        os.makedirs(args.output_dir, exist_ok=True)
        for i, traj in enumerate(trajectories):
            output_path = os.path.join(args.output_dir, f"example_{i:04d}.txt")
            with open(output_path, 'w') as f:
                f.write(traj.to_chatml())
        logger.info(f"Saved {len(trajectories)} ChatML examples to {args.output_dir}/")

    # Print summary
    print("\nSFT Data Generation Summary:")
    print(f"Total tasks processed: {len(tasks)}")
    print(f"Total training examples: {len(all_examples)}")

    if args.incremental:
        turns = [e["metadata"]["turn"] for e in all_examples]
        print(f"Examples by turn: {dict(zip(*sorted(zip(*[[t, turns.count(t)] for t in set(turns)]))))}")

    categories = {}
    for task in tasks:
        categories[task.category] = categories.get(task.category, 0) + 1
    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
