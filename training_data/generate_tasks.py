#!/usr/bin/env python3
"""
Synthetic task generator for CLAMS orchestrator training.
Based on NVIDIA ToolOrchestra data synthesis approach.
"""

import json
import os
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Task templates for different categories
TASK_TEMPLATES = {
    "slate_extraction": [
        "Extract the production metadata from the opening slate of this {genre} video from {decade}.",
        "I need to catalog this archival footage. Please identify the show title, air date, and production company from the slate.",
        "This {format} has a slate at the beginning. Extract all readable metadata fields.",
        "Parse the production slate to get episode number, title, and broadcast date for our archive catalog.",
        "We're digitizing {decade} {genre} footage. Extract slate information for the catalog record.",
    ],
    "chyron_extraction": [
        "Identify all the speakers in this {genre} video by extracting the lower-third chyrons.",
        "Extract name and title information from all on-screen graphics in this interview.",
        "Find all chyrons that identify people and extract their names and affiliations.",
        "This news segment has multiple guests. Extract their names from the lower-thirds.",
        "Parse the name supers to identify who appears in this {format}.",
    ],
    "credits_analysis": [
        "Extract the complete cast and crew list from the end credits of this {genre}.",
        "Parse the closing credits to get all production personnel with their roles.",
        "I need to identify who worked on this production from the credits sequence.",
        "Extract all names and roles from the scrolling end credits.",
        "Create a structured crew list from the credits of this {decade} {format}.",
    ],
    "transcription": [
        "Transcribe the audio from this {genre} video with timestamps.",
        "Generate a searchable transcript of this {format} for our archive.",
        "Transcribe this interview and identify speakers based on the chyrons.",
        "Create a time-coded transcript of the narration in this documentary.",
        "Transcribe the speech in this {decade} archival footage.",
    ],
    "content_indexing": [
        "Create a searchable index of this {genre} video including scenes, text, and speech.",
        "Index all text appearing on screen in this {format} for search.",
        "Generate a comprehensive content index for this archival video.",
        "Identify and index all on-screen text, graphics, and spoken content.",
        "Create metadata for full-text search of this {decade} footage.",
    ],
    "entity_extraction": [
        "Identify all people, organizations, and locations mentioned in this {genre}.",
        "Extract named entities from both the speech and on-screen text.",
        "Find all person names mentioned or shown in this {format}.",
        "Extract dates, places, and organization names from this archival footage.",
        "Identify key entities for subject cataloging of this {decade} video.",
    ],
    "quality_control": [
        "Verify that the catalog metadata matches the slate information in this video.",
        "Check if the episode number in our records matches what's shown on screen.",
        "Validate the credited producer against the production slate.",
        "Compare the transcript against the on-screen captions for accuracy.",
        "Verify speaker identifications against the chyron text.",
    ],
}

# Variables for template filling
GENRES = ["news broadcast", "documentary", "interview", "talk show", "public affairs program",
          "educational video", "historical footage", "archival recording"]
DECADES = ["1960s", "1970s", "1980s", "1990s", "2000s"]
FORMATS = ["television program", "video recording", "broadcast segment", "archived footage",
           "digitized tape", "news clip", "interview segment"]


@dataclass
class SyntheticTask:
    """Represents a synthetic training task."""
    task_id: str
    user_request: str
    category: str
    tools_sequence: List[str]
    parameters: Dict[str, Any]
    constraints: List[str]
    expected_output: str
    difficulty: str
    video_context: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulatedToolResult:
    """Simulated output from a CLAMS tool."""
    tool_name: str
    success: bool
    output: Dict[str, Any]
    confidence: float
    processing_time: float


class CLAMSTaskGenerator:
    """Generates synthetic CLAMS orchestration tasks."""

    def __init__(self, tools_path: str = "tools.json"):
        self.tools_path = Path(tools_path)
        self.tools = self._load_tools()
        self.task_counter = 0

    def _load_tools(self) -> Dict[str, Any]:
        """Load tool definitions."""
        with open(self.tools_path) as f:
            data = json.load(f)
        return {tool["name"]: tool for tool in data["tools"]}

    def _fill_template(self, template: str) -> str:
        """Fill template with random values."""
        return template.format(
            genre=random.choice(GENRES),
            decade=random.choice(DECADES),
            format=random.choice(FORMATS)
        )

    def _get_tool_sequence(self, category: str) -> List[str]:
        """Get appropriate tool sequence for category."""
        sequences = {
            "slate_extraction": ["swt_detection", "doctr_ocr", "slate_parser"],
            "chyron_extraction": ["swt_detection", "doctr_ocr", "chyron_parser"],
            "credits_analysis": ["swt_detection", "doctr_ocr", "credits_parser"],
            "transcription": ["whisper_transcription", "spacy_ner"],
            "content_indexing": ["scene_detection", "swt_detection", "whisper_transcription", "doctr_ocr", "spacy_ner"],
            "entity_extraction": ["whisper_transcription", "swt_detection", "doctr_ocr", "spacy_ner"],
            "quality_control": ["swt_detection", "doctr_ocr", "slate_parser"],
        }
        base_sequence = sequences.get(category, ["swt_detection", "doctr_ocr"])

        # Sometimes add variation
        if random.random() < 0.3:
            # Add an optional tool
            optional_tools = ["scene_detection", "forced_alignment", "smolvlm2_captioner"]
            base_sequence = base_sequence + [random.choice(optional_tools)]

        return base_sequence

    def _generate_constraints(self, category: str) -> List[str]:
        """Generate realistic constraints for a task."""
        all_constraints = {
            "slate_extraction": [
                "Slate appears in first 30 seconds",
                "Date format may be MM/DD/YYYY or written out",
                "Some fields may be partially obscured",
                "Multiple slates possible (head and tail)",
            ],
            "chyron_extraction": [
                "Chyrons may appear multiple times for same person",
                "Some lower-thirds only show name, no title",
                "Text may be in ALL CAPS",
                "Graphics may overlap with other content",
            ],
            "credits_analysis": [
                "Credits may be scrolling or static",
                "Role/name order may vary",
                "Some entries span multiple lines",
                "May include both cast and crew",
            ],
            "transcription": [
                "Multiple speakers present",
                "Background noise may affect accuracy",
                "Some speech may overlap",
                "Includes both narration and dialogue",
            ],
            "content_indexing": [
                "Video is over 30 minutes long",
                "Mix of speech, text, and graphics",
                "Some segments have no audio",
                "Quality varies throughout",
            ],
            "entity_extraction": [
                "Names may appear in both audio and text",
                "Deduplicate entities across sources",
                "Include confidence scores",
                "Flag uncertain extractions",
            ],
            "quality_control": [
                "Compare against existing catalog record",
                "Flag discrepancies",
                "Report confidence levels",
                "Note missing information",
            ],
        }

        constraints = all_constraints.get(category, [])
        # Select 2-3 random constraints
        num_constraints = random.randint(2, min(3, len(constraints)))
        return random.sample(constraints, num_constraints)

    def _generate_expected_output(self, category: str) -> str:
        """Generate expected output description."""
        outputs = {
            "slate_extraction": "JSON with fields: title, date, episode, producer, network",
            "chyron_extraction": "List of {name, title, organization, timestamp} objects",
            "credits_analysis": "Structured list of {name, role} pairs organized by category",
            "transcription": "Time-coded transcript in WebVTT or SRT format",
            "content_indexing": "Searchable index with timestamps for text, speech, and scenes",
            "entity_extraction": "Named entities with types, mentions, and confidence scores",
            "quality_control": "Validation report with matches, mismatches, and missing fields",
        }
        return outputs.get(category, "Structured JSON output")

    def _generate_video_context(self) -> Dict[str, str]:
        """Generate simulated video file context."""
        video_id = f"video_{random.randint(1000, 9999)}"
        return {
            "video_path": f"/data/archive/{video_id}.mp4",
            "duration_seconds": random.randint(60, 3600),
            "format": random.choice(["NTSC", "PAL", "HD"]),
            "year": str(random.randint(1965, 2010)),
            "collection": random.choice(["news_archive", "documentary", "public_affairs", "interviews"]),
        }

    def generate_task(self, category: Optional[str] = None) -> SyntheticTask:
        """Generate a single synthetic task."""
        if category is None:
            category = random.choice(list(TASK_TEMPLATES.keys()))

        template = random.choice(TASK_TEMPLATES[category])
        user_request = self._fill_template(template)

        tools_sequence = self._get_tool_sequence(category)
        constraints = self._generate_constraints(category)
        expected_output = self._generate_expected_output(category)
        video_context = self._generate_video_context()

        # Determine difficulty based on number of tools and constraints
        if len(tools_sequence) <= 2:
            difficulty = "easy"
        elif len(tools_sequence) <= 4:
            difficulty = "medium"
        else:
            difficulty = "hard"

        self.task_counter += 1

        return SyntheticTask(
            task_id=f"task_{self.task_counter:04d}",
            user_request=user_request,
            category=category,
            tools_sequence=tools_sequence,
            parameters={tool: {} for tool in tools_sequence},  # Will be filled during trajectory
            constraints=constraints,
            expected_output=expected_output,
            difficulty=difficulty,
            video_context=video_context,
        )

    def generate_batch(self, num_tasks: int, balanced: bool = True) -> List[SyntheticTask]:
        """Generate a batch of tasks."""
        tasks = []

        if balanced:
            # Generate equal number from each category
            categories = list(TASK_TEMPLATES.keys())
            tasks_per_category = num_tasks // len(categories)
            remainder = num_tasks % len(categories)

            for category in categories:
                for _ in range(tasks_per_category):
                    tasks.append(self.generate_task(category))

            # Add remainder from random categories
            for _ in range(remainder):
                tasks.append(self.generate_task())
        else:
            for _ in range(num_tasks):
                tasks.append(self.generate_task())

        random.shuffle(tasks)
        return tasks

    def save_tasks(self, tasks: List[SyntheticTask], output_path: str):
        """Save tasks to JSONL file."""
        with open(output_path, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task.to_dict()) + '\n')
        logger.info(f"Saved {len(tasks)} tasks to {output_path}")


class ToolResultSimulator:
    """Simulates realistic tool outputs for training data."""

    def __init__(self, tools_path: str = "tools.json"):
        with open(tools_path) as f:
            data = json.load(f)
        self.tools = {tool["name"]: tool for tool in data["tools"]}

    def simulate_swt_detection(self, video_context: Dict, category: str) -> SimulatedToolResult:
        """Simulate SWT detection results."""
        timeframes = []

        if category in ["slate_extraction", "quality_control"]:
            # Add slate at beginning
            timeframes.append({
                "start": 0,
                "end": random.randint(3000, 8000),
                "label": "slate",
                "confidence": round(random.uniform(0.85, 0.98), 2)
            })

        if category in ["chyron_extraction", "entity_extraction"]:
            # Add multiple chyrons throughout
            num_chyrons = random.randint(3, 8)
            current_time = random.randint(10000, 30000)
            for _ in range(num_chyrons):
                timeframes.append({
                    "start": current_time,
                    "end": current_time + random.randint(3000, 8000),
                    "label": "chyron",
                    "confidence": round(random.uniform(0.75, 0.95), 2)
                })
                current_time += random.randint(30000, 120000)

        if category == "credits_analysis":
            # Add credits at end
            duration = video_context.get("duration_seconds", 1800) * 1000
            credits_start = duration - random.randint(60000, 180000)
            timeframes.append({
                "start": credits_start,
                "end": duration,
                "label": "credits",
                "confidence": round(random.uniform(0.88, 0.97), 2)
            })

        return SimulatedToolResult(
            tool_name="swt_detection",
            success=True,
            output={"timeframes": timeframes},
            confidence=sum(tf["confidence"] for tf in timeframes) / max(len(timeframes), 1),
            processing_time=round(random.uniform(15, 45), 1)
        )

    def simulate_doctr_ocr(self, timeframes: List[Dict], category: str) -> SimulatedToolResult:
        """Simulate OCR results."""
        text_regions = []

        for tf in timeframes:
            if tf["label"] == "slate":
                text_regions.extend([
                    {"text": random.choice(["EVENING NEWS", "DOCUMENTARY SPECIAL", "PUBLIC AFFAIRS", "NEWS HOUR"]),
                     "confidence": round(random.uniform(0.90, 0.98), 2)},
                    {"text": f"{random.choice(['January', 'March', 'June', 'October'])} {random.randint(1, 28)}, {random.randint(1975, 2005)}",
                     "confidence": round(random.uniform(0.85, 0.95), 2)},
                    {"text": f"Episode {random.randint(100, 999)}",
                     "confidence": round(random.uniform(0.88, 0.96), 2)},
                    {"text": random.choice(["WGBH Boston", "PBS", "WNET New York", "KQED San Francisco"]),
                     "confidence": round(random.uniform(0.90, 0.97), 2)},
                ])
            elif tf["label"] == "chyron":
                first_names = ["John", "Sarah", "Michael", "Emily", "David", "Jennifer", "Robert", "Lisa"]
                last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
                titles = ["Senator", "Professor", "Dr.", "Director", "CEO", "Author", "Analyst", "Reporter"]
                orgs = ["Harvard University", "Stanford University", "The Washington Post", "MIT", "Brookings Institution"]

                text_regions.append({
                    "text": f"{random.choice(first_names)} {random.choice(last_names)}",
                    "confidence": round(random.uniform(0.88, 0.97), 2)
                })
                if random.random() > 0.3:
                    text_regions.append({
                        "text": f"{random.choice(titles)}, {random.choice(orgs)}",
                        "confidence": round(random.uniform(0.82, 0.94), 2)
                    })
            elif tf["label"] == "credits":
                roles = ["Directed by", "Produced by", "Written by", "Executive Producer", "Editor", "Camera"]
                for role in random.sample(roles, random.randint(3, 6)):
                    first_names = ["John", "Sarah", "Michael", "Emily", "David", "Jennifer"]
                    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"]
                    text_regions.append({
                        "text": f"{role}\n{random.choice(first_names)} {random.choice(last_names)}",
                        "confidence": round(random.uniform(0.80, 0.95), 2)
                    })

        full_text = "\n".join(tr["text"] for tr in text_regions)

        return SimulatedToolResult(
            tool_name="doctr_ocr",
            success=True,
            output={"text_regions": text_regions, "full_text": full_text},
            confidence=sum(tr["confidence"] for tr in text_regions) / max(len(text_regions), 1),
            processing_time=round(random.uniform(3, 12), 1)
        )

    def simulate_parser(self, parser_type: str, ocr_text: str) -> SimulatedToolResult:
        """Simulate text parser results."""
        if parser_type == "slate_parser":
            output = {
                "parsed_fields": {
                    "title": "EVENING NEWS",
                    "date": f"{random.randint(1975, 2005)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                    "episode": str(random.randint(100, 999)),
                    "producer": random.choice(["WGBH Boston", "PBS", "WNET New York"]),
                },
                "confidence": round(random.uniform(0.85, 0.95), 2)
            }
        elif parser_type == "chyron_parser":
            output = {
                "persons": [
                    {"name": "John Smith", "title": "Professor", "organization": "Harvard University"},
                    {"name": "Sarah Johnson", "title": "Director", "organization": "MIT"},
                ],
                "confidence": round(random.uniform(0.82, 0.93), 2)
            }
        elif parser_type == "credits_parser":
            output = {
                "credits": [
                    {"role": "Director", "name": "John Smith"},
                    {"role": "Producer", "name": "Sarah Johnson"},
                    {"role": "Writer", "name": "Michael Williams"},
                ],
                "confidence": round(random.uniform(0.80, 0.92), 2)
            }
        else:
            output = {"parsed": True}

        return SimulatedToolResult(
            tool_name=parser_type,
            success=True,
            output=output,
            confidence=output.get("confidence", 0.9),
            processing_time=round(random.uniform(1, 5), 1)
        )

    def simulate_whisper(self, video_context: Dict) -> SimulatedToolResult:
        """Simulate Whisper transcription results."""
        duration = video_context.get("duration_seconds", 300)

        # Generate sample transcript segments
        segments = []
        current_time = 0
        sample_texts = [
            "Good evening and welcome to tonight's program.",
            "Today we'll be discussing the latest developments.",
            "Joining us now is our first guest.",
            "Thank you for having me.",
            "Let's start with the key findings.",
            "That's an excellent question.",
            "The research shows significant progress.",
            "We'll continue after this break.",
        ]

        while current_time < duration:
            segment_duration = random.uniform(2, 8)
            segments.append({
                "start": round(current_time, 2),
                "end": round(current_time + segment_duration, 2),
                "text": random.choice(sample_texts),
                "confidence": round(random.uniform(0.85, 0.98), 2)
            })
            current_time += segment_duration + random.uniform(0.5, 2)
            if len(segments) > 20:  # Limit for demo
                break

        return SimulatedToolResult(
            tool_name="whisper_transcription",
            success=True,
            output={
                "segments": segments,
                "language": "en",
                "full_text": " ".join(s["text"] for s in segments)
            },
            confidence=sum(s["confidence"] for s in segments) / len(segments),
            processing_time=round(duration * 0.1, 1)  # ~10% of video duration
        )

    def simulate_tool(self, tool_name: str, video_context: Dict,
                      category: str, previous_results: List[SimulatedToolResult]) -> SimulatedToolResult:
        """Simulate any tool based on name."""
        if tool_name == "swt_detection":
            return self.simulate_swt_detection(video_context, category)
        elif tool_name == "doctr_ocr":
            # Get timeframes from previous SWT result
            timeframes = []
            for pr in previous_results:
                if pr.tool_name == "swt_detection":
                    timeframes = pr.output.get("timeframes", [])
            return self.simulate_doctr_ocr(timeframes, category)
        elif tool_name in ["slate_parser", "chyron_parser", "credits_parser"]:
            # Get OCR text from previous result
            ocr_text = ""
            for pr in previous_results:
                if pr.tool_name == "doctr_ocr":
                    ocr_text = pr.output.get("full_text", "")
            return self.simulate_parser(tool_name, ocr_text)
        elif tool_name == "whisper_transcription":
            return self.simulate_whisper(video_context)
        elif tool_name == "spacy_ner":
            # Simulate NER on available text
            return SimulatedToolResult(
                tool_name="spacy_ner",
                success=True,
                output={
                    "entities": [
                        {"text": "John Smith", "label": "PERSON", "confidence": 0.95},
                        {"text": "Harvard University", "label": "ORG", "confidence": 0.92},
                        {"text": "Boston", "label": "GPE", "confidence": 0.97},
                    ]
                },
                confidence=0.94,
                processing_time=round(random.uniform(1, 3), 1)
            )
        elif tool_name == "scene_detection":
            duration = video_context.get("duration_seconds", 300)
            scenes = []
            current = 0
            while current < duration:
                scene_len = random.randint(10, 60)
                scenes.append({"start": current, "end": min(current + scene_len, duration)})
                current += scene_len
            return SimulatedToolResult(
                tool_name="scene_detection",
                success=True,
                output={"scenes": scenes},
                confidence=0.92,
                processing_time=round(random.uniform(10, 30), 1)
            )
        else:
            # Generic simulation
            return SimulatedToolResult(
                tool_name=tool_name,
                success=True,
                output={"result": "processed"},
                confidence=0.9,
                processing_time=round(random.uniform(2, 10), 1)
            )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CLAMS training tasks")
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of tasks to generate")
    parser.add_argument("--output", type=str, default="tasks.jsonl", help="Output file path")
    parser.add_argument("--tools", type=str, default="tools.json", help="Tools definition file")
    parser.add_argument("--balanced", action="store_true", help="Balance tasks across categories")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    random.seed(args.seed)

    generator = CLAMSTaskGenerator(args.tools)
    tasks = generator.generate_batch(args.num_tasks, balanced=args.balanced)
    generator.save_tasks(tasks, args.output)

    # Print summary
    categories = {}
    for task in tasks:
        categories[task.category] = categories.get(task.category, 0) + 1

    print("\nTask Generation Summary:")
    print(f"Total tasks: {len(tasks)}")
    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
