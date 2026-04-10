"""
Base converter class for transforming AAPB annotations to QA format.
"""

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Any, Generator, Optional


@dataclass
class QAEntry:
    """A single question-answer pair."""
    id: str
    category: str
    question: str
    answer: dict[str, Any]
    source_project: Optional[str] = None
    source_guid: Optional[str] = None
    context: Optional[dict[str, Any]] = None
    conversation: Optional[list[dict[str, str]]] = None
    metadata: Optional[dict[str, Any]] = None
    difficulty: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class BaseConverter(ABC):
    """Abstract base class for annotation converters."""

    # Override in subclasses
    PROJECT_NAME: str = ""
    CATEGORY: str = ""

    def __init__(self, annotations_root: Path, output_dir: Path):
        """
        Initialize the converter.

        Args:
            annotations_root: Path to aapb-annotations repository
            output_dir: Path to write QA output files
        """
        self.annotations_root = Path(annotations_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._id_counter = 0

    @property
    def gold_dir(self) -> Path:
        """Path to gold annotation files for this project."""
        return self.annotations_root / self.PROJECT_NAME / "golds"

    def generate_id(self) -> str:
        """Generate a unique QA entry ID."""
        self._id_counter += 1
        prefix = self.CATEGORY.replace("_", "-")
        return f"qa-{prefix}-{self._id_counter:06d}"

    def extract_guid(self, filename: str) -> str:
        """Extract GUID from annotation filename."""
        # Remove extension and return
        return Path(filename).stem

    @abstractmethod
    def convert_file(self, filepath: Path) -> Generator[QAEntry, None, None]:
        """
        Convert a single annotation file to QA entries.

        Args:
            filepath: Path to the annotation file

        Yields:
            QAEntry objects
        """
        pass

    def convert_all(self) -> Generator[QAEntry, None, None]:
        """
        Convert all gold files in the project.

        Yields:
            QAEntry objects from all files
        """
        if not self.gold_dir.exists():
            raise FileNotFoundError(f"Gold directory not found: {self.gold_dir}")

        # Find all CSV/TSV files
        patterns = ["*.csv", "*.tsv"]
        files = []
        for pattern in patterns:
            files.extend(self.gold_dir.glob(pattern))

        for filepath in sorted(files):
            yield from self.convert_file(filepath)

    def read_csv(self, filepath: Path) -> list[dict]:
        """Read a CSV file and return list of row dicts."""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def read_tsv(self, filepath: Path) -> list[dict]:
        """Read a TSV file and return list of row dicts."""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            return list(reader)

    def create_metadata(self, filepath: Path, **extra) -> dict:
        """Create standard metadata dict."""
        return {
            "created_from": str(filepath.relative_to(self.annotations_root)),
            "conversion_date": date.today().isoformat(),
            **extra
        }

    def write_jsonl(self, entries: list[QAEntry], filename: str):
        """Write QA entries to a JSONL file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(entry.to_json() + '\n')
        print(f"Wrote {len(entries)} entries to {output_path}")

    def run(self, output_filename: Optional[str] = None) -> list[QAEntry]:
        """
        Run the full conversion and write output.

        Args:
            output_filename: Name of output file (default: {category}.jsonl)

        Returns:
            List of all generated QA entries
        """
        if output_filename is None:
            output_filename = f"{self.CATEGORY}.jsonl"

        entries = list(self.convert_all())
        self.write_jsonl(entries, output_filename)

        return entries


# Question templates for common question types
QUESTION_TEMPLATES = {
    "scene_at_time": [
        "What type of scene appears at timestamp {timestamp}?",
        "What is happening at {timestamp} in this video?",
        "Identify the scene type at {timestamp}.",
    ],
    "scene_exists": [
        "Does this video contain any {scene_type} scenes?",
        "Are there {scene_type} frames in this video?",
        "Can you find {scene_type} in this video?",
    ],
    "scene_count": [
        "How many {scene_type} scenes are in this video?",
        "Count the number of {scene_type} frames.",
    ],
    "text_at_time": [
        "What text appears at {timestamp}?",
        "Transcribe the text visible at {timestamp}.",
        "What does the {scene_type} say at {timestamp}?",
    ],
    "text_in_range": [
        "What text appears between {start} and {end}?",
        "Transcribe the {scene_type} from {start} to {end}.",
    ],
    "entities_in_transcript": [
        "What named entities appear in this transcript?",
        "Identify the people, organizations, and locations mentioned.",
        "Extract named entities from this text.",
    ],
    "entity_type": [
        "What {entity_type}s are mentioned in this transcript?",
        "List all {entity_type} entities.",
        "Find the {entity_type}s in this text.",
    ],
    "entity_link": [
        "What Wikipedia article describes '{entity_text}'?",
        "Link '{entity_text}' to Wikipedia.",
        "Find the Wikidata ID for '{entity_text}'.",
    ],
    "role_single": [
        "Who is credited as {role} in this program?",
        "What is the {role}'s name?",
        "Find the {role} in the credits.",
    ],
    "role_all": [
        "Extract all credits from this sequence.",
        "List all role-filler pairs from the credits.",
        "What production credits are shown?",
    ],
    "tool_for_task": [
        "What tool should I use to {task}?",
        "Which CLAMS app can {task}?",
        "How do I {task} with CLAMS?",
    ],
    "pipeline_for_goal": [
        "Design a pipeline to {goal}.",
        "What tools do I need to {goal}?",
        "How do I go from {input_type} to {output_type}?",
    ],
}


def select_question_template(template_key: str, index: int = 0, **kwargs) -> str:
    """
    Select a question template and fill in placeholders.

    Args:
        template_key: Key in QUESTION_TEMPLATES
        index: Which template variant to use (for variety)
        **kwargs: Values to fill in template placeholders

    Returns:
        Formatted question string
    """
    templates = QUESTION_TEMPLATES.get(template_key, [])
    if not templates:
        raise ValueError(f"Unknown template key: {template_key}")

    template = templates[index % len(templates)]
    return template.format(**kwargs)
