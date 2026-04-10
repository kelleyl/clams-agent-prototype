"""
Converter for scene-recognition annotations to QA format.

Source: aapb-annotations/scene-recognition/golds/
Format: CSV with columns: at, scene-label, scene-subtype-label, transitional
"""

from collections import defaultdict
from pathlib import Path
from typing import Generator

from base_converter import BaseConverter, QAEntry, select_question_template


# Scene label mappings from the annotation guidelines
SCENE_LABELS = {
    "S": "slate",
    "C": "chyron",
    "R": "credits",
    "P": "person",
    "I": "image",
    "O": "other",
    "N": "not-applicable",
}

SCENE_SUBTYPES = {
    "D": "digital",
    "H": "handwritten",
    "G": "generic",
    # Add more as needed from annotation guidelines
}


class SceneRecognitionConverter(BaseConverter):
    """Convert scene-recognition annotations to QA format."""

    PROJECT_NAME = "scene-recognition"
    CATEGORY = "scene_classification"

    def expand_label(self, short_label: str) -> str:
        """Expand abbreviated labels to full names."""
        return SCENE_LABELS.get(short_label, short_label)

    def expand_subtype(self, short_subtype: str) -> str:
        """Expand abbreviated subtypes to full names."""
        return SCENE_SUBTYPES.get(short_subtype, short_subtype)

    def convert_file(self, filepath: Path) -> Generator[QAEntry, None, None]:
        """
        Convert a single scene-recognition CSV file.

        Generates multiple question types:
        1. Point-in-time classification questions
        2. Existence questions (does scene type exist)
        3. Count questions (how many of each type)
        """
        guid = self.extract_guid(filepath.name)
        rows = self.read_csv(filepath)

        if not rows:
            return

        # Group by scene type for aggregation questions
        by_label: dict[str, list[dict]] = defaultdict(list)

        # Generate point-in-time questions for each row
        for i, row in enumerate(rows):
            timestamp = row.get("at", "")
            scene_label = self.expand_label(row.get("scene-label", ""))
            subtype = self.expand_subtype(row.get("scene-subtype-label", ""))
            transitional = row.get("transitional", "").lower() == "true"

            if not timestamp or not scene_label:
                continue

            by_label[scene_label].append(row)

            # Q1: What scene type at this timestamp?
            # Only generate for a subset to avoid too many similar questions
            if i % 5 == 0:  # Sample every 5th frame
                yield QAEntry(
                    id=self.generate_id(),
                    category=self.CATEGORY,
                    source_project=self.PROJECT_NAME,
                    source_guid=guid,
                    question=select_question_template(
                        "scene_at_time",
                        index=i,
                        timestamp=timestamp
                    ),
                    answer={
                        "scene_label": scene_label,
                        "scene_subtype": subtype if subtype else None,
                        "transitional": transitional,
                        "confidence": "gold"
                    },
                    context={
                        "video_guid": guid,
                        "timestamp": timestamp
                    },
                    metadata=self.create_metadata(filepath),
                    difficulty="easy",
                    tags=["frame-level", scene_label]
                )

        # Generate existence questions for each scene type found
        for scene_label, frames in by_label.items():
            timestamps = [f.get("at") for f in frames if f.get("at")]

            yield QAEntry(
                id=self.generate_id(),
                category=self.CATEGORY,
                source_project=self.PROJECT_NAME,
                source_guid=guid,
                question=select_question_template(
                    "scene_exists",
                    index=0,
                    scene_type=scene_label
                ),
                answer={
                    "exists": True,
                    "count": len(frames),
                    "sample_timestamps": timestamps[:5]  # First 5 as examples
                },
                context={"video_guid": guid},
                metadata=self.create_metadata(filepath),
                difficulty="easy",
                tags=["existence", scene_label]
            )

            # Count question
            yield QAEntry(
                id=self.generate_id(),
                category=self.CATEGORY,
                source_project=self.PROJECT_NAME,
                source_guid=guid,
                question=select_question_template(
                    "scene_count",
                    index=0,
                    scene_type=scene_label
                ),
                answer={
                    "count": len(frames),
                    "scene_label": scene_label
                },
                context={"video_guid": guid},
                metadata=self.create_metadata(filepath),
                difficulty="easy",
                tags=["count", scene_label]
            )

        # Generate negative existence questions for scene types NOT found
        all_labels = set(SCENE_LABELS.values())
        found_labels = set(by_label.keys())
        missing_labels = all_labels - found_labels - {"not-applicable", "other"}

        for missing_label in missing_labels:
            yield QAEntry(
                id=self.generate_id(),
                category=self.CATEGORY,
                source_project=self.PROJECT_NAME,
                source_guid=guid,
                question=select_question_template(
                    "scene_exists",
                    index=1,
                    scene_type=missing_label
                ),
                answer={
                    "exists": False,
                    "count": 0
                },
                context={"video_guid": guid},
                metadata=self.create_metadata(filepath),
                difficulty="easy",
                tags=["existence", "negative", missing_label]
            )


def main():
    """Run the scene recognition converter."""
    import sys

    # Default paths - adjust as needed
    annotations_root = Path("/Users/kelleylynch/clams/aapb-annotations")
    output_dir = Path("/Users/kelleylynch/clams/clams-agent-prototype/qa-data/raw")

    if len(sys.argv) > 1:
        annotations_root = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])

    converter = SceneRecognitionConverter(annotations_root, output_dir)

    print(f"Converting from: {converter.gold_dir}")
    print(f"Output to: {output_dir}")

    entries = converter.run()
    print(f"Generated {len(entries)} QA entries")

    # Print sample
    print("\nSample entries:")
    for entry in entries[:3]:
        print(f"  Q: {entry.question}")
        print(f"  A: {entry.answer}")
        print()


if __name__ == "__main__":
    main()
