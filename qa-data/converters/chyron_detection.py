"""
Converter for chyron detection annotations to QA format.

Handles both:
- newshour-chyron: Time intervals + text
- understanding-chyrons: Detailed text understanding

Source formats: CSV with columns varying by project
"""

from pathlib import Path
from typing import Generator

from base_converter import BaseConverter, QAEntry, select_question_template


class ChyronDetectionConverter(BaseConverter):
    """Convert chyron detection annotations to QA format."""

    PROJECT_NAME = "newshour-chyron"
    CATEGORY = "text_extraction"

    def convert_file(self, filepath: Path) -> Generator[QAEntry, None, None]:
        """
        Convert a single chyron CSV file.

        Generates:
        1. Text extraction questions (what text at timestamp)
        2. Time range questions (when does chyron appear)
        3. Person identification questions (who is shown in chyron)
        """
        guid = self.extract_guid(filepath.name)
        rows = self.read_csv(filepath)

        if not rows:
            return

        all_chyrons = []

        for i, row in enumerate(rows):
            start = row.get("start", "")
            end = row.get("end", "")
            text = row.get("text", "")

            if not start or not text:
                continue

            all_chyrons.append(row)

            # Q1: What text appears at this timestamp?
            yield QAEntry(
                id=self.generate_id(),
                category=self.CATEGORY,
                source_project=self.PROJECT_NAME,
                source_guid=guid,
                question=select_question_template(
                    "text_at_time",
                    index=i,
                    timestamp=start,
                    scene_type="chyron"
                ),
                answer={
                    "text": text,
                    "lines": text.split("\n") if "\n" in text else [text]
                },
                context={
                    "video_guid": guid,
                    "time_range": {"start": start, "end": end}
                },
                metadata=self.create_metadata(filepath),
                difficulty="medium",
                tags=["chyron", "text-extraction"]
            )

            # Q2: Text in time range
            if end:
                yield QAEntry(
                    id=self.generate_id(),
                    category=self.CATEGORY,
                    source_project=self.PROJECT_NAME,
                    source_guid=guid,
                    question=select_question_template(
                        "text_in_range",
                        index=0,
                        start=start,
                        end=end,
                        scene_type="chyron"
                    ),
                    answer={
                        "text": text,
                        "duration_seconds": self._time_diff_seconds(start, end)
                    },
                    context={
                        "video_guid": guid,
                        "time_range": {"start": start, "end": end}
                    },
                    metadata=self.create_metadata(filepath),
                    difficulty="medium",
                    tags=["chyron", "time-range"]
                )

            # Q3: If text contains a name pattern (NAME\nTitle), extract person
            lines = text.split("\n")
            if len(lines) >= 2 and lines[0].isupper():
                # Likely a name followed by title
                name = lines[0]
                title = lines[1] if len(lines) > 1 else ""

                yield QAEntry(
                    id=self.generate_id(),
                    category=self.CATEGORY,
                    source_project=self.PROJECT_NAME,
                    source_guid=guid,
                    question=f"Who is identified in the chyron at {start}?",
                    answer={
                        "name": name,
                        "title": title,
                        "full_text": text
                    },
                    context={
                        "video_guid": guid,
                        "timestamp": start
                    },
                    metadata=self.create_metadata(filepath),
                    difficulty="easy",
                    tags=["chyron", "person-identification"]
                )

        # Aggregation question: List all chyrons
        if all_chyrons:
            yield QAEntry(
                id=self.generate_id(),
                category=self.CATEGORY,
                source_project=self.PROJECT_NAME,
                source_guid=guid,
                question="List all chyrons that appear in this video.",
                answer={
                    "count": len(all_chyrons),
                    "chyrons": [
                        {
                            "start": c.get("start"),
                            "end": c.get("end"),
                            "text": c.get("text")
                        }
                        for c in all_chyrons
                    ]
                },
                context={"video_guid": guid},
                metadata=self.create_metadata(filepath),
                difficulty="medium",
                tags=["chyron", "aggregation"]
            )

    def _time_diff_seconds(self, start: str, end: str) -> float:
        """Calculate time difference in seconds between two timestamps."""
        try:
            def parse_time(t: str) -> float:
                parts = t.split(":")
                if len(parts) == 3:
                    h, m, s = parts
                    return int(h) * 3600 + int(m) * 60 + float(s)
                return 0.0

            return parse_time(end) - parse_time(start)
        except (ValueError, IndexError):
            return 0.0


class UnderstandingChyronsConverter(BaseConverter):
    """Convert understanding-chyrons annotations to QA format."""

    PROJECT_NAME = "understanding-chyrons"
    CATEGORY = "text_extraction"

    def convert_file(self, filepath: Path) -> Generator[QAEntry, None, None]:
        """
        Convert understanding-chyrons files with detailed text analysis.
        """
        guid = self.extract_guid(filepath.name)
        rows = self.read_csv(filepath)

        if not rows:
            return

        for i, row in enumerate(rows):
            # Adapt based on actual column names in understanding-chyrons
            timestamp = row.get("at", row.get("timestamp", row.get("start", "")))
            text = row.get("text", row.get("text-transcript", ""))
            note = row.get("note", "")

            if not timestamp or not text:
                continue

            # Basic transcription question
            yield QAEntry(
                id=self.generate_id(),
                category=self.CATEGORY,
                source_project=self.PROJECT_NAME,
                source_guid=guid,
                question=f"Transcribe the chyron text at {timestamp}.",
                answer={
                    "text": text,
                    "note": note if note else None
                },
                context={
                    "video_guid": guid,
                    "timestamp": timestamp
                },
                metadata=self.create_metadata(filepath),
                difficulty="medium",
                tags=["chyron", "transcription"]
            )


def main():
    """Run the chyron converters."""
    import sys

    annotations_root = Path("/Users/kelleylynch/clams/aapb-annotations")
    output_dir = Path("/Users/kelleylynch/clams/clams-agent-prototype/qa-data/raw")

    if len(sys.argv) > 1:
        annotations_root = Path(sys.argv[1])

    # Run both converters
    for ConverterClass in [ChyronDetectionConverter, UnderstandingChyronsConverter]:
        converter = ConverterClass(annotations_root, output_dir)

        if converter.gold_dir.exists():
            print(f"\nConverting {converter.PROJECT_NAME}...")
            print(f"  From: {converter.gold_dir}")

            entries = converter.run(f"{converter.PROJECT_NAME}.jsonl")
            print(f"  Generated {len(entries)} QA entries")

            if entries:
                print(f"  Sample: Q: {entries[0].question}")
                print(f"          A: {entries[0].answer}")
        else:
            print(f"Skipping {converter.PROJECT_NAME} - gold dir not found")


if __name__ == "__main__":
    main()
