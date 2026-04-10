"""
CLAMS annotation to QA format converters.

This package contains converter classes for transforming AAPB annotations
from the aapb-annotations repository into question-answering format for
training and evaluating the CLAMS agent.

Usage:
    from converters.scene_recognition import SceneRecognitionConverter
    from converters.chyron_detection import ChyronDetectionConverter

    converter = SceneRecognitionConverter(
        annotations_root=Path("/path/to/aapb-annotations"),
        output_dir=Path("/path/to/qa-data/raw")
    )
    entries = converter.run()
"""

from pathlib import Path

# Converter registry for easy discovery
CONVERTERS = {
    "scene-recognition": "scene_recognition.SceneRecognitionConverter",
    "newshour-chyron": "chyron_detection.ChyronDetectionConverter",
    "understanding-chyrons": "chyron_detection.UnderstandingChyronsConverter",
    # TODO: Add more converters as implemented
    # "january-slates": "slate_detection.SlateDetectionConverter",
    # "newshour-namedentity": "named_entity.NamedEntityConverter",
    # "newshour-namedentity-wikipedialink": "named_entity.EntityLinkingConverter",
    # "role-filler-binding": "role_filler.RoleFillerConverter",
}


def get_converter(project_name: str):
    """
    Get the converter class for a given project.

    Args:
        project_name: Name of the annotation project (e.g., 'scene-recognition')

    Returns:
        Converter class (not instantiated)

    Raises:
        ValueError: If no converter exists for the project
    """
    if project_name not in CONVERTERS:
        raise ValueError(
            f"No converter for project '{project_name}'. "
            f"Available: {list(CONVERTERS.keys())}"
        )

    module_name, class_name = CONVERTERS[project_name].rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def run_all_converters(
    annotations_root: Path,
    output_dir: Path,
    projects: list[str] | None = None
):
    """
    Run converters for multiple projects.

    Args:
        annotations_root: Path to aapb-annotations repository
        output_dir: Path to write QA output files
        projects: List of project names to convert (default: all)

    Returns:
        Dict mapping project names to lists of QA entries
    """
    if projects is None:
        projects = list(CONVERTERS.keys())

    results = {}
    for project in projects:
        try:
            converter_class = get_converter(project)
            converter = converter_class(annotations_root, output_dir)
            if converter.gold_dir.exists():
                results[project] = converter.run()
            else:
                print(f"Skipping {project} - gold directory not found")
        except Exception as e:
            print(f"Error converting {project}: {e}")
            results[project] = []

    return results
