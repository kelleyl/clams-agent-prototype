"""
GPU classification for CLAMS apps.
Maps app names to whether they require GPU for reasonable performance.
"""

# Static GPU classification for CLAMS apps
# Sources: Containerfile base images, README hardware requirements
GPU_CLASSIFICATION = {
    # CPU-only (always available)
    "pyscenedetect-wrapper": False,
    "simple-timepoints-stitcher": False,
    "spacy-wrapper": False,
    "tesseractocr-wrapper": False,
    "text-slicer": False,
    "tfidf-keywordextractor": False,
    "dbpedia-spotlight-wrapper": False,
    "gentle-forced-aligner-wrapper": False,
    "tonedetection": False,
    "barsdetection": False,
    "slatedetection": False,
    "chyron-detection": False,

    # GPU required or strongly preferred
    "whisper-wrapper": True,
    "distil-whisper-wrapper": True,
    "doctr-wrapper": True,
    "easyocr-wrapper": True,
    "east-textdetection": True,
    "parseqocr-wrapper": True,
    "swt-detection": True,
    "llava-captioner": True,
    "inaspeechsegmenter-wrapper": True,
    "fewshotclassifier": True,
    "brandeis-acs-wrapper": True,
    "aapb-pua-kaldi-wrapper": True,
    "smolvlm2-captioner": True,
    "smolvlm2-captioner-pytorch": True,
    "parakeet-wrapper": True,
    "spoken-lid": True,
    "nfa-wrapper": True,
    "heuristic-chyron-understanding": False,
}


def requires_gpu(app_name: str) -> bool:
    """Check if an app requires GPU. Unknown apps default to True (safe)."""
    return GPU_CLASSIFICATION.get(app_name, True)


def get_cpu_only_apps() -> list:
    """Return list of apps that can run on CPU only."""
    return [name for name, gpu in GPU_CLASSIFICATION.items() if not gpu]


def get_gpu_apps() -> list:
    """Return list of apps that require GPU."""
    return [name for name, gpu in GPU_CLASSIFICATION.items() if gpu]
