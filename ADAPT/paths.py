"""Project-root paths. All caches and datasets are resolved from Adapt-code/."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
RESULT_DIR = ROOT / "results"

FEATURE_TYPES = ("word", "sent", "doc", "ext")
# Candidate queues keep the top T pairs. Retrieval asks for a larger pool so the
# certified stopper can tell the queue apart from the retrieved tail.
RETRIEVAL_OVER_RATIO = 5.0
