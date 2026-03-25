"""custom_hf_source — NeMo Curator data-source plugin for HuggingFace datasets.

Importing this package is fast — nemo_curator is only loaded lazily when
``HuggingFaceDownloadExtractStage`` is instantiated.
"""

from .base import (
    BaseDownloader,
    BaseExtractor,
    BaseIterator,
    BaseURLGenerator,
    build_hf_pseudo_url,
    parse_hf_pseudo_url,
)
from .download import HuggingFaceDownloader
from .extract import HuggingFaceExtractor
from .iterator import HuggingFaceIterator
from .stage import HuggingFaceDownloadExtractStage
from .url_generation import HuggingFaceURLGenerator

__all__ = [
    "BaseURLGenerator",
    "BaseDownloader",
    "BaseIterator",
    "BaseExtractor",
    "HuggingFaceURLGenerator",
    "HuggingFaceDownloader",
    "HuggingFaceIterator",
    "HuggingFaceExtractor",
    "HuggingFaceDownloadExtractStage",
    "build_hf_pseudo_url",
    "parse_hf_pseudo_url",
]
