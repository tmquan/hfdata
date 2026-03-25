"""Concrete HuggingFace downloader.

Downloads a dataset split via ``datasets.load_dataset()`` and persists
the records as JSONL inside the designated *raw/* directory.
For custom download logic, subclass
:class:`~custom_hf_source.base.BaseDownloader` directly.
"""

from __future__ import annotations

from .base import BaseDownloader


class HuggingFaceDownloader(BaseDownloader):
    """Ready-to-use downloader for any public HuggingFace dataset."""
