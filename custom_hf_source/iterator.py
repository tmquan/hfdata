"""Concrete HuggingFace iterator.

Reads the JSONL files produced by :class:`HuggingFaceDownloader` and
yields one ``dict`` per record.
For datasets stored in a non-JSONL format, subclass
:class:`~custom_hf_source.base.BaseIterator` and override
``_parse_line``.
"""

from __future__ import annotations

from .base import BaseIterator


class HuggingFaceIterator(BaseIterator):
    """Ready-to-use iterator over JSONL files from HuggingFaceDownloader."""
