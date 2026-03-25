"""Concrete HuggingFace extractor.

Normalises arbitrary HuggingFace record dicts into the canonical schema
``{text, id, source, metadata_json}`` with automatic text-field detection.
For dataset-specific extraction (e.g. combining multiple columns into
*text*), subclass :class:`~custom_hf_source.base.BaseExtractor` and
override ``_resolve_text_field`` or ``_build_output``.
"""

from __future__ import annotations

from .base import BaseExtractor


class HuggingFaceExtractor(BaseExtractor):
    """Ready-to-use extractor with text-field auto-detection."""
