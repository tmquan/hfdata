"""Concrete HuggingFace URL generator.

Uses HF Hub API to auto-enumerate every config × split combination.
For dataset-specific enumeration logic, subclass
:class:`~custom_hf_source.base.BaseURLGenerator` directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseURLGenerator


@dataclass
class HuggingFaceURLGenerator(BaseURLGenerator):
    """Ready-to-use URL generator for any public HuggingFace dataset."""
