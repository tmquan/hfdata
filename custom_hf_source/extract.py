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
    """Ready-to-use extractor with text-field auto-detection.

    Pass *text_strategy* to use a named strategy function from
    ``Helper.TEXT_STRATEGIES`` instead of auto-detecting a text field.
    """

    def __init__(
        self,
        dataset_name: str = "",
        text_field: str | None = None,
        text_strategy: str = "auto",
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            text_field=text_field,
            text_strategy=text_strategy,
        )
