"""Composite stage that wires the four HF pipeline components together.

This is the single entry-point you pass to ``nemo_curator.pipeline.Pipeline``.
"""

from __future__ import annotations

from pathlib import Path

from nemo_curator.stages.text.download import DocumentDownloadExtractStage

from .download import HuggingFaceDownloader
from .extract import HuggingFaceExtractor
from .iterator import HuggingFaceIterator
from .url_generation import HuggingFaceURLGenerator


class HuggingFaceDownloadExtractStage(DocumentDownloadExtractStage):
    """End-to-end download → iterate → extract stage for HuggingFace datasets.

    Parameters
    ----------
    dataset_name:
        HuggingFace dataset identifier, e.g. ``"th1nhng0/vietnamese-legal-documents"``.
    split:
        Restrict to a single split (``None`` = enumerate all).
    config_name:
        Restrict to a single config/subset (``None`` = enumerate all).
    cache_dir:
        HuggingFace cache directory.
    output_dir:
        Directory where raw JSONL files are stored.
    text_field:
        Explicit text column name; ``None`` triggers auto-detection.
    text_strategy:
        Named text strategy from ``Helper.TEXT_STRATEGIES``.  ``"auto"``
        uses the legacy text-field auto-detection.
    url_limit / record_limit:
        Optional caps forwarded to ``DocumentDownloadExtractStage``.
    add_filename_column:
        Attach the source filename as an extra column in the output batch.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str | None = None,
        config_name: str | None = None,
        cache_dir: str | None = None,
        output_dir: str = "./downloads",
        text_field: str | None = None,
        text_strategy: str = "auto",
        url_limit: int | None = None,
        record_limit: int | None = None,
        add_filename_column: bool | str = True,
    ) -> None:
        self._dataset_name = dataset_name

        url_generator = HuggingFaceURLGenerator(
            dataset_name=dataset_name,
            config_name=config_name,
            split=split,
        )
        downloader = HuggingFaceDownloader(
            download_dir=output_dir,
            cache_dir=cache_dir,
        )
        iterator = HuggingFaceIterator()
        extractor = HuggingFaceExtractor(
            dataset_name=dataset_name,
            text_field=text_field,
            text_strategy=text_strategy,
        )

        super().__init__(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=add_filename_column,
        )
        self.name = f"hf_download_{dataset_name.replace('/', '_')}"

    # TODO: implement embeddings stage
    # embeddings_dir = Path(output_dir) / "embeddings" / model_id
    # e.g. datasets/th1nhng0/vietnamese-legal-documents/default/train/
    #          preprocessed/embeddings/nvidia/llama-embed-nemotron-8b/

    # TODO: implement embreduced stage
    # embreduced_dir = Path(output_dir) / "embreduced" / model_id
    # e.g. datasets/th1nhng0/vietnamese-legal-documents/default/train/
    #          preprocessed/embreduced/nvidia/llama-embed-nemotron-8b/

    def get_description(self) -> str:
        return f"HuggingFace download+extract pipeline for {self._dataset_name}"
