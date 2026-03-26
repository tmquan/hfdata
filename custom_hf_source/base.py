"""Abstract base classes for HuggingFace ↔ NeMo Curator integration.

Every NeMo Curator abstract method is implemented here with sensible HF-specific
defaults.  Subclasses only need to override the ``_hook_*`` / helper methods to
customise behaviour for a particular dataset.

The four ABCs mirror the NeMo Curator interfaces
(``URLGenerator``, ``DocumentDownloader``, ``DocumentIterator``,
``DocumentExtractor``) but are defined **locally** so that
``import custom_hf_source`` never triggers the heavy ``nemo_curator``
init chain (transformers → importlib.metadata scan).  NeMo Curator is
only imported lazily in ``stage.py`` when the Pipeline is built.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

# ── Pseudo-URL protocol ─────────────────────────────────────────────────────

_HF_SCHEME = "hf"
_SEP = "::"


def build_hf_pseudo_url(dataset_name: str, config: str, split: str) -> str:
    """Encode a HuggingFace triplet into a pseudo-URL."""
    return _SEP.join([_HF_SCHEME, dataset_name, config, split])


def parse_hf_pseudo_url(url: str) -> tuple[str, str, str]:
    """Decode a pseudo-URL back into *(dataset_name, config, split)*."""
    parts = url.split(_SEP)
    if len(parts) != 4 or parts[0] != _HF_SCHEME:
        raise ValueError(f"Invalid HuggingFace pseudo-URL: {url!r}")
    return parts[1], parts[2], parts[3]


# ── Shared constants ─────────────────────────────────────────────────────────

TEXT_FIELD_CANDIDATES: list[str] = [
    "text",
    "content",
    "document",
    "passage",
    "body",
    "sentence",
    "paragraph",
    "raw_text",
    "article",
]


# ═════════════════════════════════════════════════════════════════════════════
# Lightweight ABCs — same signatures as the NeMo Curator base classes so the
# objects slot straight into DocumentDownloadExtractStage at pipeline-build
# time (duck-typing / structural subtyping).
# ═════════════════════════════════════════════════════════════════════════════


class _URLGenerator(ABC):
    @abstractmethod
    def generate_urls(self) -> list[str]: ...


class _DocumentDownloader(ABC):
    """Mirrors ``nemo_curator…DocumentDownloader`` including the concrete
    ``download()`` method (temp-file + atomic rename)."""

    def __init__(self, download_dir: str, verbose: bool = False) -> None:
        self._download_dir = download_dir
        self._verbose = verbose
        os.makedirs(download_dir, exist_ok=True)

    @abstractmethod
    def _get_output_filename(self, url: str) -> str: ...

    @abstractmethod
    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]: ...

    def download(self, url: str) -> str | None:
        output_name = self._get_output_filename(url)
        output_file = os.path.join(self._download_dir, output_name)
        temp_file = output_file + ".tmp"

        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            if self._verbose:
                logger.info(f"File exists, skipping: {output_file}")
            return output_file

        success, error_message = self._download_to_path(url, temp_file)
        if success:
            os.rename(temp_file, output_file)
            if self._verbose:
                sz = os.path.getsize(output_file)
                logger.info(f"Downloaded {output_file} ({sz:,} bytes)")
            return output_file

        logger.error(f"Download failed for {output_file}: {error_message}")
        return None

    def num_workers_per_node(self) -> int | None:
        return None


class _DocumentIterator(ABC):
    @abstractmethod
    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]: ...

    @abstractmethod
    def output_columns(self) -> list[str]: ...


class _DocumentExtractor(ABC):
    @abstractmethod
    def extract(self, record: dict[str, str]) -> dict[str, Any] | None: ...

    @abstractmethod
    def input_columns(self) -> list[str]: ...

    @abstractmethod
    def output_columns(self) -> list[str]: ...


# ═════════════════════════════════════════════════════════════════════════════
# 1. URL Generation
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class BaseURLGenerator(_URLGenerator):
    """Auto-enumerates HuggingFace configs / splits and emits pseudo-URLs.

    Override ``_enumerate_configs`` or ``_enumerate_splits`` if a dataset
    needs special enumeration logic (e.g. language-based configs).
    """

    dataset_name: str
    config_name: str | None = None
    split: str | None = None

    def generate_urls(self) -> list[str]:
        configs = (
            [self.config_name] if self.config_name else self._enumerate_configs()
        )
        urls: list[str] = []
        for cfg in configs:
            splits = [self.split] if self.split else self._enumerate_splits(cfg)
            for sp in splits:
                url = build_hf_pseudo_url(self.dataset_name, cfg, sp)
                logger.info(
                    f"Enqueued {self.dataset_name}  config={cfg!r}  split={sp!r}"
                )
                urls.append(url)
        return urls

    # -- hooks ---------------------------------------------------------------

    def _enumerate_configs(self) -> list[str]:
        from datasets import get_dataset_config_names

        try:
            configs = get_dataset_config_names(
                self.dataset_name, trust_remote_code=True
            )
        except Exception as exc:
            logger.warning(
                f"Could not fetch configs for {self.dataset_name}: {exc}"
            )
            configs = []
        return configs or ["default"]

    def _enumerate_splits(self, config: str) -> list[str]:
        from datasets import get_dataset_split_names

        try:
            return get_dataset_split_names(
                self.dataset_name, config, trust_remote_code=True
            )
        except Exception:
            pass
        try:
            return get_dataset_split_names(
                self.dataset_name, trust_remote_code=True
            )
        except Exception as exc:
            logger.warning(
                f"Could not fetch splits for "
                f"{self.dataset_name}/{config}: {exc}. Falling back to 'train'."
            )
            return ["train"]


# ═════════════════════════════════════════════════════════════════════════════
# 2. Downloading
# ═════════════════════════════════════════════════════════════════════════════


class BaseDownloader(_DocumentDownloader):
    """Downloads a HuggingFace split via ``datasets.load_dataset`` and persists
    the result as JSONL in ``download_dir``.

    Override ``_load_hf_dataset`` to change how the dataset is fetched (e.g.
    streaming mode) or ``_serialize_record`` to change the on-disk format.
    """

    def __init__(
        self,
        download_dir: str,
        cache_dir: str | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(download_dir=download_dir, verbose=verbose)
        self._cache_dir = cache_dir

    def _get_output_filename(self, url: str) -> str:
        dataset_name, config, split = parse_hf_pseudo_url(url)
        safe_name = dataset_name.replace("/", "__")
        return f"{safe_name}__{config}__{split}.jsonl"

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        try:
            dataset_name, config, split = parse_hf_pseudo_url(url)
            ds = self._load_hf_dataset(dataset_name, config, split)

            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                for record in ds:
                    fh.write(self._serialize_record(record) + "\n")

            logger.info(f"Saved {len(ds)} records → {path}")
            return True, None
        except Exception as exc:
            return False, str(exc)

    def num_workers_per_node(self) -> int | None:
        return 2

    # -- hooks ---------------------------------------------------------------

    def _load_hf_dataset(
        self, dataset_name: str, config: str, split: str
    ) -> Any:
        from datasets import load_dataset

        config_arg = None if config == "default" else config
        try:
            return load_dataset(
                dataset_name,
                config_arg,
                split=split,
                cache_dir=self._cache_dir,
                trust_remote_code=True,
            )
        except Exception:
            return load_dataset(
                dataset_name,
                config,
                split=split,
                cache_dir=self._cache_dir,
                trust_remote_code=True,
            )

    def _serialize_record(self, record: dict[str, Any]) -> str:
        return json.dumps(record, ensure_ascii=False, default=str)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Iteration
# ═════════════════════════════════════════════════════════════════════════════


class BaseIterator(_DocumentIterator):
    """Reads the JSONL files produced by :class:`BaseDownloader`.

    Override ``_parse_line`` to handle non-JSON formats, or
    ``_output_columns`` to advertise a different schema.
    """

    def __init__(self, log_frequency: int = 10_000) -> None:
        self._log_frequency = log_frequency

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        count = 0
        with open(file_path, encoding="utf-8") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                record = self._parse_line(stripped, file_path)
                if record is not None:
                    count += 1
                    if count % self._log_frequency == 0:
                        logger.debug(f"Read {count:,} records from {file_path}")
                    yield record
        logger.info(f"Iterated {count:,} records from {file_path}")

    def output_columns(self) -> list[str]:
        return self._output_columns()

    # -- hooks ---------------------------------------------------------------

    def _parse_line(self, line: str, file_path: str) -> dict[str, Any] | None:
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning(f"Skipping malformed line in {file_path}: {exc}")
            return None

    def _output_columns(self) -> list[str]:
        return ["text", "id", "source", "metadata_json"]


# ═════════════════════════════════════════════════════════════════════════════
# 4. Extraction / normalisation
# ═════════════════════════════════════════════════════════════════════════════


class BaseExtractor(_DocumentExtractor):
    """Normalises arbitrary HuggingFace records into a fixed schema::

        {text, id, source, metadata_json}

    Override ``_resolve_text_field``, ``_build_id`` or ``_build_output``
    for dataset-specific normalisation.

    When *text_strategy* is not ``"auto"``, the named strategy function
    from ``Helper.TEXT_STRATEGIES`` is used to construct ``text`` from
    the raw record, bypassing ``_resolve_text_field``.
    """

    def __init__(
        self,
        dataset_name: str = "",
        text_field: str | None = None,
        text_strategy: str = "auto",
    ) -> None:
        self._dataset_name = dataset_name
        self._text_field = text_field
        self._text_strategy = text_strategy
        self._resolved_text_field: str | None = None
        self._strategy_fn = None
        if text_strategy != "auto":
            from Helper import TEXT_STRATEGIES
            self._strategy_fn = TEXT_STRATEGIES.get(text_strategy)
            if self._strategy_fn is None:
                logger.warning(
                    f"Unknown text_strategy {text_strategy!r}, falling back to auto"
                )

    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        if self._strategy_fn is not None:
            text = self._strategy_fn(record)
            if not text or not text.strip():
                return None
            text = text.strip()
            return self._build_output(record, text, None)

        if self._resolved_text_field is None:
            self._resolved_text_field = self._resolve_text_field(record)
            if self._resolved_text_field:
                logger.info(
                    f"Auto-detected text field: {self._resolved_text_field!r}"
                )

        tf = self._resolved_text_field
        if tf is None or tf not in record:
            return None

        text = str(record[tf]).strip()
        if not text:
            return None

        return self._build_output(record, text, tf)

    def input_columns(self) -> list[str]:
        return ["text", "id"]

    def output_columns(self) -> list[str]:
        return ["text", "id", "source", "metadata_json"]

    # -- hooks ---------------------------------------------------------------

    def _resolve_text_field(self, record: dict[str, Any]) -> str | None:
        if self._text_field and self._text_field in record:
            return self._text_field
        for candidate in TEXT_FIELD_CANDIDATES:
            if candidate in record:
                return candidate
        for key, value in record.items():
            if isinstance(value, str) and len(value) > 20:
                return key
        return None

    def _build_id(self, record: dict[str, Any], text: str) -> str:
        doc_id = str(record.get("id", ""))
        if not doc_id:
            doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        return doc_id

    def _build_output(
        self,
        record: dict[str, Any],
        text: str,
        text_field: str | None,
    ) -> dict[str, Any]:
        exclude = {"id"}
        if text_field is not None:
            exclude.add(text_field)
        metadata = {
            k: v for k, v in record.items() if k not in exclude
        }
        return {
            "text": text,
            "id": self._build_id(record, text),
            "source": self._dataset_name,
            "metadata_json": json.dumps(
                metadata, ensure_ascii=False, default=str
            ),
        }
