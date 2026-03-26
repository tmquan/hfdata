"""Shared helpers for Download.py and DownloadExtract.py."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Typed mirror of ``configs/default.yaml``.

    ``Download.py`` uses only the base fields; ``DownloadExtract.py`` uses
    the embedding fields too.  Unknown YAML keys are silently ignored.
    """

    datasets: list[str] = field(default_factory=list)
    output_dir: str = "./datasets"
    cache_dir: str = "~/.cache/huggingface"
    output_format: str = "parquet"
    chunk_size: int = 10000
    model_id: str = "nvidia/llama-embed-nemotron-8b"
    embedding_dim: int = 4096
    embedding_pooling: str = "mean_pooling"
    batch_size: int = 8
    num_gpus: int = 8
    max_seq_length: int = 32768
    files_per_partition: int = 1
    skip_embedding: bool = False
    reduce_methods: list[str] = field(default_factory=lambda: ["pca", "tsne", "umap"])
    reduce_n_components: int = 2
    config_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load config with ``_base`` inheritance."""
        path = Path(path)
        with open(path) as fh:
            child = yaml.safe_load(fh) or {}

        base_name = child.pop("_base", None)
        if base_name is not None:
            base_path = path.parent / base_name
            with open(base_path) as fh:
                data = yaml.safe_load(fh) or {}
            data.update({k: v for k, v in child.items() if v is not None})
        else:
            data = child

        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def for_config(self, config_name: str) -> PipelineConfig:
        """Return a copy with per-config overrides applied."""
        overrides = self.config_overrides.get(config_name, {})
        if not overrides:
            return self
        vals = {k: v for k, v in overrides.items() if k in self.__dataclass_fields__}
        return dataclasses.replace(self, **vals)


# ── Stage status ─────────────────────────────────────────────────────────────


class StageStatus(Enum):
    MISSING = "missing"
    PARTIAL = "partial"
    COMPLETE = "complete"


def check_stage_status(directory: Path) -> tuple[StageStatus, int]:
    """Return ``(status, file_count)`` for a stage output directory.

    - COMPLETE: only canonical ``part-*`` data files, no temp leftovers
    - PARTIAL:  has ``.tmp`` / ``_tmp_*`` or non-canonical data files
    - MISSING:  empty or does not exist
    """
    if not directory.exists():
        return StageStatus.MISSING, 0

    tmp_files = list(directory.glob("*.tmp")) + list(directory.glob("_tmp_*"))
    canonical = [
        f for f in directory.iterdir()
        if f.name.startswith("part-") and f.suffix in (".parquet", ".jsonl")
    ]
    other_data = [
        f for f in directory.iterdir()
        if f.suffix in (".parquet", ".jsonl")
        and not f.name.startswith("part-")
        and not f.name.startswith("_tmp_")
    ]

    if tmp_files or other_data:
        return StageStatus.PARTIAL, len(canonical) + len(other_data)
    if canonical:
        return StageStatus.COMPLETE, len(canonical)
    return StageStatus.MISSING, 0


# ── Cleanup ──────────────────────────────────────────────────────────────────


def cleanup_partial(directory: Path) -> None:
    """Remove temp files and non-canonical data files from an interrupted run."""
    for f in directory.iterdir():
        if f.name.startswith("_tmp_") or f.suffix == ".tmp":
            f.unlink()
            logger.debug(f"Removed temp: {f.name}")
        elif f.suffix in (".parquet", ".jsonl") and not f.name.startswith("part-"):
            f.unlink()
            logger.debug(f"Removed non-canonical: {f.name}")


def cleanup_raw_temps(directory: Path) -> None:
    """Remove ``.tmp`` download leftovers from raw/."""
    for tmp in directory.glob("*.tmp"):
        tmp.unlink()
        logger.debug(f"Removed raw temp: {tmp.name}")


# ── HuggingFace enumeration ─────────────────────────────────────────────────


def enumerate_dataset_splits(dataset_name: str) -> list[tuple[str, str]]:
    """Return every ``(config, split)`` pair for *dataset_name* via HF Hub."""
    from datasets import get_dataset_config_names, get_dataset_split_names

    try:
        configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
    except Exception:
        configs = []
    if not configs:
        configs = ["default"]

    pairs: list[tuple[str, str]] = []
    for cfg in configs:
        try:
            splits = get_dataset_split_names(
                dataset_name, cfg, trust_remote_code=True
            )
        except Exception:
            try:
                splits = get_dataset_split_names(
                    dataset_name, trust_remote_code=True
                )
            except Exception:
                splits = ["train"]
        for sp in splits:
            pairs.append((cfg, sp))
    return pairs


# ── Directory layout ─────────────────────────────────────────────────────────


def create_output_dirs(
    base: Path,
    dataset_name: str,
    config: str,
    split: str,
    model_id: str | None = None,
) -> tuple[Path, Path, Path | None, Path | None]:
    """Create and return ``(raw, preprocessed, embeddings, embreduced)``.

    If *model_id* is ``None``, embedding/embreduced dirs are not created.
    """
    root = base / dataset_name / config / split
    raw_dir = root / "raw"
    pre_dir = root / "preprocessed"
    dirs: list[Path] = [raw_dir, pre_dir]

    emb_dir: Path | None = None
    red_dir: Path | None = None
    if model_id is not None:
        emb_dir = root / "embeddings" / model_id
        red_dir = root / "embreduced" / model_id
        dirs += [emb_dir, red_dir]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    return raw_dir, pre_dir, emb_dir, red_dir


# ── Parquet rechunking ───────────────────────────────────────────────────────


def rechunk_parquet(directory: Path, chunk_size: int) -> int:
    """Ensure every parquet in *directory* has at most *chunk_size* rows.

    Idempotent: if all files are already canonical ``part-*`` names with
    <= *chunk_size* rows, this is a fast no-op.  Uses non-hidden ``_tmp_``
    prefix so ``cleanup_partial`` can catch interrupted rechunks.
    """
    import pandas as pd

    src_files = sorted(directory.glob("*.parquet"))
    if not src_files:
        return 0

    needs_rechunk = False
    for f in src_files:
        if not f.name.startswith("part-"):
            needs_rechunk = True
            break
        if pd.read_parquet(f, columns=[]).shape[0] > chunk_size:
            needs_rechunk = True
            break

    if not needs_rechunk:
        return len(src_files)

    logger.info(
        f"    Rechunking {len(src_files)} file(s) "
        f"→ {chunk_size} rows/chunk …"
    )

    part_idx = 0
    tmp_parts: list[Path] = []
    for src in src_files:
        df = pd.read_parquet(src)
        for start in range(0, len(df), chunk_size):
            part_idx += 1
            tmp = directory / f"_tmp_part_{part_idx:06d}.parquet"
            df.iloc[start : start + chunk_size].to_parquet(tmp, index=False)
            tmp_parts.append(tmp)
        src.unlink()

    n_parts = len(tmp_parts)
    for idx, tmp in enumerate(tmp_parts, start=1):
        dst = directory / f"part-{idx:06d}-of-{n_parts:06d}.parquet"
        tmp.rename(dst)

    return n_parts
