#!/usr/bin/env python3
"""HuggingFace dataset downloader — NeMo Curator Pipeline on Ray.

Usage
-----
    # default: download Vietnamese legal documents
    python Download.py

    # pick a config by name (resolves to configs/<name>.yaml)
    python Download.py --config-name vietnamese-legal-documents

    # or give a full path
    python Download.py --config configs/my_dataset.yaml

    # override from CLI
    python Download.py --datasets org/dataset1 org/dataset2 --output_format jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from custom_hf_source import HuggingFaceDownloadExtractStage


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Typed mirror of ``configs/default.yaml``."""

    datasets: list[str] = field(default_factory=list)
    output_dir: str = "./datasets"
    cache_dir: str = "~/.cache/huggingface"
    output_format: str = "parquet"
    model_id: str = "nvidia/llama-embed-nemotron-8b"
    embedding_dim: int = 4096

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and preprocess HuggingFace datasets"
    )
    p.add_argument(
        "--config", default=None, help="Full path to YAML config file",
    )
    p.add_argument(
        "--config-name", default="vietnamese-legal-documents",
        dest="config_name", help="Config name (looked up in configs/<name>.yaml)",
    )
    p.add_argument("--datasets", nargs="+", help="Override dataset list")
    p.add_argument("--output_dir", help="Override output directory")
    p.add_argument(
        "--output_format", choices=["parquet", "jsonl"], help="Override format",
    )
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────


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


def create_output_dirs(
    base: Path, dataset_name: str, config: str, split: str, model_id: str,
) -> tuple[Path, Path, Path, Path]:
    """Create and return ``(raw, preprocessed, embeddings, embreduced)``."""
    root = base / dataset_name / config / split
    raw_dir = root / "raw"
    pre_dir = root / "preprocessed"
    emb_dir = pre_dir / "embeddings" / model_id
    red_dir = pre_dir / "embreduced" / model_id
    for d in (raw_dir, pre_dir, emb_dir, red_dir):
        d.mkdir(parents=True, exist_ok=True)
    return raw_dir, pre_dir, emb_dir, red_dir


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # ── resolve config ───────────────────────────────────────────────────
    if args.config is not None:
        config_path = Path(args.config)
    else:
        config_path = Path("configs") / f"{args.config_name}.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    cfg = PipelineConfig.from_yaml(config_path)

    if args.datasets:
        cfg.datasets = args.datasets
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.output_format:
        cfg.output_format = args.output_format

    base_dir = Path(cfg.output_dir).expanduser().resolve()
    cache_dir = str(Path(cfg.cache_dir).expanduser())

    logger.info(f"Output directory : {base_dir}")
    logger.info(f"Output format    : {cfg.output_format}")
    logger.info(f"Datasets         : {cfg.datasets}")
    logger.info(f"Model ID         : {cfg.model_id}")

    # ── heavy nemo_curator imports (lazy — only now) ─────────────────────
    os.environ.setdefault("RAPIDS_NO_INITIALIZE", "1")
    os.environ.setdefault("CUDF_SPILL", "off")

    logger.info("Loading NeMo Curator (this may take a moment) …")
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
    from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
    logger.info("NeMo Curator loaded.")

    # ── Ray client ───────────────────────────────────────────────────────
    from nemo_curator.core.client import RayClient

    ray_client = RayClient()
    ray_client.start()

    summary: list[dict[str, Any]] = []

    try:
        for dataset_name in cfg.datasets:
            logger.info(f"{'─' * 60}")
            logger.info(f"Dataset: {dataset_name}")

            try:
                pairs = enumerate_dataset_splits(dataset_name)
            except Exception as exc:
                logger.error(f"Failed to enumerate {dataset_name}: {exc}")
                continue

            logger.info(f"  Found {len(pairs)} config/split pair(s)")

            for config_name, split in pairs:
                logger.info(f"  ▸ config={config_name!r}  split={split!r}")

                raw_dir, pre_dir, emb_dir, red_dir = create_output_dirs(
                    base_dir, dataset_name, config_name, split, cfg.model_id
                )

                # ── build pipeline ───────────────────────────────────
                pipeline = Pipeline(
                    name=(
                        f"hf_download_{dataset_name.replace('/', '_')}"
                        f"_{config_name}_{split}"
                    ),
                    description=(
                        f"Download {dataset_name} "
                        f"config={config_name} split={split}"
                    ),
                )

                pipeline.add_stage(
                    HuggingFaceDownloadExtractStage(
                        dataset_name=dataset_name,
                        split=split,
                        config_name=config_name,
                        cache_dir=cache_dir,
                        output_dir=str(raw_dir),
                    )
                )

                ### Declare Modify, Classify, Filter, Deduplicate HERE

                pipeline.add_stage(
                    ParquetWriter(path=str(pre_dir))
                    if cfg.output_format == "parquet"
                    else JsonlWriter(path=str(pre_dir))
                )

                # ── execute ──────────────────────────────────────────
                logger.info(f"    Running pipeline: {pipeline.name}")
                results = pipeline.run()

                n_tasks = len(results) if results else 0
                logger.info(
                    f"    Pipeline completed — {n_tasks} task(s)"
                )

                summary.append(
                    {
                        "dataset": dataset_name,
                        "config": config_name,
                        "split": split,
                        "tasks": n_tasks,
                        "output": str(pre_dir),
                    }
                )

    finally:
        ray_client.stop()

    # ── summary ──────────────────────────────────────────────────────────
    logger.info(f"\n{'═' * 60}")
    logger.info("Pipeline complete.  Summary:")
    logger.info(f"{'═' * 60}")
    for entry in summary:
        logger.info(
            f"  {entry['dataset']} / {entry['config']} / {entry['split']}:  "
            f"{entry['tasks']} task(s)  →  {entry['output']}"
        )
    logger.info(f"{'═' * 60}")

    total = sum(e["tasks"] for e in summary)
    print(f"\nPipeline completed successfully!")
    print(f"Processed {total} task(s) total.")


if __name__ == "__main__":
    main()
