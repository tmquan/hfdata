#!/usr/bin/env python3
"""HuggingFace dataset downloader — NeMo Curator Pipeline on Ray.

Resume-aware: re-running skips completed config/split pairs, cleans up
partial runs, and picks up where it left off.  Use ``--force`` to redo all.

Usage
-----
    python Download.py
    python Download.py --config-name vietnamese-legal-documents
    python Download.py --config configs/my_dataset.yaml
    python Download.py --force
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from custom_hf_source import HuggingFaceDownloadExtractStage
from Helper import (
    PipelineConfig,
    StageStatus,
    check_stage_status,
    cleanup_partial,
    cleanup_raw_temps,
    create_output_dirs,
    enumerate_dataset_splits,
    rechunk_parquet,
)


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
    p.add_argument(
        "--force", action="store_true",
        help="Re-run all stages even if output already exists",
    )
    return p.parse_args()


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
    if args.force:
        logger.info("Force mode       : ON")

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
                pairs = enumerate_dataset_splits(
                    dataset_name,
                    forced_configs=cfg.hf_dataset_configs.get(dataset_name),
                    split_overrides=cfg.hf_dataset_splits.get(dataset_name),
                )
            except Exception as exc:
                logger.error(f"Failed to enumerate {dataset_name}: {exc}")
                continue

            logger.info(f"  Found {len(pairs)} config/split pair(s)")

            dcfg = cfg.for_dataset(dataset_name)

            for config_name, split in pairs:
                ccfg = dcfg.for_config(config_name)
                logger.info(f"  ▸ config={config_name!r}  split={split!r}")

                raw_dir, pre_dir, _, _ = create_output_dirs(
                    base_dir, dataset_name, config_name, split,
                )

                cleanup_raw_temps(raw_dir)

                pre_status, pre_count = check_stage_status(pre_dir)
                logger.info(
                    f"    preprocessed/ {pre_status.value} "
                    f"({pre_count} file(s))"
                )

                if pre_status == StageStatus.COMPLETE and not args.force:
                    logger.info("    SKIP — already complete")
                    summary.append({
                        "dataset": dataset_name,
                        "config": config_name,
                        "split": split,
                        "tasks": pre_count,
                        "output": str(pre_dir),
                        "status": "skipped",
                    })
                    continue

                if pre_status == StageStatus.PARTIAL:
                    logger.info("    Cleaning up partial run …")
                    cleanup_partial(pre_dir)

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
                        text_strategy=ccfg.text_strategy,
                    )
                )

                ### Declare Modify, Classify, Filter, Deduplicate HERE

                pipeline.add_stage(
                    ParquetWriter(path=str(pre_dir))
                    if ccfg.output_format == "parquet"
                    else JsonlWriter(path=str(pre_dir))
                )

                logger.info(f"    Running pipeline: {pipeline.name}")
                results = pipeline.run()

                if ccfg.output_format == "parquet":
                    n = rechunk_parquet(pre_dir, ccfg.chunk_size)
                    logger.info(f"    Chunked into {n} file(s)")

                n_tasks = len(results) if results else 0
                logger.info(f"    Done — {n_tasks} task(s)")

                summary.append({
                    "dataset": dataset_name,
                    "config": config_name,
                    "split": split,
                    "tasks": n_tasks,
                    "output": str(pre_dir),
                    "status": "ran",
                })

    finally:
        ray_client.stop()

    # ── summary ──────────────────────────────────────────────────────────
    logger.info(f"\n{'═' * 60}")
    logger.info("Pipeline complete.  Summary:")
    logger.info(f"{'═' * 60}")
    for entry in summary:
        logger.info(
            f"  {entry['dataset']} / {entry['config']} / {entry['split']}:  "
            f"{entry['tasks']} task(s)  [{entry['status']}]  →  {entry['output']}"
        )
    logger.info(f"{'═' * 60}")

    total = sum(e["tasks"] for e in summary)
    print(f"\nPipeline completed successfully!")
    print(f"Processed {total} task(s) total.")


if __name__ == "__main__":
    main()
