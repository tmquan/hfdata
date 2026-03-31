#!/usr/bin/env python3
"""HuggingFace dataset downloader + embedding extraction — two-pipeline script.

Pipeline 1 (download):  HuggingFaceDownloadExtractStage -> ParquetWriter
Pipeline 2 (embed):     ParquetReader -> EmbeddingCreatorStage -> ParquetWriter

Resume-aware: re-running skips completed stages, cleans up partial runs.
Use ``--force`` to redo all.

Usage
-----
    python DownloadExtract.py
    python DownloadExtract.py --config-name vietnamese-legal-documents
    python DownloadExtract.py --config configs/my_dataset.yaml
    python DownloadExtract.py --force
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
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


# ── Library patches ──────────────────────────────────────────────────────────

_LIBRARY_PATCHES: list[tuple[str, str, str]] = [
    (
        "nemo_curator.stages.text.embedders.base",
        "AutoModel.from_pretrained(self.model_identifier, cache_dir=self.cache_dir, local_files_only=True)",
        "AutoModel.from_pretrained(self.model_identifier, cache_dir=self.cache_dir, local_files_only=True, trust_remote_code=True, attn_implementation='eager')",
    ),
    (
        "nemo_curator.stages.text.embedders.base",
        "local_files_only=True, trust_remote_code=True)",
        "local_files_only=True, trust_remote_code=True, attn_implementation='eager')",
    ),
    (
        "nemo_curator.stages.text.models.tokenizer",
        "AutoConfig.from_pretrained(\n            self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only\n        )",
        "AutoConfig.from_pretrained(\n            self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only, trust_remote_code=True\n        )",
    ),
    (
        "nemo_curator.stages.text.models.tokenizer",
        "AutoTokenizer.from_pretrained(\n            self.model_identifier,\n            padding_side=self.padding_side,\n            cache_dir=self.cache_dir,\n            local_files_only=local_files_only,\n        )",
        "AutoTokenizer.from_pretrained(\n            self.model_identifier,\n            padding_side=self.padding_side,\n            cache_dir=self.cache_dir,\n            local_files_only=local_files_only,\n            trust_remote_code=True,\n        )",
    ),
    (
        "nemo_curator.stages.text.models.tokenizer",
        "self.tokenizer.batch_encode_plus(",
        "self.tokenizer(",
    ),
]


def _patch_nemo_curator_library() -> None:
    """One-time source-level patches for trust_remote_code support."""
    for module_path, old_text, new_text in _LIBRARY_PATCHES:
        spec = importlib.util.find_spec(module_path)
        if spec is None or spec.origin is None:
            logger.warning(f"Cannot locate module {module_path} — skip patch")
            continue
        src_file = Path(spec.origin)
        source = src_file.read_text(encoding="utf-8")
        if new_text in source:
            continue
        if old_text not in source:
            logger.warning(
                f"Patch target missing in {src_file.name} — "
                f"Curator version may differ"
            )
            continue
        src_file.write_text(source.replace(old_text, new_text, 1), encoding="utf-8")
        logger.info(f"Patched {src_file.name} (trust_remote_code=True)")


def _ensure_model_cached(model_id: str) -> None:
    """Pre-download model so workers with local_files_only=True can find it."""
    logger.info(f"Caching model: {model_id}")
    snapshot_download(repo_id=model_id)
    logger.info("Model cache ready.")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download HuggingFace datasets and extract embeddings"
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
    logger.info(f"Max seq length   : {cfg.max_seq_length}")
    logger.info(f"Embedding pooling: {cfg.embedding_pooling}")
    logger.info(f"Batch size       : {cfg.batch_size}")
    logger.info(f"Num GPUs         : {cfg.num_gpus}")
    logger.info(f"Chunk size       : {cfg.chunk_size}")
    if args.force:
        logger.info("Force mode       : ON")

    # ── environment guards ───────────────────────────────────────────────
    os.environ.setdefault("RAPIDS_NO_INITIALIZE", "1")
    os.environ.setdefault("CUDF_SPILL", "off")
    os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

    _patch_nemo_curator_library()
    _ensure_model_cached(cfg.model_id)

    logger.info("Loading NeMo Curator (this may take a moment) …")
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
    from nemo_curator.stages.text.io.reader.parquet import ParquetReader
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
                logger.info(
                    f"  ▸ config={config_name!r}  split={split!r}  "
                    f"(batch_size={ccfg.batch_size}, "
                    f"max_seq_length={ccfg.max_seq_length})"
                )

                raw_dir, pre_dir, emb_dir, red_dir = create_output_dirs(
                    base_dir, dataset_name, config_name, split, ccfg.model_id
                )

                cleanup_raw_temps(raw_dir)

                # ── Pipeline 1: download + extract ───────────────────
                pre_status, pre_count = check_stage_status(pre_dir)
                logger.info(
                    f"    preprocessed/ {pre_status.value} "
                    f"({pre_count} file(s))"
                )

                if pre_status == StageStatus.COMPLETE and not args.force:
                    logger.info("    SKIP download — already complete")
                    dl_tasks = pre_count
                else:
                    if pre_status == StageStatus.PARTIAL:
                        logger.info("    Cleaning up partial download …")
                        cleanup_partial(pre_dir)

                    pipeline_dl = Pipeline(
                        name=(
                            f"hf_download_{dataset_name.replace('/', '_')}"
                            f"_{config_name}_{split}"
                        ),
                        description=(
                            f"Download {dataset_name} "
                            f"config={config_name} split={split}"
                        ),
                    )

                    pipeline_dl.add_stage(
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

                    pipeline_dl.add_stage(
                        ParquetWriter(path=str(pre_dir))
                        if ccfg.output_format == "parquet"
                        else JsonlWriter(path=str(pre_dir))
                    )

                    logger.info(f"    Running download pipeline …")
                    dl_results = pipeline_dl.run()
                    dl_tasks = len(dl_results) if dl_results else 0
                    logger.info(f"    Download complete — {dl_tasks} task(s)")

                    if ccfg.output_format == "parquet":
                        n = rechunk_parquet(pre_dir, ccfg.chunk_size)
                        logger.info(f"    Split into {n} preprocessed chunk(s)")

                # ── Pipeline 2: embedding generation ─────────────────
                if ccfg.skip_embedding:
                    logger.info("    SKIP embedding — skip_embedding=true")
                    summary.append({
                        "dataset": dataset_name,
                        "config": config_name,
                        "split": split,
                        "dl_tasks": dl_tasks,
                        "emb_tasks": 0,
                        "dl_status": (
                            "skipped" if pre_status == StageStatus.COMPLETE
                            and not args.force else "ran"
                        ),
                        "emb_status": "skip_embedding",
                        "output": str(pre_dir),
                    })
                    continue

                n_chunks = rechunk_parquet(pre_dir, ccfg.chunk_size)
                logger.info(f"    preprocessed/ {n_chunks} chunk(s)")

                emb_status, emb_count = check_stage_status(emb_dir)
                logger.info(
                    f"    embeddings/   {emb_status.value} "
                    f"({emb_count} file(s))"
                )

                if emb_status == StageStatus.COMPLETE and not args.force:
                    logger.info("    SKIP embedding — already complete")
                    emb_tasks = emb_count
                else:
                    if emb_status == StageStatus.PARTIAL:
                        logger.info("    Cleaning up partial embeddings …")
                        cleanup_partial(emb_dir)

                    input_files = sorted(
                        str(p) for p in pre_dir.glob("*.parquet")
                    )

                    if not input_files:
                        logger.warning(
                            f"    No parquet files in {pre_dir}, "
                            f"skipping embedding stage"
                        )
                        summary.append({
                            "dataset": dataset_name,
                            "config": config_name,
                            "split": split,
                            "dl_tasks": dl_tasks,
                            "emb_tasks": 0,
                            "dl_status": "done",
                            "emb_status": "no_input",
                            "output": str(pre_dir),
                        })
                        continue

                    logger.info(
                        f"    Found {len(input_files)} parquet file(s) "
                        f"for embedding"
                    )

                    pipeline_emb = Pipeline(
                        name=(
                            f"embedding_{dataset_name.replace('/', '_')}"
                            f"_{config_name}_{split}"
                        ),
                        description=(
                            f"Embeddings for {dataset_name} "
                            f"config={config_name} split={split}"
                        ),
                        stages=[
                            ParquetReader(
                                file_paths=input_files,
                                files_per_partition=ccfg.files_per_partition,
                                fields=["text"],
                                _generate_ids=False,
                                read_kwargs={"dtype_backend": "numpy_nullable"},
                            ),
                            EmbeddingCreatorStage(
                                model_identifier=ccfg.model_id,
                                use_sentence_transformer=False,
                                text_field="text",
                                max_seq_length=ccfg.max_seq_length,
                                max_chars=None,
                                embedding_pooling=ccfg.embedding_pooling,
                                model_inference_batch_size=ccfg.batch_size,
                            ),
                            ParquetWriter(
                                path=str(emb_dir),
                                fields=["embeddings"],
                            ),
                        ],
                    )

                    logger.info(f"    Running embedding pipeline …")
                    emb_results = pipeline_emb.run()
                    emb_tasks = len(emb_results) if emb_results else 0
                    rechunk_parquet(emb_dir, ccfg.chunk_size)
                    logger.info(
                        f"    Embedding complete — {emb_tasks} task(s)"
                    )

                summary.append({
                    "dataset": dataset_name,
                    "config": config_name,
                    "split": split,
                    "dl_tasks": dl_tasks,
                    "emb_tasks": emb_tasks,
                    "dl_status": (
                        "skipped" if pre_status == StageStatus.COMPLETE
                        and not args.force else "ran"
                    ),
                    "emb_status": (
                        "skipped" if emb_status == StageStatus.COMPLETE
                        and not args.force else "ran"
                    ),
                    "output": str(pre_dir),
                    "embeddings": str(emb_dir),
                })

    finally:
        ray_client.stop()

    # ── summary ──────────────────────────────────────────────────────────
    logger.info(f"\n{'═' * 60}")
    logger.info("Pipeline complete.  Summary:")
    logger.info(f"{'═' * 60}")
    for entry in summary:
        logger.info(
            f"  {entry['dataset']} / {entry['config']} / {entry['split']}:"
        )
        logger.info(
            f"    download  : {entry['dl_tasks']} task(s)  "
            f"[{entry['dl_status']}]  →  {entry['output']}"
        )
        logger.info(
            f"    embedding : {entry['emb_tasks']} task(s)  "
            f"[{entry['emb_status']}]  →  {entry.get('embeddings', 'n/a')}"
        )
    logger.info(f"{'═' * 60}")

    print(f"\nPipeline completed successfully!")
    print(
        f"Download: {sum(e['dl_tasks'] for e in summary)} task(s), "
        f"Embedding: {sum(e['emb_tasks'] for e in summary)} task(s)"
    )


if __name__ == "__main__":
    main()
