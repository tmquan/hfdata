#!/usr/bin/env python3
"""HuggingFace dataset: download + embed + dimensionality-reduce.

Pipeline 1 (download):  HuggingFaceDownloadExtractStage -> ParquetWriter
Pipeline 2 (embed):     ParquetReader -> EmbeddingCreatorStage -> ParquetWriter
Pipeline 3 (reduce):    embeddings -> PCA / t-SNE / UMAP (2D) -> ParquetWriter

GPU (cuML/cuDF) preferred; falls back to sklearn / umap-learn / pandas.
Resume-aware.  Use ``--force`` to redo all.

Usage
-----
    python DownloadExtractReduce.py
    python DownloadExtractReduce.py --config-name vietnamese-legal-documents
    python DownloadExtractReduce.py --force
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
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


# ── Library patches (same as DownloadExtract.py) ─────────────────────────────

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


# ── Dimensionality reduction ─────────────────────────────────────────────────


_PCA_PRE_DIM = 50


def _pca_prereduction(X: np.ndarray) -> np.ndarray:
    """Reduce high-dim embeddings to 50D via PCA before t-SNE / UMAP."""
    if X.shape[1] <= _PCA_PRE_DIM:
        return X
    try:
        from cuml import PCA as cuPCA
        logger.info(f"      PCA pre-reduction {X.shape[1]}→{_PCA_PRE_DIM} via cuML")
        return cuPCA(n_components=_PCA_PRE_DIM).fit_transform(X)
    except Exception:
        pass
    from sklearn.decomposition import PCA
    logger.info(f"      PCA pre-reduction {X.shape[1]}→{_PCA_PRE_DIM} via sklearn")
    return PCA(n_components=_PCA_PRE_DIM).fit_transform(X)


def _run_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """PCA: try cuML, fall back to sklearn."""
    try:
        from cuml import PCA as cuPCA
        logger.info(f"      PCA via cuML (n={n_components})")
        return cuPCA(n_components=n_components).fit_transform(X)
    except Exception:
        pass
    from sklearn.decomposition import PCA
    logger.info(f"      PCA via sklearn (n={n_components})")
    return PCA(n_components=n_components).fit_transform(X)


def _run_tsne(X: np.ndarray, n_components: int) -> np.ndarray:
    """t-SNE: try cuML, fall back to sklearn.  PCA pre-reduces first."""
    X_pre = _pca_prereduction(X)
    try:
        from cuml import TSNE as cuTSNE
        logger.info(f"      t-SNE via cuML (n={n_components})")
        return cuTSNE(n_components=n_components).fit_transform(X_pre)
    except Exception:
        pass
    from sklearn.manifold import TSNE
    logger.info(f"      t-SNE via sklearn (n={n_components})")
    return TSNE(
        n_components=n_components,
        init="pca",
        learning_rate="auto",
        n_jobs=1,
    ).fit_transform(X_pre)


def _run_umap(X: np.ndarray, n_components: int) -> np.ndarray:
    """UMAP: try cuML, fall back to umap-learn, or skip.  PCA pre-reduces first."""
    X_pre = _pca_prereduction(X)
    try:
        from cuml import UMAP as cuUMAP
        logger.info(f"      UMAP via cuML (n={n_components})")
        return cuUMAP(n_components=n_components).fit_transform(X_pre)
    except Exception:
        pass
    try:
        from umap import UMAP
        logger.info(f"      UMAP via umap-learn (n={n_components})")
        return UMAP(n_components=n_components).fit_transform(X_pre)
    except Exception:
        logger.warning("      UMAP skipped — neither cuml.UMAP nor umap-learn available")
        return None


_AXIS_NAMES: dict[int, tuple[str, ...]] = {
    2: ("x", "y"),
    3: ("x", "y", "z"),
}

_REDUCERS: dict[str, Any] = {
    "pca": _run_pca,
    "tsne": _run_tsne,
    "umap": _run_umap,
}


def reduce_embeddings(
    emb_dir: Path,
    red_dir: Path,
    methods: list[str],
    n_components: int,
    chunk_size: int,
) -> int:
    """Read all embedding parquets, reduce to 2D, write to *red_dir*.

    Returns the number of output parquet files written.
    """
    import pandas as pd

    if n_components != 2:
        raise NotImplementedError(
            f"3D reduction not yet implemented (n_components={n_components})"
        )

    emb_files = sorted(emb_dir.glob("*.parquet"))
    if not emb_files:
        logger.warning(f"    No embedding files in {emb_dir}")
        return 0

    logger.info(f"    Loading {len(emb_files)} embedding file(s) …")
    all_emb: list[np.ndarray] = []
    for f in emb_files:
        df = pd.read_parquet(f)
        all_emb.append(np.stack(df["embeddings"].values))
    X = np.vstack(all_emb).astype(np.float32)
    logger.info(f"    Embedding matrix: {X.shape}")

    result_cols: dict[str, np.ndarray] = {}
    for method in methods:
        fn = _REDUCERS.get(method)
        if fn is None:
            logger.warning(f"    Unknown reduce method: {method!r} — skipping")
            continue
        logger.info(f"    Running {method} …")
        reduced = fn(X, n_components)
        if reduced is None:
            for i, axis in enumerate(_AXIS_NAMES[n_components]):
                result_cols[f"{method}_{n_components}d_{axis}"] = np.full(len(X), np.nan)
            continue
        if hasattr(reduced, "to_numpy"):
            reduced = reduced.to_numpy()
        reduced = np.asarray(reduced, dtype=np.float32)
        for i, axis in enumerate(_AXIS_NAMES[n_components]):
            result_cols[f"{method}_{n_components}d_{axis}"] = reduced[:, i]

    out_df = pd.DataFrame(result_cols)
    logger.info(f"    Reduced DataFrame: {out_df.shape} cols={list(out_df.columns)}")

    red_dir.mkdir(parents=True, exist_ok=True)
    n_parts = 0
    for start in range(0, len(out_df), chunk_size):
        n_parts += 1
        tmp = red_dir / f"_tmp_part_{n_parts:06d}.parquet"
        out_df.iloc[start : start + chunk_size].to_parquet(tmp, index=False)

    for idx in range(1, n_parts + 1):
        src = red_dir / f"_tmp_part_{idx:06d}.parquet"
        dst = red_dir / f"part-{idx:06d}-of-{n_parts:06d}.parquet"
        src.rename(dst)

    logger.info(f"    Wrote {n_parts} reduced parquet(s) to {red_dir}")
    return n_parts


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download, embed, and reduce HuggingFace datasets"
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

    logger.info(f"Output directory  : {base_dir}")
    logger.info(f"Output format     : {cfg.output_format}")
    logger.info(f"Datasets          : {cfg.datasets}")
    logger.info(f"Model ID          : {cfg.model_id}")
    logger.info(f"Reduce methods    : {cfg.reduce_methods}")
    logger.info(f"Reduce components : {cfg.reduce_n_components}")
    if args.force:
        logger.info("Force mode        : ON")

    # ── environment guards ───────────────────────────────────────────────
    os.environ.setdefault("RAPIDS_NO_INITIALIZE", "1")
    os.environ.setdefault("CUDF_SPILL", "off")
    os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
    os.environ.setdefault("MKL_NUM_THREADS", "64")
    os.environ.setdefault("NUMBA_CUDA_LOG_LEVEL", "ERROR")

    import warnings
    warnings.filterwarnings("ignore", message=".*cuDriverGetVersion.*")
    warnings.filterwarnings("ignore", message=".*Not patching Numba.*")

    # cuda-bindings v13 requires explicit cudart load before cuDF/cuML
    try:
        from cuda.pathfinder import load_nvidia_dynamic_lib
        load_nvidia_dynamic_lib("cudart")
    except Exception:
        pass

    _patch_nemo_curator_library()
    _ensure_model_cached(cfg.model_id)

    logger.info("Loading NeMo Curator (this may take a moment) …")
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
    from nemo_curator.stages.text.io.reader.parquet import ParquetReader
    from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
    from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
    logger.info("NeMo Curator loaded.")

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
                ccfg = cfg.for_config(config_name)
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
                        )
                    )

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
                    logger.info(
                        "    SKIP embedding + reduce — skip_embedding=true"
                    )
                    summary.append({
                        "dataset": dataset_name,
                        "config": config_name,
                        "split": split,
                        "dl_status": (
                            "skipped" if pre_status == StageStatus.COMPLETE
                            and not args.force else "ran"
                        ),
                        "emb_status": "skip",
                        "red_status": "skip",
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
                            f"skipping embedding + reduce"
                        )
                        summary.append({
                            "dataset": dataset_name,
                            "config": config_name,
                            "split": split,
                            "dl_status": "done",
                            "emb_status": "no_input",
                            "red_status": "skip",
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
                    rechunk_parquet(emb_dir, ccfg.chunk_size)
                    logger.info(
                        f"    Embedding complete — "
                        f"{len(emb_results) if emb_results else 0} task(s)"
                    )

                # ── Pipeline 3: dimensionality reduction ─────────────
                red_status, red_count = check_stage_status(red_dir)
                logger.info(
                    f"    embreduced/   {red_status.value} "
                    f"({red_count} file(s))"
                )

                if red_status == StageStatus.COMPLETE and not args.force:
                    logger.info("    SKIP reduce — already complete")
                    red_n = red_count
                else:
                    if red_status == StageStatus.PARTIAL:
                        logger.info("    Cleaning up partial reduction …")
                        cleanup_partial(red_dir)

                    logger.info(
                        f"    Running reduce: {ccfg.reduce_methods} "
                        f"→ {ccfg.reduce_n_components}D"
                    )
                    red_n = reduce_embeddings(
                        emb_dir=emb_dir,
                        red_dir=red_dir,
                        methods=ccfg.reduce_methods,
                        n_components=ccfg.reduce_n_components,
                        chunk_size=ccfg.chunk_size,
                    )

                summary.append({
                    "dataset": dataset_name,
                    "config": config_name,
                    "split": split,
                    "dl_status": (
                        "skipped" if pre_status == StageStatus.COMPLETE
                        and not args.force else "ran"
                    ),
                    "emb_status": (
                        "skipped" if emb_status == StageStatus.COMPLETE
                        and not args.force else "ran"
                    ),
                    "red_status": (
                        "skipped" if red_status == StageStatus.COMPLETE
                        and not args.force else "ran"
                    ),
                    "red_files": red_n,
                    "output": str(red_dir),
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
        logger.info(f"    download  [{entry['dl_status']}]")
        logger.info(f"    embedding [{entry['emb_status']}]")
        logger.info(
            f"    reduce    [{entry['red_status']}]"
            + (f"  {entry.get('red_files', 0)} file(s)  →  {entry.get('output', 'n/a')}" if "output" in entry else "")
        )
    logger.info(f"{'═' * 60}")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
