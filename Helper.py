"""Shared helpers for Download.py and DownloadExtract.py."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import yaml
from loguru import logger

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_torch_dtype(name: str) -> torch.dtype:
    """Map a human-friendly dtype string to a ``torch.dtype``."""
    dt = _DTYPE_MAP.get(name.lower())
    if dt is None:
        raise ValueError(
            f"Unknown model_dtype {name!r}; choose from {list(_DTYPE_MAP)}"
        )
    return dt


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
    model_dtype: str = "bfloat16"
    batch_size: int = 8
    num_gpus: int = 8
    max_seq_length: int = 32768
    files_per_partition: int = 1
    skip_embedding: bool = False
    text_strategy: str = "auto"
    reduce_methods: list[str] = field(default_factory=lambda: ["pca", "tsne", "umap"])
    reduce_n_components: int = 2
    config_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    dataset_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # If set for a repo id, use these HF config names instead of Hub list_dataset_configs
    # (avoids bogus "default" when the API fails or the dataset has no default config).
    hf_dataset_configs: dict[str, list[str]] = field(default_factory=dict)
    # Optional: repo_id -> HF config name -> split name (when Hub/API says "train" but
    # the dataset only exposes e.g. "data", or configs disagree on split names).
    hf_dataset_splits: dict[str, dict[str, str]] = field(default_factory=dict)

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

    def for_dataset(self, dataset_name: str) -> PipelineConfig:
        """Return a copy with per-dataset overrides applied."""
        overrides = self.dataset_overrides.get(dataset_name, {})
        if not overrides:
            return self
        vals = {k: v for k, v in overrides.items() if k in self.__dataclass_fields__}
        return dataclasses.replace(self, **vals)

    def for_config(self, config_name: str) -> PipelineConfig:
        """Return a copy with per-config overrides applied."""
        overrides = self.config_overrides.get(config_name, {})
        if not overrides:
            return self
        vals = {k: v for k, v in overrides.items() if k in self.__dataclass_fields__}
        return dataclasses.replace(self, **vals)


# ── Text strategy registry ───────────────────────────────────────────────────


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Convert ``[{role, content}, ...]`` into ``"Role: content\\n..."``.

    Handles nested content (list-of-dicts), ``tool_calls`` structs, and
    content that is ``None`` (common in tool-call turns).
    """
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "unknown")).capitalize()
        content = msg.get("content")

        if content is None:
            content = ""
        elif isinstance(content, list):
            content = " ".join(
                str(c.get("text", c)) if isinstance(c, dict) else str(c)
                for c in content
            )
        else:
            content = str(content)

        tool_calls = msg.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            tc_parts = []
            for tc in tool_calls:
                fn = tc.get("function", tc) if isinstance(tc, dict) else tc
                if isinstance(fn, dict):
                    name = fn.get("name", "")
                    args = fn.get("arguments", "")
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False, default=str)
                    tc_parts.append(f"{name}({args})" if name else str(fn))
                else:
                    tc_parts.append(str(fn))
            if tc_parts:
                content = f"{content}\n[tool_calls: {', '.join(tc_parts)}]" if content else f"[tool_calls: {', '.join(tc_parts)}]"

        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _serialize_tools(tools: list[dict[str, Any]]) -> str:
    """Compact text summary of tool/function definitions."""
    parts: list[str] = []
    for tool in tools:
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            parts.append(f"Tool: {fn}")
            continue
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        parts.append(f"Tool: {name} - {desc}" if desc else f"Tool: {name}")
    return "\n".join(parts)


def _strategy_messages_list(record: dict[str, Any]) -> str | None:
    """Flatten the ``messages`` column (list of role/content dicts).

    Handles tool_calls in message turns (Nemotron-Post-Training v1 tool
    subset) and skips the ``metadata`` column (JSON string) automatically.
    """
    messages = record.get("messages")
    if not messages or not isinstance(messages, list):
        return None
    return _flatten_messages(messages)


def _strategy_messages_concat(record: dict[str, Any]) -> str | None:
    """Flatten ``input`` message list and append ``output``.

    Used by Llama-Nemotron-Post-Training-Dataset:
    - SFT subsets: ``input`` (list of messages) + ``output`` (string)
    - RL/when2call subsets: ``messages`` (list) — auto-detected as fallback

    Strips ``system_prompt`` if it duplicates the first system message.
    """
    inp = record.get("input")
    if isinstance(inp, list):
        sys_prompt = record.get("system_prompt", "")
        if (
            sys_prompt
            and inp
            and isinstance(inp[0], dict)
            and inp[0].get("role") == "system"
            and str(inp[0].get("content", "")).strip() == str(sys_prompt).strip()
        ):
            pass
        elif sys_prompt:
            inp = [{"role": "system", "content": sys_prompt}] + list(inp)

        text = _flatten_messages(inp)
        output = record.get("output", "")
        if output:
            text += f"\nAssistant: {output}"
        return text

    messages = record.get("messages")
    if isinstance(messages, list):
        return _flatten_messages(messages)
    # Plain document corpora (e.g. HF ``content`` config with markdown in ``content``/``text``)
    for key in ("text", "content", "document", "body"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _strategy_rl_blend(record: dict[str, Any]) -> str | None:
    """Extract messages from ``responses_create_params.input`` + ground_truth.

    Used by Nemotron-3-Nano-RL-Training-Blend. The nested dict structure:
    ``responses_create_params["input"]`` contains the message list.
    ``ground_truth`` is a list of tool-call dicts appended as structured text.
    """
    rcp = record.get("responses_create_params")
    if isinstance(rcp, str):
        try:
            rcp = json.loads(rcp)
        except (json.JSONDecodeError, TypeError):
            rcp = None
    if isinstance(rcp, dict):
        messages = rcp.get("input")
        if isinstance(messages, list):
            text = _flatten_messages(messages)
            gt = record.get("ground_truth")
            if gt:
                if isinstance(gt, str):
                    text += f"\nGround truth: {gt}"
                else:
                    text += f"\nGround truth: {json.dumps(gt, ensure_ascii=False, default=str)}"
            return text
    return _strategy_messages_list(record)


def _strategy_agentic(record: dict[str, Any]) -> str | None:
    """Serialize tool definitions as prefix, then flatten messages.

    Used by Nemotron-Agentic-v1 and Nemotron-SWE-v1. Captures the full
    agentic task structure: available tools + multi-turn conversation
    including tool calls and tool outputs.

    WARNING: SWE-v1 messages can exceed 100k chars. The embedding model's
    tokenizer will truncate to ``max_seq_length``.
    """
    sections: list[str] = []
    tools = record.get("tools")
    if isinstance(tools, list) and tools:
        sections.append(_serialize_tools(tools))
    messages = record.get("messages")
    if isinstance(messages, list):
        sections.append(_flatten_messages(messages))
    return "\n\n".join(sections) if sections else None


def _strategy_math_proof(record: dict[str, Any]) -> str | None:
    """Concatenate ``problem`` + Lean 4 ``formal_statement``.

    Used by Nemotron-Math-Proofs-v1. Falls back to ``messages`` if the
    primary fields are empty (some rows have ``messages`` populated instead).
    Skips None-valued fields (url, user_name, sft_line_number).
    """
    problem = str(record.get("problem", "") or "").strip()
    header = str(record.get("lean_header", "") or "").strip()
    formal = str(record.get("formal_statement", "") or "").strip()
    if not problem and not formal:
        return _strategy_messages_list(record)
    parts = []
    if problem:
        parts.append(f"Problem: {problem}")
    if header or formal:
        parts.append("Formal Statement (Lean 4):")
        if header:
            parts.append(header)
        if formal:
            parts.append(formal)
    return "\n\n".join(parts)


def _strategy_math_v2(record: dict[str, Any]) -> str | None:
    """Flatten ``messages`` if present, else fall back to ``problem``.

    Used by Nemotron-Math-v2 (low/medium/high difficulty tiers).
    Messages typically contain the user prompt with \\\\boxed{} instruction
    and the assistant solution.
    """
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        return _flatten_messages(messages)
    problem = str(record.get("problem", "") or "").strip()
    return problem if problem else None


def _strategy_raw_text(record: dict[str, Any]) -> str | None:
    """Use the ``text`` field directly.

    Used by pretraining datasets where the text column already contains
    the full document. Skips records with no ``text`` column (e.g.
    Code-Metadata subset which only has repo/commit_id/rel_path).
    """
    text = record.get("text")
    if text is None:
        return None
    text = str(text).strip()
    return text if text else None


def _strategy_auto_detect(record: dict[str, Any]) -> str | None:
    """Smart auto-detection: infer the best strategy from available columns.

    Checks for common column patterns across Nemotron datasets and
    dispatches to the appropriate strategy function.
    """
    keys = set(record.keys())

    if "responses_create_params" in keys:
        return _strategy_rl_blend(record)
    if "tools" in keys and "messages" in keys:
        return _strategy_agentic(record)
    if "formal_statement" in keys or "lean_header" in keys:
        return _strategy_math_proof(record)
    if "input" in keys and isinstance(record["input"], list):
        return _strategy_messages_concat(record)
    if "messages" in keys and isinstance(record["messages"], list):
        return _strategy_messages_list(record)
    if "problem" in keys:
        return _strategy_math_v2(record)
    if "text" in keys:
        return _strategy_raw_text(record)
    return None


TEXT_STRATEGIES: dict[str, Callable[[dict[str, Any]], str | None] | None] = {
    "auto": None,
    "smart": _strategy_auto_detect,
    "messages_list": _strategy_messages_list,
    "messages_concat": _strategy_messages_concat,
    "rl_blend": _strategy_rl_blend,
    "agentic": _strategy_agentic,
    "math_proof": _strategy_math_proof,
    "math_v2": _strategy_math_v2,
    "raw_text": _strategy_raw_text,
}


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


def enumerate_dataset_splits(
    dataset_name: str,
    forced_configs: list[str] | None = None,
    split_overrides: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Return every ``(config, split)`` pair for *dataset_name* via HF Hub.

    If *forced_configs* is set (from YAML ``hf_dataset_configs``), use it instead
    of ``datasets.get_dataset_config_names`` so multi-config datasets are not
    collapsed to a single misleading ``default`` config.

    If *split_overrides* is set (from YAML ``hf_dataset_splits``), keys are HF
    config names and values are the exact split to use (e.g. ``content`` → ``data``
    when ``get_dataset_split_names`` wrongly returns ``train``).
    """
    from datasets import get_dataset_config_names, get_dataset_split_names

    if forced_configs:
        configs = list(forced_configs)
        logger.info(
            f"Using {len(configs)} HF config(s) from config file "
            f"for {dataset_name!r}: {configs}"
        )
    else:
        try:
            configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
        except Exception:
            configs = []
        if not configs:
            configs = ["default"]
            logger.warning(
                f"No HF config names from Hub for {dataset_name!r} — "
                f"using {configs!r}. Set ``hf_dataset_configs`` in YAML if this "
                f"dataset needs metadata/content (or other) configs."
            )

    so = split_overrides or {}
    pairs: list[tuple[str, str]] = []
    for cfg in configs:
        if cfg in so:
            splits = [so[cfg]]
            logger.info(
                f"Using split override for {dataset_name!r} config={cfg!r}: {splits[0]!r}"
            )
        else:
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


# ── Embedding NaN audit ──────────────────────────────────────────────────────


def audit_nan_embeddings(directory: Path, embedding_col: str = "embeddings") -> int:
    """Log and count rows with NaN/None in *embedding_col* across parquets in *directory*.

    Returns the total number of bad rows found.
    """
    import numpy as np
    import pandas as pd

    def _is_bad(v: Any) -> bool:
        arr = np.asarray(v)
        if arr.dtype == object:
            return any(x is None for x in arr.flat)
        return bool(np.isnan(arr).any())

    total_bad = 0
    total_rows = 0
    for f in sorted(directory.glob("*.parquet")):
        df = pd.read_parquet(f, columns=[embedding_col])
        n_bad = int(df[embedding_col].apply(_is_bad).sum())
        total_bad += n_bad
        total_rows += len(df)
    if total_bad:
        logger.warning(
            f"    {total_bad}/{total_rows} embedding rows contain NaN/None in {directory}"
        )
    else:
        logger.info(f"    All {total_rows} embedding rows are finite")
    return total_bad


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
        nrows = len(df)
        if nrows == 0:
            # Old bug: inner loop did nothing but we still unlinked → 0 output files.
            logger.warning(
                f"    Empty parquet (0 rows), removing {src.name!r}. "
                f"Fix upstream download (wrong HF config?) — see hf_dataset_configs in YAML."
            )
            src.unlink()
            continue
        for start in range(0, nrows, chunk_size):
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
