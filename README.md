# hfdata

Best Practice for Hugging Face dataset handling with NeMo Curator — download,
embed, reduce, and explore.

## Setup

```bash
conda create -n nemo python=3.12
conda activate nemo

pip install uv

# ─── PyTorch (CUDA 13.0) ────────────────────────────────────────────────────
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# or use the NGC container instead:
# sudo docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.11-py3

pip install flash-attn --no-build-isolation 
# ─── build deps (needed by native extensions) ───────────────────────────────
uv pip install setuptools setuptools_scm setuptools_rust \
    wheel wheel_stub psutil pybind11

# ─── NeMo Curator (from source — base only, no [all] extras) ────────────────
git clone https://github.com/NVIDIA-NeMo/Curator.git /tmp/nemo-curator 2>/dev/null || true
pip install --no-build-isolation /tmp/nemo-curator

# ─── (optional) RAPIDS CUDA 13 for GPU-accelerated dedup / fuzzy matching ───
# uv pip install cudf-cu13 dask-cudf-cu13 cuml-cu13 cugraph-cu13 \
#     --extra-index-url https://pypi.nvidia.com

# ─── remaining deps ─────────────────────────────────────────────────────────
uv pip install -r requirements.txt

# ─── Plotly static-image export (Kaleido + Chromium) ─────────────────────────
# Kaleido ≥ 1.0 needs Chrome/Chromium at runtime. The [chromium] extra
# auto-downloads a bundled copy. If you already have Chrome installed system-
# wide you can skip this, but the extra is included in requirements.txt by
# default so it should just work.
# On headless Linux you also need Chromium's shared-library deps:
sudo apt-get install -y libnss3 libatk-bridge2.0-0 libcups2 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libxkbcommon0 libpango-1.0-0 libcairo2 libasound2t64
```

## Project Structure

```
hfdata/
├── configs/
│   ├── default.yaml                       # base pipeline configuration
│   ├── vietnamese-legal-documents.yaml    # dataset-specific overrides
│   ├── llama-nemotron-post-training.yaml  # Llama-Nemotron Post-Training
│   ├── nemotron-post-training-v1.yaml     # Nemotron Post-Training v1
│   ├── nemotron-post-training-v2.yaml     # Nemotron Post-Training v2
│   └── nemotron-post-training-v3.yaml     # Nemotron v3 collection (8 datasets)
├── custom_hf_source/                      # custom HuggingFace download stage
├── datasets/                              # output directory (created by pipeline)
├── Download.py                            # Stage 1: download dataset to parquet
├── DownloadExtract.py                     # Stage 1+2: download + compute embeddings
├── DownloadExtractReduce.py               # Stage 1+2+3: download + embed + dim-reduce
├── Helper.py                              # shared utilities, config, text strategies
├── Explorer.ipynb                         # interactive EDA notebook
├── requirements.txt                       # pinned Python dependencies
└── README.md
```

## Usage

### 1. Download only

```bash
python Download.py
python Download.py --config-name vietnamese-legal-documents
python Download.py --config configs/my_dataset.yaml
```

### 2. Download + Embed

```bash
python DownloadExtract.py
python DownloadExtract.py --config-name vietnamese-legal-documents
```

### 3. Download + Embed + Dimensionality Reduction

Runs the full pipeline: download, compute embeddings with
`nvidia/llama-embed-nemotron-8b`, then reduce to 2D via PCA, t-SNE, and UMAP.

```bash
python DownloadExtractReduce.py
python DownloadExtractReduce.py --config-name vietnamese-legal-documents
python DownloadExtractReduce.py --force   # redo all stages
```

#### Nemotron Post-Training Datasets

```bash
# Llama-Nemotron Post-Training (SFT + RL, messages_concat strategy)
python DownloadExtractReduce.py --config-name llama-nemotron-post-training

# Nemotron Post-Training v1 (code/math/stem/tool/chat, messages_list strategy)
python DownloadExtractReduce.py --config-name nemotron-post-training-v1

# Nemotron Post-Training v2 (+ multilingual subsets, messages_list strategy)
python DownloadExtractReduce.py --config-name nemotron-post-training-v2

# Nemotron v3 collection — 8 datasets with per-dataset text strategies
# (Science, Agentic, Math-Proofs, Competitive-Programming, SWE, etc.)
python DownloadExtractReduce.py --config-name nemotron-post-training-v3
```

All three scripts are **resume-aware** — re-running skips completed stages and
cleans up partial runs. Pass `--force` to reprocess everything.

### 4. Explore

Open `Explorer.ipynb` in Jupyter to run interactive EDA: schema overview, text
length distribution, category bar charts, cross-tab heatmaps, and 2D embedding
scatter plots with algorithm/color toggles.

```bash
jupyter notebook Explorer.ipynb
```

## Configuration

Pipeline settings live in `configs/default.yaml`. Dataset-specific overrides
(e.g. batch size, sequence length, which configs to skip embedding for) go in a
child YAML that sets `_base: default.yaml`:

| Key                | Default                            | Description                              |
| ------------------ | ---------------------------------- | ---------------------------------------- |
| `datasets`         | `[]`                               | HuggingFace dataset IDs to process       |
| `output_dir`       | `./datasets`                       | Root output directory                    |
| `model_id`         | `nvidia/llama-embed-nemotron-8b`   | Embedding model                          |
| `embedding_dim`    | `4096`                             | Model embedding dimension                |
| `text_strategy`    | `auto`                             | Text extraction strategy (see below)     |
| `model_dtype`      | `bfloat16`                         | Model precision: `bfloat16` / `float16` / `float32` |
| `batch_size`       | `1`                                | Inference batch size (must fit GPU VRAM)  |
| `num_gpus`         | `8`                                | Number of GPUs for Ray pipeline          |
| `max_seq_length`   | `8192`                             | Max input tokens                         |
| `reduce_methods`   | `[pca, tsne, umap]`               | Dimensionality reduction algorithms      |
| `reduce_n_components` | `2`                             | Target dimensions (2D)                   |
| `config_overrides` | `{}`                               | Per-HF-config overrides (keyed by config name) |
| `dataset_overrides`| `{}`                               | Per-dataset overrides (keyed by HF repo id) |

### Text Strategies

The `text_strategy` field controls how raw HuggingFace records are converted
into a single `text` column for embedding:

| Strategy           | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `auto`             | Legacy auto-detection: pick first string column > 20 chars   |
| `smart`            | Column-aware dispatch: infers strategy from available fields  |
| `messages_list`    | Flatten `messages` list of `{role, content}` dicts           |
| `messages_concat`  | Flatten `input` messages + append `output` string            |
| `rl_blend`         | Extract from `responses_create_params.input` + `ground_truth`|
| `agentic`          | Serialize `tools` definitions + flatten `messages`           |
| `math_proof`       | Concatenate `problem` + Lean 4 `formal_statement`           |
| `math_v2`          | Flatten `messages`, fallback to `problem` field              |
| `raw_text`         | Use the `text` column directly                               |

## License

See [LICENSE](LICENSE).
