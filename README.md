# hfdata

Best Practice for Huggingface dataset handle with NeMo Curator

## Setup

```bash
conda create -n nemo python=3.12
conda activate nemo

pip install uv

# ─── PyTorch (CUDA 13.0) ────────────────────────────────────────────────────
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# or use the NGC container instead:
# sudo docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.11-py3

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
```
