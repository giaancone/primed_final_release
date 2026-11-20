# Environment of WaveFM. Usage: source setup.sh   (login or compute node)
# First time: source setup.sh; uv sync --frozen   (main env, PyTorch, in .venv)
# Compression env (TensorFlow/QKeras, not activated here): cd compression && uv sync --frozen, then
#   uv run --project $WAVEFM_ROOT/compression python ...   (or $WAVEFM_ROOT/compression/.venv/bin/python)

export WAVEFM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# uv cache and uv-managed Pythons on scratch: the home quota is small
export UV_CACHE_DIR=${UV_CACHE_DIR:-${PSCRATCH:-$WAVEFM_ROOT}/.cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-${PSCRATCH:-$WAVEFM_ROOT}/.local/share/uv/python}
export UV_PYTHON_PREFERENCE=${UV_PYTHON_PREFERENCE:-only-managed}
if [ -f "$WAVEFM_ROOT/.venv/bin/activate" ]; then
    source "$WAVEFM_ROOT/.venv/bin/activate"
else
    echo "setup.sh: no environment in $WAVEFM_ROOT/.venv; run: (cd $WAVEFM_ROOT && uv sync --frozen)" >&2
fi

export WAVEFM_OUTPUT=${WAVEFM_OUTPUT:-$WAVEFM_ROOT/outputs}    # one directory per run
export HF_HOME=${HF_HOME:-$WAVEFM_ROOT/.cache/huggingface}
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
# torch.compile in-process: a SIGTERM to the job (Slurm, timeout) cannot kill compile workers and deadlock the run
export TORCHINDUCTOR_COMPILE_THREADS=1
# the TimesFM weights are cached in HF_HOME: no network calls from compute nodes
export HF_HUB_OFFLINE=1
