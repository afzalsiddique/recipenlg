#!/bin/bash
# Sourced by every PBS script in this directory. Edit the activation block
# below to point at the env that has the project's requirements.txt installed.
# Project standard: a venv at $PROJECT_ROOT/.venv, or a conda env named
# recipenlg-replicate. The default below tries the venv first, then conda.

set -e

export HF_HOME=${HF_HOME:-/mmfs1/scratch/m.afzalsiddique/hfcache}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

PROJECT_ROOT="/mmfs1/projects/changhui.yan/m.afzalsiddique/codes/recipe-nlg-github-replicate/recipenlg"
VENV_PATH="$PROJECT_ROOT/.venv"
CONDA_ENV_NAME="recipenlg-replicate"

if [ -f "$VENV_PATH/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
elif command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
else
    echo "ERROR: no venv at $VENV_PATH and conda not on PATH." >&2
    exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"

cd "$PROJECT_ROOT"
echo "Project dir: $PWD"
echo "Python: $(which python)"
python --version
