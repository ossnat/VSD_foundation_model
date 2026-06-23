#!/bin/bash
# Shared Python env activation for Slurm / cluster scripts.
#
# Source after setting REPO (and optionally ENV_KIND, CONDA_SH, CONDA_ENV):
#   ENV_KIND=conda source scripts/activate_cluster_env.sh
#   ENV_KIND=venv  source scripts/activate_cluster_env.sh
#
# ENV_KIND: venv (repo .venv) | conda (named conda env). Default: venv.

: "${REPO:?REPO must be set before sourcing activate_cluster_env.sh}"

ENV_KIND="${ENV_KIND:-venv}"
CONDA_ENV="${CONDA_ENV:-vsd_conda_env}"

_activate_conda() {
  local conda_sh="${CONDA_SH:-}"
  if [[ -z "${conda_sh}" || ! -f "${conda_sh}" ]]; then
    for candidate in \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh" \
      "${HOME}/miniforge3/etc/profile.d/conda.sh" \
      "${HOME}/mambaforge/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh"; do
      if [[ -f "${candidate}" ]]; then
        conda_sh="${candidate}"
        break
      fi
    done
  fi
  if [[ ! -f "${conda_sh}" ]]; then
    echo "ERROR: conda.sh not found. Install conda or set CONDA_SH=/path/to/conda.sh" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${CONDA_ENV}"
}

case "${ENV_KIND}" in
  venv)
    if [[ ! -f "${REPO}/.venv/bin/activate" ]]; then
      echo "ERROR: venv missing at ${REPO}/.venv — create with: python3 -m venv .venv && pip install -r requirements.txt" >&2
      return 1
    fi
    # shellcheck disable=SC1091
    source "${REPO}/.venv/bin/activate"
    ;;
  conda)
    _activate_conda || return 1
    ;;
  *)
    echo "ERROR: ENV_KIND must be 'venv' or 'conda', got: ${ENV_KIND}" >&2
    return 1
    ;;
esac

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
