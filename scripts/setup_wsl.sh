#!/usr/bin/env bash
# Phase 0: build the svml toolchain inside WSL2 (finding M4).
#
# User-space only -- no sudo. Installs micromamba to ~/.local/bin and the environment
# to ~/micromamba/envs/svml. Undo with:  rm -rf ~/micromamba ~/.local/bin/micromamba
#
# Usage (from the repo root, inside WSL):  bash scripts/setup_wsl.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
BIN="$HOME/.local/bin"
mkdir -p "$BIN"

if [ ! -x "$BIN/micromamba" ]; then
  echo "[setup] installing micromamba"
  # Decompress with Python's bz2 module rather than `tar -j`: a minimal Ubuntu image may
  # lack the bzip2 binary, and installing it would need sudo.
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | python3 -c "
import sys, tarfile
with tarfile.open(fileobj=sys.stdin.buffer, mode='r|bz2') as tar:
    for member in tar:
        if member.name == 'bin/micromamba':
            tar.extract(member, '$HOME/.local')
            break
"
  chmod +x "$BIN/micromamba"
fi
MM="$BIN/micromamba"
"$MM" --version

if "$MM" env list | grep -qE '(^|/)svml\s*$|envs/svml$'; then
  echo "[setup] updating existing svml environment"
  "$MM" install -y -n svml -f "$REPO/environment.yml"
else
  echo "[setup] creating svml environment (several minutes)"
  "$MM" create -y -f "$REPO/environment.yml"
fi

echo "[setup] installing the svml package in editable mode"
"$MM" run -n svml python -m pip install --no-deps -e "$REPO"

echo "[setup] Phase 0 exit test"
cd "$REPO"
"$MM" run -n svml svml check-env --config config/config.yaml
