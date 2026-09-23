#!/usr/bin/env bash
# Run a Snakemake target inside the svml environment (WSL2).
#
#   bash scripts/run_pipeline.sh              # full run (rule `all`)
#   bash scripts/run_pipeline.sh milestone1   # FASTQ -> candidate VCFs
#   CORES=6 bash scripts/run_pipeline.sh milestone2
#
# All tools live in the single `svml` environment built by scripts/setup_wsl.sh, so
# `--use-conda` is not needed. The log is kept under ~/svml-work/logs/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
MM="$HOME/.local/bin/micromamba"
TARGET="${1:-all}"
CORES="${CORES:-10}"

LOGDIR="$HOME/svml-work/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/snakemake_${TARGET}_$(date +%Y%m%d_%H%M%S).log"

# No `conda:` directive appears in any rule (all tools live in the one `svml` env), so
# Snakemake never needs to shell out to `conda info --json`. An earlier attempt to shim
# that call (for the case where a directive was present) was itself buggy -- it supplied
# only the "conda_prefix" key and Snakemake's conda.py also reads "platform", which
# crashed with a second KeyError. Removing the directives outright is more robust than
# chasing every key that internal Snakemake code happens to read.

cd "$REPO"
echo "[run] target=$TARGET cores=$CORES log=$LOG"
# No --rerun-incomplete: a job killed mid-write leaves large, already-verified downloads
# marked incomplete, and rerunning would re-fetch gigabytes. Verify such a file (e.g.
# `samtools quickcheck`) and clear its marker with `snakemake --cleanup-metadata FILE`.
"$MM" run -n svml snakemake "$TARGET" \
  --cores "$CORES" \
  --keep-going \
  --printshellcmds \
  2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
