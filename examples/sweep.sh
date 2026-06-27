#!/usr/bin/env bash
# Full SupCon-vs-baseline verification sweep across all example datasets.
#
# For each dataset it uses a domain-appropriate backbone and runs the
# frozen / cosine / supcon x {8, 32} grid over several seeds, writing one CSV per dataset.
# This is the matrix to confirm whether the SupCon win on MSWC generalizes to sound-event
# and emotion tasks (or whether CLAP's near-ceiling leaves no room on ESC-50/UrbanSound8K).
#
# Usage:
#   bash examples/sweep.sh
#   SEEDS="41 42 43" DEVICE=cuda OUTDIR=results bash examples/sweep.sh
#
# Env overrides:
#   SEEDS        space-separated seeds            (default "41 42 43")
#   LOSSES       grid of phase-1 losses           (default "frozen cosine supcon")
#   BATCH_SIZES  grid of --batch-size values      (default "8 32")
#   DEVICE       cpu / cuda / mps                 (default: auto)
#   OUTDIR       directory for per-dataset CSVs   (default "results")
#
# Note: 'frozen' ignores batch size, so frozen x {8,32} produces two identical (cheap) runs.
set -euo pipefail

SEEDS="${SEEDS:-41 42 43}"
LOSSES="${LOSSES:-frozen cosine supcon}"
BATCH_SIZES="${BATCH_SIZES:-8 32}"
OUTDIR="${OUTDIR:-results}"

DEVICE_ARG=()
if [[ -n "${DEVICE:-}" ]]; then DEVICE_ARG=(--device "${DEVICE}"); fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTDIR"

run () {
  local dataset="$1" backbone="$2"
  local tag="${dataset}__$(echo "$backbone" | tr '/:' '__')"
  echo ""
  echo "############################################################"
  echo "# sweep: dataset=$dataset  backbone=$backbone"
  echo "############################################################"
  # shellcheck disable=SC2086  # word-splitting of SEEDS/LOSSES/BATCH_SIZES is intentional
  python "$HERE/benchmark.py" \
    --dataset "$dataset" \
    --backbones "$backbone" \
    --seeds $SEEDS \
    --losses $LOSSES \
    --batch-sizes $BATCH_SIZES \
    --quiet "${DEVICE_ARG[@]}" \
    --csv "$OUTDIR/${tag}.csv"
}

# Domain-appropriate backbone per dataset (edit / add lines to broaden the sweep).
run esc50         laion/clap-htsat-unfused      # sound events  -> CLAP
run urbansound8k  laion/clap-htsat-unfused      # urban sounds  -> CLAP
run cremad        microsoft/wavlm-base          # speech emotion-> WavLM
run mswc          microsoft/wavlm-base          # keyword spot. -> WavLM

echo ""
echo "All sweeps complete. Per-dataset CSVs written to: $OUTDIR/"
