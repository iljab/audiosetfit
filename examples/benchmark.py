"""Reproducible few-shot benchmark harness for audiosetfit.

Runs one of the example training scripts across a grid of backbones x seeds, parses the
reported eval metrics, and prints a mean +/- std table. Because it simply drives the existing
``examples/train_*.py`` scripts as subprocesses, each dataset keeps its own split logic
(ESC-50/UrbanSound8K folds, CREMA-D speaker-disjoint, MSWC predefined splits).

It can also sweep a grid of training conditions in one invocation via ``--losses`` and
``--batch-sizes``; results are grouped per (backbone, loss, batch-size).

Examples:
    # CLAP vs wav2vec2 on CREMA-D over 3 seeds
    python examples/benchmark.py --dataset cremad \
        --backbones laion/clap-htsat-unfused facebook/wav2vec2-base \
        --seeds 41 42 43

    # Keyword spotting, speech encoders, write a CSV
    python examples/benchmark.py --dataset mswc \
        --backbones facebook/wav2vec2-base microsoft/wavlm-base \
        --seeds 42 43 --csv results.csv

    # Loss/batch-size grid: frozen vs cosine vs supcon (large batch) on MSWC
    python examples/benchmark.py --dataset mswc --backbones microsoft/wavlm-base \
        --seeds 41 42 43 --losses frozen cosine supcon --batch-sizes 8 32

    # Frozen-backbone baseline (pass-through flag after `--`)
    python examples/benchmark.py --dataset esc50 --seeds 41 42 43 -- --no-embedding-finetuning
"""

import argparse
import ast
import csv
import os
import re
import statistics
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

DATASET_SCRIPTS = {
    "esc50": "train_esc50.py",
    "urbansound8k": "train_urbansound8k.py",
    "cremad": "train_cremad.py",
    "mswc": "train_mswc_keywords.py",
}

DEFAULT_BACKBONES = ["laion/clap-htsat-unfused", "facebook/wav2vec2-base"]

# Matches the line printed by the example scripts, e.g. "Eval metrics: {'test_accuracy': 0.44}".
METRICS_RE = re.compile(r"Eval metrics:\s*(\{.*\})")


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-backbone / multi-seed benchmark for audiosetfit examples",
        epilog="Any arguments after a literal `--` are forwarded verbatim to the training script.",
    )
    p.add_argument("--dataset", choices=sorted(DATASET_SCRIPTS), required=True, help="Which example to run")
    p.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES, help="HF backbone ids to compare")
    p.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds to average over")
    p.add_argument(
        "--losses",
        nargs="+",
        default=None,
        help="Grid over phase-1 losses, e.g. 'cosine supcon frozen' "
        "('frozen' maps to --no-embedding-finetuning).",
    )
    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Grid over --batch-size values (supcon benefits from larger batches).",
    )
    p.add_argument("--device", default=None, help="cpu / cuda / mps (auto if omitted)")
    p.add_argument("--csv", default=None, help="Optional path to write per-run results as CSV")
    p.add_argument("--quiet", action="store_true", help="Suppress each run's stdout (still parses metrics)")
    p.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the training script (place after `--`).",
    )
    return p.parse_args()


def _clean_passthrough(passthrough: List[str]) -> List[str]:
    # argparse.REMAINDER keeps a leading "--" if present; drop it.
    return passthrough[1:] if passthrough and passthrough[0] == "--" else passthrough


def run_one(script: str, backbone: str, seed: int, device: Optional[str], extra: List[str], quiet: bool, tag: str = ""):
    """Run a single training script and return its parsed metrics dict (or None on failure)."""
    cmd = [sys.executable, script, "--backbone", backbone, "--seed", str(seed)]
    if device:
        cmd += ["--device", device]
    cmd += extra

    print(f"\n>>> {backbone} | seed={seed}{(' | ' + tag) if tag else ''}")
    print("    " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not quiet and proc.stdout:
        print(proc.stdout.rstrip())

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        print(f"    !! run failed (exit {proc.returncode}):")
        for line in tail:
            print("       " + line)
        return None

    matches = METRICS_RE.findall(proc.stdout)
    if not matches:
        print("    !! could not find 'Eval metrics:' in output")
        return None
    try:
        return ast.literal_eval(matches[-1])
    except (ValueError, SyntaxError) as e:
        print(f"    !! failed to parse metrics: {e}")
        return None


def aggregate(values: List[float]) -> Tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


# A condition is one cell of the loss x batch-size grid: a label plus the extra CLI args it adds.
FROZEN_TOKENS = {"frozen", "none", "no-ft"}


def expand_conditions(args, base_extra: List[str]) -> List[Dict]:
    """Cartesian product of --losses x --batch-sizes, each as (loss, bs, extra-args)."""
    losses = args.losses if args.losses else [None]
    batch_sizes = args.batch_sizes if args.batch_sizes else [None]
    conditions: List[Dict] = []
    for loss in losses:
        for bs in batch_sizes:
            extra = list(base_extra)
            if loss is None:
                loss_label = "default"
            elif loss.lower() in FROZEN_TOKENS:
                extra += ["--no-embedding-finetuning"]
                loss_label = "frozen"
            else:
                extra += ["--loss", loss]
                loss_label = loss
            if bs is not None:
                extra += ["--batch-size", str(bs)]
            conditions.append({"loss": loss_label, "bs": ("-" if bs is None else str(bs)), "extra": extra})
    return conditions


def main():
    args = parse_args()
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATASET_SCRIPTS[args.dataset])
    base_extra = _clean_passthrough(args.passthrough)
    conditions = expand_conditions(args, base_extra)

    rows: List[Dict] = []  # one per (backbone, condition, seed)
    for backbone in args.backbones:
        for cond in conditions:
            for seed in args.seeds:
                tag = f"loss={cond['loss']} bs={cond['bs']}"
                metrics = run_one(script, backbone, seed, args.device, cond["extra"], args.quiet, tag=tag)
                row = {"backbone": backbone, "loss": cond["loss"], "bs": cond["bs"], "seed": seed}
                if metrics:
                    row["accuracy"] = metrics.get("test_accuracy")
                    row["f1_macro"] = metrics.get("test_f1_macro")
                rows.append(row)

    # ---- summary table (mean +/- std over seeds, per backbone x condition) ----
    print("\n" + "=" * 86)
    grid = f"losses={args.losses or '(script default)'}  batch_sizes={args.batch_sizes or '(script default)'}"
    print(f"Benchmark: {args.dataset}  |  seeds={args.seeds}  |  {grid}")
    if base_extra:
        print(f"           extra args: {base_extra}")
    print("=" * 86)
    header = f"{'backbone':38s} {'loss':>8s} {'bs':>4s} {'accuracy':>16s} {'f1_macro':>16s} {'n':>3s}"
    print(header)
    print("-" * len(header))
    for backbone in args.backbones:
        for cond in conditions:
            sel = [
                r for r in rows
                if r["backbone"] == backbone and r["loss"] == cond["loss"] and r["bs"] == cond["bs"]
            ]
            accs = [r["accuracy"] for r in sel if r.get("accuracy") is not None]
            f1s = [r["f1_macro"] for r in sel if r.get("f1_macro") is not None]
            name = backbone if len(backbone) <= 38 else backbone[:37] + "\u2026"
            if not accs:
                print(f"{name:38s} {cond['loss']:>8s} {cond['bs']:>4s} {'FAILED':>16s} {'FAILED':>16s} {0:>3d}")
                continue
            a_m, a_s = aggregate(accs)
            f_m, f_s = aggregate(f1s) if f1s else (float("nan"), 0.0)
            print(
                f"{name:38s} {cond['loss']:>8s} {cond['bs']:>4s} "
                f"{a_m:7.4f} +/-{a_s:5.4f} {f_m:7.4f} +/-{f_s:5.4f} {len(accs):>3d}"
            )

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["backbone", "loss", "bs", "seed", "accuracy", "f1_macro"])
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in writer.fieldnames})
        print(f"\nWrote per-run results to {args.csv}")


if __name__ == "__main__":
    main()
