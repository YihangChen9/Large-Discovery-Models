#!/usr/bin/env python3
"""Plot live/final progress of an engine-native antibody run from events.jsonl.

Reads candidate_evaluated events (metric `absolut_energy`, lower is better) and
writes:

  plots/antibody_progress_<run>.png  -- 2 panels: energy trajectory (per-eval
                              and best-so-far), and a final candidate summary.
  plots/antibody_progress_<run>.csv  -- per-evaluation (iteration, energy,
                              best_so_far, sequence) rows.

Usage:
  python scripts/plot_antibody_progress.py <run_dir> [--out <plots_dir>]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="Engine run directory (contains events.jsonl).")
    p.add_argument("--out", type=Path, default=None, help="Output directory (default: <run_dir>/plots).")
    return p.parse_args(argv)


def load_energy_rows(run_dir: Path) -> list[dict]:
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        raise SystemExit(f"events.jsonl not found: {events_path}")
    rows: list[dict] = []
    with events_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event_type") != "candidate_evaluated":
                continue
            payload = ev.get("payload") or {}
            evaluation = payload.get("evaluation") or {}
            if evaluation.get("status") != "succeeded":
                continue
            metrics = evaluation.get("metrics") or {}
            energy = metrics.get("absolut_energy")
            if energy is None:
                continue
            try:
                energy_f = float(energy)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(energy_f):
                continue
            seq = (evaluation.get("metadata") or {}).get("sequence", "")
            rows.append(
                {
                    "iteration": ev.get("iteration"),
                    "candidate_id": evaluation.get("candidate_id", ""),
                    "sequence": seq,
                    "energy": energy_f,
                }
            )
    rows.sort(key=lambda r: (r["iteration"] is None, r["iteration"] or 0))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_energy_rows(args.run_dir)
    if not rows:
        raise SystemExit("No finite succeeded evaluations found in events.jsonl.")
    out_dir = args.out or (args.run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_dir.name

    energies = [r["energy"] for r in rows]
    best_so_far: list[float] = []
    running_best = float("inf")
    for e in energies:
        running_best = min(running_best, e)
        best_so_far.append(running_best)
    iters = [r["iteration"] if r["iteration"] is not None else i + 1 for i, r in enumerate(rows)]

    csv_path = out_dir / f"antibody_progress_{run_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["iteration", "candidate_id", "sequence", "energy", "best_so_far"])
        writer.writeheader()
        for r, b in zip(rows, best_so_far):
            writer.writerow({**r, "best_so_far": b})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Antibody real run (UCB, 1ADQ_A): {run_name}\n"
        f"successful evals: {len(rows)} | best energy: {min(energies):.2f}",
        fontsize=13,
    )

    ax = axes[0]
    ax.plot(iters, energies, "o-", color="#1f77b4", alpha=0.7, label="per-evaluation energy")
    ax.plot(iters, best_so_far, "-", color="#d62728", linewidth=2, label="best so far")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Absolut energy (lower better)")
    ax.set_title("Binding energy trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    best_idx = int(np.argmin(energies))
    ax.barh([0], [energies[best_idx]], color="#2ca02c", height=0.5)
    ax.set_yticks([])
    ax.set_xlabel("Absolut energy")
    ax.set_title(f"Best candidate (iter {iters[best_idx]}): {rows[best_idx]['sequence']}\nenergy={energies[best_idx]:.2f}")
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    png_path = out_dir / f"antibody_progress_{run_name}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"rows={len(rows)} best_energy={min(energies):.3f} best_iter={iters[best_idx]} best_seq={rows[best_idx]['sequence']}")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
