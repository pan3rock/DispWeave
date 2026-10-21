#!/usr/bin/env python3
"""Plot evaluation counts and convergence in paired fixed-layer LM regressions."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("baseline", type=Path)
parser.add_argument("updated", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

def read(path):
    with (path / "summary.csv").open() as f:
        return list(csv.DictReader(f))

old, new = read(args.baseline), read(args.updated)
cases = list(dict.fromkeys(r["case"] for r in old if r["case"].startswith(("fundamental_", "multimode_"))))
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
x = np.arange(len(cases))
for rows, offset, label in [(old, -.18, "Before"), (new, .18, "After")]:
    for axis, key, title in [(axes[0], "evaluations", "Total forward/Jacobian evaluations"), (axes[1], "converged", "Starts passing unchanged gradient tolerance")]:
        values = [sum(int(r[key]) for r in rows if r["case"] == case) for case in cases]
        bars = axis.bar(x + offset, values, width=.36, label=label)
        axis.bar_label(bars, fontsize=8)
        axis.set_title(title)
        axis.set_xticks(x, [c.replace("fundamental_", "F\n").replace("multimode_", "F+1\n") for c in cases], rotation=25, ha="right")
        axis.legend()
        axis.grid(axis="y", alpha=.2)
axes[1].set_ylim(0, 3.6)
axes[1].set_yticks([0, 1, 2, 3])
fig.savefig(args.output, dpi=160)
plt.close(fig)
print(args.output)
