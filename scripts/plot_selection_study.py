#!/usr/bin/env python3
"""Summarize paired complete candidate sets; keep incomplete comparisons separate."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    study = json.loads((root / "study.json").read_text())
    with (root / "candidates.csv").open() as f:
        candidates = list(csv.DictReader(f))
    with (root / "selections.csv").open() as f:
        selections = list(csv.DictReader(f))
    size = {key: len(value) for key, value in study["candidates"].items()}
    sigmas = sorted(study["sigma_km_s"])
    models = list(study["models"])
    report = {"reasons": dict(Counter(row["reason"] for row in candidates)),
              "forward_failures": sum(int(row["forward_failures"]) for row in candidates),
              "gradient_failures": sum(int(row["gradient_failures"]) for row in candidates),
              "converged_starts": sum(int(row["converged_starts"]) for row in candidates),
              "paired_noise": [], "paired_modes": []}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for model, axis in zip(models, axes.flat):
        subset = [row for row in selections if row["model"] == model]
        for criterion, label in [("bic", "BIC"), ("validation_nll", "Validation NLL")]:
            pairs = {}
            for row in subset:
                key = (row["seed"], row["modes"], row["coverage"])
                pairs.setdefault(key, {})[float(row["sigma"])] = row
            changes = []
            for pair in pairs.values():
                if all(sigma in pair and int(pair[sigma][f"{criterion}_eligible"]) == len(size) for sigma in sigmas):
                    changes.append([size[pair[sigma][f"{criterion}_winner"]] for sigma in sigmas])
            if changes:
                values = np.asarray(changes)
                axis.plot(sigmas, np.mean(values, axis=0), "o-", label=f"{label}: {len(values)} paired sets")
                report["paired_noise"].append(dict(model=model, criterion=criterion, pairs=len(values),
                    simpler=int(np.sum(values[:, -1] < values[:, 0])), unchanged=int(np.sum(values[:, -1] == values[:, 0])),
                    more_complex=int(np.sum(values[:, -1] > values[:, 0]))))
            else:
                axis.plot([], [], label=f"{label}: no complete pairs")
        axis.set(title=model, xlabel="Observation sigma (km/s)", ylabel="Mean selected parameter count", ylim=(2.7, 6.3), yticks=[3, 4, 5, 6])
        axis.set_xscale("log")
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.suptitle("Noise comparison: same scenarios, all candidates eligible at both noise levels")
    fig.savefig(root / "complexity.png", dpi=160)
    plt.close(fig)
    # Mode comparison uses BIC and only pairs with all candidates eligible at m=1 and m=3.
    groups = {}
    for row in selections:
        key = (row["model"], row["seed"], row["coverage"], row["sigma"])
        groups.setdefault(key, {})[int(row["modes"])] = row
    for key, group in groups.items():
        if all(m in group and int(group[m]["bic_eligible"]) == len(size) for m in (1, 3)):
            report["paired_modes"].append(dict(model=key[0], seed=key[1], coverage=key[2], sigma=key[3],
                fundamental=size[group[1]["bic_winner"]], three_modes=size[group[3]["bic_winner"]]))
    common = [r for r in selections if int(r["bic_eligible"]) == len(size) and int(r["validation_nll_eligible"]) == len(size)]
    report["complete_common_scenarios"] = len(common)
    report["bic_validation_disagreements"] = sum(r["bic_winner"] != r["validation_nll_winner"] for r in common)
    report["by_model"] = {}
    for model in models:
        rows = [r for r in candidates if r["model"] == model]
        fitted = [r for r in rows if r["fitted"] == "1"]
        report["by_model"][model] = dict(total=len(rows), fitted=len(fitted), regular=sum(r["regular"] == "1" for r in rows),
            median_profile_rmse=float(np.median([float(r["rmse_vs"]) for r in fitted])))
    (root / "analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "paired_modes"}, indent=2))


if __name__ == "__main__":
    main()
