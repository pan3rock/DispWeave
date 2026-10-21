#!/usr/bin/env python3
"""Plot a completed DispWeave LM run; no inversion or uncertainty inference."""
import argparse
from pathlib import Path
import shlex
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(directory, destination=None):
    directory = Path(directory)
    meta = dict(line.split("=", 1) for line in (directory / "metadata.txt").read_text().splitlines())
    best = int(meta["best_run"])
    if best < 0:
        raise ValueError("No converged run to plot; inspect runs.txt and traces")
    model = np.loadtxt(directory / f"run_{best}_model.txt", ndmin=2)
    data = np.loadtxt(directory / "observations.txt", ndmin=2)
    prediction = []
    for line in (directory / f"run_{best}_prediction.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = shlex.split(line)
        if fields[4] != "valid":
            raise ValueError("Invalid predicted mode")
        prediction.append(float(fields[3]))
    prediction = np.asarray(prediction)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    bottom = max(model[-1, 0] * 1.3, 1.0)
    depth = np.r_[model[:, 0], bottom]
    reference = "Truth" if meta["synthetic"] == "1" else "Reference"
    for col, label, style in [(1, reference, "k--"), (2, "Initial", "C1:"), (3, "LM fit", "C0-")]:
        axes[0].step(np.r_[model[:, col], model[-1, col]], depth, style, where="pre", label=label)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Vs (km/s)", ylabel="Depth (km)", title="Fixed layer boundaries")
    axes[0].legend()
    for mode in np.unique(data[:, 2]).astype(int):
        mask = data[:, 2] == mode
        order = np.argsort(data[mask, 0])
        freq = data[mask, 0][order]
        color = f"C{mode % 10}"
        axes[1].errorbar(freq, data[mask, 1][order], yerr=data[mask, 3][order], fmt="o", color=color, label=f"Mode {mode}")
        axes[1].plot(freq, prediction[mask][order], color=color)
        axes[2].plot(freq, (prediction[mask] - data[mask, 1])[order] / data[mask, 3][order], "o-", color=color)
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Phase velocity (km/s)", title="Data (1 sigma) and prediction")
    axes[1].legend()
    axes[2].axhline(0, color="k", linewidth=.8)
    axes[2].set(xlabel="Frequency (Hz)", ylabel="Residual / marginal sigma", title="Not whitened if errors correlate")
    for axis in axes:
        axis.grid(alpha=.2)
    destination = Path(destination) if destination else directory / "inversion.png"
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    print(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(args.directory, args.output)
