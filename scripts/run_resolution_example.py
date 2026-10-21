#!/usr/bin/env python3
"""Compare reference-model local information, modes, noise, and ridge strength."""
import argparse
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build/dispweave"))
    parser.add_argument("--output", type=Path, default=Path("results/m4a"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    template = (source / "examples/resolution/reference.toml").read_text()
    frequencies = "0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0"
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for mode in ("fundamental", "multimode"):
        text = template
        if mode == "fundamental":
            text = text.replace(f"[{frequencies}, {frequencies}]", f"[{frequencies}]")
            text = text.replace("[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]", "[0, 0, 0, 0, 0, 0, 0]")
        for ridge in (1., 10.):
            dimensions = []
            for sigma in (.005, .02, .1):
                name = f"{mode}_sigma{sigma}_ridge{ridge}"
                config = args.output / f"{name}.toml"
                config.write_text(text.replace("sigma_km_s = 0.005", f"sigma_km_s = {sigma}").replace("ridge = 1.0", f"ridge = {ridge}"))
                folder = args.output / name
                subprocess.run([str(args.executable.resolve()), "resolution", "--config", str(config), "--output", str(folder)], check=True)
                meta = dict(line.split("=", 1) for line in (folder / "metadata.txt").read_text().splitlines())
                dimension = float(meta["effective_dimension"])
                dimensions.append(dimension)
                spectrum = np.loadtxt(folder / "spectrum.txt", ndmin=2)
                layers = np.loadtxt(folder / "layers.txt", ndmin=2)
                rows.append([mode, sigma, ridge, int(meta["numerical_rank"]), dimension, *spectrum[:, 1], *layers[:, 4]])
                if ridge == 1. and sigma == .005:
                    axes[0].semilogy(spectrum[:, 0] + 1, spectrum[:, 1], "o-", label=mode)
                    axes[2].plot(layers[:, 4], layers[:, 0], "o-", label=mode)
            axes[1].semilogx([.005, .02, .1], dimensions, "o-" if ridge == 1 else "o--", label=f"{mode}, ridge={ridge:g}")
    axes[0].axhline(1, color="gray", linestyle=":", label="ridge=1")
    axes[0].set(xlabel="Singular component", ylabel="Singular value", title="Scaled whitened spectrum (sigma=.005)", xticks=[1, 2, 3])
    axes[1].set(xlabel="Observation sigma (km/s)", ylabel="Sum of ridge filter factors", title="Effective dimension depends on ridge", ylim=(0, 3.1))
    axes[2].set(xlabel="Diagonal of ridge resolution matrix", ylabel="Existing model layer", title="Local diagnostic, not depth resolution", yticks=[0, 1, 2], yticklabels=["0–2 km", "2–4 km", ">4 km (halfspace)"], xlim=(0, 1.05))
    axes[2].invert_yaxis()
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.savefig(args.output / "comparison.png", dpi=160)
    plt.close(fig)
    with (args.output / "comparison.txt").open("w") as f:
        f.write("# mode sigma ridge numerical_rank effective_dimension s0 s1 s2 R00 R11 R22\n")
        for row in rows:
            f.write(" ".join(map(str, row)) + "\n")
    print(args.output / "comparison.png")


if __name__ == "__main__":
    main()
