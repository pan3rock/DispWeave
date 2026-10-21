#!/usr/bin/env python3
"""Build, test and run paired fundamental/multimode synthetic experiments."""
import argparse
from pathlib import Path
import subprocess
import sys


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/m3"))
    parser.add_argument("--build", type=Path, default=Path("build"))
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    output, build = args.output.resolve(), args.build.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be new or empty")
    run("cmake", "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release")
    run("cmake", "--build", build, "-j", "2")
    run("ctest", "--test-dir", build, "--output-on-failure")
    output.mkdir(parents=True, exist_ok=True)
    template = (source / "examples/inversion/synthetic.toml").read_text()
    for mode in ("fundamental", "multimode"):
        config = template
        if mode == "fundamental":
            config = config.replace("0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 0.2", "0.2")
            config = config.replace("0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1", "0, 0, 0, 0, 0, 0, 0")
        for noise in ("noiseless", "independent", "correlated"):
            name = f"{mode}_{noise}"
            text = config
            if noise != "noiseless":
                text = text.replace("add_noise = false", "add_noise = true")
            if noise == "correlated":
                text = text.replace("correlation = 0.0", "correlation = 0.5")
            path = output / f"{name}.toml"
            path.write_text(text)
            run(build / "dispweave", "synthetic", "--config", path, "--output", output / name)
            run(sys.executable, source / "scripts/plot_inversion.py", output / name)
    # Replay the full covariance as an external observation inversion.
    result = output / "multimode_correlated"
    run(build / "dispweave", "invert", "--config", result / "config.toml", "--data", result / "observations.txt",
        "--covariance", result / "covariance.txt", "--output", output / "replayed")


if __name__ == "__main__":
    main()
