#!/usr/bin/env python3
"""Run the six fixed-layer LM regression cases, retaining every failed start."""
import argparse
import csv
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build/dispweave"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extended", action="store_true", help="Also exercise four six-layer profiles at two noise levels and coverages")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    template = (source / "examples/inversion/synthetic.toml").read_text()
    rows = []
    cases = []
    for mode in ("fundamental", "multimode"):
        config = template
        if mode == "fundamental":
            config = config.replace("0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 0.2", "0.2")
            config = config.replace("0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1", "0, 0, 0, 0, 0, 0, 0")
        for noise in ("noiseless", "independent", "correlated"):
            name = f"{mode}_{noise}"
            text = config.replace("add_noise = false", f"add_noise = {'false' if noise == 'noiseless' else 'true'}")
            if noise == "correlated":
                text = text.replace("correlation = 0.0", "correlation = 0.5")
            cases.append((name, text))
    if args.extended:
        profiles = {
            "gradient": [1., 1.2, 1.45, 1.8, 2.3, 3.],
            "interface": [1., 1., 1., 2.2, 2.2, 3.],
            "low_velocity": [1.1, 1.8, 1.1, 1.8, 2.4, 3.],
            "shallow_detail": [1., 1.8, 1.2, 1.65, 2.2, 3.],
        }
        queries = [(f, 0) for f in [.12,.15,.18,.22,.28,.35,.45,.6,.8,1.,1.3,1.7]]
        queries += [(f, 1) for f in [.3,.35,.45,.6,.8,1.,1.3,1.7]]
        queries += [(f, 2) for f in [.6,.7,.8,1.,1.3,1.7]]
        initial = [1.1 + i * (2.9-1.1)/5 for i in range(6)]
        starts = [[value * factor for value in initial] for factor in [1., .95, 1.05]]
        for name, vs in profiles.items():
            for sigma in [.005, .03]:
                for coverage in ["full", "high_band"]:
                    selected = queries if coverage == "full" else [(f,m) for f,m in queries if f >= .35]
                    text = (f"[model]\ndepth = [0.0,0.5,1.0,2.0,3.0,4.0]\nvs = {json.dumps(vs)}\n"
                            f"vp = {[6.] * 6}\nrho = {[2.5] * 6}\n[queries]\n"
                            f"frequency_hz = {json.dumps([f for f,m in selected])}\nmode = {json.dumps([m for f,m in selected])}\n"
                            f"[inversion]\nlower_vs = {[.6] * 6}\nupper_vs = {[3.8] * 6}\nstarts_vs = {json.dumps(starts)}\n"
                            "max_iterations = 200\ngradient_tolerance = 0.0001\n"
                            f"[synthetic]\nsigma_km_s = {sigma}\nadd_noise = true\nseed = 2026\n")
                    cases.append((f"{name}_{coverage}_sigma{sigma}", text))
    for name, text in cases:
        path = args.output / f"{name}.toml"
        path.write_text(text)
        out = args.output / name
        p = subprocess.run([str(args.executable.resolve()), "synthetic", "--config", str(path), "--output", str(out)], capture_output=True, text=True)
        if p.returncode not in (0, 4):
            raise RuntimeError(p.stderr)
        lines = (out / "runs.txt").read_text().splitlines()
        for line in lines[1:]:
            row = dict(zip(lines[0][2:].split(), line.split()))
            rows.append(dict(case=name, run=row["run"], converged=row["converged"], cost=row["cost"],
                evaluations=row["evaluations"], accepted=row["accepted"], rejected=row["rejected"],
                projected_gradient=row["projected_gradient"], reason=row["reason"],
                termination=row.get("termination", "converged" if row["converged"] == "1" else "unclassified")))
        print(p.stdout.strip())
    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
