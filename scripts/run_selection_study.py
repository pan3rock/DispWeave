#!/usr/bin/env python3
"""Fixed candidate comparison; all physics, optimization and scores run in C++."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import time
import numpy as np


def array(values):
    return json.dumps([float(v) for v in values])


def model_config(depth, vs, study):
    n = len(depth)
    return (f"[model]\ndepth = {array(depth)}\nvs = {array(vs)}\n"
            f"vp = {array([study['fixed_vp']] * n)}\nrho = {array([study['fixed_rho']] * n)}\n")


def metadata(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def command(exe, *args, allow_failure=False):
    result = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    allowed = (0, 4) if allow_failure else (0,)
    if result.returncode not in allowed:
        raise RuntimeError(f"{args}: {result.stdout}\n{result.stderr}")
    return result


def profile_error(true_depth, true_vs, fitted, bottom):
    # Exact integral of squared error for piecewise constant profiles on [0,bottom].
    edges = np.unique(np.r_[true_depth, fitted[:, 0], bottom])
    edges = edges[(edges >= 0) & (edges <= bottom)]
    mids = (edges[:-1] + edges[1:]) / 2
    truth = np.asarray(true_vs)[np.searchsorted(true_depth, mids, side="right") - 1]
    estimate = fitted[np.searchsorted(fitted[:, 0], mids, side="right") - 1, 3]
    return np.sqrt(np.sum(np.diff(edges) * (truth - estimate)**2) / bottom)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = Path(__file__).resolve().parents[1]
    parser.add_argument("--study", type=Path, default=source / "examples/selection/study.json")
    parser.add_argument("--executable", type=Path, default=Path("build/dispweave"))
    parser.add_argument("--output", type=Path, default=Path("results/m4b"))
    parser.add_argument("--smoke", action="store_true", help="First model, one seed, full band, two modes, both noise levels")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    study = json.loads(args.study.read_text())
    (args.output / "study.json").write_text(json.dumps(study, indent=2))
    exe = args.executable.resolve()
    version = subprocess.check_output([str(exe), "--version"], text=True).strip()
    begin = time.perf_counter()
    query = [(freq, mode) for mode, freqs in enumerate(study["frequencies"]) for freq in freqs]
    freq, modes = np.asarray(query).T
    all_rows, winners = [], []
    models = list(study["models"].items())[:1] if args.smoke else list(study["models"].items())
    for model_index, (name, truth_vs) in enumerate(models):
        folder = args.output / name
        folder.mkdir()
        truth_config = folder / "truth.toml"
        truth_config.write_text(model_config(study["depth"], truth_vs, study) +
                                f"[queries]\nfrequency_hz = {array(freq)}\nmode = {json.dumps(modes.astype(int).tolist())}\n")
        command(exe, "forward", "--config", truth_config, "--output", folder / "truth.txt")
        parsed = [shlex.split(line) for line in (folder / "truth.txt").read_text().splitlines() if not line.startswith("#")]
        if not all(row[4] == "valid" for row in parsed):
            raise RuntimeError("Truth contains unresolved roots; cannot silently trim study")
        phase = np.asarray([float(row[3]) for row in parsed])
        for seed in study["seeds"][:1] if args.smoke else study["seeds"]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, model_index]))
            # Same master errors are retained across nested mode sets, masks and noise levels.
            train_z, validation_z = rng.standard_normal((2, len(query)))
            np.savetxt(folder / f"errors_{seed}.txt", np.c_[train_z, validation_z], header="train_z independent_validation_z")
            for mode_count in ([2] if args.smoke else [1, 2, 3]):
                for coverage in (["full"] if args.smoke else ["full", "high_band", "gaps"]):
                    mask = modes < mode_count
                    if coverage == "high_band":
                        mask &= freq >= .35
                    elif coverage == "gaps":
                        mask &= np.arange(len(query)) % 3 != 1
                    for sigma in study["sigma_km_s"]:
                        scenario = f"seed{seed}_m{mode_count}_{coverage}_s{sigma}"
                        work = folder / scenario
                        work.mkdir()
                        for label, errors in [("train", train_z), ("validation", validation_z)]:
                            np.savetxt(work / f"{label}.txt", np.c_[freq[mask], (phase + sigma * errors)[mask], modes[mask], np.full(mask.sum(), sigma)],
                                       fmt=["%.17g", "%.17g", "%d", "%.17g"], header="frequency_hz phase_km_s mode sigma_km_s")
                        candidates = []
                        for candidate, depth in study["candidates"].items():
                            n = len(depth)
                            initial = np.linspace(1.1, 2.9, n)
                            text = model_config(depth, initial, study)
                            text += (f"[inversion]\nlower_vs = {array([study['lower_vs']] * n)}\nupper_vs = {array([study['upper_vs']] * n)}\n"
                                     f"starts_vs = {json.dumps([(initial * factor).tolist() for factor in (1., .95, 1.05)])}\n"
                                     f"max_iterations = {study['max_iterations']}\ngradient_tolerance = {study['gradient_tolerance']}\n"
                                     f"[assessment]\nscale_log_vs = {array([study['scale_log_vs']] * n)}\nridge = {float(study['ridge'])}\n")
                            config = work / f"{candidate}.toml"
                            config.write_text(text)
                            result = work / candidate
                            command(exe, "invert", "--config", config, "--data", work / "train.txt", "--validation-data", work / "validation.txt", "--output", result, allow_failure=True)
                            assessment = metadata(result / "assessment.txt")
                            meta = metadata(result / "metadata.txt")
                            fitted = assessment["fitted"] == "1"
                            row = dict(model=name, seed=seed, modes=mode_count, coverage=coverage, sigma=sigma, candidate=candidate, parameters=n,
                                       fitted=int(fitted), regular=int(assessment["regular"]), reason=assessment["reason"],
                                       aic=float(assessment.get("aic", "nan")), bic=float(assessment.get("bic", "nan")),
                                       validation_nll=-float(assessment.get("validation_log_likelihood", "nan")),
                                       rank=float(assessment.get("numerical_rank", "nan")), effective_dimension=float(assessment.get("effective_dimension", "nan")),
                                       rmse_vs=float("nan"), cost=float("nan"), forward_failures=0, gradient_failures=0, converged_starts=0, seconds=float(meta["seconds"]))
                            runs = [line.split() for line in (result / "runs.txt").read_text().splitlines() if not line.startswith("#")]
                            row.update(forward_failures=sum(int(r[6]) for r in runs), gradient_failures=sum(int(r[7]) for r in runs), converged_starts=sum(int(r[1]) for r in runs))
                            if fitted:
                                best = int(meta["best_run"])
                                row["cost"] = float(runs[best][2])
                                fitted_model = np.loadtxt(result / f"run_{best}_model.txt", ndmin=2)
                                row["rmse_vs"] = profile_error(study["depth"], truth_vs, fitted_model, study["error_depth_km"])
                            candidates.append(row)
                            all_rows.append(row)
                        selection = dict(model=name, seed=seed, modes=mode_count, coverage=coverage, sigma=sigma)
                        for criterion in ("aic", "bic", "validation_nll"):
                            eligible = [r for r in candidates if np.isfinite(r[criterion])]
                            selection[f"{criterion}_eligible"] = len(eligible)
                            selection[f"{criterion}_winner"] = min(eligible, key=lambda r: (r[criterion], r["parameters"]))["candidate"] if eligible else "none"
                        winners.append(selection)
                        # Save incrementally so failed scenarios leave inspectable previous results.
                        write_csv(args.output / "candidates.csv", all_rows)
                        write_csv(args.output / "selections.csv", winners)
        print(f"completed {name}: {len(all_rows)} candidate fits", flush=True)
    info = dict(version=version, numpy=np.__version__, rng="NumPy PCG64; saved master z and data are authoritative", smoke=args.smoke,
                candidate_fits=len(all_rows), scenarios=len(winners), seconds=time.perf_counter()-begin,
                fitted=sum(r["fitted"] for r in all_rows), regular=sum(r["regular"] for r in all_rows),
                complete_bic=sum(r["bic_eligible"] == len(study["candidates"]) for r in winners),
                complete_validation=sum(r["validation_nll_eligible"] == len(study["candidates"]) for r in winners))
    (args.output / "summary.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
