"""End-to-end CLI contract checks (stdlib only)."""
from pathlib import Path
import subprocess
import sys
import tempfile

exe, source = Path(sys.argv[1]), Path(sys.argv[2])
template = (source / "examples/inversion/synthetic.toml").read_text()


def run(*args, code=0):
    p = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    assert p.returncode == code, (p.returncode, p.stdout, p.stderr, args)
    return p


def table(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def metadata(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


with tempfile.TemporaryDirectory(prefix="dispweave-workflow-") as tmp:
    work = Path(tmp)
    config = work / "case.toml"
    config.write_text(template)
    result = work / "noiseless"
    run("synthetic", "--config", config, "--output", result)
    runs = table(result / "runs.txt")
    assert len(runs) == 3 and all(r[1] == "1" and r[-2] == "converged" for r in runs)
    assert max(float(r[2]) for r in runs) < 1e-12
    best = metadata(result / "metadata.txt")["best_run"]
    for row in table(result / f"run_{best}_model.txt"):
        assert abs(float(row[3]) - float(row[1])) < 1e-7
    run("synthetic", "--config", config, "--output", result, code=1)
    for name, correlation in [("noise", "0.0"), ("correlated", "0.5")]:
        config.write_text(template.replace("add_noise = false", "add_noise = true").replace("correlation = 0.0", f"correlation = {correlation}"))
        result = work / name
        run("synthetic", "--config", config, "--output", result)
        best = int(metadata(result / "metadata.txt")["best_run"])
        cost = float(table(result / "runs.txt")[best][2])
        assert 0 < cost < 30
        replay = work / f"{name}_replay"
        run("invert", "--config", config, "--output", replay, "--data", result / "observations.txt", "--covariance", result / "covariance.txt")
        assert (result / "runs.txt").read_bytes() == (replay / "runs.txt").read_bytes()
        # Exact saved observations/covariance enable replay independent of RNG libraries.
        assert (result / "covariance.txt").read_bytes() == (replay / "covariance.txt").read_bytes()
    # Regression: formerly clipped coupled steps needed 78--151 evaluations at the bound.
    fundamental = template.replace("0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 0.2", "0.2").replace(
        "0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1", "0, 0, 0, 0, 0, 0, 0")
    for name, correlation in [("bounded_independent", "0.0"), ("bounded_correlated", "0.5")]:
        config.write_text(fundamental.replace("add_noise = false", "add_noise = true").replace("correlation = 0.0", f"correlation = {correlation}"))
        result = work / name
        run("synthetic", "--config", config, "--output", result)
        bounded = table(result / "runs.txt")
        assert all(r[1] == "1" and r[-2] == "converged" for r in bounded)
        assert max(int(r[3]) for r in bounded) < 25
        assert max(float(r[2]) for r in bounded) - min(float(r[2]) for r in bounded) < 1e-7
        for i in range(3):
            model = table(result / f"run_{i}_model.txt")
            assert abs(float(model[-1][3]) - 3.5) < 1e-12
    config.write_text(template.replace("max_iterations = 150", "max_iterations = 1"))
    run("synthetic", "--config", config, "--output", work / "limited", code=4)
    assert all(r[-1] == "max_iterations" for r in table(work / "limited/runs.txt"))
    config.write_text(template.replace("starts_vs = [[1.1, 1.8, 2.8], [0.9, 2.2, 3.2], [1.2, 2.1, 3.1]]", "starts_vs = [[0.1, 2.0, 3.0]]"))
    run("synthetic", "--config", config, "--output", work / "invalid", code=4)
    assert table(work / "invalid/runs.txt")[0][-1] == "invalid_initial_bounds"
    fixed_template = template.replace("[inversion]", "[inversion]\nfixed_vs_layers = [2]").replace(
        "starts_vs = [[1.1, 1.8, 2.8], [0.9, 2.2, 3.2], [1.2, 2.1, 3.1]]",
        "starts_vs = [[1.1, 1.8, 3.0], [0.9, 2.2, 3.0], [1.2, 2.1, 3.0]]")
    config.write_text(fixed_template)
    fixed_result = work / "fixed"
    run("synthetic", "--config", config, "--output", fixed_result)
    assert all(r[1] == "1" for r in table(fixed_result / "runs.txt"))
    assert metadata(fixed_result / "metadata.txt")["free_vs_parameters"] == "2"
    assert metadata(fixed_result / "metadata.txt")["fixed_vs_layers"] == "2"
    for i in range(3):
        model = table(fixed_result / f"run_{i}_model.txt")
        assert abs(float(model[-1][3]) - 3.) < 1e-14
        assert max(abs(float(r[3]) - float(r[1])) for r in model) < 1e-7
    for mask in ["[-1]", "[3]", "[2, 2]", "[0, 1, 2]"]:
        config.write_text(fixed_template.replace("fixed_vs_layers = [2]", f"fixed_vs_layers = {mask}"))
        run("synthetic", "--config", config, "--output", work / "invalid_fixed", code=1)
    # No silent replacement of a supplied fixed-layer initial value.
    config.write_text(fixed_template.replace("[1.1, 1.8, 3.0]", "[1.1, 1.8, 2.9]"))
    run("synthetic", "--config", config, "--output", work / "mismatch_fixed", code=1)
    config.write_text(template)
    bad = work / "bad.txt"
    bad.write_text("0.2 1.5 0 0.01 extra\n")
    run("invert", "--config", config, "--data", bad, "--output", work / "bad", code=1)
    bad.write_text("0.2 1.5 0 0.01\n0.3 1.4 0 0.01\n")
    covariance = work / "badcov.txt"
    covariance.write_text("1 2\n2 1\n")
    run("invert", "--config", config, "--data", bad, "--covariance", covariance, "--output", work / "badcov", code=1)
    # Requested overtone cannot be recovered at a low frequency; no rows can be dropped.
    bad.write_text("0.01 2.9 1 0.01\n")
    run("invert", "--config", config, "--data", bad, "--output", work / "missing", code=4)
    assert all(r[-1] == "invalid_initial_evaluation" and r[-2] == "failed" and r[6] == "1" for r in table(work / "missing/runs.txt"))
print("workflow passed: noiseless/noisy/correlated, exact replay, failures, missing roots")
