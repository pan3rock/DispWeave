"""Assessment cannot use validation observations to choose or fit a training solution."""
from pathlib import Path
import subprocess
import sys
import tempfile

exe, source = Path(sys.argv[1]), Path(sys.argv[2])
text = (source / "examples/inversion/synthetic.toml").read_text()
text += "\n[assessment]\nscale_log_vs = [0.1, 0.1, 0.1]\nridge = 1.0\n"


def run(*args, code=0):
    p = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    assert p.returncode == code, (p.stdout, p.stderr)


def meta(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    config = root / "config.toml"
    config.write_text(text)
    base = root / "base"
    run("synthetic", "--config", config, "--output", base)
    validation = root / "validation.txt"
    validation.write_text((base / "observations.txt").read_text())
    a = root / "a"
    run("invert", "--config", config, "--output", a, "--data", base / "observations.txt", "--validation-data", validation)
    score_a = meta(a / "assessment.txt")
    assert score_a["regular"] == "1" and score_a["validation_status"] == "valid"
    data = []
    for line in validation.read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split()
        fields[1] = str(float(fields[1]) + .1)
        data.append(" ".join(fields))
    validation.write_text("\n".join(data))
    b = root / "b"
    run("invert", "--config", config, "--output", b, "--data", base / "observations.txt", "--validation-data", validation)
    score_b = meta(b / "assessment.txt")
    assert (a / "runs.txt").read_bytes() == (b / "runs.txt").read_bytes()
    assert score_a["aic"] == score_b["aic"] and score_a["best_training_run"] == score_b["best_training_run"]
    assert float(score_a["validation_log_likelihood"]) > float(score_b["validation_log_likelihood"]) + 100
    validation.write_text("0.01 2.9 1 .005\n")
    missing = root / "missing"
    run("invert", "--config", config, "--output", missing, "--data", base / "observations.txt", "--validation-data", validation)
    assert meta(missing / "assessment.txt")["validation_status"] == "forward_failure"
    config.write_text(text.replace("max_iterations = 150", "max_iterations = 1"))
    failed = root / "failed"
    run("synthetic", "--config", config, "--output", failed, code=4)
    assert meta(failed / "assessment.txt")["fitted"] == "0"
print("selection workflow passed")
