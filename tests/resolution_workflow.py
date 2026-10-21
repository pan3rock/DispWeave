"""CLI rejects unresolved rows and preserves diagnostic configuration."""
from pathlib import Path
import subprocess
import sys
import tempfile

exe, source = sys.argv[1:]
text = (Path(source) / "examples/resolution/reference.toml").read_text()
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    config = work / "input.toml"
    config.write_text(text)
    output = work / "result"
    def run(out, code=0):
        p = subprocess.run([exe, "resolution", "--config", str(config), "--output", str(out)], capture_output=True, text=True)
        assert p.returncode == code, (p.stdout, p.stderr)
    run(output)
    assert (output / "config.toml").read_text() == text
    meta = dict(line.split("=", 1) for line in (output / "metadata.txt").read_text().splitlines())
    assert meta["numerical_rank"] == "3" and 0 < float(meta["effective_dimension"]) < 3
    run(output, 1)
    # Diagonal full covariance must reproduce scalar sigma; exercise TOML matrix input.
    covariance = [[0.005**2 if i == j else 0.0 for j in range(14)] for i in range(14)]
    full = text.replace("sigma_km_s = 0.005", "covariance = " + str(covariance))
    config.write_text(full)
    run(work / "full")
    assert (output / "spectrum.txt").read_bytes() == (work / "full/spectrum.txt").read_bytes()
    covariance[0][0] = -1.0
    config.write_text(text.replace("sigma_km_s = 0.005", "covariance = " + str(covariance)))
    run(work / "nonpositive", 1)
    config.write_text(text.replace("sigma_km_s = 0.005", "sigma_km_s = -1.0"))
    run(work / "negative", 1)
    config.write_text(text.replace("scale_log_vs = [0.1, 0.1, 0.1]", "scale_log_vs = [0.1]"))
    run(work / "dimension", 1)
    config.write_text(text.replace("0.2, 0.25", "0.01, 0.25"))
    run(work / "missing", 1)
    assert not (work / "missing").exists()
print("resolution workflow passed")
