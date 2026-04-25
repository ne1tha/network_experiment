from pathlib import Path
import subprocess
import sys


def test_generate_synthetic_dataset(tmp_path: Path) -> None:
    script = Path("/mnt/nvme/jiwang/route2_swinjscc_gan/scripts/generate_synthetic_dataset.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(tmp_path / "synthetic"),
            "--train-count",
            "3",
            "--test-count",
            "1",
            "--size",
            "128",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "created synthetic dataset" in result.stdout
    assert len(list((tmp_path / "synthetic" / "train").glob("*.png"))) == 3
    assert len(list((tmp_path / "synthetic" / "test").glob("*.png"))) == 1
