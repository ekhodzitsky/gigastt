#!/usr/bin/env python3
"""Run the standalone synthetic resampling audit without changing product Cargo files."""
import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/gigastt-small-diagnosis/resampling-probe-repro")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    work = args.cache.resolve()
    (work / "src").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "scripts/resampling_length_probe.rs", work / "src/main.rs")
    shutil.copyfile(root / "crates/gigastt-core/src/inference/audio/resample.rs", work / "src/production.rs")
    (work / "Cargo.toml").write_text('''[package]
name = "resampling-length-probe"
version = "0.0.0"
edition = "2024"
[features]
file-decode = []
[dependencies]
rubato = "=5.0.0"
anyhow = "=1.0.104"
serde_json = "=1.0.151"
''')
    subprocess.run(["cargo", "run", "--quiet", "--offline", "--manifest-path", str(work / "Cargo.toml")], check=True)


if __name__ == "__main__":
    main()
