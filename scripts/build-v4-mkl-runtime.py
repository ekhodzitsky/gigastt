#!/usr/bin/env python3
"""Build the exact opt-in GigaAM v4 MKL side-load directory from a local wheel.

No downloads, system installation, or modification of the vendor libraries.
The first profile admits only the pinned compiler-produced bridge bytes.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "crates/gigastt-core/src/inference/v4/frontend"
PROFILE = json.loads((SOURCE / "mkl-profile.json").read_text())


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(path, pin):
    if path.stat().st_size != pin["bytes"] or digest(path) != pin["sha256"]:
        raise ValueError(f"Unqualified artifact: {pin['file']} (SHA-256 {digest(path)})")


def build(wheel, runtime, cc):
    verify(wheel, PROFILE["wheel"])
    if runtime.exists() or runtime.is_symlink():
        raise ValueError("Output runtime directory must not exist")
    prefix = "mkl-2024.2.0.data/data/"
    metadata = "mkl-2024.2.0.dist-info/"
    with tempfile.TemporaryDirectory(prefix="v4-mkl-", dir=runtime.parent) as temp:
        directory = Path(temp) / "runtime"
        directory.mkdir()
        with zipfile.ZipFile(wheel) as archive:
            rows = dict((row[0], row[1:]) for row in csv.reader(io.StringIO(archive.read(metadata + "RECORD").decode())))

            def extract(member, destination):
                hash_field, size = rows[member]
                if not hash_field.startswith("sha256="):
                    raise ValueError(f"Missing wheel RECORD digest: {member}")
                digest_field = hash_field.removeprefix("sha256=")
                expected = base64.urlsafe_b64decode(digest_field + "=" * (-len(digest_field) % 4)).hex()
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
                verify(destination, {"file": member, "bytes": int(size), "sha256": expected})

            for pin in [*PROFILE["libraries"], PROFILE["message_catalog"]]:
                destination = directory / pin["file"]
                extract(prefix + "lib/" + pin["file"], destination)
                verify(destination, pin)
            # Retain the wheel's Intel license and its third-party notices.
            license_members = [metadata + "LICENSE.txt"] + sorted(
                name for name in rows if name.startswith(prefix + "share/doc/mkl/licensing/")
            )
            for member in license_members:
                extract(member, directory / "license" / Path(member).name)

        shim = directory / PROFILE["shim"]["file"]
        subprocess.run(
            [cc, "-std=c11", "-O2", "-fPIC", "-shared", "-fvisibility=hidden",
             "-Wall", "-Wextra", "-Werror",
             "-Wl,-z,defs,-z,now,--no-as-needed,--disable-new-dtags,--build-id=none,-soname,libgigastt_v4_mkl.so",
             "-Wl,-rpath,$ORIGIN", "-L" + str(directory), str(SOURCE / "mkl_bridge.c"),
             *["-l:" + pin["file"] for pin in PROFILE["libraries"]],
             "-lm", "-ldl", "-pthread", "-o", str(shim)],
            check=True,
        )
        verify(shim, PROFILE["shim"])
        os.rename(directory, runtime)
    print(f"Qualified runtime created: {runtime}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Exact local Intel MKL 2024.2.0 x86-64 wheel")
    parser.add_argument("runtime", type=Path, help="New runtime directory (parent must exist)")
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"), help="Qualified C compiler executable")
    args = parser.parse_args()
    build(args.wheel.resolve(strict=True), args.runtime.absolute(), args.cc)


if __name__ == "__main__":
    main()
