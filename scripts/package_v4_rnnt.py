#!/usr/bin/env python3
"""Build, install, or explicitly download the pinned GigaAM v4 RNNT FP32 CPU bundle."""
import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zlib

CATALOGUE = Path(__file__).resolve().parents[1] / "models/gigaam-v4-rnnt-fp32-cpu-v1.json"
CHUNK = 1024 * 1024
MAX_ARCHIVE_BYTES = 2_000_000_000
DOWNLOAD_DEADLINE_SECONDS = 900


@dataclass(frozen=True)
class Payload:
    size: int
    sha256: str
    data: bytes | None = None


def payload(data):
    return Payload(len(data), hashlib.sha256(data).hexdigest(), data)


def manifest(profile, provenance):
    # These fields are strings, ints or flat string/int arrays, all valid with
    # the same spelling in UTF-8 JSON and TOML. No JSON object is emitted.
    fields = ["schema_version", "family", "head", "styles", "mode", "precision", "provider", "abi"]
    if profile["mode"] == "streaming":
        fields += ["future_context_ms", "left_context_frames", "state_schema"]
    fields += ["external_data"]
    assignments = lambda mapping, keys: [f"{k} = {json.dumps(mapping[k], ensure_ascii=False)}" for k in keys]
    lines = assignments(profile, fields) + ["", "[provenance]"] + assignments(provenance, provenance)
    for role, art in profile["artifacts"].items():
        lines += ["", f"[artifacts.{role}]"] + assignments(art, ("file", "bytes", "sha256"))
    return ("\n".join(lines) + "\n").encode("utf8")


def qualified_plan():
    raw = CATALOGUE.read_bytes()
    catalogue = json.loads(raw)
    notice = catalogue["license"]
    license_bytes = (CATALOGUE.parent / notice["notice"]).read_bytes()
    license_entry = payload(license_bytes)
    if (license_entry.size, license_entry.sha256) != (notice["bytes"], notice["sha256"]):
        raise ValueError("Pinned GigaAM MIT notice mismatch")
    plan = {"LICENSE-GigaAM": license_entry, "PROVENANCE.json": payload(raw)}
    for profile, config in catalogue["profiles"].items():
        plan[f"{profile}/manifest.toml"] = payload(manifest(config, catalogue["provenance"]))
        for artifact in config["artifacts"].values():
            name = artifact["file"]
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name) or f"{profile}/{name}" in plan:
                raise ValueError("Invalid or duplicate pinned model basename")
            plan[f"{profile}/{name}"] = Payload(artifact["bytes"], expected_digest(artifact["sha256"]))
    plan["SHA256SUMS"] = payload("".join(f"{p.sha256}  {name}\n" for name, p in sorted(plan.items())).encode("ascii"))
    return catalogue, plan


def expected_digest(value):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("Expected SHA-256 must contain exactly 64 hexadecimal characters")
    return value.lower()


@contextmanager
def regular_reader(path):
    # Reject symlinks and special files, including non-blocking opens of a FIFO.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Only regular input files are accepted")
        yield stream


def header(name, size):
    item = tarfile.TarInfo(name)
    item.size, item.mode, item.uid, item.gid, item.mtime = size, 0o644, 0, 0, 0
    item.uname = item.gname = ""
    return item.tobuf(tarfile.USTAR_FORMAT)


def padding(size):
    return (-size) % tarfile.BLOCKSIZE


def end_padding(total):
    end = 2 * tarfile.BLOCKSIZE
    return end + (-(total + end)) % tarfile.RECORDSIZE


def transfer(source, output, expected, name):
    digest = hashlib.sha256()
    remaining = expected.size
    while remaining:
        part = source.read(min(CHUNK, remaining))
        if not part:
            raise ValueError(f"Truncated payload: {name}")
        output.write(part)
        digest.update(part)
        remaining -= len(part)
    if digest.hexdigest() != expected.sha256:
        raise ValueError(f"Payload digest mismatch: {name}")


def write_bundle(output, plan, roots=None):
    total = 0
    with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0, compresslevel=1) as tar:
        for name, entry in sorted(plan.items()):
            record = header(name, entry.size)
            tar.write(record)
            if entry.data is not None:
                with io.BytesIO(entry.data) as source:
                    transfer(source, tar, entry, name)
            else:
                profile, basename = name.split("/", 1)
                with regular_reader(roots[profile] / basename) as source:
                    if os.fstat(source.fileno()).st_size != entry.size:
                        raise ValueError(f"Wrong size for pinned export: {name}")
                    transfer(source, tar, entry, name)
                    if source.read(1):
                        raise ValueError(f"Unexpected extra export bytes: {name}")
            pad = padding(entry.size)
            tar.write(bytes(pad))
            total += len(record) + entry.size + pad
        tar.write(bytes(end_padding(total)))


def require_bytes(source, expected, label):
    got = source.read(len(expected))
    if len(got) != len(expected):
        raise ValueError(f"Truncated archive {label}")
    if got != expected:
        raise ValueError(f"Unexpected archive {label}")


class SingleGzip:
    """Read one bounded gzip member, not gzip's permissive concatenation."""

    def __init__(self, source):
        self.source = source
        self.decoder = zlib.decompressobj(wbits=31)
        self.pending = b""
        self.read_size = 0
        self.compressed_digest = hashlib.sha256()

    def read(self, size):
        out = bytearray()
        while len(out) < size:
            if self.decoder.eof:
                if self.decoder.unused_data or self.source.read(1):
                    raise ValueError("Unexpected trailing compressed bytes")
                break
            if not self.pending:
                self.pending = self.source.read(CHUNK)
                self.read_size += len(self.pending)
                self.compressed_digest.update(self.pending)
                if self.read_size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive exceeds compressed size limit")
                if not self.pending:
                    raise ValueError("Truncated gzip archive")
            try:
                out.extend(self.decoder.decompress(self.pending, size - len(out)))
            except zlib.error as error:
                raise ValueError("Invalid gzip archive") from error
            self.pending = self.decoder.unconsumed_tail
        return bytes(out)


def unpack_exact(source, staging, plan):
    # The canonical member list and USTAR headers are known in advance.
    # Do not ask a generic tar extractor to parse links, sparse or PAX records.
    total = 0
    tar = SingleGzip(source)
    for name, entry in sorted(plan.items()):
        record = header(name, entry.size)
        require_bytes(tar, record, f"header for {name}")
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as out:
            transfer(tar, out, entry, name)
            out.flush()
            os.fsync(out.fileno())
        path.chmod(0o644)
        pad = padding(entry.size)
        require_bytes(tar, bytes(pad), "member padding")
        total += len(record) + entry.size + pad
    require_bytes(tar, bytes(end_padding(total)), "end padding")
    if tar.read(1):
        raise ValueError("Unexpected trailing archive payload")
    return tar.compressed_digest.hexdigest()


def archive_digest(stream):
    h = hashlib.sha256()
    total = 0
    while part := stream.read(CHUNK):
        total += len(part)
        if total > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive exceeds compressed size limit")
        h.update(part)
    stream.seek(0)
    return h.hexdigest()


def publish_noreplace(staging, destination):
    # Linux/glibc with filesystem RENAME_NOREPLACE support is part of the
    # supported installer contract; never fall back to a clobbering rename.
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise OSError("Atomic no-replace install needs Linux/glibc renameat2") from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd, noreplace = -100, 1
    if renameat2(at_fdcwd, os.fsencode(staging), at_fdcwd, os.fsencode(destination), noreplace):
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e), destination)


@contextmanager
def prepared_destination(destination):
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Outer private directory is never renamed; cleanup cannot touch a
    # concurrently recreated target or a caller-owned previous installation.
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as work:
        staging = Path(work) / "complete"
        staging.mkdir(mode=0o700)
        yield staging
        staging.chmod(0o755)
        publish_noreplace(staging, destination)


def verify_signature(source, signature, public_key):
    fd = source.fileno()
    # Use the same open regular file as hashing and extraction. Optional
    # signature uses the existing minisign release format, not a new format.
    subprocess.run(["minisign", "-H", "-Vm", f"/proc/self/fd/{fd}", "-x", str(signature.absolute()),
                    "-p", str(public_key.absolute()), "-q"], pass_fds=(fd,), check=True, timeout=90)


def install_archive(archive, expected_sha256, destination, plan, *, signature=None, public_key=None):
    if (signature is None) != (public_key is None):
        raise ValueError("Both signature and trusted public key are required for minisign verification")
    with prepared_destination(destination) as staging, regular_reader(archive) as source:
        if os.fstat(source.fileno()).st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive exceeds compressed size limit")
        expected_sha256 = expected_digest(expected_sha256)
        if archive_digest(source) != expected_sha256:
            raise ValueError("Archive SHA-256 mismatch")
        if signature is not None:
            verify_signature(source, signature, public_key)
        if unpack_exact(source, staging, plan) != expected_sha256:
            raise ValueError("Archive SHA-256 changed during extraction")
    return Path(destination)


def build_bundle(catalogue, plan, file_dir, stream_dir, output_dir):
    filename = catalogue["archive"]
    with prepared_destination(output_dir) as staging:
        archive = staging / filename
        with archive.open("xb") as stream:
            write_bundle(stream, plan, {"file": file_dir, "native": stream_dir})
            stream.flush()
            os.fsync(stream.fileno())
        with regular_reader(archive) as stream:
            sha256 = archive_digest(stream)
        with (staging / "SHA256SUMS.txt").open("x", encoding="ascii") as sidecar:
            sidecar.write(f"{sha256}  {filename}\n")
            sidecar.flush()
            os.fsync(sidecar.fileno())
    return Path(output_dir) / filename


def require_https(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise ValueError("Only explicit HTTPS URLs without user info or fragment are accepted")
    return url


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_archive(url, output):
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_SECONDS
    opener = urllib.request.build_opener(HTTPSRedirects())
    with opener.open(require_https(url), timeout=30) as response, Path(output).open("xb") as stream:
        size = 0
        while part := response.read1(CHUNK):
            size += len(part)
            if size > MAX_ARCHIVE_BYTES:
                raise ValueError("Download exceeds compressed size limit")
            if time.monotonic() > deadline:
                raise TimeoutError("Download deadline exceeded")
            stream.write(part)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Prohibit HTTPS download; build and install are always local")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Package exact previously qualified local export bytes")
    build.add_argument("--file-dir", type=Path, required=True)
    build.add_argument("--stream-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    install = commands.add_parser("install", help="Side-load a local archive with full verification")
    install.add_argument("--archive", type=Path, required=True)
    download = commands.add_parser("download", help="Fetch a caller-selected HTTPS URL and install the verified bundle")
    download.add_argument("--url", required=True)
    for command in (install, download):
        command.add_argument("--sha256", required=True, type=expected_digest)
        command.add_argument("--destination", type=Path, required=True)
        command.add_argument("--signature", type=Path, help="Optional local minisign signature of the archive")
        command.add_argument("--public-key", type=Path, help="Trusted minisign public key; required with --signature")
    args = parser.parse_args(argv)
    if args.command == "download" and (args.offline or os.environ.get("GIGASTT_OFFLINE", "").casefold() in ("1", "true", "yes", "on")):
        parser.error("Offline mode forbids download; use install --archive instead")
    if args.command != "build" and (args.signature is None) != (args.public_key is None):
        parser.error("--signature and --public-key must be specified together")
    try:
        cat, plan = qualified_plan()
        if args.command == "build":
            result = build_bundle(cat, plan, args.file_dir, args.stream_dir, args.output_dir)
        else:
            verification = {"signature": args.signature, "public_key": args.public_key}
            if args.command == "install":
                result = install_archive(args.archive, args.sha256, args.destination, plan, **verification)
            else:
                if os.path.lexists(args.destination):
                    raise FileExistsError(f"Destination already exists: {args.destination}")
                with tempfile.TemporaryDirectory(prefix="gigastt-v4-download-") as work:
                    archive = Path(work) / cat["archive"]
                    download_archive(args.url, archive)
                    result = install_archive(archive, args.sha256, args.destination, plan, **verification)
        print(result)
    except (OSError, ValueError, EOFError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
