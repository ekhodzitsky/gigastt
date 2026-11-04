#!/usr/bin/env python3
"""Build and securely install the standalone selective-precision small pack.

Uses only the Python standard library. No training, hosting, model downloads
during build, or changes to existing installations.
"""
import argparse
import gzip
import hashlib
import io
from pathlib import Path
import os
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request


ENCODER = 'multilingual_small_selective.int8.onnx'
RUNTIME_FILES = (ENCODER, 'multilingual_vocab.txt', 'manifest.toml')
MANIFEST = (f'architecture = "ml_ctc"\n[files]\nencoder = "{ENCODER}"\n'
            f'encoder_int8 = "{ENCODER}"\nvocab = "multilingual_vocab.txt"\n').encode()
ENCODER_SHA = '18fa5fab6ece0123d7813f7abc8da129530f6d1a7da17c85c4b3948f27b8e3e0'
VOCAB_SHA = '4d130287892e1099fedfb3f93c4b4cf8a263151158801680b28977d1be4133f4'
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
LIMITS = {ENCODER: 512 * 1024 * 1024, 'multilingual_vocab.txt': 1024 * 1024,
          'manifest.toml': 64 * 1024, 'SHA256SUMS': 4096}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def expected_digest(value):
    if not re.fullmatch('[0-9a-fA-F]{64}', value):
        raise ValueError('Expected SHA-256 must contain exactly64 hexadecimal characters')
    return value.lower()


def unpack_verified(archive, expected_sha256, staging):
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('Archive exceeds size limit')
    if digest(archive) != expected_digest(expected_sha256):
        raise ValueError('Archive SHA-256 mismatch')
    seen = set()
    with tarfile.open(archive, mode='r|gz') as bundle:
        for member in bundle:
            if member.name not in LIMITS or member.name in seen:
                raise ValueError('Unexpected or duplicate archive member')
            if not member.isfile() or member.issparse() or not 0 < member.size <= LIMITS[member.name]:
                raise ValueError('Only bounded regular-file members are accepted')
            seen.add(member.name)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError('Missing archive payload')
            with source, (staging / member.name).open('xb') as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if (staging / member.name).stat().st_size != member.size:
                raise ValueError('Truncated archive member')
            (staging / member.name).chmod(0o644)
    if seen != set(LIMITS):
        raise ValueError('Incomplete archive')
    checksums = {}
    for line in (staging / 'SHA256SUMS').read_text(encoding='ascii').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  ([^/\\]+)', line)
        if not match or match[2] not in RUNTIME_FILES or match[2] in checksums:
            raise ValueError('Invalid internal checksum manifest')
        checksums[match[2]] = match[1]
    if set(checksums) != set(RUNTIME_FILES):
        raise ValueError('Incomplete internal checksums')
    for name, expected in checksums.items():
        if digest(staging / name) != expected:
            raise ValueError('Internal SHA-256 mismatch: ' + name)
    if (staging / 'manifest.toml').read_bytes() != MANIFEST:
        raise ValueError('Unexpected model manifest layout')
    (staging / 'SHA256SUMS').unlink()


def install_archive(archive, expected_sha256, destination):
    """Verify into a sibling directory and publish via one directory rename."""
    archive, destination = Path(archive), Path(destination)
    expected_sha256 = expected_digest(expected_sha256)
    if destination.is_symlink() or (destination.exists() and
                                   (not destination.is_dir() or any(destination.iterdir()))):
        raise FileExistsError('Destination must be absent or an empty real directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.install-', dir=destination.parent))
    try:
        unpack_verified(archive, expected_sha256, staging)
        staging.chmod(0o755)
        # On POSIX, rename cannot replace a nonempty directory or a symlink
        # with this directory, including one created while validation ran.
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def build_archive(encoder, vocab, output, encoder_sha256=ENCODER_SHA, vocab_sha256=VOCAB_SHA):
    encoder, vocab, output = Path(encoder), Path(vocab), Path(output)
    for path, expected in [(encoder, encoder_sha256), (vocab, vocab_sha256)]:
        if digest(path) != expected_digest(expected):
            raise ValueError('Pinned source digest mismatch: ' + path.name)
    sidecar = output.with_name(output.name + '.sha256')
    if output.exists() or output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError('Refusing to overwrite an existing archive or checksum')
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {ENCODER: encoder_sha256.lower(), 'multilingual_vocab.txt': vocab_sha256.lower(),
              'manifest.toml': hashlib.sha256(MANIFEST).hexdigest()}
    checksums = ''.join(f'{hashes[name]}  {name}\n' for name in sorted(hashes)).encode()
    payloads = {ENCODER: encoder, 'multilingual_vocab.txt': vocab, 'manifest.toml': MANIFEST, 'SHA256SUMS': checksums}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.' + output.name, dir=output.parent, delete=False) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=6) as compressed:
                with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as bundle:
                    for name, payload in sorted(payloads.items()):
                        member = tarfile.TarInfo(name)
                        member.size = len(payload) if isinstance(payload, bytes) else payload.stat().st_size
                        member.mode = 0o644
                        member.uid = member.gid = member.mtime = 0
                        member.uname = member.gname = ''
                        with (io.BytesIO(payload) if isinstance(payload, bytes) else payload.open('rb')) as source:
                            bundle.addfile(member, source)
            raw.flush()
            os.fsync(raw.fileno())
        checksum = digest(temporary)
        with tempfile.TemporaryDirectory(prefix='.verify-', dir=output.parent) as verify:
            unpack_verified(temporary, checksum, Path(verify))
        # Exclusive atomic publication: no existing file can be replaced.
        os.link(temporary, output)
        with sidecar.open('x') as stream:
            stream.write(f'{checksum}  {output.name}\n')
        return checksum
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def require_https(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('Only explicit HTTPS URLs without embedded credentials are accepted')


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_archive(url, output):
    require_https(url)
    opener = urllib.request.build_opener(HTTPSRedirects())
    with opener.open(url, timeout=60) as response, Path(output).open('xb') as stream:
        size = 0
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise ValueError('Download exceeds archive size limit')
            stream.write(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='Build from the pinned validated local weights')
    build.add_argument('--encoder', type=Path, required=True)
    build.add_argument('--vocab', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    install = commands.add_parser('install', help='Verify and atomically install into a separate directory')
    source = install.add_mutually_exclusive_group(required=True)
    source.add_argument('--archive', type=Path)
    source.add_argument('--url')
    install.add_argument('--sha256', required=True, type=expected_digest)
    install.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        print(build_archive(args.encoder, args.vocab, args.output))
    elif args.archive:
        print(install_archive(args.archive, args.sha256, args.destination))
    else:
        with tempfile.TemporaryDirectory(prefix='gigastt-pack-download-') as work:
            archive = Path(work) / 'pack.tar.gz'
            download_archive(args.url, archive)
            print(install_archive(archive, args.sha256, args.destination))


if __name__ == '__main__':
    main()
