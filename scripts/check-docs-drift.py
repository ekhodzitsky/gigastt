#!/usr/bin/env python3
"""Docs drift gate: fail when documentation drifts away from the code.

Six axes, all stdlib-only (no third-party deps, no network):

  1. CLI flags/envs: every clap flag + GIGASTT_* env in crates/gigastt/src/main.rs
     is documented in docs/cli.md, and cli.md names no flag/env that does not
     exist (intentional exceptions live in scripts/check-docs-drift.allowlist).
  2. WS error codes: the enum in docs/asyncapi.yaml == the table in docs/api.md
     == the codes emitted by crates/gigastt/src/server/ws.rs (plus allowlisted
     doc-only entries).
  3. Audio formats: the canonical FORMATS list below == the `// docs-drift: codecs`
     marker block in crates/gigastt-core/src/inference/audio.rs, and every format
     is named in docs/api.md and docs/cli.md. When adding a codec, update all
     three places (marker, FORMATS, docs) in the same commit.
  4. mdBook TOCs: every chapter file is listed in its book's SUMMARY.md, every
     SUMMARY.md entry points at an existing file, and `mdbook build` succeeds
     for both books (skipped with a warning when mdbook is not on PATH).
  5. EN/RU parity: docs/workbook/en/src and docs/workbook/ru/src hold identical
     file names, and paired files have the same heading count (structural
     control only; translation freshness is a review responsibility).
  6. Relative links: every relative markdown link in docs/**, the root README*,
     and packaging/**/README* resolves to an existing file/directory, and
     #anchors resolve to a heading in the target file.

Exit code 0 when everything is in sync, 1 otherwise. Runs in seconds.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent

MAIN_RS = ROOT / "crates/gigastt/src/main.rs"
WS_RS = ROOT / "crates/gigastt/src/server/ws.rs"
AUDIO_RS = ROOT / "crates/gigastt-core/src/inference/audio.rs"
CLI_MD = ROOT / "docs/cli.md"
API_MD = ROOT / "docs/api.md"
ASYNCAPI_YAML = ROOT / "docs/asyncapi.yaml"
ALLOWLIST = ROOT / "scripts/check-docs-drift.allowlist"
WORKBOOK = ROOT / "docs/workbook"

# Canonical audio decode surface. The token list must match the
# `// docs-drift: codecs` marker block in inference/audio.rs exactly; each
# needle regex must appear in the named doc. Update this table, the marker,
# and the docs together whenever a codec or container is added/removed.
FORMATS = {
    # token: (api.md needle, cli.md needle)
    "wav": (r"WAV \(PCM", r"Supports: WAV"),
    "wav-g711": (r"G\.711 A-law / μ-law", r"G\.711 A-law/μ-law"),
    "wav-g722": (r"WAV with G\.722", r"G\.722 payloads"),
    "mp3": (r"MP3", r"MP3"),
    "m4a": (r"M4A", r"M4A"),
    "ogg-vorbis": (r"OGG/Vorbis", r"OGG/Vorbis"),
    "ogg-opus": (r"OGG/Opus", r"OGG/Opus"),
    "flac": (r"FLAC", r"FLAC"),
    "raw-pcmu": (r"\.ulaw", r"\.ulaw"),
    "raw-pcma": (r"\.alaw", r"\.alaw"),
    "raw-g722": (r"\.g722", r"\.g722"),
}

MARKER_BEGIN = "// docs-drift: codecs"
MARKER_END = "// docs-drift: end"


def load_allowlist(path: Path) -> dict[str, dict[str, str]]:
    """Parse the allowlist: `[section]` headers, `value  # reason` lines.

    Returns {section: {value: reason}}. Every entry must carry a justification
    comment — an unexplained exception is a parse error.
    """
    sections: dict[str, dict[str, str]] = {}
    if not path.exists():
        return sections
    current: str | None = None
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"\[([a-z0-9-]+)\]", line)
        if m:
            current = m.group(1)
            sections.setdefault(current, {})
            continue
        if current is None:
            raise SystemExit(f"{path}:{lineno}: entry outside a [section]: {line!r}")
        value, sep, reason = line.partition("#")
        if not sep or not reason.strip():
            raise SystemExit(f"{path}:{lineno}: allowlist entry lacks a justification: {line!r}")
        sections[current][value.strip()] = reason.strip()
    return sections


# ---------------------------------------------------------------------------
# 1. CLI flags / env vars
# ---------------------------------------------------------------------------

ARG_RE = re.compile(r"#\[arg\((.*?)\)\]\s*(?:pub\s+)?(\w+)\s*:", re.S)
LONG_NAME_RE = re.compile(r'long\s*=\s*"([^"]+)"')
ENV_RE = re.compile(r'env\s*=\s*"(GIGASTT_[A-Z0-9_]+)"')
ENV_CALL_RE = re.compile(r'env::(?:var|var_os|set_var)\(\s*"(GIGASTT_[A-Z0-9_]+)"')
CLI_FLAG_TOKEN_RE = re.compile(r"--([a-z0-9][a-z0-9-]*)")
ENV_TOKEN_RE = re.compile(r"\bGIGASTT_[A-Z0-9_]+\b")


def parse_cli_definition() -> tuple[set[str], set[str]]:
    """Extract clap long-flag names and GIGASTT_* env vars from main.rs."""
    src = MAIN_RS.read_text(encoding="utf-8")
    flags: set[str] = set()
    envs: set[str] = set(ENV_CALL_RE.findall(src))
    for attrs, field in ARG_RE.findall(src):
        if "long" not in attrs:
            continue
        m = LONG_NAME_RE.search(attrs)
        flags.add(m.group(1) if m else field.replace("_", "-"))
        envs.update(ENV_RE.findall(attrs))
    return flags, envs


def check_cli(allow: dict[str, dict[str, str]]) -> list[str]:
    flags, envs = parse_cli_definition()
    doc = CLI_MD.read_text(encoding="utf-8")
    doc_flags = set(CLI_FLAG_TOKEN_RE.findall(doc))
    doc_envs = set(ENV_TOKEN_RE.findall(doc))

    ok_flags = set(allow.get("flags-undocumented-ok", {}))
    ok_envs = set(allow.get("envs-undocumented-ok", {}))
    phantom_flags = set(allow.get("doc-flags-phantom-ok", {}))
    phantom_envs = set(allow.get("doc-envs-phantom-ok", {}))

    failures: list[str] = []
    for flag in sorted(flags - doc_flags - ok_flags):
        failures.append(f"cli.md: flag --{flag} (crates/gigastt/src/main.rs) is not documented")
    for env in sorted(envs - doc_envs - ok_envs):
        failures.append(f"cli.md: env var {env} (crates/gigastt/src/main.rs) is not documented")
    for flag in sorted(doc_flags - flags - phantom_flags):
        failures.append(f"cli.md: --{flag} does not match any clap flag in main.rs")
    for env in sorted(doc_envs - envs - phantom_envs):
        failures.append(f"cli.md: {env} is not a GIGASTT_* env var read by main.rs")
    return failures


# ---------------------------------------------------------------------------
# 2. WebSocket error codes
# ---------------------------------------------------------------------------

WS_CODE_RE = re.compile(r'\bcode:\s*"([a-z_]+)"')


def asyncapi_codes() -> set[str]:
    """Pull the error-code enum out of docs/asyncapi.yaml (regex, no yaml dep)."""
    lines = ASYNCAPI_YAML.read_text(encoding="utf-8").splitlines()
    codes: set[str] = set()
    in_enum = False
    for line in lines:
        if re.match(r"^\s+enum:\s*$", line):
            in_enum = True
            continue
        if in_enum:
            m = re.match(r"^\s+- ([a-z_]+)\s*$", line)
            if m:
                codes.add(m.group(1))
                continue
            break
    return codes


def api_md_ws_codes() -> set[str]:
    """Pull the code column out of the '### Error codes' table in api.md."""
    doc = API_MD.read_text(encoding="utf-8")
    m = re.search(r"^### Error codes\s*$", doc, re.M)
    if not m:
        raise SystemExit("docs/api.md: '### Error codes' section not found")
    section = doc[m.end():]
    end = re.search(r"^## ", section, re.M)
    if end:
        section = section[: end.start()]
    return {row.group(1) for row in re.finditer(r"^\|\s*`([a-z_]+)`\s*\|", section, re.M)}


def check_ws_error_codes(allow: dict[str, dict[str, str]]) -> list[str]:
    async_codes = asyncapi_codes()
    api_codes = api_md_ws_codes()
    emitted = set(WS_CODE_RE.findall(WS_RS.read_text(encoding="utf-8")))
    doc_only = set(allow.get("ws-codes-doc-only", {}))
    undoc_ok = set(allow.get("ws-codes-undocumented-ok", {}))

    failures: list[str] = []
    for code in sorted(async_codes - api_codes):
        failures.append(f"asyncapi.yaml: `{code}` is missing from the docs/api.md error-code table")
    for code in sorted(api_codes - async_codes):
        failures.append(f"api.md: `{code}` is missing from the docs/asyncapi.yaml enum")
    for code in sorted(emitted - api_codes - undoc_ok):
        failures.append(f"ws.rs: `{code}` is emitted but not documented in api.md/asyncapi.yaml")
    for code in sorted(api_codes - emitted - doc_only):
        failures.append(f"api.md: `{code}` is documented but never emitted by ws.rs (allowlist if REST-only)")
    return failures


# ---------------------------------------------------------------------------
# 3. Audio formats
# ---------------------------------------------------------------------------


def check_formats() -> list[str]:
    src = AUDIO_RS.read_text(encoding="utf-8")
    begin = src.find(MARKER_BEGIN)
    end = src.find(MARKER_END)
    failures: list[str] = []
    if begin == -1 or end == -1 or end < begin:
        return [f"audio.rs: '{MARKER_BEGIN}' / '{MARKER_END}' marker block not found"]
    block = src[begin + len(MARKER_BEGIN):end]
    # One token per comment line: `// wav-g722`. Prose lines never match
    # because they contain spaces or uppercase letters.
    marker_tokens = set(re.findall(r"^//\s*([a-z0-9-]+)\s*$", block, re.M))

    canonical = set(FORMATS)
    for token in sorted(canonical - marker_tokens):
        failures.append(f"audio.rs: `{token}` is in FORMATS but missing from the docs-drift marker block")
    for token in sorted(marker_tokens - canonical):
        failures.append(f"audio.rs: `{token}` is in the docs-drift marker block but not in FORMATS")

    api_doc = API_MD.read_text(encoding="utf-8")
    cli_doc = CLI_MD.read_text(encoding="utf-8")
    for token, (api_needle, cli_needle) in FORMATS.items():
        if not re.search(api_needle, api_doc):
            failures.append(f"api.md: format `{token}` not found (needle: {api_needle!r})")
        if not re.search(cli_needle, cli_doc):
            failures.append(f"cli.md: format `{token}` not found (needle: {cli_needle!r})")
    return failures


# ---------------------------------------------------------------------------
# 4. mdBook SUMMARY + build
# ---------------------------------------------------------------------------

SUMMARY_LINK_RE = re.compile(r"\]\(([^)#]+)\)")


def check_workbook(skip_mdbook: bool) -> list[str]:
    failures: list[str] = []
    for lang in ("en", "ru"):
        src_dir = WORKBOOK / lang / "src"
        summary = src_dir / "SUMMARY.md"
        entries = set(SUMMARY_LINK_RE.findall(summary.read_text(encoding="utf-8")))
        for entry in sorted(entries):
            if not (src_dir / entry).is_file():
                failures.append(f"{summary.relative_to(ROOT)}: entry `{entry}` does not exist")
        chapters = {
            p.name
            for p in src_dir.glob("*.md")
            if p.name not in {"SUMMARY.md", "_template.md"} and not p.name.startswith("_")
        }
        for chapter in sorted(chapters - entries):
            failures.append(f"{summary.relative_to(ROOT)}: chapter `{chapter}` is not listed")

    mdbook = shutil.which("mdbook")
    if skip_mdbook or mdbook is None:
        note = "--skip-mdbook" if skip_mdbook else "mdbook not on PATH"
        print(f"warning: mdbook build check skipped ({note})", file=sys.stderr)
        return failures
    for lang in ("en", "ru"):
        book = WORKBOOK / lang
        # Build a throwaway copy: mdbook auto-creates stub files for SUMMARY
        # entries whose chapter is missing, which would mutate the real src/.
        with tempfile.TemporaryDirectory(prefix=f"mdbook-{lang}-") as tmp:
            copy = Path(tmp) / "book"
            shutil.copytree(book, copy)
            proc = subprocess.run(
                [mdbook, "build", str(copy)],
                capture_output=True,
                text=True,
            )
        if proc.returncode != 0:
            failures.append(f"mdbook build {book.relative_to(ROOT)} failed:\n{proc.stderr.strip()}")
    return failures


# ---------------------------------------------------------------------------
# 5. EN/RU parity
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^#{1,6} ", re.M)


def check_parity() -> list[str]:
    failures: list[str] = []
    en_dir = WORKBOOK / "en/src"
    ru_dir = WORKBOOK / "ru/src"
    en_files = {p.name for p in en_dir.glob("*.md")}
    ru_files = {p.name for p in ru_dir.glob("*.md")}
    for name in sorted(en_files - ru_files):
        failures.append(f"workbook parity: {name} exists in en/ but not ru/")
    for name in sorted(ru_files - en_files):
        failures.append(f"workbook parity: {name} exists in ru/ but not en/")
    for name in sorted(en_files & ru_files):
        en_heads = len(HEADING_RE.findall((en_dir / name).read_text(encoding="utf-8")))
        ru_heads = len(HEADING_RE.findall((ru_dir / name).read_text(encoding="utf-8")))
        if en_heads != ru_heads:
            failures.append(
                f"workbook parity: {name} has {en_heads} headings in en/ but {ru_heads} in ru/"
            )
    return failures


# ---------------------------------------------------------------------------
# 6. Relative links
# ---------------------------------------------------------------------------

LINK_RE = re.compile(r"!?\[[^\]]*\]\((<[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\)")


def github_slug(heading: str) -> str:
    """GitHub-style anchor slug: strip markup, lowercase, drop punctuation,
    spaces become hyphens. Unicode letters (e.g. Cyrillic) are kept."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # [text](url) -> text
    text = re.sub(r"<[^>]+>", "", text)  # inline HTML
    text = text.replace("`", "").replace("*", "").replace("_", "")
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s", "-", text)


def heading_slugs(path: Path) -> set[str]:
    slugs: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            slugs.add(github_slug(m.group(2)))
    return slugs


def link_check_files() -> list[Path]:
    files = sorted((ROOT / "docs").rglob("*.md"))
    files += sorted(ROOT.glob("README*.md"))
    files += sorted((ROOT / "packaging").rglob("README*"))
    return [f for f in files if f.is_file()]


def check_links() -> list[str]:
    failures: list[str] = []
    slug_cache: dict[Path, set[str]] = {}
    for md in link_check_files():
        text = md.read_text(encoding="utf-8")
        for m in LINK_RE.finditer(text):
            target = m.group(1).strip("<>")
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("//"):
                continue  # external URL, mailto:, etc.
            file_part, _, anchor = target.partition("#")

            resolved = (md.parent / unquote(file_part)).resolve() if file_part else md
            if file_part and not resolved.exists():
                failures.append(f"{md.relative_to(ROOT)}: link target `{target}` does not exist")
                continue
            if anchor and resolved.is_file() and resolved.suffix == ".md":
                if resolved not in slug_cache:
                    slug_cache[resolved] = heading_slugs(resolved)
                if unquote(anchor) not in slug_cache[resolved]:
                    failures.append(
                        f"{md.relative_to(ROOT)}: anchor `#{anchor}` not found in {resolved.relative_to(ROOT)}"
                    )
    return failures


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-mdbook", action="store_true", help="skip the mdbook build step")
    args = parser.parse_args()

    os.chdir(ROOT)
    allow = load_allowlist(ALLOWLIST)

    results: list[tuple[str, list[str]]] = []
    results.append(("CLI flags/envs (cli.md == main.rs)", check_cli(allow)))
    results.append(("WS error codes (asyncapi.yaml == api.md == ws.rs)", check_ws_error_codes(allow)))
    results.append(("audio formats (api.md/cli.md == audio.rs marker)", check_formats()))
    results.append(("mdBook SUMMARY + build", check_workbook(args.skip_mdbook)))
    results.append(("workbook EN/RU parity", check_parity()))
    results.append(("relative links", check_links()))

    failed = 0
    for name, failures in results:
        if failures:
            failed += 1
            print(f"FAIL {name}")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print(f"PASS {name}")
    total = sum(len(f) for _, f in results)
    if failed:
        print(f"\ndocs drift detected: {total} problem(s) in {failed} check(s)", file=sys.stderr)
        return 1
    print("\nno docs drift detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
