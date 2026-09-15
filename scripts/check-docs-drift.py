#!/usr/bin/env python3
"""Docs drift gate: fail when documentation drifts away from the code.

Ten checks, all stdlib-only (no third-party deps, no network):

  1. CLI flags/envs: every clap flag + GIGASTT_* env in the CLI sources
     (crates/gigastt/src/{main,serve,serve/bind,transcribe_cmd}.rs) is
     documented in docs/cli.md, and cli.md names no flag/env that does not
     exist (intentional exceptions live in scripts/check-docs-drift.allowlist).
  2. CLI flag scoping: a `--flag` documented under a `gigastt <subcommand>`
      section of cli.md must exist on that subcommand's clap struct (flattened
      arg structs and top-level global flags included). Axis 1 matches tokens
      globally, so it cannot catch a serve-only flag documented under
      `gigastt transcribe` — this one fails on it.
  3. CLI defaults: a `[default: X]` marker in cli.md must match the clap
      `default_value` / `default_value_t` literal for the flag in that
      section (UPPER_CASE constants are resolved from the CLI sources;
      non-literal expressions like `model::default_model_dir()` are skipped).
      A missing marker is never a failure — only a value mismatch is.
  4. WS error codes: the enum in docs/asyncapi.yaml == the table in docs/api.md
     == the codes emitted under crates/gigastt/src/server/ws/ (plus allowlisted
     doc-only entries).
  5. Audio formats: the canonical FORMATS list below == the `// docs-drift: codecs`
     marker block in crates/gigastt-core/src/inference/audio/decode.rs, and every format
     is named in docs/api.md and docs/cli.md. When adding a codec, update all
     three places (marker, FORMATS, docs) in the same commit.
  6. Relative links: every relative markdown link in docs/**, .github/*.md,
     the root README*, and packaging/**/README* resolves to an existing file/directory, and
     #anchors resolve to a heading in the target file.
  7. OpenAPI paths: every `paths:` key in docs/openapi.yaml is registered in
     crates/gigastt/src/server/{mod,router,listen}.rs (or is the separate metrics listener),
     and OpenAPI must describe the supported formats.
  8. Duration claims: contract docs must not state an unconditional audio limit
     when the default file-transcription path has no duration cap.
  9. .github/SECURITY.md supported-version table marks the workspace Cargo.toml major.minor
     as current and the previous minor as previous.
  10. Crate version pins: `gigastt-core = "X.Y"` in README*/architecture.md must
     match the workspace package major.minor.

Exit code 0 when everything is in sync, 1 otherwise. Runs in seconds.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent

CLI_SOURCES = (
    ROOT / "crates/gigastt/src/main.rs",
    ROOT / "crates/gigastt/src/serve.rs",
    ROOT / "crates/gigastt/src/serve/bind.rs",
    ROOT / "crates/gigastt/src/transcribe_cmd.rs",
)
WS_DIR = ROOT / "crates/gigastt/src/server/ws"
SERVER_ROUTE_SOURCES = (
    ROOT / "crates/gigastt/src/server/mod.rs",
    ROOT / "crates/gigastt/src/server/router.rs",
    ROOT / "crates/gigastt/src/server/listen.rs",
)
# Codec marker lives in the decode module after the audio/ feature split.
AUDIO_RS = ROOT / "crates/gigastt-core/src/inference/audio/decode.rs"
CARGO_TOML = ROOT / "Cargo.toml"
SECURITY_MD = ROOT / ".github/SECURITY.md"
OPENAPI_YAML = ROOT / "docs/openapi.yaml"
CLI_MD = ROOT / "docs/cli.md"
API_MD = ROOT / "docs/api.md"
ASYNCAPI_YAML = ROOT / "docs/asyncapi.yaml"
ALLOWLIST = ROOT / "scripts/check-docs-drift.allowlist"
PIN_FILES = [
    ROOT / "README.md",
    ROOT / "docs/architecture.md",
]

# Canonical audio decode surface. The token list must match the
# `// docs-drift: codecs` marker block in inference/audio/decode.rs exactly; each
# needle regex must appear in the named doc. Update this table, the marker,
# and the docs together whenever a codec or container is added/removed.
FORMATS = {
    # token: (api.md needle, cli.md needle)
    "wav": (r"WAV \(PCM", r"Supports: WAV"),
    "wav-g711": (r"G\.711 A-law / μ-law", r"G\.711 A-law/μ-law"),
    "wav-g722": (r"WAV with G\.722", r"G\.722 payloads"),
    "wav-gsm": (r"GSM 06\.10", r"GSM 06\.10"),
    "mp3": (r"MP3", r"MP3"),
    "m4a": (r"M4A", r"M4A"),
    "ogg-vorbis": (r"OGG/Vorbis", r"OGG/Vorbis"),
    "ogg-opus": (r"OGG/Opus", r"OGG/Opus"),
    "webm-opus": (r"WebM/Opus", r"WebM/Opus"),
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
# 1–3. CLI flags / env vars, scoping and defaults
# ---------------------------------------------------------------------------

ARG_RE = re.compile(
    r"#\[arg\((.*?)\)\]\s*(?:pub(?:\([^)]+\))?\s+)?(\w+)\s*:",
    re.S,
)
LONG_NAME_RE = re.compile(r'long\s*=\s*"([^"]+)"')
ENV_RE = re.compile(r'env\s*=\s*"(GIGASTT_[A-Z0-9_]+)"')
ENV_CALL_RE = re.compile(r'env::(?:var|var_os|set_var)\(\s*"(GIGASTT_[A-Z0-9_]+)"')
CLI_FLAG_TOKEN_RE = re.compile(r"--([a-z0-9][a-z0-9-]*)")
ENV_TOKEN_RE = re.compile(r"\bGIGASTT_[A-Z0-9_]+\b")


def parse_cli_definition() -> tuple[set[str], set[str]]:
    """Extract clap long-flag names and GIGASTT_* env vars from CLI sources."""
    flags: set[str] = set()
    envs: set[str] = set()
    for path in CLI_SOURCES:
        src = path.read_text(encoding="utf-8")
        envs.update(ENV_CALL_RE.findall(src))
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
        failures.append(f"cli.md: flag --{flag} (CLI sources) is not documented")
    for env in sorted(envs - doc_envs - ok_envs):
        failures.append(f"cli.md: env var {env} (CLI sources) is not documented")
    for flag in sorted(doc_flags - flags - phantom_flags):
        failures.append(f"cli.md: --{flag} does not match any clap flag in CLI sources")
    for env in sorted(doc_envs - envs - phantom_envs):
        failures.append(f"cli.md: {env} is not a GIGASTT_* env var read by CLI sources")
    return failures


# --- 1b/1c: per-subcommand scoping + defaults ------------------------------

# Section headers inside cli.md: `gigastt serve [OPTIONS]`, `gigastt
# transcribe-batch [OPTIONS] <…>`, and the top-level `gigastt [OPTIONS]
# <COMMAND>` (no subcommand group). Example invocations are indented, so a
# column-0 anchor is enough.
SECTION_HEADER_RE = re.compile(r"^gigastt(?:[ \t]+([a-z][a-z0-9-]*))?[^\n]*$", re.M)
FLATTEN_RE = re.compile(r"#\[command\(flatten\)\]\s*(?:pub(?:\([^)]*\))?\s+)?\w+\s*:\s*(\w+)")
DEFAULT_VALUE_RE = re.compile(r'default_value\s*=\s*"([^"]*)"')
DEFAULT_VALUE_T_RE = re.compile(r"default_value_t\s*=\s*([^\s,]+)")
CONST_DEF_RE = re.compile(r'\bconst\s+([A-Z][A-Z0-9_]*)\s*:[^=;]+=\s*("[^"]*"|-?\d+|true|false)\s*;')
DOC_DEFAULT_RE = re.compile(r"\[default:\s*([^\]]+)\]")
# A flag *definition* line in cli.md: the flag opens the line (small indent,
# optional short alias). Prose mentions like "stay at --pool-size)" are
# mid-line and must not start a defaults window — otherwise a mention swallows
# the next flag's `[default: …]` marker and misfires.
DEF_TOKEN_RE = re.compile(r"(?m)^[ \t]{0,6}(?:-[a-zA-Z],[ \t]*)?--([a-z0-9][a-z0-9-]*)")


def _code_skeleton(src: str) -> str:
    """Same-length copy of `src` with line comments and string-literal
    contents blanked to spaces. Structural scans (brace matching, variant
    headers) run on this so braces inside comments/strings cannot confuse
    them; attribute parsing runs on the original text — indices are
    preserved, so spans map 1:1."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        if src[i] == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j == -1 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif src[i] == '"':
            out[i] = " "
            i += 1
            while i < n and src[i] != '"':
                if src[i] == "\\":
                    out[i] = " "
                    i += 1
                if i < n and src[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


def _brace_span(skel: str, open_idx: int) -> tuple[int, int] | None:
    """(start, end) of the text inside the braces opened at skel[open_idx]."""
    depth = 0
    for i in range(open_idx, len(skel)):
        if skel[i] == "{":
            depth += 1
        elif skel[i] == "}":
            depth -= 1
            if depth == 0:
                return open_idx + 1, i
    return None


def collect_clap_items() -> tuple[dict[str, str], dict[str, str]]:
    """Struct name → body and Commands variant name → body (original text).

    Tuple variants such as `Serve(ServeArgs)` are resolved to the referenced
    struct body; inline variants (`Download { … }`) keep their own body.
    """
    sources: list[tuple[str, str]] = []
    for path in CLI_SOURCES:
        if path.exists():
            src = path.read_text(encoding="utf-8")
            sources.append((src, _code_skeleton(src)))

    structs: dict[str, str] = {}
    for src, skel in sources:
        for m in re.finditer(r"\bstruct\s+(\w+)\s*\{", skel):
            span = _brace_span(skel, skel.index("{", m.end() - 1))
            if span:
                structs[m.group(1)] = src[span[0]:span[1]]

    variants: dict[str, str] = {}
    for src, skel in sources:
        em = re.search(r"\benum\s+Commands\s*\{", skel)
        if not em:
            continue
        span = _brace_span(skel, skel.index("{", em.end() - 1))
        if not span:
            continue
        base, ebody = span[0], skel[span[0]:span[1]]
        for vm in re.finditer(r"(?m)^[ \t]*(\w+)[ \t]*([({])", ebody):
            name, kind = vm.group(1), vm.group(2)
            if kind == "(":
                tm = re.match(r"\s*(\w+)\s*\)", ebody[vm.end():])
                if tm and tm.group(1) in structs:
                    variants[name] = structs[tm.group(1)]
            else:
                vspan = _brace_span(ebody, vm.end() - 1)
                if vspan:
                    variants[name] = src[base + vspan[0]:base + vspan[1]]
    return structs, variants


def flags_in_body(body: str, structs: dict[str, str], _depth: int = 0) -> dict[str, str]:
    """Long-flag name → raw `#[arg(…)]` attrs for one struct/variant body,
    following `#[command(flatten)]` references into other structs."""
    flags: dict[str, str] = {}
    if _depth > 4:
        return flags
    for attrs, field in ARG_RE.findall(body):
        if "long" not in attrs:
            continue
        m = LONG_NAME_RE.search(attrs)
        flags[m.group(1) if m else field.replace("_", "-")] = attrs
    for ty in FLATTEN_RE.findall(body):
        sub = structs.get(ty)
        if sub is not None:
            flags.update(flags_in_body(sub, structs, _depth + 1))
    return flags


def cli_md_sections(doc: str) -> list[tuple[str, str]]:
    """Split cli.md into (`subcommand`, section text); top-level is `gigastt`."""
    headers = list(SECTION_HEADER_RE.finditer(doc))
    sections = []
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(doc)
        sections.append((m.group(1) or "gigastt", doc[m.start():end]))
    return sections


def _variant_name(section: str) -> str:
    """cli.md section → Commands variant: transcribe-batch → TranscribeBatch."""
    return "".join(part.capitalize() for part in section.split("-"))


def _section_flags(
    name: str,
    structs: dict[str, str],
    variants: dict[str, str],
    global_flags: dict[str, str],
) -> dict[str, str] | None:
    """flag → attrs allowed in one cli.md section, or None when the section
    has no matching Commands variant."""
    if name == "gigastt":
        return dict(global_flags)
    body = variants.get(_variant_name(name))
    if body is None:
        return None
    allowed = flags_in_body(body, structs)
    allowed.update(global_flags)
    return allowed


def check_cli_sections(allow: dict[str, dict[str, str]]) -> list[str]:
    structs, variants = collect_clap_items()
    global_flags = flags_in_body(structs.get("Cli", ""), structs)
    phantom = set(allow.get("section-flags-phantom-ok", {}))
    doc = CLI_MD.read_text(encoding="utf-8")

    failures: list[str] = []
    for name, text in cli_md_sections(doc):
        allowed = _section_flags(name, structs, variants, global_flags)
        if allowed is None:
            failures.append(
                f"cli.md: section `gigastt {name}` has no matching Commands variant in CLI sources"
            )
            continue
        for flag in sorted(set(CLI_FLAG_TOKEN_RE.findall(text)) - set(allowed) - phantom):
            failures.append(
                f"cli.md: --{flag} is documented under `gigastt {name}` but that "
                "subcommand has no such clap flag (wrong section or stale flag?)"
            )
    return failures


def collect_consts() -> dict[str, str]:
    """Literal `const NAME: … = <literal>;` definitions in the CLI sources."""
    consts: dict[str, str] = {}
    for path in CLI_SOURCES:
        if not path.exists():
            continue
        for m in CONST_DEF_RE.finditer(path.read_text(encoding="utf-8")):
            value = m.group(2)
            consts[m.group(1)] = value.strip('"') if value.startswith('"') else value
    return consts


def _resolve_default(attrs: str, consts: dict[str, str]) -> str | None:
    """The literal clap prints as `[default: …]`, or None when the default is
    not safely comparable (no default, function call, enum path, unresolved
    constant) — ambiguity skips the flag rather than failing falsely."""
    m = DEFAULT_VALUE_RE.search(attrs)
    if m:
        return m.group(1)
    m = DEFAULT_VALUE_T_RE.search(attrs)
    if not m:
        return None
    expr = m.group(1).strip()
    if re.fullmatch(r"-?\d+|true|false", expr):
        return expr
    if re.fullmatch(r"(?:\w+::)*[A-Z][A-Z0-9_]*", expr):
        return consts.get(expr.rsplit("::", 1)[-1])
    return None


def _normalize_doc_default(raw: str) -> str:
    """The literal part of a `[default: …]` marker. Docs append a human
    reading after the literal (`[default: 524288 = 512 KiB]`, `[default:
    auto = on for rnnt, …]`); clap prints only the part before ` = `."""
    return re.sub(r"\s+", " ", raw).strip().split(" = ", 1)[0].strip()


def check_cli_defaults() -> list[str]:
    structs, variants = collect_clap_items()
    consts = collect_consts()
    global_flags = flags_in_body(structs.get("Cli", ""), structs)
    doc = CLI_MD.read_text(encoding="utf-8")

    failures: set[str] = set()
    for name, text in cli_md_sections(doc):
        allowed = _section_flags(name, structs, variants, global_flags)
        if allowed is None:
            continue  # reported by check_cli_sections
        tokens = list(DEF_TOKEN_RE.finditer(text))
        for i, tok in enumerate(tokens):
            attrs = allowed.get(tok.group(1))
            if attrs is None:
                continue  # phantom in this section — the scoping check reports it
            code_default = _resolve_default(attrs, consts)
            if code_default is None:
                continue
            end = tokens[i + 1].start() if i + 1 < len(tokens) else len(text)
            for dm in DOC_DEFAULT_RE.finditer(text[tok.start():end]):
                doc_default = _normalize_doc_default(dm.group(1))
                if doc_default != code_default:
                    failures.add(
                        f"cli.md: --{tok.group(1)} under `gigastt {name}` documents "
                        f"[default: {doc_default}] but the clap default is {code_default}"
                    )
    return sorted(failures)


# ---------------------------------------------------------------------------
# 4. WebSocket error codes
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


def emitted_ws_codes() -> set[str]:
    codes: set[str] = set()
    if not WS_DIR.is_dir():
        return codes
    for path in sorted(WS_DIR.rglob("*.rs")):
        codes.update(WS_CODE_RE.findall(path.read_text(encoding="utf-8")))
    return codes


def check_ws_error_codes(allow: dict[str, dict[str, str]]) -> list[str]:
    async_codes = asyncapi_codes()
    api_codes = api_md_ws_codes()
    emitted = emitted_ws_codes()
    doc_only = set(allow.get("ws-codes-doc-only", {}))
    undoc_ok = set(allow.get("ws-codes-undocumented-ok", {}))

    failures: list[str] = []
    if not WS_DIR.is_dir():
        failures.append("crates/gigastt/src/server/ws/: directory missing")
        return failures
    for code in sorted(async_codes - api_codes):
        failures.append(f"asyncapi.yaml: `{code}` is missing from the docs/api.md error-code table")
    for code in sorted(api_codes - async_codes):
        failures.append(f"api.md: `{code}` is missing from the docs/asyncapi.yaml enum")
    for code in sorted(emitted - api_codes - undoc_ok):
        failures.append(f"ws/: `{code}` is emitted but not documented in api.md/asyncapi.yaml")
    for code in sorted(api_codes - emitted - doc_only):
        failures.append(f"api.md: `{code}` is documented but never emitted by ws/ (allowlist if REST-only)")
    return failures


# ---------------------------------------------------------------------------
# 5. Audio formats
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
# 6. Relative links
# ---------------------------------------------------------------------------

LINK_RE = re.compile(r"!?\[[^\]]*\]\((<[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\)")


def github_slug(heading: str) -> str:
    """GitHub-style anchor slug: strip markup, lowercase, drop punctuation,
    spaces become hyphens. Unicode letters (e.g. Cyrillic) are kept, and
    underscores are preserved (GitHub does not strip them)."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # [text](url) -> text
    text = re.sub(r"<[^>]+>", "", text)  # inline HTML
    text = text.replace("`", "").replace("*", "")
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
    files += sorted((ROOT / ".github").glob("*.md"))
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


def workspace_version() -> tuple[int, int, int]:
    text = CARGO_TOML.read_text(encoding="utf-8")
    m = re.search(r'(?m)^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', text)
    if not m:
        raise SystemExit("Cargo.toml: workspace version not found")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def server_routes() -> set[str]:
    """Collect string route paths from the server route table `.route("…")` calls."""
    routes: set[str] = set()
    for path in SERVER_ROUTE_SOURCES:
        if path.exists():
            routes.update(re.findall(r'\.route\(\s*"([^"]+)"', path.read_text(encoding="utf-8")))
    return routes


def openapi_paths() -> set[str]:
    """Collect top-level OpenAPI path keys (lines like `  /health:` under paths:)."""
    text = OPENAPI_YAML.read_text(encoding="utf-8")
    # Only path keys under the paths: block — skip components and examples.
    in_paths = False
    found: set[str] = set()
    for line in text.splitlines():
        if re.match(r"^paths:\s*$", line):
            in_paths = True
            continue
        if in_paths and re.match(r"^[A-Za-z]", line):
            # next top-level key ends paths:
            break
        if in_paths:
            m = re.match(r"^  (/[^:]+):\s*$", line)
            if m:
                found.add(m.group(1))
    return found


def check_openapi() -> list[str]:
    failures: list[str] = []
    if not OPENAPI_YAML.exists() or not any(p.exists() for p in SERVER_ROUTE_SOURCES):
        failures.append("openapi.yaml or server route sources missing")
        return failures

    routes = server_routes()
    # Metrics is served on a separate listener; still a real HTTP path.
    routes.add("/metrics")
    # OpenAPI uses templated job paths; normalize server routes that use {id}.
    # The route table already uses `/v1/jobs/{id}` style axum paths.
    oas = openapi_paths()
    for path in sorted(oas):
        if path not in routes:
            failures.append(
                f"openapi.yaml: path `{path}` is not registered in the server route table "
                f"(known routes include {sorted(p for p in routes if p.startswith('/v1') or p in ('/health', '/ready', '/metrics'))})"
            )

    # Required product surfaces that must stay documented.
    for required in (
        "/health",
        "/ready",
        "/v1/models",
        "/v1/transcribe",
        "/v1/transcribe/stream",
        "/v1/admin/reload",
        "/v1/jobs",
    ):
        if required not in oas:
            failures.append(f"openapi.yaml: missing required path `{required}`")

    oas_text = OPENAPI_YAML.read_text(encoding="utf-8")

    # Formats intro must mention Opus (full surface is in FORMATS gate for api/cli).
    if not re.search(r"Opus", oas_text):
        failures.append("openapi.yaml: audio formats description must mention Opus")

    return failures


# Words that legitimately scope a duration ceiling to the paths that really have
# one. A claim carrying any of these is correct; a bare one is not.
_DURATION_SCOPE = re.compile(
    r"whole[- ]buffer|diariz|channels=split|telephony|G\.722|max-audio-secs|"
    r"MAX_AUDIO_SECS|safety ceiling",
    re.IGNORECASE,
)

# "<something> cap: 10 minutes", "Max audio duration: 30 min", "audio limit: 1800 s".
_DURATION_CLAIM = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z /]*(?:cap|limit|duration|ceiling))\s*[:=]\s*"
    r"~?\s*(?P<num>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>minutes?|mins?\b|hours?|hrs?\b|seconds?|secs?\b|[smh]\b)",
    re.IGNORECASE,
)


def check_duration_claims() -> list[str]:
    """Forbid unconditional audio-duration ceilings in the contract docs.

    The default file path, the VAD path and OGG/Opus all decode in bounded
    windows and have no duration limit; only the whole-buffer paths keep one.
    A claim like "File transcription cap: 10 minutes" is therefore always wrong
    unless it names the paths it applies to.

    This deliberately checks every contract surface, not just openapi.yaml: the
    previous version of this gate scanned that one file for the literal string
    "30 minutes", so a bogus 10-minute cap sat in asyncapi.yaml unnoticed.
    """
    failures: list[str] = []
    for path in (OPENAPI_YAML, ASYNCAPI_YAML, API_MD, CLI_MD):
        if not path.exists():
            failures.append(f"{path.name} missing")
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = _DURATION_CLAIM.search(line)
            if not m or _DURATION_SCOPE.search(line):
                continue
            # Session/idle/request budgets are unrelated to audio length.
            if re.search(r"session|idle|drain|checkout|retry|ttl|timeout", line, re.IGNORECASE):
                continue
            failures.append(
                f"{path.name}:{lineno}: unconditional duration claim "
                f"{m.group('label').strip()!r} = {m.group('num')} {m.group('unit')}; "
                "audio length is unlimited by default — name the whole-buffer paths "
                "(diarization / channels=split / telephony) or --max-audio-secs"
            )
    return failures


def check_security_versions() -> list[str]:
    failures: list[str] = []
    if not SECURITY_MD.exists():
        return [".github/SECURITY.md missing"]
    major, minor, _patch = workspace_version()
    current = f"{major}.{minor}.x"
    text = SECURITY_MD.read_text(encoding="utf-8")
    # Expect a table row like: | 2.14.x  | Yes (current)  |
    if not re.search(rf"\|\s*{re.escape(current)}\s*\|\s*Yes\s*\(current\)", text):
        failures.append(
            f".github/SECURITY.md: supported-versions table must mark `{current}` as Yes (current) "
            f"(workspace version is {major}.{minor}.*)"
        )
    if minor > 0:
        previous = f"{major}.{minor - 1}.x"
        if not re.search(
            rf"\|\s*{re.escape(previous)}\s*\|\s*Yes\s*\(previous\)", text
        ):
            failures.append(
                f".github/SECURITY.md: supported-versions table must mark `{previous}` as Yes (previous) "
                f"(workspace minor is {major}.{minor})"
            )
    return failures


def check_crate_pins() -> list[str]:
    failures: list[str] = []
    major, minor, _patch = workspace_version()
    expected = f'{major}.{minor}'
    pin_re = re.compile(r'gigastt-core\s*=\s*"(\d+)\.(\d+)(?:\.\d+)?"')
    for path in PIN_FILES:
        if not path.exists():
            failures.append(f"{path.relative_to(ROOT)}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        pins = pin_re.findall(text)
        if not pins:
            # architecture + README are expected to show an embed pin
            if path.name in ("README.md", "architecture.md"):
                failures.append(
                    f"{path.relative_to(ROOT)}: expected a `gigastt-core = \"{expected}\"` pin"
                )
            continue
        for maj_s, min_s in pins:
            if int(maj_s) != major or int(min_s) != minor:
                failures.append(
                    f'{path.relative_to(ROOT)}: pin gigastt-core = "{maj_s}.{min_s}" '
                    f'does not match workspace {expected} (use gigastt-core = "{expected}")'
                )
    return failures


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    os.chdir(ROOT)
    allow = load_allowlist(ALLOWLIST)

    results: list[tuple[str, list[str]]] = []
    results.append(("CLI flags/envs (cli.md == CLI sources)", check_cli(allow)))
    results.append(("CLI flag scoping (cli.md sections == clap structs)", check_cli_sections(allow)))
    results.append(("CLI defaults (cli.md [default: …] == clap)", check_cli_defaults()))
    results.append(("WS error codes (asyncapi.yaml == api.md == ws/)", check_ws_error_codes(allow)))
    results.append(("audio formats (api.md/cli.md == audio.rs marker)", check_formats()))
    results.append(("relative links", check_links()))
    results.append(("OpenAPI paths + format claims", check_openapi()))
    results.append(("no unconditional duration caps (all contract docs)", check_duration_claims()))
    results.append((".github/SECURITY.md supported versions", check_security_versions()))
    results.append(("crate version pins (README/architecture)", check_crate_pins()))

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
