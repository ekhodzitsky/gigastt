"""Common utilities for benchmark: text normalization, WER, RTF, metadata."""

import datetime
import hashlib
import json
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Optional


def home_dir() -> Path:
    return Path.home()


_MANIFESTS_DIR = Path(__file__).parent / "manifests"


DATASETS: dict[str, Path] = {
    "golos_crowd": _MANIFESTS_DIR / "golos_crowd.json",
}


def _register_committed_datasets() -> None:
    """Auto-register any JSON manifests found in benchmark/manifests/."""
    if _MANIFESTS_DIR.exists():
        for path in sorted(_MANIFESTS_DIR.glob("*.json")):
            DATASETS[path.stem] = path


def _legacy_crowd_path() -> Path:
    return home_dir() / ".gigastt/benchmarks/golos_wav/manifest.json"


def manifest_path(dataset: str = "golos_crowd") -> Path:
    """Resolve the manifest path for a dataset.

    Defaults to ``golos_crowd`` for backward compatibility. If the committed
    manifest does not exist, falls back to the legacy crowd manifest or the
    bundled fixtures. Unknown datasets raise ``ValueError``.
    """
    _register_committed_datasets()

    if dataset in DATASETS:
        p = DATASETS[dataset]
        if p.exists():
            return p

    if dataset == "golos_crowd":
        legacy = _legacy_crowd_path()
        if legacy.exists():
            return legacy
        return Path(__file__).parent.parent / "crates/gigastt/tests/fixtures/manifest.json"

    known = ", ".join(sorted(DATASETS))
    raise ValueError(f"Unknown dataset {dataset!r}. Known datasets: {known}")


def load_manifest(max_samples: Optional[int] = None, dataset: str = "golos_crowd") -> dict:
    """Load a benchmark manifest.

    Supports both the new registry format (JSON object with ``audio_root`` and
    ``samples``) and the legacy list format (list of ``{"filename", "reference"}``).
    Filenames are resolved to absolute paths. Samples whose ``reference`` is empty
    or whitespace-only are skipped and counted.

    Returns a dict with ``samples`` (list of non-empty samples with resolved paths)
    and ``skipped_empty_refs`` (int).
    """
    path = manifest_path(dataset)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        audio_root = Path(data.get("audio_root", "~")).expanduser().resolve()
        raw_samples = data.get("samples", [])
        base_dir = audio_root
    elif isinstance(data, list):
        raw_samples = data
        base_dir = path.parent
    else:
        raise ValueError(f"Manifest {path} must be a JSON object or list")

    result = []
    skipped_empty_refs = 0
    for s in raw_samples:
        reference = s.get("reference", "")
        if not reference.strip():
            skipped_empty_refs += 1
            continue

        filename = s["filename"]
        fp = Path(filename)
        if not fp.is_absolute():
            fp = base_dir / fp
        wav_path = str(fp.expanduser().resolve())

        sample = {
            "filename": wav_path,
            "reference": reference,
        }
        if "duration" in s:
            sample["duration"] = s["duration"]
        result.append(sample)

    if max_samples and max_samples > 0:
        result = result[:max_samples]
    return {"samples": result, "skipped_empty_refs": skipped_empty_refs}


# --- Russian words-to-digits ITN for symmetric WER normalization ---

_UNIT = 1
_TEEN = 10
_TEN = 10
_HUNDRED = 100
_THOUSAND = 1000
_MILLION = 1_000_000

_NUMBER_WORDS: dict[str, tuple[int, int]] = {}


def _add_num(value: int, scale: int, forms: list[str]) -> None:
    for form in forms:
        _NUMBER_WORDS[form] = (value, scale)


# Cardinals 0-9 with common case/gender forms.
_add_num(0, _UNIT, ["ноль", "ноля", "нолю", "нолем", "нолём", "ноле"])
_add_num(1, _UNIT, [
    "один", "одна", "одно", "одного", "одной", "одному", "одном", "одним",
])
_add_num(2, _UNIT, ["два", "две", "двух", "двум", "двумя"])
_add_num(3, _UNIT, ["три", "трех", "трёх", "трем", "трём", "тремя"])
_add_num(4, _UNIT, ["четыре", "четырёх", "четырех", "четырём", "четырем", "четырьмя"])
_add_num(5, _UNIT, ["пять", "пяти", "пятью"])
_add_num(6, _UNIT, ["шесть", "шести", "шестью"])
_add_num(7, _UNIT, ["семь", "семи", "семью"])
_add_num(8, _UNIT, ["восемь", "восьми", "восьмью"])
_add_num(9, _UNIT, ["девять", "девяти", "девятью"])

# Teens 10-19.
for _v, _forms in [
    (10, ["десять", "десяти", "десятью"]),
    (11, ["одиннадцать", "одиннадцати"]),
    (12, ["двенадцать", "двенадцати"]),
    (13, ["тринадцать", "тринадцати"]),
    (14, ["четырнадцать", "четырнадцати"]),
    (15, ["пятнадцать", "пятнадцати"]),
    (16, ["шестнадцать", "шестнадцати"]),
    (17, ["семнадцать", "семнадцати"]),
    (18, ["восемнадцать", "восемнадцати"]),
    (19, ["девятнадцать", "девятнадцати"]),
]:
    _add_num(_v, _TEEN, _forms)

# Tens 20-90.
for _v, _forms in [
    (20, ["двадцать", "двадцати"]),
    (30, ["тридцать", "тридцати"]),
    (40, ["сорок", "сорока"]),
    (50, ["пятьдесят", "пятидесяти"]),
    (60, ["шестьдесят", "шестидесяти"]),
    (70, ["семьдесят", "семидесяти"]),
    (80, ["восемьдесят", "восьмидесяти"]),
    (90, ["девяносто", "девяноста"]),
]:
    _add_num(_v, _TEN, _forms)

# Hundreds 100-900.
for _v, _forms in [
    (100, ["сто", "ста"]),
    (200, ["двести", "двухсот"]),
    (300, ["триста", "трехсот", "трёхсот"]),
    (400, ["четыреста", "четырёхсот"]),
    (500, ["пятьсот", "пятисот"]),
    (600, ["шестьсот", "шестисот"]),
    (700, ["семьсот", "семисот"]),
    (800, ["восемьсот", "восьмисот"]),
    (900, ["девятьсот", "девятисот"]),
]:
    _add_num(_v, _HUNDRED, _forms)

# Scale words (all common case forms).
_add_num(1000, _THOUSAND, [
    "тысяча", "тысячи", "тысяч", "тысяче", "тысячу", "тысячей",
    "тысячам", "тысячами", "тысячах",
])
_add_num(1_000_000, _MILLION, [
    "миллион", "миллиона", "миллионов", "миллиону", "миллионе",
    "миллионам", "миллионами", "миллионах",
])

# Ordinals 1-9 (nominative and common case forms).
_ORDINAL_UNIT_FORMS: dict[int, list[str]] = {
    1: ["первый", "первая", "первое", "первого", "первой", "первому", "первом", "первым"],
    2: ["второй", "вторая", "второе", "второго", "второй", "второму", "втором", "вторым"],
    3: ["третий", "третья", "третье", "третьего", "третьей", "третьему", "третьем", "третьим"],
    4: ["четвертый", "четвертая", "четвертое", "четвертого", "четвертой", "четвертому", "четвертом", "четвертым"],
    5: ["пятый", "пятая", "пятое", "пятого", "пятой", "пятому", "пятом", "пятым"],
    6: ["шестой", "шестая", "шестое", "шестого", "шестой", "шестому", "шестом", "шестым"],
    7: ["седьмой", "седьмая", "седьмое", "седьмого", "седьмой", "седьмому", "седьмом", "седьмым"],
    8: ["восьмой", "восьмая", "восьмое", "восьмого", "восьмой", "восьмому", "восьмом", "восьмым"],
    9: ["девятый", "девятая", "девятое", "девятого", "девятой", "девятому", "девятым"],
}
for _v, _forms in _ORDINAL_UNIT_FORMS.items():
    _add_num(_v, _UNIT, _forms)

# Ordinal teens 11-19.
_ORDINAL_TEEN_BASES = {
    11: "одиннадцат", 12: "двенадцат", 13: "тринадцат", 14: "четырнадцат",
    15: "пятнадцат", 16: "шестнадцат", 17: "семнадцат", 18: "восемнадцат", 19: "девятнадцат",
}
for _v, _base in _ORDINAL_TEEN_BASES.items():
    _add_num(_v, _TEEN, [
        f"{_base}ый", f"{_base}ая", f"{_base}ое", f"{_base}ого",
        f"{_base}ой", f"{_base}ому", f"{_base}ом", f"{_base}ым",
    ])

# Ordinal tens 20-90.
_ORDINAL_TEN_BASES = {
    20: "двадцат", 30: "тридцат", 40: "сороков", 50: "пятидесят",
    60: "шестидесят", 70: "семидесят", 80: "восьмидесят", 90: "девяност",
}
for _v, _base in _ORDINAL_TEN_BASES.items():
    _add_num(_v, _TEN, [
        f"{_base}ый", f"{_base}ая", f"{_base}ое", f"{_base}ого",
        f"{_base}ой", f"{_base}ому", f"{_base}ом", f"{_base}ым",
    ])

# Ordinal hundreds.
_ORDINAL_HUNDRED_FORMS = {
    100: ["сотый", "сотая", "сотое", "сотого", "сотой", "сотому", "сотом", "сотым"],
    200: ["двухсотый", "двухсотая", "двухсотое", "двухсотого", "двухсотой", "двухсотому", "двухсотом", "двухсотым"],
    300: ["трёхсотый", "трехсотый", "трёхсотая", "трехсотая", "трёхсотое", "трехсотое"],
    400: ["четырёхсотый", "четырехсотый", "четырёхсотая", "четырехсотая", "четырёхсотое", "четырехсотое"],
    500: ["пятисотый", "пятисотая", "пятисотое"],
    600: ["шестисотый", "шестисотая", "шестисотое"],
    700: ["семисотый", "семисотая", "семисотое"],
    800: ["восьмисотый", "восьмисотая", "восьмисотое"],
    900: ["девятисотый", "девятисотая", "девятисотое"],
}
for _v, _forms in _ORDINAL_HUNDRED_FORMS.items():
    _add_num(_v, _HUNDRED, _forms)


# Symbols and punctuation that are removed after digit-group merging.  Keeping
# them as separate tokens during merging prevents "15% 180" from collapsing
# into "15180" just because the percent sign was stripped too early.
_SYMBOL_STRIP = '+№%-€₽.,!?;:"'"'"'()[]{}«»—…'

_EMPTY_TOKENS = {
    "плюс", "плюса", "плюсу", "плюсом", "плюсе",
    "минус", "минуса", "минусу", "минусом", "минусе",
    "номер", "номера", "номеру", "номером", "номере",
    "процент", "процента", "процентов", "проценту", "процентом", "проценте", "процентами",
    # Spoken equivalents of currency symbols listed in the normalization spec.
    "рубль", "рубля", "рублей", "рубли", "рублем", "рублями",
    "доллар", "доллара", "долларов", "доллары", "долларом", "долларами",
    "евро",
    # Latin abbreviation for "номер" often output by engines.
    "no",
    # Wake-word artifacts that are inconsistently transcribed across engines.
    "джой",
}

# FROZEN map — do NOT extend. Each entry is a widely-used Latin brand/abbreviation
# that recurs in references in Latin spelling while some engines transliterate it to
# Cyrillic: device/service brands (apple, iphone, samsung, sony), social/media
# (youtube, facebook, instagram, telegram, vk, ok, netflix, spotify, whatsapp), the
# aliexpress marketplace, and the tv/synergy/pink tokens. INVARIANT: do not add new
# entries to lower WER. The words-to-digits ITN and this map only fire on hypotheses
# containing Latin/digit tokens, so growing the map silently rewards the output style
# of digit/Latin engines (notably gigastt). Any extension MUST be disclosed in the
# "Word-error normalization" section of benchmark/README.md.
ANGLICISMS = {
    "synergy": "синергия", "tv": "тв", "pink": "пинк", "sony": "сони",
    "samsung": "самсунг", "apple": "эпл", "iphone": "айфон", "google": "гугл",
    "youtube": "ютуб", "facebook": "фейсбук", "instagram": "инстаграм",
    "netflix": "нетфликс", "spotify": "спотифай", "whatsapp": "ватсап",
    "telegram": "телеграм", "vk": "вк", "ok": "ок", "aliexpress": "алиэкспресс",
}


def _words_to_numbers(tokens: list[str]) -> list[str]:
    """Convert Russian number-word sequences into Arabic digit tokens.

    Compound numbers such as "две тысячи двадцать" become a single token
    "2020", while independent digit groups (e.g. phone-number chunks) are
    emitted separately based on scale-order jumps.
    """
    result: list[str] = []
    current = 0
    running_total = 0
    prev_scale = 0
    in_number = False

    def flush() -> None:
        nonlocal current, running_total, prev_scale, in_number
        total = running_total + current
        if in_number:
            result.append(str(total))
        current = 0
        running_total = 0
        prev_scale = 0
        in_number = False

    for token in tokens:
        num = _NUMBER_WORDS.get(token)
        if num is not None:
            in_number = True
            value, scale = num
            if scale in (_THOUSAND, _MILLION):
                if current == 0:
                    current = 1
                running_total += current * scale
                current = 0
                prev_scale = scale
            else:
                if current > 0 and scale >= prev_scale:
                    flush()
                    current = value
                    in_number = True
                else:
                    current += value
                prev_scale = scale
        else:
            flush()
            result.append(token)

    flush()
    return result


def _merge_digit_groups(tokens: list[str]) -> list[str]:
    """Merge adjacent digit tokens when every token has length <= 3."""
    result: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i].isdigit():
            j = i
            group: list[str] = []
            while j < n and tokens[j].isdigit():
                group.append(tokens[j])
                j += 1
            if group and all(len(t) <= 3 for t in group):
                result.append("".join(group))
            else:
                result.extend(group)
            i = j
        else:
            result.append(tokens[i])
            i += 1
    return result


# Short ordinal suffixes that commonly follow a digit ("36-я", "5-й", "3-го").
_ORDINAL_SUFFIXES = {
    "я", "й", "е", "го", "му", "ми", "ом", "ым", "их", "ыми",
}


def _drop_ordinal_suffixes(tokens: list[str]) -> list[str]:
    """Drop short ordinal suffixes that stand immediately after a digit token."""
    result: list[str] = []
    for i, token in enumerate(tokens):
        if i > 0 and tokens[i - 1].isdigit() and token in _ORDINAL_SUFFIXES:
            continue
        result.append(token)
    return result


def _strip_empty_tokens(tokens: list[str]) -> list[str]:
    """Drop symbol tokens and their Russian word equivalents."""
    result: list[str] = []
    for token in tokens:
        stripped = token.strip(_SYMBOL_STRIP)
        if stripped in _EMPTY_TOKENS:
            continue
        if stripped:
            result.append(stripped)
    return result


# Matches letter sequences (Latin or Cyrillic), digit sequences, or any
# single non-whitespace character (punctuation / symbol) as its own token.
_TOKEN_RE = re.compile(r"[a-zа-яё]+|\d+|[^a-zа-яё\d\s]", re.UNICODE)


def normalize_for_wer(text: str) -> list[str]:
    """Normalize text to word list for WER computation.

    Performs symmetric words-to-digits ITN on both reference and hypothesis,
    so Russian number words and Arabic digits become comparable tokens.
    """
    text = text.lower()
    text = text.replace("ё", "е")
    # Normalize various dash characters to spaces.
    text = re.sub(r"[-\u2010-\u2015\u2212]", " ", text)

    # Split letters, digits, and symbols/punctuation into separate tokens.
    # This keeps symbols as explicit separators during digit-group merging.
    tokens = _TOKEN_RE.findall(text)

    tokens = _words_to_numbers(tokens)
    tokens = _merge_digit_groups(tokens)
    tokens = _drop_ordinal_suffixes(tokens)
    tokens = _strip_empty_tokens(tokens)
    tokens = [ANGLICISMS.get(w, w) for w in tokens]
    return tokens


# Verbatim normalization keeps only word characters: letters (Latin or
# Cyrillic), digits, and whitespace. Everything else (punctuation, currency,
# percent, dashes) is removed without acting as a separator, matching the
# "naive" baseline described in benchmark/README.md.
_NAIVE_STRIP_RE = re.compile(r"[^a-zа-я0-9\s]", re.UNICODE)


def normalize_for_wer_naive(text: str) -> list[str]:
    """Normalize text to a word list with verbatim ("naive") rules only.

    Applies lowercasing, ``ё``→``е`` folding, removal of every non-word
    character, and whitespace splitting — but NONE of the words-to-digits ITN,
    digit-group merging, ordinal resolution, or anglicism mapping that
    :func:`normalize_for_wer` performs. The difference between the WER computed
    over this pass and the ITN pass isolates how much of the apparent error is a
    writing-convention mismatch (number style, punctuation, transliteration)
    rather than an acoustic recognition error.
    """
    text = text.lower().replace("ё", "е")
    text = _NAIVE_STRIP_RE.sub("", text)
    return text.split()


def word_edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance at word level."""
    m, n = len(reference), len(hypothesis)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def compute_wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Returns (wer_percent, errors, ref_word_count)."""
    ref_words = normalize_for_wer(reference)
    hyp_words = normalize_for_wer(hypothesis)
    errors = word_edit_distance(ref_words, hyp_words)
    ref_count = len(ref_words)
    wer = (errors / ref_count * 100.0) if ref_count > 0 else 0.0
    return wer, errors, ref_count


def compute_wer_naive(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Returns (wer_percent, errors, ref_word_count) under verbatim rules.

    Mirrors :func:`compute_wer` but uses :func:`normalize_for_wer_naive`, so no
    words-to-digits ITN or anglicism mapping is applied. Reported alongside the
    normalized WER to expose the writing-convention share of the error.
    """
    ref_words = normalize_for_wer_naive(reference)
    hyp_words = normalize_for_wer_naive(hypothesis)
    errors = word_edit_distance(ref_words, hyp_words)
    ref_count = len(ref_words)
    wer = (errors / ref_count * 100.0) if ref_count > 0 else 0.0
    return wer, errors, ref_count


_U64_MASK = (1 << 64) - 1


def bootstrap_ci(per_sample: list[tuple[int, int]], iterations: int = 1000) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for WER via resampling with replacement.

    Mirrors the implementation in crates/gigastt/tests/benchmark.rs so that
    Python and Rust benchmark suites produce comparable intervals.  Each tuple
    is (ref_word_count, errors) for a single sample; failures are represented
    as errors == ref_word_count (100% WER for that sample).
    """
    n = len(per_sample)
    if n == 0:
        return (0.0, 0.0)

    rng: int = 123456789
    wers: list[float] = []
    for _ in range(iterations):
        total_ref = 0
        total_err = 0
        for _ in range(n):
            rng = (rng * 6364136223846793005 + 1) & _U64_MASK
            idx = (rng >> 32) % n
            total_ref += per_sample[idx][0]
            total_err += per_sample[idx][1]
        wer = (total_err / total_ref * 100.0) if total_ref > 0 else 0.0
        wers.append(wer)

    wers.sort()
    lo = wers[(iterations * 25) // 1000]
    hi = wers[(iterations * 975) // 1000]
    return (lo, hi)


def audio_duration(wav_path: str) -> float:
    """Get duration in seconds using ffprobe or wave module."""
    try:
        import wave
        with wave.open(wav_path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return frames / rate
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# --- Reproducibility metadata helpers ---


def file_sha256(path: str) -> Optional[str]:
    """Return the SHA-256 hex digest of a file, or None if unavailable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def collect_host_metadata() -> dict:
    """Collect host hardware and OS metadata."""
    ram_bytes: Optional[int] = None
    try:
        import psutil

        ram_bytes = psutil.virtual_memory().total
    except Exception:
        pass

    return {
        "cpu": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "ram_bytes": ram_bytes,
        "os": platform.platform(),
        "python_version": platform.python_version(),
    }


def collect_dataset_metadata(
    dataset_name: str = "golos_crowd", version: Optional[str] = None
) -> dict:
    """Collect dataset source metadata from the manifest.

    Defaults to the Golos crowd subset by SberDevices.
    """
    path = manifest_path(dataset_name)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}

    if isinstance(data, list):
        # Legacy crowd manifest.
        return {
            "name": dataset_name,
            "version": version,
            "source": "https://github.com/sberdevices/golos",
            "attribution": "Golos by SberDevices",
            "license": "Sber Public License (attribution/non-commercial/share-alike)",
            "manifest_path": str(path),
        }

    return {
        "name": data.get("dataset", dataset_name),
        "version": version,
        "source": data.get("source", ""),
        "attribution": data.get("attribution", ""),
        "license": data.get("license", ""),
        "manifest_path": str(path),
        "slice_seed": data.get("slice_seed"),
        "slice_size": data.get("slice_size"),
        "total_available": data.get("total_available"),
    }


def collect_engine_metadata(runner) -> dict:
    """Collect engine name, binary/model paths, version, and model hashes."""
    meta: dict[str, object] = {"name": getattr(runner, "name", type(runner).__name__)}

    # Model identifiers
    for attr in ("model_dir", "model_name", "model_size", "download_dir"):
        val = getattr(runner, attr, None)
        if val is not None:
            meta[attr] = str(val)

    private_model_path = getattr(runner, "_model_path", None)
    if private_model_path is not None:
        meta["model_path"] = str(private_model_path)

    # Binary / version
    binary = getattr(runner, "_binary", None)
    if binary is not None:
        meta["binary"] = str(binary)
        try:
            result = subprocess.run(
                [str(binary), "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = (result.stdout + result.stderr).strip()
            if output:
                meta["version"] = output.splitlines()[0]
        except Exception:
            pass

    # Model hash (file or directory contents)
    model_path = private_model_path or getattr(runner, "model_dir", None)
    if model_path is None and hasattr(runner, "model_size"):
        # faster-whisper stores under cache by model_size
        model_path = Path.home() / ".cache" / "huggingface" / "hub"
    if model_path is not None:
        p = Path(model_path)
        if p.is_file():
            meta["model_sha256"] = file_sha256(str(p))
        elif p.is_dir():
            meta["model_path"] = str(p)
            # Hash the first ONNX/bin file found to give a stable fingerprint
            for candidate in sorted(p.rglob("*")):
                if candidate.is_file() and candidate.suffix in {".onnx", ".bin", ".pt", ".ckpt"}:
                    meta["model_sha256"] = file_sha256(str(candidate))
                    meta["model_hashed_file"] = str(candidate.relative_to(p))
                    break

    return meta


def collect_repro_metadata(
    runners: list,
    dataset_name: str = "golos_crowd",
    dataset_version: Optional[str] = None,
) -> dict:
    """Aggregate reproducibility metadata for a benchmark run."""
    return {
        "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": collect_host_metadata(),
        "dataset": collect_dataset_metadata(dataset_name, dataset_version),
        "engines": [collect_engine_metadata(r) for r in runners],
    }


# --- WER histograms ---------------------------------------------------------


HistogramBucket = dict[str, object]


_AUDIO_DURATION_BUCKETS: list[tuple[str, float | None, float | None]] = [
    ("0-5s", 0.0, 5.0),
    ("5-15s", 5.0, 15.0),
    ("15-30s", 15.0, 30.0),
    ("30s+", 30.0, None),
]


_REF_WORD_BUCKETS: list[tuple[str, float | None, float | None]] = [
    ("1-5 words", 1.0, 6.0),
    ("6-15 words", 6.0, 16.0),
    ("16-30 words", 16.0, 31.0),
    ("30+ words", 31.0, None),
]


_WER_BUCKETS: list[tuple[str, float | None, float | None]] = [
    ("0%", 0.0, 0.0),
    ("1-10%", 0.0, 10.0),
    ("10-20%", 10.0, 20.0),
    ("20-50%", 20.0, 50.0),
    ("50-100%", 50.0, 100.0),
    ("100%+", 100.0, None),
]


def _bucket_value(
    value: float,
    buckets: list[tuple[str, float | None, float | None]],
) -> str:
    """Return the matching bucket label for a value."""
    for label, low, high in buckets:
        # Degenerate bucket with identical bounds matches that exact value.
        if low is not None and high is not None and low == high:
            if value == low:
                return label
            continue
        if low is not None and value < low:
            return label
        if high is not None and value >= high:
            continue
        return label
    return buckets[-1][0]


def _empty_bucket_counts(
    buckets: list[tuple[str, float | None, float | None]],
) -> dict[str, dict[str, int]]:
    return {
        label: {"samples": 0, "ref_words": 0, "errors": 0}
        for label, _, _ in buckets
    }


def _build_histogram(
    details: list[dict],
    buckets: list[tuple[str, float | None, float | None]],
    value_key: str,
) -> list[HistogramBucket]:
    """Aggregate per-sample details into a histogram.

    ``value_key`` selects the numeric field used to assign each sample to a
    bucket (e.g. ``"audio_sec"``, ``"ref_words"``, ``"wer"``).
    """
    counts = _empty_bucket_counts(buckets)
    for d in details:
        value = d.get(value_key, 0.0)
        label = _bucket_value(value, buckets)
        counts[label]["samples"] += 1
        counts[label]["ref_words"] += d.get("ref_words", 0)
        counts[label]["errors"] += d.get("errors", 0)

    result: list[HistogramBucket] = []
    for label, low, high in buckets:
        c = counts[label]
        ref_words = c["ref_words"]
        errors = c["errors"]
        wer = (errors / ref_words * 100.0) if ref_words > 0 else 0.0
        entry: HistogramBucket = {
            "bucket": label,
            "samples": c["samples"],
            "ref_words": ref_words,
            "errors": errors,
            "wer": round(wer, 2),
        }
        if low is not None:
            entry["low_inclusive"] = low
        if high is not None:
            entry["high_exclusive"] = high
        result.append(entry)
    return result


def compute_histograms(details: list[dict]) -> dict[str, list[HistogramBucket]]:
    """Compute WER histograms by audio duration, reference length, and WER.

    Each histogram is a list of buckets ordered from easiest to hardest.
    Samples with ``failed=True`` are included in the WER bucket (they count as
    100% WER for that sample) so the failure rate is visible.
    """
    return {
        "audio_duration": _build_histogram(details, _AUDIO_DURATION_BUCKETS, "audio_sec"),
        "ref_words": _build_histogram(details, _REF_WORD_BUCKETS, "ref_words"),
        "wer": _build_histogram(details, _WER_BUCKETS, "wer"),
    }
