"""Disk cache for benchmark transcription results.

The cache lets repeated benchmark runs skip re-transcribing files that have
already been processed by the same runner configuration.  This is the biggest
practical speed-up for iterative benchmark development: after the first full
run, subsequent runs only compute WER on cached hypotheses.
"""

import hashlib
import json
import os
import re
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from common import file_sha256


class DiskCache:
    """Simple JSON-on-disk cache with atomic writes and SHA256-derived keys."""

    def __init__(self, cache_dir: Path | str, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, runner: Any, wav_path: str) -> str:
        """Build a stable cache key for a runner + audio file pair."""
        parts = [getattr(runner, "name", type(runner).__name__)]
        config = getattr(runner, "cache_config", None)
        if config:
            parts.append(str(config))
        parts.append(file_sha256(wav_path) or wav_path)
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, runner: Any, wav_path: str) -> Optional[dict[str, Any]]:
        """Return cached result dict or None."""
        if not self.enabled:
            return None
        path = self._path(self._key(runner, wav_path))
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # Corrupt cache entry is treated as a miss.
            return None

    def set(
        self,
        runner: Any,
        wav_path: str,
        hypothesis: str,
        proc_time: float,
    ) -> None:
        """Atomically write a cache entry."""
        if not self.enabled:
            return
        key = self._key(runner, wav_path)
        path = self._path(key)
        entry = {
            "hypothesis": hypothesis,
            "proc_time": round(proc_time, 6),
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runner": getattr(runner, "name", type(runner).__name__),
            "config": getattr(runner, "cache_config", None),
            "wav_sha256": file_sha256(wav_path),
        }
        # Atomic write: write to temp in same directory, then rename.
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def clear(self) -> int:
        """Remove recognizable cache entries. Returns number of files removed."""
        try:
            paths = list(self.cache_dir.iterdir())
        except FileNotFoundError:
            return 0
        removed = 0
        for path in paths:
            if not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
                continue
            try:
                if not stat.S_ISREG(path.lstat().st_mode):
                    continue
                with path.open(encoding="utf-8") as handle:
                    entry = json.load(handle)
                # Keep ambiguous files, even when they have a digest-like name.
                if not isinstance(entry, dict) or entry.keys() != {
                    "hypothesis", "proc_time", "cached_at", "runner", "config", "wav_sha256",
                }:
                    continue
                path.unlink()
            except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            removed += 1
        return removed
