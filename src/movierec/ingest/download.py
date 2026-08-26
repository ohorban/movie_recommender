"""Resumable large-file downloads for the bulk datasets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import requests

from ..logging_utils import get_logger

log = get_logger("ingest.download")

ProgressFn = Callable[[str, float], None]


def download(
    url: str,
    dest: Path,
    *,
    label: str = "",
    progress: ProgressFn | None = None,
    progress_span: tuple[float, float] = (0.0, 1.0),
    chunk: int = 1 << 20,
    timeout: float = 60.0,
) -> Path:
    """Download ``url`` to ``dest``, resuming a partial file when possible."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    label = label or dest.name

    if dest.exists() and dest.stat().st_size > 0:
        log.info("%s already downloaded (%.1f MB)", label, dest.stat().st_size / 1e6)
        return dest

    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    lo, hi = progress_span

    with requests.get(url, stream=True, headers=headers, timeout=timeout) as resp:
        if resp.status_code == 416:  # already complete
            part.rename(dest)
            return dest
        resp.raise_for_status()
        # A server that ignores Range restarts the file.
        if have and resp.status_code != 206:
            have = 0
        total = int(resp.headers.get("Content-Length", 0)) + have
        mode = "ab" if have and resp.status_code == 206 else "wb"
        written = have
        with open(part, mode) as fh:
            for block in resp.iter_content(chunk_size=chunk):
                if not block:
                    continue
                fh.write(block)
                written += len(block)
                if progress and total:
                    frac = min(1.0, written / total)
                    progress(
                        f"Downloading {label} · {written / 1e6:.0f}/{total / 1e6:.0f} MB",
                        lo + (hi - lo) * frac,
                    )

    part.rename(dest)
    log.info("downloaded %s (%.1f MB)", label, dest.stat().st_size / 1e6)
    return dest
