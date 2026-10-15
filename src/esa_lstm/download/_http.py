"""
The one download helper every downloader uses: fetch a URL into a file, with retries.

A file is first written as "<name>.part" and only renamed when complete, so an
interrupted run never leaves a half-written file that looks finished. Files that
already exist are skipped, which is what makes every downloader resumable.
"""
import time

import requests


class LoginError(Exception):
    """The server refused our credentials: retrying will not help, so stop the run."""


def download_file(url, dest, session=None, retries=3, timeout=(15, 300)):
    """Download `url` to `dest` (a Path). Returns "cached", "ok" or "missing" (HTTP 404).

    session: a requests.Session carrying a login (optional).
    """
    if dest.exists():
        return "cached"
    get = session.get if session else requests.get
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    return "missing"
                if r.status_code in (401, 403):
                    raise LoginError(f"HTTP {r.status_code}: login rejected for {url}")
                r.raise_for_status()
                if "text/html" in r.headers.get("Content-Type", ""):
                    raise LoginError(f"Got a web page instead of a data file (a login page?) for {url}")
                with open(part, "wb") as f:
                    f.writelines(r.iter_content(1 << 20))
            part.replace(dest)
            return "ok"
        except (requests.RequestException, OSError):
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)  # wait 2 s, 4 s, ... before retrying
