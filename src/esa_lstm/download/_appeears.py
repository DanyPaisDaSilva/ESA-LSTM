"""
Small client for NASA AppEEARS (Application for Extracting and Exploring Analysis
Ready Samples): you send an "area request" (products, layers, dates, box), NASA crops
the data on its servers, and you download the finished files.

Used by modis_ndvi.py and ecostress.py.
Needs a free NASA Earthdata login, read from EARTHDATA_USER and EARTHDATA_PASS
(put them in .env and run with `uv run --env-file .env`, see docs/setup.md).

Typical use:
    api = AppEEARS(); api.login()
    task_id = api.submit_or_resume(area_task(...), out_dir)
    api.wait(task_id)
    api.download_bundle(task_id, out_dir)
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

from esa_lstm.download._http import download_file

API = "https://appeears.earthdatacloud.nasa.gov/api/"


# ---------------------------------------------------------------- building a request
def area_task(name, layers, dates, bbox):
    """An AppEEARS area request. layers: list of (product, layer); dates: see date_range()."""
    lon_min, lat_min, lon_max, lat_max = bbox
    ring = [[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max],
            [lon_min, lat_max], [lon_min, lat_min]]
    box = {"type": "FeatureCollection",
           "features": [{"type": "Feature", "properties": {},
                         "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    return {
        "task_type": "area",
        "task_name": name,
        "params": {
            "dates": dates,
            "layers": [{"product": p, "layer": l} for p, l in layers],
            # GeoTIFFs in plain lon/lat, so all datasets share one coordinate system
            "output": {"format": {"type": "geotiff"}, "projection": "geographic"},
            "geo": box,
        },
    }


def date_range(start_year, end_year, start_mmdd="01-01", end_mmdd="12-31"):
    """One continuous period, e.g. 01-01-2000 .. 12-31-2025."""
    return [{"startDate": f"{start_mmdd}-{start_year}", "endDate": f"{end_mmdd}-{end_year}"}]


# ---------------------------------------------------------------- talking to the API
class AppEEARS:
    def __init__(self):
        self.session = requests.Session()  # carries the login token once login() ran

    def login(self):
        user, pw = os.environ.get("EARTHDATA_USER"), os.environ.get("EARTHDATA_PASS")
        if not user or not pw:
            sys.exit("Set EARTHDATA_USER and EARTHDATA_PASS environment variables first.")
        r = requests.post(API + "login", auth=(user, pw), timeout=60)
        if r.status_code in (400, 401, 403):
            sys.exit(f"Earthdata login rejected (HTTP {r.status_code}). Check your credentials.")
        r.raise_for_status()
        self.session.headers["Authorization"] = f"Bearer {r.json()['token']}"

    def _get(self, path):
        r = self.session.get(API + path, timeout=120)
        if r.status_code == 401 and "Authorization" in self.session.headers:
            self.login()  # tokens expire after ~48 h; log in again and retry once
            r = self.session.get(API + path, timeout=120)
        r.raise_for_status()
        return r.json()

    def layers(self, product):
        """Layer names offered for a product, e.g. layers("MOD13Q1.061")."""
        return set(self._get(f"product/{product}").keys())

    def check_layers(self, layers):
        """Stop early, with the valid names, if a product or layer name is wrong."""
        products = {p["ProductAndVersion"] for p in self._get("product")}
        problems = []
        for product, layer in layers:
            if product not in products:
                problems.append(f"Product {product} is not offered by AppEEARS.")
            elif layer not in self.layers(product):
                problems.append(f"Layer '{layer}' not in {product}. Available: {sorted(self.layers(product))}")
        if problems:
            sys.exit("\n".join(problems + ["Fix the names above (see --layers) and run again."]))

    def submit_or_resume(self, task, out_dir, new_task=False):
        """Submit the request once and save its id, so re-running resumes instead of resubmitting."""
        out_dir = Path(out_dir)
        id_file, task_file = out_dir / "appeears_task_id.txt", out_dir / "appeears_task.json"
        if id_file.exists() and not new_task:
            saved = json.loads(task_file.read_text())["params"]
            if saved != json.loads(json.dumps(task["params"])):  # json round trip: tuples -> lists
                sys.exit(f"The saved AppEEARS task in {out_dir} was for a different box, dates or layers.\n"
                         "Add --new-task to submit this request, or use a different --out folder.")
            task_id = id_file.read_text().strip()
            print(f"Resuming AppEEARS task {task_id} (use --new-task to submit a fresh one)")
            return task_id
        r = self.session.post(API + "task", json=task, timeout=120)
        if r.status_code >= 400:
            sys.exit(f"Task rejected (HTTP {r.status_code}): {r.text}")
        task_id = r.json()["task_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        id_file.write_text(task_id)
        task_file.write_text(json.dumps(task, indent=2))
        print(f"Submitted AppEEARS task {task_id}")
        return task_id

    def wait(self, task_id, poll_s=60):
        """Wait until NASA has processed the request (minutes to hours)."""
        last = None
        while True:
            status = self._get(f"task/{task_id}").get("status")
            if status != last:
                print(f"  task status: {status}", flush=True)
                last = status
            if status == "done":
                return
            if status in ("error", "expired", "deleted"):
                sys.exit(f"AppEEARS task ended with status '{status}'. Check it on the AppEEARS website.")
            time.sleep(poll_s)

    def download_bundle(self, task_id, out_dir):
        """Download every file of a finished request; files already on disk are skipped."""
        files = self._get(f"bundle/{task_id}")["files"]
        print(f"{len(files)} files in the result bundle")
        for i, f in enumerate(files, 1):
            url = f"{API}bundle/{task_id}/{f['file_id']}"
            try:
                download_file(url, Path(out_dir) / f["file_name"], session=self.session)
            except (requests.RequestException, OSError) as e:
                print(f"  ! failed {f['file_name']}: {e} (re-run to retry)")
            if i % 50 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] done", flush=True)
