from ndvi_service.search import search_day
from ndvi_service.hls import build_farm_cog_bytes_per_index
from ndvi_service.indices import TARGET_INDICES
import os, time, glob, json
import pandas as pd

MISS_EXT = ".miss.json"  # tiny JSON with a reason + timestamp

def _is_fresh(path: str, ttl_seconds: int) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path) <= ttl_seconds)

def _miss_path(out_dir: str, day_str: str) -> str:
    return os.path.join(out_dir, f"{day_str}{MISS_EXT}")

def build_and_save_farm_day(farm_gdf, day_str, out_dir, use_fmask=True, count_per_day=20) -> list[str] | None:
    os.makedirs(out_dir, exist_ok=True)
    granules = search_day(tuple(farm_gdf.total_bounds), day_str, count=count_per_day)

    cog_map = build_farm_cog_bytes_per_index(granules, farm_gdf, use_fmask=use_fmask)  # { "NDVI": bytes, "EVI": bytes, ... }
    if not cog_map:
        # sentinel for "no valid data that day"
        open(os.path.join(out_dir, f"{day_str}.miss"), "a").close()
        return None

    out_paths = []
    for idx in TARGET_INDICES:
        data = cog_map.get(idx)
        if not data:
            continue
        fpath = os.path.join(out_dir, f"{day_str}_{idx}_cropped.tif")
        with open(fpath, "wb") as f:
            f.write(data)
        out_paths.append(fpath)
    return out_paths

def ensure_range_days_on_disk(farm_gdf, start_str, end_str, out_dir, ttl_seconds, count_per_day=20, canonical_idx="NDVI"):
    os.makedirs(out_dir, exist_ok=True)
    purge_older_than(out_dir, ttl_seconds, patterns=("*.tif","*.csv","*.miss"))
    start = pd.to_datetime(start_str); end = pd.to_datetime(end_str)
    day = start
    while day <= end:
        d = day.strftime("%Y-%m-%d")
        marker = os.path.join(out_dir, f"{d}.miss")
        fpath  = os.path.join(out_dir, f"{d}_{canonical_idx}_cropped.tif")
        fresh  = os.path.exists(fpath) and (pd.Timestamp.now().timestamp() - os.path.getmtime(fpath) <= ttl_seconds)
        missed = os.path.exists(marker) and (pd.Timestamp.now().timestamp() - os.path.getmtime(marker) <= ttl_seconds)
        if not (fresh or missed):
            build_and_save_farm_day(farm_gdf, d, out_dir, count_per_day=count_per_day)
        day += pd.Timedelta(days=1)
    return out_dir


# Helper for clearing cache:
def purge_older_than(dirpath: str, ttl_seconds: int, patterns=("*.tif", "*.csv")):
    now = time.time()
    for pat in patterns:
        for p in glob.glob(os.path.join(dirpath, pat)):
            try:
                if now - os.path.getmtime(p) > ttl_seconds:
                    os.remove(p)
            except OSError:
                pass
