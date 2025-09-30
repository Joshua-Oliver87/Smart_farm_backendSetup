import os
import json
import argparse
import rasterio as rio
import rioxarray as rxr
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely import wkt as shapely_wkt

from ndvi_service.config import (
    CACHE_DIR, COUNT_PER_DAY, CACHE_TTL_SECONDS, rasterio_env_kwargs
)
from ndvi_service.day_build import build_and_save_farm_day, ensure_range_days_on_disk
from ndvi_service.indices import TARGET_INDICES, INDEX_REQUIREMENTS
from ndvi_service.helper import time_index_from_filenames
from ndvi_service.search import login

# -------------------------
# Helpers to build a GeoDataFrame from user input
# -------------------------
def gdf_from_geojson_path(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path)

def gdf_from_wkt(wkt_str: str, crs="EPSG:4326") -> gpd.GeoDataFrame:
    geom = shapely_wkt.loads(wkt_str)
    return gpd.GeoDataFrame({"name": ["geom"]}, geometry=[geom], crs=crs)

def gdf_from_bbox_str(bbox_csv: str, crs="EPSG:4326") -> gpd.GeoDataFrame:
    # "minx,miny,maxx,maxy"
    minx, miny, maxx, maxy = [float(v.strip()) for v in bbox_csv.split(",")]
    geom = box(minx, miny, maxx, maxy)
    return gpd.GeoDataFrame({"name": ["bbox"]}, geometry=[geom], crs=crs)

def gdf_from_coords(coords: list, crs="EPSG:4326") -> gpd.GeoDataFrame:
    # coords = [[x1,y1], [x2,y2], ...]
    poly = Polygon(coords)
    return gpd.GeoDataFrame({"name": ["polygon"]}, geometry=[poly], crs=crs)

# -------------------------
# IO helpers
# -------------------------
def _list_index_files_in_range(cache_dir: str, idx: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    pattern = f"_{idx}_cropped.tif"
    all_files = [
        os.path.join(cache_dir, o)
        for o in os.listdir(cache_dir)
        if o.endswith(pattern)
    ]
    all_times = time_index_from_filenames(all_files)
    filtered = [(f, t) for f, t in zip(all_files, all_times) if (t >= start_ts and t <= end_ts)]
    filtered.sort(key=lambda ft: ft[1])
    return [f for f, _ in filtered]

def _load_band1(path: str, chunks: dict) -> xr.DataArray:
    # All per-index COGs are single-band float32 with nodata=-9999
    da = rxr.open_rasterio(path, mask_and_scale=True, chunks=chunks)
    if "band" in da.dims:
        if da.sizes.get("band", 1) == 1:
            return da.squeeze("band", drop=True)
        # Fallback: if somehow multiband, take band 1
        return da.isel(band=0).squeeze(drop=True)
    return da.squeeze(drop=True)

def _stats_dataframe(ts: xr.DataArray, idx_name: str) -> pd.DataFrame:
    ts = ts.sortby("time")
    dmin    = ts.min(("y","x"))
    dmax    = ts.max(("y","x"))
    dmean   = ts.mean(("y","x"))
    dsd     = ts.std(("y","x"))
    dcount  = ts.count(("y","x"))
    dmedian = ts.median(("y","x"))

    df = pd.DataFrame({
        f"Min {idx_name}":              dmin.values,
        f"Max {idx_name}":              dmax.values,
        f"Mean {idx_name}":             dmean.values,
        f"Standard Deviation {idx_name}": dsd.values,
        f"Median {idx_name}":           dmedian.values,
        "Count":                        dcount.values,
    })
    df.index = pd.to_datetime(ts.time.values)
    return df

# -------------------------
# Core runner
# -------------------------
def run(
    farm_gdf: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    refresh_today: bool = True,
    cache_dir: str = CACHE_DIR,
    count_per_day: int = COUNT_PER_DAY,
    ttl_seconds: int = CACHE_TTL_SECONDS,
    indices: list[str] | None = None,
) -> dict:
    """
    - Ensures (and optionally refreshes) per-day per-index COGs in cache for [start_date, end_date]
    - Builds a time series per requested index from cached COGs
    - Computes summary stats per index and writes per-index CSVs into cache
    - Returns paths + quick summary
    """
    # Sanity: INDEX_REQUIREMENTS must use aliases ('nir','red',...)
    expects_aliases = any(k in ("nir","red","blue","green") for s in INDEX_REQUIREMENTS.values() for k in s)
    if not expects_aliases:
        raise RuntimeError("indices.INDEX_REQUIREMENTS must use aliases like {'nir','red'}.")

    indices = list(indices or TARGET_INDICES)

    # Earthdata auth (persisted)
    login()

    env_kwargs = rasterio_env_kwargs()
    os.makedirs(cache_dir, exist_ok=True)

    with rio.Env(**env_kwargs):
        if refresh_today:
            today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
            build_and_save_farm_day(
                farm_gdf, today, cache_dir, use_fmask=True, count_per_day=count_per_day
            )

        # Fill the range into cache (per-day granules, 1-hour TTL)
        ensure_range_days_on_disk(
            farm_gdf, start_date, end_date, cache_dir,
            ttl_seconds=ttl_seconds, count_per_day=count_per_day
        )

    # ---- Build timeseries and stats per index ----
    start_ts = pd.to_datetime(start_date)
    end_ts   = pd.to_datetime(end_date)
    chunks   = dict(band=1, x=512, y=512)

    summary = {
        "message": "OK",
        "start": start_date,
        "end": end_date,
        "indices": indices,
        "per_index": {}
    }

    for idx in indices:
        idx_files = _list_index_files_in_range(cache_dir, idx, start_ts, end_ts)
        if not idx_files:
            summary["per_index"][idx] = {
                "files": [],
                "stats_csv": None,
                "n_scenes": 0,
                "note": f"No {idx} COGs available in cache for the requested range."
            }
            continue

        print(f"[ndvi_service] [{idx}] Found {len(idx_files)} files in cache for {start_date}..{end_date}")

        time_coord = xr.Variable("time", time_index_from_filenames(idx_files))
        ts = xr.concat([_load_band1(f, chunks) for f in idx_files], dim=time_coord)
        ts.name = idx

        df = _stats_dataframe(ts, idx)
        stats_csv = os.path.join(cache_dir, f"{idx.lower()}_stats_{start_date}_{end_date}.csv")
        df.to_csv(stats_csv, index=True)
        print(f"[ndvi_service] [{idx}] Wrote stats CSV: {stats_csv}")

        summary["per_index"][idx] = {
            "files": idx_files,
            "stats_csv": stats_csv,
            "n_scenes": len(idx_files)
        }

    return summary

# -------------------------
# CLI entrypoint
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Per-index per-day cache + timeseries builder")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--geojson", type=str, help="Path to a GeoJSON file for the farm geometry")
    src.add_argument("--wkt", type=str, help="WKT polygon")
    src.add_argument("--bbox", type=str, help="Bounding box 'minx,miny,maxx,maxy'")
    src.add_argument("--coords", type=str, help="JSON array of [x,y] vertices (lon/lat), e.g. [[-118,46.19], ...]")

    p.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end",   type=str, required=True, help="End date YYYY-MM-DD")
    p.add_argument("--indices", type=str, default=None,
                   help="Comma-separated list of indices to compute/stack timeseries for (default: indices.TARGET_INDICES)")
    p.add_argument("--no-today", action="store_true", help="Do not force-refresh today's COG")
    p.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    p.add_argument("--ttl-seconds", type=int, default=CACHE_TTL_SECONDS)
    p.add_argument("--count-per-day", type=int, default=COUNT_PER_DAY)
    return p.parse_args()

def main():
    args = parse_args()

    # Build farm GeoDataFrame
    if args.geojson:
        farm = gdf_from_geojson_path(args.geojson)
    elif args.wkt:
        farm = gdf_from_wkt(args.wkt)
    elif args.bbox:
        farm = gdf_from_bbox_str(args.bbox)
    elif args.coords:
        coords = json.loads(args.coords)
        farm = gdf_from_coords(coords)
    else:
        raise SystemExit("Provide one of --geojson, --wkt, --bbox, or --coords")

    indices = [s.strip() for s in args.indices.split(",")] if args.indices else None

    summary = run(
        farm_gdf=farm,
        start_date=args.start,
        end_date=args.end,
        refresh_today=(not args.no_today),
        cache_dir=args.cache_dir,
        count_per_day=args.count_per_day,
        ttl_seconds=args.ttl_seconds,
        indices=indices,
    )
    print(json.dumps(summary, default=str, indent=2))

if __name__ == "__main__":
    main()
