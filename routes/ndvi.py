from __future__ import annotations

import os
import rasterio as rio
import pandas as pd
import geopandas as gpd
from flask import Blueprint, request, jsonify
from typing import Optional, List
from routes.classes import BaseGeo, RangeRequest
from pydantic import ValidationError
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon, box
from ndvi_service.config import (
    CACHE_DIR, COUNT_PER_DAY, CACHE_TTL_SECONDS, rasterio_env_kwargs
)
from ndvi_service.indices import TARGET_INDICES, INDEX_REQUIREMENTS
from ndvi_service.search import login
from ndvi_service.search import search_day
from ndvi_service.hls import build_mask_payloads_and_stats_for_day

ndvi_bp = Blueprint("ndvi", __name__)

def _maybe_swap_latlon(x, y):
    """
    Return (lon, lat). If input looks like (lat, lon) (second magnitude > 90),
    swap them. Accepts strings/numbers.
    """
    x = float(x); y = float(y)
    # Expected: lon in [-180,180], lat in [-90,90]
    if abs(x) <= 90 and abs(y) > 90:
        # looks like (lat, lon)
        return y, x
    return x, y  # already (lon, lat) or ambiguous-but-valid

def _normalize_ring(ring_pairs):
    """
    Normalize a ring of pairs into a valid Polygon.
    - Accept [lon,lat] or [lat,lon] pairs; swap when needed.
    - Ensure closed ring.
    """
    pts = []
    for p in ring_pairs:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            raise ValueError("Each coordinate must be a [x,y] pair.")
        lon, lat = _maybe_swap_latlon(p[0], p[1])
        # basic bounds check
        if abs(lon) > 180 or abs(lat) > 90:
            raise ValueError(f"Coordinate out of range: ({lon}, {lat})")
        pts.append((lon, lat))

    if pts[0] != pts[-1]:
        pts.append(pts[0])

    poly = Polygon(pts)
    if not poly.is_valid:
        raise ValueError("Provided coordinates do not form a valid polygon.")
    return poly

def _normalize_bbox4(a, b, c, d):
    """
    Accept 4-number bbox in either order:
      - [minLon, minLat, maxLon, maxLat]  (standard)
      - [minLat, minLon, maxLat, maxLon]  (swapped)
    Decide by looking at magnitudes (|lon| can be >90, |lat| ≤90).
    """
    a = float(a); b = float(b); c = float(c); d = float(d)

    # If a/c look like lats and b/d look like lons, swap interpretation.
    if max(abs(a), abs(c)) <= 90 and max(abs(b), abs(d)) > 90:
        minLat, maxLat = sorted([a, c])
        minLon, maxLon = sorted([b, d])
    else:
        # assume a/c are lons and b/d are lats
        minLon, maxLon = sorted([a, c])
        minLat, maxLat = sorted([b, d])

    if any(abs(x) > 180 for x in (minLon, maxLon)):
        raise ValueError("Longitude out of range in bbox.")
    if any(abs(y) > 90 for y in (minLat, maxLat)):
        raise ValueError("Latitude out of range in bbox.")

    return box(minLon, minLat, maxLon, maxLat)

def gdf_from_body(body: BaseGeo, crs="EPSG:4326") -> gpd.GeoDataFrame:
    # 1) GeoJSON
    if body.geojson is not None:
        # We assume incoming GeoJSON follows lon,lat per spec.
        # (If you want to “auto-fix” bad GeoJSON too, that’s possible but more invasive.)
        if "type" in body.geojson and body.geojson["type"] == "FeatureCollection":
            return gpd.GeoDataFrame.from_features(body.geojson, crs=crs)
        return gpd.GeoDataFrame.from_features([body.geojson], crs=crs)

    # 2) WKT
    if body.wkt is not None:
        geom = shapely_wkt.loads(body.wkt)
        return gpd.GeoDataFrame({"name": ["geom"]}, geometry=[geom], crs=crs)

    # 3) bbox: either 4-number bbox OR ring of pairs
    if body.bbox is not None:
        # 3a) 4-number bbox
        if isinstance(body.bbox, (list, tuple)) and len(body.bbox) == 4 and all(
            isinstance(v, (int, float, str)) for v in body.bbox
        ):
            geom = _normalize_bbox4(*body.bbox)
            return gpd.GeoDataFrame({"name": ["bbox"]}, geometry=[geom], crs=crs)

        # 3b) bbox ring: [[x1,y1], [x2,y2], ...]
        if isinstance(body.bbox, (list, tuple)) and body.bbox and isinstance(body.bbox[0], (list, tuple)):
            poly = _normalize_ring(body.bbox)
            return gpd.GeoDataFrame({"name": ["polygon"]}, geometry=[poly], crs=crs)

        raise ValueError("bbox must be either [minx,miny,maxx,maxy] or [[x,y], ...]")

    # 4) coords: explicit polygon ring
    if body.coords is not None:
        poly = _normalize_ring(body.coords)
        return gpd.GeoDataFrame({"name": ["polygon"]}, geometry=[poly], crs=crs)

    raise ValueError("Provide one of: geojson, wkt, bbox, coords")

def _normalize_indices(requested: Optional[List[str]]) -> List[str]:
    if not requested:
        return list(TARGET_INDICES)
    tgt = set(TARGET_INDICES)
    out = []
    seen = set()
    for s in requested:
        k = s.upper()
        if k in tgt and k not in seen:
            out.append(k); seen.add(k)
    return out


@ndvi_bp.route("/indices/range", methods=["POST"])
def indices_range():
    try:
        model = RangeRequest(**(request.get_json(force=True) or {}))
    except ValidationError as e:
        return jsonify({"error": "invalid_request", "detail": e.errors()}), 400

    farm = gdf_from_body(model)
    res = run(
        farm_gdf=farm,
        start_date=model.start,
        end_date=model.end,
        count_per_day=model.count_per_day,
        indices=_normalize_indices(model.indices),
    )
    return jsonify(res), 200

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
    (MODIFIED)
    - Builds day-by-day, per-index payloads (compressed bytes) + stats in-memory (no file writes)
    - Leaves previous cache write/read code commented for easy rollback
    """
    # --- imports needed for the new in-memory path ---
    
    # from ndvi_service.day_build import build_and_save_farm_day, ensure_range_days_on_disk  # (kept for rollback)

    # Sanity: INDEX_REQUIREMENTS must use aliases ('nir','red',...)
    expects_aliases = any(k in ("nir","red","blue","green") for s in INDEX_REQUIREMENTS.values() for k in s)
    if not expects_aliases:
        raise RuntimeError("indices.INDEX_REQUIREMENTS must use aliases like {'nir','red'}.")

    # Only compute indices that are both requested and in TARGET_INDICES
    indices_req = list(indices or TARGET_INDICES)
    indices = [i for i in indices_req if i in TARGET_INDICES]

    # Earthdata auth (persisted)
    login()

    env_kwargs = rasterio_env_kwargs()
    os.makedirs(cache_dir, exist_ok=True)

    with rio.Env(**env_kwargs):
        # ----------------------------
        # COMMENT OUT: disk writes
        # ----------------------------
        # if refresh_today:
        #     today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        #     build_and_save_farm_day(
        #         farm_gdf, today, cache_dir, use_fmask=True, count_per_day=count_per_day
        #     )
        #
        # # Fill the range into cache (per-day granules, 1-hour TTL)
        # ensure_range_days_on_disk(
        #     farm_gdf, start_date, end_date, cache_dir,
        #     ttl_seconds=ttl_seconds, count_per_day=count_per_day
        # )

        # ----------------------------
        # NEW: in-memory bytes workflow (no file IO)
        # ----------------------------
        start_dt = pd.to_datetime(start_date)
        end_dt   = pd.to_datetime(end_date)

        days = []
        cur = start_dt
        while cur <= end_dt:
            day_str = cur.strftime("%Y-%m-%d")

            # Find intersecting granules for this day (no writes)
            granules_links = search_day(farm_gdf, day_str, count_per_day=count_per_day)

            # Build bytes payloads + stats per index
            per_day = build_mask_payloads_and_stats_for_day(
                granules_links,
                farm_gdf=farm_gdf,
                indices=indices,
                use_fmask=True,   # keep your default masking behavior
                quantize=True,    # send compact int16 by default
                scale=10000
            )

            days.append({"date": day_str, "per_index": per_day})
            cur += pd.Timedelta(days=1)

    # ----------------------------
    # COMMENT OUT: stats CSV write & file list summary
    # ----------------------------
    # start_ts = pd.to_datetime(start_date)
    # end_ts   = pd.to_datetime(end_date)
    # chunks   = dict(band=1, x=512, y=512)
    #
    # for idx in indices:
    #     idx_files = _list_index_files_in_range(cache_dir, idx, start_ts, end_ts)
    #     if not idx_files:
    #         summary["per_index"][idx] = {
    #             "files": [],
    #             "stats_csv": None,
    #             "n_scenes": 0,
    #             "note": f"No {idx} COGs available in cache for the requested range."
    #         }
    #         continue
    #
    #     time_coord = xr.Variable("time", time_index_from_filenames(idx_files))
    #     ts = xr.concat([_load_band1(f, chunks) for f in idx_files], dim=time_coord)
    #     ts.name = idx
    #
    #     df = _stats_dataframe(ts, idx)
    #     stats_csv = os.path.join(cache_dir, f"{idx.lower()}_stats_{start_date}_{end_date}.csv")
    #     df.to_csv(stats_csv, index=True)
    #     summary["per_index"][idx] = {
    #         "files": idx_files,
    #         "stats_csv": stats_csv,
    #         "n_scenes": len(idx_files)
    #     }

    summary = {
        "message": "OK",
        "start": start_date,
        "end": end_date,
        "indices": indices,
        "days": days
    }
    return summary
