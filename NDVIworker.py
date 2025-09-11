#!/usr/bin/env python
"""
Fetch high-res NDVI (HLS) for a batch of bounding boxes and a single date.
- Unions all bboxes into one AOI (to minimize NASA queries/downloads)
- Searches HLSS30 (Sentinel-2) + HLSL30 (Landsat 8/9) for that AOI/date
- Downloads intersecting tiles
- Computes NDVI per tile, optionally applies QA (Fmask) to drop clouds/shadow/snow
- Reprojects to EPSG:4326, mosaics tiles, clips mosaic to AOI bbox
- Writes a single GeoTIFF (tiled/deflate) to --outdir
- Prints JSON to stdout: {"aoi_tif": "...", "tiles_used": N, "granule_count": M, "granule_ids":[...]}

Usage:
  python NDVIworker.py \
    --bboxes_json '[[minx,miny,maxx,maxy],[minx,miny,maxx,maxy]]' \
    --date 2025-08-01 \
    --outdir /path/to/output \
    --apply_qa --qa_keep_water \
    --edl_user YOUR_USERNAME --edl_pass YOUR_PASSWORD

NOTE: This script intentionally does NOT clip per input bbox yet.
"""

import argparse, os, sys, json, glob, tempfile
from datetime import date as date_cls
import numpy as np

import earthaccess
from shapely.geometry import box
from shapely.ops import unary_union
import rioxarray as rxr
import xarray as xr
import zipfile, shutil
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from pathlib import Path
import re
from collections import defaultdict
from shapely.geometry import Polygon, mapping
import math
import rasterio
from contextlib import contextmanager
from datetime import date as date_cls, timedelta

STEM_RE = re.compile(r'^(HLS\.[SL]30\.[^.]+\.\d{7}T\d{6}\.v[0-9.]+)\.', re.IGNORECASE)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


OUT_NODATA = -9999.0

def deg_res_for_meters(target_m, lat_hint):
    dlat = target_m / 110540.0
    dlon = target_m / (111320.0 * max(0.1, math.cos(math.radians(lat_hint))))
    return (dlon, dlat)


#-------For COG support-------

@contextmanager
def _cog_env():
    """
    Makes GDAL behave well with COGs over HTTP(S).
    Safe to use for local files too.
    """
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",  # avoid directory reads on HTTP
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.TIF,.TIFF",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="10000000"  # ~10 MB
    ):
        yield

def open_rioxarray(path, **kwargs):
    """
    Open either a local .tif or an http(s) COG URL with rioxarray.
    """
    with _cog_env():
        return rxr.open_rasterio(path, masked=True, **kwargs)
    
# ---------------------Helpers--------------------

def load_polys_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect dict: { "<id>": [[lon,lat], ... closed ring], ... }
    polys = []
    for fid, ring in data.items():
        if not (isinstance(ring, list) and len(ring) >= 4):
            raise ValueError(f"Field {fid} ring invalid")
        # ensure it’s closed
        if ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        # build polygon
        poly = Polygon(ring)
        if not poly.is_valid:
            raise ValueError(f"Field {fid} polygon invalid")
        polys.append(poly)
    return polys  # list[Polygon]


# For getting union of multiple polygons
def aoi_union_from_fields(polys):
    if not polys:
        raise ValueError("No polygons provided")
    return unary_union(polys)

# ----------Pad bbox in meters ----------
def pad_bbox_meters(bbox, pad_m, lat_hint=None):
    """
    bbox = [minx, miny, maxx, maxy] in lon/lat (EPSG:4326)
    pad_m in meters. Converts to deg using latitude.
    """
    minx, miny, maxx, maxy = bbox
    if lat_hint is None:
        lat_hint = (miny + maxy) / 2.0
    # rough meters->deg conversions
    deg_lat_per_m = 1.0 / 110540.0   # ~110.54 km per deg lat
    deg_lon_per_m = 1.0 / (111320.0 * max(0.1, math.cos(math.radians(lat_hint))))
    dlat = pad_m * deg_lat_per_m
    dlon = pad_m * deg_lon_per_m
    return [minx - dlon, miny - dlat, maxx + dlon, maxy + dlat]

# -------------------- Auth --------------------
def login_edl_or_die(edl_user: str | None, edl_pass: str | None):
    """
    Priority:
      1) explicit --edl_user/--edl_pass (set env + environment strategy)
      2) EARTHDATA_USERNAME / EARTHDATA_PASSWORD env vars
      3) _netrc (default strategy)
    """
    # If provided explicitly, set env vars for this process
    if edl_user and edl_pass:
        os.environ["EARTHDATA_USERNAME"] = edl_user
        os.environ["EARTHDATA_PASSWORD"] = edl_pass
        auth = earthaccess.login(strategy="environment")
    else:
        # Let earthaccess auto-detect (env, then _netrc, then interactive)
        auth = earthaccess.login()

    if not getattr(auth, "authenticated", False):
        eprint("Earthdata Login failed. Provide --edl_user/--edl_pass, set EARTHDATA_* env vars, or configure _netrc.")
        sys.exit(1)
    return auth

def _ndvi_u8_rgb_alpha(ndvi_da: xr.DataArray):
    """
    Convert NDVI float32 [-1,1] to:
      - ndvi_u8: 0..255
      - R,G,B: 0..255 using a simple ramp (blue->yellow->green)
      - A: 0..255 alpha (0 for NaN/masked)
    """
    # Scale [-1,1] -> [0,255]
    ndvi_u8 = ((ndvi_da.clip(-1, 1) + 1.0) * 127.5).round().astype("uint8")

    # Build a simple perceptible ramp:
    #   0   = blue, ~128 = yellow, 255 = green
    # piecewise linear, no external cmap dependency
    mid = 128
    # Red: rises to yellow at mid, then falls towards green
    R = xr.where(ndvi_u8 < mid, ndvi_u8 * 2, 255 - ((ndvi_u8 - mid) * 2)).clip(0, 255).astype("uint8")
    # Green: increases to max and stays high
    G = xr.where(ndvi_u8 < mid, ndvi_u8, 255).clip(0, 255).astype("uint8")
    # Blue: high for low NDVI, fades out towards high NDVI
    B = xr.where(ndvi_u8 < mid, 255, 255 - ((ndvi_u8 - mid) * 2)).clip(0, 255).astype("uint8")

    # Alpha: 0 where NDVI is NaN (masked by QA), else 255
    A = xr.where(np.isnan(ndvi_da), 0, 255).astype("uint8")

    return ndvi_u8, R, G, B, A


def _granule_stem(name: str) -> str | None:
    m = STEM_RE.match(os.path.basename(name))
    return m.group(1) if m else None

def _group_paths_by_granule(files_or_dirs, scratch_dir):
    groups = defaultdict(list)
    for f in files_or_dirs:
        f = _maybe_extract_zip(f, scratch_dir)
        if os.path.isdir(f):
            # take all tifs in the dir; we’ll sub-group by stem below
            cand = glob.glob(os.path.join(f, "**", "*.tif"), recursive=True) + \
                   glob.glob(os.path.join(f, "**", "*.TIF"), recursive=True)
        else:
            # include siblings in the same directory
            d = os.path.dirname(f)
            cand = glob.glob(os.path.join(d, "*.tif")) + glob.glob(os.path.join(d, "*.TIF"))

        for p in cand:
            stem = _granule_stem(p)
            if stem:
                groups[stem].append(p)
    return groups  # {stem: [paths...]}

def _maybe_extract_zip(path, dest_root):
    """
    If 'path' is a .zip, extract into a unique subfolder under dest_root and return that folder.
    Otherwise, return the original path.
    """
    if path.lower().endswith(".zip"):
        base = os.path.splitext(os.path.basename(path))[0]
        outdir = os.path.join(dest_root, f"unz_{base}")
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(outdir)
        return outdir
    return path

# -------------------- Args --------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch NDVI (HLS) for a union of bboxes or field polygons on a single date"
    )
    # make bboxes_json and polys_file mutually exclusive (one required)
    mx = p.add_mutually_exclusive_group(required=True)
    mx.add_argument("--bboxes_json", help="JSON array of bboxes: [[minx,miny,maxx,maxy], ...]")
    mx.add_argument("--polys_file", help="Path to JSON of field polygons {id:[[lon,lat]...],...}")

    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--outdir", required=True, help="Directory to write the AOI NDVI GeoTIFF")
    p.add_argument("--max_granules", type=int, default=40, help="Max HLS granules (HLSS30+HLSL30)")
    p.add_argument("--apply_qa", action="store_true", help="Apply HLS Fmask to drop cloud/shadow/snow")
    p.add_argument("--qa_keep_water", action="store_true", help="Keep water pixels (Fmask==1)")
    p.add_argument("--direct_red")
    p.add_argument("--direct_nir")
    p.add_argument("--direct_fmask")
    p.add_argument("--make_rgb_quicklook", action="store_true",
               help="Also write a 4-band RGBA quicklook GeoTIFF")
    p.add_argument("--out_driver", choices=["GTiff","COG"], default="GTiff")
    p.add_argument("--edl_user", help="Earthdata Login username")
    p.add_argument("--edl_pass", help="Earthdata Login password")
    p.add_argument("--search_pad_m", type=float, default=300.0, help="Pad (meters) added to search bbox")
    p.add_argument("--window_days", type=int, default=0,
               help="Half-window around --date (0 = exact day)")
    p.add_argument("--start_date", help="YYYY-MM-DD (overrides --date/--window_days if set)")
    p.add_argument("--end_date", help="YYYY-MM-DD (overrides --date/--window_days if set)")
    # in parse_args()
    p.add_argument(
        "--per_granule_out",
        action="store_true",
        help="Write one COG per granule (no mosaic). Each granule is reprojected "
            "and optionally clipped to the AOI."
    )
    return p.parse_args()


def parse_bboxes(bboxes_json):
    try:
        arr = json.loads(bboxes_json)
        assert isinstance(arr, list) and len(arr) > 0
        for bb in arr:
            assert isinstance(bb, (list, tuple)) and len(bb) == 4
        return arr
    except Exception as e:
        eprint(f"Invalid --bboxes_json: {e}")
        sys.exit(1)

# -------------------- Search --------------------

def find_hls_granules(aoi_bbox, start_iso, end_iso, max_n):
    # Search both HLS Sentinel-2 and Landsat collections over a date RANGE
    q_s2 = (earthaccess.granule_query()
            .short_name("HLSS30")
            .cloud_hosted(True)
            .bounding_box(*aoi_bbox)
            .temporal(start_iso, end_iso))
    q_l8 = (earthaccess.granule_query()
            .short_name("HLSL30")
            .cloud_hosted(True)
            .bounding_box(*aoi_bbox)
            .temporal(start_iso, end_iso))

    # Pull up to max_n from each, then dedupe & trim
    g_s2 = q_s2.get(max_n)
    g_l8 = q_l8.get(max_n)
    grans = g_s2 + g_l8

    # Deduplicate by GranuleUR (or concept-id)
    seen = set()
    uniq = []
    for g in grans:
        gid = None
        try:
            gid = g["umm"].get("GranuleUR") or g["meta"].get("concept-id")
        except Exception:
            pass
        if gid and gid in seen:
            continue
        seen.add(gid)
        uniq.append(g)
    grans = uniq

    # If you want a hard overall cap, trim here:
    if len(grans) > max_n:
        grans = grans[:max_n]

    # Helpful metadata for diagnostics
    gran_ids = []
    for g in grans:
        cid = None
        try:
            cid = g["umm"].get("GranuleUR") or g["meta"].get("concept-id")
        except Exception:
            pass
        gran_ids.append(cid)

    return grans, gran_ids


# -------------------- Band detection --------------------
import re

def detect_red_nir_fmask_tifs(paths):
    """
    HLS v2 naming:
      - HLSS30 (Sentinel-2):   RED=B04, NIR=B08 (10m). Also contains B8A (20m) but we prefer B08.
      - HLSL30 (Landsat 8/9):  RED=B04, NIR=B05.
      - QA band: Fmask
    We accept either ".Bxx." or "_Bxx." just in case.
    """
    red = nir = fmask = None
    for p in paths:
        name = os.path.basename(p).upper()
        if not name.endswith(".TIF"):
            continue

        # match .B04. or _B04. (case-insensitive handled by upper())
        if red is None and re.search(r"[._]B04\.", name):
            red = p

        # Prefer B08 (S2 10m). If not found yet, allow B05 (L8/9 30m). If neither, allow B8A as a fallback.
        if nir is None and re.search(r"[._]B08\.", name):
            nir = p
        if nir is None and re.search(r"[._]B05\.", name):
            nir = p
        if nir is None and re.search(r"[._]B8A\.", name):
            nir = p  # last resort (S2 20m NIR)

        if fmask is None and "FMASK" in name:
            fmask = p

        if red and nir and fmask:
            break

    return red, nir, fmask

# -------------------- NDVI compute --------------------

def compute_ndvi_tile(red_path, nir_path, fmask_path=None, keep_water=False):
    """
    Compute NDVI for a single tile. If fmask_path is provided and apply_qa=True,
    drop pixels where Fmask ∈ {cloud(4), cloud shadow(2), snow/ice(3)}.
    Keep 'clear'(0) and optionally 'water'(1) if keep_water=True.
    """
    da_red = open_rioxarray(red_path).squeeze()
    da_nir = open_rioxarray(nir_path).squeeze()

    # Align grids (common with HLS pairs)
    if da_red.rio.crs != da_nir.rio.crs or da_red.rio.transform() != da_nir.rio.transform():
        da_nir = da_nir.rio.reproject_match(da_red)

    # Scale reflectance (HLS uses scale factor ~ 0.0001)
    red_f = (da_red.astype("float32") * 1e-4)
    nir_f = (da_nir.astype("float32") * 1e-4)

    ndvi = (nir_f - red_f) / (nir_f + red_f)
    ndvi = ndvi.clip(min=-1.0, max=1.0)

    # Optional QA mask
    if fmask_path is not None:
        qa = open_rioxarray(fmask_path).squeeze()
        # Reproject QA to the red grid if needed
        if qa.rio.crs != da_red.rio.crs or qa.rio.transform() != da_red.rio.transform():
            qa = qa.rio.reproject_match(da_red)

        # Fmask codes: 0=clear, 1=water, 2=cloud shadow, 3=snow/ice, 4=cloud
        if keep_water:
            keep = (qa == 0) | (qa == 1)
        else:
            keep = (qa == 0)

        ndvi = ndvi.where(keep)  # non-keep pixels -> NaN (nodata)

    return ndvi

# -------------------- Mosaic --------------------
def build_ndvi_mosaic(
    granules, aoi_bbox, scratch_dir,
    apply_qa=False, keep_water=False,
    res_deg=None, per_granule_out=False,
    outdir=None, clip_geom=None
):
    if not granules:
        raise RuntimeError("No HLS granules for AOI/date")

    files = earthaccess.download(granules, local_path=scratch_dir)
    groups = _group_paths_by_granule(files, scratch_dir)

    tiles_used = 0
    ndvi_tiles_ll = []

    # ------- per-granule (no mosaic) -------
    if per_granule_out:
        if not outdir:
            raise ValueError("--per_granule_out requires a valid --outdir")
        os.makedirs(outdir, exist_ok=True)

        written = []
        for stem, tif_paths in groups.items():
            red, nir, fmask = detect_red_nir_fmask_tifs(tif_paths)
            if not red or not nir:
                eprint(f"Skipping granule (missing bands): STEM={stem}")
                continue

            ndvi = compute_ndvi_tile(
                red, nir,
                fmask_path=(fmask if (apply_qa and fmask) else None),
                keep_water=keep_water
            )

            ndvi_ll = (ndvi.rio.reproject("EPSG:4326", resolution=res_deg)
                       if res_deg else ndvi.rio.reproject("EPSG:4326"))

            if clip_geom is not None:
                ndvi_ll = ndvi_ll.rio.clip([mapping(clip_geom)], crs="EPSG:4326")

            ndvi_ll.rio.write_nodata(OUT_NODATA, inplace=True)
            out_path = os.path.join(outdir, f"{stem}.tif")
            ndvi_ll.rio.to_raster(
                out_path,
                driver="COG",
                dtype="float32",
                compress="deflate",
                BIGTIFF="IF_SAFER",
                NUM_THREADS="ALL_CPUS",
            )
            written.append(out_path)
            tiles_used += 1

        return None, tiles_used, written

    # ------- mosaic path (unchanged idea) -------
    for stem, tif_paths in groups.items():
        red, nir, fmask = detect_red_nir_fmask_tifs(tif_paths)
        if not red or not nir:
            eprint(f"Skipping granule (missing bands): STEM={stem}")
            continue

        ndvi = compute_ndvi_tile(
            red, nir,
            fmask_path=(fmask if (apply_qa and fmask) else None),
            keep_water=keep_water
        )

        ndvi_ll = (ndvi.rio.reproject("EPSG:4326", resolution=res_deg)
                   if res_deg else ndvi.rio.reproject("EPSG:4326"))
        ndvi_tiles_ll.append(ndvi_ll)
        tiles_used += 1

    if not ndvi_tiles_ll:
        raise RuntimeError("No usable RED/NIR bands found")

    template = merge_arrays(
        ndvi_tiles_ll,
        bounds=tuple(aoi_bbox),
        res=res_deg,
        nodata=np.nan
    )
    aligned = [a.rio.reproject_match(template) for a in ndvi_tiles_ll]
    stack = xr.concat(aligned, dim="stack")
    mosaic = stack.median(dim="stack", skipna=True)

    return mosaic, tiles_used, None
# -------------------- Main --------------------
def main():
    args = parse_args()
    direct_mode = bool(getattr(args, "direct_red", None) and getattr(args, "direct_nir", None))

    # validate date
    try:
        _ = date_cls.fromisoformat(args.date)
    except Exception:
        eprint("Invalid --date (expected YYYY-MM-DD)")
        sys.exit(1)

    # Decide temporal range
    center = date_cls.fromisoformat(args.date)
    if args.start_date and args.end_date:
        start_iso = args.start_date
        end_iso   = args.end_date
    elif getattr(args, "window_days", 0) and args.window_days > 0:
        start_iso = (center - timedelta(days=args.window_days)).isoformat()
        end_iso   = (center + timedelta(days=args.window_days)).isoformat()
    else:
        start_iso = end_iso = args.date


    os.makedirs(args.outdir, exist_ok=True)

    # 1) AOI from polygons OR bboxes
    if getattr(args, "polys_file", None):
        field_polys = load_polys_json(args.polys_file)
        aoi_geom = aoi_union_from_fields(field_polys)
    else:
        bboxes = parse_bboxes(args.bboxes_json)
        aoi_geom = unary_union([box(*bb) for bb in bboxes])

    minx, miny, maxx, maxy = aoi_geom.bounds
    aoi_bbox = pad_bbox_meters([minx, miny, maxx, maxy], args.search_pad_m, lat_hint=(miny+maxy)/2.0)
    clip_geom = aoi_geom

    # 2) Build NDVI in one of two ways → set ndvi_aoi, tiles_used, grans, gran_ids
    if direct_mode:
        # no Earthdata auth needed
        ndvi_single = compute_ndvi_tile(
            args.direct_red, args.direct_nir,
            fmask_path=(args.direct_fmask if args.apply_qa else None),
            keep_water=args.qa_keep_water
        )
        ndvi_ll = ndvi_single.rio.reproject("EPSG:4326")
        ndvi_aoi = ndvi_ll.rio.clip([mapping(clip_geom)], crs="EPSG:4326")
        tiles_used = 1
        grans, gran_ids = [], []
    else:
        # 2a) Auth + search + download + mosaic
        login_edl_or_die(args.edl_user, args.edl_pass)
        grans, gran_ids = find_hls_granules(aoi_bbox, start_iso, end_iso, args.max_granules)
        if not grans:
            print(json.dumps({"error": "No granules found", "aoi_bbox": aoi_bbox, "date": args.date}))
            sys.exit(0)

        scratch = tempfile.mkdtemp(prefix="ndvi_hls_")
        lat_hint = (miny + maxy) / 2.0
        res_deg = deg_res_for_meters(30.0, lat_hint)
        try:
            mosaic_ll, tiles_used, written = build_ndvi_mosaic(
                grans, aoi_bbox, scratch,
                apply_qa=args.apply_qa,
                keep_water=args.qa_keep_water,
                res_deg=res_deg,
                per_granule_out=args.per_granule_out,
                outdir=args.outdir,
                clip_geom=clip_geom,
            )
        except Exception as ex:
            eprint(f"NDVI mosaic failed: {ex}")
            print(json.dumps({"error": str(ex)}))
            sys.exit(1)

        # If per-granule mode, we're done: emit JSON and exit
        if args.per_granule_out:
            print(json.dumps({
                "granule_cogs": written,
                "aoi_bbox": aoi_bbox,
                "date": args.date,
                "date_start": start_iso,
                "date_end": end_iso,
                "tiles_used": int(tiles_used),
                "granule_count": int(len(grans)),
                "granule_ids": gran_ids,
                "apply_qa": bool(args.apply_qa),
                "qa_keep_water": bool(args.qa_keep_water)
            }))
            return

        ndvi_aoi = mosaic_ll.rio.clip([mapping(clip_geom)], crs="EPSG:4326")
        eprint("mosaic bounds:", mosaic_ll.rio.bounds())

    # 3) Write outputs (shared for both paths)
    driver = getattr(args, "out_driver", "GTiff")  # add choices in parse_args
    ndvi_aoi.rio.write_nodata(OUT_NODATA, inplace=True)
    date_tag = args.date if start_iso == end_iso else f"{start_iso}_{end_iso}"
    out_tif = os.path.join(args.outdir, f"ndvi_aoi_{date_tag}.tif")
    ndvi_aoi.rio.to_raster(
        out_tif,
        driver=driver,
        dtype="float32",
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        nodata=OUT_NODATA,            # <- use same value, not np.nan
        OVERVIEW_RESAMPLING="average",
        NUM_THREADS="ALL_CPUS",
        BIGTIFF="IF_SAFER"
    )

    # RGB quicklook (optional)
    out_rgb = None  # <- make sure this exists for the JSON
    if args.make_rgb_quicklook:
        ndvi_u8, R, G, B, A = _ndvi_u8_rgb_alpha(ndvi_aoi)
        rgb = xr.concat([R, G, B, A], dim="band").assign_coords(band=["R", "G", "B", "A"])
        rgb = rgb.rio.write_crs(ndvi_aoi.rio.crs).rio.write_transform(ndvi_aoi.rio.transform())
        out_rgb = os.path.join(args.outdir, f"ndvi_aoi_{args.date}_rgb.tif")
        rgb.rio.to_raster(
            out_rgb,
            driver=driver,
            dtype="uint8",
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            OVERVIEW_RESAMPLING="nearest",
            NUM_THREADS="ALL_CPUS",
            BIGTIFF="IF_SAFER"
        )

    # 4) Emit JSON
    print(json.dumps({
        "aoi_tif": out_tif,
        "aoi_rgb": out_rgb,
        "aoi_bbox": aoi_bbox,
        "date": args.date,
        "tiles_used": int(tiles_used),
        "granule_count": int(len(grans)),
        "granule_ids": gran_ids,
        "apply_qa": bool(args.apply_qa),
        "qa_keep_water": bool(args.qa_keep_water)
    }))


if __name__ == "__main__":
    main()
