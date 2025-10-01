import os
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio as rio
from ndvi_service.indices import TARGET_INDICES, INDEX_REQUIREMENTS, compute_indices_by_alias
from ndvi_service.helper import (
    S30_ALIAS_TO_CODE, open_hls_band, REFLECTANCE_CODES,
    create_quality_mask, scaling, get_alias_to_code, detect_hls_product
)
from rioxarray.merge import merge_arrays
import base64, zlib

"""

For setting up/stacking COG's, applying FMASK, etc, after querying NASA HLS repo for raw bands used for index compute
Relies on index's specified in index.py
Has support for mosaicing tiles per day and streaming COG bytes for frontend use

"""
# Returns per-index DataArrays for frontend visualizing
def _indices_from_links(links: list[str], farm_gdf: gpd.GeoDataFrame, use_fmask=True) -> dict[str, xr.DataArray] | None:
    product = detect_hls_product(links)
    alias_to_code = get_alias_to_code(product)

    needed_aliases = set()
    for idx in TARGET_INDICES:
        needed_aliases |= INDEX_REQUIREMENTS.get(idx, set())
    needed_codes = {alias_to_code[a] for a in needed_aliases if a in alias_to_code}
    if use_fmask and "fmask" in alias_to_code:
        needed_codes.add(alias_to_code["fmask"])

    band_urls = [u for u in links if any(f".{code}.tif" in u for code in needed_codes)]
    if not band_urls:
        return None

    loaded_by_code = {}
    for url in band_urls:
        code, da = open_hls_band(url)
        loaded_by_code[code] = da

    loaded_by_alias = {a: loaded_by_code[c] for a, c in alias_to_code.items() if c in loaded_by_code}
    nir = loaded_by_alias.get("nir")
    if nir is None:
        return None

    farm_utm = farm_gdf.to_crs(nir.rio.crs)
    cropped_by_alias = {a: da.rio.clip(farm_utm.geometry.values, farm_utm.crs, all_touched=True)
                        for a, da in loaded_by_alias.items()}

    # scale reflectance where needed
    scaled_by_alias = {}
    for a, da in cropped_by_alias.items():
        code = alias_to_code.get(a, "")
        scaled_by_alias[a] = scaling(da) if code in REFLECTANCE_CODES else da

    computed = compute_indices_by_alias(scaled_by_alias)
    if not computed:
        return None

    # optional Fmask
    fm = cropped_by_alias.get("fmask")
    if use_fmask and fm is not None:
        mask_layer = create_quality_mask(fm.data, [1,2,3,4,5])
        for k in list(computed.keys()):
            computed[k] = computed[k].where(~mask_layer)

    # normalize dtype/nodata
    out = {}
    for idx_name, da in computed.items():
        if da.dtype != np.float32:
            da = da.astype("float32")
        for k in ("_FillValue","missing_value","scale_factor","add_offset"):
            da.attrs.pop(k, None); da.encoding.pop(k, None)
        out[idx_name] = da.fillna(-9999.0).rio.write_nodata(-9999.0, encoded=True)
    return out


# Creates timeseries stack for given index
def _stack_from_links(links: list[str], farm_gdf: gpd.GeoDataFrame, use_fmask=True) -> xr.DataArray | None:
    product = detect_hls_product(links)
    alias_to_code = get_alias_to_code(product)

    needed_aliases = set()
    for idx in TARGET_INDICES:
        needed_aliases |= INDEX_REQUIREMENTS.get(idx, set())
    needed_codes = {alias_to_code[a] for a in needed_aliases if a in alias_to_code}
    if use_fmask and "fmask" in alias_to_code:
        needed_codes.add(alias_to_code["fmask"])

    # filter URLs for those codes
    band_urls = [u for u in links if any(f".{code}.tif" in u for code in needed_codes)]
    if not band_urls:
        return None

    # load
    loaded_by_code = {}
    for url in band_urls:
        code, da = open_hls_band(url)
        loaded_by_code[code] = da

    # view by alias
    loaded_by_alias = {a: loaded_by_code[c] for a, c in alias_to_code.items() if c in loaded_by_code}
    nir = loaded_by_alias.get("nir")
    if nir is None:
        return None

    # crop to farm
    farm_utm = farm_gdf.to_crs(nir.rio.crs)
    cropped_by_alias = {a: da.rio.clip(farm_utm.geometry.values, farm_utm.crs, all_touched=True)
                        for a, da in loaded_by_alias.items()}

    # scale reflectance
    scaled_by_alias = {}
    for a, da in cropped_by_alias.items():
        code = alias_to_code.get(a, "")
        scaled_by_alias[a] = scaling(da) if code in REFLECTANCE_CODES else da

    # indices
    computed = compute_indices_by_alias(scaled_by_alias)
    if not computed:
        return None

    # Fmask
    fm = cropped_by_alias.get("fmask")
    if use_fmask and fm is not None:
        mask_layer = create_quality_mask(fm.data, [1,2,3,4,5])
        for k in list(computed.keys()):
            computed[k] = computed[k].where(~mask_layer)

    # stack
    da_list, names = [], []
    for idx_name in TARGET_INDICES:
        if idx_name in computed:
            da = computed[idx_name]
            if da.dtype != np.float32: da = da.astype("float32")
            for k in ("_FillValue","missing_value","scale_factor","add_offset"):
                da.attrs.pop(k, None); da.encoding.pop(k, None)
            da_list.append(da.fillna(-9999.0).rio.write_nodata(-9999.0, encoded=True))
            names.append(idx_name)

    # include fmask as last band for QA
    if use_fmask and fm is not None:
        fm_da = fm.astype("float32")
        for k in ("_FillValue","missing_value","scale_factor","add_offset"):
            fm_da.attrs.pop(k, None); fm_da.encoding.pop(k, None)
        da_list.append(fm_da.fillna(-9999.0).rio.write_nodata(-9999.0, encoded=True))
        names.append("fmask")

    if not da_list:
        return None

    stack = xr.concat(da_list, dim="band").assign_coords(band=("band", names))
    stack.attrs.pop("_FillValue", None); stack.encoding.pop("_FillValue", None)
    return stack


def mosaic_and_write_bytes(stacks: list[xr.DataArray]) -> bytes:
    import tempfile, os
    if not stacks: 
        return None

    merged = stacks[0]
    for st in stacks[1:]:
        merged = merged.rio.merge(st, method="first")

    # enforce dtype & nodata right before write
    merged = merged.astype("float32").rio.write_nodata(-9999.0, encoded=True)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        path = tmp.name
    merged.rio.to_raster(
        path,
        driver="COG",
        compress="DEFLATE",
        BIGTIFF="IF_SAFER",
        nodata=-9999.0,
        dtype="float32",
        OVERVIEWS="AUTO",
        NUM_THREADS="ALL_CPUS",
    )
    with open(path, "rb") as f:
        data = f.read()
    os.unlink(path)
    return data


#MULTIBAND
def build_farm_cog_bytes(granules_links: list[list[str]], farm_gdf: gpd.GeoDataFrame, use_fmask=True) -> bytes | None:
    per = []
    for links in granules_links:
        st = _stack_from_links(links, farm_gdf, use_fmask=use_fmask)
        if st is not None:
            per.append(st)
    return mosaic_and_write_bytes(per)

def _to_cog_bytes(da: xr.DataArray) -> bytes | None:
    import tempfile, os
    da = da.astype("float32").rio.write_nodata(-9999.0, encoded=True)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        path = tmp.name
    da.rio.to_raster(
        path,
        driver="COG",
        compress="DEFLATE",
        BIGTIFF="IF_SAFER",
        nodata=-9999.0,
        dtype="float32",
        OVERVIEWS="AUTO",
        NUM_THREADS="ALL_CPUS",
    )
    with open(path, "rb") as f:
        data = f.read()
    os.unlink(path)
    return data



# PER INDEX
def build_farm_cog_bytes_per_index(granules_links: list[list[str]], farm_gdf: gpd.GeoDataFrame, use_fmask=True) -> dict[str, bytes]:
    # accumulate per-index mosaics across all intersecting granules
    acc: dict[str, list[xr.DataArray]] = {}
    for links in granules_links:
        dct = _indices_from_links(links, farm_gdf, use_fmask=use_fmask)
        if not dct:
            continue
        for idx in TARGET_INDICES:
            da = dct.get(idx)
            if da is not None:
                acc.setdefault(idx, []).append(da)

    out: dict[str, bytes] = {}
    for idx, lst in acc.items():
        merged = lst[0] if len(lst) == 1 else merge_arrays(lst, method="first", nodata=np.nan)
        out[idx] = _to_cog_bytes(merged)
    return out


def collect_needed_codes(alias_to_code: dict, use_fmask: bool, extra_aliases: set[str] | None = None) -> list[str]:
    needed_aliases = set()
    for idx in TARGET_INDICES:
        needed_aliases |= INDEX_REQUIREMENTS.get(idx, set())
    if extra_aliases:
        needed_aliases |= set(extra_aliases)
    codes = {alias_to_code[a] for a in needed_aliases if a in alias_to_code}
    if use_fmask and "fmask" in alias_to_code:
        codes.add(alias_to_code["fmask"])
    return sorted(codes)


################# FOR RETURNING RAW BYTES DIRECTLY TO FRONTEND BYPASS WRITING TIFS #################

def _stats_from_da(da) -> dict:
    """
    Compute robust per-raster stats from a single-band DataArray.
    Assumes float values with nodata encoded as -9999 and/or NaNs.
    Returns JSON-serializable scalars (or None when no valid pixels).
    """
    # ensure we’re working on the last two dims (y, x)
    arr = np.asarray(da.values)

    # valid pixels: finite and not nodata
    valid = np.isfinite(arr) & (arr != -9999)
    n = int(valid.sum())
    if n == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "count": 0,
            "q05": None,
            "q95": None,
        }

    vals = arr[valid].astype(np.float64, copy=False)

    # ddof=0 => population std (consistent for imagery)
    return {
        "min":   float(np.nanmin(vals)),
        "max":   float(np.nanmax(vals)),
        "mean":  float(np.nanmean(vals)),
        "median":float(np.nanmedian(vals)),
        "std":   float(np.nanstd(vals, ddof=0)),
        "count": n,
        "q05":   float(np.nanpercentile(vals, 5)),
        "q95":   float(np.nanpercentile(vals, 95)),
    }

def _to_payload_from_da(da: xr.DataArray, *, quantize=True, scale=10000):
    """
    Convert a single-band DataArray into a compact, JSON-safe payload.
    - If quantize=True, store NDVI [-1,1] as int16 using scale (e.g., 10000).
      nodata uses -32768.
    - Otherwise, send float32 (bigger payload).
    """
    da = da.rio.reproject(da.rio.crs)  # ensure transform exists
    transform = da.rio.transform()
    crs = da.rio.crs.to_string()
    height, width = da.shape[-2], da.shape[-1]
    bounds = list(da.rio.bounds())

    arr = da.values  # (y,x), float
    # treat -9999 and non-finite as nodata
    mask = ~np.isfinite(arr) | (arr == -9999)

    if quantize:
        nodata = np.int16(-32768)
        q = np.full((height, width), nodata, dtype=np.int16)
        good = ~mask
        # clip to int16 safe range after scaling
        q[good] = np.clip(np.round(arr[good] * scale), -32767, 32767).astype(np.int16)
        raw = q.tobytes(order="C")
        meta = {"dtype":"int16", "scale":1/scale, "offset":0, "nodata":int(nodata)}
    else:
        nodata_f = np.float32(-9999.0)
        f = np.full((height, width), nodata_f, dtype=np.float32)
        f[~mask] = arr[~mask].astype(np.float32)
        raw = f.tobytes(order="C")
        meta = {"dtype":"float32", "scale":1.0, "offset":0.0, "nodata":float(nodata_f)}

    compressed = zlib.compress(raw)
    b64 = base64.b64encode(compressed).decode("ascii")

    payload = {
        "crs": crs,
        "width": int(width),
        "height": int(height),
        "transform": [transform.a, transform.b, transform.c,
                      transform.d, transform.e, transform.f],  # GDAL affine
        "bounds": bounds,  # [minx, miny, maxx, maxy]
        "encoding": "zlib",
        "data_b64": b64,
        **meta
    }
    return payload


def build_mask_payload_per_index(granules_links: list[list[str]],
                                 farm_gdf: gpd.GeoDataFrame,
                                 use_fmask=True,
                                 quantize=True,
                                 scale=10000) -> dict[str, dict]:
    """
    Returns a dict {index_name: payload}, where payload contains a compressed typed
    array plus georeferencing metadata. No files are written.
    """
    # Accumulate per-index mosaics across intersecting granules
    acc: dict[str, list[xr.DataArray]] = {}
    for links in granules_links:
        dct = _indices_from_links(links, farm_gdf, use_fmask=use_fmask)
        if not dct:
            continue
        for idx in TARGET_INDICES:
            da = dct.get(idx)
            if da is not None:
                acc.setdefault(idx, []).append(da)

    out: dict[str, dict] = {}
    for idx, lst in acc.items():
        if not lst:
            continue
        merged = lst[0] if len(lst) == 1 else merge_arrays(lst, method="first", nodata=np.nan)
        # Enforce float32 + NaN nodata for clean transparency in the browser
        merged = merged.astype("float32").rio.write_nodata(np.nan, encoded=False)
        out[idx] = _to_payload_from_da(merged, quantize=quantize, scale=scale)
    return out

def build_mask_payloads_and_stats_for_day(
    granules_links: list[list[str]],
    farm_gdf,
    *,
    indices=None,
    use_fmask=True,
    quantize=True,
    scale=10000
) -> dict[str, dict]:
    """
    For one day: { index_name: { payload: {...}, stats: {...} } }
    """
    indices = list(indices or TARGET_INDICES)

    # Reuse your existing per-index builder, but keep the merged DA
    acc: dict[str, list[xr.DataArray]] = {}
    per_day: dict[str, dict] = {}

    dct_all = _indices_from_links  # just a shorthand

    # accumulate per-index mosaics across intersecting granules
    for links in granules_links:
        dct = dct_all(links, farm_gdf, use_fmask=use_fmask)  # computes indices on the clip
        if not dct:
            continue
        for idx in indices:
            da = dct.get(idx)
            if da is not None:
                acc.setdefault(idx, []).append(da)

    for idx, lst in acc.items():
        if not lst:
            continue

        # ensure each tile has float and nodata=NaN before merge
        lst = [da.astype("float32").rio.write_nodata(np.nan, encoded=False) for da in lst]

        # merge with nodata respected
        merged = lst[0] if len(lst) == 1 else merge_arrays(lst, method="first", nodata=np.nan)

        px = abs(float(merged.rio.resolution()[0]))  # e.g., 30.0
        farm_clip = farm_gdf.to_crs(merged.rio.crs).copy()
        farm_clip["geometry"] = farm_clip.geometry.buffer(-0.5 * px)  # try 0.5..1.0 * px

        merged = merged.rio.clip(
            farm_clip.geometry, crs=merged.rio.crs,
            all_touched=False, drop=False
        )
        merged = merged.astype("float32").rio.write_nodata(np.nan, encoded=False)
        print("CRS:", merged.rio.crs, "res:", merged.rio.resolution(), "bounds:", merged.rio.bounds())
        payload = _to_payload_from_da(merged, quantize=quantize, scale=scale)
        stats   = _stats_from_da(merged)
        per_day[idx] = {"payload": payload, "stats": stats}

    return per_day

"""
def get_raw_bands(hls_results_urls, field, USE_FMASK=True):
    ############################################################################ MAIN LOOP #########################################################################
    idx_suffix = "_".join(TARGET_INDICES) or "bands"
    out_folder = 'C:/Users/joshua.oliver/Desktop/ndvi_cog_out'
    os.makedirs(out_folder, exist_ok=True)

    for j, links in enumerate(hls_results_urls):
        # Detect product for THESE links and choose the right band codes
        product = detect_hls_product(links)                 # 'HLSS30' or 'HLSL30'
        alias_to_code = get_alias_to_code(product)
        needed_codes = collect_needed_codes(alias_to_code, use_fmask=USE_FMASK)
        
        out_name = links[0].split('/')[-1].split('v2.0')[0] + f'v2.0_{idx_suffix}_cropped.tif'
        out_path = os.path.join(out_folder, out_name)
        print(out_name)

        # Filter URLs down to only the bands we need for THIS product
        band_urls = [u for u in links if any(f".{code}.tif" in u for code in needed_codes)]
        if not band_urls:
            print("[SKIP] No needed bands present.")
            continue

        if os.path.exists(out_path):
            print(f"{out_name} already available, skipping.")
            continue

        ####### Load HLS COGS INTO MEMORY #########
        # Define chunk size of an HLS tile
        # mask NaN values
        # read files using rioxarray and name them based on the band
        # squeeze object to remove band dimension since files are 1 band
        # FOR SCALING: set mask_and_scale=True. Default scale factor is 1
        # NOTE: Some scale factors for HLS granules are found in file metadata, but rioxarray always looks in band metadata
                # Manually set scale factor if needed to avoid this

        loaded_by_code = {}
        for url in band_urls:
            code, da = open_hls_band(url)
            loaded_by_code[code] = da

        # Build alias view (nir/red/blue...) independent of product
        loaded_by_alias = {
            alias: loaded_by_code[code]
            for alias, code in alias_to_code.items()
            if code in loaded_by_code
        }

        ########################################

        for alias, da in loaded_by_alias.items():
            print(f"Loaded {alias} ({[k for k,v in S30_ALIAS_TO_CODE.items() if v==da.attrs.get('HLS_CODE','')]}): shape={tuple(da.shape)}, dtype={da.dtype}")


        ###################### Subset spatially ###########################

        # First need to convert geopandas dataframe from lat/lon into UTM. Handle zonal projection
        # Extract unique UTM zonal projection parameters rom input HLS files & use them to transform the coordinate of input farm field

        # Use NIR’s CRS (any reflectance band is fine—they share the same CRS per granule)
        nir_da = loaded_by_alias["nir"]
        if nir_da is None:
            print("[SKIP] Missing NIR for this granule.")
            continue
        fsUTM = field.to_crs(nir_da.spatial_ref.crs_wkt)
        # Now use our field ROI to mask any pixels that fall outside it and crop to bounding box via rasterio
        # GREATLY REDUCES AMOUNT OF DATA THAT IS NEEDED TO LOAD INTO MEMORY

        # Batch clip
        cropped_by_code = {
            a: da.rio.clip(fsUTM.geometry.values, fsUTM.crs, all_touched=True)
            for a, da in loaded_by_alias.items()
        }

        # Batch scale (this is where scaling() is actually used)
        scaled_by_code = {}
        for a, da in cropped_by_code.items():
            code = alias_to_code.get(a, "")
            scaled_by_code[a] = scaling(da) if code in REFLECTANCE_CODES else da


        computed = compute_indices_by_alias(scaled_by_code)
        if not computed:
            print("[WARN] No indices computed; skipping this granule.")
            continue

        # apply Fmask cloud mask to every computed index if requested & available
        fmask_cropped = cropped_by_code.get("fmask")
        if USE_FMASK and fmask_cropped is not None:
            mask_layer = create_quality_mask(fmask_cropped.data, [1,2,3,4,5])
            for k in list(computed.keys()):
                computed[k] = computed[k].where(~mask_layer)

        da_list, names = [], []

        for idx_name in TARGET_INDICES:
            if idx_name in computed:
                da = computed[idx_name]
                if da.dtype != np.float32:
                    da = da.astype("float32")
                for k in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
                    da.attrs.pop(k, None); da.encoding.pop(k, None)
                NODATA = -9999.0
                da = da.fillna(NODATA).rio.write_nodata(NODATA, encoded=True)
                da_list.append(da); names.append(idx_name)

        '''
        # 4b) optionally append raw reflectance bands you actually loaded
        if SAVE_RAW_BANDS:
            for code in sorted(REFLECTANCE_CODES & set(scaled_by_code.keys())):
                da = scaled_by_code[code]
                if da.dtype != np.float32:
                    da = da.astype("float32")
                for k in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
                    da.attrs.pop(k, None); da.encoding.pop(k, None)
                NODATA = -9999.0
                da = da.fillna(NODATA).rio.write_nodata(NODATA, encoded=True)
                da_list.append(da); names.append(code)
        '''

        # 4c) optionally include Fmask as a band for debugging/QA
        if USE_FMASK and "fmask" in cropped_by_code:
            da = cropped_by_code["Fmask"].astype("float32")
            for k in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
                da.attrs.pop(k, None); da.encoding.pop(k, None)
            NODATA = -9999.0
            da = da.fillna(NODATA).rio.write_nodata(NODATA, encoded=True)
            da_list.append(da); names.append("fmask")

        if not da_list:
            print("[WARN] Nothing to write; skipping.")
            continue

        stack = xr.concat(da_list, dim="band").assign_coords(band=("band", names))
        stack.attrs.pop("_FillValue", None); stack.encoding.pop("_FillValue", None)

        '''
        # Stack all bands + NDVI (unchanged)
        stack_order = ["NDVI","B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","fmask","B01"]
        da_list, names = [], []
        for code in stack_order:
            if code == "NDVI":
                da = ndvi_cropped_qf
            elif code in scaled_by_code:
                da = scaled_by_code[code]
            else:
                continue
            if da.dtype != np.float32:
                da = da.astype("float32")

            # Remove CF attrs/encoding that conflict on write
            for k in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
                da.attrs.pop(k, None)
                da.encoding.pop(k, None)

            # Use a single finite nodata value for all bands in the stack
            NODATA = -9999.0
            da = da.fillna(NODATA).rio.write_nodata(NODATA, encoded=True)

            da_list.append(da); names.append(code)

        if not da_list:
            print("[WARN] Nothing to write; skipping.")
            continue
        '''
        # Nice for COGs
        stack.rio.to_raster(
            raster_path=out_path,
            driver="COG",
            compress="DEFLATE",
            BIGTIFF="IF_SAFER"
        )

        del stack, da_list, names, computed
        print(f"Processed file {j+1} of {len(hls_results_urls)}")
"""