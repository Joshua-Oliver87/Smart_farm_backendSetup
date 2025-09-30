from datetime import datetime
import numpy as np
import rioxarray as rxr
import io, geopandas as gpd, rasterio as rio
from rasterio.mask import mask
import os

REFLECTANCE_CODES = {"B01","B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"}
QA_CODES = {"Fmask"}
ANGLE_CODES = {"SZA","SAA","VZA","VAA"}  # don't scale these

# For HLSS30:
S30_ALIAS_TO_CODE = {
    "nir":  "B08",
    "red":  "B04",
    "green":"B03",
    "blue": "B02",
    "swir1":"B11",
    "swir2":"B12",
    "ca":   "B01",   # coastal aerosol
    "re1":  "B05",
    "re2":  "B06",
    "re3":  "B07",
    "wv":   "B09",   # water vapor
    "cirrus":"B10",
    "fmask":"Fmask"
}

L30_ALIAS_TO_CODE = {  # Landsat 8/9 (HLSL30)
    "nir":"B05", "red":"B04", "blue":"B02", "green":"B03",
    "swir1":"B06","swir2":"B07","ca":"B01","cirrus":"B09",
    "fmask":"Fmask"
}

def detect_hls_product(links: list[str]) -> str:
    """Return 'HLSS30' or 'HLSL30' based on any URL in the list."""
    for u in links:
        if "HLSS30." in u: return "HLSS30"
        if "HLSL30." in u: return "HLSL30"
    return "HLSS30"  # default

def get_alias_to_code(product: str) -> dict:
    return S30_ALIAS_TO_CODE if product == "HLSS30" else L30_ALIAS_TO_CODE

def time_index_from_filenames(paths):
    """
    Accepts either:
      - HLS original names with ...YYYYJJJTHHMMSS...
      - Cached names like YYYY-MM-DD_<suffix>_cropped.tif
    """
    out = []
    for f in paths:
        name = os.path.basename(f)
        # new pattern
        try:
            out.append(datetime.strptime(name.split("_", 1)[0], "%Y-%m-%d"))
            continue
        except Exception:
            pass
        # legacy HLS pattern (…YYYYJJJTHHMMSS…)
        try:
            out.append(datetime.strptime(name.split('.')[-4], '%Y%jT%H%M%S'))
        except Exception:
            # last resort: file mtime
            out.append(datetime.fromtimestamp(os.path.getmtime(f)))
    return out



def create_quality_mask(qa, bit_nums=(1,2,3,4,5)):
    """
    Return True where the pixel should be masked (bad).
    HLS Fmask v2 is class-based: 0=clear, 1=water, 2=shadow, 3=snow/ice, 4=cloud, 255=fill.
    Fallback to bitfield interpretation if needed.
    """
    arr = np.nan_to_num(qa, 255).astype(np.int16)

    # Heuristic: class map with small integers
    uniq = np.unique(arr[arr != 255])
    if uniq.size > 0 and uniq.max() <= 8:
        # keep only 'clear' (0). Mask water/cloud/shadow/snow and fill.
        return (arr != 0) | (arr == 255)

    # Otherwise treat as bitfield
    bad = np.zeros_like(arr, dtype=bool)
    for b in bit_nums:
        bad |= (arr & (1 << b)) > 0
    bad |= (arr == 255)
    return bad

# Define function to scale 
def scaling(band):
    scale_factor = band.attrs['scale_factor'] 
    band_out = band.copy()
    band_out.data = band.data * scale_factor
    band_out.attrs['scale_factor'] = 1
    return band_out

# Checks if mask_and_scale applied and if not manually applies scale
def open_hls_band(url):
    # IMPORTANT: mask_and_scale=False so we do NOT auto-scale on read
    da = rxr.open_rasterio(
        url,
        chunks=dict(band=1, x=512, y=512),              # chunk size
        masked=True,
        mask_and_scale=False
    ).squeeze("band", drop=True)

    code = url.rsplit('.', 2)[-2]  # e.g. "B08", "Fmask", "SZA", ...
    # If it’s a reflectance band, ensure scale metadata is present.
    if code in REFLECTANCE_CODES and 'scale_factor' not in da.attrs:
        da.attrs['scale_factor'] = 0.0001

    return code, da 

def clip_block_from_path(cog_path: str, block_gdf: gpd.GeoDataFrame) -> bytes | None:
    with rio.open(cog_path) as src:
        block = block_gdf.to_crs(src.crs)
        out_arr, out_transform = mask(src, block.geometry.values, crop=True, all_touched=True)
        meta = src.meta.copy()
        meta.update({"height": out_arr.shape[1], "width": out_arr.shape[2], "transform": out_transform})
        with rio.MemoryFile() as mem:
            with mem.open(**meta) as dst:
                dst.write(out_arr)
            return mem.read()