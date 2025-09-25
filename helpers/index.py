from datetime import datetime
import numpy as np

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

def time_index_from_filenames(ndvi_files):
    '''
    Helper function to create a pandas DatetimeIndex
    '''
    return [datetime.strptime(f.split('.')[-4], '%Y%jT%H%M%S') for f in ndvi_files]

def calculate_ndvi(nir, red):
    # Create EVI xarray.DataArray that has the same coordinates and metadata
      ndvi = red.copy()
      # Calculate the NDVI
      ndvi_data = nir.data + red.data
      # Replace the Red xarray.DataArray data with the new NDVI data
      with np.errstate(divide='ignore', invalid='ignore'):
        ndvi_data = (nir.data - red.data) / ndvi_data
      ndvi.data = ndvi_data.astype("float32") 
      # exclude the inf values
      ndvi = xr.where(np.isfinite(ndvi), ndvi, np.nan, keep_attrs=True)
      # change the long_name in the attributes
      ndvi.attrs['long_name'] = 'NDVI'
      ndvi.attrs['scale_factor'] = 1
      return ndvi

def compute_indices(scaled):
    out = {}
    if "NDVI" in TARGET_INDICES:
        nir = scaled.get("B08"); red = scaled.get("B04")
        if nir is not None and red is not None:
            out["NDVI"] = calculate_ndvi(nir, red)
    # if you add EVI later, do it here similarly:
    # if "EVI" in TARGET_INDICES:
    #     nir = scaled.get("B08"); red = scaled.get("B04"); blue = scaled.get("B02")
    #     if nir is not None and red is not None and blue is not None:
    #         out["EVI"] = calculate_evi(nir, red, blue)
    return out

def create_quality_mask(quality_data, bit_nums: list = [1, 2, 3, 4, 5]):
    """
    Uses the Fmask layer and bit numbers to create a binary mask of good pixels.
    By default, bits 1-5 are used.
    """
    mask_array = np.zeros((quality_data.shape[0], quality_data.shape[1]))
    # Remove/Mask Fill Values and Convert to Integer
    quality_data = np.nan_to_num(quality_data, 255).astype(np.int8)
    for bit in bit_nums:
        # Create a Single Binary Mask Layer
        mask_temp = np.array(quality_data) & 1 << bit > 0
        mask_array = np.logical_or(mask_array, mask_temp)
    return mask_array

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