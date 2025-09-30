import xarray as xr
import numpy as np

"""
For adding additional index's that will be computed and returned from hls.py for frontend visualization (Like NDVI, EVI, etc)
"""

TARGET_INDICES = ["NDVI"]           # Add "EVI", or other index as needed

INDEX_REQUIREMENTS = {
    "NDVI": {"nir", "red"},          # NIR, RED
    "EVI":  {"nir", "red", "blue"},   # NIR, RED, BLUE
}   

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

def compute_indices_by_alias(scaled_by_alias: dict) -> dict:
    out = {}
    if "NDVI" in TARGET_INDICES:
        nir = scaled_by_alias.get("nir")
        red = scaled_by_alias.get("red")
        if nir is not None and red is not None:
            out["NDVI"] = calculate_ndvi(nir, red)
    # if "EVI" in TARGET_INDICES:
    #     blue = scaled_by_alias.get("blue")
    #     if nir is not None and red is not None and blue is not None:
    #         out["EVI"] = calculate_evi(nir, red, blue)
    return out
