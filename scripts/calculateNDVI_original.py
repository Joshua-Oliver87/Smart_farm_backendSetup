import os, rasterio as rio
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gp
from skimage import io
import matplotlib.pyplot as plt
#from osgeo import gdal
import xarray as xr
import rioxarray as rxr
import hvplot.xarray
import hvplot.pandas
import earthaccess
from shapely.geometry import Polygon
import geopandas as gp

WA_NE_1_coords =  [[-118.107964276, 46.193713997],
  [-118.109828261, 46.193714049],
  [-118.109484915, 46.191497798],
  [-118.109270883, 46.191323941],
  [-118.108931971, 46.191181275],
  [-118.108673304, 46.191025177],
  [-118.108294322, 46.190690763],
  [-118.108107051, 46.19046781],
  [-118.107772659, 46.190307268],
  [-118.106702412, 46.190079869],
  [-118.105596494, 46.1896473],
  [-118.101275631, 46.188180252],
  [-118.100383896, 46.187836938],
  [-118.099777405, 46.187721004],
  [-118.099220056, 46.187591644],
  [-118.097431915, 46.186976327],
  [-118.095795409, 46.186499186],
  [-118.095795435, 46.186566096],
  [-118.095795535, 46.187444522],
  [-118.09579991, 46.188853624],
  [-118.096125438, 46.189094373],
  [-118.096299318, 46.189272778],
  [-118.096415275, 46.18938865],
  [-118.09660256, 46.189504619],
  [-118.096847839, 46.189513549],
  [-118.097476551, 46.189602724],
  [-118.09798931, 46.189745419],
  [-118.09799381, 46.189745404],
  [-118.098314839, 46.189905975],
  [-118.098551227, 46.190155649],
  [-118.098684979, 46.190235883],
  [-118.099469724, 46.190423204],
  [-118.100593455, 46.19105636],
  [-118.100936758, 46.191390826],
  [-118.101583417, 46.192336155],
  [-118.101828645, 46.192933688],
  [-118.101895516, 46.193714011],
  [-118.107964276, 46.193713997]]
roi_poly = Polygon(WA_NE_1_coords)
field = gp.GeoDataFrame({"name": ["WA_NE_1"]}, geometry=[roi_poly], crs="EPSG:4326")
"""
NEEDS: Thermal image fetching from Landsat for crop water stress
       To know if we want all bands + NDVI for our ROI to be returned in a single GeoTIFF
       Figure out good count number, currently 100, takes a long time to compute but very accurate for given time range
"""

######## MAPPING INFO FOR SENTINEL ##########
'''
Coastal Aerosol: B01
Blue:            B02
Green:           B03
Red:             B04
Red Edge 1:      B05
Red Edge 2:      B06
Red Edge 3:      B07
NIR Broad:       B08
NIR Narrow:      B8A
SWIR 1:          B11
SWIR 2:          B12
Water vapor:     B09 
Cirrus:          B10
'''

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

TARGET_INDICES = ["NDVI"]           # Add "EVI", or other index as needed
SAVE_RAW_BANDS = False             # True => also include the raw bands loaded
USE_FMASK = True
INDEX_REQUIREMENTS = {
    "NDVI": {"B08", "B04"},          # NIR, RED
    "EVI":  {"B08", "B04", "B02"},   # NIR, RED, BLUE
}                  

#########################

######## HELPERS ##########

REFLECTANCE_CODES = {"B01","B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"}
QA_CODES = {"Fmask"}
ANGLE_CODES = {"SZA","SAA","VZA","VAA"}  # don't scale these

###################### SETUP ##########################


earthaccess.login(persist=True)

# Use geojson file that has our ROI (region of interest) to query HLS data for files that intersect
# Read in geojson file:
#field = gp.read_file('C:/Users/joshua.oliver/Documents/WA-field-boundaries.geojson')

#bbox is tuple holding list of field boundaries
bbox = tuple(field.total_bounds)
print(bbox)

#Specify time range
temporal = ("2021-05-01T00:00:00", "2021-09-30T23:59:59")

#Search JUST Sentinel data for now (add HLSL30 later if needed)
# Do we want to specify a count????????????????
results = earthaccess.search_data(
    short_name=['HLSS30'],
    bounding_box=bbox,
    temporal=temporal,
    count=100
    )

# View metadata if needed-->(only first 5 entries)
pd.json_normalize(results).head(5)

# Or just view each individual result with data link and browse image:
results[0]

# Grab all url's for the data
hls_results_urls = [granule.data_links() for granule in results]
hls_results_urls[0:1] # subset

# Url's for browse images
browse_urls = [granule.dataviz_links()[0] for granule in results] # 0 gets only the https links
browse_urls[0:2] # subset

# Read HLS COG's from Earthdata Cloud
# Access cloud assets we want & read into memory without needing to download files!

# GDAL configurations used to successfully access LP DAAC Cloud Assets via vsicurl 
# NOTE: Changed from gdal. to use rio.env.set instead
rio.env.set_gdal_config('GDAL_HTTP_COOKIEFILE',  os.path.expanduser('~/cookies.txt'))        # Tells GDAL HTTP layer where to read cookies from
rio.env.set_gdal_config('GDAL_HTTP_COOKIEJAR',   os.path.expanduser('~/cookies.txt'))        # Where to write updated cookies too
rio.env.set_gdal_config('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')   # Skip remote directory listings and pretend the directory is empty. Makes open by exact URL fast
rio.env.set_gdal_config('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', 'TIF')      # Performance guard: Stops GDAL from probing non-TIF URL's & cuts useless HEAD/GETs
rio.env.set_gdal_config('GDAL_HTTP_UNSAFESSL', 'YES')                  # Allows insecure SSL.
rio.env.set_gdal_config('GDAL_HTTP_MAX_RETRY', '10')                   # Retry failed HTTP requests up to 10 times
rio.env.set_gdal_config('GDAL_HTTP_RETRY_DELAY', '0.5')                # Delay between retries

# Look at url's for a returned granule
h = hls_results_urls[10]
print(h)


############################################################################ MAIN LOOP #########################################################################
idx_suffix = "_".join(TARGET_INDICES) or "bands"
out_folder = 'C:/Users/joshua.oliver/Desktop/ndvi_cog_out'
os.makedirs(out_folder, exist_ok=True)

for j, h in enumerate(hls_results_urls):
    out_name = h[0].split('/')[-1].split('v2.0')[0] + f'v2.0_{idx_suffix}_cropped.tif'
    out_path = os.path.join(out_folder, out_name)
    print(out_name)

    ### Calculating NDVI ###
    ndvi_band_links = []
    # Which HLS product is being accessed
    if h[0].split('/')[4] == 'HLSS30.020':
        # build the minimal set of bands needed for the requested indices
        needed = set()
        for idx in TARGET_INDICES:
            needed |= INDEX_REQUIREMENTS.get(idx, set())
        if USE_FMASK:
            needed.add("Fmask")
        if SAVE_RAW_BANDS:
            needed |= REFLECTANCE_CODES  # Grab all reflectance bands if specified

        ndvi_bands = sorted(needed)
    else:
        # Add Landsat mapping here when needed
        ndvi_bands = []

    # Subset assets in this granule down to only desired bands (unchanged loop)
    for a in h:
        # match cleanly on ".Bxx.tif" or ".Fmask.tif" to avoid false positives
        if any(f".{b}.tif" in a for b in ndvi_bands):
            ndvi_band_links.append(a)

    print(ndvi_band_links)

    print(out_name)

    if os.path.exists(out_path):
        print(f"{out_name} has already been processed and is available in this directory, moving to next file.")
        continue


    ####### Load HLS COGS INTO MEMORY #########
    # Define chunk size of an HLS tile
    # mask NaN values
    # read files using rioxarray and name them based on the band
    # squeeze object to remove band dimension since files are 1 band
    # FOR SCALING: set mask_and_scale=True. Default scale factor is 1
    # NOTE: Some scale factors for HLS granules are found in file metadata, but rioxarray always looks in band metadata
            # Manually set scale factor if needed to avoid this

    loaded = {}
    loaded_alias = {}

    for url in ndvi_band_links:
        code, da = open_hls_band(url)
        loaded[code] = da

    # To be able to reference loaded_alias["nir"] for example
    for alias, code in S30_ALIAS_TO_CODE.items():
        if code in loaded:
            loaded_alias[alias] = loaded[code]

    ########################################

    for alias, da in loaded_alias.items():
        print(f"Loaded {alias} ({[k for k,v in S30_ALIAS_TO_CODE.items() if v==da.attrs.get('HLS_CODE','')]}): shape={tuple(da.shape)}, dtype={da.dtype}")


    ###################### Subset spatially ###########################

    # First need to convert geopandas dataframe from lat/lon into UTM. Handle zonal projection
    # Extract unique UTM zonal projection parameters rom input HLS files & use them to transform the coordinate of input farm field

    # Use NIR’s CRS (any reflectance band is fine—they share the same CRS per granule)
    nir_da = loaded_alias["nir"]
    fsUTM = field.to_crs(nir_da.spatial_ref.crs_wkt)
    # Now use our field ROI to mask any pixels that fall outside it and crop to bounding box via rasterio
    # GREATLY REDUCES AMOUNT OF DATA THAT IS NEEDED TO LOAD INTO MEMORY

    # Batch clip
    cropped_by_code = {}
    for alias, code in S30_ALIAS_TO_CODE.items():
        if alias in loaded_alias:
            cropped_by_code[code] = loaded_alias[alias].rio.clip(
                fsUTM.geometry.values, fsUTM.crs, all_touched=True
            )

    # Batch scale (this is where scaling() is actually used)
    scaled_by_code = {}
    for code, da in cropped_by_code.items():
        if code in REFLECTANCE_CODES:
            scaled_by_code[code] = scaling(da)  # <-- your scaling() used here
        elif code == "Fmask":
            scaled_by_code[code] = da
        else:
            scaled_by_code[code] = da

    computed = compute_indices(scaled_by_code)
    da_list, names = [], []
    if not computed:
        print("[WARN] No indices computed; skipping this granule.")
        continue

    # apply Fmask cloud mask to every computed index if requested & available
    fmask_cropped = cropped_by_code.get("Fmask")
    if USE_FMASK and fmask_cropped is not None:
        mask_layer = create_quality_mask(fmask_cropped.data, [1,2,3,4,5])
        for k in list(computed.keys()):
            computed[k] = computed[k].where(~mask_layer)

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

    # 4c) optionally include Fmask as a band for debugging/QA
    if USE_FMASK and "Fmask" in cropped_by_code:
        da = cropped_by_code["Fmask"].astype("float32")
        for k in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
            da.attrs.pop(k, None); da.encoding.pop(k, None)
        NODATA = -9999.0
        da = da.fillna(NODATA).rio.write_nodata(NODATA, encoded=True)
        da_list.append(da); names.append("Fmask")

    if not da_list:
        print("[WARN] Nothing to write; skipping.")
        continue

    stack = xr.concat(da_list, dim="band").assign_coords(band=("band", names))
    stack.attrs.pop("_FillValue", None); stack.encoding.pop("_FillValue", None)

    """
    # Stack all bands + NDVI (unchanged)
    stack_order = ["NDVI","B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","Fmask","B01"]
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
    """
    # optional but nice for COGs
    stack.rio.to_raster(
        raster_path=out_path,
        driver="COG",
        compress="DEFLATE",
        BIGTIFF="IF_SAFER"
    )

    del stack, da_list, names, computed
    print(f"Processed file {j+1} of {len(hls_results_urls)}")

##### Stack HLS data #####
ndvi_dir = 'C:/Users/joshua.oliver/Desktop/ndvi_cog_out'
ndvi_files = [os.path.abspath(os.path.join(ndvi_dir, o)) for o in os.listdir(ndvi_dir) if o.endswith(f'{idx_suffix}_cropped.tif')]  # List COGs
print(f"There are {len(ndvi_files)} {idx_suffix} files.")

# Make time index as xarray variable from the filenames
time = xr.Variable('time', time_index_from_filenames(ndvi_files))
chunks=dict(band=1, x=512, y=512)

# MAKE TIMESERIES
ts_index = TARGET_INDICES[0]  # e.g., "NDVI"
# If files may have multiple bands, select that band explicitly:
ndvi_ts = xr.concat(
    [rxr.open_rasterio(f, mask_and_scale=True, chunks=chunks).sel(band=ts_index).squeeze('band', drop=True)
     for f in ndvi_files],
    dim=time
)
ndvi_ts.name = ts_index

ndvi_ts = ndvi_ts.sortby(ndvi_ts.time)
print("ndvi_ts:",ndvi_ts)

### Visualize stacked time series ###
title = f'HLS-derived {ts_index} over agricultural fields in WA state'
ndvi_ts.hvplot.image(x='x', y='y', groupby='time', frame_width= 800, cmap='YlGn', fontscale=1.6, crs='EPSG:32610', tiles = 'EsriImagery')

# Slice one specific range in timeseries
title = 'HLS-derived NDVI'
ndvi_ts.isel(time=1).hvplot.image(x='x', y='y', cmap='YlGn', frame_width= 600, fontscale=1.6, crs='EPSG:32610', tiles = 'EsriImagery').opts(title=f'{title}, {ndvi_ts.isel(time=4).SENSING_TIME}')
# Plot time series as boxplots to show distribution of NDVI values from our field
ndvi_ts.hvplot.box(ts_index, by=['time'], rot=90, box_fill_color='lightblue', width=900, height=450).opts(ylim=(-0.5,1.5)).opts(title='NDVI Timeseries')

######## Export Statistics ########
# xarray allows you to easily calculate a number of statistics
ndvi_min = ndvi_ts.min(('y', 'x'))
ndvi_max = ndvi_ts.max(('y', 'x'))
ndvi_mean = ndvi_ts.mean(('y', 'x'))
ndvi_sd = ndvi_ts.std(('y', 'x'))
ndvi_count = ndvi_ts.count(('y', 'x'))
ndvi_median = ndvi_ts.median(('y', 'x'))

# Make simple & interactive plots for min, max, mean, sd
ndvi_mean.hvplot.line()

# Combine line plots for different statistics
stats = (ndvi_mean.hvplot.line(height=350, width=450, line_width=1.5, color='red', grid=True, padding=0.05).opts(title='Mean')+ 
    ndvi_sd.hvplot.line(height=350, width=450, line_width=1.5, color='red', grid=True, padding=0.05).opts(title='Standard Deviation')
    + ndvi_max.hvplot.line(height=350, width=450, line_width=1.5, color='red', grid=True, padding=0.05).opts(title='Max') + 
    ndvi_min.hvplot.line(height=350, width=450, line_width=1.5, color='red', grid=True, padding=0.05).opts(title='Min')).cols(2)

print("stats:", stats)

# Make pandas dataframe of stats and export to CSV
df = pd.DataFrame({'Min '+ts_index: ndvi_min, 'Max '+ts_index: ndvi_max,
                   'Mean '+ts_index: ndvi_mean, 'Standard Deviation '+ts_index: ndvi_sd,
                   'Median '+ts_index: ndvi_median, 'Count': ndvi_count})

df.index = ndvi_ts.time.data    # Set observation date as the index
df.to_csv('C:/Users/joshua.oliver/Desktop/ndvi_stats.csv', index=True)
