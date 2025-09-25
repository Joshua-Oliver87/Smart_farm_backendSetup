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

WA_NE_1_coords = [[-120.71660407248436,49.00018838095098],[-120.3762180247982,49.00070607907161],[-120.0355101664815,48.99953685629129],[-120.00119928628976,48.99941910732785],[-119.9192635254378,49.0000939823249],[-119.87619734283753,49.000448699798774],[-119.70201959176373,49.00026917233873],[-119.70170641261808,49.00026485469535],[-119.70122157757136,49.00025817051834],[-119.45769956513949,49.000260967906094],[-119.44208734879831,49.00025675446278],[-119.4286772647843,49.00025313533422],[-119.1807290576803,49.000292106857984],[-119.13727567317949,49.0002989830027],[-119.13210386793862,49.00026402036364],[-118.90278292411631,49.00029998100113],[-118.83661995294645,49.00030988026254],[-118.50374400442398,49.00035796699147],[-118.49877305980223,49.000358685679814],[-118.37043206122537,49.000377990280796],[-118.36469310582845,49.000378853514164],[-118.22461692015168,49.000400075524816],[-118.22258191340305,49.00040039382375],[-118.19738683430234,49.000404334629565],[-118.03993231029129,49.00042896236157],[-118.00205715896449,49.00043488647846],[-118.00111714911962,48.99990888695902],[-117.88440457663826,48.99991135250553],[-117.8761062578375,49.00054545808796],[-117.63091021014428,49.00081822844113],[-117.6073228728829,49.000844339563145],[-117.42996807416876,49.00036544773212],[-117.29982619690212,49.000013210064786],[-117.26819500264355,48.99992759818124],[-117.26825000044292,48.99981759879106],[-117.12607615916798,48.998888330955864],[-117.0323505245854,48.99918900098305],[-117.03210664992885,48.874926574198334],[-117.03238430380763,48.84666576883208],[-117.03333478346204,48.74992114175074],[-117.03367082154203,48.65690195942106],[-117.03370868361827,48.655337956337746],[-117.03435783412408,48.628522903581946],[-117.03449883743488,48.620768888385484],[-117.03459505138619,48.60821236348799],[-117.03489639752577,48.56888478607007],[-117.03542488599892,48.49991365291365],[-117.03538260450549,48.478823609764454],[-117.03530121130959,48.438224645413435],[-117.03528494812875,48.43011256788452],[-117.03528494839367,48.42981556752288],[-117.03525395428908,48.4231435594011],[-117.03528895471956,48.42273155889687],[-117.0351780004028,48.371220496207954],[-117.03517800070834,48.37087749579047],[-117.03751177064237,48.25982460353439],[-117.03860228657707,48.20793835105764],[-117.03908080529504,48.19663546380341],[-117.0391204195343,48.19569975278978],[-117.03921565202602,48.1934503068423],[-117.039281594303,48.191892712561184],[-117.03928650959257,48.19177661061435],[-117.03946387752703,48.18758707882425],[-117.03954866384134,48.18558437780749],[-117.03959938338838,48.18438635325153],[-117.03961538492139,48.18401435328647],[-117.03958239616541,48.181123353676156],[-117.03958239723258,48.18085235371037],[-117.03958339936332,48.18031235377778],[-117.03961840806296,48.17814135402579],[-117.03941341069316,48.177249354290616],[-117.03952044964352,48.1747160896705],[-117.03955242424657,48.17395935460256],[-117.03955742346513,48.17380166320046],[-117.03979160215809,48.16641491626629],[-117.04110762471785,48.12490335974225],[-117.04140151897963,48.085534571557545],[-117.04140178180504,48.08549936460901],[-117.04167694108453,48.04555936958542],[-117.04167776803537,48.04546364119769],[-117.04226526419025,47.97745076373546],[-117.0423612089444,47.96634219128552],[-117.04236812102972,47.958363900499904],[-117.04243068614723,47.88614788578342],[-117.04244081751851,47.874453700562235],[-117.04245220570374,47.861308826592186],[-117.04247152666137,47.83900749602457],[-117.04228684353481,47.8324939865678],[-117.04200056654075,47.82239740444633],[-117.04206567600824,47.77862816534641],[-117.04248670723192,47.766523100295444],[-117.04252271137405,47.76489409149699],[-117.04262472075584,47.76122107172079],[-117.04265872174521,47.76085506981537],[-117.04206073963087,47.74509801366904],[-117.04213673819882,47.74409801486292],[-117.04167109416578,47.73591929276128],[-117.04166784904812,47.73586229446638],[-117.04163572299646,47.735298022173225],[-117.04165036453732,47.73110663705945],[-117.04167970292738,47.72270803473112],[-117.0416356882301,47.70676446421327],[-117.0416346766885,47.70639805074701],[-117.04162770491358,47.70479679435941],[-117.04162657040423,47.70453622356187],[-117.04161450824202,47.70176582199271],[-117.04161168874826,47.70111824910648],[-117.0415973974954,47.697835876777425],[-117.04159671665306,47.697679502988976],[-117.04159352236125,47.696945847644336],[-117.04158678935491,47.69539943090193],[-117.04153363924053,47.68319207349864],[-117.04143263385275,47.67999807647133],[-117.04143263094204,47.678183078271125],[-117.04143263086988,47.67813807831575],[-117.04112778369367,47.62357739905235],[-117.04111811112705,47.621846231313235],[-117.04098779325258,47.598522322875915],[-117.04091076586073,47.58473619122638],[-117.04088294129363,47.57975623426208],[-117.04087649855826,47.57860313310337],[-117.04085146208557,47.57412218122597],[-117.04090565382494,47.5715137625966],[-117.04117543815019,47.55852819687609],[-117.04127743799445,47.55820819721306],[-117.04074639543325,47.53290722244805],[-117.04054638608727,47.52756022785074],[-117.04052121440758,47.52333011468234],[-117.04051537758126,47.52234923309245],[-117.0405145687083,47.52228594174877],[-117.03994632970542,47.47782121262975],[-117.0399723234838,47.46330718429514],[-117.03994931986536,47.434894054417654],[-117.03994931101954,47.434883128847645],[-117.03995130124088,47.4124100849975],[-117.03991431102438,47.405161176996515],[-117.03988329519653,47.39908305902623],[-117.0401772856866,47.374898011705824],[-117.04007061972179,47.36602585260939],[-117.03984442875985,47.34721195622716],[-117.03984427248564,47.347198957800565],[-117.0399176338787,47.31053993446631],[-117.03999272387739,47.27301710629588],[-120.82458086539867,47.27301710629588],[-120.82458086539867,49.00026178878592],[-120.71660407248436,49.00018838095098]]
roi_poly = Polygon(WA_NE_1_coords)
field = gp.GeoDataFrame({"name": ["WA_NE_1"]}, geometry=[roi_poly], crs="EPSG:4326")


# Output JSON
results = {}

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

TARGET_INDICES = ["NDVI"]           # Add "EVI", or other index as needed
SAVE_RAW_BANDS = False             # True => also include the raw bands loaded
USE_FMASK = True
INDEX_REQUIREMENTS = {
    "NDVI": {"B08", "B04"},          # NIR, RED
    "EVI":  {"B08", "B04", "B02"},   # NIR, RED, BLUE
}
PERSIST_RASTER = True                # Whether to save the output raster to disk     

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
    # Nice for COGs
    if PERSIST_RASTER:
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

#################### For visualizing timeseries ####################
'''

### Visualize stacked time series ###
title = f'HLS-derived {ts_index} over agricultural fields in WA state'
ndvi_ts.hvplot.image(x='x', y='y', groupby='time', frame_width= 800, cmap='YlGn', fontscale=1.6, crs='EPSG:32610', tiles = 'EsriImagery')

# Slice one specific range in timeseries
title = 'HLS-derived NDVI'
ndvi_ts.isel(time=1).hvplot.image(x='x', y='y', cmap='YlGn', frame_width= 600, fontscale=1.6, crs='EPSG:32610', tiles = 'EsriImagery').opts(title=f'{title}, {ndvi_ts.isel(time=4).SENSING_TIME}')
# Plot time series as boxplots to show distribution of NDVI values from our field
ndvi_ts.hvplot.box(ts_index, by=['time'], rot=90, box_fill_color='lightblue', width=900, height=450).opts(ylim=(-0.5,1.5)).opts(title='NDVI Timeseries')
'''

######## Export Statistics ########
# xarray allows you to easily calculate a number of statistics
ndvi_min = ndvi_ts.min(('y', 'x'))
ndvi_max = ndvi_ts.max(('y', 'x'))
ndvi_mean = ndvi_ts.mean(('y', 'x'))
ndvi_sd = ndvi_ts.std(('y', 'x'))
ndvi_count = ndvi_ts.count(('y', 'x'))
ndvi_median = ndvi_ts.median(('y', 'x'))

results["timeseries"].append
({
    "name": ndvi_ts.name,
    "date": ndvi_ts.time,
    "min": ndvi_min,
    "max": ndvi_max,
    "mean": ndvi_mean,
    "sd": ndvi_sd,
    "count": ndvi_count,
    "median": ndvi_median
})

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
