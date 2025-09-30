import os

CACHE_DIR          = os.getenv("NDVI_CACHE_DIR", r"C:\ndvi_cache")
COUNT_PER_DAY      = int(os.getenv("NDVI_COUNT_PER_DAY", 20))
CACHE_TTL_SECONDS  = int(os.getenv("NDVI_CACHE_TTL_SEC", 3600)) # 1 hour
RECENT_LOOKBACK_D  = int(os.getenv("NDVI_RECENT_LOOKBACK_D", 10))

# GDAL configurations used to successfully access LP DAAC Cloud Assets via vsicurl 
def rasterio_env_kwargs():
    cookies = os.path.expanduser('~/cookies.txt')
    return dict(
        GDAL_HTTP_COOKIEFILE=cookies,   # Tells GDAL HTTP layer where to read cookies from
        GDAL_HTTP_COOKIEJAR=cookies,     # Where to write updated cookies too
        GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',   # Skip remote directory listings and pretend the directory is empty. Makes open by exact URL fast
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS='TIF',     # Performance guard: Stops GDAL from probing non-TIF URL's & cuts useless HEAD/GETs
        GDAL_HTTP_UNSAFESSL='YES',                  # Allows insecure SSL.
        GDAL_HTTP_MAX_RETRY='10',                   # Retry failed HTTP requests up to 10 times
        GDAL_HTTP_RETRY_DELAY='0.5',                # Delay between retries
    )