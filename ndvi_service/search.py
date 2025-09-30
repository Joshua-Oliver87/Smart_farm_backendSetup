import earthaccess
import geopandas as gpd

"""
Query Earthdata (HLSS30 and (later) HLSL30) for granules intersecting the bounding box
"""

def login():
    earthaccess.login(persist=True)

def search_hlss30(bbox, temporal, count=100):
    # bbox = (minx, miny, maxx, maxy), temporal=("YYYY-MM-DD","YYYY-MM-DD" or full ISO)
    results = earthaccess.search_data(
        short_name=['HLSS30'],
        bounding_box=tuple(bbox),
        temporal=temporal,
        count=count
    )
    # list[list[str]] where inner list = links within a granule
    return [granule.data_links() for granule in results]

def search_day(gdf_or_bbox, day_str: str, *, count: int = 20, count_per_day: int | None = None):
    # alias support
    if count_per_day is not None:
        count = count_per_day

    # accept either a GeoDataFrame or a bbox list/tuple
    if isinstance(gdf_or_bbox, gpd.GeoDataFrame):
        gdf4326 = gdf_or_bbox.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = gdf4326.total_bounds
        bbox = (float(minx), float(miny), float(maxx), float(maxy))
    else:
        bbox = tuple(gdf_or_bbox)  # assume bbox-like

    temporal = (f"{day_str}T00:00:00", f"{day_str}T23:59:59")
    results = earthaccess.search_data(
        short_name=['HLSS30'],
        bounding_box=bbox,
        temporal=temporal,
        count=count
    )
    return [g.data_links() for g in results]