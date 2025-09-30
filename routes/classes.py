from typing import Optional, List
from pydantic import BaseModel, Field
from ndvi_service.indices import TARGET_INDICES

class BaseGeo(BaseModel):
    # Provide exactly one of these geometry sources
    geojson: Optional[dict] = Field(
        default=None, description="GeoJSON Geometry, Feature, or FeatureCollection (EPSG:4326)"
    )
    wkt: Optional[str] = Field(default=None, description="WKT polygon (EPSG:4326)")
    bbox: Optional[List[float]] = Field(
        default=None, min_items=4, max_items=4, description="[minx,miny,maxx,maxy] (EPSG:4326)"
    )
    coords: Optional[List[List[float]]] = Field(
        default=None, description="[[lon,lat], ...] polygon ring (EPSG:4326)"
    )

class CommonParams(BaseGeo):
    indices: Optional[List[str]] = Field(
        default=None,
        description=f"Subset of available indices (default: {TARGET_INDICES})",
    )
    use_fmask: bool = True
    quantize: bool = True
    scale: int = 10000
    count_per_day: int = 20  # search cap

class DayRequest(CommonParams):
    date: str = Field(..., description="YYYY-MM-DD")

class RangeRequest(CommonParams):
    start: str = Field(..., description="YYYY-MM-DD")
    end: str = Field(..., description="YYYY-MM-DD")
    include_payloads: bool = True  # stats-only if False
