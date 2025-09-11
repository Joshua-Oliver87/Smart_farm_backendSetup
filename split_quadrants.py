#!/usr/bin/env python3
import json, argparse, os
from shapely.geometry import Polygon, MultiPolygon, shape, box
from shapely.ops import unary_union

def load_polys(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case A: your current format -> {"WA": [[lon,lat], ...]}
    if isinstance(data, dict) and "type" not in data:
        polys = []
        for _, ring in data.items():
            if ring[0] != ring[-1]:
                ring = ring + [ring[0]]
            polys.append(Polygon(ring))
        return unary_union(polys), (list(data.keys())[0] if len(data) == 1 else "AOI")

    # Case B: GeoJSON (FeatureCollection/Polygon/MultiPolygon)
    geom = shape(data["features"][0]["geometry"]) if data.get("type") == "FeatureCollection" else shape(data)
    return geom, data.get("name", "AOI")

def to_mapping(geom, label_prefix):
    if geom.is_empty:
        return {}
    geoms = [geom] if isinstance(geom, Polygon) else (list(geom.geoms) if isinstance(geom, MultiPolygon) else [])
    out = {}
    for i, g in enumerate(geoms, 1):
        coords = [list(c) for c in g.exterior.coords]  # exterior only (matches your loader)
        out[f"{label_prefix}_{i}"] = coords
    return out

def main():
    ap = argparse.ArgumentParser(description="Split AOI polygon into exact quadrants and write JSON per quadrant.")
    ap.add_argument("--polys_file", required=True, help="Input polygon JSON (your WA file).")
    ap.add_argument("--outdir", required=True, help="Where to write quadrant JSONs.")
    ap.add_argument("--label", default=None, help="Base label (defaults to first key or 'AOI').")
    args = ap.parse_args()

    aoi, default_label = load_polys(args.polys_file)
    minx, miny, maxx, maxy = aoi.bounds
    midx = (minx + maxx) / 2.0
    midy = (miny + maxy) / 2.0

    os.makedirs(args.outdir, exist_ok=True)
    base = (args.label or default_label).strip()

    quads = {
        "SW": box(minx, miny, midx, midy),
        "NW": box(minx, midy, midx, maxy),
        "SE": box(midx, miny, maxx, midy),
        "NE": box(midx, midy, maxx, maxy),
    }

    reports = []
    for name, rect in quads.items():
        clipped = aoi.intersection(rect)
        mapping = to_mapping(clipped, f"{base}_{name}")
        outpath = os.path.join(args.outdir, f"{base.lower()}_{name.lower()}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(mapping, f, separators=(",", ":"))
        reports.append((name, outpath, len(mapping)))

    print("Wrote: " + " | ".join([f"{n}: {cnt} polys -> {p}" for n, p, cnt in reports]))

if __name__ == "__main__":
    main()
