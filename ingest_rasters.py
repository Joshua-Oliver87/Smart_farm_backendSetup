# insert_quadrant_masks.py
import json, os
from pathlib import Path
import psycopg

# --- edit these paths/names to your files/mask_ids ---
QUADS = [
    {"mask_id": "WA_NE", "mask_name": "Washington NE", "geojson": r"C:\Users\joshua.oliver\Desktop\smartFarm_backend\helpers\wa_quadrants\wa_ne.json"},
    {"mask_id": "WA_NW", "mask_name": "Washington NW", "geojson": r"C:\Users\joshua.oliver\Desktop\smartFarm_backend\helpers\wa_quadrants\wa_nw.json"},
    {"mask_id": "WA_SE", "mask_name": "Washington SE", "geojson": r"C:\Users\joshua.oliver\Desktop\smartFarm_backend\helpers\wa_quadrants\wa_se.json"},
    {"mask_id": "WA_SW", "mask_name": "Washington SW", "geojson": r"C:\Users\joshua.oliver\Desktop\smartFarm_backend\helpers\wa_quadrants\wa_sw.json"},
]

DSN = os.getenv("PG_DSN", "postgresql://smart_farm:new_password87$@10.107.5.35/ndvi")

def polygon_bbox_from_geojson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    # accept Feature or Polygon
    if gj.get("type") == "Feature":
        geom = gj["geometry"]
    else:
        geom = gj
    if geom["type"] != "Polygon":
        raise ValueError(f"{path} is not a Polygon GeoJSON")
    ring = geom["coordinates"][0]
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    return [min(xs), min(ys), max(xs), max(ys)]

def upsert_mask(conn, mask_id, mask_name, bbox):
    minx, miny, maxx, maxy = bbox
    conn.execute("""
      INSERT INTO lorawan_smart_farm.raster_masks (mask_id, mask_name, bbox)
      VALUES (%s, %s, ST_MakeEnvelope(%s,%s,%s,%s,4326))
      ON CONFLICT (mask_id) DO UPDATE
        SET bbox = EXCLUDED.bbox,
            mask_name = COALESCE(EXCLUDED.mask_name, raster_masks.mask_name),
            updated_at = now();
    """, (mask_id, mask_name, minx, miny, maxx, maxy))

if __name__ == "__main__":
    with psycopg.connect(DSN) as conn:
        conn.execute("SET search_path TO public, lorawan_smart_farm;")
        for q in QUADS:
            bbox = polygon_bbox_from_geojson(q["geojson"])
            upsert_mask(conn, q["mask_id"], q["mask_name"], bbox)
        conn.commit()
        print("Inserted/updated 4 quadrant masks.")
