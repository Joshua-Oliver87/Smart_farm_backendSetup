# ingest_rasters.py
import argparse, math, os
from datetime import date
import rasterio as rio
from rasterio.warp import transform_bounds
import psycopg2

def approx_pixel_size_m(ds):
    # Works for both projected and EPSG:4326 rasters
    xres, yres = ds.res
    if ds.crs and ds.crs.is_geographic:
        # meters per degree (approx) at scene center
        cx = (ds.bounds.left + ds.bounds.right) / 2.0
        cy = (ds.bounds.top  + ds.bounds.bottom) / 2.0
        m_per_deg_lat = 111_132.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(cy))
        return int(round(max(abs(yres) * m_per_deg_lat, abs(xres) * m_per_deg_lon)))
    else:
        return int(round(max(abs(xres), abs(yres))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", required=True, help="postgresql://user:pass@host/dbname")
    ap.add_argument("--dataset", required=True, help="NDVI or ET")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("tiffs", nargs="+", help="Paths to GeoTIFF/COG files")
    args = ap.parse_args()

    dt = date.fromisoformat(args.date)
    conn = psycopg2.connect(args.conn)
    conn.autocommit = True
    cur = conn.cursor()

    # ensure helpful indexes (no-op if already present)
    cur.execute("""
      CREATE INDEX IF NOT EXISTS raster_asset_bbox_gix
      ON lorawan_smart_farm.raster_asset USING GIST (bbox);
    """)
    cur.execute("""
      DO $$
      BEGIN
        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint
          WHERE conname='raster_asset_uq'
        ) THEN
          ALTER TABLE lorawan_smart_farm.raster_asset
          ADD CONSTRAINT raster_asset_uq UNIQUE (dataset, tile_id, dt);
        END IF;
      END$$;
    """)

    for path in args.tiffs:
        with rio.open(path) as ds:
            # bounds in 4326
            left, bottom, right, top = transform_bounds(ds.crs, "EPSG:4326",
                                                        ds.bounds.left, ds.bounds.bottom,
                                                        ds.bounds.right, ds.bounds.top,
                                                        densify_pts=0)
            pixel_m = approx_pixel_size_m(ds)
            nodata = ds.nodata
            driver = ds.driver or "GTiff"

        tile_id = os.path.splitext(os.path.basename(path))[0].upper()  # e.g. WA_NE
        sql = """
        INSERT INTO lorawan_smart_farm.raster_asset
          (dataset, tile_id, dt, uri, driver, storage, pixel_size_m, nodata, bbox,
           qa_applied, qa_keep_water)
        VALUES
          (%s, %s, %s, %s, %s, 'local', %s, %s,
           ST_MakeEnvelope(%s,%s,%s,%s,4326),
           TRUE, FALSE)
        ON CONFLICT (dataset, tile_id, dt) DO UPDATE
          SET uri           = EXCLUDED.uri,
              driver        = EXCLUDED.driver,
              storage       = EXCLUDED.storage,
              pixel_size_m  = EXCLUDED.pixel_size_m,
              nodata        = EXCLUDED.nodata,
              bbox          = EXCLUDED.bbox,
              updated_at    = now();
        """
        cur.execute(sql, (args.dataset, tile_id, dt, os.path.abspath(path), driver,
                          pixel_m, nodata, left, bottom, right, top))
        print(f"Upserted {args.dataset} {dt} {tile_id} -> {path}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
