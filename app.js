

import 'dotenv/config';
import express from "express";
import {polygonToBbox, findLatestDtForDatasetIntersectingBbox, fetchTileUris,mosaicAndClip} from "./helpers/helper.js";

const app = express();
app.use(express.json({ limit: "2mb" }));

app.post("/clip-by-bbox", async (req, res) => {
  try {
    const {
      dataset = "NDVI",          // "NDVI" or "ET"
      dt,                        // optional: "YYYY-MM-DD"
      bbox,                      // [minx,miny,maxx,maxy] (required unless polygon provided)
      polygon,                   // optional GeoJSON Polygon
      crs = "EPSG:4326",
      driver = "GTiff",          // "GTiff" or "COG"
      resampling = "nearest"
    } = req.body || {};

    if (!bbox && !polygon) {
      return res.status(400).json({ error: "either bbox or polygon is required" });
    }
    if (bbox && (!Array.isArray(bbox) || bbox.length !== 4)) {
      return res.status(400).json({ error: "bbox must be [minx,miny,maxx,maxy]" });
    }

    // Resolve date if missing: use latest dt that intersects bbox for this dataset
    let useDt = dt;
    if (!useDt) {
      useDt = await findLatestDtForDatasetIntersectingBbox(dataset, bbox || polygonToBbox(polygon));
      if (!useDt) return res.status(404).json({ error: "No tiles found for dataset/bbox." });
    }

    // Fetch URIs for intersecting tiles on that date
    const uris = await fetchTileUris(dataset, useDt, bbox || polygonToBbox(polygon));
    if (!uris.length) return res.status(404).json({ error: "No tiles intersect your bbox for the given date." });

    // Build + return clipped mosaic
    const outFile = await mosaicAndClip({ sources: uris, bbox, polygon, crs, driver, resampling });

    const filename = `${dataset.toLowerCase()}_${useDt.toISOString?.().slice(0,10) || String(useDt)}.tif`;
    res.setHeader("Content-Type", "image/tiff");
    res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
    fs.createReadStream(outFile).on("close", () => fs.unlink(outFile, () => {})).pipe(res);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: String(err.message || err) });
  }
});

// Healthcheck
app.get("/healthz", (_req, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Raster clip service on :${PORT}`));