// server.js
import express from "express";
import crypto from "crypto";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";

/*
  Will take in a tif file and set of coordinates and return a new tif file clipped to those coordinates.
  Uses gdalwarp and gdal_translate, so make sure those are installed and in your PATH.
  Depending on which is more optimal, may be modified to return raw raster pixels instead of a file.
*/

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
app.use(express.json({ limit: "2mb" }));

const GDAL_WARP = process.env.GDALWARP_PATH || "gdalwarp";
const GDAL_TRANSLATE = process.env.GDALTRANSLATE_PATH || "gdal_translate"

// --- Helpful util: run a process and stream stdio on error
function run(cmd, args, extraEnv = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {
      env: { ...process.env, ...extraEnv },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true
    });
    let stderr = "";
    child.stderr.on("data", (d) => (stderr += d.toString()));
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} exited ${code}\n${stderr}`));
    });
  });
}

// Convert plain URL to GDAL /vsicurl/ URL (for better HTTP range reads)
function toVsiCurl(src) {
  if (/^https?:\/\//i.test(src)) return `/vsicurl/${src}`;
  return src; // local path is fine
}

// Create a temp file path
function tmpPath(ext = ".tif") {
  const name = `clip_${Date.now()}_${crypto.randomBytes(4).toString("hex")}${ext}`;
  return path.join(os.tmpdir(), name);
}

// Write a small GeoJSON file when a polygon cutline is requested
function writeTempGeoJSON(geojson) {
  const p = tmpPath(".geojson");
  fs.writeFileSync(p, JSON.stringify(geojson));
  return p;
}

async function getGdalInfo(src, env) {
  return await new Promise((resolve, reject) => {
    const child = spawn(GDAL_TRANSLATE.replace("gdal_translate","gdalinfo"),
      ["-json", toVsiCurl(src)],
      { env: { ...process.env, ...env } }
    );
    let out = "", err = "";
    child.stdout.on("data", d => out += d.toString());
    child.stderr.on("data", d => err += d.toString());
    child.on("close", code => {
      if (code === 0) {
        try { resolve(JSON.parse(out)); } catch (e) { reject(e); }
      } else reject(new Error(err || `gdalinfo exited ${code}`));
    });
  });
}

function buildBandArgs(info) {
  const bands = info.bands || [];
  if (bands.length === 0) return [];

  // Look for alpha interpretation
  const hasAlpha = bands.some(b => (b.colorInterpretation || "").toUpperCase() === "ALPHA");
  const isRGB = bands.length >= 3 &&
    ["RED","GREEN","BLUE"].every(ci =>
      bands.some(b => (b.colorInterpretation || "").toUpperCase() === ci)
    );

  // Single-band NDVI (float)
  if (bands.length === 1 && !isRGB) {
    return ["-b","1","-srcnodata","nan","-dstnodata","-9999"];
  }

  // RGB(A) quicklook
  if (isRGB) {
    // Keep RGB and generate a fresh alpha from the cutline
    return ["-b","1","-b","2","-b","3","-dstalpha"];
    // If you prefer to keep the original alpha (when present), use:
    // return hasAlpha ? [] : ["-dstalpha"];
  }

  // Fallback: just take band 1 and treat it like grayscale
  return ["-b","1","-dstnodata","0"];
}

function writeTempCutlineFromPolygon(polygonGeom) {
  // Ensure valid polygon geometry
  if (!polygonGeom || polygonGeom.type !== "Polygon" || !Array.isArray(polygonGeom.coordinates)) {
    throw new Error("polygon must be a GeoJSON Polygon with 'coordinates'");
  }
  // auto-close the outer ring (and any inner rings) if needed
  const closedCoords = polygonGeom.coordinates.map(ring => {
    if (!ring.length) return ring;
    const first = ring[0], last = ring[ring.length - 1];
    const isClosed = first[0] === last[0] && first[1] === last[1];
    return isClosed ? ring : ring.concat([first]);
  });

  const featureCollection = {
    type: "FeatureCollection",
    features: [{
      type: "Feature",
      properties: {},
      geometry: { type: "Polygon", coordinates: closedCoords }
    }]
  };
  const p = tmpPath(".geojson");
  fs.writeFileSync(p, JSON.stringify(featureCollection));
  return p;
}

function writeTempCutlineFromBbox([minx, miny, maxx, maxy]) {
  const poly = {
    type: "Polygon",
    coordinates: [[
      [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]
    ]]
  };
  return writeTempCutlineFromPolygon(poly);
}


// Main endpoint
app.post("/clip-raster", async (req, res) => {
  const {
    source,
    bbox,
    polygon,
    crs = "EPSG:4326",
    driver = "GTiff", // or "COG"
    nodata = -9999,
    resampling = "nearest"
  } = req.body || {};

  if (!source) {
    return res.status(400).json({ error: "source is required" });
  }
  if (!bbox && !polygon) {
    return res.status(400).json({ error: "either bbox or polygon is required" });
  }
  if (bbox && (!Array.isArray(bbox) || bbox.length !== 4)) {
    return res.status(400).json({ error: "bbox must be [minx,miny,maxx,maxy]" });
  }

  // Output file
  const outFile = tmpPath(".tif");
  // For polygon cutline, we’ll use gdalwarp; for bbox we can use gdal_translate -projwin
  // (Both support input COGs via /vsicurl/)

  // GDAL env for COGs over HTTP (avoid directory reads, enable curl extension cache)
  const gdalEnv = {
    GDAL_DISABLE_READDIR_ON_OPEN: "EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS: ".tif,.tiff,.TIF,.TIFF",
    VSI_CACHE: "TRUE",
    VSI_CACHE_SIZE: "10000000",
  };

  // If driver is COG, use gdalwarp or gdal_translate with COG creation options
  const cogCreate =
    driver === "COG"
      ? [
          "-of", "COG",
          "-co", "COMPRESS=DEFLATE",
          "-co", "LEVEL=9",
          "-co", `RESAMPLING=${resampling}`,
          "-co", "BIGTIFF=IF_SAFER",
          "-co", "NUM_THREADS=ALL_CPUS",
        ]
      : [
          "-of", "GTiff",
          "-co", "TILED=YES",
          "-co", "COMPRESS=DEFLATE",
          "-co", "BIGTIFF=IF_SAFER"
        ];
    const input = toVsiCurl(source);

    // build a cutline file if polygon OR bbox was provided
    let cutPath = null;
    if (polygon) {
      cutPath = writeTempCutlineFromPolygon(polygon);
    } else if (bbox) {
      if (!Array.isArray(bbox) || bbox.length !== 4) {
        return res.status(400).json({ error: "bbox must be [minx,miny,maxx,maxy]" });
      }
      cutPath = writeTempCutlineFromBbox(bbox);
    }

    if (cutPath) {
      // polygon cut → gdalwarp -cutline
      const tempWarp = tmpPath(".tif");  // always warp to GTiff first
      const info = await getGdalInfo(source, gdalEnv);
      const bandArgs = buildBandArgs(info);

      const warpArgs = [
        "-q",
        "-overwrite",
        "-cutline", cutPath,
        "-crop_to_cutline",
        "-cutline_srs", "EPSG:4326",   // <<< IMPORTANT: declare cutline CRS
        // if your source is already EPSG:4326, you may omit -t_srs; otherwise keep:
        "-t_srs", crs,
        "-r", resampling,
        "-multi",
        "-wo", "NUM_THREADS=ALL_CPUS",
        ...bandArgs,
        "-of", "GTiff",
        input,
        tempWarp
      ];

      await run(GDAL_WARP, warpArgs, gdalEnv);

      // Optional: compute stats so QGIS renders immediately
      await run(GDAL_TRANSLATE.replace("gdal_translate","gdalinfo"), ["-stats", tempWarp], gdalEnv);

      // Convert to COG if requested, else rename to final
      if (driver === "COG") {
        const toCOG = [
          "-q", "-of", "COG",
          "-co", "COMPRESS=DEFLATE",
          "-co", `RESAMPLING=${resampling}`,
          "-co", "BIGTIFF=IF_SAFER",
          "-co", "NUM_THREADS=ALL_CPUS",
          tempWarp,
          outFile
        ];
        await run(GDAL_TRANSLATE, toCOG, gdalEnv);
        fs.unlink(tempWarp, () => {});
      } else {
        fs.renameSync(tempWarp, outFile);
      }

      // sanity check
      const outInfo = await getGdalInfo(outFile, gdalEnv);
      if (!outInfo.size || outInfo.size[0] === 0 || outInfo.size[1] === 0) {
        throw new Error("Clip produced an empty raster: does the polygon intersect the source?");
      }

      fs.unlink(cutPath, () => {});
    } else {
      // bbox/polygon not provided? (guard above already ensures one is present)
      return res.status(400).json({ error: "either bbox or polygon is required" });
    }

    // stream result
    const filename = `clip_${Date.now()}.tif`;
    res.setHeader("Content-Type", "image/tiff");
    res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
    fs.createReadStream(outFile).on("close", () => fs.unlink(outFile, () => {})).pipe(res);
});

// Healthcheck
app.get("/healthz", (_req, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Raster clip service on :${PORT}`));
