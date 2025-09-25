import os from "os";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import {pool} from "../dbSetup.js";


// --- Util to run a process ---
export function run(cmd, args, extraEnv = {})
{
    return new Promise((resolve, reject) => 
        {
            const child = spawn(cmd, args, 
            {
                env: { ...process.env, ...extraEnv },
                stdio: ["ignore", "pipe", "pipe"],
                windowsHide: true
            });
            let stderr = "";
            child.stderr.on("data", (d) => stderr += d.toString());
            child.on("close", (code) => {
                if(code === 0)
                    resolve();
                else
                    reject(new Error(`${cmd} exited ${code}\n${stderr}`));
            });
        });
}

// --- Helper for converting plain URL to GDAL /vsicurl/ ---
export function toVsiCurl(src)
{
    if(/^https?:\/\//i.test(src))
        return `/vsicurl/${src}`;
    return src; // local path is fine
}

// --- Temp path creation helpers ---
export function tmpPath(ext = ".tif") {
  const name = `clip_${Date.now()}_${crypto.randomBytes(4).toString("hex")}${ext}`;
  return path.join(os.tmpdir(), name);
}

export function writeTempGeoJSON(geojson) {
  const p = tmpPath(".geojson");
  fs.writeFileSync(p, JSON.stringify(geojson));
  return p;
}

// ---------- DB helpers ----------
export async function findLatestDtForDatasetIntersectingBbox(dataset, bbox) {
  // bbox: [minx,miny,maxx,maxy], EPSG:4326
  const [minx, miny, maxx, maxy] = bbox;
  const sql = `
    SELECT dt
    FROM lorawan_smart_farm.raster_asset
    WHERE dataset = $1
      AND ST_Intersects(
            bbox,
            ST_MakeEnvelope($2, $3, $4, $5, 4326)
          )
    ORDER BY dt DESC
    LIMIT 1
  `;
  const { rows } = await pool.query(sql, [dataset, minx, miny, maxx, maxy]);
  return rows[0]?.dt || null;
}

export async function fetchTileUris(dataset, dt, bbox) {
  const [minx, miny, maxx, maxy] = bbox;
  const sql = `
    SELECT uri
    FROM lorawan_smart_farm.raster_asset
    WHERE dataset = $1
      AND dt = $2
      AND ST_Intersects(
            bbox,
            ST_MakeEnvelope($3, $4, $5, $6, 4326)
          )
    ORDER BY tile_id ASC
  `;
  const { rows } = await pool.query(sql, [dataset, dt, minx, miny, maxx, maxy]);
  return rows.map(r => r.uri);
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

function buildBandArgs(info)
{
    const bands = info.bands || {};
    if (bands.length == 0 ) return [];

    const hasAlpha = bands.some(b => (b.colorInterpretation || "").toUpperCase() === "ALPHA");
    const isRGB = bands.length >= 3 &&
        ["RED","GREEN","BLUE"].every(ci =>
        bands.some(b => (b.colorInterpretation || "").toUpperCase() === ci)
        );

    // Single-band NDVI (float)
    if (bands.length === 1 && !isRGB) {
        return ["-b","1","-srcnodata","nan","-dstnodata","-9999"];
    }
    if (isRGB) {
        return ["-b","1","-b","2","-b","3","-dstalpha"];
    }
    return ["-b","1","-dstnodata","0"];
}


// ---------- Mosaic and Clip ----------
export async function mosaicAndClip({ sources, bbox, polygon, crs = "EPSG:4326", driver = "GTiff", resampling = "nearest" }) {
  if (!sources?.length) throw new Error("No sources to mosaic/clip");
  const gdalEnv = {
    GDAL_DISABLE_READDIR_ON_OPEN: "EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS: ".tif,.tiff,.TIF,.TIFF",
    VSI_CACHE: "TRUE",
    VSI_CACHE_SIZE: "10000000",
  };

  // prepare cutline
  let cutPath = null;
  if (polygon) cutPath = writeTempCutlineFromPolygon(polygon);
  else if (bbox) cutPath = writeTempCutlineFromBbox(bbox);
  else throw new Error("either bbox or polygon is required");

  // band args based on first source (assume homogeneous stack)
  const info = await getGdalInfo(sources[0], gdalEnv);
  const bandArgs = buildBandArgs(info);

  // warp all sources together + cutline in one go
  const tempWarp = tmpPath(".tif");
  const inputs = sources.map(toVsiCurl);
  const warpArgs = [
    "-q",
    "-overwrite",
    "-multi",
    "-wo", "NUM_THREADS=ALL_CPUS",
    "-cutline", cutPath,
    "-crop_to_cutline",
    "-cutline_srs", "EPSG:4326",
    "-t_srs", crs,
    "-r", resampling,
    ...bandArgs,
    ...inputs,
    tempWarp
  ];
  await run(GDAL_WARP, warpArgs, gdalEnv);

  // Build COG if needed
  /*
  const outFile = tmpPath(".tif");
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
  */

  if (cutPath) fs.unlink(cutPath, () => {});

  // sanity check
  const outInfo = await getGdalInfo(outFile, gdalEnv);
  if (!outInfo.size || outInfo.size[0] === 0 || outInfo.size[1] === 0) {
    fs.unlink(outFile, () => {});
    throw new Error("Clip produced an empty raster (no intersection?)");
  }
  return outFile;
}

// Helper to get bbox from polygon (outer ring only)
export function polygonToBbox(poly) {
  if (!poly || poly.type !== "Polygon" || !poly.coordinates?.length) return null;
  const ring = poly.coordinates[0] || [];
  let minx=Infinity, miny=Infinity, maxx=-Infinity, maxy=-Infinity;
  for (const [x,y] of ring) {
    if (x < minx) minx = x;
    if (y < miny) miny = y;
    if (x > maxx) maxx = x;
    if (y > maxy) maxy = y;
  }
  if (!isFinite(minx) || !isFinite(miny) || !isFinite(maxx) || !isFinite(maxy)) return null;
  return [minx, miny, maxx, maxy];
}