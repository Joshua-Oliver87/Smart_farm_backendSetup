import crypto from "crypto";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import { toVsiCurl, writeTempGeoJSON, run } from "../helpers/helper.js";

/*
  Adds /clip-by-bbox:
  - Looks up tile URIs in lorawan_smart_farm.raster_asset by dataset, date (optional), bbox
  - Mosaics intersecting tiles (if >1) and clips to bbox/polygon
  - Returns GTiff or COG
*/

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const GDAL_WARP = process.env.GDALWARP_PATH || "gdalwarp";
const GDAL_TRANSLATE = process.env.GDALTRANSLATE_PATH || "gdal_translate";


function writeTempCutlineFromPolygon(polygonGeom) {
  if (!polygonGeom || polygonGeom.type !== "Polygon" || !Array.isArray(polygonGeom.coordinates)) {
    throw new Error("polygon must be a GeoJSON Polygon with 'coordinates'");
  }
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

