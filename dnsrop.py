#!/usr/bin/env python3

from __future__ import annotations

import base64
import bz2
import gzip
import io
import json
import math
import os
import re
import struct
import subprocess
import tempfile
import time
import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import threading
import concurrent.futures


try:
    import h5py  # type: ignore
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore
    HAS_RADAR_DEPS = True
    RADAR_DEPS_ERROR = ""
except Exception as exc:  # pragma: no cover
    HAS_RADAR_DEPS = False
    RADAR_DEPS_ERROR = str(exc)

try:
    import netCDF4  # type: ignore
    HAS_NETCDF4 = True
    NETCDF4_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    netCDF4 = None  # type: ignore
    HAS_NETCDF4 = False
    NETCDF4_IMPORT_ERROR = str(exc)

try:
    import xarray as xr  # type: ignore
    import xradar as xd  # type: ignore
    HAS_XRADAR = True
    XRADAR_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    xr = None  # type: ignore
    xd = None  # type: ignore
    HAS_XRADAR = False
    XRADAR_IMPORT_ERROR = str(exc)

try:
    import unravel  # type: ignore
    HAS_UNRAVEL = True
    UNRAVEL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    unravel = None  # type: ignore
    HAS_UNRAVEL = False
    UNRAVEL_IMPORT_ERROR = str(exc)

try:
    import rasterio  # type: ignore
    from rasterio.windows import from_bounds as _rio_win_from_bounds  # type: ignore
    from rasterio.transform import from_bounds as _rio_transform_from_bounds  # type: ignore
    from rasterio.enums import Resampling as _Resampling  # type: ignore
    from rasterio.warp import reproject as _rio_reproject  # type: ignore
    from rasterio.crs import CRS as _CRS  # type: ignore
    HAS_RASTERIO = True
except Exception:  # pragma: no cover
    rasterio = None  # type: ignore
    _Resampling = None  # type: ignore
    HAS_RASTERIO = False

try:
    from scipy.ndimage import zoom as _scipy_zoom          # type: ignore
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter  # type: ignore
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_zoom = None              # type: ignore
    _scipy_gaussian_filter = None   # type: ignore
    HAS_SCIPY = False

# ── ICON-D2 post-processing knobs ─────────────────────────────────────────────
# Upsampling factor applied to the native ICON-D2 regular lat/lon grid before
# the PNG is encoded.  3× raises the effective resolution from ~2 km to ~0.7 km
# so Mapbox's WebGL raster resampling never blurs visible grid squares.
ICON_D2_UPSAMPLE_FACTOR: int = 3
# Gaussian sigma in upsampled pixels.  1.5 px ≈ one native cell → smooths
# the staircase artefacts introduced by the zoom without smearing fronts.
ICON_D2_SMOOTH_SIGMA: float = 1.5
# Any simulated reflectivity below this threshold is treated as no-echo (NaN).
ICON_D2_MIN_DBZ: float = 5.0


def _icon_d2_postprocess(grid: "np.ndarray") -> "np.ndarray":
    """Upsample, smooth, and threshold an ICON-D2 dBZ grid.

    Steps
    -----
    1. Mask values below ICON_D2_MIN_DBZ as NaN (remove noise / clear-air
       return artefacts).
    2. Upsample by ICON_D2_UPSAMPLE_FACTOR using bicubic interpolation on the
       filled data and bilinear interpolation on the validity mask so the
       rendered image is crisp at the zoom levels typically used in the
       dashboard.
    3. Apply a light Gaussian blur (sigma = ICON_D2_SMOOTH_SIGMA upsampled
       pixels) so the blocky native GRIB cells blend into a smooth mesh.

    Falls back to the threshold-only operation when scipy is not available.
    """
    # Step 1 – threshold
    below = np.isfinite(grid) & (grid < ICON_D2_MIN_DBZ)
    grid = grid.copy()
    grid[below] = np.nan

    if not HAS_SCIPY or ICON_D2_UPSAMPLE_FACTOR <= 1:
        return grid

    valid = np.isfinite(grid)

    # Fill NaN with 0 for the cubic-spline zoom (avoids NaN propagation).
    # We upscale the validity mask separately (bilinear → threshold at 0.5)
    # to recover which output pixels are actually valid.
    filled = np.where(valid, grid, 0.0).astype(np.float64)

    factor = float(ICON_D2_UPSAMPLE_FACTOR)

    # Upsample data with bicubic (order=3); use prefilter=False to avoid
    # ringing at the data/zero boundary.
    up_data = _scipy_zoom(filled, factor, order=3, prefilter=False)

    # Upsample mask with bilinear (order=1) and threshold at 0.5 so mask
    # edges stay tight.
    up_mask = _scipy_zoom(valid.astype(np.float64), factor, order=1) >= 0.5

    # Step 3 – Gaussian smoothing in float space (only on valid pixels).
    # We smooth the filled field and then re-apply the mask so edge pixels
    # don't bleed into the no-echo region.
    up_data = _scipy_gaussian_filter(up_data, sigma=ICON_D2_SMOOTH_SIGMA)
    up_data[~up_mask] = np.nan

    return up_data.astype(np.float32)


# Path to the Blue Marble GeoTIFF (same directory as this script)
_BLUEMARBLE_TIFF: Optional[Path] = None
_BLUEMARBLE_DS = None   # open rasterio dataset, lazily loaded
_BLUEMARBLE_LOCK = threading.Lock()

BLUEMARBLE_TIFF_NAME = "world.topo.bathy.200407.3x21600x21600.C1_geo.tiff"

def _get_bluemarble_ds():
    """Lazily open the Blue Marble GeoTIFF dataset (thread-safe)."""
    global _BLUEMARBLE_DS, _BLUEMARBLE_TIFF
    with _BLUEMARBLE_LOCK:
        if _BLUEMARBLE_DS is not None:
            return _BLUEMARBLE_DS
        if not HAS_RASTERIO:
            return None
        candidate = Path(__file__).resolve().parent / BLUEMARBLE_TIFF_NAME
        if not candidate.exists():
            return None
        try:
            _BLUEMARBLE_TIFF = candidate
            _BLUEMARBLE_DS = rasterio.open(str(candidate))
            return _BLUEMARBLE_DS
        except Exception as exc:
            print(f"[bluemarble] Failed to open GeoTIFF: {exc}")
            return None


def _serve_bluemarble_tile(z: int, x: int, y: int) -> Optional[bytes]:
    """Return a 256×256 PNG for the given Web-Mercator tile.

    The C1 GeoTIFF is in WGS84 (EPSG:4326) and covers 0°–90°E, 0°–90°N.
    Tiles are properly warped to Web Mercator (EPSG:3857) via rasterio.warp
    so that Mercator latitude stretching is correctly compensated.
    Tiles outside the C1 footprint receive a solid-black fill.
    """
    if not HAS_RASTERIO:
        return None
    try:
        import math as _math

        # ── C1 GeoTIFF coverage (WGS84) ──────────────────────────────────────
        TIFF_LON_MIN, TIFF_LON_MAX =  0.0,  90.0
        TIFF_LAT_MIN, TIFF_LAT_MAX =  0.0,  90.0

        # ── WGS84 lat/lon bounds of the requested tile (for coverage test) ────
        n = 2 ** z
        lon_w = x / n * 360.0 - 180.0
        lon_e = (x + 1) / n * 360.0 - 180.0
        lat_n = _math.degrees(_math.atan(_math.sinh(_math.pi * (1 - 2 * y / n))))
        lat_s = _math.degrees(_math.atan(_math.sinh(_math.pi * (1 - 2 * (y + 1) / n))))

        # Helper: solid-black 256×256 PNG
        def _black_tile() -> bytes:
            img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8), "RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=False)
            return buf.getvalue()

        # Tile completely outside C1 footprint → black
        if (lon_e <= TIFF_LON_MIN or lon_w >= TIFF_LON_MAX
                or lat_n <= TIFF_LAT_MIN or lat_s >= TIFF_LAT_MAX):
            return _black_tile()

        ds = _get_bluemarble_ds()
        if ds is None:
            return None

        # -Web Mercator bounds of this tile in metres (EPSG:3857)
        # Tile row/col maps linearly to Mercator Y/X:
        #   merc_x = R * lon_rad
        #   merc_y = π * R * (1 − 2y/n)   (exact, no transcendentals needed)
        R = 6_378_137.0                       # WGS84 semi-major axis (metres)
        MERC_MAX = _math.pi * R               # ≈ 20 037 508 m
        merc_w = _math.radians(lon_w) * R
        merc_e = _math.radians(lon_e) * R
        merc_n_m = MERC_MAX * (1.0 - 2.0 * y / n)
        merc_s_m = MERC_MAX * (1.0 - 2.0 * (y + 1) / n)

        # Destination affine transform: 256×256 pixels → the mercator tile
        dst_transform = _rio_transform_from_bounds(merc_w, merc_s_m, merc_e, merc_n_m,
                                                   256, 256)
        dst_crs = _CRS.from_epsg(3857)
        dst_data = np.zeros((3, 256, 256), dtype=np.uint8)

        # -Read a padded WGS84 window from the GeoTIFF
        # Clip to the C1 extent and add 10 % padding so the warp has sufficient
        # source pixels at all four edges (avoids black fringe artefacts).
        read_lon_w = max(lon_w, TIFF_LON_MIN)
        read_lon_e = min(lon_e, TIFF_LON_MAX)
        read_lat_s = max(lat_s, TIFF_LAT_MIN)
        read_lat_n = min(lat_n, TIFF_LAT_MAX)
        pad = max((read_lon_e - read_lon_w), (read_lat_n - read_lat_s)) * 0.10
        read_lon_w = max(read_lon_w - pad, TIFF_LON_MIN)
        read_lon_e = min(read_lon_e + pad, TIFF_LON_MAX)
        read_lat_s = max(read_lat_s - pad, TIFF_LAT_MIN)
        read_lat_n = min(read_lat_n + pad, TIFF_LAT_MAX)

        with _BLUEMARBLE_LOCK:
            src_crs = ds.crs if ds.crs else _CRS.from_epsg(4326)
            win = _rio_win_from_bounds(read_lon_w, read_lat_s,
                                       read_lon_e, read_lat_n,
                                       ds.transform)
            win_transform = ds.window_transform(win)
            # Read at capped resolution (max 1024 px per side) to bound memory
            win_width  = max(1, int(round(win.width)))
            win_height = max(1, int(round(win.height)))
            read_w = min(win_width,  1024)
            read_h = min(win_height, 1024)
            src_data = ds.read(
                [1, 2, 3],
                window=win,
                out_shape=(3, read_h, read_w),
                resampling=_Resampling.bilinear,
            )
            # Recompute the transform for the (possibly downsampled) read
            x_scale = win_width  / read_w
            y_scale = win_height / read_h
            from rasterio.transform import Affine as _Affine  # type: ignore
            scaled_transform = win_transform * _Affine.scale(x_scale, y_scale)

        # -Warp WGS84 source → Web Mercator destination
        _rio_reproject(
            source=src_data,
            destination=dst_data,
            src_transform=scaled_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=_Resampling.bilinear,
        )

        img = Image.fromarray(np.transpose(dst_data, (1, 2, 0)).astype("uint8"), "RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return buf.getvalue()

    except Exception as exc:
        print(f"[bluemarble tile] z={z} x={x} y={y} error: {exc}")
        return None


MAPBOX_STYLES = {
    "basemap": {
        "label": "Basemap",
        "style": "mapbox://styles/noabaltzer/cmmkdfjlu007401sb6b2j6ajo",
    },
    "greyscale": {
        "label": "DNSROP",
        "style": "mapbox://styles/noabaltzer/cmolnvfp2002901qy58wzeyl4",
    },
}
DEFAULT_MAPBOX_STYLE_KEY = "basemap"
MAPBOX_ACCESS_TOKEN = (
    "pk.eyJ1Ijoibm9hYmFsdHplciIsImEiOiJjbW5xemI1eGMwYXBpMnBzZDh4MDFtcXlwIn0."
    "5GgkXGzlxewQa21Cdi6Pvw"
)
DMI_API_KEY = "dnsrop-dmi"
# Norway Frost API client ID – register free at https://frost.met.no/auth/requestCredentials
# Leave empty to disable Norway metObs.
FROST_CLIENT_ID = "b05080c1-d8fb-4485-81be-824a27e73b40"
# KNMI Open Data API key (anonymous demo key; replace with your own registered key for production)
KNMI_API_KEY = (
    "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjY1ODRmNTRmOGU2ODQ2NDg4OWYyOGI5NmVkOGE1MzlkIiwiaCI6Im11cm11cjEyOCJ9"
)
# DWD synoptic GeoJSON (bz2-compressed, server-side fetch)
DWD_METOBS_BASE = "https://opendata.dwd.de/weather/weather_reports/synoptic/germany/geojson/"
DMI_VOLUME_ITEMS_URL = "https://opendataapi.dmi.dk/v1/radardata/collections/volume/items"

# ESWD (European Severe Weather Database) reports — https://eswd.eu/en/api/docs
# The v2 pull API expects an Authorization **Bearer** token (account API access),
# not HTTP Basic with a short word. Set `ESWD_API_TOKEN` in the environment before
# starting the server; without it the overlay returns an empty layer and skips the request.
ESWD_API_BASE = "https://eswd.eu/api/v2/reportList"
ESWD_TYPES = "HAIL,PRECIP,TORNADO,WIND"
ESWD_LEVELS = "QC0,QC0+,QC1,QC2"
# Denmark bounding box: lat_north, lat_south, lon_west, lon_east
ESWD_DK_X0    = 58.0   # lat north
ESWD_DK_X1    = 54.5   # lat south
ESWD_DK_Y0    =  7.5   # lon west
ESWD_DK_Y1    = 15.5   # lon east

# ICON-D2 NWP products (Germany regular lat/lon grid)
ICON_D2_BASE_URL = "https://opendata.dwd.de/weather/nwp/icon-d2/grib"
ICON_D2_RUN_HOURS = ("00", "03", "06", "09", "12", "15", "18", "21")
ICON_D2_INVENTORY_TTL_S = 5 * 60
ICON_D2_FILE_CACHE_MAX = 10

# Per-product configuration:
#   dir            – subdirectory under the run-hour folder on DWD opendata
#   filename_suffix – the variable part of the GRIB filename (after "2d_")
#   label           – human-readable display label
#   colormap        – key into COLORMAPS / JS getColormap()
#   unit_scale      – multiply raw GRIB values by this before encoding
#                     (1.0 = no change, 0.001 = m → km for echotop)
ICON_D2_PRODUCTS: Dict[str, Dict[str, Any]] = {
    "dbz_cmax": {
        "dir": "dbz_cmax",
        "filename_suffix": "dbz_cmax",
        "label": "Simulated Reflectivity",
        "colormap": "reflectivity",
        "unit_scale": 1.0,
    },
    "cape_ml": {
        "dir": "cape_ml",
        "filename_suffix": "cape_ml",
        "label": "CAPE (ML)",
        "colormap": "cape_ml",
        "unit_scale": 1.0,
    },
    "echotop": {
        "dir": "echotop",
        "filename_suffix": "echotop",
        "label": "Echo Top",
        "colormap": "echo_tops",
        "unit_scale": 0.001,   # raw GRIB values in metres - convert to km
    },
}
ICON_D2_DEFAULT_PRODUCT = "dbz_cmax"

def _icon_d2_file_re(filename_suffix: str) -> "re.Pattern[str]":
    """Return a compiled regex for a given ICON-D2 GRIB filename suffix."""
    return re.compile(
        rf"icon-d2_germany_regular-lat-lon_single-level_(?P<run>\d{{10}})_(?P<fh>\d{{3}})_2d_{re.escape(filename_suffix)}\.grib2\.bz2$",
        re.IGNORECASE,
    )

# Per-product inventory and file caches (keyed by product key)
_ICON_D2_INVENTORY_CACHES: Dict[str, Dict[str, Any]] = {
    k: {"expires": 0.0, "data": None} for k in ICON_D2_PRODUCTS
}
_ICON_D2_INVENTORY_LOCK = threading.Lock()
_ICON_D2_FILE_CACHES: Dict[str, Dict[str, Dict[str, Any]]] = {
    k: {} for k in ICON_D2_PRODUCTS
}
_ICON_D2_FILE_CACHE_LOCK = threading.Lock()

# SMHI single-site radar (Sweden) – volume product "qcvol"
SMHI_VERSION = "latest"
SMHI_BASE_URL = "https://opendata-download-radar.smhi.se/api/version"
SMHI_QCVOL_PRODUCT = "qcvol"
# DWD (Germany) single-site sweep volumes
DWD_BASE_URL = "https://opendata.dwd.de/weather/radar/sites"
DWD_SWEEP_DIRS = {
    "reflectivity": "sweep_vol_z",
    "velocity": "sweep_vol_v",
    "correlation": "sweep_vol_rhohv",
}
DWD_SITE_DISCOVERY_PRODUCT = "reflectivity"
DWD_SITE_LIMIT = 32
DWD_HREF_RE = re.compile(r"""href=["']([^"']+)["']""", flags=re.IGNORECASE)
DWD_TS_RE = re.compile(r"(20\d{12,14})")
# FMI (Finland) – public S3 bucket
FMI_S3_BASE = "https://fmi-opendata-radar-volume-hdf5.s3.amazonaws.com"
FMI_S3_REGION_BASE = "https://fmi-opendata-radar-volume-hdf5.s3.eu-west-1.amazonaws.com"
# Match any HDF5 key in an S3 ListObjectsV2 XML response.
# FMI names volume files like:  202503071200_fikor_PVOL.h5
# (some older files used _pvol.h5 lower-case; both matched by case-insensitive flag)
FMI_S3_KEY_RE = re.compile(r"<Key>([^<]+\.h5(?:\.bz2|\.gz)?)</Key>", re.IGNORECASE)

KNMI_BASE_URL = "https://api.dataplatform.knmi.nl/open-data/v1"

# DISABLED: Meteo-France (AERIS/SEDOO) radial products — commented out (constantly failing)
# FR_RADAR_DOWNLOAD_URL = "https://api.sedoo.fr/radarsmf-rest/v3/download"
# FR_SELECTION_FILENAME = "selections_radars_*.txt"  # glob – auto-discovers any downloaded selection file
# FR_SELECTION_URL_RE = re.compile(
#     r'https?://api\.sedoo\.fr/radarsmf-rest/v3/download\?([^"\s]+)',
#     re.IGNORECASE,
# )
# FR_PRODUCT_CODE_RE = re.compile(r"(PAG_\d+)_", re.IGNORECASE)

# CHMI (Czech Republic) radar – directory-served HDF5 volumes
CHMI_RADAR_BASE = "https://opendata.chmi.cz/meteorology/weather/radar/sites"
# Per-variable subdirectory names for CHMI
CHMI_VAR_DIRS = {
    "reflectivity":   "vol_z",
    "velocity":       "vol_v",
    "spectrum_width": "vol_w",
    "zdr":            "vol_zdr",
    "correlation":    "vol_rhohv",
    "phidp":          "vol_phidp",
    "u":              "vol_u",
}
CHMI_HDF5_RE = re.compile(
    r'href=["\']([^"\']*\.hdf5?(?:\.gz|\.bz2)?)["\']', re.IGNORECASE
)

# SHMU (Slovakia) radar – directory-served HDF5 volumes
SHMU_RADAR_BASE = "https://opendata.shmu.sk/meteorology/weather/radar/volume"
# SHMU listings use both day directories (YYYYMMDD) and file stamps
# (YYYYMMDDHHMM[SS]); accept all of them for discovery/sorting.
SHMU_TS_RE = re.compile(r"(20\d{6,12})")

# Romanian radar (MeteoRomania) – Apache-style directory listings with HDF files
ROMANIA_RADAR_BASE = "https://opendata.meteoromania.ro/radar"
ROMANIA_HDF_RE = re.compile(
    r'href=["\']([^"\']*\.hdf(?:5)?(?:\.gz|\.bz2)?)["\']',
    re.IGNORECASE,
)
ROMANIA_TS_RE = re.compile(r"_(20\d{12,14})")
ROMANIA_MOMENT_MARKERS = {
    "reflectivity": ("DBZH", "DBZ", "ZH", "REFL", "TH"),
    # boundary-aware handler (requires [_-]V(?:[._-]|$)) which is exact.
    "velocity": ("VRAD", "VEL", "V", "_V."),
    "correlation": ("RHOHV", "RHO", "CC"),
    "spectrum_width": ("WRAD", "WIDTH", "SW", "W"),
    "zdr": ("ZDR",),
    "kdp": ("KDP",),
}

# KNMI Cabauw (CESAR IDRA) NetCDF dataset
CABAUW_DATASET = "cesar_idra_reflctivity_la1_t00"
CABAUW_VERSION = "v1.0"

# GeoSphere Austria (Austria) – HTTP filelisting, Bearer-token auth
GEOSPHERE_BUCKET_URL = "https://public.hub.geosphere.at/datahub"
GEOSPHERE_BUCKET_ROOT = "resources/"
GEOSPHERE_LISTING_BASE = "https://public.hub.geosphere.at/resources"
GEOSPHERE_TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJqdGkiOiJLMmlvOHZkb2J0VXhOMlpxUDNKYmY3SFRwRmZlY1pJQVBQVHJILVV1T29FIiwi"
    "aWF0IjoxNzc0NDY2OTc2fQ"
    ".UcLgLP6IgV2hB_jkWZwbyUE4_mXMIklohZUNcMWB6sA"
)
# Matches filenames like WXRHOF_202603230000.hdf inside the listing HTML
GEOSPHERE_HDF_RE = re.compile(
    r'href=["\']([^"\']*WXRHOF_(\d{12})\.hdf)["\']', re.IGNORECASE
)
GEOSPHERE_S3_KEY_RE = re.compile(
    r"<Key>([^<]*WXRHOF_(\d{12})\.hdf)</Key>", re.IGNORECASE
)
GEOSPHERE_S3_NEXT_TOKEN_RE = re.compile(
    r"<NextContinuationToken>([^<]+)</NextContinuationToken>", re.IGNORECASE
)

# Meteogate (EUMETNet) OGC API – Norwegian (MET Norway) volume radar -- 
# do note, this is a test integration and does not work
METEOGATE_BASE_URL = "https://api.meteogate.eu/test/radar"
METEOGATE_COLL_BASE = (
    f"{METEOGATE_BASE_URL}/eu-eumetnet-weather-radar/collections/observations"
)
METEOGATE_LOCATIONS_URL = f"{METEOGATE_COLL_BASE}/locations"
METEOGATE_ITEMS_URL = f"{METEOGATE_BASE_URL}/collections/observations/items"
# Norway bounding box used for radar discovery (lon_min,lat_min,lon_max,lat_max)
_NORWAY_BBOX = "4.0,57.0,32.0,72.0"
# Norwegian Meteogate test endpoint requested by user.
METEOGATE_TEST_LOCATION_URL = (
    "https://radar.meteogate.eu/api/collections/observations/locations"
)
METEOGATE_TEST_DEFAULT_PARAMS = {
    "datetime": "2026-01-01T00:00Z/2029-01-01T00:00Z",
    "f": "CoverageJSON",
    "standard_name": "DBZH, TH, VRAD, RATE",
    "level": "1.0/10.0",
    "format": "ODIM",
    "method": "scan",
}

# Estonian KAIA API endpoint (download file by item ID)
ESTONIA_KAIA_BASE = "https://avaandmed.keskkonnaportaal.ee/api/lists/active/items"

# ARPA Piemonte (Italy) – Apache-style directory listings with HDF5 volume files
# Two C-band radars: Bric della Croce (bric) and Settepani (sett)
ARPA_PIE_BASE = "https://www.arpa.piemonte.it/rischi_naturali/radar"
# Moment suffixes embedded in ARPA Piemonte filenames
# Files follow: {PAGZ4x}_C_PIEM_{YYYYMMDDHHMMSS}{moment}.h5
ARPA_PIE_MOMENTS = ("dBZ", "dBuZ", "Vu", "RhoHVu", "ZDRu", "uPhiDPu")
# Regex to extract the 14-digit timestamp (YYYYMMDDHHMMSS) from the filename
ARPA_PIE_TS_RE = re.compile(r"(20\d{12})")
# Regex to match HDF5 hrefs in Apache directory listings
ARPA_PIE_HDF5_RE = re.compile(r'href=["\']([^"\']*\.h5)["\']', re.IGNORECASE)

# ARPA Lombardia (Italy) – Apache-style directory listings with HDF5 volume files
# Two C-band radars: Desio (near Milan) and Flero (near Brescia)
# New filename convention: {StationName}.{YYYYMMDD}T{HHMMSS}Z_{MOMENT}.h5.gz
#   e.g. Desio.20250929T143000Z_DBZH.h5.gz
ARPA_LOM_BASE = "https://radarlive.arpalombardia.it/Volumi"
# ISO-8601 timestamp embedded in new filenames: YYYYMMDDTHHMMSSZ
ARPA_LOM_TS_RE = re.compile(r"(20\d{6}T\d{6}Z)")
# Match both plain .h5 and gzip-compressed .h5.gz hrefs in Apache directory listings
ARPA_LOM_HDF5_RE = re.compile(r'href=["\']([^"\']*\.h5(?:\.gz)?)["\']', re.IGNORECASE)

# Cartesian resolution for DERIVED products only (echo tops, nrot, etc.)
DERIVED_CARTESIAN_SIZE = 2048

MAX_CS_BINS = 300
MAX_CS_SWEEPS = 20

_BUILD_LOCK = threading.Lock()
_DWD_STATION_CACHE: Optional[Dict[str, dict]] = None
# DISABLED (SEDOO): _FR_STATION_CACHE: Optional[Dict[str, dict]] = None
# DISABLED (SEDOO): _FR_SELECTION_CACHE: Optional[Dict[str, List[dict]]] = None
_FR_STATION_CACHE: Optional[Dict[str, dict]] = {}      # stubbed out — SEDOO disabled
_FR_SELECTION_CACHE: Optional[Dict[str, List[dict]]] = {}  # stubbed out — SEDOO disabled
_METEOGATE_STATION_CACHE: Optional[Dict[str, dict]] = None
_TVS_ICON_CACHE: Optional[Dict[str, str]] = None
# Guards concurrent writes to the three station discovery caches above so that
# simultaneous requests cannot trigger duplicate expensive network fetches.
_STATION_CACHE_LOCK = threading.Lock()
# Keep --serve startup responsive by prefetching only core providers.
# Heavier sources are fetched on-demand when a station is selected.
SERVE_PREFETCH_PROVIDERS = {
    "dmi", "smhi", "fmi", "dwd", "knmi", "knmi_cabauw",
    "chmi", "shmu", "geosphere", "meteogate",  # "fr_mf" disabled — SEDOO constantly failing
    "romania", "estonia", "arpa_piemonte", "arpa_lombardia",
}

# -Live-payload cache (serves latest data instantly)
_PAYLOAD_CACHE: Optional[bytes] = None          # JSON bytes of last build
_PAYLOAD_CACHE_TIME: float = 0.0                # time.time() of last build
_PAYLOAD_CACHE_LOCK = threading.Lock()
_PAYLOAD_CACHE_INTERVAL = 15 * 60              # seconds between background refreshes

# -Force-refresh scheduling 
# A force-refresh sets _NEXT_PREFETCH_TIME into the future so the background
# prefetch thread doesn't immediately clobber the freshly-built cache.
_NEXT_PREFETCH_TIME: float = 0.0               # earliest time bg thread may next build
_NEXT_PREFETCH_LOCK = threading.Lock()

# -Azimuthal shear composite ring buffer 
# Stores the last _AZSHEAR_HISTORY_MAXLEN raw polar float32 arrays per station
# so the composite endpoint can merge them without extra network fetches.
_AZSHEAR_HISTORY: Dict[str, List[dict]] = {}   # station -> [{"polar":…,"meta":…}, …]
_AZSHEAR_HISTORY_MAXLEN = 10
_AZSHEAR_HISTORY_LOCK = threading.Lock()

# -Custom radar uploads (in-memory; cleared on server restart) 
# Each entry: {"provider": "custom", "file_bytes": bytes, "lat": float,
#              "lon": float, "range_km": float, "filename": str}
_CUSTOM_STATIONS: Dict[str, dict] = {}
_CUSTOM_RENDERED: Dict[str, dict] = {}
_CUSTOM_LOCK = threading.Lock()

STATIONS = {
    # Denmark (DMI)
    "Rømø":     {"provider": "dmi",  "code": "dkrom", "lat": 55.1725903,  "lon":  8.55052996, "range_km": 120.0},
    "Sindal":   {"provider": "dmi",  "code": "dksin", "lat": 57.48876226, "lon": 10.13511376, "range_km": 120.0},
    "Stevns":   {"provider": "dmi",  "code": "dkste", "lat": 55.32561875, "lon": 12.44817293, "range_km": 120.0},
    "Bornholm": {"provider": "dmi",  "code": "dkbor", "lat": 55.11283297, "lon": 14.8874575,  "range_km": 120.0},
    "Samsø":    {"provider": "dmi",  "code": "dksam", "lat": 55.812009,   "lon": 10.585485,   "range_km": 120.0},

    # Sweden (SHMI)
    "Ängelholm":   {"provider": "smhi", "area": "angelholm",    "lat": 56.29, "lon": 12.85, "range_km": 240.0},
    "Åtvidaberg":  {"provider": "smhi", "area": "atvidaberg",   "lat": 58.21, "lon": 16.01, "range_km": 240.0},
    "Bålsta":      {"provider": "smhi", "area": "balsta",       "lat": 59.57, "lon": 17.52, "range_km": 240.0},
    "Hemse":       {"provider": "smhi", "area": "hemse",        "lat": 57.25, "lon": 18.37, "range_km": 240.0},
    "Hudiksvall":  {"provider": "smhi", "area": "hudiksvall",   "lat": 61.73, "lon": 17.08, "range_km": 240.0},
    "Karlskrona":  {"provider": "smhi", "area": "karlskrona",   "lat": 56.17, "lon": 15.58, "range_km": 240.0},
    "Kiruna":      {"provider": "smhi", "area": "kiruna",       "lat": 67.86, "lon": 20.23, "range_km": 240.0},
    "Leksand":     {"provider": "smhi", "area": "leksand",      "lat": 60.73, "lon": 14.99, "range_km": 240.0},
    "Luleå":       {"provider": "smhi", "area": "lulea",        "lat": 65.58, "lon": 22.17, "range_km": 240.0},
    "Örnsköldsvik":{"provider": "smhi", "area": "ornskoldsvik", "lat": 63.29, "lon": 18.72, "range_km": 240.0},
    "Östersund":   {"provider": "smhi", "area": "ostersund",    "lat": 63.18, "lon": 14.50, "range_km": 240.0},
    "Vara":        {"provider": "smhi", "area": "vara",         "lat": 58.25, "lon": 12.95, "range_km": 240.0},

    # Finland (FMI)
    "Korppoo":    {"provider": "fmi", "site": "fikor", "lat": 60.1288, "lon": 21.6339, "range_km": 250.0},
    "Vihti":      {"provider": "fmi", "site": "fivnt", "lat": 60.4699, "lon": 24.2581, "range_km": 250.0},
    "Anjalankoski":{"provider": "fmi","site": "fianj", "lat": 60.9056, "lon": 26.8924, "range_km": 250.0},
    "Kankaanpää": {"provider": "fmi", "site": "fikan", "lat": 61.7863, "lon": 22.6913, "range_km": 250.0},
    "Kesälahti":  {"provider": "fmi", "site": "fikes", "lat": 61.9368, "lon": 29.4447, "range_km": 250.0},
    "Petäjävesi": {"provider": "fmi", "site": "fipet", "lat": 62.3019, "lon": 25.4465, "range_km": 250.0},
    "Kuopio":     {"provider": "fmi", "site": "fikuo", "lat": 63.1183, "lon": 27.3808, "range_km": 250.0},
    "Vimpeli":    {"provider": "fmi", "site": "fivim", "lat": 63.1025, "lon": 23.8218, "range_km": 250.0},
    "Nurmes":     {"provider": "fmi", "site": "finur", "lat": 63.8399, "lon": 29.5157, "range_km": 250.0},
    "Utajärvi":   {"provider": "fmi", "site": "fiuta", "lat": 64.7744, "lon": 26.3196, "range_km": 250.0},
    "Luosto":     {"provider": "fmi", "site": "filuo", "lat": 67.1394, "lon": 26.8964, "range_km": 250.0},
    "Kaunispää":  {"provider": "fmi", "site": "fiika", "lat": 68.4703, "lon": 27.1068, "range_km": 250.0},

    # Netherlands (KNMI)
    "Herwijnen": {
        "provider": "knmi",
        "dataset": "radar_volume_full_herwijnen",
        "version": "1.0",
        "lat": 51.837,
        "lon": 5.138,
        "range_km": 320.0,
    },
    "Den Helder": {
        "provider": "knmi",
        "dataset": "radar_volume_denhelder",
        "version": "2.0",
        "lat": 52.955,
        "lon": 4.788,
        "range_km": 320.0,
    },

    # Austria (GeoSphere Austria)
    "Hochficht": {
        "provider": "geosphere",
        "lat": 48.6280,
        "lon": 13.9703,
        "range_km": 200.0,
    },

    # Czech Republic (CHMI)
    "Skalky": {
        "provider": "chmi",
        "site": "ska",
        "lat": 49.5017,
        "lon": 17.8447,
        "range_km": 256.0,
    },

    # Slovakia (SHMU)
    "Javorník": {"provider": "shmu", "radar": "skjav", "lat": 49.20, "lon": 18.20, "range_km": 250.0},
    "Kojšovská hoľa": {"provider": "shmu", "radar": "skkoj", "lat": 48.80, "lon": 20.95, "range_km": 250.0},
    "Kubínska hoľa": {"provider": "shmu", "radar": "skkub", "lat": 49.20, "lon": 19.30, "range_km": 250.0},
    "Lazany": {"provider": "shmu", "radar": "sklaz", "lat": 48.65, "lon": 18.60, "range_km": 250.0},

    # Romania (MeteoRomania)
    "Romania BAR": {"provider": "romania", "site": "BAR", "lat": 47.12, "lon": 27.63, "range_km": 250.0},
    "Romania BOB": {"provider": "romania", "site": "BOB", "lat": 46.53, "lon": 24.34, "range_km": 250.0},
    "Romania BUC": {"provider": "romania", "site": "BUC", "lat": 44.50, "lon": 26.13, "range_km": 250.0},
    "Romania CRA": {"provider": "romania", "site": "CRA", "lat": 44.32, "lon": 23.88, "range_km": 250.0},
    "Romania MED": {"provider": "romania", "site": "MED", "lat": 44.25, "lon": 28.27, "range_km": 250.0},
    "Romania ORA": {"provider": "romania", "site": "ORA", "lat": 47.05, "lon": 21.90, "range_km": 250.0},
    "Romania TIM": {"provider": "romania", "site": "TIM", "lat": 45.76, "lon": 21.25, "range_km": 250.0},

    # Estonia (KAIA)
    "Estonia HAR (KAIA)": {
        "provider": "estonia",
        "item_id": 6522843,
        "lat": 59.40,
        "lon": 24.60,
        "range_km": 250.0,
    },
    "Estonia SUR (KAIA)": {
        "provider": "estonia",
        "item_id": 6522907,
        "lat": 58.48,
        "lon": 25.52,
        "range_km": 250.0,
    },

    # Norway (Meteogate test URL) - does not work, but included for demonstration of OGC API approach
    "Norway Test 0-20000-0-01498": {
        "provider": "meteogate_test",
        "wigos_id": "0-20000-0-01498",
        "lat": 63.0,
        "lon": 11.0,
        "range_km": 240.0,
    },

    # Italy - ARPA Piemonte open-data radar
    "Bric della Croce": {
        "provider": "arpa_piemonte",
        "site": "bric",
        "lat": 44.684,
        "lon": 7.761,
        "range_km": 200.0,
    },
    "Settepani": {
        "provider": "arpa_piemonte",
        "site": "sett",
        "lat": 44.263,
        "lon": 8.183,
        "range_km": 200.0,
    },

    # Italy - ARPA Lombardia open-data radar
    # Files live at ARPA_LOM_BASE/DES/ and ARPA_LOM_BASE/FLE/
    # New filename convention: {StationName}.{YYYYMMDD}T{HHMMSS}Z_{MOMENT}.h5.gz
    #   e.g. https://radarlive.arpalombardia.it/Volumi/DES/Desio.20250929T143000Z_DBZH.h5.gz
    "Desio": {
        "provider": "arpa_lombardia",
        "site": "DES",
        "lat": 45.617,
        "lon": 9.211,
        "range_km": 200.0,
    },
    "Flero": {
        "provider": "arpa_lombardia",
        "site": "FLE",
        "lat": 45.494,
        "lon": 10.132,
        "range_km": 200.0,
    },

    # Netherlands (Cabauw / CESAR IDRA)
    "Cabauw": {"provider": "knmi_cabauw", "lat": 51.971, "lon": 4.927, "range_km": 60.0},
}

PRODUCTS = {
    "reflectivity":   {"label": "Reflectivity (dBZ)",                "quantities": ["DBZHC", "DBZH", "TH", "DBTH", "DBZ", "ZH", "dBZ", "DBZH_corr"], "base": True },
    "velocity":       {"label": "Doppler velocity (V)",              "quantities": ["VRADC", "VRADH", "VRAD", "VR", "V", "VRAD_corr"],   "base": True },
    "correlation":    {"label": "Correlation Coefficient (CC)",      "quantities": ["RHOHV", "RHOHVH", "URHOHV", "RHO", "CCORH", "CC"], "base": True },
    "spectrum_width": {"label": "Spectrum width (SW)",               "quantities": ["WRADH", "WRAD", "W", "WH"],   "base": True },
    "zdr":            {"label": "Diff. Reflectivity (ZDR)",          "quantities": ["ZDRC", "ZDRHC", "ZDR", "ZDRHV", "ZDR_corr"],  "base": True },
    "kdp":            {"label": "Specific Diff. Phase (KDP)",        "quantities": ["KDP"],                        "base": True },
    "echo_tops":      {"label": "Echo tops (>18 dBZ, km)",           "quantities": [],                  "base": False},
    "echo_tops_0":    {"label": "Echo tops (>0 dBZ, km)",            "quantities": [],                  "base": False},
    "nrot":           {"label": "Normalized rotation (m/s)",         "quantities": [],                  "base": False},
    "srv":            {"label": "Storm-relative velocity (m/s)",     "quantities": [],                  "base": False},
    "meso":           {"label": "Mesocyclone probability (%)",       "quantities": [],                  "base": False},
    "azimuthal_shear":{"label": "Azimuthal Shear (×10⁻³/s)",         "quantities": [],                  "base": False},
    "mesh":           {"label": "Est. max hail size (MESH, cm)",      "quantities": [],                  "base": False},
}

def _norm_qty(q: str) -> str:
    """Normalize quantity strings for fuzzy matching."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(q or "")).upper()


def _match_quantity(quantities: List[str], sweeps_by_qty: Dict[str, List[dict]]) -> Optional[str]:
    """Find the best matching quantity key in sweeps_by_qty."""
    if not sweeps_by_qty:
        return None
    keys = list(sweeps_by_qty.keys())
    # Exact / case-insensitive match
    upper_map = {str(k).upper(): k for k in keys}
    for q in quantities:
        if q in sweeps_by_qty:
            return q
        uq = str(q).upper()
        if uq in upper_map:
            return upper_map[uq]
        nq = _norm_qty(q)
        for k in keys:
            if _norm_qty(k) == nq:
                return k
    # Fuzzy contains/prefix match
    for q in quantities:
        nq = _norm_qty(q)
        if not nq:
            continue
        for k in keys:
            nk = _norm_qty(k)
            if nk.startswith(nq) or nq in nk:
                return k
    return None

# Knots - m/s conversion factor.
# Velocity colormaps (velocity, srv) are defined with breakpoints in knots
# (matching the original implementation intent of ±140 kt ~ ±72 m/s),
# then converted to m/s at load time via _KT_TO_MS so the rendered scale
# stays within the Nyquist of typical European C-band radars (~72 m/s).
_KT_TO_MS: float = 0.514444  # 1 knot in m/s

COLORMAPS: Dict[str, List[Tuple[float, Tuple[int, int, int]]]] = {
    "reflectivity": [(-35,(0,0,0)),(-27.5,(0,0,0)),(-27.5,(32,0,32)),(-22.5,(32,0,32)),(-22.5,(33,22,37)),(-10,(33,22,37)),(-10,(73,73,73)),(5,(99,63,192)),(15,(0,64,128)),(15,(0,128,0)),(30,(255,255,128)),(40,(191,130,0)),(40,(255,128,0)),(50,(193,0,0)),(60,(128,0,0)),(60,(128,0,255)),(70,(255,255,255)),(80,(128,64,64))],
    "velocity":     [(-140,(255,255,255)),(-100,(0,14,146)),(-64,(0,69,10)),(-60,(5,107,0)),(-35,(145,192,147)),(-15,(77,124,79)),(-7,(46,68,52)),(0,(109,109,124)),(7,(80,67,52)),(15,(209,143,36)),(35,(250,250,0)),(60,(226,200,255)),(95,(255,10,225)),(140,(0,0,0))],
    "correlation":  [(0,(16,14,18)),(40,(0,0,0)),(54,(41,158,95)),(65,(148,186,124)),(75,(246,225,174)),(80,(183,121,50)),(88,(172,32,32)),(90,(234,0,6)),(100,(243,37,171)),(102,(167,10,143))],
    "spectrum_width":[(0,(5,5,6)),(2,(42,42,49)),(4,(78,77,91)),(6,(115,114,133)),(8,(152,148,173)),(10,(200,109,105)),(12,(246,75,42)),(14,(248,116,52)),(16,(250,161,63)),(18,(252,203,73)),(20,(255,244,83)),(22,(255,248,111)),(24,(255,250,140)),(26,(255,251,170)),(28,(255,252,199)),(30,(255,254,226)),(32,(246,246,246))],
    # Echo tops – solid/step colour table
    # Each integer km band gets an identical pair (N, colour) + (N+0.999, colour) so
    # linear interpolation inside the GPU texture still produces sharp hard edges.
    "echo_tops":    [(0,(55,37,59)),(0.999,(55,37,59)),(1,(0,72,73)),(1.999,(0,72,73)),(2,(21,122,0)),(2.999,(21,122,0)),(3,(40,233,0)),(3.999,(40,233,0)),(4,(149,255,127)),(4.999,(149,255,127)),(5,(255,111,0)),(5.999,(255,111,0)),(6,(255,167,0)),(6.999,(255,167,0)),(7,(255,199,0)),(7.999,(255,199,0)),(8,(255,223,0)),(8.999,(255,223,0)),(9,(255,254,0)),(9.999,(255,254,0)),(10,(255,0,0)),(10.999,(255,0,0)),(11,(171,0,0)),(11.999,(171,0,0)),(12,(122,0,0)),(12.999,(122,0,0)),(13,(122,0,84)),(13.999,(122,0,84)),(14,(141,0,131)),(14.999,(141,0,131)),(15,(199,0,185)),(15.999,(199,0,185)),(16,(255,21,238)),(16.999,(255,21,238)),(17,(255,99,244)),(17.999,(255,99,244)),(18,(255,255,255))],
    "echo_tops_0":  [(0,(55,37,59)),(0.999,(55,37,59)),(1,(0,72,73)),(1.999,(0,72,73)),(2,(21,122,0)),(2.999,(21,122,0)),(3,(40,233,0)),(3.999,(40,233,0)),(4,(149,255,127)),(4.999,(149,255,127)),(5,(255,111,0)),(5.999,(255,111,0)),(6,(255,167,0)),(6.999,(255,167,0)),(7,(255,199,0)),(7.999,(255,199,0)),(8,(255,223,0)),(8.999,(255,223,0)),(9,(255,254,0)),(9.999,(255,254,0)),(10,(255,0,0)),(10.999,(255,0,0)),(11,(171,0,0)),(11.999,(171,0,0)),(12,(122,0,0)),(12.999,(122,0,0)),(13,(122,0,84)),(13.999,(122,0,84)),(14,(141,0,131)),(14.999,(141,0,131)),(15,(199,0,185)),(15.999,(199,0,185)),(16,(255,21,238)),(16.999,(255,21,238)),(17,(255,99,244)),(17.999,(255,99,244)),(18,(255,255,255))],
    "nrot":         [(35,(255,0,0)),(42.5,(128,0,0)),(50,(128,0,128)),(70,(255,0,255)),(90,(0,0,253)),(100,(0,0,0))],
    "srv":          [(-140,(255,255,255)),(-100,(0,14,146)),(-64,(0,69,10)),(-60,(5,107,0)),(-35,(145,192,147)),(-15,(77,124,79)),(-7,(46,68,52)),(0,(109,109,124)),(7,(80,67,52)),(15,(209,143,36)),(35,(250,250,0)),(60,(226,200,255)),(95,(255,10,225)),(140,(0,0,0))],
    "meso":         [(0,(0,64,0)),(10,(125,255,129)),(35,(255,0,0)),(42.5,(128,0,0)),(50,(128,0,128)),(70,(255,0,255)),(90,(0,0,253)),(100,(0,0,0))],
    # Azimuthal shear (negatives fixed to positives, ×10⁻³ s⁻¹)
    # Smooth interpolation within bands; step boundaries replicated at X and X+0.001
    # to create the hard breaks at 4 and 10 ×10⁻³ s⁻¹.
    "azimuthal_shear": [(0,(136,147,126)),(1,(0,105,20)),(3.9,(0,198,38)),(3.901,(144,133,68)),(9.9,(249,255,8)),(9.901,(255,0,0)),(20,(82,0,0)),(26,(0,255,236)),(32,(255,255,255))],
    "zdr":          [(-4,(0,0,0)),(0,(142,121,181)),(0.25,(10,10,155)),(1,(68,248,212)),(1.5,(90,221,98)),(2,(255,255,100)),(3,(220,10,5)),(4,(175,0,0)),(5,(240,120,180)),(6,(255,255,255)),(8,(145,45,150))],
    "kdp":          [(-2,(0,0,100)),(-1,(0,60,200)),(0,(100,160,255)),(0.1,(200,230,255)),(0.25,(255,255,255)),(0.5,(200,255,200)),(1,(0,200,0)),(2,(200,255,0)),(3,(255,220,0)),(4,(255,140,0)),(5,(200,0,0)),(6,(130,0,60))],
    # MESH (0–15 cm): black → blue → cyan → green → yellow → orange → red → magenta → white
    # Thresholds loosely follow NWS/Gibson Ridge conventions (pea≈0.6 cm, dime≈1.8 cm,
    # golf-ball≈4.3 cm, baseball≈7.4 cm); colour saturates toward white at the 15 cm cap.
    "mesh":         [(0,(0,0,0)),(0.5,(0,0,160)),(1.0,(0,80,255)),(1.5,(0,200,220)),(2.0,(0,200,40)),(2.5,(140,230,0)),(3.0,(255,255,0)),(4.0,(255,165,0)),(5.0,(255,60,0)),(6.0,(210,0,0)),(7.5,(180,0,180)),(10.0,(255,0,255)),(12.5,(255,160,255)),(15.0,(255,255,255))],
    # CAPE ML (0–5000 J/kg): black → dark purple → blue → cyan → green → yellow → orange → red → pink → white
    # Thresholds loosely follow operational conventions:
    #   <100 J/kg  – marginal/no instability (dark tones)
    #   250–1000   – moderate (blue-cyan-green)
    #   1000–2500  – significant (yellow-orange)
    #   >2500      – extreme instability (red-pink-white)
    "cape_ml":      [(0,(0,0,0)),(50,(20,0,40)),(100,(60,0,100)),(250,(0,0,200)),(500,(0,100,255)),(750,(0,200,220)),(1000,(0,210,80)),(1500,(160,240,0)),(2000,(255,240,0)),(2500,(255,160,0)),(3000,(255,60,0)),(3500,(210,0,0)),(4000,(160,0,160)),(4500,(255,100,220)),(5000,(255,255,255))],
}
# Convert velocity & srv colormap breakpoints from knots - m/s.
# The values above (e.g. ±140) are in knots; multiplying by _KT_TO_MS
# gives the correct physical range (±140 kt ~ ±72 m/s) without touching
# the colour stops manually.
for _vel_qty in ("velocity", "srv"):
    COLORMAPS[_vel_qty] = [
        (round(v * _KT_TO_MS, 2), c) for v, c in COLORMAPS[_vel_qty]
    ]

LEGEND_STEPS = {
    "reflectivity": 5,
    "velocity": 10,
    "correlation": 10,
    "spectrum_width": 2,
    "zdr": 1,
    "kdp": 1,
    "echo_tops": 1,
    "echo_tops_0": 1,
    "nrot": 5,
    "srv": 10,
    "meso": 10,
    "azimuthal_shear": 2,
    "mesh": 1,
    "cape_ml": 250,
}


def _json_get(url: str, params: dict) -> dict:
    with urlopen(f"{url}?{urlencode(params)}", timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _bytes_get(url: str, params: dict) -> bytes:
    qs = urlencode(params)
    full_url = f"{url}?{qs}" if qs else url
    with urlopen(full_url, timeout=120) as response:
        return response.read()


def _find_latest_volume_asset(station_code: str) -> Tuple[str, str]:
    data = _json_get(DMI_VOLUME_ITEMS_URL, {"api-key": DMI_API_KEY, "limit": 250, "sortorder": "datetime,DESC"})
    for feature in data.get("features", []):
        if str(feature.get("id", "")).startswith(f"{station_code}_"):
            href = feature.get("asset", {}).get("data", {}).get("href")
            if href:
                return href, feature.get("properties", {}).get("datetime", "")
    raise RuntimeError(f"No volume data found for station code {station_code}")


def _parse_rfc3339(dt: str) -> datetime:
    s = dt.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _find_dmi_volume_asset_near_datetime(
    station_code: str, target_utc: datetime, window_minutes: int = 20
) -> Tuple[str, str]:
    if target_utc.tzinfo is None:
        target_utc = target_utc.replace(tzinfo=timezone.utc)
    target_utc = target_utc.astimezone(timezone.utc)

    start = (target_utc - timedelta(minutes=window_minutes)).isoformat().replace("+00:00", "Z")
    end = (target_utc + timedelta(minutes=window_minutes)).isoformat().replace("+00:00", "Z")

    # Note: do NOT include sortorder here - some DMI API versions reject it
    # when combined with a datetime interval filter and return HTTP 400.
    data = _json_get(
        DMI_VOLUME_ITEMS_URL,
        {
            "api-key": DMI_API_KEY,
            "limit": 500,
            "datetime": f"{start}/{end}",
        },
    )

    best: Optional[Tuple[float, str, str]] = None
    for feature in data.get("features", []):
        if not str(feature.get("id", "")).startswith(f"{station_code}_"):
            continue
        dtstr = str((feature.get("properties") or {}).get("datetime") or "")
        href = ((feature.get("asset") or {}).get("data") or {}).get("href")
        if not dtstr or not href:
            continue
        try:
            obs = _parse_rfc3339(dtstr)
        except Exception:
            continue
        delta = abs((obs - target_utc).total_seconds())
        if best is None or delta < best[0]:
            best = (delta, str(href), dtstr)

    if best is None:
        raise RuntimeError(
            f"No volume data found for {station_code} near {target_utc.isoformat()} "
            f"(±{window_minutes} min). Check the station code and datetime."
        )
    return best[1], best[2]


def _smhi_find_latest_qcvol_asset(area_key: str) -> Tuple[str, str]:
    product_url = f"{SMHI_BASE_URL}/{SMHI_VERSION}/area/{area_key}/product/{SMHI_QCVOL_PRODUCT}.json"
    data = _json_get(product_url, {})
    last_files = data.get("lastFiles") or []
    if not last_files:
        raise RuntimeError(f"No SMHI qcvol files found for area {area_key}")
    latest = last_files[0]
    formats = latest.get("formats") or []
    h5 = next((f for f in formats if f.get("key") == "h5" and f.get("link")), None)
    if not h5:
        raise RuntimeError(f"No SMHI qcvol HDF5 format for area {area_key}")
    link = str(h5["link"])
    valid_str = str(latest.get("valid") or latest.get("updated") or "")
    return link, valid_str


def _fmi_find_latest_asset(site_code: str) -> Tuple[str, str]:
    """Find the latest PVOL HDF5 file for a given FMI radar site on S3.

    FMI naming convention:  {YYYY}/{MM}/{DD}/{site}/{YYYYMMDDHHmm}_{site}_PVOL.h5
    Tries today first, steps back up to 2 days to cope with late uploads.
    """
    from urllib.error import HTTPError, URLError

    def _try_list(base: str, prefix: str) -> Optional[str]:
        """Return raw XML from S3 listing or None."""
        url = f"{base}/?list-type=2&prefix={prefix}&max-keys=200"
        try:
            with urlopen(url, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, OSError):
            return None

    for delta in range(3):
        dt = datetime.now(timezone.utc) - timedelta(days=delta)
        prefix = f"{dt.year}/{dt.month:02d}/{dt.day:02d}/{site_code}/"

        xml = _try_list(FMI_S3_BASE, prefix)
        if not xml:
            xml = _try_list(FMI_S3_REGION_BASE, prefix)
        if not xml:
            continue

        keys = FMI_S3_KEY_RE.findall(xml)
        if not keys:
            continue

        # Prefer PVOL files; if none, fall back to whatever .h5 exists
        pvol_keys = [k for k in keys if re.search(r'pvol', k, re.IGNORECASE)]
        chosen_keys = pvol_keys if pvol_keys else keys

        # Sort descending: timestamp is the first 12-digit token in the filename
        def _fmi_ts(key: str) -> str:
            m = re.search(r"(\d{12})", key.split("/")[-1])
            return m.group(1) if m else "000000000000"

        chosen_keys.sort(key=_fmi_ts, reverse=True)
        latest_key = chosen_keys[0]

        # Build download URL using the regional endpoint for reliability
        file_url = f"{FMI_S3_REGION_BASE}/{latest_key}"

        ts = _fmi_ts(latest_key)
        try:
            dt_val = datetime.strptime(ts, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            dt_str = dt_val.isoformat().replace("+00:00", "Z")
        except Exception:
            dt_str = ts

        return file_url, dt_str

    raise RuntimeError(f"No FMI PVOL data found for site {site_code}")

# -SEDOO / Meteo-France helpers DISABLED - API constantly failing 
#
# def _fr_selection_paths() -> List[Path]:
#     """Return candidate paths for the SEDOO selection file.
#
#     The filename embeds a per-subscription token and changes whenever the user
#     re-generates their selection, so we glob for ``selections_radars_*.txt``
#     in the usual locations and return the newest match in each directory.
#     An explicit override via the ``FR_PVOL_SELECTION_FILE`` env var is always
#     tried first.
#     """
#     import glob as _glob
#     env_path = str((os.environ.get("FR_PVOL_SELECTION_FILE") or "")).strip()
#     search_dirs: List[Path] = [
#         Path.cwd(),
#         Path(__file__).resolve().parent,
#         Path.home() / "Downloads",
#     ]
#     out: List[Path] = []
#     seen: set = set()
#
#     def _add(p: Optional[Path]) -> None:
#         if p is None:
#             return
#         try:
#             rp = p.resolve()
#         except Exception:
#             rp = p
#         key = str(rp).lower()
#         if key in seen:
#             return
#         seen.add(key)
#         out.append(rp)
#
#     # 1. Explicit env override (exact path)
#     if env_path:
#         _add(Path(env_path))
#
#     # 2. Glob for any selections_radars_*.txt in each search dir, newest first
#     for d in search_dirs:
#         try:
#             matches = sorted(
#                 _glob.glob(str(d / "selections_radars_*.txt")),
#                 key=lambda f: Path(f).stat().st_mtime,
#                 reverse=True,
#             )
#             for m in matches:
#                 _add(Path(m))
#         except Exception:
#             pass
#     return out
#
#
# def _load_fr_selection_entries() -> Dict[str, List[dict]]:
#     """Parse the SEDOO selection file (wget format) into a product→entries dict.
#
#     Handles both the old per-scan format (year+month+day+hour+hot) and the new
#     monthly-archive format (year+month only, response is a .tar of NetCDF files).
#     """
#     global _FR_SELECTION_CACHE
#     if _FR_SELECTION_CACHE is not None:
#         return _FR_SELECTION_CACHE
#
#     src_path: Optional[Path] = None
#     for p in _fr_selection_paths():
#         if p.exists():
#             src_path = p
#             break
#
#     if src_path is None:
#         _FR_SELECTION_CACHE = {}
#         return _FR_SELECTION_CACHE
#
#     try:
#         lines = src_path.read_text(encoding="utf-8", errors="ignore").splitlines()
#     except Exception:
#         _FR_SELECTION_CACHE = {}
#         return _FR_SELECTION_CACHE
#
#     entries: Dict[str, List[dict]] = {}
#
#     for line in lines:
#         m = FR_SELECTION_URL_RE.search(line)
#         if not m:
#             continue
#         params = parse_qs(m.group(1))
#
#         product = str((params.get("product") or [""])[0]).strip()
#         year    = str((params.get("year")    or [""])[0]).strip()
#         month   = str((params.get("month")   or [""])[0]).strip().zfill(2)
#         day     = str((params.get("day")     or [""])[0]).strip().zfill(2)
#         hour    = str((params.get("hour")    or [""])[0]).strip().zfill(4)
#         hot     = str((params.get("hot")     or [""])[0]).strip().lower()
#
#         if not product or not year or not month:
#             continue
#
#         # Detect format: new API has no day/hour → monthly tar archive
#         is_monthly = not (day.strip("0") and hour.strip("0"))
#
#         if is_monthly:
#             try:
#                 dt_val = datetime(int(year), int(month), 1, tzinfo=timezone.utc)
#             except Exception:
#                 continue
#         else:
#             try:
#                 dt_val = datetime.strptime(
#                     f"{year}{month}{day}{hour}", "%Y%m%d%H%M"
#                 ).replace(tzinfo=timezone.utc)
#             except Exception:
#                 continue
#
#         # Preserve any extra auth params (e.g. uuid)
#         _known = {"product", "year", "month", "day", "hour", "hot"}
#         extra_params = {k: v[0] for k, v in params.items() if k not in _known and v}
#
#         # Reconstruct canonical URL
#         url_params: dict = {**extra_params, "product": product, "year": year, "month": month}
#         if not is_monthly:
#             url_params.update({"day": day, "hour": hour})
#             if hot:
#                 url_params["hot"] = hot
#
#         url = f"{FR_RADAR_DOWNLOAD_URL}?" + urlencode(url_params)
#
#         entries.setdefault(product, []).append({
#             "url":          url,
#             "datetime":     dt_val,
#             "is_monthly":   is_monthly,
#             "extra_params": extra_params,
#             "base_params":  {"year": year, "month": month},  # for constructing live URLs
#         })
#
#     for product in entries:
#         entries[product].sort(key=lambda e: e["datetime"], reverse=True)
#
#     _FR_SELECTION_CACHE = entries
#     return entries
# def _fr_find_latest_asset(product: str, max_candidates: int = 5) -> List[Tuple[str, str]]:
#     """Return (url, dt_str, is_monthly) tuples for *product*, newest first.
#
#     Strategy (tried in order):
#       1. Per-scan URLs: current time walking back in 5-min steps with
#          year+month+day+hour params — works if SEDOO still accepts granular requests.
#       2. Monthly TAR URLs for the current and previous month — caller must handle
#          the TAR by streaming-extracting the most recent NetCDF inside.
#     """
#     entries = _load_fr_selection_entries().get(product) or []
#     if not entries:
#         raise RuntimeError(
#             f"No Meteo-France selection entry for product '{product}'. "
#             f"Expected a 'selections_radars_*.txt' file in the working directory."
#         )
#
#     extra_params: dict = entries[0].get("extra_params") or {}
#     candidates: List[Tuple[str, str]] = []
#
#     # ── Per-scan candidates (dynamic timestamps, newest first) ──────────────
#     now_utc = datetime.now(timezone.utc)
#     floored = now_utc.replace(second=0, microsecond=0)
#     floored -= timedelta(minutes=floored.minute % 5)
#
#     for step in range(36):
#         dt = floored - timedelta(minutes=step * 5)
#         url = (
#             f"{FR_RADAR_DOWNLOAD_URL}?"
#             + urlencode({
#                 **extra_params,
#                 "product": product,
#                 "year":  str(dt.year),
#                 "month": f"{dt.month:02d}",
#                 "day":   f"{dt.day:02d}",
#                 "hour":  f"{dt.hour:02d}{dt.minute:02d}",
#             })
#         )
#         candidates.append((url, dt.isoformat().replace("+00:00", "Z")))
#         if len(candidates) >= max_candidates:
#             break
#
#     # ── Monthly TAR fallback (current month, then previous month) ───────────
#     for delta_months in (0, -1):
#         m_dt = now_utc
#         if delta_months:
#             # Roll back one month
#             if m_dt.month == 1:
#                 m_dt = m_dt.replace(year=m_dt.year - 1, month=12)
#             else:
#                 m_dt = m_dt.replace(month=m_dt.month - 1)
#         monthly_url = (
#             f"{FR_RADAR_DOWNLOAD_URL}?"
#             + urlencode({
#                 **extra_params,
#                 "product": product,
#                 "year":  str(m_dt.year),
#                 "month": f"{m_dt.month:02d}",
#             })
#         )
#         dt_str = m_dt.replace(day=1, hour=0, minute=0, second=0).isoformat().replace("+00:00", "Z")
#         candidates.append((monthly_url, dt_str))
#
#     return candidates
# def _discover_fr_stations() -> Dict[str, dict]:
#     stations: Dict[str, dict] = {}
#     for product in sorted(_load_fr_selection_entries().keys()):
#         m = FR_PRODUCT_CODE_RE.search(product)
#         code = m.group(1).upper() if m else product
#         name = f"France {code}"
#         stations[name] = {
#             "provider": "fr_mf",
#             "product": product,
#             # Updated from file metadata after first successful render.
#             "lat": 46.50,
#             "lon": 2.50,
#             "range_km": 256.0,
#         }
#     return stations
#

def _knmi_dt_from_filename(name: str) -> Optional[str]:
    """
    Extract RFC3339 timestamp from KNMI radar filename.

    Example: RAD_NL61_VOL_NA_202603151035.h5 - 2026-03-15T10:35:00Z
    """
    m = re.search(r"(20\d{10,14})", name)
    if not m:
        return None
    token = m.group(1)
    try:
        if len(token) == 12:
            dt = datetime.strptime(token, "%Y%m%d%H%M")
        elif len(token) == 14:
            dt = datetime.strptime(token, "%Y%m%d%H%M%S")
        else:
            return None
        return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _knmi_json_get(path: str, params: dict) -> dict:
    """
    Small helper for KNMI Open Data API using urllib only (no requests dependency).
    """
    qs = urlencode(params)
    url = f"{KNMI_BASE_URL}{path}"
    if qs:
        url = f"{url}?{qs}"
    req = Request(url, headers={"Authorization": KNMI_API_KEY})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _knmi_fetch_latest_volume(dataset: str, version: str) -> Tuple[bytes, str, str]:
    """
    Fetch the most recent KNMI radar volume file for a dataset/version.

    Returns (file_bytes, datetime_rfc3339, filename).
    """
    listing = _knmi_json_get(
        f"/datasets/{dataset}/versions/{version}/files",
        {"maxKeys": 1, "orderBy": "created", "sorting": "desc"},
    )
    files = listing.get("files") or []
    if not files:
        raise RuntimeError(f"No KNMI files found for dataset {dataset}/{version}")

    fname = str(files[0].get("filename") or "").strip()
    if not fname:
        raise RuntimeError(f"KNMI listing missing filename for dataset {dataset}/{version}")

    url_info = _knmi_json_get(
        f"/datasets/{dataset}/versions/{version}/files/{fname}/url",
        {},
    )
    download_url = str(url_info.get("temporaryDownloadUrl") or "").strip()
    if not download_url:
        raise RuntimeError(f"KNMI did not return download URL for {fname}")

    file_bytes = _bytes_get(download_url, {})
    dt_str = _knmi_dt_from_filename(fname) or ""
    return file_bytes, dt_str, fname


def _decompress_if_needed(data: bytes) -> bytes:
    """Transparently decompress gzip or bzip2 bytes, return raw bytes otherwise."""
    if data[:2] == b'\x1f\x8b':        # gzip magic
        return gzip.decompress(data)
    if data[:2] == b'BZ' and data[2:3] == b'h':  # bzip2 magic
        import bz2
        return bz2.decompress(data)
    return data


def _shmu_read_url(url: str, timeout: int = 30) -> bytes:
    """
    Read SHMU URL bytes with certificate fallback.

    First attempts normal TLS verification. If the endpoint presents an
    incomplete certificate chain (seen on SHMU), retry with an unverified
    context for that request only.
    """
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.read()
    except ssl.SSLCertVerificationError:
        insecure_ctx = ssl._create_unverified_context()
        with urlopen(url, timeout=timeout, context=insecure_ctx) as response:
            return response.read()
    except URLError as exc:
        reason = getattr(exc, "reason", None)
        msg = str(exc)
        is_cert_error = isinstance(reason, ssl.SSLCertVerificationError) or (
            "CERTIFICATE_VERIFY_FAILED" in msg
        )
        if not is_cert_error:
            raise
        insecure_ctx = ssl._create_unverified_context()
        with urlopen(url, timeout=timeout, context=insecure_ctx) as response:
            return response.read()


def _http_dir_list(url: str) -> List[str]:
    if not url.endswith("/"):
        url += "/"
    with urlopen(url, timeout=60) as response:
        html = response.read().decode("utf-8", errors="ignore")
    return [
        href.strip()
        for href in DWD_HREF_RE.findall(html)
        if href and href not in ("../", "./") and not href.startswith("?")
    ]


def _grib_signed_magnitude(raw: bytes) -> int:
    value = int.from_bytes(raw, "big", signed=False)
    sign = -1 if (value & (1 << (len(raw) * 8 - 1))) else 1
    magnitude = value & ((1 << (len(raw) * 8 - 1)) - 1)
    return sign * magnitude


def _grib_time_code_to_minutes(unit_code: int, value: int) -> int:
    if unit_code == 0:   # minute
        return int(value)
    if unit_code == 1:   # hour
        return int(value) * 60
    if unit_code == 2:   # day
        return int(value) * 24 * 60
    raise RuntimeError(f"Unsupported ICON-D2 time unit code: {unit_code}")


def _grib2_unpack_values(payload: bytes, bits_per_value: int, count: int) -> "np.ndarray":
    if count <= 0:
        return np.empty((0,), dtype=np.uint32)
    if bits_per_value == 0:
        return np.zeros((count,), dtype=np.uint32)
    if bits_per_value == 8:
        return np.frombuffer(payload[:count], dtype=np.uint8).astype(np.uint32)
    if bits_per_value == 16:
        return np.frombuffer(payload[: count * 2], dtype=">u2").astype(np.uint32)
    if bits_per_value == 32:
        return np.frombuffer(payload[: count * 4], dtype=">u4").astype(np.uint32)

    raw = np.frombuffer(payload, dtype=np.uint8)
    bit_count = count * bits_per_value
    bits = np.unpackbits(raw, bitorder="big")
    if bits.size < bit_count:
        raise RuntimeError(
            f"GRIB payload too short for {count} values at {bits_per_value} bits"
        )
    bits = bits[:bit_count].reshape(count, bits_per_value).astype(np.uint32)
    weights = (1 << np.arange(bits_per_value - 1, -1, -1, dtype=np.uint32))
    return bits.dot(weights)


def _icon_d2_parse_file_messages(file_bytes: bytes, run_dt: datetime, unit_scale: float = 1.0) -> List[Dict[str, Any]]:
    if not HAS_RADAR_DEPS:
        raise RuntimeError(f"Missing dependencies: {RADAR_DEPS_ERROR}")

    raw = _decompress_if_needed(file_bytes)
    out: List[Dict[str, Any]] = []
    pos = 0

    while True:
        start = raw.find(b"GRIB", pos)
        if start < 0:
            break

        msg_len = int.from_bytes(raw[start + 8 : start + 16], "big")
        msg = raw[start : start + msg_len]
        pos = start + msg_len
        if msg[:4] != b"GRIB" or msg[-4:] != b"7777":
            continue

        sections: Dict[int, bytes] = {}
        off = 16
        while off < len(msg) - 4:
            if msg[off : off + 4] == b"7777":
                break
            sec_len = int.from_bytes(msg[off : off + 4], "big")
            sec_no = msg[off + 4]
            sections[sec_no] = msg[off : off + sec_len]
            off += sec_len

        sec3 = sections.get(3)
        sec4 = sections.get(4)
        sec5 = sections.get(5)
        sec6 = sections.get(6)
        sec7 = sections.get(7)
        if not all((sec3, sec4, sec5, sec6, sec7)):
            continue

        grid_template = int.from_bytes(sec3[12:14], "big")
        data_template = int.from_bytes(sec5[9:11], "big")
        if grid_template != 0:
            raise RuntimeError(f"Unsupported ICON-D2 grid template: {grid_template}")
        if data_template != 0:
            raise RuntimeError(f"Unsupported ICON-D2 data template: {data_template}")

        nx = int.from_bytes(sec3[30:34], "big")
        ny = int.from_bytes(sec3[34:38], "big")
        if nx <= 0 or ny <= 0:
            raise RuntimeError("ICON-D2 grid has invalid dimensions")

        lat1 = int.from_bytes(sec3[46:50], "big", signed=True) / 1_000_000.0
        lon1 = int.from_bytes(sec3[50:54], "big", signed=True) / 1_000_000.0
        lat2 = int.from_bytes(sec3[55:59], "big", signed=True) / 1_000_000.0
        lon2 = int.from_bytes(sec3[59:63], "big", signed=True) / 1_000_000.0

        def _norm_lon(lon: float) -> float:
            while lon > 180.0:
                lon -= 360.0
            while lon <= -180.0:
                lon += 360.0
            return lon

        west = min(_norm_lon(lon1), _norm_lon(lon2))
        east = max(_norm_lon(lon1), _norm_lon(lon2))
        south = min(lat1, lat2)
        north = max(lat1, lat2)

        time_unit = sec4[17]
        forecast_minutes = _grib_time_code_to_minutes(
            time_unit,
            int.from_bytes(sec4[18:22], "big"),
        )

        number_of_values = int.from_bytes(sec5[5:9], "big")
        reference_value = struct.unpack(">f", sec5[11:15])[0]
        binary_scale = _grib_signed_magnitude(sec5[15:17])
        decimal_scale = _grib_signed_magnitude(sec5[17:19])
        bits_per_value = int(sec5[19])

        bitmap_indicator = sec6[5]
        if bitmap_indicator == 255:
            bitmap = np.ones((nx * ny,), dtype=bool)
        elif bitmap_indicator == 0:
            bitmap = (
                np.unpackbits(np.frombuffer(sec6[6:], dtype=np.uint8), bitorder="big")[
                    : nx * ny
                ].astype(bool)
            )
        else:
            raise RuntimeError(
                f"Unsupported ICON-D2 bitmap indicator: {bitmap_indicator}"
            )

        expected_values = int(bitmap.sum())
        if number_of_values != expected_values:
            expected_values = min(number_of_values, expected_values)

        packed_values = _grib2_unpack_values(sec7[5:], bits_per_value, expected_values)
        values = np.full((nx * ny,), np.nan, dtype=np.float32)
        if expected_values > 0:
            decoded = reference_value + (
                packed_values.astype(np.float32) * (2.0 ** binary_scale)
            )
            if decimal_scale != 0:
                decoded = decoded / (10.0 ** decimal_scale)
            valid_idx = np.flatnonzero(bitmap)[:expected_values]
            values[valid_idx] = decoded

        grid = values.reshape((ny, nx))
        # GRIB scans south - north for ICON-D2 regular lat/lon, while Mapbox
        # image sources expect row 0 at the northern edge.
        if lat1 < lat2:
            grid = np.flipud(grid)

        # Post-processing intentionally skipped - raw native GRIB grid is
        # encoded and sent to the client without any thresholding, upsampling,
        # or smoothing so the data is displayed exactly as received.
        # Apply per-product unit scaling (e.g. m - km for echotop).
        if unit_scale != 1.0:
            grid = grid * np.float32(unit_scale)
        out_ny, out_nx = grid.shape

        out.append(
            {
                "forecastMinutes": int(forecast_minutes),
                "validIso": (run_dt + timedelta(minutes=forecast_minutes))
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "bounds": [
                    [west, north],
                    [east, north],
                    [east, south],
                    [west, south],
                ],
                "valueEncoded": _encode_values_png(grid),
                "gridShape": [int(out_ny), int(out_nx)],
            }
        )

    if not out:
        raise RuntimeError("ICON-D2 file did not contain any readable GRIB messages")

    out.sort(key=lambda item: item["forecastMinutes"])
    return out


def _icon_d2_inventory_payload(inventory: Dict[str, Any], product_key: str) -> Dict[str, Any]:
    prod_cfg = ICON_D2_PRODUCTS.get(product_key, ICON_D2_PRODUCTS[ICON_D2_DEFAULT_PRODUCT])
    return {
        "generatedAtIso": inventory["generatedAtIso"],
        "latestCompletedRunIso": inventory["latestCompletedRunIso"],
        "product": product_key,
        "label": prod_cfg["label"],
        "colormap": prod_cfg["colormap"],
        "frames": [
            {
                "key": frame["key"],
                "validIso": frame["validIso"],
                "runIso": frame["runIso"],
                "forecastMinutes": frame["forecastMinutes"],
            }
            for frame in inventory["frames"]
        ],
    }


def _icon_d2_build_inventory(product_key: str) -> Dict[str, Any]:
    prod_cfg = ICON_D2_PRODUCTS.get(product_key, ICON_D2_PRODUCTS[ICON_D2_DEFAULT_PRODUCT])
    product_dir = prod_cfg["dir"]
    file_re = _icon_d2_file_re(prod_cfg["filename_suffix"])
    runs: List[Dict[str, Any]] = []

    for run_hour in ICON_D2_RUN_HOURS:
        dir_url = f"{ICON_D2_BASE_URL}/{run_hour}/{product_dir}/"
        try:
            hrefs = _http_dir_list(dir_url)
        except Exception:
            continue

        grouped: Dict[str, Dict[int, str]] = {}
        for href in hrefs:
            name = href.rstrip("/").split("/")[-1]
            match = file_re.match(name)
            if not match:
                continue
            run_id = match.group("run")
            forecast_hour = int(match.group("fh"))
            grouped.setdefault(run_id, {})[forecast_hour] = urljoin(dir_url, name)

        completed: List[Tuple[str, Dict[int, str]]] = [
            (run_id, files)
            for run_id, files in grouped.items()
            if 48 in files
        ]
        if not completed:
            continue

        latest_run_id, files = max(completed, key=lambda item: item[0])
        run_dt = datetime.strptime(latest_run_id, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        runs.append(
            {
                "runIso": run_dt.isoformat().replace("+00:00", "Z"),
                "runDt": run_dt,
                "files": files,
            }
        )

    runs.sort(key=lambda item: item["runDt"], reverse=True)

    frames_by_valid: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        run_iso = run["runIso"]
        for forecast_hour, file_url in sorted(run["files"].items()):
            minute_offsets = (0,) if forecast_hour >= 48 else (0, 15, 30, 45)
            for message_index, minute_offset in enumerate(minute_offsets):
                forecast_minutes = forecast_hour * 60 + minute_offset
                valid_dt = run["runDt"] + timedelta(minutes=forecast_minutes)
                valid_iso = valid_dt.isoformat().replace("+00:00", "Z")
                if valid_iso in frames_by_valid:
                    continue
                frames_by_valid[valid_iso] = {
                    "key": f"{run['runDt'].strftime('%Y%m%d%H')}|{forecast_hour:03d}|{message_index}",
                    "validIso": valid_iso,
                    "runIso": run_iso,
                    "forecastMinutes": int(forecast_minutes),
                    "fileUrl": file_url,
                    "messageIndex": int(message_index),
                }

    frames = sorted(frames_by_valid.values(), key=lambda item: item["validIso"])
    by_key = {frame["key"]: frame for frame in frames}

    return {
        "generatedAtIso": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "latestCompletedRunIso": runs[0]["runIso"] if runs else None,
        "frames": frames,
        "byKey": by_key,
    }


def _icon_d2_get_inventory(product_key: str = ICON_D2_DEFAULT_PRODUCT, force: bool = False) -> Dict[str, Any]:
    if product_key not in ICON_D2_PRODUCTS:
        product_key = ICON_D2_DEFAULT_PRODUCT
    now_ts = time.time()
    with _ICON_D2_INVENTORY_LOCK:
        cache = _ICON_D2_INVENTORY_CACHES[product_key]
        cached = cache.get("data")
        expires = float(cache.get("expires") or 0.0)
        if not force and cached is not None and now_ts < expires:
            return cached

    inventory = _icon_d2_build_inventory(product_key)
    with _ICON_D2_INVENTORY_LOCK:
        _ICON_D2_INVENTORY_CACHES[product_key]["data"] = inventory
        _ICON_D2_INVENTORY_CACHES[product_key]["expires"] = now_ts + ICON_D2_INVENTORY_TTL_S
    return inventory


def _icon_d2_get_frame_payload(frame_key: str, product_key: str = ICON_D2_DEFAULT_PRODUCT) -> Dict[str, Any]:
    if product_key not in ICON_D2_PRODUCTS:
        product_key = ICON_D2_DEFAULT_PRODUCT
    prod_cfg = ICON_D2_PRODUCTS[product_key]
    unit_scale = float(prod_cfg.get("unit_scale", 1.0))

    inventory = _icon_d2_get_inventory(product_key)
    frame_meta = inventory["byKey"].get(frame_key)
    if frame_meta is None:
        raise KeyError(frame_key)

    file_url = frame_meta["fileUrl"]
    with _ICON_D2_FILE_CACHE_LOCK:
        file_cache = _ICON_D2_FILE_CACHES[product_key]
        cached = file_cache.get(file_url)
        if cached is not None:
            cached["last_used"] = time.time()

    if cached is None:
        run_dt = _parse_rfc3339(frame_meta["runIso"])
        raw_bytes = _bytes_get(file_url, {})
        messages = _icon_d2_parse_file_messages(raw_bytes, run_dt, unit_scale=unit_scale)
        cached = {
            "messages": messages,
            "last_used": time.time(),
        }
        with _ICON_D2_FILE_CACHE_LOCK:
            file_cache = _ICON_D2_FILE_CACHES[product_key]
            file_cache[file_url] = cached
            if len(file_cache) > ICON_D2_FILE_CACHE_MAX:
                oldest_url = min(
                    file_cache.items(),
                    key=lambda item: float(item[1].get("last_used") or 0.0),
                )[0]
                file_cache.pop(oldest_url, None)

    message_index = max(
        0,
        min(int(frame_meta["messageIndex"]), len(cached["messages"]) - 1),
    )
    message = cached["messages"][message_index]
    return {
        "key": frame_meta["key"],
        "validIso": message["validIso"],
        "runIso": frame_meta["runIso"],
        "forecastMinutes": int(message["forecastMinutes"]),
        "bounds": message["bounds"],
        "valueEncoded": message["valueEncoded"],
        "gridShape": message["gridShape"],
    }


def _dwd_dt_from_name(name: str) -> Optional[datetime]:
    m = DWD_TS_RE.search(name)
    if not m:
        return None
    token = m.group(1)[:14]
    try:
        return datetime.strptime(token, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _dwd_discover_site_codes(limit: int = DWD_SITE_LIMIT) -> List[str]:
    root = f"{DWD_BASE_URL}/{DWD_SWEEP_DIRS[DWD_SITE_DISCOVERY_PRODUCT]}/"
    entries = _http_dir_list(root)
    out: List[str] = []
    for entry in entries:
        if not entry.endswith("/"):
            continue
        code = entry.strip("/").split("/")[-1].lower()
        if not code or code == "filter_polarimetric":
            continue
        if not any(ch.isalpha() for ch in code):
            continue
        if len(code) > 8:
            continue
        out.append(code)
    out = sorted(set(out))
    return out[:max(1, int(limit))]


def _dwd_find_latest_site_asset(site_code: str, product_key: str) -> Tuple[str, str]:
    if product_key not in DWD_SWEEP_DIRS:
        raise RuntimeError(f"Unsupported DWD product key: {product_key}")

    root = f"{DWD_BASE_URL}/{DWD_SWEEP_DIRS[product_key]}/{site_code}/"

    def _looks_like_hdf5(name: str) -> bool:
        nl = name.lower()
        if nl.endswith(("/", ".html", ".htm", ".txt", ".xml", ".json", ".css", ".js")):
            return False
        if nl.endswith((".hd5", ".h5", ".hdf5", "-hd5", "-h5",
                         ".hd5.bz2", ".h5.bz2", ".hdf5.bz2",
                         "-hd5.bz2", "-h5.bz2",
                         ".hd5.gz", ".h5.gz", ".hdf5.gz")):
            return True
        # DWD files sometimes have no conventional extension - accept any
        # non-directory entry whose name contains a timestamp-like token
        # or the word "sweep"/"h5"/"hdf" in any casing.
        if re.search(r'hdf5?|sweep|20\d{10}', nl):
            return True
        return False

    def _find_files_in_tree(start_root: str, start_entries: List[str],
                             max_depth: int = 5) -> Tuple[str, List[str]]:
        """Navigate directory tree looking for HDF5-like files.

        When the current level contains subdirectories but no HDF5 files
        (e.g. DWD's per-elevation layout sweep_vol_z_0_0/, sweep_vol_z_0_1/,
        …), we collect files from ALL subdirectories and return them with
        their relative subdir prefix intact so that _elev_sort_key can pick
        the lowest elevation across the full set.  Only if that shallow scan
        yields nothing do we recurse — into the first (lowest-elevation)
        subdirectory rather than the last, to avoid landing in the highest
        tilt by default.
        """
        cur_root, cur_entries = start_root, start_entries
        for _ in range(max_depth):
            # Strict pass first (known extensions)
            files = [e for e in cur_entries
                     if (not e.endswith("/")) and e.lower().endswith(
                         (".hd5", ".h5", ".hdf5", "-hd5", "-h5",
                          ".hd5.bz2", ".h5.bz2", ".hdf5.bz2",
                          "-hd5.bz2", "-h5.bz2",
                          ".hd5.gz", ".h5.gz", ".hdf5.gz"))]
            if not files:
                # Permissive pass for unusual naming
                files = [e for e in cur_entries
                         if (not e.endswith("/")) and _looks_like_hdf5(e)]
            if files:
                return cur_root, files
            dirs = sorted(
                e for e in cur_entries
                if e.endswith("/") and e not in ("../", "./")
            )
            if not dirs:
                break
            # Collect HDF5 files from ALL subdirectories so that
            # _elev_sort_key can compare across every elevation tier.
            # Files are returned as "<subdir>/<filename>" relative paths so
            # urljoin(cur_root, name) resolves them correctly.
            all_subdir_files: List[str] = []
            for d in dirs:
                try:
                    sub_entries = _http_dir_list(urljoin(cur_root, d))
                    sub_files = [
                        e for e in sub_entries
                        if (not e.endswith("/")) and _looks_like_hdf5(e)
                    ]
                    all_subdir_files.extend(d + e for e in sub_files)
                except Exception:
                    continue
            if all_subdir_files:
                return cur_root, all_subdir_files
            # Fan-out found only further subdirectories (no HDF5 files at
            # depth 1).  Two cases require different descent strategies:
            #
            # A) Elevation tier subdirs with HDF5 files buried deeper.
            #    We must descend into the LOWEST elevation tier.
            #    Signatures:
            #      - Named dirs: any name contains "sweep", "swp", or "_EL"
            #        e.g. sweep_vol_z_0_0/, sweep_vol_z_0_24/
            #      - Plain zero-indexed numeric dirs: the set contains a
            #        dir whose bare name is literally "0" or "00" (no leading
            #        zeros beyond that), which is never a valid date component
            #        (months/days start at 01; years are 4-digit).
            #
            # B) Pure temporal navigation (2025/ -> 04/ -> 29/).
            #    Descend into the LAST (alphabetically newest) directory.
            def _dir_elev_index(d: str) -> int:
                nums = re.findall(r'\d+', d.rstrip('/'))
                return int(nums[-1]) if nums else 0

            bare_names = {d.rstrip('/') for d in dirs}
            is_elev_dirs = (
                any(re.search(r'sweep|swp|_EL\d', d, re.IGNORECASE) for d in dirs)
                or '0' in bare_names
                or '00' in bare_names
            )
            if is_elev_dirs:
                # Case A: elevation tier dirs -- pick numerically lowest
                best = min(dirs, key=_dir_elev_index)
            else:
                # Case B: date/temporal dirs -- pick alphabetically last (newest)
                best = dirs[-1]
            cur_root = urljoin(cur_root, best)
            cur_entries = _http_dir_list(cur_root)
        return cur_root, []

    entries = _http_dir_list(root)

    # Check for filter_polarimetric subdirectory (present in some DWD products)
    filt_dirs = [e for e in entries
                 if e.endswith("/") and "filter_polarimetric" in e.lower()]

    files: List[str] = []
    found_root = root

    if filt_dirs:
        filt_root = urljoin(root, sorted(filt_dirs)[0])
        try:
            filt_entries = _http_dir_list(filt_root)
            found_root, files = _find_files_in_tree(filt_root, filt_entries)
        except Exception:
            pass

    # Fall back to the non-filtered tree if filter_polarimetric gave nothing
    if not files:
        found_root, files = _find_files_in_tree(root, entries)

    if not files:
        raise RuntimeError(
            f"No DWD HDF5 sweep file for site {site_code} ({product_key})"
        )
    _ELEV_KEY_EXTS = re.compile(
        r'((?:[.-](?:hd5|h5|hdf5|bz2|gz))+$)', re.IGNORECASE
    )
    _DWD_FLAT_ELEV_RE = re.compile(
        r'_(?P<elev>\d{1,3})-(?:20\d{12,14}|LATEST)\b',
        re.IGNORECASE,
    )
    # DWD's current "vol5minng01" flat-file token order is not ascending by
    # physical elevation.  Embedded elangle values observed on 2026-04-29:
    # 00=5.5°, 01=4.5°, 02=3.5°, 03=2.5°, 04=1.5°, 05=0.5°,
    # 06=8.0°, 07=12.0°, 08=17.0°, 09=25.0°.
    _DWD_FLAT_TOKEN_ELANGLE = {
        0: 5.5, 1: 4.5, 2: 3.5, 3: 2.5, 4: 1.5,
        5: 0.5, 6: 8.0, 7: 12.0, 8: 17.0, 9: 25.0,
    }

    def _elev_sort_key(name: str) -> float:
        """Return the elevation-sweep index encoded in a DWD path.

        DWD stores one sweep per subdirectory named like:
            sweep_vol_z_0_0/  sweep_vol_z_0_1/  ...  sweep_vol_z_0_24/
        The elevation tier index is the LAST numeric token in the directory
        name.  Every file within that subdir has "_00-" as its within-file
        sweep index, so the filename alone always returns 0 and cannot
        distinguish tiers.

        Strategy:
        1. If the path contains an explicit elevation directory, use the last
           number from that directory name as the elevation tier index.
        2. For current flat DWD files, use the token immediately before the
           scan timestamp/LATEST marker (e.g. "_00-202604..." or "_09-LATEST").
        3. Fall back to a conservative basename scan for older file layouts.
        """
        parts = name.split('/')
        if len(parts) >= 2:
            dir_name = parts[-2]          # e.g. "sweep_vol_z_0_24"
            dir_nums = re.findall(r'\d+', dir_name)
            if dir_nums and re.search(r'sweep|swp|_EL\d', dir_name, re.IGNORECASE):
                return int(dir_nums[-1])  # last number = elevation tier index

        base = _ELEV_KEY_EXTS.sub('', parts[-1])
        m = _DWD_FLAT_ELEV_RE.search(base)
        if m:
            token = int(m.group("elev"))
            return _DWD_FLAT_TOKEN_ELANGLE.get(token, float(token))

        # Flat-file fallback: walk backwards through numeric tokens in the
        # stripped basename, skipping timestamps (>= 8 digits).
        nums = re.findall(r'\d+', base)
        for tok in reversed(nums):
            if len(tok) < 8:
                return int(tok)
        return 0

    def _ts(name: str) -> datetime:
        return _dwd_dt_from_name(name) or datetime.min.replace(tzinfo=timezone.utc)

    # Find the lowest elevation index present across all candidate files.
    all_elev_keys = {f: _elev_sort_key(f) for f in files}
    min_elev = min(all_elev_keys.values())

    # Among files at the lowest elevation tier, take the one with the newest
    # scan timestamp so we always serve the most recently published low-tilt
    # sweep (handles day rollover, stale cached files, etc.).
    lowest_elev_files = [f for f in files if all_elev_keys[f] == min_elev]
    latest_name = max(lowest_elev_files, key=lambda f: _ts(f))

    latest_url = urljoin(found_root, latest_name)
    dt = _dwd_dt_from_name(latest_name)
    dt_str = dt.isoformat().replace("+00:00", "Z") if dt else latest_name
    return latest_url, dt_str


def _discover_dwd_stations() -> Dict[str, dict]:
    stations: Dict[str, dict] = {}
    try:
        codes = _dwd_discover_site_codes()
    except Exception:
        return stations

    for code in codes:
        name = f"DWD {code.upper()}"
        stations[name] = {
          "provider": "dwd",
          "site": code,
          "lat": 51.0,
          "lon": 10.0,
          "range_km": 180.0,
        }
    return stations


def _all_stations() -> Dict[str, dict]:
    global _DWD_STATION_CACHE, _METEOGATE_STATION_CACHE  # _FR_STATION_CACHE removed — SEDOO disabled
    stations = dict(STATIONS)

    with _STATION_CACHE_LOCK:
        # -DWD 
        if _DWD_STATION_CACHE is None or not _DWD_STATION_CACHE:
            discovered = _discover_dwd_stations()
            if discovered:
                _DWD_STATION_CACHE = discovered
            elif _DWD_STATION_CACHE is None:
                _DWD_STATION_CACHE = {}

        # DISABLED (SEDOO): France station discovery
        # if _FR_STATION_CACHE is None:
        #     discovered_fr = _discover_fr_stations()
        #     _FR_STATION_CACHE = discovered_fr if discovered_fr else {}

        # -Meteogate (Norway) ───────────────────────────────────────────────
        if _METEOGATE_STATION_CACHE is None:
            try:
                discovered_mg = _meteogate_discover_norway_stations()
            except Exception:
                discovered_mg = {}
            _METEOGATE_STATION_CACHE = discovered_mg if discovered_mg else {}

    stations.update(_DWD_STATION_CACHE or {})
    # stations.update(_FR_STATION_CACHE or {})  # DISABLED — SEDOO
    stations.update(_METEOGATE_STATION_CACHE or {})

    with _CUSTOM_LOCK:
        stations.update(_CUSTOM_STATIONS)

    return stations


def _load_tvs_icons() -> Dict[str, str]:
    global _TVS_ICON_CACHE
    if _TVS_ICON_CACHE is not None:
        return _TVS_ICON_CACHE

    base_dir = Path(__file__).resolve().parent
    icons: Dict[str, str] = {}
    for level in range(1, 6):
        icon_path = base_dir / f"tvs{level}.png"
        if not icon_path.exists():
            continue
        raw = icon_path.read_bytes()
        icons[str(level)] = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

    _TVS_ICON_CACHE = icons
    return icons

def _summarize_hdf_bytes(file_bytes: bytes, max_items: int = 25) -> str:
    """Return a short structural summary of an HDF5 file for debugging."""
    try:
        with h5py.File(io.BytesIO(file_bytes), "r") as hdf:
            keys = list(hdf.keys())[:20]
            attrs = list(getattr(hdf, "attrs", {}).keys())[:20]
            items: List[str] = []

            def _visit(name, obj):
                if len(items) >= max_items:
                    return
                try:
                    if isinstance(obj, h5py.Dataset):
                        items.append(f"{name} [D] shape={getattr(obj, 'shape', None)}")
                    else:
                        items.append(f"{name} [G]")
                except Exception:
                    items.append(f"{name} [?]")

            try:
                hdf.visititems(_visit)
            except Exception:
                pass
            # Add a focused peek into scan*/radar groups if present
            scan_peek: List[str] = []
            for gname in ("scan1", "scan2", "scan3", "radar1", "overview", "image1", "geographic"):
                try:
                    grp = hdf.get(gname)
                except Exception:
                    grp = None
                if grp is None:
                    continue
                try:
                    sub_keys = list(grp.keys())[:20]
                except Exception:
                    sub_keys = []
                sub_items: List[str] = []
                if sub_keys:
                    for sk in sub_keys[:10]:
                        try:
                            sobj = grp.get(sk)
                        except Exception:
                            sobj = None
                        if sobj is None:
                            continue
                        try:
                            if isinstance(sobj, h5py.Dataset):
                                sub_items.append(f"{gname}/{sk}[D] shape={getattr(sobj,'shape',None)}")
                            else:
                                sub_items.append(f"{gname}/{sk}[G]")
                        except Exception:
                            sub_items.append(f"{gname}/{sk}[?]")
                scan_peek.append(f"{gname}:keys={sub_keys}, items={sub_items}")
            return f"keys={keys} attrs={attrs} items={items} scan_peek={scan_peek}"
    except Exception as exc:
        return f"summary-failed:{exc}"


_CUSTOM_IRIS_SIGMET_EXTENSIONS = {
    "RAWUCGE",
    "RAWUGG",
    "RAWUCFR",
    "RAWUCGA",
    "RAWUCGC",
    "RAWUCEG",
    "RAWUCFL",
    "RAWUCCW",
    "RAWUCCY",
    "RAWUCD0",
    "RAWUCD2",
    "RAWUCBG",
    "RAWUCCK",
    "RAWUCCS",
    "RAWUCCU",
    "RAWUCBF",
}
_CUSTOM_FURUNO_EXTENSIONS = {".scn", ".scnx"}
_CUSTOM_BUFR_EXTENSIONS = {".bufr", ".buf", ".bfr"}
# NEXRAD Level-2 MSG31 file extensions.
# ".msg31" – raw concatenated 2620-byte message frames, no compression/wrapper.
# ".ar2v"  – Archive II wrapper (LDM bz2-compressed records, "AR2V" magic header).
# Files from NCEI/AWS with no extension are handled via payload magic detection.
_CUSTOM_MSG31_EXTENSIONS = {".msg31", ".ar2v", ".bz2.ar2v"}
_CUSTOM_COMPRESSED_EXTENSIONS = {".gz", ".gzip", ".bz2", ".bzip2"}
_BUFR_CONVERTER_ENV_VARS = (
    "EMRAW_BUFR_CONVERTER",
    "CUSTOM_RADAR_BUFR_CONVERTER",
    "OPERA_BUFR_CONVERTER",
)
_XRADAR_QTY_ALIASES = {
    "DB_DBZ": "DBZH",
    "DB_DBZC": "DBZH_corr",
    "DB_DBTH": "DBTH",
    "DB_TH": "TH",
    "DB_VEL": "VRADH",
    "DB_VELC": "VRAD_corr",
    "DB_WIDTH": "WRADH",
    "DB_ZDR": "ZDR",
    "DB_KDP": "KDP",
    "DB_PHIDP": "PHIDP",
    "DB_PHIDP2": "PHIDP",
    "DB_RHOHV": "RHOHV",
    "DB_SQI": "SQIH",
}

# NEXRAD MSG31 3-character data-block name → internal canonical quantity name.
# "SW " (with trailing space) is the raw encoding used in the ICD for spectrum width.
_MSG31_MOMENT_NAMES: Dict[str, str] = {
    "REF": "DBZH",    # Reflectivity
    "VEL": "VRADH",   # Radial velocity
    "SW ": "WRADH",   # Spectrum width  (note: trailing space is part of the ICD encoding)
    "ZDR": "ZDR",     # Differential reflectivity
    "PHI": "PHIDP",   # Differential phase
    "RHO": "RHOHV",   # Correlation coefficient
    "SNR": "SNR",     # Signal-to-noise ratio
    "CFP": "CFP",     # Clutter filter power removed
    "HHC": "HHC",     # Hybrid hydrometeor classification
}


def _strip_compression_suffixes(filename: str) -> str:
    base = Path(str(filename or "").strip() or "custom").name
    while True:
        suffix = Path(base).suffix
        if suffix and suffix.lower() in _CUSTOM_COMPRESSED_EXTENSIONS:
            base = Path(base).stem
            continue
        return base


def _custom_primary_suffix(filename: str) -> str:
    return Path(_strip_compression_suffixes(filename)).suffix.lower()


def _looks_like_iris_sigmet_filename(filename: str) -> bool:
    base = _strip_compression_suffixes(filename)
    token = (Path(base).suffix.lstrip(".") or Path(base).name).upper()
    if token in _CUSTOM_IRIS_SIGMET_EXTENSIONS:
        return True
    # Covers both older and newer SIGMET/IRIS RAW* variants, e.g. RAW2049 / RAWKPJV.
    return bool(re.fullmatch(r"RAW[A-Z0-9]{3,8}", token))


def _looks_like_furuno_filename(filename: str) -> bool:
    return _custom_primary_suffix(filename) in _CUSTOM_FURUNO_EXTENSIONS


def _looks_like_bufr_payload(filename: str, file_bytes: bytes) -> bool:
    if _custom_primary_suffix(filename) in _CUSTOM_BUFR_EXTENSIONS:
        return True
    return bytes(file_bytes[:4]) == b"BUFR"


def _looks_like_msg31_filename(filename: str) -> bool:
    """Return True when the filename looks like a NEXRAD Level-2 Archive II file.

    NCEI/AWS raw downloads are often named like ``KBMX20230601_120001_V06``
    (no extension); newer exports may carry ``.ar2v``.  We match both patterns.
    """
    base = _strip_compression_suffixes(filename)
    stem = Path(base).stem  # filename without any extension
    suffix = Path(base).suffix.lower()
    if suffix in _CUSTOM_MSG31_EXTENSIONS:
        return True
    # Bare NEXRAD filenames: 4-char ICAO ID + 8-digit date + _ + 6-digit time + _V06
    if re.fullmatch(r"[A-Z]{4}\d{8}_\d{6}(?:_V\d+)?", stem, re.IGNORECASE):
        return True
    return False


def _looks_like_msg31_payload(file_bytes: bytes) -> bool:
    """Return True when the payload looks like a NEXRAD Level-2 MSG31 file.

    Two sub-formats are recognised:

    * **Archive II** (``AR2V`` / ``ARCHIVE2`` magic, LDM bz2-compressed records)
    * **Raw MSG31** (bare concatenated 2620-byte message frames, no header/compression)
      Identified by: file length is a multiple of 2620 *and* the message-type byte
      at offset 15 of the first frame equals 31 (0x1F).
    """
    if len(file_bytes) < 24:
        return False
    # Archive II magic
    header = file_bytes[:8]
    if header[:4] == b"AR2V" or header[:8] == b"ARCHIVE2":
        return True
    # Raw MSG31: file is a multiple of 2620-byte frames; message-type byte = 31
    if (len(file_bytes) >= 2620
            and len(file_bytes) % 2620 == 0
            and file_bytes[15] == 31):
        return True
    return False


def _temp_filename_for_custom_reader(filename: str, fallback_suffix: str) -> str:
    base = _strip_compression_suffixes(filename)
    if not base:
        base = f"custom{fallback_suffix}"
    if not Path(base).suffix and fallback_suffix:
        base += fallback_suffix
    return Path(base).name


def _xradar_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if hasattr(value, "values"):
            value = value.values
    except Exception:
        pass
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore").strip()
        except Exception:
            return value
    if isinstance(value, str):
        return value.strip()
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        scalar = arr.reshape(-1)[0]
        if hasattr(scalar, "item"):
            scalar = scalar.item()
        if isinstance(scalar, bytes):
            try:
                return scalar.decode("utf-8", errors="ignore").strip()
            except Exception:
                return scalar
        return scalar
    except Exception:
        return value


def _xradar_find_scalar(obj: Any, *keys: str) -> Any:
    for key in keys:
        try:
            attrs = getattr(obj, "attrs", None)
            if attrs is not None and key in attrs:
                val = _xradar_scalar(attrs.get(key))
                if val not in (None, ""):
                    return val
        except Exception:
            pass
        try:
            coords = getattr(obj, "coords", None)
            if coords is not None and key in coords:
                val = _xradar_scalar(coords[key])
                if val not in (None, ""):
                    return val
        except Exception:
            pass
        try:
            if key in obj:
                val = _xradar_scalar(obj[key])
                if val not in (None, ""):
                    return val
        except Exception:
            pass
        try:
            val = _xradar_scalar(getattr(obj, key))
            if val not in (None, ""):
                return val
        except Exception:
            pass
    return None


def _xradar_find_float(obj: Any, *keys: str) -> Optional[float]:
    val = _xradar_find_scalar(obj, *keys)
    if val in (None, ""):
        return None
    try:
        f = float(val)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return None


def _iter_xradar_sweep_datasets(dtree: Any) -> List[Any]:
    datasets: List[Any] = []
    seen: Set[str] = set()

    def _add_node(node: Any) -> None:
        if node is None:
            return
        name = str(getattr(node, "name", "") or "")
        path = str(getattr(node, "path", "") or name)
        if not name.startswith("sweep_") and "/sweep_" not in path:
            return
        ds = getattr(node, "ds", None)
        if ds is None:
            try:
                ds = node.to_dataset()
            except Exception:
                ds = None
        if ds is None:
            return
        key = path or name or str(id(node))
        if key in seen:
            return
        seen.add(key)
        datasets.append(ds)

    try:
        subtree = getattr(dtree, "subtree", None)
        if subtree is not None:
            iterator = subtree if not callable(subtree) else subtree()
            for node in iterator:
                _add_node(node)
    except Exception:
        pass

    if not datasets:
        try:
            children = getattr(dtree, "children", None)
            if hasattr(children, "items"):
                for _, node in children.items():
                    _add_node(node)
        except Exception:
            pass

    if not datasets:
        try:
            for key in list(dtree.keys()):
                try:
                    _add_node(dtree[key])
                except Exception:
                    continue
        except Exception:
            pass

    return datasets


def _xradar_quantity_name(var_name: str, data_var: Any) -> str:
    raw_name = str(var_name or "").strip()
    upper_name = raw_name.upper()
    if upper_name in _XRADAR_QTY_ALIASES:
        return _XRADAR_QTY_ALIASES[upper_name]

    attrs = getattr(data_var, "attrs", {}) or {}
    std = str(attrs.get("standard_name") or "").strip().lower()
    long_name = str(attrs.get("long_name") or "").strip().lower()
    if "specific_differential_phase" in std:
        return "KDP"
    if "differential_phase" in std:
        return "PHIDP"
    if "correlation_coefficient" in std:
        return "RHOHV"
    if "differential_reflectivity" in std:
        return "ZDR"
    if "doppler_spectrum_width" in std:
        return "WRADH"
    if "radial_velocity" in std:
        return "VRADH"
    if "reflectivity" in std:
        if "uncorrected" in long_name or "total power" in long_name:
            return "TH"
        return "DBZH"
    return upper_name or raw_name


def _extract_xradar_tree_sweeps(dtree: Any) -> Tuple[Dict[str, List[dict]], dict]:
    sweeps_by_qty: Dict[str, List[dict]] = {}
    meta: dict = {"lat": None, "lon": None, "max_range_km": None}
    lowest_angle = 999.0

    root_ds = getattr(dtree, "ds", None)
    if root_ds is not None:
        meta["lat"] = _xradar_find_float(root_ds, "latitude", "lat")
        meta["lon"] = _xradar_find_float(root_ds, "longitude", "lon")

    sweep_datasets = _iter_xradar_sweep_datasets(dtree)
    if not sweep_datasets:
        raise RuntimeError("xradar did not expose any sweep datasets")

    for ds in sweep_datasets:
        if meta["lat"] is None:
            meta["lat"] = _xradar_find_float(ds, "latitude", "lat")
        if meta["lon"] is None:
            meta["lon"] = _xradar_find_float(ds, "longitude", "lon")

        range_da = None
        try:
            coords = getattr(ds, "coords", None)
            if coords is not None and "range" in coords:
                range_da = coords["range"]
        except Exception:
            range_da = None
        if range_da is None:
            try:
                if "range" in ds:
                    range_da = ds["range"]
            except Exception:
                range_da = None
        if range_da is None:
            continue

        try:
            range_vals = np.asarray(getattr(range_da, "values", range_da), dtype=np.float64).ravel()
        except Exception:
            continue
        if range_vals.size == 0:
            continue

        step_raw = np.nan
        if range_vals.size >= 2:
            try:
                diffs = np.diff(range_vals.astype(np.float64))
                diffs = diffs[np.isfinite(diffs)]
                if diffs.size:
                    step_raw = float(np.nanmedian(np.abs(diffs)))
            except Exception:
                step_raw = np.nan
        if not np.isfinite(step_raw) or step_raw <= 0:
            step_raw = float(
                _xradar_find_float(
                    range_da,
                    "meters_between_gates",
                    "resolution_range_direction",
                    "gate_spacing",
                    "gate_size",
                )
                or 1000.0
            )

        first_center_raw = float(range_vals[0])
        if abs(step_raw) > 20.0 or abs(first_center_raw) > 20.0:
            rscale_km = float(step_raw) / 1000.0
            rstart_km = max(0.0, (first_center_raw - 0.5 * float(step_raw)) / 1000.0)
        else:
            rscale_km = float(step_raw)
            rstart_km = max(0.0, first_center_raw - 0.5 * float(step_raw))

        nbins = int(range_vals.size)
        elev = _xradar_find_float(ds, "sweep_fixed_angle", "fixed_angle")
        if elev is None:
            try:
                elev_coord = getattr(ds, "coords", {}).get("elevation")
                if elev_coord is not None:
                    elev_vals = np.asarray(elev_coord.values, dtype=np.float64).ravel()
                    elev_vals = elev_vals[np.isfinite(elev_vals)]
                    if elev_vals.size:
                        elev = float(np.nanmedian(elev_vals))
            except Exception:
                elev = None
        if elev is None:
            elev = 0.0

        max_rng = float(rstart_km + nbins * rscale_km)
        if elev < lowest_angle:
            lowest_angle = elev
            meta["max_range_km"] = max_rng

        nyquist = _xradar_find_float(ds, "nyquist_velocity", "unambiguous_velocity")
        low_nyquist = _xradar_find_float(ds, "low_nyquist_velocity", "low_unambiguous_velocity")
        if low_nyquist is None and nyquist:
            prf1 = _xradar_find_float(ds, "prf_1", "prf1", "lowprf")
            prf2 = _xradar_find_float(ds, "prf_2", "prf2", "highprf")
            try:
                if prf1 and prf2:
                    low_nyquist = abs(float(nyquist)) * min(float(prf1), float(prf2)) / max(float(prf1), float(prf2))
            except Exception:
                low_nyquist = None

        data_vars = getattr(ds, "data_vars", {}) or {}
        for var_name, data_var in data_vars.items():
            if getattr(data_var, "ndim", 0) != 2:
                continue
            dims = [str(dim) for dim in getattr(data_var, "dims", ())]
            if len(dims) != 2 or "range" not in dims:
                continue

            try:
                raw = data_var.values
            except Exception:
                continue
            if np.ma.isMaskedArray(raw):
                arr = np.asarray(raw.filled(np.nan), dtype=np.float32)
            else:
                arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim != 2:
                continue

            range_axis = dims.index("range")
            ray_axis = 1 - range_axis
            if (ray_axis, range_axis) != (0, 1):
                arr = np.moveaxis(arr, (ray_axis, range_axis), (0, 1))

            nrays, nbins_arr = arr.shape
            if nrays <= 0 or nbins_arr <= 0:
                continue
            if nbins_arr != nbins:
                common_bins = min(nbins_arr, nbins)
                if common_bins <= 0:
                    continue
                arr = arr[:, :common_bins]
                nbins_eff = common_bins
            else:
                nbins_eff = nbins

            qty = _xradar_quantity_name(var_name, data_var)
            if not qty:
                continue

            units = str((getattr(data_var, "attrs", {}) or {}).get("units") or "").strip().lower()
            if qty in ("VRAD", "VRADH", "VRADV", "VRADC", "VR", "V", "VRAD_corr") and units in ("km/h", "kmh", "kph", "km h-1"):
                arr = arr / 3.6

            if qty in ("RHOHV", "RHOHVH", "URHOHV", "RHO", "CCORH", "CC"):
                valid = arr[np.isfinite(arr)]
                if valid.size > 0 and float(np.nanmax(valid)) <= 1.5:
                    arr = arr * 100.0

            max_rng_eff = float(rstart_km + nbins_eff * rscale_km)
            sweeps_by_qty.setdefault(qty, []).append(
                {
                    "elevation": elev,
                    "rscale_km": rscale_km,
                    "rstart_km": rstart_km,
                    "max_range_km": max_rng_eff,
                    "nrays": int(nrays),
                    "nbins": int(nbins_eff),
                    "_polar": arr,
                    "nyquist_m_s": None if nyquist is None else float(abs(nyquist)),
                    "low_nyquist_m_s": None if low_nyquist is None else float(abs(low_nyquist)),
                }
            )

    for qty in sweeps_by_qty:
        sweeps_by_qty[qty].sort(key=lambda sweep: float(sweep.get("elevation", 0.0)))

    if not sweeps_by_qty:
        raise RuntimeError("xradar opened the file but found no 2-D radar moments")

    return sweeps_by_qty, meta


def _extract_xradar_sweeps(file_bytes: bytes, filename: str, engine: str) -> Tuple[Dict[str, List[dict]], dict]:
    if not HAS_XRADAR:
        raise RuntimeError(f"Missing xradar/xarray dependency: {XRADAR_IMPORT_ERROR}")

    fallback_suffix = ".raw" if engine == "iris" else ".scn"
    temp_name = _temp_filename_for_custom_reader(filename, fallback_suffix)

    with tempfile.TemporaryDirectory(prefix=f"emraw_{engine}_") as tmpdir:
        temp_path = Path(tmpdir) / temp_name
        temp_path.write_bytes(file_bytes)
        try:
            io_mod = getattr(xd, "io", None)
            if engine == "iris":
                opener = getattr(io_mod, "open_iris_datatree", None)
                if opener is None:
                    raise RuntimeError("xradar iris reader is not available in this installation")
                dtree = opener(str(temp_path), sweep=None)
            elif engine == "furuno":
                opener = getattr(io_mod, "open_furuno_datatree", None)
                if opener is None:
                    raise RuntimeError("xradar furuno reader is not available in this installation")
                dtree = opener(str(temp_path), sweep=None)
            else:
                raise RuntimeError(f"Unsupported xradar engine '{engine}'")
            return _extract_xradar_tree_sweeps(dtree)
        except Exception as exc:
            raise RuntimeError(f"xradar {engine} reader failed for {Path(temp_name).name}: {exc}") from exc


def _bufr_converter_command() -> str:
    for env_key in _BUFR_CONVERTER_ENV_VARS:
        value = str(os.environ.get(env_key) or "").strip()
        if value:
            return value
    return ""


def _extract_bufr_sweeps(file_bytes: bytes, filename: str) -> Tuple[Dict[str, List[dict]], dict]:
    converter = _bufr_converter_command()
    if not converter:
        raise RuntimeError(
            "BUFR requires an external converter. Set EMRAW_BUFR_CONVERTER "
            "(or CUSTOM_RADAR_BUFR_CONVERTER / OPERA_BUFR_CONVERTER) to a command "
            "that accepts {input} and {output}, or to an executable that reads "
            "<input> and writes <output>."
        )

    input_name = _temp_filename_for_custom_reader(filename or "custom.bufr", ".bufr")
    with tempfile.TemporaryDirectory(prefix="emraw_bufr_") as tmpdir:
        input_path = Path(tmpdir) / input_name
        output_path = Path(tmpdir) / f"{Path(input_name).stem}.h5"
        input_path.write_bytes(file_bytes)

        if "{input}" in converter or "{output}" in converter:
            cmd = converter.format(input=str(input_path), output=str(output_path))
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            proc = subprocess.run(
                [converter, str(input_path), str(output_path)],
                capture_output=True,
                text=True,
            )

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"BUFR converter failed (exit {proc.returncode}): {detail[:400]}")

        if not output_path.exists():
            candidates = sorted(Path(tmpdir).glob("*.h5")) + sorted(Path(tmpdir).glob("*.hdf5"))
            if not candidates:
                raise RuntimeError("BUFR converter did not produce an HDF5 file")
            output_path = candidates[0]

        converted_bytes = _decompress_if_needed(output_path.read_bytes())
        try:
            return _extract_all_sweeps(converted_bytes)
        except Exception as exc:
            try:
                return _extract_knmi_sweeps(converted_bytes)
            except Exception as exc2:
                raise RuntimeError(
                    f"Converted BUFR output was not readable as radar HDF5 ({exc}; {exc2})"
                ) from exc2


# -MSG31 (NEXRAD Level-2 Archive II) reader
#
# Reference: WSR-88D Interface Control Document for the RDA/RPG, Build 19.0+,
# Message Type 31 – Digital Radar Data.
#
# File layout
#   [24 bytes]  Volume Scan Title  (e.g. "AR2V0006.xxx")
#   [N records] Each record is preceded by a signed 32-bit big-endian integer:
#               positive → the record is bz2-compressed (size = abs value)
#               negative → the record is uncompressed   (size = abs value)
#     Each decompressed record is a multiple of 2620 bytes; one message per
#     2620-byte frame.  Multi-segment MSG31 radials span consecutive frames with
#     the same ID-sequence number.
#
# MSG31 Radial Header (offsets from the start of the message body, i.e. after
# the 12-byte CTM header and 16-byte message header):
#   00-03  Radar Identifier (ASCII, 4 chars, e.g. "KBMX")
#   04-07  Collection time (ms past midnight, uint32 BE)
#   08-09  Modified Julian date (days since 1/1/70, uint16 BE)
#   10-11  Azimuth number (1-based sequence, uint16 BE)
#   12-15  Azimuth angle (IEEE float32 BE, degrees 0–360)
#   16     Compression indicator (0=none, 1=bz2, 2=zlib)
#   17     Spare
#   18-19  Radial length (uint16 BE, bytes of MSG31 data)
#   20     Azimuth resolution spacing (1 = 0.5°, 2 = 1.0°)
#   21     Radial status (0=start-of-elev, 1=inter, 2=end-of-elev, …)
#   22     Elevation number (1-based)
#   23     Cut sector number
#   24-27  Elevation angle (IEEE float32 BE, degrees)
#   28     Radial spot blanking status
#   29     Azimuth indexing mode
#   30-31  Data block count N (uint16 BE)
#   32+    N × 4-byte data block pointers (uint32 BE, offset from byte 0 here)
#
# Data Block types (identified by [ptr+0] type char and [ptr+1:ptr+4] name):
#   'R'+"VOL"  Volume Data Constant  – carries radar lat/lon
#   'R'+"ELV"  Elevation Data Constant
#   'R'+"RAD"  Radial Data Constant  – carries Nyquist velocity
#   'D'+name   Moment data (REF, VEL, SW , ZDR, PHI, RHO, SNR, CFP, HHC)
#
# Volume Data Constant Block offsets (relative to the block pointer):
#   00     'R'
#   01-03  "VOL"
#   04-05  Block size (uint16)
#   06-07  LRTUP
#   08-09  Version major
#   10-11  Version minor
#   12-15  Latitude  (float32 BE, degrees N)
#   16-19  Longitude (float32 BE, degrees E)
#   …
#
# Radial Data Constant Block offsets:
#   00     'R'
#   01-03  "RAD"
#   04-05  Block size
#   06-07  Unambiguous range (uint16, 0.1 km)
#   08-09  Noise horizontal
#   10-11  Noise vertical
#   12-13  Nyquist velocity (uint16, 0.01 m/s)
#   …
#
# Data Moment Block offsets:
#   00     'D'
#   01-03  data name (e.g. "REF", "VEL", "SW ", "ZDR", "PHI", "RHO")
#   04-07  Reserved
#   08-09  Number of gates (uint16)
#   10-11  Range to first gate (uint16, 0.001 km)
#   12-13  Gate size (uint16, 0.001 km)
#   14-15  RF threshold (uint16, 0.1 dB)
#   16-17  SNR threshold (int16, 0.1 dB)
#   18     Control flags
#   19     Word size (8 or 16 bits)
#   20-23  Scale  (float32 BE)
#   24-27  Offset (float32 BE)
#   28+    Gate data: n_gates × (word_size/8) bytes
#
# Physical value formula:  value = (raw_code − offset) / scale
#   raw codes 0 and 1 are reserved (0 = below-threshold, 1 = range-folded).

def _extract_msg31_sweeps(file_bytes: bytes) -> Tuple[Dict[str, List[dict]], dict]:
    """Parse a NEXRAD Level-2 Archive II (MSG31) file into sweeps.

    Returns the same ``(sweeps_by_qty, meta)`` tuple format as every other
    sweep-extraction function so it can be dropped into the normal pipeline.
    """
    if not HAS_RADAR_DEPS:
        raise RuntimeError(
            f"MSG31 parser requires numpy (missing radar dependencies: {RADAR_DEPS_ERROR})"
        )
    if not (_looks_like_msg31_payload(file_bytes) or _looks_like_msg31_filename("")):
        raise RuntimeError("File does not look like a NEXRAD MSG31 file")

    has_ar2v_header = (len(file_bytes) >= 8
                       and (file_bytes[:4] == b"AR2V" or file_bytes[:8] == b"ARCHIVE2"))

    # 1. Collect 2620-byte message frames 
    all_frames: List[bytes] = []

    if has_ar2v_header:
        # Archive II format: 24-byte Volume Scan Title followed by LDM records.
        # Each record is headed by a signed 32-bit big-endian length; positive
        # means the payload is bz2-compressed, negative means it is raw.
        pos = 24
        while pos < len(file_bytes):
            if pos + 4 > len(file_bytes):
                break
            rec_size_signed = struct.unpack(">i", file_bytes[pos:pos + 4])[0]
            pos += 4
            if rec_size_signed == 0:
                continue
            abs_size = abs(rec_size_signed)
            if pos + abs_size > len(file_bytes):
                abs_size = len(file_bytes) - pos  # tolerate truncated final record
            chunk = file_bytes[pos:pos + abs_size]
            pos += abs_size

            if rec_size_signed > 0:
                try:
                    chunk = bz2.decompress(chunk)
                except Exception:
                    continue  # skip corrupted record

            frame_pos = 0
            while frame_pos + 2620 <= len(chunk):
                all_frames.append(chunk[frame_pos:frame_pos + 2620])
                frame_pos += 2620
    else:
        # Raw MSG31 format: plain concatenated 2620-byte frames, no wrapper.
        # The file begins directly at the first CTM header (byte 0).
        frame_pos = 0
        while frame_pos + 2620 <= len(file_bytes):
            all_frames.append(file_bytes[frame_pos:frame_pos + 2620])
            frame_pos += 2620

    if not all_frames:
        raise RuntimeError(
            "No MSG31 frames found – the file may be corrupt or use an unsupported "
            "sub-format (expected Archive II bz2-records or raw 2620-byte frames)"
        )

    # 2. Reassemble multi-segment MSG31 radials
    # Message header layout (bytes 12–27 of each 2620-byte frame):
    #   12-13  message size in half-words (uint16 BE)
    #   14     RDA channel
    #   15     message type
    #   16-17  ID sequence number (uint16 BE)
    #   18-19  Julian date
    #   20-23  ms past midnight
    #   24-25  number of message segments (uint16 BE)
    #   26-27  segment number (uint16 BE, 1-based)
    # The message body starts at byte 28.

    # segments[seq] → list of (seg_index, body_bytes)
    segments: Dict[int, Dict[int, bytes]] = {}
    n_segs_for: Dict[int, int] = {}

    for frame in all_frames:
        if len(frame) < 28:
            continue
        msg_type = frame[15]
        if msg_type != 31:
            continue
        seq    = struct.unpack(">H", frame[16:18])[0]
        n_segs = struct.unpack(">H", frame[24:26])[0]
        seg    = struct.unpack(">H", frame[26:28])[0]  # 1-based
        body   = frame[28:]

        n_segs_for[seq] = n_segs
        segments.setdefault(seq, {})[seg] = body

    # Join complete radials and sort by sequence number
    radial_bodies: List[Tuple[int, bytes]] = []
    for seq, seg_dict in segments.items():
        n_segs = n_segs_for.get(seq, 1)
        if len(seg_dict) < n_segs:
            continue  # incomplete multi-segment radial – skip
        assembled = b"".join(seg_dict[i] for i in range(1, n_segs + 1))
        radial_bodies.append((seq, assembled))
    radial_bodies.sort(key=lambda t: t[0])

    if not radial_bodies:
        raise RuntimeError("No complete MSG31 radials found in the file")

    # 3. Parse radials
    meta: dict = {"lat": None, "lon": None, "max_range_km": None}

    # Per elevation-cut accumulator
    # elev_cuts[elev_num] = {
    #     "elevation_angle": float,
    #     "azimuths":  List[float],
    #     "nyquist_m_s": Optional[float],
    #     "moments": {
    #         block_name: {
    #             "n_gates": int, "range_km": float, "gate_size_km": float,
    #             "rays": List[np.ndarray]  (float32, NaN for invalid)
    #         }
    #     }
    # }
    elev_cuts: Dict[int, dict] = {}

    for _seq, body in radial_bodies:
        if len(body) < 32:
            continue

        # --- Radial header ---
        az_angle   = struct.unpack(">f", body[12:16])[0]
        elev_num   = body[22]
        elev_angle = struct.unpack(">f", body[24:28])[0]
        n_blocks   = struct.unpack(">H", body[30:32])[0]

        if elev_num not in elev_cuts:
            elev_cuts[elev_num] = {
                "elevation_angle": elev_angle,
                "azimuths":        [],
                "nyquist_m_s":     None,
                "moments":         {},
            }
        cut = elev_cuts[elev_num]
        cut["azimuths"].append(az_angle)

        # --- Walk data block pointers ---
        for i in range(n_blocks):
            ptr_off = 32 + 4 * i
            if ptr_off + 4 > len(body):
                break
            ptr = struct.unpack(">I", body[ptr_off:ptr_off + 4])[0]
            if ptr + 4 > len(body):
                continue

            blk_type = chr(body[ptr]) if body[ptr] < 128 else ""
            blk_name = body[ptr + 1:ptr + 4].decode("ascii", errors="replace")

            # -Volume Data Constant Block (lat/lon) 
            if blk_type == "R" and blk_name == "VOL":
                if meta["lat"] is None and ptr + 20 <= len(body):
                    lat = struct.unpack(">f", body[ptr + 12:ptr + 16])[0]
                    lon = struct.unpack(">f", body[ptr + 16:ptr + 20])[0]
                    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                        meta["lat"] = float(lat)
                        meta["lon"] = float(lon)
                continue

            # -Radial Data Constant Block (Nyquist velocity)
            if blk_type == "R" and blk_name == "RAD":
                if cut["nyquist_m_s"] is None and ptr + 14 <= len(body):
                    nyq_raw = struct.unpack(">H", body[ptr + 12:ptr + 14])[0]
                    if nyq_raw > 0:
                        # ICD: stored in units of 0.01 m/s
                        cut["nyquist_m_s"] = float(nyq_raw) * 0.01
                continue

            # -Data Moment Block
            if blk_type != "D":
                continue
            if ptr + 28 > len(body):
                continue

            n_gates   = struct.unpack(">H", body[ptr + 8:ptr + 10])[0]
            rng_first = struct.unpack(">H", body[ptr + 10:ptr + 12])[0]  # 0.001 km
            gate_size = struct.unpack(">H", body[ptr + 12:ptr + 14])[0]  # 0.001 km
            word_size = body[ptr + 19]                                    # 8 or 16

            if n_gates == 0 or word_size not in (8, 16):
                continue

            scale  = struct.unpack(">f", body[ptr + 20:ptr + 24])[0]
            offset = struct.unpack(">f", body[ptr + 24:ptr + 28])[0]

            if scale == 0.0:
                continue  # guard against divide-by-zero

            bytes_per_gate = word_size // 8
            data_start     = ptr + 28
            available      = (len(body) - data_start) // bytes_per_gate
            n_gates_eff    = min(n_gates, available)
            if n_gates_eff <= 0:
                continue

            raw_bytes = body[data_start:data_start + n_gates_eff * bytes_per_gate]
            if word_size == 8:
                raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
            else:
                raw_arr = np.frombuffer(raw_bytes, dtype=">u2").astype(np.float32)

            # ICD decoding: physical = (raw − offset) / scale
            # raw codes 0 (below threshold) and 1 (range-folded) → NaN
            decoded = np.where(raw_arr > 1.0, (raw_arr - offset) / scale, np.nan).astype(np.float32)

            # Pad to declared gate count if we were truncated
            if n_gates_eff < n_gates:
                padded_ray = np.full(n_gates, np.nan, dtype=np.float32)
                padded_ray[:n_gates_eff] = decoded
                decoded = padded_ray

            mdict = cut["moments"]
            if blk_name not in mdict:
                mdict[blk_name] = {
                    "n_gates":     n_gates,
                    "range_km":    float(rng_first) * 0.001,
                    "gate_size_km": float(gate_size) * 0.001,
                    "rays":        [],
                }
            mdict[blk_name]["rays"].append(decoded)

    if not elev_cuts:
        raise RuntimeError("No elevation cuts parsed from MSG31 file")

    # 4. Build sweeps_by_qty 
    sweeps_by_qty: Dict[str, List[dict]] = {}
    lowest_angle = 999.0

    for elev_num in sorted(elev_cuts.keys()):
        cut        = elev_cuts[elev_num]
        elev_angle = float(cut["elevation_angle"])
        azimuths   = cut["azimuths"]
        nyquist    = cut["nyquist_m_s"]

        for blk_name, mdata in cut["moments"].items():
            rays = mdata["rays"]
            if not rays:
                continue

            # Normalise to a rectangular array (all rays padded to equal length)
            max_bins = max(len(r) for r in rays)
            polar = np.full((len(rays), max_bins), np.nan, dtype=np.float32)
            for idx, r in enumerate(rays):
                polar[idx, :len(r)] = r

            # Roll so that the ray nearest to azimuth 0° (North) is first
            if azimuths and len(azimuths) == len(rays):
                az_arr   = np.array(azimuths, dtype=np.float32)
                dist     = np.minimum(az_arr, 360.0 - az_arr)
                north_idx = int(np.argmin(dist))
                polar    = np.roll(polar, -north_idx, axis=0)

            # Map block name to canonical quantity name
            canonical = _MSG31_MOMENT_NAMES.get(blk_name, blk_name.strip())

            # RHOHV: if encoded as 0–1 fractions, scale to 0–100 to match colourmap
            if canonical in ("RHOHV", "RHO"):
                valid = polar[np.isfinite(polar)]
                if valid.size > 0 and float(np.nanmax(valid)) <= 1.5:
                    polar = polar * 100.0

            n_rays, n_bins = polar.shape
            rstart_km  = float(mdata["range_km"])
            rscale_km  = float(mdata["gate_size_km"])
            max_rng_km = rstart_km + n_bins * rscale_km

            if elev_angle < lowest_angle:
                lowest_angle        = elev_angle
                meta["max_range_km"] = max_rng_km

            sweeps_by_qty.setdefault(canonical, []).append({
                "elevation":        elev_angle,
                "rscale_km":        rscale_km,
                "rstart_km":        rstart_km,
                "max_range_km":     max_rng_km,
                "nrays":            n_rays,
                "nbins":            n_bins,
                "_polar":           polar,
                "nyquist_m_s":      nyquist,
                "low_nyquist_m_s":  None,
            })

    for qty in sweeps_by_qty:
        sweeps_by_qty[qty].sort(key=lambda s: float(s["elevation"]))

    if not sweeps_by_qty:
        raise RuntimeError("MSG31 parser: no moment data could be assembled from radials")

    return sweeps_by_qty, meta


def _extract_xradar_nexrad_sweeps(file_bytes: bytes, filename: str) -> Tuple[Dict[str, List[dict]], dict]:
    """Try to read a NEXRAD Level-2 file via xradar's NEXRAD reader (optional fast path).

    xradar ≥ 0.4 ships ``xd.io.open_nexradlevel2_datatree``; older builds may not
    have it.  Raises RuntimeError if xradar is unavailable or the reader is absent.
    """
    if not HAS_XRADAR:
        raise RuntimeError(f"xradar not available: {XRADAR_IMPORT_ERROR}")

    io_mod = getattr(xd, "io", None)
    opener = (
        getattr(io_mod, "open_nexradlevel2_datatree", None)
        or getattr(io_mod, "open_nexrad_archive_datatree", None)
    )
    if opener is None:
        raise RuntimeError(
            "xradar NEXRAD Level-2 reader not available in this installation "
            "(need xradar >= 0.4 with open_nexradlevel2_datatree)"
        )

    temp_name = _temp_filename_for_custom_reader(filename, ".ar2v")
    with tempfile.TemporaryDirectory(prefix="emraw_nexrad_") as tmpdir:
        temp_path = Path(tmpdir) / temp_name
        temp_path.write_bytes(file_bytes)
        try:
            dtree = opener(str(temp_path))
            return _extract_xradar_tree_sweeps(dtree)
        except Exception as exc:
            raise RuntimeError(
                f"xradar NEXRAD reader failed for {Path(temp_name).name}: {exc}"
            ) from exc


def _extract_custom_sweeps(file_bytes: bytes, filename: str) -> Tuple[Dict[str, List[dict]], dict]:
    readers: List[Tuple[str, Any]] = []
    errors: List[str] = []
    added: Set[str] = set()

    def _add_reader(label: str, func) -> None:
        if label in added:
            return
        readers.append((label, func))
        added.add(label)

    is_iris   = _looks_like_iris_sigmet_filename(filename)
    is_furuno = _looks_like_furuno_filename(filename)
    is_bufr   = _looks_like_bufr_payload(filename, file_bytes)
    is_msg31  = _looks_like_msg31_payload(file_bytes) or _looks_like_msg31_filename(filename)
    known_native_suffixes = {
        ".h5",
        ".hdf5",
        ".hdf",
        ".nc",
        ".nc4",
        ".cdf",
        ".cf",
    }

    # MSG31 (NEXRAD Level-2) – try native parser first, then xradar as fallback.
    # Placed before HDF5/NetCDF readers so we don't waste time on archive bytes.
    if is_msg31:
        _add_reader("MSG31(native)", lambda: _extract_msg31_sweeps(file_bytes))
        _add_reader("MSG31(xradar)", lambda: _extract_xradar_nexrad_sweeps(file_bytes, filename))

    if is_iris:
        _add_reader("Iris/Sigmet", lambda: _extract_xradar_sweeps(file_bytes, filename, "iris"))
    if is_furuno:
        _add_reader("Furuno", lambda: _extract_xradar_sweeps(file_bytes, filename, "furuno"))
    if is_bufr:
        _add_reader("BUFR", lambda: _extract_bufr_sweeps(file_bytes, filename))

    _add_reader("HDF5(ODIM)", lambda: _extract_all_sweeps(file_bytes))
    _add_reader("HDF5(KNMI)", lambda: _extract_knmi_sweeps(file_bytes))
    if HAS_NETCDF4:
        _add_reader("NetCDF(CF/Radial)", lambda: _extract_cabauw_netcdf_sweeps(file_bytes))

    # Last-resort fallbacks for unfamiliar raw vendor suffixes.
    if not (is_iris or is_furuno or is_bufr or is_msg31) and _custom_primary_suffix(filename) not in known_native_suffixes:
        _add_reader("Iris/Sigmet", lambda: _extract_xradar_sweeps(file_bytes, filename, "iris"))
        _add_reader("Furuno", lambda: _extract_xradar_sweeps(file_bytes, filename, "furuno"))

    for label, reader in readers:
        try:
            sweeps_by_qty, meta = reader()
            if sweeps_by_qty:
                return sweeps_by_qty, meta
            errors.append(f"{label}: no readable sweeps")
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    raise RuntimeError("; ".join(errors) if errors else "No readable sweeps found")


def _chmi_find_latest_asset(site_code: str, var_dir: str) -> Tuple[str, str]:
    """
    Find the latest HDF5 volume file for a CHMI radar site and variable.

    CHMI directory structure:
      {CHMI_RADAR_BASE}/{site}/{var_dir}/
      └── hdf5/              (may be nested under a date sub-dir or directly here)
            └── *.hdf5

    Walks up to 3 levels deep to locate HDF5 files, picks the most recent
    by the 12-digit timestamp embedded in the filename.
    """
    from urllib.error import HTTPError, URLError

    base_url = f"{CHMI_RADAR_BASE}/{site_code}/{var_dir}/"

    def _list_safe(url: str) -> Optional[str]:
        if not url.endswith("/"):
            url += "/"
        try:
            with urlopen(url, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, OSError):
            return None

    def _chmi_ts(name: str) -> str:
        m = re.search(r"(\d{12,14})", name.split("/")[-1])
        return m.group(1)[:12] if m else "000000000000"

    def _find_in_html(html: str, parent_url: str) -> List[str]:
        found = []
        for m in CHMI_HDF5_RE.finditer(html):
            href = m.group(1) or ""
            if not href or href.startswith("?") or href in ("../", "./"):
                continue
            full = urljoin(parent_url, href)
            found.append(full)
        return found

    def _walk(url: str, depth: int = 0) -> List[str]:
        if depth > 4:
            return []
        html = _list_safe(url)
        if not html:
            return []
        files = _find_in_html(html, url)
        if files:
            return files
        # Descend into sub-directories (look for /hdf5/ or date dirs)
        dirs: List[str] = []
        for m in DWD_HREF_RE.finditer(html):
            href = m.group(1).strip()
            if href.endswith("/") and href not in ("../", "./") and not href.startswith("?"):
                dirs.append(urljoin(url, href))
        # Prefer a subdir called "hdf5" if present
        hdf5_dirs = [d for d in dirs if "hdf5" in d.lower()]
        ordered = hdf5_dirs + [d for d in dirs if d not in hdf5_dirs]
        for d in sorted(ordered, reverse=True)[:5]:
            result = _walk(d, depth + 1)
            if result:
                return result
        return []

    all_files = _walk(base_url)
    if not all_files:
        raise RuntimeError(
            f"No CHMI HDF5 files found for site {site_code} var {var_dir}"
        )

    all_files.sort(key=lambda u: _chmi_ts(u), reverse=True)
    latest_url = all_files[0]

    ts = _chmi_ts(latest_url)
    try:
        dt_val = datetime.strptime(ts, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        dt_str = dt_val.isoformat().replace("+00:00", "Z")
    except Exception:
        dt_str = ts

    return latest_url, dt_str


def _shmu_list_dir(url: str) -> List[str]:
    """List directory entries (files and subdirs) for SHMU-style HTTP listings."""
    try:
        if not url.endswith("/"):
            url += "/"
        html = _shmu_read_url(url, timeout=30).decode("utf-8", errors="ignore")
        return [
            href.strip()
            for href in DWD_HREF_RE.findall(html)
            if href and href not in ("../", "./") and not href.startswith("?")
        ]
    except Exception:
        # Fallback: best-effort html scan (SHMU listings behave like apache autoindex)
        if not url.endswith("/"):
            url += "/"
        html = _shmu_read_url(url, timeout=30).decode("utf-8", errors="ignore")
        return [
            href.strip()
            for href in DWD_HREF_RE.findall(html)
            if href and href not in ("../", "./") and not href.startswith("?")
        ]


def _shmu_ts(name: str) -> str:
    m = SHMU_TS_RE.search(name)
    return m.group(1)[:14] if m else "00000000000000"


def _shmu_find_latest_asset(radar_code: str, product_dir: str) -> Tuple[str, str]:
    """
    Find the latest SHMU HDF5 volume file URL for a radar/product directory.

    SHMU structure (as described by user):
      /meteorology/weather/radar/volume/[radar]/[product]/[datetime]
    """
    r = str(radar_code or "").strip().lower()
    p = str(product_dir or "").strip().strip("/")
    if not r or not p:
        raise RuntimeError("Missing SHMU radar/product")

    root = f"{SHMU_RADAR_BASE}/{r}/{p}/"
    entries = _shmu_list_dir(root)

    # Accept both directories (datetime/) and files (datetime.h5)
    candidates = [e for e in entries if _shmu_ts(e) != "00000000000000"]
    if not candidates:
        raise RuntimeError(f"No SHMU volumes found at {root}")

    candidates.sort(key=_shmu_ts, reverse=True)
    latest = candidates[0]

    # If the latest entry is a directory, descend and pick the first HDF5-ish file.
    if latest.endswith("/"):
        droot = urljoin(root, latest)
        dentries = _shmu_list_dir(droot)
        files = [
            e
            for e in dentries
            if (not e.endswith("/"))
            and re.search(r"\.(h5|hdf5|hd5)(?:\.(gz|bz2))?$", e, re.IGNORECASE)
        ]
        if not files:
            # Some listings might expose a single file without extension; take the newest timestamped entry.
            files = [e for e in dentries if not e.endswith("/")]
        if not files:
            raise RuntimeError(f"No SHMU HDF5 files found under {droot}")
        files.sort(key=_shmu_ts, reverse=True)
        latest_url = urljoin(droot, files[0])
        dt_token = _shmu_ts(files[0])
    else:
        latest_url = urljoin(root, latest)
        dt_token = _shmu_ts(latest)

    # Convert token to RFC3339 when possible
    dt_str = dt_token
    try:
        if len(dt_token) >= 12:
            dt_val = datetime.strptime(dt_token[:12], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            dt_str = dt_val.isoformat().replace("+00:00", "Z")
    except Exception:
        pass
    return latest_url, dt_str


def _shmu_discover_products(radar_code: str) -> List[str]:
    """Return a list of product directory names for a SHMU radar."""
    r = str(radar_code or "").strip().lower()
    if not r:
        return []
    root = f"{SHMU_RADAR_BASE}/{r}/"
    entries = _shmu_list_dir(root)
    prods = [e.strip("/").strip() for e in entries if e.endswith("/") and e not in ("../", "./")]
    # Keep stable ordering
    return sorted({p for p in prods if p})


def _pick_shmu_product_dir(all_dirs: List[str], moment: str) -> Optional[str]:
    """
    Heuristic mapping from base moment -> SHMU product directory name.
    Prefers common naming patterns but falls back to substring matches.
    """
    dirs = [d.lower() for d in all_dirs]
    # Common-ish tokens
    want = {
        "reflectivity": ("z", "dbz", "refl", "reflect"),
        "velocity": ("v", "vel", "vrad"),
        "correlation": ("rho", "cc", "corr", "rhohv"),
        "spectrum_width": ("w", "width", "sw"),
        "zdr": ("zdr",),
    }.get(moment, ())
    if not want:
        return None

    # Prefer exact-ish matches
    for tok in want:
        for d in all_dirs:
            dl = d.lower()
            if dl == tok or dl.endswith(f"_{tok}") or dl.startswith(f"{tok}_"):
                return d
    # Substring fallback
    for tok in want:
        for d in all_dirs:
            if tok in d.lower():
                return d
    # Last-resort: short token matches
    if moment == "reflectivity":
        for d in all_dirs:
            if re.search(r"(?:^|_)(z|zh)(?:$|_)", d.lower()):
                return d
    if moment == "velocity":
        for d in all_dirs:
            if re.search(r"(?:^|_)(v|vr)(?:$|_)", d.lower()):
                return d
    return None


def _romania_ts(name: str) -> str:
    m = ROMANIA_TS_RE.search(name.split("/")[-1])
    token = m.group(1) if m else ""
    return token[:14] if token else "00000000000000"


def _romania_list_assets(site_code: str) -> List[str]:
    site = str(site_code or "").strip().upper()
    if not site:
        return []
    root = f"{ROMANIA_RADAR_BASE}/{site}/"
    try:
        html = _bytes_get(root, {}).decode("utf-8", errors="ignore")
    except Exception:
        return []

    out: List[str] = []
    for m in ROMANIA_HDF_RE.finditer(html):
        href = (m.group(1) or "").strip()
        if not href or href in ("../", "./") or href.startswith("?") or href.endswith("/"):
            continue
        out.append(urljoin(root, href))

    out = sorted(set(out))
    out.sort(key=_romania_ts, reverse=True)
    return out


def _romania_has_moment_marker(filename: str, markers: Tuple[str, ...]) -> bool:
    name_up = str(filename or "").upper()
    for marker in markers:
        mk = marker.upper()
        if len(mk) == 1:
            # Use a negative lookbehind so the marker is not immediately
            # preceded by a letter.  This lets "V" match "...0200V.hdf"
            # (digit before V) while correctly rejecting "RhoHV.hdf"
            # (letter H before V).
            if re.search(rf"(?<![A-Za-z]){re.escape(mk)}(?:[._-]|$)", name_up):
                return True
            continue
        if mk in name_up:
            return True
    return False


def _romania_find_latest_asset(site_code: str, moment_key: str) -> Tuple[str, str]:
    markers = ROMANIA_MOMENT_MARKERS.get(moment_key) or ()
    if not markers:
        raise RuntimeError(f"Unsupported Romanian moment key: {moment_key}")

    assets = _romania_list_assets(site_code)
    if not assets:
        raise RuntimeError(f"No Romanian HDF assets found for site {site_code}")

    filtered = [
        u for u in assets
        if _romania_has_moment_marker(u.split("/")[-1], markers)
    ]
    if not filtered:
        raise RuntimeError(
            f"No Romanian asset matched '{moment_key}' for site {site_code}"
        )

    latest_url = sorted(filtered, key=_romania_ts, reverse=True)[0]
    token = _romania_ts(latest_url)
    try:
        dt = datetime.strptime(token, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        dt_str = dt.isoformat().replace("+00:00", "Z")
    except Exception:
        dt_str = token if token != "00000000000000" else "n/a"
    return latest_url, dt_str


def _arpa_piemonte_list_dir(site_dir: str) -> List[str]:
    """List all HDF5 file URLs from an ARPA Piemonte Apache directory listing.

    Site directories: bric → Bric della Croce (PAGZ41), sett → Settepani (PAGZ42).
    The listing is a plain Apache index; no auth required.
    """
    url = f"{ARPA_PIE_BASE}/{site_dir}/"
    try:
        html = _bytes_get(url, {}).decode("utf-8", errors="ignore")
    except Exception:
        return []
    out: List[str] = []
    for m in ARPA_PIE_HDF5_RE.finditer(html):
        href = (m.group(1) or "").strip()
        if not href or href.startswith("?") or href in ("../", "./") or href.endswith("/"):
            continue
        out.append(urljoin(url, href))
    return sorted(set(out))


def _arpa_piemonte_ts(url_or_name: str) -> str:
    """Extract 14-digit timestamp (YYYYMMDDHHMMSS) from an ARPA Piemonte filename."""
    m = ARPA_PIE_TS_RE.search(url_or_name.split("/")[-1])
    return m.group(1) if m else "00000000000000"


def _arpa_piemonte_find_asset(site_dir: str, moment_suffix: str) -> Tuple[str, str]:
    """Return (url, dt_rfc3339) for the latest ARPA Piemonte file matching *moment_suffix*.

    ARPA Piemonte filename pattern:
        PAGZ41_C_PIEM_20260422162002dBZ.h5
    The moment suffix immediately follows the 14-digit timestamp and precedes '.h5'.
    Files appear every ~5 minutes; we sort by embedded timestamp descending.
    """
    all_files = _arpa_piemonte_list_dir(site_dir)
    if not all_files:
        raise RuntimeError(f"No ARPA Piemonte files listed at {site_dir}/")

    # Match files whose basename ends with exactly {moment_suffix}.h5, preceded by a
    # digit (the tail of the 14-digit timestamp).  This prevents 'Vu' from also
    # matching 'RhoHVu.h5', 'dBuZ' from matching 'Z.h5', etc.
    matching = [
        f for f in all_files
        if re.search(r"\d" + re.escape(moment_suffix) + r"\.h5$", f.split("/")[-1], re.IGNORECASE)
    ]
    if not matching:
        raise RuntimeError(
            f"No ARPA Piemonte {moment_suffix!r} files at {site_dir}/"
        )

    matching.sort(key=_arpa_piemonte_ts, reverse=True)
    latest_url = matching[0]
    ts = _arpa_piemonte_ts(latest_url)
    try:
        dt = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        dt_str = dt.isoformat().replace("+00:00", "Z")
    except Exception:
        dt_str = ts
    return latest_url, dt_str


def _arpa_lombardia_list_dir(site_code: str) -> List[str]:
    """List all HDF5/HDF5-gz file URLs from an ARPA Lombardia Apache directory listing.

    site_code: 'DES' (Desio, near Milan) or 'FLE' (Flero, near Brescia).
    Files are at ARPA_LOM_BASE/{site_code}/, e.g.:
        https://radarlive.arpalombardia.it/Volumi/DES/Desio.20250929T143000Z_DBZH.h5.gz
    The listing is a plain Apache index; no auth required.
    """
    url = f"{ARPA_LOM_BASE}/{site_code}/"
    try:
        req = Request(url, headers={"User-Agent": "radr/1.0"})
        try:
            with urlopen(req, timeout=30) as r:
                html = r.read().decode("utf-8", errors="ignore")
        except ssl.SSLCertVerificationError:
            ctx = ssl._create_unverified_context()
            with urlopen(req, timeout=30, context=ctx) as r:
                html = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    out: List[str] = []
    for m in ARPA_LOM_HDF5_RE.finditer(html):
        href = (m.group(1) or "").strip()
        if not href or href.startswith("?") or href in ("../", "./") or href.endswith("/"):
            continue
        out.append(urljoin(url, href))
    return sorted(set(out))


def _arpa_lombardia_ts(url_or_name: str) -> str:
    """Extract ISO-8601 timestamp token (YYYYMMDDTHHMMSSz) from an ARPA Lombardia filename.

    New convention: Desio.20250929T143000Z_DBZH.h5.gz → '20250929T143000Z'
    Returns '00000000T000000Z' if not found (sorts as oldest).
    """
    m = ARPA_LOM_TS_RE.search(url_or_name.split("/")[-1])
    return m.group(1) if m else "00000000T000000Z"


def _arpa_lombardia_find_asset(site_code: str, moment_suffix: str) -> Tuple[str, str]:
    """Return (url, dt_rfc3339) for the latest ARPA Lombardia file matching *moment_suffix*.

    Filename convention:
        {StationName}.{YYYYMMDD}T{HHMMSS}Z_{MOMENT}.h5.gz
        e.g. Desio.20250929T143000Z_DBZH.h5.gz  (in the DES/ directory)
             Flero.20250929T143000Z_DBZH.h5.gz  (in the FLE/ directory)

    The moment token follows an underscore after the ISO timestamp and precedes '.h5.gz'.
    Files appear every ~5 minutes; sorted by embedded timestamp descending.
    """
    all_files = _arpa_lombardia_list_dir(site_code)
    if not all_files:
        raise RuntimeError(f"No ARPA Lombardia files listed at {ARPA_LOM_BASE}/{site_code}/")

    # Match files whose basename ends with _{moment_suffix}.h5.gz (case-insensitive).
    # Using a word-boundary-style check so e.g. 'DBZH' does not match 'DBZH_corr'.
    matching = [
        f for f in all_files
        if re.search(
            r"_" + re.escape(moment_suffix) + r"\.h5(?:\.gz)?$",
            f.split("/")[-1],
            re.IGNORECASE,
        )
    ]
    if not matching:
        raise RuntimeError(
            f"No ARPA Lombardia {moment_suffix!r} files at {ARPA_LOM_BASE}/{site_code}/"
        )

    matching.sort(key=_arpa_lombardia_ts, reverse=True)
    latest_url = matching[0]
    ts = _arpa_lombardia_ts(latest_url)
    try:
        # Parse ISO token: YYYYMMDDTHHMMSSz  e.g. '20250929T143000Z'
        dt = datetime.strptime(ts, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        dt_str = dt.isoformat().replace("+00:00", "Z")
    except Exception:
        dt_str = ts
    return latest_url, dt_str


def _estonia_fetch_file_by_item_id(item_id: int) -> Tuple[bytes, str, str]:
    iid = int(item_id)
    url = f"{ESTONIA_KAIA_BASE}/{iid}/files/0"
    req = Request(
        url,
        headers={"Accept": "application/octet-stream, */*"},
    )
    with urlopen(req, timeout=180) as response:
        raw = response.read()
        cd = str(response.headers.get("Content-Disposition") or "")

    if not raw:
        raise RuntimeError(f"Estonia KAIA returned no content for item {iid}")

    fname = f"item_{iid}.h5"
    m_name = re.search(
        r"filename\*?=(?:UTF-8''|\"|')?([^\"';\r\n]+)",
        cd,
        re.IGNORECASE,
    )
    if m_name:
        fname = m_name.group(1).strip().strip("\"'")

    dt_str = "n/a"
    m_dt = re.search(r"\.(20\d{10,14})\.VOL\.h5$", fname, re.IGNORECASE)
    if m_dt:
        tok = m_dt.group(1)[:12]
        try:
            dt = datetime.strptime(tok, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            dt_str = dt.isoformat().replace("+00:00", "Z")
        except Exception:
            dt_str = tok
    return raw, dt_str, fname


def _meteogate_test_fetch_raw(wigos_id: str) -> Tuple[bytes, str]:
    wid = str(wigos_id or "").strip()
    if not wid:
        raise RuntimeError("Missing Meteogate test WIGOS station identifier")
    url = f"{METEOGATE_TEST_LOCATION_URL}/{wid}"
    params = dict(METEOGATE_TEST_DEFAULT_PARAMS)
    raw = _bytes_get(url, params)
    if not raw:
        raise RuntimeError(
            f"Meteogate test endpoint returned no content for {wid} "
            f"({params.get('datetime')})"
        )
    # We return the requested interval start as a display token.
    dt_str = str(params.get("datetime") or "").split("/", 1)[0] or "n/a"
    return raw, dt_str


def _velocity_polar_nrays_nbins(sw: dict) -> Tuple["np.ndarray", int, int, bool]:
    """
    Normalise sweep polar velocity to shape (nrays, nbins).

    Some readers occasionally produce transposed (nbins, nrays) arrays; UNRAVEL
    requires velocity shaped (len(azimuth), len(range)) == (nrays, nbins).

    Returns (polar_work, nrays, nbins, transpose_back_to_storage).
    """
    polar = sw.get("_polar")
    if polar is None:
        raise ValueError("missing _polar")
    arr = np.asarray(polar, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("polar not 2-D")
    nr = int(sw.get("nrays") or arr.shape[0])
    nb = int(sw.get("nbins") or arr.shape[1])
    if arr.shape == (nr, nb):
        return arr, nr, nb, False
    if arr.shape == (nb, nr):
        return arr.T.copy(), nr, nb, True
    return arr.copy(), arr.shape[0], arr.shape[1], False


def _unravel_dealias_polar_slice(
    sw: dict, polar_work: "np.ndarray", nyquist_m_s: float
) -> Optional["np.ndarray"]:
    """
    Run UNRAVEL (Louf et al., 2020) on a canonical (nrays × nbins) velocity slice.

    Tries ``dealiasing_process_2D`` first, then ``dealias_long_range`` if the
    primary pipeline raises.
    """
    if not HAS_UNRAVEL:
        return None
    nyq = float(abs(float(nyquist_m_s)))
    if not np.isfinite(nyq) or nyq <= 0:
        return None

    nr, nb = polar_work.shape
    az = np.linspace(0.0, 360.0, nr, endpoint=False, dtype=np.float32)
    rstart_km = float(sw.get("rstart_km", 0.0))
    rscale_km = float(sw.get("rscale_km", 1.0))
    r_m = (rstart_km + (np.arange(nb, dtype=np.float32) + 0.5) * rscale_km) * 1000.0
    elev = float(sw.get("elevation", 0.5))

    mask_missing = ~np.isfinite(polar_work)
    vel64 = np.asarray(polar_work, dtype=np.float64)

    try:
        deal, _flag = unravel.dealiasing_process_2D(
            r=r_m,
            azimuth=az,
            elevation=elev,
            velocity=vel64,
            nyquist_velocity=nyq,
            alpha=0.6,
            debug=False,
        )
    except Exception as exc_primary:
        dl_fn = getattr(unravel, "dealias_long_range", None)
        if dl_fn is None:
            print(f"[dealias] UNRAVEL dealiasing_process_2D failed: {exc_primary}")
            return None
        try:
            deal, _flag = dl_fn(
                r=r_m,
                azimuth=az,
                elevation=elev,
                velocity=vel64,
                nyquist_velocity=nyq,
                alpha=0.6,
                debug=False,
            )
        except Exception as exc:
            print(f"[dealias] UNRAVEL long-range fallback failed: {exc}")
            return None

    out = np.asarray(deal, dtype=np.float32)
    out[mask_missing] = np.nan
    return out


def _apply_velocity_dealias(sweeps_by_qty: Dict[str, List[dict]]) -> bool:
    """
    Dealias velocity-like quantities in-place when the user enables dealiasing.

    Strategy:
      1) NLradar dual-PRF correction when low/high Nyquist metadata indicates
         dual-PRF sampling (often the best match for European operational volumes).
      2) Else UNRAVEL 2-D when the unravel package is available and Nyquist is known.

    Returns True when at least one sweep was modified.
    """
    changed = False
    vel_qtys = ("VRADC", "VRADH", "VRAD", "VR", "V", "VRAD_corr")
    for vq in vel_qtys:
        sweeps = sweeps_by_qty.get(vq)
        if not sweeps:
            continue
        for sw in sweeps:
            try:
                polar_work, _nr, _nb, transpose_back = _velocity_polar_nrays_nbins(sw)
            except Exception:
                continue

            out: Optional["np.ndarray"] = None
            ni = sw.get("nyquist_m_s")
            lni = sw.get("low_nyquist_m_s")

            try:
                if ni is not None and lni is not None and float(lni) < float(ni):
                    out = _dealias_velocity_nlr(
                        polar_work,
                        nyquist=float(ni),
                        low_nyquist=float(lni),
                    )
            except Exception as exc:
                print(f"[dealias] NLradar dual-PRF failed: {exc}")
                out = None

            if out is None and ni is not None:
                out = _unravel_dealias_polar_slice(sw, polar_work, float(ni))

            if out is not None:
                if transpose_back:
                    out = np.asarray(out, dtype=np.float32).T
                sw["_polar"] = out
                changed = True
        break
    return changed


def _cabauw_fetch_latest_netcdf() -> Tuple[bytes, str, str]:
    """Fetch latest Cabauw (CESAR IDRA) NetCDF file via KNMI Open Data."""
    listing = _knmi_json_get(
        f"/datasets/{CABAUW_DATASET}/versions/{CABAUW_VERSION}/files",
        {"maxKeys": 1, "orderBy": "created", "sorting": "desc"},
    )
    files = listing.get("files") or []
    if not files:
        raise RuntimeError(f"No Cabauw files found for dataset {CABAUW_DATASET}/{CABAUW_VERSION}")
    fname = str(files[0].get("filename") or "").strip()
    if not fname:
        raise RuntimeError("Cabauw listing missing filename")
    url_info = _knmi_json_get(
        f"/datasets/{CABAUW_DATASET}/versions/{CABAUW_VERSION}/files/{fname}/url",
        {},
    )
    download_url = str(url_info.get("temporaryDownloadUrl") or "").strip()
    if not download_url:
        raise RuntimeError(f"Cabauw did not return download URL for {fname}")
    file_bytes = _decompress_if_needed(_bytes_get(download_url, {}))
    dt_str = _knmi_dt_from_filename(fname) or ""
    return file_bytes, dt_str, fname


def _extract_cabauw_netcdf_sweeps(file_bytes: bytes) -> Tuple[Dict[str, List[dict]], dict]:
    """
    Best-effort extraction for Cabauw IDRA NetCDF to a single 'DBZH' sweep.
    """
    if not HAS_NETCDF4:
        raise RuntimeError(f"Missing netCDF4 dependency: {NETCDF4_IMPORT_ERROR}")

    sweeps_by_qty: Dict[str, List[dict]] = {}
    meta: dict = {"lat": None, "lon": None, "max_range_km": None}

    ds = None
    tmp_path: Optional[str] = None

    def _to_scalar(val):
        if isinstance(val, bytes):
            try:
                return val.decode("utf-8", errors="ignore")
            except Exception:
                return str(val)
        try:
            arr = np.asarray(val)
            if arr.size == 1:
                return arr.flat[0].item() if hasattr(arr.flat[0], "item") else arr.flat[0]
        except Exception:
            pass
        return val

    def _safe_attr(obj, key: str):
        try:
            if hasattr(obj, "ncattrs") and key in obj.ncattrs():
                return _to_scalar(obj.getncattr(key))
        except Exception:
            pass
        try:
            return _to_scalar(getattr(obj, key))
        except Exception:
            return None

    def _attr_text(obj, *keys: str) -> str:
        for k in keys:
            v = _safe_attr(obj, k)
            if v is None:
                continue
            try:
                s = str(v).strip()
                if s:
                    return s
            except Exception:
                continue
        return ""

    def _attr_float(obj, *keys: str) -> Optional[float]:
        for k in keys:
            v = _safe_attr(obj, k)
            if v is None:
                continue
            try:
                f = float(v)
                if np.isfinite(f):
                    return f
            except Exception:
                continue
        return None

    try:
        try:
            ds = netCDF4.Dataset("inmemory_cabauw.nc", mode="r", memory=file_bytes)
        except Exception:
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            ds = netCDF4.Dataset(tmp_path, mode="r")

        # Lat/lon (fallback to station defaults later)
        meta["lat"] = _attr_float(ds, "latitude", "lat", "radar_latitude")
        meta["lon"] = _attr_float(ds, "longitude", "lon", "radar_longitude")

        # Find candidate reflectivity-like 2D variable
        chosen = None
        for var_name, var in ds.variables.items():
            if getattr(var, "ndim", 0) != 2:
                continue
            standard_name = _attr_text(var, "standard_name", "quantity")
            long_name = _attr_text(var, "long_name", "description")
            units = _attr_text(var, "units", "unit")
            qty = _fr_netcdf_quantity(var_name, standard_name, long_name, units)
            if qty == "DBZH":
                chosen = (var_name, var)
                break
        if chosen is None:
            raise RuntimeError("Cabauw NetCDF: no reflectivity-like 2D field found")
        var_name, var = chosen

        raw = var[:]
        if np.ma.isMaskedArray(raw):
            arr = np.asarray(raw.filled(np.nan), dtype=np.float32)
        else:
            arr = np.asarray(raw, dtype=np.float32)

        # Try to detect which axis is azimuth vs range
        dims = [str(d).lower() for d in getattr(var, "dimensions", ())]
        az = None
        rg = None
        for cand in ("azimuth", "azi", "ray_azimuth", "bearing"):
            if cand in ds.variables:
                try:
                    a = np.asarray(ds.variables[cand][:], dtype=np.float32).ravel()
                    if a.size:
                        az = a
                        break
                except Exception:
                    pass
        for cand in ("range", "gate", "r", "radial_range", "distance"):
            if cand in ds.variables:
                try:
                    a = np.asarray(ds.variables[cand][:], dtype=np.float32).ravel()
                    if a.size:
                        rg = a
                        break
                except Exception:
                    pass

        ray_axis = None
        bin_axis = None
        for i, dname in enumerate(dims):
            if ray_axis is None and "azi" in dname:
                ray_axis = i
            if bin_axis is None and "range" in dname:
                bin_axis = i

        if ray_axis is None and az is not None:
            if arr.shape[0] == az.size:
                ray_axis = 0
            elif arr.shape[1] == az.size:
                ray_axis = 1
        if bin_axis is None and rg is not None:
            if arr.shape[1] == rg.size:
                bin_axis = 1
            elif arr.shape[0] == rg.size:
                bin_axis = 0

        if ray_axis is None or bin_axis is None or ray_axis == bin_axis:
            # fallback: assume (ray, range)
            ray_axis, bin_axis = (0, 1) if arr.shape[0] >= arr.shape[1] else (1, 0)

        arr = np.moveaxis(arr, (ray_axis, bin_axis), (0, 1))
        nrays, nbins = arr.shape

        # Range scaling (meters->km if needed)
        rstart_km = 0.0
        rscale_km = 1.0
        if rg is not None and rg.size >= 2:
            step = float(np.nanmedian(np.abs(np.diff(rg[: min(rg.size, nbins)]))))
            first = float(rg.flat[0])
            if abs(step) > 20 or abs(first) > 20:
                rscale_km = step / 1000.0
                rstart_km = first / 1000.0
            else:
                rscale_km = step
                rstart_km = first
        else:
            gate_step_m = _attr_float(var, "meters_between_gates", "gate_size", "range_resolution")
            first_gate_m = _attr_float(var, "meters_to_center_of_first_gate", "range_first_gate")
            if gate_step_m is not None:
                rscale_km = float(gate_step_m) / 1000.0 if abs(float(gate_step_m)) > 20 else float(gate_step_m)
            if first_gate_m is not None:
                rstart_km = float(first_gate_m) / 1000.0 if abs(float(first_gate_m)) > 20 else float(first_gate_m)

        if not np.isfinite(rscale_km) or rscale_km <= 0:
            rscale_km = 1.0
        if not np.isfinite(rstart_km):
            rstart_km = 0.0
        max_rng = rstart_km + nbins * rscale_km
        meta["max_range_km"] = float(max_rng)

        sweeps_by_qty["DBZH"] = [
            {
                "elevation": float(_attr_float(var, "elevation", "fixed_angle") or 0.5),
                "rscale_km": float(rscale_km),
                "rstart_km": float(rstart_km),
                "max_range_km": float(max_rng),
                "nrays": int(nrays),
                "nbins": int(nbins),
                "_polar": arr,
                "nyquist_m_s": None,
                "low_nyquist_m_s": None,
            }
        ]
        return sweeps_by_qty, meta
    finally:
        try:
            if ds is not None:
                ds.close()
        except Exception:
            pass
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _geosphere_find_latest_asset(dataset_path: str) -> Tuple[str, str]:
    """
    Find the most recent HDF volume from a GeoSphere Austria listing.

    Primary source (current public setup):
      https://public.hub.geosphere.at/datahub/?list-type=2&prefix=resources/...
    Fallback source (legacy):
      https://public.hub.geosphere.at/resources/...

    Returns (file_url, datetime_rfc3339).
    """
    from urllib.error import HTTPError, URLError

    raw_path = (dataset_path or "").strip().strip("/")
    if not raw_path:
        raise RuntimeError("GeoSphere dataset path is empty")

    path_candidates: List[str] = []

    def _add_path(path_value: str) -> None:
        p = (path_value or "").strip().strip("/")
        if p and p not in path_candidates:
            path_candidates.append(p)

    _add_path(raw_path)
    if raw_path.endswith("filelisting"):
        parent = raw_path.rsplit("filelisting", 1)[0].rstrip("/")
        _add_path(parent)
    else:
        _add_path(f"{raw_path}/filelisting")

    def _ts_to_rfc3339(ts12: str) -> str:
        try:
            dt = datetime.strptime(ts12, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except Exception:
            return ts12

    errors: List[str] = []

    # 1) Current GeoSphere public S3-style listing endpoint
    for p in path_candidates:
        prefix = f"{GEOSPHERE_BUCKET_ROOT}{p}/"
        continuation_token: Optional[str] = None
        pages = 0
        best_key = ""
        best_ts = ""
        while True:
            pages += 1
            params = {
                "list-type": "2",
                "max-keys": "1000",
                "prefix": prefix,
            }
            if continuation_token:
                params["continuation-token"] = continuation_token

            listing_url = f"{GEOSPHERE_BUCKET_URL}/?{urlencode(params)}"
            req = Request(
                listing_url,
                headers={"Authorization": f"Bearer {GEOSPHERE_TOKEN}"},
            )
            try:
                with urlopen(req, timeout=60) as resp:
                    xml = resp.read().decode("utf-8", errors="ignore")
            except HTTPError as exc:
                errors.append(f"{listing_url} -> HTTP {exc.code}")
                break
            except (URLError, OSError) as exc:
                errors.append(f"{listing_url} -> {exc}")
                break

            matches = GEOSPHERE_S3_KEY_RE.findall(xml)
            for key, ts in matches:
                if ts > best_ts:
                    best_key = key
                    best_ts = ts

            m_next = GEOSPHERE_S3_NEXT_TOKEN_RE.search(xml)
            next_token = m_next.group(1).strip() if m_next else ""
            if not next_token:
                break
            if next_token == continuation_token:
                errors.append(f"{listing_url} -> repeated continuation token")
                break
            continuation_token = next_token
            if pages >= 50:
                errors.append(f"{listing_url} -> pagination limit reached")
                break

        if best_key and best_ts:
            file_url = f"{GEOSPHERE_BUCKET_URL}/{best_key.lstrip('/')}"
            return file_url, _ts_to_rfc3339(best_ts)

        errors.append(f"S3 listing prefix '{prefix}' -> no WXRHOF HDF keys")

    # 2) Legacy HTML listing fallback
    base = GEOSPHERE_LISTING_BASE.rstrip("/")
    url_candidates: List[str] = []

    def _add_url(url_value: str) -> None:
        if url_value and url_value not in url_candidates:
            url_candidates.append(url_value)

    for p in path_candidates:
        u = f"{base}/{p}"
        _add_url(u)
        _add_url(u + "/")

    for listing_url in url_candidates:
        req = Request(
            listing_url,
            headers={"Authorization": f"Bearer {GEOSPHERE_TOKEN}"},
        )
        try:
            with urlopen(req, timeout=60) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
        except HTTPError as exc:
            errors.append(f"{listing_url} -> HTTP {exc.code}")
            continue
        except (URLError, OSError) as exc:
            errors.append(f"{listing_url} -> {exc}")
            continue

        matches = GEOSPHERE_HDF_RE.findall(html)
        if not matches:
            errors.append(f"{listing_url} -> no WXRHOF HDF links")
            continue

        matches.sort(key=lambda m: m[1], reverse=True)
        latest_href, latest_ts = matches[0]
        if latest_href.startswith("http"):
            file_url = latest_href
        else:
            join_base = listing_url if listing_url.endswith("/") else f"{listing_url}/"
            file_url = urljoin(join_base, latest_href)
        return file_url, _ts_to_rfc3339(latest_ts)

    detail = "; ".join(errors[-4:]) if errors else "no candidate URLs generated"
    raise RuntimeError(f"GeoSphere listing failed for '{raw_path}': {detail}")


def _geosphere_bytes_get(url: str) -> bytes:
    """Fetch a GeoSphere file URL with Bearer-token auth."""
    req = Request(url, headers={"Authorization": f"Bearer {GEOSPHERE_TOKEN}"})
    with urlopen(req, timeout=120) as resp:
        return resp.read()

# Meteogate (EUMETNet) OGC API – MET Norway volume radar

def _meteogate_json_get(url: str, params: dict) -> dict:
    """Minimal OGC API JSON GET against the Meteogate sandbox."""
    qs = urlencode(params)
    full_url = f"{url}?{qs}" if qs else url
    req = Request(full_url, headers={"Accept": "application/geo+json,application/json"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _meteogate_discover_norway_stations() -> Dict[str, dict]:
    """
    Query Meteogate locations endpoint and return all MET Norway radar stations
    as a provider dict keyed by display name.

    Uses the Norway bounding box to filter; falls back to accepting any
    feature whose WIGOS ID starts with the Norwegian prefix ``0-20000-0-no``.
    """
    try:
        data = _meteogate_json_get(
            METEOGATE_LOCATIONS_URL,
            {
                "parameter-name": "DBZH:scan",
                "method": "scan",
                "bbox": _NORWAY_BBOX,
                "f": "json",
            },
        )
    except Exception as exc:
        # Try wider: fetch all locations and filter client-side
        try:
            data = _meteogate_json_get(
                METEOGATE_LOCATIONS_URL,
                {"parameter-name": "DBZH:scan", "method": "scan", "f": "json"},
            )
        except Exception:
            return {}

    stations: Dict[str, dict] = {}
    _NO_PREFIXES = ("0-20000-0-no", "0-578-")  # WMO block 01 = Norway; also 0-578-*

    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        try:
            lon, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            continue

        # Resolve WIGOS ID from multiple possible property names
        wigos_id = str(
            props.get("wigosStationIdentifier")
            or props.get("wigos_station_identifier")
            or props.get("stationId")
            or props.get("id")
            or feature.get("id")
            or ""
        ).strip()

        if not wigos_id:
            continue

        # If we got back global results (bbox ignored), filter to Norway only
        wid_lo = wigos_id.lower()
        in_norway_box = (4.0 <= lon <= 32.0) and (57.0 <= lat <= 72.0)
        is_no_wigos = any(wid_lo.startswith(pfx) for pfx in _NO_PREFIXES)
        if not in_norway_box and not is_no_wigos:
            continue

        # Build a human-readable display name
        raw_name = str(
            props.get("stationName")
            or props.get("station_name")
            or props.get("name")
            or props.get("shortName")
            or wigos_id
        ).strip()
        display_name = f"NO {raw_name}" if not raw_name.upper().startswith("NO") else raw_name

        # Deduplicate: if name already taken, append WIGOS suffix
        base_name = display_name
        suffix = 2
        while display_name in stations:
            display_name = f"{base_name} ({suffix})"
            suffix += 1

        stations[display_name] = {
            "provider": "meteogate",
            "wigos_id": wigos_id,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "range_km": 240.0,
        }

    return stations


def _meteogate_latest_scan_dt(wigos_id: str) -> Optional[str]:
    """
    Ask the Meteogate *items* endpoint for the most recent DBZH scan
    for the given WIGOS station and return an ISO 8601 datetime string,
    or None on failure.
    """
    try:
        data = _meteogate_json_get(
            METEOGATE_ITEMS_URL,
            {
                "stationId": wigos_id,
                "standard-name": "DBZH",
                "method": "scan",
                "limit": 1,
                "sortby": "-datetime",
                "f": "json",
            },
        )
        for feat in data.get("features", []):
            dt_raw = str(
                (feat.get("properties") or {}).get("datetime")
                or (feat.get("properties") or {}).get("resultTime")
                or (feat.get("properties") or {}).get("phenomenonTime")
                or ""
            ).strip()
            if dt_raw:
                return dt_raw
    except Exception:
        pass
    return None


def _meteogate_fetch_pvol(wigos_id: str) -> Tuple[bytes, str]:
    """
    Fetch the latest PVOL (ODIM HDF5) for a Meteogate-registered Norwegian
    radar.

    Strategy:
      1. Ask the items endpoint for the latest scan time for this station.
      2. Fall back to *now* rounded down to the nearest 5 minutes.
      3. Download the PVOL via the locations/{wigos_id} endpoint with
         format=ODIM.

    Returns (file_bytes, datetime_str_rfc3339).
    """
    # 1. Resolve nominal scan time
    dt_str = _meteogate_latest_scan_dt(wigos_id)

    if not dt_str:
        # Fall back: round current UTC time to nearest 5-minute boundary
        _now = datetime.now(timezone.utc)
        _mins = (_now.minute // 5) * 5
        _now = _now.replace(minute=_mins, second=0, microsecond=0)
        dt_str = _now.isoformat().replace("+00:00", "Z")

    # Normalise to the form expected by the API (no fractional seconds)
    try:
        _parsed = _parse_rfc3339(dt_str)
        dt_str = _parsed.strftime("%Y-%m-%dT%H:%MZ")
    except Exception:
        pass  # leave dt_str as-is

    # 2. Download PVOL 
    pvol_url = f"{METEOGATE_LOCATIONS_URL}/{wigos_id}"
    pvol_params = {
        "parameter-name": "DBZH:scan",
        "format": "ODIM",
        "datetime": dt_str,
        "f": "application/x-hdf5",
    }
    qs = urlencode(pvol_params)
    full_url = f"{pvol_url}?{qs}"
    req = Request(
        full_url,
        headers={"Accept": "application/x-hdf5, application/octet-stream, */*"},
    )
    with urlopen(req, timeout=120) as resp:
        raw_bytes = resp.read()

    return raw_bytes, dt_str


# Maps KNMI dataset base-names → ODIM quantity keys used by PRODUCTS / _match_quantity.
# KNMI naming is case-sensitive; include all observed variants.
_KNMI_QTY_MAP: Dict[str, str] = {
    # Horizontal reflectivity
    "scan_Z_data":      "DBZH",   "scan_Zh_data":    "DBZH",
    # Uncorrected horizontal reflectivity
    "scan_uZ_data":     "TH",     "scan_uZh_data":   "TH",
    # Vertical reflectivity
    "scan_Zv_data":     "DBZV",
    # Uncorrected vertical reflectivity
    "scan_uZv_data":    "TV",
    # Horizontal Doppler velocity
    "scan_V_data":      "VRADH",  "scan_Vh_data":    "VRADH",
    # Vertical Doppler velocity  (KNMI uses lowercase-v second letter)
    "scan_Vv_data":     "VRADV",
    # Unfiltered velocities
    "scan_uV_data":     "VRADH",  "scan_uVv_data":   "VRADV",
    # Spectrum width
    "scan_W_data":      "WRADH",  "scan_Wh_data":    "WRADH",
    "scan_Wv_data":     "WRADV",
    # ZDR – dual-pol
    "scan_ZDR_data":    "ZDR",
    # KDP
    "scan_KDP_data":    "KDP",
    # PhiDP – KNMI uses mixed-case "PhiDP"
    "scan_PhiDP_data":  "PHIDP",  "scan_PHIDP_data": "PHIDP",
    # Unfiltered PhiDP
    "scan_uPhiDP_data": "UPHIDP",
    # Cross-correlation (CCOR) – maps to CCORH in PRODUCTS correlation list
    "scan_CCOR_data":   "CCORH",  "scan_CCORh_data": "CCORH",
    "scan_CCORv_data":  "CCORV",
    # RhoHV – KNMI uses mixed-case "RhoHV"
    "scan_RhoHV_data":  "RHOHV",  "scan_RHOHV_data": "RHOHV",
    # Quality / ancillary
    "scan_CPA_data":    "CPA",
    "scan_CPAv_data":   "CPAv",
    "scan_SQI_data":    "SQI",
    "scan_SQIv_data":   "SQIv",
}

def _parse_knmi_calibration_formula(formula: Any) -> Optional[Tuple[float, float]]:
    """Parse KNMI formulas like ``GEO=0.00193793*PV+-31.5019``.

    KNMI's non-ODIM HDF5 volumes store the physical-value transform as a
    string formula rather than ODIM ``gain`` / ``offset`` attributes.  The
    formula maps packed pixel values (PV) to geophysical values (GEO).
    """
    if formula is None:
        return None
    try:
        text = formula.decode("utf-8", errors="ignore") if isinstance(formula, bytes) else str(formula)
    except Exception:
        return None
    text = text.strip().strip("\x00")
    if not text:
        return None
    if "=" in text:
        text = text.split("=", 1)[1]
    expr = re.sub(r"\s+", "", text)
    num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    scale = None
    offset = 0.0

    m = re.search(rf"(?P<scale>{num})\*PV(?P<offset>[+-].+)?$", expr)
    if m:
        scale = float(m.group("scale"))
        tail = m.group("offset")
        if tail:
            tail = tail.replace("+-", "-").replace("-+", "-").replace("++", "+").replace("--", "+")
            offset = float(tail)
    else:
        m = re.search(rf"PV\*(?P<scale>{num})(?P<offset>[+-].+)?$", expr)
        if m:
            scale = float(m.group("scale"))
            tail = m.group("offset")
            if tail:
                tail = tail.replace("+-", "-").replace("-+", "-").replace("++", "+").replace("--", "+")
                offset = float(tail)
        elif expr == "PV":
            scale = 1.0

    if scale is None or not math.isfinite(scale) or not math.isfinite(offset):
        return None
    return float(scale), float(offset)

def _extract_knmi_sweeps(file_bytes: bytes) -> Tuple[Dict[str, List[dict]], dict]:
    """Best-effort extraction for KNMI radar volumes (non-ODIM layout)."""
    sweeps_by_qty: Dict[str, List[dict]] = {}
    meta: dict = {"lat": None, "lon": None, "max_range_km": None}
    lowest_angle = 999.0

    def _safe_attr(obj, key: str):
        try:
            return obj.attrs.get(key)
        except Exception:
            return None

    def _safe_attr_float(obj, key: str) -> Optional[float]:
        try:
            v = obj.attrs.get(key)
            if v is None:
                return None
            if isinstance(v, (bytes, str)):
                v = v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else v
                return float(str(v).strip().strip("\x00"))
            arr = np.asarray(v)
            return float(arr.flat[0])
        except Exception:
            return None

    def _safe_attr_text(obj, key: str) -> Optional[str]:
        try:
            v = obj.attrs.get(key)
            if v is None:
                return None
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="ignore").strip().strip("\x00")
            if isinstance(v, str):
                return v.strip().strip("\x00")
            arr = np.asarray(v)
            if arr.size == 0:
                return None
            first = arr.flat[0]
            if isinstance(first, bytes):
                return first.decode("utf-8", errors="ignore").strip().strip("\x00")
            return str(first).strip().strip("\x00")
        except Exception:
            return None

    with h5py.File(io.BytesIO(file_bytes), "r") as hdf:
        # Lat/lon: try radar1 or geographic groups
        for grp_name in ("radar1", "geographic"):
            grp = hdf.get(grp_name)
            if grp is None:
                continue
            for k in ("lat", "lon", "latitude", "longitude", "radar_latitude", "radar_longitude"):
                if meta["lat"] is None and k in ("lat", "latitude", "radar_latitude"):
                    v = _safe_attr_float(grp, k)
                    if v is not None:
                        meta["lat"] = v
                if meta["lon"] is None and k in ("lon", "longitude", "radar_longitude"):
                    v = _safe_attr_float(grp, k)
                    if v is not None:
                        meta["lon"] = v
            if meta["lat"] is not None and meta["lon"] is not None:
                break

        scan_names = sorted([k for k in hdf.keys() if re.match(r"scan\d+$", str(k))])

        def _find_range_array(group) -> Optional[np.ndarray]:
            """Return a 1D range/gate array if present."""
            for name in group.keys():
                try:
                    obj = group[name]
                except Exception:
                    continue
                if not isinstance(obj, h5py.Dataset):
                    continue
                if obj.ndim != 1:
                    continue
                lname = name.lower()
                if "range" in lname or "gate" in lname or lname in ("r", "rg", "rng"):
                    try:
                        return np.asarray(obj[:], dtype=np.float32)
                    except Exception:
                        return None
            return None

        def _range_info_from_array(arr: np.ndarray) -> Tuple[Optional[float], float]:
            """Return (rscale_km, rstart_km) from a 1D range array."""
            rscale_km = None
            rstart_km = 0.0
            try:
                if arr.size >= 2:
                    step = float(arr[1] - arr[0])
                    if abs(step) > 0:
                        rscale_km = step / 1000.0 if abs(step) > 10 else step
                        rstart_km = float(arr[0] / 1000.0) if abs(arr[0]) > 10 else float(arr[0])
            except Exception:
                pass
            return rscale_km, rstart_km

        def _find_range_info(group, nbins: int, range_arr: Optional[np.ndarray]) -> Tuple[Optional[float], float]:
            """Return (rscale_km, rstart_km) from range arrays or attrs."""
            rscale_km = None
            rstart_km = 0.0
            if range_arr is not None and range_arr.size == nbins:
                rscale_km, rstart_km = _range_info_from_array(range_arr)
                if rscale_km is not None:
                    return rscale_km, rstart_km
            # Attr fallbacks
            for key in ("scan_range_bin", "range_step", "range_resolution", "gate_size", "rscale", "bin_size"):
                v = _safe_attr_float(group, key)
                if v is None:
                    continue
                rscale_km = v / 1000.0 if v > 10 else v
                return rscale_km, rstart_km
            return rscale_km, rstart_km

        def _find_azimuth_array(group, nrays: int) -> Optional[np.ndarray]:
            for name in group.keys():
                try:
                    obj = group[name]
                except Exception:
                    continue
                if not isinstance(obj, h5py.Dataset):
                    continue
                if obj.ndim != 1 or obj.shape[0] != nrays:
                    continue
                lname = name.lower()
                if "azimuth" in lname or lname in ("az", "azi"):
                    try:
                        return np.asarray(obj[:], dtype=np.float32)
                    except Exception:
                        return None
            return None

        for scan_name in scan_names:
            scan = hdf.get(scan_name)
            if scan is None:
                continue

            # Elevation angle
            elev = None
            for key in ("scan_elevation", "elevation", "elevation_angle", "elangle", "tilt", "elev"):
                v = _safe_attr_float(scan, key)
                if v is not None:
                    elev = v
                    break
            if elev is None:
                elev = 0.0

            # Find 2D datasets within scan group
            candidates: List[Tuple[str, h5py.Dataset]] = []

            def _visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                    candidates.append((name, obj))

            try:
                scan.visititems(_visit)
            except Exception:
                pass

            # Optional SQI mask (quality) for the scan
            sqi_arr = None
            for n, d in candidates:
                if "sqi" in n.lower():
                    try:
                        sqi_arr = np.asarray(d[:], dtype=np.float32)
                    except Exception:
                        sqi_arr = None
                    break

            for name, ds in candidates:
                try:
                    raw = ds[:]
                except Exception:
                    continue
                if raw is None or raw.ndim != 2:
                    continue
                nrays, nbins = raw.shape

                # Quantity name: try dataset attrs first, fall back to dataset name,
                # then normalise to ODIM via _KNMI_QTY_MAP.
                qty = None
                for key in ("quantity", "moment", "product", "type", "data_type"):
                    v = _safe_attr(ds, key)
                    if v is not None:
                        qty = str(v.decode() if isinstance(v, bytes) else v)
                        break
                raw_dsname = name.split("/")[-1]   # e.g. "scan_Z_data"
                if not qty:
                    qty = raw_dsname
                qty = _KNMI_QTY_MAP.get(qty, qty)
                qty_norm = qty.lower()

                # -Calibration
                # KNMI stores gain/offset/nodata/undetect in a sibling Group
                # called "calibration" (or "Calibration") whose ATTRIBUTES are
                # named calibration_<moment>_gain, calibration_<moment>_offset, etc.
                # The <moment> key is extracted from the raw dataset name.
                # If that group is absent or missing the attribute, fall back to
                # dataset-level attrs (scale_factor / add_offset / _FillValue) and
                # finally to sensible defaults.

                # Find calibration containers: sub-group/dataset OR direct scan attrs
                _cal_srcs = []
                for _cal_key in scan.keys():
                    if "calibration" in str(_cal_key).lower():
                        try:
                            _cal_srcs.append(scan[_cal_key])
                        except Exception:
                            pass

                # Derive the KNMI <moment> token from the raw dataset name.
                # e.g. "scan_Z_data" → "Z", "scan_PhiDP_data" → "PhiDP"
                _moment_m = re.match(r"scan_([A-Za-z0-9]+)_data$", raw_dsname)
                _mk = _moment_m.group(1) if _moment_m else None

                def _cal_attr(suffix: str) -> Optional[float]:
                    """Read calibration_<moment>_<suffix> from cal group or scan attrs."""
                    if _mk is None:
                        return None
                    attr_name = f"calibration_{_mk}_{suffix}"
                    # Try calibration subgroup/dataset first
                    for _cal_src in _cal_srcs:
                        v = _safe_attr_float(_cal_src, attr_name)
                        if v is not None:
                            return v
                    # Try directly on the scan group (some KNMI file variants)
                    v = _safe_attr_float(scan, attr_name)
                    return v

                def _cal_attr_text(suffix: str) -> Optional[str]:
                    """Read calibration_<moment>_<suffix> text from cal group or scan attrs."""
                    if _mk is None:
                        return None
                    attr_name = f"calibration_{_mk}_{suffix}"
                    for _cal_src in _cal_srcs:
                        v = _safe_attr_text(_cal_src, attr_name)
                        if v:
                            return v
                    return _safe_attr_text(scan, attr_name)

                parsed_formula = _parse_knmi_calibration_formula(
                    _cal_attr_text("formulas") or _cal_attr_text("formula")
                )
                if parsed_formula is not None:
                    gain, offset = parsed_formula
                else:
                    gain = _cal_attr("gain")
                    if gain is None:
                        gain = _cal_attr("scale")
                    if gain is None:
                        gain = _safe_attr_float(ds, "gain")
                    if gain is None:
                        gain = _safe_attr_float(ds, "scale_factor")
                    if gain is None:
                        gain = 1.0

                    offset = _cal_attr("offset")
                    if offset is None:
                        offset = _safe_attr_float(ds, "offset")
                    if offset is None:
                        offset = _safe_attr_float(ds, "add_offset")
                    if offset is None:
                        offset = 0.0

                # Nodata / undetect: raw uint8 values stored in calibration group
                _raw_nodata   = _safe_attr(ds, "nodata")   or _safe_attr(ds, "_FillValue")
                _raw_undetect = _safe_attr(ds, "undetect")
                if _raw_nodata is None:
                    _nd = _cal_attr("nodata")
                    if _nd is None:
                        _nd = _cal_attr("missing_data")
                    if _nd is None:
                        _nd = _safe_attr_float(scan, "calibration_missing_data")
                    if _nd is None:
                        for _cal_src in _cal_srcs:
                            _nd = _safe_attr_float(_cal_src, "calibration_missing_data")
                            if _nd is not None:
                                break
                    _raw_nodata = _nd
                if _raw_undetect is None:
                    _ud = _cal_attr("undetect")
                    _raw_undetect = _ud

                units = _safe_attr(ds, "units") or _safe_attr(ds, "unit")

                arr = raw.astype(np.float32) * float(gain) + float(offset)
                try:
                    if _raw_nodata is not None:
                        arr[raw == float(_raw_nodata)] = np.nan
                    if _raw_undetect is not None:
                        arr[raw == float(_raw_undetect)] = np.nan
                except Exception:
                    pass

                # Range + azimuth (and orientation)
                range_arr = _find_range_array(scan)
                az_any = _find_azimuth_array(scan, nrays)
                az = az_any
                transpose = False
                if az is not None and az.size == nbins:
                    transpose = True
                elif range_arr is not None and range_arr.size == nrays:
                    # Range array matches first axis => likely (range, azimuth)
                    transpose = True
                if transpose:
                    arr = arr.T
                    if sqi_arr is not None and sqi_arr.shape == raw.shape:
                        sqi_arr = sqi_arr.T
                nrays, nbins = arr.shape
                if range_arr is not None and range_arr.size == nrays:
                    # Range array was for pre-transpose axis; recompute after swap
                    range_arr = _find_range_array(scan)
                az = _find_azimuth_array(scan, nrays)
                rscale_km, rstart_km = _find_range_info(scan, nbins, range_arr)
                polar = arr
                if az is not None and az.size == nrays:
                    dist = np.minimum(az % 360.0, 360.0 - (az % 360.0))
                    north_idx = int(np.argmin(dist))
                    polar = np.roll(arr, -north_idx, axis=0)

                # SQI quality mask (suppress noise)
                if sqi_arr is not None and sqi_arr.shape == polar.shape:
                    try:
                        polar = np.where(sqi_arr < 0.3, np.nan, polar)
                    except Exception:
                        pass

                # Convert linear Z to dBZ if needed
                if qty_norm.endswith("_z_data") or qty_norm.endswith("_uz_data"):
                    u = str(units.decode() if isinstance(units, bytes) else units).lower() if units else ""
                    if "db" not in u:
                        try:
                            safe = np.where(polar > 0, polar, np.nan)
                            polar = 10.0 * np.log10(safe)
                        except Exception:
                            pass

                max_rng = None
                if rscale_km is not None:
                    max_rng = rstart_km + nbins * rscale_km

                if elev < lowest_angle and max_rng is not None:
                    lowest_angle = elev
                    meta["max_range_km"] = max_rng

                sweeps_by_qty.setdefault(qty, []).append({
                    "elevation": elev,
                    "rscale_km": rscale_km if rscale_km is not None else 1.0,
                    "rstart_km": rstart_km,
                    "max_range_km": max_rng if max_rng is not None else (rstart_km + nbins),
                    "nrays": nrays,
                    "nbins": nbins,
                    "_polar": polar,
                    "nyquist_m_s": None,
                    "low_nyquist_m_s": None,
                })

    for q in sweeps_by_qty:
        sweeps_by_qty[q].sort(key=lambda s: s["elevation"])
    return sweeps_by_qty, meta


def _fr_netcdf_quantity(var_name: str, standard_name: str, long_name: str, units: str) -> Optional[str]:
    text = " ".join([var_name, standard_name, long_name]).lower()
    u = units.lower()

    if "zdr" in text or "differential_reflectivity" in text:
        return "ZDR"
    if "kdp" in text or "specific_differential_phase" in text:
        return "KDP"
    if "phidp" in text or "differential_phase" in text:
        return "PHIDP"
    if "rhohv" in text or "cross_correlation" in text or "correlation" in text:
        return "RHOHV"
    if "spectrum_width" in text or ("width" in text and "velocity" in text):
        return "WRADH"
    if "velocity" in text or "vrad" in text or "doppler" in text:
        return "VRADH"
    if "reflectivity" in text or "dbz" in text or "equivalent_reflectivity_factor" in text:
        return "DBZH"
    if "m/s" in u and "velocity" in text:
        return "VRADH"
    if ("dbz" in u or "dBZ" in units) and "reflect" in text:
        return "DBZH"
    return None


# _extract_fr_netcdf_sweeps DISABLED — SEDOO/Meteo-France commented out
# def _extract_fr_netcdf_sweeps(file_bytes: bytes) -> Tuple[Dict[str, List[dict]], dict]:
#     """Extract sweeps from Meteo-France NetCDF4 CF/Radial files."""
#     if not HAS_NETCDF4:
#         raise RuntimeError(f"Missing netCDF4 dependency: {NETCDF4_IMPORT_ERROR}")
#
#     sweeps_by_qty: Dict[str, List[dict]] = {}
#     meta: dict = {"lat": None, "lon": None, "max_range_km": None}
#     lowest_angle = 999.0
#
#     ds = None
#     tmp_path: Optional[str] = None
#
#     def _to_scalar(val):
#         if isinstance(val, bytes):
#             try:
#                 return val.decode("utf-8", errors="ignore")
#             except Exception:
#                 return str(val)
#         try:
#             arr = np.asarray(val)
#             if arr.size == 1:
#                 return arr.flat[0].item() if hasattr(arr.flat[0], "item") else arr.flat[0]
#         except Exception:
#             pass
#         return val
#
#     def _safe_attr(obj, key: str):
#         try:
#             if hasattr(obj, "ncattrs") and key in obj.ncattrs():
#                 return _to_scalar(obj.getncattr(key))
#         except Exception:
#             pass
#         try:
#             return _to_scalar(getattr(obj, key))
#         except Exception:
#             return None
#
#     def _attr_float(obj, *keys: str) -> Optional[float]:
#         for key in keys:
#             v = _safe_attr(obj, key)
#             if v is None:
#                 continue
#             try:
#                 f = float(v)
#                 if np.isfinite(f):
#                     return f
#             except Exception:
#                 continue
#         return None
#
#     def _attr_text(obj, *keys: str) -> str:
#         for key in keys:
#             v = _safe_attr(obj, key)
#             if v is None:
#                 continue
#             try:
#                 s = str(v).strip()
#                 if s:
#                     return s
#             except Exception:
#                 continue
#         return ""
#
#     def _var_1d_float(*names: str) -> Optional[np.ndarray]:
#         for name in names:
#             v = ds.variables.get(name) if ds is not None else None
#             if v is None or getattr(v, "ndim", 0) != 1:
#                 continue
#             try:
#                 a = v[:]
#                 if np.ma.isMaskedArray(a):
#                     a = a.filled(np.nan)
#                 arr = np.asarray(a, dtype=np.float32).ravel()
#                 if arr.size:
#                     return arr
#             except Exception:
#                 continue
#         return None
#
#     def _var_1d_int(*names: str) -> Optional[np.ndarray]:
#         arrf = _var_1d_float(*names)
#         if arrf is None:
#             return None
#         try:
#             return arrf.astype(np.int64)
#         except Exception:
#             return None
#
#     def _scalar_from_var_or_attr(names: Tuple[str, ...]) -> Optional[float]:
#         for name in names:
#             v = ds.variables.get(name) if ds is not None else None
#             if v is not None:
#                 try:
#                     a = v[:]
#                     if np.ma.isMaskedArray(a):
#                         a = a.filled(np.nan)
#                     arr = np.asarray(a, dtype=np.float64).ravel()
#                     if arr.size:
#                         f = float(arr[0])
#                         if np.isfinite(f):
#                             return f
#                 except Exception:
#                     pass
#             if ds is not None:
#                 f = _attr_float(ds, name)
#                 if f is not None:
#                     return f
#         return None
#
#     try:
#         try:
#             ds = netCDF4.Dataset("inmemory_fr_pvol.nc", mode="r", memory=file_bytes)
#         except Exception:
#             with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
#                 tmp.write(file_bytes)
#                 tmp_path = tmp.name
#             ds = netCDF4.Dataset(tmp_path, mode="r")
#
#         # Radar position metadata
#         for lat_key in ("latitude", "lat", "radar_latitude", "latitude_of_projection_origin"):
#             v = _scalar_from_var_or_attr((lat_key,))
#             if v is not None:
#                 meta["lat"] = v
#                 break
#         for lon_key in ("longitude", "lon", "radar_longitude", "longitude_of_projection_origin"):
#             v = _scalar_from_var_or_attr((lon_key,))
#             if v is not None:
#                 meta["lon"] = v
#                 break
#
#         az = _var_1d_float("azimuth", "ray_azimuth_angle", "radial_azimuth_coordinate")
#         elev_by_ray = _var_1d_float("elevation", "ray_elevation_angle", "radial_elevation_coordinate")
#         fixed_angles = _var_1d_float("fixed_angle", "sweep_fixed_angle")
#         sweep_start = _var_1d_int("sweep_start_ray_index", "index_of_first_ray_in_sweep")
#         sweep_end = _var_1d_int("sweep_end_ray_index", "index_of_last_ray_in_sweep")
#         range_axis = _var_1d_float(
#             "range",
#             "radial_range_coordinate",
#             "range_to_measurement_volume",
#             "projection_range_coordinate",
#         )
#
#         first_gate_m = _scalar_from_var_or_attr(("meters_to_center_of_first_gate", "rstart", "range_first_gate"))
#         gate_step_m = _scalar_from_var_or_attr(("meters_between_gates", "rscale", "range_resolution", "gate_size"))
#
#         for var_name, var in ds.variables.items():
#             if getattr(var, "ndim", 0) != 2:
#                 continue
#
#             standard_name = _attr_text(var, "standard_name", "quantity", "moment")
#             long_name = _attr_text(var, "long_name", "description", "product")
#             units = _attr_text(var, "units", "unit")
#             qty = _fr_netcdf_quantity(var_name, standard_name, long_name, units)
#             if qty is None:
#                 continue
#
#             try:
#                 raw = var[:]
#             except Exception:
#                 continue
#
#             if raw is None:
#                 continue
#             if np.ma.isMaskedArray(raw):
#                 arr = np.asarray(raw.filled(np.nan), dtype=np.float32)
#             else:
#                 arr = np.asarray(raw, dtype=np.float32)
#             if arr.ndim != 2:
#                 continue
#
#             dims = [str(d).lower() for d in getattr(var, "dimensions", ())]
#             ray_axis = None
#             bin_axis = None
#             for i, dname in enumerate(dims):
#                 if ray_axis is None and any(tok in dname for tok in ("time", "ray", "azimuth")):
#                     ray_axis = i
#                 if bin_axis is None and any(tok in dname for tok in ("range", "gate", "bin")):
#                     bin_axis = i
#
#             if ray_axis is None and az is not None:
#                 if arr.shape[0] == az.size:
#                     ray_axis = 0
#                 elif arr.shape[1] == az.size:
#                     ray_axis = 1
#
#             if bin_axis is None and range_axis is not None:
#                 if arr.shape[0] == range_axis.size and arr.shape[1] != range_axis.size:
#                     bin_axis = 0
#                 elif arr.shape[1] == range_axis.size:
#                     bin_axis = 1
#
#             if ray_axis is None or bin_axis is None or ray_axis == bin_axis:
#                 if arr.shape[0] >= arr.shape[1]:
#                     ray_axis, bin_axis = 0, 1
#                 else:
#                     ray_axis, bin_axis = 1, 0
#
#             try:
#                 arr = np.moveaxis(arr, (ray_axis, bin_axis), (0, 1))
#             except Exception:
#                 continue
#
#             scale = _attr_float(var, "scale_factor", "gain")
#             offset = _attr_float(var, "add_offset", "offset")
#             if scale is not None or offset is not None:
#                 arr = arr * float(scale if scale is not None else 1.0) + float(offset if offset is not None else 0.0)
#
#             fill_values: List[float] = []
#             for key in ("_FillValue", "missing_value", "nodata", "undetect"):
#                 fv = _attr_float(var, key)
#                 if fv is not None:
#                     fill_values.append(float(fv))
#             if fill_values:
#                 for fv in fill_values:
#                     arr[np.isclose(arr, fv, rtol=0.0, atol=1e-6)] = np.nan
#
#             if qty in ("RHOHV", "RHOHVH", "URHOHV", "RHO", "CCORH", "CC"):
#                 finite = arr[np.isfinite(arr)]
#                 if finite.size > 0 and float(np.nanmax(finite)) <= 1.5:
#                     arr = arr * 100.0
#
#             nrays_total, nbins = arr.shape
#             rstart_km = 0.0
#             rscale_km = 1.0
#
#             if range_axis is not None and range_axis.size >= 2:
#                 try:
#                     ra = np.asarray(range_axis[:nbins], dtype=np.float32)
#                     if ra.size >= 2:
#                         diffs = np.diff(ra)
#                         diffs = diffs[np.isfinite(diffs)]
#                         if diffs.size:
#                             step = float(np.nanmedian(np.abs(diffs)))
#                             first = float(ra[0])
#                             if abs(step) > 20.0 or abs(first) > 20.0:
#                                 rscale_km = step / 1000.0
#                                 rstart_km = first / 1000.0
#                             else:
#                                 rscale_km = step
#                                 rstart_km = first
#                 except Exception:
#                     pass
#             elif gate_step_m is not None:
#                 rscale_km = float(gate_step_m) / 1000.0 if abs(float(gate_step_m)) > 20.0 else float(gate_step_m)
#                 if first_gate_m is not None:
#                     rstart_km = float(first_gate_m) / 1000.0 if abs(float(first_gate_m)) > 20.0 else float(first_gate_m)
#
#             if not np.isfinite(rscale_km) or rscale_km <= 0:
#                 rscale_km = 1.0
#             if not np.isfinite(rstart_km):
#                 rstart_km = 0.0
#             max_rng = rstart_km + nbins * rscale_km
#
#             def _append_sweep(chunk: np.ndarray, elev: float, az_part: Optional[np.ndarray]) -> None:
#                 nonlocal lowest_angle
#                 local = np.asarray(chunk, dtype=np.float32)
#                 if az_part is not None and az_part.size == local.shape[0]:
#                     dist = np.minimum(az_part % 360.0, 360.0 - (az_part % 360.0))
#                     north_idx = int(np.argmin(dist))
#                     local = np.roll(local, -north_idx, axis=0)
#                 sweeps_by_qty.setdefault(qty, []).append(
#                     {
#                         "elevation": float(elev),
#                         "rscale_km": float(rscale_km),
#                         "rstart_km": float(rstart_km),
#                         "max_range_km": float(max_rng),
#                         "nrays": int(local.shape[0]),
#                         "nbins": int(local.shape[1]),
#                         "_polar": local,
#                         "nyquist_m_s": None,
#                         "low_nyquist_m_s": None,
#                     }
#                 )
#                 if elev < lowest_angle:
#                     lowest_angle = elev
#                     meta["max_range_km"] = float(max_rng)
#
#             if (
#                 sweep_start is not None
#                 and sweep_end is not None
#                 and sweep_start.size > 0
#                 and sweep_end.size > 0
#             ):
#                 nsw = int(min(sweep_start.size, sweep_end.size))
#                 for i in range(nsw):
#                     s = int(max(0, sweep_start[i]))
#                     e = int(min(nrays_total - 1, sweep_end[i]))
#                     if e < s:
#                         continue
#                     chunk = arr[s : e + 1, :]
#                     if chunk.size == 0:
#                         continue
#                     az_part = az[s : e + 1] if az is not None and az.size >= (e + 1) else None
#                     if fixed_angles is not None and fixed_angles.size > i and np.isfinite(fixed_angles[i]):
#                         elev_val = float(fixed_angles[i])
#                     elif elev_by_ray is not None and elev_by_ray.size >= (e + 1):
#                         elev_val = float(np.nanmedian(elev_by_ray[s : e + 1]))
#                     else:
#                         elev_val = _attr_float(var, "elevation", "elevation_angle", "fixed_angle") or 0.5
#                     _append_sweep(chunk, elev_val, az_part)
#             else:
#                 if fixed_angles is not None and fixed_angles.size:
#                     elev_val = float(fixed_angles[0])
#                 elif elev_by_ray is not None and elev_by_ray.size:
#                     elev_val = float(np.nanmedian(elev_by_ray))
#                 else:
#                     elev_val = _attr_float(var, "elevation", "elevation_angle", "fixed_angle") or 0.5
#                 _append_sweep(arr, elev_val, az)
#
#         for q in sweeps_by_qty:
#             sweeps_by_qty[q].sort(key=lambda s: s["elevation"])
#
#         if not sweeps_by_qty:
#             raise RuntimeError("No readable sweeps found in Meteo-France NetCDF file")
#
#         return sweeps_by_qty, meta
#     finally:
#         try:
#             if ds is not None:
#                 ds.close()
#         except Exception:
#             pass
#         if tmp_path:
#             try:
#                 Path(tmp_path).unlink(missing_ok=True)
#             except Exception:
#                 pass
#
#
def _extract_all_sweeps(file_bytes: bytes) -> Tuple[Dict[str, List[dict]], dict]:
    """Extract ALL elevation sweeps from HDF5 volume file."""
    sweeps_by_qty: Dict[str, List[dict]] = {}
    meta: dict = {"lat": None, "lon": None, "max_range_km": None}
    lowest_angle = 999.0
    hdf = None
    debug_summary: Optional[str] = None
    try:
        hdf = h5py.File(io.BytesIO(file_bytes), "r")
        def _iter_sweep_groups() -> List["h5py.Group"]:
            groups: List["h5py.Group"] = []
            for k in sorted(hdf.keys()):
                if not k.startswith("dataset"):
                    continue
                obj = hdf[k]
                if isinstance(obj, h5py.Group):
                    groups.append(obj)
            if groups:
                return groups

            def _looks_like_sweep(grp: "h5py.Group") -> bool:
                try:
                    if "where" not in grp:
                        return False
                    for kk in grp.keys():
                        if str(kk).startswith("data"):
                            return True
                    # Some providers place "data" at the group root without data1/data2
                    return "data" in grp and "what" in grp
                except Exception:
                    return False

            # Fallback: scan all groups for sweep-like structure
            seen = set()
            found: List["h5py.Group"] = []
            def _visitor(_name, obj) -> None:
                if not isinstance(obj, h5py.Group):
                    return
                if _looks_like_sweep(obj):
                    oid = id(obj)
                    if oid not in seen:
                        seen.add(oid)
                        found.append(obj)
            try:
                hdf.visititems(_visitor)
            except Exception:
                pass
            return found

        def _safe_attr_float(grp: "h5py.Group", key: str) -> Optional[float]:
            try:
                v = grp.attrs.get(key)
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        def _hdf_summary(max_items: int = 25) -> str:
            keys = []
            attrs = []
            items: List[str] = []
            try:
                keys = list(hdf.keys())[:20]
                attrs = list(getattr(hdf, "attrs", {}).keys())[:20]
            except Exception:
                pass
            try:
                def _visit(name, obj):
                    if len(items) >= max_items:
                        return
                    try:
                        if isinstance(obj, h5py.Dataset):
                            items.append(f"{name} [D] shape={getattr(obj, 'shape', None)}")
                        else:
                            items.append(f"{name} [G]")
                    except Exception:
                        items.append(f"{name} [?]")
                hdf.visititems(_visit)
            except Exception:
                pass
            return f"keys={keys} attrs={attrs} items={items}"

        # Prefer root /where when present, else dataset-level /where
        _root_where = None
        try:
            _root_where = hdf.get("where")
        except Exception:
            _root_where = None
        if _root_where is not None:
            _lat = _safe_attr_float(_root_where, "lat")
            _lon = _safe_attr_float(_root_where, "lon")
            if _lat is not None:
                meta["lat"] = _lat
            if _lon is not None:
                meta["lon"] = _lon

        if meta["lat"] is None or meta["lon"] is None:
            for _ds in _iter_sweep_groups():
                try:
                    _wh = _ds.get("where")
                except Exception:
                    _wh = None
                if _wh is None:
                    continue
                _lat = _safe_attr_float(_wh, "lat")
                _lon = _safe_attr_float(_wh, "lon")
                if _lat is not None and _lon is not None:
                    meta["lat"] = _lat
                    meta["lon"] = _lon
                    break

        # Read radar wavelength (cm) from root /how – needed for Nyquist computation.
        # ODIM convention: wavelength stored in centimetres.
        _root_wl_cm: Optional[float] = None
        if "how" in hdf:
            _wl = hdf["how"].attrs.get("wavelength")
            if _wl is not None:
                _root_wl_cm = float(_wl)

        for ds in _iter_sweep_groups():
            try:
                wh = ds.get("where")
            except Exception:
                wh = None
            if wh is None:
                continue
            elev      = float(wh.attrs.get("elangle", 90.0))
            nbins     = int(wh.attrs.get("nbins", 0))
            nrays     = int(wh.attrs.get("nrays", 360))
            rscale_m  = float(wh.attrs.get("rscale", 1000.0))
            rstart_km = float(wh.attrs.get("rstart", 0.0))
            a1gate    = int(wh.attrs.get("a1gate", 0))
            max_rng   = rstart_km + nbins * rscale_m / 1000.0

            # -Nyquist / dual-PRF parameters 
            # NI  = extended (dual-PRF) unambiguous velocity [m/s]
            # lowprf / highprf = PRF values [Hz]; wavelength [cm] may override root value.
            _ds_ni_m_s:  Optional[float] = None   # extended Nyquist
            _ds_low_m_s: Optional[float] = None   # single (low-PRF) Nyquist

            def _attr_float(grp, *names) -> Optional[float]:
                """Read first matching attr from an h5py group; handles scalar/array/bytes."""
                for nm in names:
                    try:
                        val = grp.attrs.get(nm)
                        if val is None:
                            continue
                        if isinstance(val, (bytes, str)):
                            return float(val.decode() if isinstance(val, bytes) else val)
                        v = float(np.asarray(val).flat[0])
                        if np.isfinite(v):
                            return v
                    except Exception:
                        pass
                return None

            # Prefer dataset-level /how, fall back to root /how
            _how_src = (ds["how"] if "how" in ds else None) or (hdf["how"] if "how" in hdf else None)
            if _how_src is not None:
                _ds_ni_m_s = _attr_float(_how_src, "NI", "nyquist_velocity", "unambiguous_velocity")
                if _ds_ni_m_s is not None:
                    _ds_ni_m_s = abs(_ds_ni_m_s)

                _wl2 = _attr_float(_how_src, "wavelength")
                _ds_wl_cm = _wl2 if _wl2 is not None else _root_wl_cm
                _wl_m = (_ds_wl_cm / 100.0) if _ds_wl_cm else None

                _lprf = _attr_float(_how_src, "lowprf", "prf_low")
                if _lprf is not None and _lprf > 0 and _wl_m:
                    _ds_low_m_s = _wl_m * _lprf / 4.0
                else:
                    _hprf = _attr_float(_how_src, "highprf", "prf_high", "prf")
                    if _hprf is not None and _hprf > 0 and _wl_m and _ds_ni_m_s:
                        high_ni = _wl_m * _hprf / 4.0
                        _ratio = max(1, round(_ds_ni_m_s / high_ni)) if high_ni > 0 else 3
                        _ds_low_m_s = _ds_ni_m_s / _ratio
                    if _ds_low_m_s is None and _ds_ni_m_s is not None and _ds_ni_m_s > 0:
                        # Last resort: assume 3:2 dual-PRF ratio (FMI C-band default)
                        _ds_low_m_s = _ds_ni_m_s / 3.0
            # Clamp to physically reasonable range
            if _ds_low_m_s is not None:
                _ds_low_m_s = float(np.clip(_ds_low_m_s, 3.0, 50.0))

            if elev < lowest_angle:
                lowest_angle = elev
                meta["max_range_km"] = max_rng

            for key in sorted(ds.keys()):
                if not key.startswith("data"):
                    continue
                moment = ds[key]
                if "what" not in moment or "data" not in moment:
                    continue
                what  = moment["what"]
                qty_raw = what.attrs.get("quantity", "")
                qty   = qty_raw.decode() if isinstance(qty_raw, bytes) else str(qty_raw)
                # Strip null bytes and whitespace that some encoders add
                qty   = qty.strip('\x00 \t\r\n')

                raw     = moment["data"][:]
                gain    = float(what.attrs.get("gain",    1.0))
                offset  = float(what.attrs.get("offset",  0.0))
                nodata  = float(what.attrs.get("nodata",  -9999))
                undetect= float(what.attrs.get("undetect",-9998))

                arr = raw.astype(np.float32) * gain + offset
                arr[(raw == nodata) | (raw == undetect)] = np.nan

                # Convert velocity from km/h → m/s when the /what units attribute
                # says so (e.g. some MeteoRomania ODIM files encode VRAD in km/h).
                _vel_qtys = {"VRAD", "VRADH", "VRADV", "VRADC", "VR", "V", "VRAD_corr"}
                if qty in _vel_qtys:
                    _units_raw = what.attrs.get("units", "") or ""
                    _units_str = (_units_raw.decode() if isinstance(_units_raw, bytes) else str(_units_raw)).strip().lower()
                    if _units_str in ("km/h", "kmh", "km h-1", "kph"):
                        arr = arr / 3.6
                    elif not _units_str or _units_str in ("m/s", "ms-1", "m s-1", ""):
                        # Heuristic: if the decoded velocity span far exceeds a
                        # realistic Nyquist (>60 m/s) but fits neatly after /3.6,
                        # the file is almost certainly encoded in km/h without a
                        # proper units label (common MeteoRomania quirk).
                        _valid = arr[np.isfinite(arr)]
                        if _valid.size > 0:
                            _absmax = float(np.nanmax(np.abs(_valid)))
                            if _absmax > 60.0 and (_absmax / 3.6) < 50.0:
                                arr = arr / 3.6

                # Scale correlation coefficient from fractional (0-1) to
                # percentage (0-100) so it matches the colormap range.
                if qty in ("RHOHV", "RHOHVH", "URHOHV", "RHO", "CCORH", "CC"):
                    valid = arr[np.isfinite(arr)]
                    if valid.size > 0 and float(np.nanmax(arr)) <= 1.5:
                        arr = arr * 100.0

                # -Azimuth alignment
                # Prefer per-ray azimuth array from /how (more reliable than
                # a1gate, fixes SMHI scans that start at varying azimuths).
                startaz: Optional[np.ndarray] = None
                if "how" in ds:
                    how_grp = ds["how"]
                    for az_key in ("startazA", "stopazA", "azangles", "astart"):
                        az_attr = how_grp.attrs.get(az_key)
                        if az_attr is not None:
                            az_arr = np.asarray(az_attr, dtype=np.float32).ravel()
                            if len(az_arr) == nrays:
                                startaz = az_arr % 360.0
                                break

                if startaz is not None:
                    # Roll so that the ray nearest to azimuth 0° (North) is first
                    dist = np.minimum(startaz, 360.0 - startaz)
                    north_idx = int(np.argmin(dist))
                    polar = np.roll(arr, -north_idx, axis=0)
                else:
                    polar = np.roll(arr, -a1gate, axis=0)

                if qty not in sweeps_by_qty:
                    sweeps_by_qty[qty] = []
                sweeps_by_qty[qty].append({
                    "elevation":      elev,
                    "rscale_km":      rscale_m / 1000.0,
                    "rstart_km":      rstart_km,
                    "max_range_km":   max_rng,
                    "nrays":          nrays,
                    "nbins":          nbins,
                    "_polar":         polar,
                    "nyquist_m_s":    _ds_ni_m_s,
                    "low_nyquist_m_s": _ds_low_m_s,
                })

        for q in sweeps_by_qty:
            sweeps_by_qty[q].sort(key=lambda s: s["elevation"])
        if not sweeps_by_qty:
            meta["hdf_summary"] = _hdf_summary()
        try:
            if hdf is not None:
                hdf.close()
        except Exception:
            pass
        if not sweeps_by_qty:
            # Build a quick structure summary to help diagnose non-ODIM files (e.g., KNMI netCDF)
            try:
                with h5py.File(io.BytesIO(file_bytes), "r") as _hdf:
                    keys = list(_hdf.keys())[:25]
                    attrs = list(getattr(_hdf, "attrs", {}).keys())[:25]
                    # Try to list one group's keys for context
                    grp_keys = []
                    if keys:
                        try:
                            obj = _hdf.get(keys[0])
                            if isinstance(obj, h5py.Group):
                                grp_keys = list(obj.keys())[:25]
                        except Exception:
                            pass
                    debug_summary = f"hdf keys={keys} attrs={attrs} first_group_keys={grp_keys}"
            except Exception:
                debug_summary = None
        if not sweeps_by_qty and debug_summary:
            raise RuntimeError(f"No ODIM sweeps found ({debug_summary})")
        return sweeps_by_qty, meta
    except Exception as exc:
        # Surface some structure hints for non-ODIM files (e.g., KNMI)
        keys = []
        attrs = []
        try:
            try:
                if hdf is not None:
                    hdf.close()
            except Exception:
                pass
            with h5py.File(io.BytesIO(file_bytes), "r") as hdf:
                keys = list(hdf.keys())[:20]
                attrs = list(getattr(hdf, "attrs", {}).keys())[:20]
        except Exception:
            pass
        hint = ""
        if keys or attrs:
            hint = f" | hdf keys={keys} attrs={attrs}"
        raise RuntimeError(f"{exc}{hint}") from exc

# Velocity dealiasing a la NLradar

def _apply_velocity_dual_prf_nlr_only(sweeps_by_qty: Dict[str, List[dict]]) -> None:
    """Apply NLradar dual-PRF dealiasing using sweep Nyquist metadata (no UNRAVEL)."""
    vel_qtys = ("VRADC", "VRADH", "VRAD", "VR", "V", "VRAD_corr")
    for vq in vel_qtys:
        sweeps = sweeps_by_qty.get(vq)
        if not sweeps:
            continue
        for sw in sweeps:
            try:
                polar_work, _, _, transpose_back = _velocity_polar_nrays_nbins(sw)
            except Exception:
                continue
            ni = sw.get("nyquist_m_s")
            lni = sw.get("low_nyquist_m_s")
            try:
                if ni is not None and lni is not None and float(lni) < float(ni):
                    out = _dealias_velocity_nlr(
                        polar_work,
                        nyquist=float(ni),
                        low_nyquist=float(lni),
                    )
                    if transpose_back:
                        out = np.asarray(out, dtype=np.float32).T
                    sw["_polar"] = out
            except Exception:
                pass
        break


def _dealias_velocity_nlr(
    polar: "np.ndarray",
    nyquist: float,
    low_nyquist: float,
    n_iterations: int = 5,
    win_az: int = 4,
    win_rng: int = 4,
    min_neighbors: int = 3,
) -> "np.ndarray":
    """
    Dealias dual-PRF aliased radial velocities using the NLradar method.

    Reference: B. van der Schalie, NLradar (2018).
    https://github.com/Bram94/NLradar/blob/main/Python_files/dealiasing/nlr_dealiasing.py

    Algorithm (repeated for n_iterations):
      1. Convert each gate's velocity to a phase angle scaled by the extended
         Nyquist velocity:  phase = v / nyquist * π  ∈ [−π, π].
      2. Compute a circular (vector-mean) average over a window of
         (2·win_az+1) rays × (2·win_rng+1) range gates, wrapping in azimuth.
      3. Flag each gate whose velocity deviates from the window mean by more
         than `low_nyquist` (the single-PRF Nyquist) as aliased.
      4. Recompute the window mean excluding flagged gates (clean mean).
      5. Correct each flagged gate: add the nearest integer multiple of
         2·low_nyquist that minimises |v_corrected − v_clean_mean|.

    Parameters
    ----------
    polar        : (nrays, nbins) float32 array of radial velocities [m/s], NaN = nodata.
    nyquist      : Extended (dual-PRF) Nyquist velocity [m/s].
    low_nyquist  : Single-PRF (low-PRF) Nyquist velocity [m/s].
                   Dual-PRF aliasing errors appear as integer multiples of
                   2·low_nyquist offset from the true velocity.
    n_iterations : Number of correction passes (default 5, usually sufficient).
    win_az       : Half-width of the averaging window in rays.
    win_rng      : Half-width of the averaging window in range bins.
    min_neighbors: Minimum number of non-aliased neighbours required before a
                   gate is actually corrected (avoids spurious fixes in sparse data).

    Returns
    -------
    Dealiased velocity array with the same shape and NaN positions as `polar`.
    """
    if nyquist <= 0 or low_nyquist <= 0 or low_nyquist >= nyquist:
        return polar.copy()

    out  = polar.copy()
    step = 2.0 * low_nyquist          # correction step size for dual-PRF errors
    max_n = int(np.ceil(150.0 / step)) + 1  # safeguard: no correction larger than ±150 m/s

    win_h = 2 * win_az  + 1
    win_w = 2 * win_rng + 1

    def _box_sum(padded: "np.ndarray") -> "np.ndarray":
        """Integral-image box sum on a pre-padded array → shape (nrays, nbins)."""
        # Axis 0
        cs = np.concatenate(
            [np.zeros((1, padded.shape[1]), dtype=np.float64), padded.cumsum(axis=0)],
            axis=0
        )
        s0 = cs[win_h:] - cs[:-win_h]        # shape (nrays, nbins + 2·win_rng)
        # Axis 1
        cs2 = np.concatenate(
            [np.zeros((s0.shape[0], 1), dtype=np.float64), s0.cumsum(axis=1)],
            axis=1
        )
        return cs2[:, win_w:] - cs2[:, :-win_w]   # shape (nrays, nbins)

    def _pad(arr: "np.ndarray") -> "np.ndarray":
        """Wrap in azimuth, zero-pad in range."""
        # Azimuth: circular (weather wraps around 360°)
        top = arr[-win_az:, :]  if win_az  > 0 else arr[:0, :]
        bot = arr[:win_az,  :]  if win_az  > 0 else arr[:0, :]
        az  = np.concatenate([top, arr, bot], axis=0)
        # Range: outside the scan → treat as no-data (zero weight)
        return np.pad(az, ((0, 0), (win_rng, win_rng)), mode="constant", constant_values=0.0)

    for _it in range(n_iterations):
        valid = np.isfinite(out)
        if not np.any(valid):
            break

        phase = out / nyquist * np.pi                     # ∈ [−π, π]
        cos_p = np.where(valid, np.cos(phase), 0.0)
        sin_p = np.where(valid, np.sin(phase), 0.0)
        cnt   = valid.astype(np.float64)

        # -Full-window circular mean
        sc  = _box_sum(_pad(cos_p))
        ss  = _box_sum(_pad(sin_p))
        n1  = _box_sum(_pad(cnt))
        v_avg = np.where(n1 > 0, np.arctan2(ss, sc) / np.pi * nyquist, 0.0)

        # -Detect aliased gates
        aliased = valid & (np.abs(out - v_avg) > low_nyquist)
        if not np.any(aliased):
            break                                         # converged

        # -Clean-window circular mean (excluding aliased gates)
        valid2 = valid & ~aliased
        sc2 = _box_sum(_pad(np.where(valid2, cos_p, 0.0)))
        ss2 = _box_sum(_pad(np.where(valid2, sin_p, 0.0)))
        n2  = _box_sum(_pad(valid2.astype(np.float64)))

        has_clean = n2 >= min_neighbors
        v_avg_clean = np.where(
            has_clean,
            np.arctan2(ss2, sc2) / np.pi * nyquist,
            v_avg,                                        # fall back to full mean
        )

        # -Apply correction 
        # Only fix aliased gates where we have a reliable reference mean.
        to_fix = aliased & (has_clean | (n1 >= min_neighbors))
        diff   = v_avg_clean - out
        # Replace NaN (where v_avg_clean was 0/0 and out is NaN) with 0 before cast
        diff_safe = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        n_corr = np.clip(np.rint(diff_safe / step).astype(np.int32), -max_n, max_n)
        out    = np.where(to_fix, out + n_corr.astype(np.float32) * np.float32(step), out)

    return out


def _encode_sweep_for_overlay(sweep: dict) -> Optional[dict]:
    polar    = sweep["_polar"]
    rscale   = sweep["rscale_km"]
    nrays, nbins = polar.shape

    # Geometry-based rendering draws each gate as an exact trapezoid polygon,
    # so no decimation is needed — send the full native-resolution polar array.
    finite = np.isfinite(polar)
    if not np.any(finite):
        return None

    vmin = float(np.nanmin(polar))
    vmax = float(np.nanmax(polar))
    if vmax <= vmin:
        vmax = vmin + 1.0

    enc = np.zeros(polar.shape, dtype=np.uint16)
    vals = (polar[finite] - vmin) / (vmax - vmin) * 65534.0
    enc[finite] = np.clip(np.round(vals), 0, 65534).astype(np.uint16) + 1

    # compresslevel=1 is ~10× faster than 6 with minimal size penalty on
    # uint16 big-endian binary that already has low entropy.
    compressed = gzip.compress(enc.astype(">u2").tobytes(), compresslevel=1)
    b64 = base64.b64encode(compressed).decode()
    return {
        "b64":         b64,
        "nrays":       int(nrays),
        "nbins":       int(nbins),
        "rscaleKm":    float(rscale),
        "rstartKm":    float(sweep["rstart_km"]),
        "maxRangeKm":  float(sweep["max_range_km"]),
        "elevationDeg": float(sweep.get("elevation", 0.0)),
        "minVal":      vmin,
        "maxVal":      vmax,
    }


def _encode_sweep_for_cs(polar: "np.ndarray", rscale_km: float,
                          max_bins: int = MAX_CS_BINS) -> Optional[dict]:
    nrays, nbins = polar.shape
    effective_rscale = rscale_km

    if nbins > max_bins:
        step = max(1, nbins // max_bins)
        polar = polar[:, ::step]
        nbins = polar.shape[1]
        effective_rscale = rscale_km * step

    finite = np.isfinite(polar)
    if not np.any(finite):
        return None

    vmin = float(np.nanmin(polar))
    vmax = float(np.nanmax(polar))
    if vmax <= vmin:
        vmax = vmin + 1.0

    enc = np.zeros(polar.shape, dtype=np.uint16)
    vals = (polar[finite] - vmin) / (vmax - vmin) * 65534.0
    enc[finite] = np.clip(np.round(vals), 0, 65534).astype(np.uint16) + 1

    compressed = gzip.compress(enc.astype(">u2").tobytes(), compresslevel=1)
    b64 = base64.b64encode(compressed).decode()
    return {
        "b64": b64, "nrays": nrays, "nbins": nbins,
        "rscaleKm": effective_rscale, "minVal": vmin, "maxVal": vmax,
    }


# Cache for _polar_to_cartesian pre-computed index arrays.
# Key: (output_size, nrays, nbins) → (azimuth_bins, range_bins, mask_outside)
# The 2048×2048 index arrays are ~32 MB; caching by key avoids recomputing them
# for every station that shares the same geometry.
_P2C_INDEX_CACHE: Dict[Tuple[int, int, int], Tuple["np.ndarray", "np.ndarray", "np.ndarray"]] = {}
_P2C_INDEX_CACHE_LOCK = threading.Lock()


def _polar_to_cartesian(polar: "np.ndarray",
                         output_size: int = DERIVED_CARTESIAN_SIZE) -> "np.ndarray":
    nrays, nbins = polar.shape
    cache_key = (output_size, nrays, nbins)

    with _P2C_INDEX_CACHE_LOCK:
        cached = _P2C_INDEX_CACHE.get(cache_key)

    if cached is None:
        y, x = np.indices((output_size, output_size), dtype=np.float32)
        center_x = (output_size - 1) / 2
        center_y = (output_size - 1) / 2
        dx = x - center_x
        dy = center_y - y
        radius      = np.sqrt(dx * dx + dy * dy)
        max_radius  = output_size / 2 - 2
        range_bins  = np.clip(np.rint((radius / max_radius) * (nbins - 1)).astype(np.int32), 0, nbins - 1)
        azimuth     = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
        azimuth_bins = np.clip(np.rint((azimuth / 360.0) * (nrays - 1)).astype(np.int32), 0, nrays - 1)
        outside     = radius > max_radius
        cached      = (azimuth_bins, range_bins, outside)
        with _P2C_INDEX_CACHE_LOCK:
            _P2C_INDEX_CACHE[cache_key] = cached

    azimuth_bins, range_bins, outside = cached
    cartesian = polar[azimuth_bins, range_bins]
    cartesian[outside] = np.nan
    return cartesian


def _png_data_url(rgba: "np.ndarray") -> str:
    image  = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_values_png(values: "np.ndarray") -> dict:
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return {"url": None, "min": None, "max": None}
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if max_value <= min_value:
        max_value = min_value + 1.0
    encoded = np.full(values.shape, 65535, dtype=np.uint16)
    scaled  = ((values[finite_mask] - min_value) / (max_value - min_value)) * 65534.0
    encoded[finite_mask] = np.clip(np.rint(scaled), 0, 65534).astype(np.uint16)
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    rgba[..., 0] = (encoded >> 8).astype(np.uint8)
    rgba[..., 1] = (encoded & 255).astype(np.uint8)
    rgba[..., 3] = np.where(finite_mask, 255, 0).astype(np.uint8)
    return {"url": _png_data_url(rgba), "min": min_value, "max": max_value}


STORM_MOTION: Tuple[float, float] = (240.0, 15.0)  # direction°, speed m/s


def _beam_height_km(range_km: "np.ndarray", elev_deg: float) -> "np.ndarray":
    el_rad = np.deg2rad(elev_deg)
    ke = 4.0 / 3.0
    Re = 6371.0
    keRe = ke * Re
    return np.sqrt(range_km**2 + keRe**2 + 2.0 * range_km * keRe * np.sin(el_rad)) - keRe


def _compute_echo_tops_km(sweeps: List[dict], min_dbz: float = 18.0) -> Optional["np.ndarray"]:
    tops: Optional["np.ndarray"] = None
    ref_nbins: Optional[int] = None

    for sw in sweeps:
        polar = sw["_polar"]
        nrays, nbins = polar.shape

        ranges = sw["rstart_km"] + (
            np.arange(nbins, dtype=np.float32) + 0.5
        ) * sw["rscale_km"]
        heights = _beam_height_km(ranges[None, :], sw["elevation"])
        mask = polar > float(min_dbz)

        if tops is None:
            tops = np.zeros_like(polar, dtype=np.float32)
            ref_nbins = nbins
        else:
            assert ref_nbins is not None
            common = min(ref_nbins, nbins)
            if common <= 0:
                continue
            tops = tops[:, :common]
            mask = mask[:, :common]
            heights = heights[:, :common]
            ref_nbins = common

        tops = np.where(mask, np.maximum(tops, heights), tops)

    if tops is None:
        return None
    tops[tops == 0.0] = np.nan
    return tops


def _compute_mesh_polar(
    sweeps: List[dict],
    h0_km: float = 2.0,
    h_neg20_km: float = 5.5,
) -> "Optional[np.ndarray]":
    """Compute Maximum Expected Size of Hail (MESH) in cm.

    Implements the Witt et al. (1998, *Weather and Forecasting*, 13, 286-303)
    algorithm adapted for polar multi-sweep volume data.

    MESH [mm] = 2.54 * sqrt(SHI)
    Returned in **cm** (MESH_mm / 10), capped at 15 cm.

    SHI  =  0.1 * integral_{H0}^{Htop}  W(H) * E(Z)  dH   [J m⁻¹ s⁻¹]

    Hail kinetic energy flux (Waldvogel et al. 1978):
        E(Z)  =  5e-6 * 10^(0.084 * Z_dBZ)           [J m⁻² s⁻¹]

    Thermodynamic weight:
        W(H) = 0                               H < H_0   (below 0 °C isotherm)
             = (H − H_0) / (H_{-20} − H_0)    H_0 ≤ H ≤ H_{-20}
             = 1                               H > H_{-20}

    Default isotherm heights are climatological summer values for the
    Northern-European / Scandinavian radar domain:
        H_0   = 2.0 km (0 °C)
        H_{-20} = 5.5 km (-20 °C)

    The vertical integral is performed column-by-column using the
    trapezoidal rule over sweeps sorted by ascending elevation angle.
    Gates with NaN reflectivity (no echo) contribute zero to the integral.
    """
    if not sweeps:
        return None

    # Sort ascending by elevation so beam heights are monotone per column
    sorted_sweeps = sorted(sweeps, key=lambda s: float(s.get("elevation", 0.0)))

    base_sw = sorted_sweeps[0]
    nrays_ref: int = base_sw["_polar"].shape[0]
    nbins_ref: int = base_sw["_polar"].shape[1]

    shi = np.zeros((nrays_ref, nbins_ref), dtype=np.float64)
    h_span = max(float(h_neg20_km - h0_km), 0.001)  # guard against degenerate config

    # Trackers for trapezoidal integration across sweeps
    prev_h: "np.ndarray" = np.full(nbins_ref, np.nan, dtype=np.float64)   # (nbins,)
    prev_we: "np.ndarray" = np.full((nrays_ref, nbins_ref), 0.0, dtype=np.float64)

    for sw in sorted_sweeps:
        polar = sw["_polar"]
        nrays, nbins = polar.shape
        cr = min(nrays_ref, nrays)
        cb = min(nbins_ref, nbins)

        # Beam-centre height in km; azimuth-independent so computed per bin
        ranges_1d = (
            sw["rstart_km"]
            + (np.arange(nbins, dtype=np.float64) + 0.5) * sw["rscale_km"]
        )
        h_1d = _beam_height_km(ranges_1d, float(sw["elevation"])).astype(np.float64)  # (nbins,)
        h_cb = h_1d[:cb]  # (cb,)

        z_dbz = polar[:cr, :cb].astype(np.float64)
        valid = np.isfinite(z_dbz)

        # Hail kinetic energy flux E(Z) [J m⁻² s⁻¹]
        e_z = np.where(
            valid,
            5e-6 * np.power(10.0, 0.084 * np.where(valid, z_dbz, 0.0)),
            0.0,
        )

        # Thermodynamic weight W(H) – broadcast height (bins only) over all rays
        w_h = np.clip((h_cb - float(h0_km)) / h_span, 0.0, 1.0)[None, :]  # (1, cb)

        we = w_h * e_z  # (cr, cb); zero where no echo
        we = np.where(valid, we, 0.0)

        # Trapezoidal integration using height difference to previous sweep
        h_prev_cb = prev_h[:cb]
        finite_prev = np.isfinite(h_prev_cb)
        if np.any(finite_prev):
            dh_m = np.where(
                finite_prev,
                np.maximum(h_cb - h_prev_cb, 0.0) * 1000.0,  # km → m
                0.0,
            )  # (cb,)
            trap = 0.5 * (we + prev_we[:cr, :cb]) * dh_m[None, :]  # (cr, cb)
            shi[:cr, :cb] += np.nan_to_num(trap, nan=0.0, posinf=0.0, neginf=0.0)

        prev_h[:cb] = h_cb
        prev_we[:cr, :cb] = we

    # SHI = 0.1 * vertical integral
    shi_final = 0.1 * shi

    # MESH [mm] = 2.54 * sqrt(SHI); convert to cm and cap at 15 cm
    mesh_mm = 2.54 * np.sqrt(np.maximum(shi_final, 0.0))
    mesh_cm = np.clip(mesh_mm / 10.0, 0.0, 15.0).astype(np.float32)

    # Suppress sub-threshold noise (< 0.1 cm ≈ MESH values from very weak Z)
    mesh_cm[mesh_cm < 0.1] = np.nan

    return mesh_cm


def _compute_local_shear(
    vel_field: "np.ndarray",
    *,
    az_window: int = 2,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Gate-to-gate azimuthal shear proxy in polar (rays × bins) space.

    Searches within ±az_window rays at the **same range bin** for the maximum
    outbound (positive) and maximum inbound (negative) velocities.  Only
    azimuthal neighbours are considered because the rotational velocity of a
    mesocyclone / TVS couplet is defined by the velocity difference between
    gates that are separated *across* the beam (azimuthally), not along the
    beam (in range).  Mixing range-direction neighbours would conflate radial
    divergence / convergence with azimuthal rotation, inflating the apparent
    shear and producing physically meaningless couplets at diagonal offsets.

    Azimuthal wrap-around (np.roll) is correct for a 360° polar scan.
    No range wrap-around is applied.

    Parameters
    ----------
    vel_field : 2-D array (nrays × nbins) of dealiased radial velocity [m/s].
                NaN / Inf are treated as missing gates.
    az_window : Number of adjacent rays on each side to search (default 2,
                giving a 5-ray window that spans ~5° for 1°-resolution data).
    """
    v = np.array(vel_field, dtype=np.float32)
    nrays, nbins = v.shape

    v_pos_max = np.full((nrays, nbins), -np.inf, dtype=np.float32)
    v_neg_min = np.full((nrays, nbins),  np.inf, dtype=np.float32)

    for di in range(-az_window, az_window + 1):
        # np.roll wraps azimuth correctly for a 360° scan.
        # No dj (range) offset: only same-bin neighbours count.
        shifted = np.roll(v, -di, axis=0)          # shifted[r, b] = v[(r+di)%nrays, b]
        valid_shift = np.isfinite(shifted)

        pos = np.where(valid_shift & (shifted > 0.0), shifted, -np.inf)
        neg = np.where(valid_shift & (shifted < 0.0), shifted,  np.inf)
        np.maximum(v_pos_max, pos, out=v_pos_max)
        np.minimum(v_neg_min, neg, out=v_neg_min)

    # A valid couplet requires at least one positive and one negative gate in
    # the azimuthal window.
    valid = (v_pos_max > 0.0) & (v_neg_min < 0.0)
    shear = np.zeros((nrays, nbins), dtype=np.float32)
    shear[valid] = v_pos_max[valid] - v_neg_min[valid]
    return shear, v_pos_max, v_neg_min


def _compute_nrot_polar(
    vel_field: "np.ndarray",
    *,
    pair_threshold: float = 5.0,
    az_window: int = 2,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    """Compute rotational velocity proxy (NROT = Vrot = 0.5 · ΔV) from radial velocity.

    NROT is the standard rotational velocity estimate used by WSR-88D and NSSL
    mesocyclone/TVS algorithms.  For a couplet with peak outbound velocity Vmax
    and peak inbound velocity Vmin (both measured azimuthally adjacent to the
    same range gate):

        NROT = 0.5 · (Vmax − Vmin)   [m/s]

    This is range-independent, which is why it is preferred over azimuthal
    shear for operational level thresholding.

    Parameters
    ----------
    vel_field      : 2-D array (nrays × nbins) of dealiased radial velocity [m/s].
    pair_threshold : Minimum absolute velocity [m/s] that each arm of the
                     couplet must reach independently.  Values below this in
                     *both* the positive and negative direction are treated as
                     noise and excluded from couplet detection.
                     5 m/s is sufficient to reject sub-noise returns while
                     allowing detection of weak rotation onset at ~11 m/s NROT.
    az_window      : Passed to _compute_local_shear; defines the azimuthal
                     search half-width in rays (default 2 → 5-ray window ≈ 5°).

    Returns
    -------
    nrot, shear, v_pos_max, v_neg_min, mask_rot
      nrot     – rotational velocity [m/s], NaN where couplet criterion not met
      shear    – gate-to-gate azimuthal ΔV = 2 · NROT [m/s]
      v_pos_max, v_neg_min – peak outbound / inbound in the azimuthal window
      mask_rot – boolean couplet validity mask
    """
    shear, v_pos_max, v_neg_min = _compute_local_shear(vel_field, az_window=az_window)
    # Both arms of the couplet must independently exceed the noise floor.
    mask_rot = (v_pos_max >= pair_threshold) & (v_neg_min <= -pair_threshold)
    nrot = np.full_like(shear, np.nan, dtype=np.float32)
    nrot[mask_rot] = 0.5 * shear[mask_rot]
    return nrot, shear, v_pos_max, v_neg_min, mask_rot


def _box_mean_3x3(arr: "np.ndarray") -> "np.ndarray":
    """3x3 mean filter without az/range wraparound artifacts."""
    a = np.array(arr, dtype=np.float32, copy=False)
    nr, nc = a.shape
    vals = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.isfinite(a).astype(np.float32)

    pv = np.pad(vals, ((1, 1), (1, 1)), mode="edge")
    pw = np.pad(w, ((1, 1), (1, 1)), mode="edge")

    num = np.zeros((nr, nc), dtype=np.float32)
    den = np.zeros((nr, nc), dtype=np.float32)
    for di in range(3):
        for dj in range(3):
            num += pv[di:di + nr, dj:dj + nc]
            den += pw[di:di + nr, dj:dj + nc]

    out = np.full((nr, nc), np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=den > 0.0)
    return out


def _compute_meso_probability(
    nrot_polar: "np.ndarray",
    shear_polar: "np.ndarray",
    refl_polar: "np.ndarray",
    continuity: Optional["np.ndarray"] = None,
    azshear_polar: Optional["np.ndarray"] = None,
) -> "np.ndarray":
    """
    Mesocyclone probability proxy [0..100].

    Modelled after WSR-88D/NSSL mesocyclone detection criteria:

      - Rotational velocity (NROT = 0.5·ΔV):
            ≥ 11 m/s → contributes (weak meso onset), saturates at 45 m/s.
            Operational weak-meso threshold is typically 11–14 m/s.
      - Gate-to-gate velocity difference (shear):
            Scales 15 → 70 m/s.
      - Reflectivity co-location:
            Scales 20 → 50 dBZ (lowered from 30 dBZ to catch rain-wrapped
            and marginal convection).
      - Vertical continuity (fraction of upper tilts with NROT ≥ 20 m/s):
            Raised weight to 0.15; deeper-column rotation is a strong
            mesocyclone discriminator.
      - Azimuthal shear bonus (tight couplet indicator, ×10⁻³ s⁻¹):
            ≥ 5 × 10⁻³ s⁻¹ adds up to 0.10 of the total score.
      - Spatial coherence (neighbourhood rotation support):
            Changed from a *multiplicative* penalty (which crushed isolated
            but real rotation to near-zero) to an *additive* factor.
            Formula: score × (0.55 + 0.45 · coherence).
            A single isolated couplet (coherence ≈ 1/9 ≈ 0.11) now reaches
            ~60 % of its intrinsic core score rather than ~11 %.
            The coherence mask uses the lower 11 m/s onset threshold so that
            weak surrounding gates still count toward neighbourhood support.
    """
    # NROT: onset 11 m/s (operational weak-meso), saturates at 45 m/s
    rot_norm = np.clip((nrot_polar - 11.0) / 34.0, 0.0, 1.0)
    # Gate-to-gate shear: onset 15 m/s, saturates at 70 m/s
    shear_norm = np.clip((shear_polar - 15.0) / 55.0, 0.0, 1.0)
    # Reflectivity: onset 20 dBZ, saturates at 50 dBZ
    refl_norm = np.clip((refl_polar - 20.0) / 30.0, 0.0, 1.0)

    if continuity is None:
        cont_norm = np.zeros_like(rot_norm, dtype=np.float32)
    else:
        cont_norm = np.clip(np.nan_to_num(continuity, nan=0.0), 0.0, 1.0)

    # Azimuthal shear (tight-couplet bonus): onset 5×10⁻³, saturates at 25×10⁻³ s⁻¹
    if azshear_polar is not None:
        cr = min(rot_norm.shape[0], azshear_polar.shape[0])
        cb = min(rot_norm.shape[1], azshear_polar.shape[1])
        azsh_norm = np.zeros_like(rot_norm, dtype=np.float32)
        azsh_norm[:cr, :cb] = np.clip(
            np.nan_to_num(azshear_polar[:cr, :cb], nan=0.0) / 25.0, 0.0, 1.0
        )
    else:
        azsh_norm = np.zeros_like(rot_norm, dtype=np.float32)

    # Core weighted score
    core = (
        0.38 * np.nan_to_num(rot_norm,   nan=0.0)
        + 0.20 * np.nan_to_num(shear_norm, nan=0.0)
        + 0.14 * np.nan_to_num(refl_norm,  nan=0.0)
        + 0.18 * cont_norm
        + 0.10 * azsh_norm
    )

    # Spatial coherence: fraction of 3×3 neighbourhood with NROT ≥ 11 m/s.
    # Additive formulation: score × (0.55 + 0.45 · coherence).
    # This prevents real but compact rotation couplets from being penalised
    # to near-zero by a multiplicative factor.
    rot_present = np.isfinite(nrot_polar) & (nrot_polar >= 11.0)
    coherence = np.clip(
        np.nan_to_num(_box_mean_3x3(rot_present.astype(np.float32)), nan=0.0),
        0.0,
        1.0,
    )
    score = 100.0 * core * (0.55 + 0.45 * coherence)

    # Validity gate: finite NROT and reflectivity ≥ 20 dBZ
    valid = np.isfinite(nrot_polar) & np.isfinite(refl_polar) & (refl_polar >= 20.0)
    out = np.where(valid, score.astype(np.float32), np.nan)
    return np.clip(out, 0.0, 100.0)


def _compute_azimuthal_shear_polar(
    vel_field: "np.ndarray",
    *,
    rstart_km: float,
    rscale_km: float,
) -> "np.ndarray":
    """
    Azimuthal shear [×10⁻³ s⁻¹] from radial velocity using the standard
    central-difference formula in polar coordinates:

        AzShear(r, θ) = [ V(r, θ+Δθ) − V(r, θ−Δθ) ] / (2 · r · Δθ)

    where r is the slant range (m) at each bin and Δθ is the azimuthal step
    in radians.  The result is multiplied by 1 000 to give ×10⁻³ s⁻¹, which
    is the conventional display unit (a tornadic couplet typically reaches
    10–30 × 10⁻³ s⁻¹).

    Only cyclonic (positive) shear is retained; NaN otherwise.
    Requires at least 10 valid neighbours in the azimuthal direction to avoid
    noise at near-zero ranges.
    """
    v = np.array(vel_field, dtype=np.float32)
    nrays, nbins = v.shape

    # Azimuthal step in radians (assumes uniform ray spacing)
    az_step_rad = (2.0 * np.pi) / max(1, nrays)

    # Slant-range at each bin centre (m); clip to ≥500 m to avoid /0 near radar
    ranges_m = (rstart_km + (np.arange(nbins, dtype=np.float32) + 0.5) * rscale_km) * 1000.0
    ranges_m = np.maximum(ranges_m, 500.0)

    # Arc distance between adjacent azimuths at each range (m)
    arc_m = ranges_m * az_step_rad  # shape (nbins,)

    # Central-difference neighbours with azimuthal wrap-around
    v_prev = np.roll(v, 1, axis=0)   # V[ray−1, bin]
    v_next = np.roll(v, -1, axis=0)  # V[ray+1, bin]

    # Both neighbours must be finite
    valid = np.isfinite(v) & np.isfinite(v_prev) & np.isfinite(v_next)

    delta_v = np.where(valid, v_next - v_prev, np.nan)

    # AzShear in s⁻¹, then scale to ×10⁻³ s⁻¹
    az_shear = delta_v / (2.0 * arc_m[None, :]) * 1000.0

    # Retain only cyclonic (positive) shear
    az_shear = np.where(valid & (az_shear > 0.0), az_shear, np.nan)

    return az_shear.astype(np.float32)


def _composite_azshear_polar(polar_list: List["np.ndarray"]) -> "np.ndarray":
    """
    Merge a list of azimuthal shear polar arrays by taking the element-wise
    maximum, ignoring NaN (missing).  Arrays are aligned to the shape of the
    first (most recent) entry; smaller arrays are zero-padded, larger ones
    cropped.
    """
    if not polar_list:
        return np.full((1, 1), np.nan, dtype=np.float32)
    ref = polar_list[0]
    nr, nb = ref.shape
    out = np.where(np.isfinite(ref), ref, np.nan).astype(np.float32)
    for other in polar_list[1:]:
        or_, ob_ = other.shape
        cr = min(nr, or_)
        cb = min(nb, ob_)
        patch = other[:cr, :cb]
        mask = np.isfinite(patch)
        out_view = out[:cr, :cb]
        out[:cr, :cb] = np.where(mask, np.maximum(np.where(np.isfinite(out_view), out_view, -np.inf), patch), out_view)
    return out


def _compute_srv_polar(vel_sweep: dict, storm_motion: Tuple[float, float]) -> "np.ndarray":
    direction_deg, speed = storm_motion
    polar = vel_sweep["_polar"]
    nrays, _ = polar.shape
    azis = np.linspace(0.0, 360.0, nrays, endpoint=False, dtype=np.float32)
    srv_term = speed * np.cos(np.deg2rad(direction_deg - azis))[:, None]
    return polar + srv_term


def _destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> Tuple[float, float]:
    Re = 6371.0
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    brng = np.radians(bearing_deg)
    d = distance_km / Re

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(brng)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(d) * np.cos(lat1),
        np.cos(d) - np.sin(lat1) * np.sin(lat2),
    )
    return float(np.degrees(lat2)), float(np.degrees(lon2))


def _compute_tvs_markers(
    nrot_polar: "np.ndarray",
    vel_sweep: dict,
    station_lat: float,
    station_lon: float,
    refl_polar: Optional["np.ndarray"] = None,
    azshear_polar: Optional["np.ndarray"] = None,
) -> List[dict]:
    """
    Detect Tornado Vortex Signature (TVS) and mesocyclone couplet markers.

    Operational two-tier framework consistent with WSR-88D/NSSL TVS detection.
    NROT = Vrot = 0.5 · ΔV is the range-independent rotational velocity [m/s].

    Level classification (NROT baseline, upgradeable by azimuthal shear):
      1 – Weak rotation       NROT  11–19 m/s   (~22–38 m/s gate-to-gate ΔV)
      2 – Moderate meso       NROT  20–27 m/s   (~40–54 m/s ΔV)
      3 – Strong meso         NROT  28–35 m/s   (~56–70 m/s ΔV)
      4 – TVS                 NROT  36–47 m/s   (~72–94 m/s ΔV; WSR-88D TDA regime)
      5 – Tornadic TVS (ETVS) NROT ≥ 48 m/s    (≥96 m/s ΔV; exceptional rotation)

    Azimuthal shear upgrade (tightness bonus):
      A high AzShear at moderate NROT implies the couplet is spatially compact
      (sub-beam-width scale), which operationally warrants a higher classification
      regardless of the NROT level.  The required NROT base at each upgrade tier
      matches the *lower* boundary of the target tier's predecessor so the upgrade
      is physically grounded:

        AzShear ≥ 15 × 10⁻³ s⁻¹  AND NROT ≥ 11 m/s  → promote to level 3
        AzShear ≥ 25 × 10⁻³ s⁻¹  AND NROT ≥ 20 m/s  → promote to level 4
        AzShear ≥ 40 × 10⁻³ s⁻¹  AND NROT ≥ 28 m/s  → promote to level 5

      At 50 km range / 1° spacing, 15 × 10⁻³ s⁻¹ corresponds to ΔV ≈ 26 m/s
      (NROT ≈ 13 m/s); 40 × 10⁻³ s⁻¹ corresponds to ΔV ≈ 70 m/s (NROT ≈ 35 m/s),
      so the upgrade thresholds are consistent with the underlying rotation signal.

    Detection quality controls:
      • Strict azimuthal opposite-sign velocity couplet (inherited from NROT)
      • Local maximum within a 5-ray × 7-bin window; range axis uses edge-pad
        (no wrap-around) so near-range and far-range gates are not spuriously
        compared against the opposite edge of the scan
      • Reflectivity co-location ≥ 20 dBZ (suppresses false detections in clear air)
      • Minimum slant range ≥ 3 km (eliminates near-radar aliasing artefacts)
      • Minimum couplet separation: 6 rays azimuthally, 10 bins in range
        (avoids duplicate markers on the same physical couplet)
      • Output includes azShear [×10⁻³ s⁻¹] at the couplet centre for display
    """
    if nrot_polar.size == 0:
        return []

    rstart = float(vel_sweep.get("rstart_km", 0.0))
    rscale = float(vel_sweep.get("rscale_km", 1.0))
    nrays  = int(vel_sweep.get("nrays", nrot_polar.shape[0]))
    nbins  = nrot_polar.shape[1]

    # -Minimum-range mask (≥ 3 km suppresses near-radar noise) 
    bin_ranges_km = rstart + (np.arange(nbins, dtype=np.float32) + 0.5) * rscale
    range_ok = bin_ranges_km >= 3.0  # shape (nbins,)

    # -Baseline NROT level classification 
    # Thresholds calibrated to WSR-88D/NSSL operational values:
    #   weak onset 11 m/s (Brown & Wood 2012), TVS ~36 m/s, ETVS ~48 m/s.
    levels = np.zeros(nrot_polar.shape, dtype=np.uint8)
    levels[nrot_polar >= 11.0] = 1
    levels[nrot_polar >= 20.0] = 2
    levels[nrot_polar >= 28.0] = 3
    levels[nrot_polar >= 36.0] = 4
    levels[nrot_polar >= 48.0] = 5

    # -Azimuthal-shear level upgrade 
    # High AzShear at moderate NROT indicates a spatially tight couplet that
    # may be unresolved by the beam width but still warrants a higher severity.
    # Each upgrade tier requires a minimum NROT equal to the lower boundary of
    # the tier below the target so that the upgrade is physically consistent.
    if azshear_polar is not None:
        cr = min(levels.shape[0], azshear_polar.shape[0])
        cb = min(levels.shape[1], azshear_polar.shape[1])
        azsh = np.nan_to_num(azshear_polar[:cr, :cb], nan=0.0)
        lv   = levels[:cr, :cb]
        # AzShear ≥ 15×10⁻³ with NROT ≥ 11 m/s (level ≥ 1) → promote to ≥ level 3
        lv[:] = np.where((azsh >= 15.0) & (lv >= 1), np.maximum(lv, 3), lv)
        # AzShear ≥ 25×10⁻³ with NROT ≥ 20 m/s (level ≥ 2) → promote to ≥ level 4
        lv[:] = np.where((azsh >= 25.0) & (lv >= 2), np.maximum(lv, 4), lv)
        # AzShear ≥ 40×10⁻³ with NROT ≥ 28 m/s (level ≥ 3) → promote to ≥ level 5
        lv[:] = np.where((azsh >= 40.0) & (lv >= 3), np.maximum(lv, 5), lv)
        levels[:cr, :cb] = lv

    # Apply range mask (broadcast over rays)
    mask = (levels > 0) & range_ok[np.newaxis, :]

    # -Reflectivity co-location (≥ 20 dBZ) 
    if refl_polar is not None:
        common_rays = min(mask.shape[0], refl_polar.shape[0])
        common_bins = min(mask.shape[1], refl_polar.shape[1])
        mask       = mask[:common_rays, :common_bins]
        levels     = levels[:common_rays, :common_bins]
        nrot_polar = nrot_polar[:common_rays, :common_bins]
        refl_sub   = refl_polar[:common_rays, :common_bins]
        mask &= np.isfinite(refl_sub) & (refl_sub >= 20.0)

    if not np.any(mask):
        return []

    # -Local maximum: 5-ray × 7-bin window 
    # Azimuth wraps (360° scan); range does NOT wrap (edge-pad with -inf so
    # near-range gates are never compared against the far edge of the array).
    field = np.nan_to_num(nrot_polar, nan=-np.inf)
    nrot_rows, nrot_cols = field.shape

    # Pad range dimension with -inf (edge of scan, no wrap)
    # Pad azimuth with wrap-around (roll handles this in the loop)
    locmax = np.ones((nrot_rows, nrot_cols), dtype=bool)
    for di in range(-2, 3):         # ±2 rays  → 5-ray azimuthal window
        for dj in range(-3, 4):     # ±3 bins  → 7-bin range window
            if di == 0 and dj == 0:
                continue
            # Azimuth: wrap-around via roll
            rolled_az = np.roll(field, di, axis=0)
            # Range: shift with edge fill (-inf at boundaries, no wrap)
            if dj == 0:
                neighbor = rolled_az
            elif dj > 0:
                neighbor = np.full_like(rolled_az, -np.inf)
                neighbor[:, dj:] = rolled_az[:, :-dj]
            else:
                neighbor = np.full_like(rolled_az, -np.inf)
                neighbor[:, :dj] = rolled_az[:, -dj:]
            locmax &= (field >= neighbor)
    mask &= locmax

    ray_idx, bin_idx = np.where(mask)
    if ray_idx.size == 0:
        return []

    vals  = field[ray_idx, bin_idx]
    lvls  = levels[ray_idx, bin_idx]
    # Sort descending by level first, then NROT value
    order = np.lexsort((vals, lvls))[::-1]

    # Pre-flatten azshear for fast lookup
    _azsh_flat: Optional["np.ndarray"] = None
    if azshear_polar is not None:
        _azsh_flat = np.nan_to_num(azshear_polar, nan=0.0)

    out:  List[dict]           = []
    kept: List[Tuple[int,int]] = []

    for oi in order:
        r   = int(ray_idx[oi])
        b   = int(bin_idx[oi])
        lvl = int(lvls[oi])
        val = float(vals[oi])

        # Suppress duplicates within 6 rays × 10 bins of an already-kept couplet
        too_close = False
        for kr, kb in kept:
            dr = abs(r - kr)
            dr = min(dr, max(1, nrays - dr))  # azimuthal wrap-around
            if dr <= 6 and abs(b - kb) <= 10:
                too_close = True
                break
        if too_close:
            continue

        azimuth  = (r / max(1, nrays)) * 360.0
        range_km = rstart + (b + 0.5) * rscale
        lat, lon = _destination_point(station_lat, station_lon, azimuth, range_km)

        entry: dict = {
            "level":   lvl,
            "lat":     lat,
            "lon":     lon,
            "nrot":    round(val, 2),
            "rangeKm": round(range_km, 2),
        }
        # Include azimuthal shear at the couplet centre for UI display / alerting
        if _azsh_flat is not None:
            rr = min(r, _azsh_flat.shape[0] - 1)
            bb = min(b, _azsh_flat.shape[1] - 1)
            entry["azShear"] = round(float(_azsh_flat[rr, bb]), 2)

        out.append(entry)
        kept.append((r, b))
        if len(out) >= 30:
            break

    return out

def _bounds_from_range(lat: float, lon: float, max_range_km: float) -> List[List[float]]:
    delta_lat = max_range_km / 111.0
    delta_lon = max_range_km / (111.0 * max(0.3, abs(np.cos(np.radians(lat)))))
    return [
        [lon - delta_lon, lat + delta_lat],
        [lon + delta_lon, lat + delta_lat],
        [lon + delta_lon, lat - delta_lat],
        [lon - delta_lon, lat - delta_lat],
    ]


def _render_station_products(
    station_info: dict,
    *,
    dmi_target_utc: Optional[datetime] = None,
    dealias_unravel: bool = False,
) -> dict:
    provider = station_info.get("provider", "dmi").lower()

    if provider == "smhi":
        asset_url, datetime_value = _smhi_find_latest_qcvol_asset(station_info["area"])
        raw_bytes = _bytes_get(asset_url, {})
        file_bytes = _decompress_if_needed(raw_bytes)
        sweeps_by_qty, meta = _extract_all_sweeps(file_bytes)
        meta.setdefault("max_range_km", station_info.get("range_km", 240.0))
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 240.0)
        qty_source_file: Dict[str, str] = {
            q: asset_url.rstrip("/").split("/")[-1] for q in sweeps_by_qty
        }

    elif provider == "dwd":
        site_code = str(station_info.get("site", "")).strip().lower()
        if not site_code:
            raise RuntimeError("Missing DWD site code")

        sweeps_by_qty: Dict[str, List[dict]] = {}
        # Maps quantity string → source filename for display in the UI
        qty_source_file: Dict[str, str] = {}
        meta = {
          "lat": station_info.get("lat"),
          "lon": station_info.get("lon"),
          # Always use 180 km for DWD range rings regardless of file content
          "max_range_km": 180.0,
        }
        dts: List[str] = []

        for moment_key in ("reflectivity", "velocity", "correlation"):
            try:
                asset_url, dt_value = _dwd_find_latest_site_asset(site_code, moment_key)
                raw_bytes = _bytes_get(asset_url, {})
                file_bytes = _decompress_if_needed(raw_bytes)
                part_sweeps, part_meta = _extract_all_sweeps(file_bytes)
                fname = asset_url.rstrip("/").split("/")[-1]
                for qty, sweeps in part_sweeps.items():
                    sweeps_by_qty.setdefault(qty, []).extend(sweeps)
                    qty_source_file[qty] = fname
                # Update lat/lon from the actual file, but NOT max_range_km
                if part_meta.get("lat") is not None:
                    meta["lat"] = float(part_meta["lat"])
                if part_meta.get("lon") is not None:
                    meta["lon"] = float(part_meta["lon"])
                dts.append(dt_value)
            except Exception:
                continue

        if not sweeps_by_qty:
            raise RuntimeError(f"No DWD sweep volumes available for site {site_code}")

        datetime_value = max(dts) if dts else "n/a"

        # -Low-tilt preference 
        # DWD sweep volumes sometimes lead with a high-elevation tilt before
        # the 0.3°/0.5° scan is published.  Re-sort each quantity so that the
        # sweep closest to the target low angles comes first (used as the
        # default display tilt), while still keeping all available tilts so
        # the user can navigate up/down.  If no low-tilt scan exists the
        # lowest available angle is used instead of skipping the station.
        _DWD_PREFERRED_TILTS = (0.3, 0.5)
        _DWD_TILT_TOL = 0.2   # ± degrees tolerance when matching preferred tilt

        def _dwd_tilt_sort_key(sweep: dict) -> float:
            elev = sweep["elevation"]
            for pref in _DWD_PREFERRED_TILTS:
                if abs(elev - pref) <= _DWD_TILT_TOL:
                    return -1.0   # preferred tilt → sort to front
            return elev           # otherwise ascending by elevation

        for qty in sweeps_by_qty:
            sweeps_by_qty[qty].sort(key=_dwd_tilt_sort_key)

        _min_dwd_elev = min(
            (s["elevation"] for sweeps in sweeps_by_qty.values() for s in sweeps),
            default=999.0,
        )
        if _min_dwd_elev > max(_DWD_PREFERRED_TILTS) + _DWD_TILT_TOL:
            pass  # No preferred low-tilt scan; render with lowest available

    elif provider == "fmi":
        site_code = str(station_info.get("site", "")).strip().lower()
        if not site_code:
            raise RuntimeError("Missing FMI site code")
        asset_url, datetime_value = _fmi_find_latest_asset(site_code)
        raw_bytes = _bytes_get(asset_url, {})
        file_bytes = _decompress_if_needed(raw_bytes)
        sweeps_by_qty, meta = _extract_all_sweeps(file_bytes)
        meta.setdefault("max_range_km", station_info.get("range_km", 250.0))
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 250.0)
        fname = asset_url.rstrip("/").split("/")[-1]
        qty_source_file: Dict[str, str] = {q: fname for q in sweeps_by_qty}

    # -elif provider == 'fr_mf' DISABLED — SEDOO constantly failing
    # product = str(station_info.get("product", "")).strip()
    # if not product:
    # raise RuntimeError("Missing Meteo-France product code")
    # if not HAS_NETCDF4:
    # raise RuntimeError(f"Missing netCDF4 dependency: {NETCDF4_IMPORT_ERROR}")
    #
    # candidates = _fr_find_latest_asset(product)
    # raw_bytes: Optional[bytes] = None
    # asset_url = ""
    # datetime_value = ""
    # _fr_last_exc: Optional[Exception] = None
    #
    # for _fr_url, _fr_dt in candidates:
    # _is_monthly = ("day=" not in _fr_url and "hour=" not in _fr_url)
    # try:
    # print(f"[SEDOO] Trying ({'monthly tar' if _is_monthly else 'per-scan'}): {_fr_url}")
    # if _is_monthly:
    # # Stream the monthly TAR and extract the most recent NetCDF
    # from urllib.request import urlopen as _urlopen2
    # import tarfile as _tarfile
    # with _urlopen2(_fr_url, timeout=180) as _resp:
    # _buf = io.BytesIO(_resp.read())
    # with _tarfile.open(fileobj=_buf, mode="r:*") as _tf:
    # _nc_members = sorted(
    # [_m for _m in _tf.getmembers()
    # if _m.isfile() and re.search(r'\.nc(\.gz|\.bz2)?$', _m.name)],
    # key=lambda _m: _m.name,
    # )
    # if not _nc_members:
    # raise RuntimeError("No NetCDF files in SEDOO monthly TAR")
    # _latest = _nc_members[-1]
    # _f = _tf.extractfile(_latest)
    # if _f is None:
    # raise RuntimeError(f"Cannot extract {_latest.name} from TAR")
    # raw_bytes = _f.read()
    # print(f"[SEDOO] Extracted {_latest.name} ({len(raw_bytes)} bytes) from TAR")
    # else:
    # raw_bytes = _bytes_get(_fr_url, {})
    # print(f"[SEDOO] Per-scan success: {len(raw_bytes)} bytes")
    # asset_url = _fr_url
    # datetime_value = _fr_dt
    # break
    # except Exception as _exc:
    # from urllib.error import HTTPError as _HTTPError2
    # if isinstance(_exc, _HTTPError2):
    # try:
    # _body = _exc.read(512).decode("utf-8", errors="replace")
    # except Exception:
    # _body = "(unreadable)"
    # print(f"[SEDOO] HTTP {_exc.code} for {_fr_url} — body: {_body!r}")
    # else:
    # print(f"[SEDOO] Error for {_fr_url}: {_exc}")
    # _fr_last_exc = _exc
    # continue
    #
    # if raw_bytes is None:
    # raise RuntimeError(
    # f"All SEDOO candidates failed for '{product}': {_fr_last_exc}"
    # )
    # file_bytes = _decompress_if_needed(raw_bytes)
    # sweeps_by_qty, meta = _extract_fr_netcdf_sweeps(file_bytes)
    # meta.setdefault("max_range_km", station_info.get("range_km", 256.0))
    # if meta["max_range_km"] is None:
    # meta["max_range_km"] = station_info.get("range_km", 256.0)
    # if meta.get("lat") is None:
    # meta["lat"] = station_info.get("lat")
    # if meta.get("lon") is None:
    # meta["lon"] = station_info.get("lon")
    # fname = asset_url.rstrip("/").split("/")[-1] or f"{product}.nc"
    # qty_source_file: Dict[str, str] = {q: fname for q in sweeps_by_qty}
    #
    elif provider == "chmi":
        site_code = str(station_info.get("site", "")).strip().lower()
        if not site_code:
            raise RuntimeError("Missing CHMI site code")

        sweeps_by_qty: Dict[str, List[dict]] = {}
        qty_source_file: Dict[str, str] = {}
        meta = {
            "lat": station_info.get("lat"),
            "lon": station_info.get("lon"),
            "max_range_km": station_info.get("range_km", 256.0),
        }
        dts: List[str] = []

        for _chmi_moment, _chmi_var in CHMI_VAR_DIRS.items():
            if _chmi_moment not in ("reflectivity", "velocity", "correlation",
                                    "spectrum_width", "zdr"):
                continue
            try:
                asset_url, dt_value = _chmi_find_latest_asset(site_code, _chmi_var)
                raw_bytes = _bytes_get(asset_url, {})
                file_bytes = _decompress_if_needed(raw_bytes)
                part_sweeps, part_meta = _extract_all_sweeps(file_bytes)
                fname = asset_url.rstrip("/").split("/")[-1]
                for qty, sweeps in part_sweeps.items():
                    sweeps_by_qty.setdefault(qty, []).extend(sweeps)
                    qty_source_file[qty] = fname
                if part_meta.get("lat") is not None:
                    meta["lat"] = float(part_meta["lat"])
                if part_meta.get("lon") is not None:
                    meta["lon"] = float(part_meta["lon"])
                dts.append(dt_value)
            except Exception:
                continue

        if not sweeps_by_qty:
            raise RuntimeError(f"No CHMI sweep volumes available for site {site_code}")

        datetime_value = max(dts) if dts else "n/a"
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 256.0)

    elif provider == "geosphere":
        dataset_path = "radar_volumen_hochficht-v1-5min/filelisting"
        asset_url, datetime_value = _geosphere_find_latest_asset(dataset_path)
        raw_bytes = _geosphere_bytes_get(asset_url)
        file_bytes = _decompress_if_needed(raw_bytes)
        sweeps_by_qty, meta = _extract_all_sweeps(file_bytes)
        meta.setdefault("max_range_km", station_info.get("range_km", 200.0))
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 200.0)
        if meta.get("lat") is None:
            meta["lat"] = station_info.get("lat")
        if meta.get("lon") is None:
            meta["lon"] = station_info.get("lon")
        fname = asset_url.rstrip("/").split("/")[-1]
        qty_source_file: Dict[str, str] = {q: fname for q in sweeps_by_qty}

    elif provider == "shmu":
        radar_code = str(station_info.get("radar", "")).strip().lower()
        if not radar_code:
            raise RuntimeError("Missing SHMU radar code")

        sweeps_by_qty = {}
        qty_source_file = {}
        meta = {
            "lat": station_info.get("lat"),
            "lon": station_info.get("lon"),
            "max_range_km": station_info.get("range_km", 250.0),
        }
        dts: List[str] = []

        all_prod_dirs = _shmu_discover_products(radar_code)
        # Try to fetch up to 3 core moments and merge sweeps
        for moment_key in ("reflectivity", "velocity", "correlation"):
            pdir = _pick_shmu_product_dir(all_prod_dirs, moment_key)
            if not pdir and all_prod_dirs:
                # Fallback: just try the newest-looking directory (often contains multiple quantities)
                pdir = all_prod_dirs[-1]
            if not pdir:
                continue
            try:
                asset_url, dt_value = _shmu_find_latest_asset(radar_code, pdir)
                raw_bytes = _shmu_read_url(asset_url, timeout=120)
                file_bytes = _decompress_if_needed(raw_bytes)
                part_sweeps, part_meta = _extract_all_sweeps(file_bytes)
                fname = asset_url.rstrip("/").split("/")[-1]
                for qty, sweeps in part_sweeps.items():
                    sweeps_by_qty.setdefault(qty, []).extend(sweeps)
                    qty_source_file[qty] = fname
                if part_meta.get("lat") is not None:
                    meta["lat"] = float(part_meta["lat"])
                if part_meta.get("lon") is not None:
                    meta["lon"] = float(part_meta["lon"])
                dts.append(dt_value)
            except Exception:
                continue

        if not sweeps_by_qty:
            raise RuntimeError(f"No SHMU sweep volumes available for radar {radar_code}")

        datetime_value = max(dts) if dts else "n/a"
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 250.0)

    elif provider == "romania":
        site_code = str(station_info.get("site", "")).strip().upper()
        if not site_code:
            raise RuntimeError("Missing Romanian radar site code")

        sweeps_by_qty = {}
        qty_source_file = {}
        meta = {
            "lat": station_info.get("lat"),
            "lon": station_info.get("lon"),
            "max_range_km": station_info.get("range_km", 250.0),
        }
        dts: List[str] = []

        for moment_key in ("reflectivity", "velocity", "correlation", "spectrum_width", "zdr", "kdp"):
            try:
                asset_url, dt_value = _romania_find_latest_asset(site_code, moment_key)
                raw_bytes = _bytes_get(asset_url, {})
                file_bytes = _decompress_if_needed(raw_bytes)
                part_sweeps, part_meta = _extract_all_sweeps(file_bytes)
                fname = asset_url.rstrip("/").split("/")[-1]
                for qty, sweeps in part_sweeps.items():
                    sweeps_by_qty.setdefault(qty, []).extend(sweeps)
                    # Only record the source file for a quantity once – the
                    # first (most-specific) file wins.  Without setdefault a
                    # later multi-moment file (e.g. RhoHV.hdf also containing
                    # VRADH) would overwrite the correct velocity source.
                    qty_source_file.setdefault(qty, fname)
                if part_meta.get("lat") is not None:
                    meta["lat"] = float(part_meta["lat"])
                if part_meta.get("lon") is not None:
                    meta["lon"] = float(part_meta["lon"])
                dts.append(dt_value)
            except Exception:
                continue

        if not sweeps_by_qty:
            raise RuntimeError(f"No Romanian sweep volumes available for site {site_code}")

        datetime_value = max(dts) if dts else "n/a"
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 250.0)

    elif provider == "arpa_piemonte":
        # ARPA Piemonte (Italy) – Bric della Croce and Settepani C-band radars.
        # Each moment is stored as a separate HDF5 file in an Apache directory.
        # Moment→filename suffix mapping (subset of ARPA_PIE_MOMENTS):
        #   reflectivity  → dBZ   (clutter-filtered)
        #   velocity      → Vu
        #   correlation   → RhoHVu
        #   zdr           → ZDRu
        #   phidp         → uPhiDPu  (only on Settepani)
        site_dir = str(station_info.get("site", "")).strip().lower()
        if not site_dir:
            raise RuntimeError("Missing ARPA Piemonte site directory (bric or sett)")

        sweeps_by_qty: Dict[str, List[dict]] = {}
        qty_source_file: Dict[str, str] = {}
        meta = {
            "lat": station_info.get("lat"),
            "lon": station_info.get("lon"),
            "max_range_km": station_info.get("range_km", 200.0),
        }
        dts: List[str] = []

        # Map moment suffix → moment key used by _extract_all_sweeps / PRODUCTS
        arpa_moment_map = [
            ("dBZ",      "reflectivity"),
            ("Vu",       "velocity"),
            ("RhoHVu",   "correlation"),
            ("ZDRu",     "zdr"),
            ("uPhiDPu",  "phidp"),
        ]
        for _arpa_suffix, _arpa_moment in arpa_moment_map:
            try:
                asset_url, dt_value = _arpa_piemonte_find_asset(site_dir, _arpa_suffix)
                raw_bytes = _bytes_get(asset_url, {})
                file_bytes = _decompress_if_needed(raw_bytes)
                part_sweeps, part_meta = _extract_all_sweeps(file_bytes)
                fname = asset_url.rstrip("/").split("/")[-1]
                for qty, sweeps in part_sweeps.items():
                    sweeps_by_qty.setdefault(qty, []).extend(sweeps)
                    qty_source_file[qty] = fname
                if part_meta.get("lat") is not None:
                    meta["lat"] = float(part_meta["lat"])
                if part_meta.get("lon") is not None:
                    meta["lon"] = float(part_meta["lon"])
                dts.append(dt_value)
            except Exception:
                continue

        if not sweeps_by_qty:
            raise RuntimeError(
                f"No ARPA Piemonte sweep volumes available for site {site_dir!r}"
            )

        datetime_value = max(dts) if dts else "n/a"
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 200.0)

    elif provider == "arpa_lombardia":
        # ARPA Lombardia (Italy) – Desio and Flero C-band radars.
        # Files served at ARPA_LOM_BASE/{site_code}/ where site_code is DES or FLE.
        # New filename convention: {StationName}.{YYYYMMDD}T{HHMMSS}Z_{MOMENT}.h5.gz
        #   e.g. https://radarlive.arpalombardia.it/Volumi/DES/Desio.20250929T143000Z_DBZH.h5.gz
        # Each moment is a separate gzip-compressed HDF5 file; files appear every ~5 min.
        site_code = str(station_info.get("site", "")).strip().upper()
        if not site_code:
            raise RuntimeError("Missing ARPA Lombardia site code (DES or FLE)")

        sweeps_by_qty: Dict[str, List[dict]] = {}
        qty_source_file: Dict[str, str] = {}
        meta = {
            "lat": station_info.get("lat"),
            "lon": station_info.get("lon"),
            "max_range_km": station_info.get("range_km", 200.0),
        }
        dts: List[str] = []

        # ODIM moment names used in the new filename convention.
        # CLASS is a hydrometeor classification product — skip (not a standard radar qty).
        arpa_lom_moment_map = [
            ("DBZH",  "reflectivity"),
            ("TH",    "reflectivity"),   # total (unfiltered) reflectivity fallback
            ("VRADH", "velocity"),
            ("RHOHV", "correlation"),
            ("ZDR",   "zdr"),
            ("KDP",   "kdp"),
            ("PHIDP", "phidp"),
            ("WRADH", "spectrum_width"),
        ]
        for _arpa_suffix, _arpa_moment in arpa_lom_moment_map:
            try:
                asset_url, dt_value = _arpa_lombardia_find_asset(site_code, _arpa_suffix)
                raw_bytes = _bytes_get(asset_url, {})
                file_bytes = _decompress_if_needed(raw_bytes)
                part_sweeps, part_meta = _extract_all_sweeps(file_bytes)
                fname = asset_url.rstrip("/").split("/")[-1]
                for qty, sweeps in part_sweeps.items():
                    sweeps_by_qty.setdefault(qty, []).extend(sweeps)
                    qty_source_file[qty] = fname
                if part_meta.get("lat") is not None:
                    meta["lat"] = float(part_meta["lat"])
                if part_meta.get("lon") is not None:
                    meta["lon"] = float(part_meta["lon"])
                dts.append(dt_value)
            except Exception:
                continue

        if not sweeps_by_qty:
            raise RuntimeError(
                f"No ARPA Lombardia sweep volumes available for site {site_code!r}"
            )

        datetime_value = max(dts) if dts else "n/a"
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 200.0)

    elif provider == "estonia":
        item_id = station_info.get("item_id")
        if item_id is None:
            raise RuntimeError("Missing Estonia KAIA item ID")

        raw_bytes, datetime_value, fname = _estonia_fetch_file_by_item_id(int(item_id))
        file_bytes = _decompress_if_needed(raw_bytes)
        sweeps_by_qty, meta = _extract_all_sweeps(file_bytes)
        meta.setdefault("max_range_km", station_info.get("range_km", 250.0))
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 250.0)
        if meta.get("lat") is None:
            meta["lat"] = station_info.get("lat")
        if meta.get("lon") is None:
            meta["lon"] = station_info.get("lon")
        qty_source_file = {q: fname for q in sweeps_by_qty}

    elif provider == "meteogate_test":
        wigos_id = str(station_info.get("wigos_id", "")).strip()
        raw_bytes, datetime_value = _meteogate_test_fetch_raw(wigos_id)

        # This endpoint often responds with CoverageJSON or HTTP 204; only
        # ODIM/HDF bytes are usable in the current renderer.
        if raw_bytes[:1] in (b"{", b"["):
            try:
                parsed = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
                ptype = parsed.get("type", "CoverageJSON")
            except Exception:
                ptype = "CoverageJSON"
            raise RuntimeError(
                f"Meteogate test endpoint returned {ptype} instead of ODIM HDF5 for {wigos_id}"
            )

        file_bytes = _decompress_if_needed(raw_bytes)
        sweeps_by_qty, meta = _extract_all_sweeps(file_bytes)
        if not sweeps_by_qty:
            raise RuntimeError(
                f"No readable sweeps in Meteogate test response for WIGOS {wigos_id}"
            )
        meta.setdefault("max_range_km", station_info.get("range_km", 240.0))
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 240.0)
        if meta.get("lat") is None:
            meta["lat"] = station_info.get("lat")
        if meta.get("lon") is None:
            meta["lon"] = station_info.get("lon")
        qty_source_file = {q: f"{wigos_id}_test.h5" for q in sweeps_by_qty}

    elif provider == "knmi_cabauw":
        if not HAS_NETCDF4:
            raise RuntimeError(f"Missing netCDF4 dependency: {NETCDF4_IMPORT_ERROR}")
        file_bytes, datetime_value, fname = _cabauw_fetch_latest_netcdf()
        sweeps_by_qty, meta = _extract_cabauw_netcdf_sweeps(file_bytes)
        # Ensure station defaults if NetCDF lacks coordinates
        if meta.get("lat") is None:
            meta["lat"] = station_info.get("lat")
        if meta.get("lon") is None:
            meta["lon"] = station_info.get("lon")
        meta.setdefault("max_range_km", station_info.get("range_km", 60.0))
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 60.0)
        qty_source_file = {q: fname for q in sweeps_by_qty}

    elif provider == "knmi":
        dataset = str(station_info.get("dataset", "")).strip()
        version = str(station_info.get("version", "")).strip()
        if not dataset or not version:
            raise RuntimeError("Missing KNMI dataset/version")

        file_bytes, datetime_value, fname = _knmi_fetch_latest_volume(dataset, version)
        file_bytes = _decompress_if_needed(file_bytes)
        summary = _summarize_hdf_bytes(file_bytes)
        try:
            sweeps_by_qty, meta = _extract_knmi_sweeps(file_bytes)
        except Exception:
            sweeps_by_qty, meta = {}, {"lat": None, "lon": None, "max_range_km": None}
        if not sweeps_by_qty:
            try:
                sweeps_by_qty, meta = _extract_all_sweeps(file_bytes)
            except Exception:
                sweeps_by_qty, meta = {}, {"lat": None, "lon": None, "max_range_km": None}
        if not sweeps_by_qty:
            raise RuntimeError(f"KNMI file has no readable sweeps | {summary}")
        meta.setdefault("max_range_km", station_info.get("range_km", 320.0))
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 320.0)
        if meta.get("lat") is None:
            meta["lat"] = station_info.get("lat")
        if meta.get("lon") is None:
            meta["lon"] = station_info.get("lon")
        qty_source_file: Dict[str, str] = {q: fname for q in sweeps_by_qty}

    elif provider == "meteogate":
        wigos_id = str(station_info.get("wigos_id", "")).strip()
        if not wigos_id:
            raise RuntimeError("Missing Meteogate WIGOS station identifier")

        raw_bytes, datetime_value = _meteogate_fetch_pvol(wigos_id)
        file_bytes = _decompress_if_needed(raw_bytes)
        sweeps_by_qty, meta = _extract_all_sweeps(file_bytes)

        if not sweeps_by_qty:
            raise RuntimeError(
                f"No readable sweeps in Meteogate PVOL for WIGOS {wigos_id}"
            )

        meta.setdefault("max_range_km", station_info.get("range_km", 240.0))
        if meta["max_range_km"] is None:
            meta["max_range_km"] = station_info.get("range_km", 240.0)
        if meta.get("lat") is None:
            meta["lat"] = station_info.get("lat")
        if meta.get("lon") is None:
            meta["lon"] = station_info.get("lon")

        qty_source_file: Dict[str, str] = {
            q: f"{wigos_id}_PVOL.h5" for q in sweeps_by_qty
        }

    elif provider == "custom":
        # -Custom file upload
        # file_bytes stored directly in station_info; dispatch to the best matching
        # reader (ODIM/KNMI HDF5, NetCDF, IRIS/Sigmet RAW*, Furuno, BUFR).
        fname = str(station_info.get("filename", "custom.h5"))
        datetime_value = str(station_info.get("datetime", "n/a"))
        raw_file_bytes = station_info.get("file_bytes")
        if not raw_file_bytes:
            raise RuntimeError("Custom radar: no file bytes in station info")
        file_bytes = _decompress_if_needed(raw_file_bytes)
        try:
            sweeps_by_qty, meta = _extract_custom_sweeps(file_bytes, fname)
        except Exception as _e:
            raise RuntimeError(f"Custom radar: no readable sweeps. {_e}") from _e
        if meta.get("lat") is None:
            meta["lat"] = station_info.get("lat")
        if meta.get("lon") is None:
            meta["lon"] = station_info.get("lon")
        if meta.get("max_range_km") is None:
            meta["max_range_km"] = station_info.get("range_km", 200.0)
        qty_source_file: Dict[str, str] = {q: fname for q in sweeps_by_qty}

        # -Merge sweeps from optional per-product extra file
        # Each entry is (raw_bytes, filename). Quantities found in extra files
        # are merged into sweeps_by_qty; existing quantities are NOT overwritten
        # so the primary file always wins for shared quantities.
        for _extra_raw, _extra_fname in (station_info.get("extra_product_files") or []):
            try:
                _extra_bytes = _decompress_if_needed(_extra_raw)
                try:
                    _extra_sweeps, _ = _extract_custom_sweeps(_extra_bytes, _extra_fname)
                except Exception:
                    _extra_sweeps = {}
                for _qty, _sw_list in _extra_sweeps.items():
                    if _qty not in sweeps_by_qty:
                        sweeps_by_qty[_qty] = _sw_list
                        qty_source_file[_qty] = _extra_fname
            except Exception as _merge_exc:
                print(f"[custom radar] Could not merge extra file '{_extra_fname}': {_merge_exc}")

    else:
        if dmi_target_utc is not None:
            asset_url, datetime_value = _find_dmi_volume_asset_near_datetime(
                station_info["code"], dmi_target_utc
            )
        else:
            asset_url, datetime_value = _find_latest_volume_asset(station_info["code"])
        file_bytes = _decompress_if_needed(_bytes_get(asset_url, {"api-key": DMI_API_KEY}))
        sweeps_by_qty, meta = _extract_all_sweeps(file_bytes)
        qty_source_file: Dict[str, str] = {
            q: asset_url.rstrip("/").split("/")[-1] for q in sweeps_by_qty
        }

    # ── Velocity dealiasing (user toggle or KNMI/DMI dual-PRF auto-path) ──────
    if dealias_unravel:
        _apply_velocity_dealias(sweeps_by_qty)
    elif provider in ("knmi", "dmi"):
        _apply_velocity_dual_prf_nlr_only(sweeps_by_qty)

    if meta["max_range_km"] is None or meta["lat"] is None or meta["lon"] is None:
        raise RuntimeError("Could not derive radar coverage bounds")

    refl_qty = next((q for q in ("DBZHC", "DBZH", "TH", "DBTH") if q in sweeps_by_qty), None)
    vel_qty = next((q for q in ("VRADC", "VRADH", "VRAD") if q in sweeps_by_qty), None)

    products: Dict[str, Optional[dict]] = {}

    _knmi_qmap = {
        "reflectivity": ["scan_Z_data", "scan_uZ_data"],
        "velocity": ["scan_V_data", "scan_Vv_data"],
        "correlation": ["scan_RhoHV_data", "scan_CCOR_data", "scan_CCORv_data"],
        "spectrum_width": ["scan_W_data", "scan_Wv_data"],
        "kdp": ["scan_KDP_data"],
        # zdr handled later if both Z and Zv exist
    }

    for pk in ("reflectivity", "velocity", "correlation", "spectrum_width", "zdr", "kdp"):
        quantities = PRODUCTS[pk]["quantities"]
        matched = _match_quantity(quantities, sweeps_by_qty)
        if matched is None:
            products[pk] = None
            continue
        sweeps = sweeps_by_qty[matched]
        overlay_data = _encode_sweep_for_overlay(sweeps[0])
        cs_sweeps = []
        for sw in sweeps[:MAX_CS_SWEEPS]:
            enc = _encode_sweep_for_cs(sw["_polar"], sw["rscale_km"])
            if enc:
                cs_sweeps.append(
                    {
                        "elevation": round(sw["elevation"], 3),
                        "rstartKm": sw["rstart_km"],
                        "maxRangeKm": sw["max_range_km"],
                        **enc,
                    }
                )
        # Encode ALL tilts so the front-end can navigate up/down through elevations
        all_tilt_overlays: List[dict] = []
        for sw in sweeps:
            od = _encode_sweep_for_overlay(sw)
            if od:
                all_tilt_overlays.append(od)
        products[pk] = {
            "type": "polar",
            "overlayData": overlay_data,
            "allTiltOverlays": all_tilt_overlays,
            "sweeps": cs_sweeps,
            "sourceFile": qty_source_file.get(matched, ""),
            "matchedQty": matched,
        }

    if provider == "knmi":
        base_keys = ("reflectivity", "velocity", "correlation", "spectrum_width", "zdr", "kdp")
        if not any(products.get(k) for k in base_keys):
            pass  # No base product matched; quantities already in errors

    def _overlay_from_polar_array(arr: "np.ndarray", template_sw: dict) -> Optional[dict]:
        if arr is None:
            return None
        sweep = {
            "elevation": template_sw["elevation"],
            "rscale_km": template_sw["rscale_km"],
            "rstart_km": template_sw["rstart_km"],
            "max_range_km": template_sw["max_range_km"],
            "nrays": template_sw["nrays"],
            "nbins": template_sw["nbins"],
            "_polar": arr,
        }
        return _encode_sweep_for_overlay(sweep)

    def _derived_polar(overlay_data: Optional[dict]) -> Optional[dict]:
        if not overlay_data:
            return None
        return {
            "type": "polar",
            "overlayData": overlay_data,
            "sweeps": [],
        }

    if provider == "knmi":
        z = None
        zv = None
        # After qty mapping, horizontal Z is stored as "DBZH", vertical as "DBZV"
        _z_key  = next((q for q in ("DBZH",  "TH",  "scan_Z_data",  "scan_uZ_data")
                        if q in sweeps_by_qty), None)
        _zv_key = next((q for q in ("DBZV",  "TV",  "scan_Zv_data", "scan_uZv_data")
                        if q in sweeps_by_qty), None)
        if _z_key:
            z    = sweeps_by_qty[_z_key][0]["_polar"]
            z_sw = sweeps_by_qty[_z_key][0]
        else:
            z_sw = None
        if _zv_key:
            zv = sweeps_by_qty[_zv_key][0]["_polar"]
        if z is not None and zv is not None and z_sw is not None:
            common_rays = min(z.shape[0], zv.shape[0])
            common_bins = min(z.shape[1], zv.shape[1])
            zdr = z[:common_rays, :common_bins] - zv[:common_rays, :common_bins]
            zdr_overlay = _overlay_from_polar_array(zdr, z_sw)
            products["zdr"] = _derived_polar(zdr_overlay)

    echo18_overlay = echo0_overlay = None
    tops18_polar = tops0_polar = None
    nrot_overlay = srv_overlay = meso_overlay = azshear_overlay = mesh_overlay = None
    tvs_markers: List[dict] = []

    if refl_qty is not None:
        refl_sweeps = sweeps_by_qty[refl_qty]
        base_refl_sw = refl_sweeps[0]
        tops18_polar = _compute_echo_tops_km(refl_sweeps, min_dbz=18.0)
        if tops18_polar is not None:
            echo18_overlay = _overlay_from_polar_array(tops18_polar, base_refl_sw)
        tops0_polar = _compute_echo_tops_km(refl_sweeps, min_dbz=0.0)
        if tops0_polar is not None:
            echo0_overlay = _overlay_from_polar_array(tops0_polar, base_refl_sw)

        mesh_polar = _compute_mesh_polar(refl_sweeps)
        if mesh_polar is not None:
            mesh_overlay = _overlay_from_polar_array(mesh_polar, base_refl_sw)

    if vel_qty is not None:
        vel_sweeps = sweeps_by_qty[vel_qty]
        base_vel_sw = vel_sweeps[0]
        vel_polar0 = base_vel_sw["_polar"]

        nrot_polar, shear, _, _, mask_rot = _compute_nrot_polar(
            vel_polar0, pair_threshold=5.0
        )
        nrot_overlay = _overlay_from_polar_array(nrot_polar, base_vel_sw)

        srv_polar = _compute_srv_polar(base_vel_sw, STORM_MOTION)
        srv_overlay = _overlay_from_polar_array(srv_polar, base_vel_sw)

        if refl_qty is not None:
            refl_sweeps = sweeps_by_qty[refl_qty]
            refl_polar0 = refl_sweeps[0]["_polar"]

            common_rays = min(nrot_polar.shape[0], refl_polar0.shape[0])
            common_bins = min(nrot_polar.shape[1], refl_polar0.shape[1])
            nrot_sub = nrot_polar[:common_rays, :common_bins]
            mask_sub = mask_rot[:common_rays, :common_bins]
            refl_sub = refl_polar0[:common_rays, :common_bins]
            shear_sub = shear[:common_rays, :common_bins]

            # Vertical continuity: require rotation support in one or two higher tilts.
            continuity = np.zeros((common_rays, common_bins), dtype=np.float32)
            continuity_count = 0
            for upper_sw in vel_sweeps[1:3]:
                upper_nrot, _, _, _, _ = _compute_nrot_polar(
                    upper_sw["_polar"], pair_threshold=5.0
                )
                ur = min(common_rays, upper_nrot.shape[0])
                ub = min(common_bins, upper_nrot.shape[1])
                if ur <= 0 or ub <= 0:
                    continue
                continuity[:ur, :ub] += (
                    np.isfinite(upper_nrot[:ur, :ub]) & (upper_nrot[:ur, :ub] >= 20.0)
                ).astype(np.float32)
                continuity_count += 1
            if continuity_count > 0:
                continuity /= float(continuity_count)
            else:
                continuity[:] = 0.0

            tops18_sub = None
            if tops18_polar is not None:
                tr = min(common_rays, tops18_polar.shape[0])
                tb = min(common_bins, tops18_polar.shape[1])
                tops18_sub = np.full((common_rays, common_bins), np.nan, dtype=np.float32)
                tops18_sub[:tr, :tb] = tops18_polar[:tr, :tb]

            # -Azimuthal shear (×10⁻³ s⁻¹)
            # Computed BEFORE meso/TVS so it can be passed as a bonus input to
            # both _compute_meso_probability and _compute_tvs_markers.
            azshear_polar = _compute_azimuthal_shear_polar(
                vel_polar0[:common_rays, :common_bins],
                rstart_km=float(base_vel_sw.get("rstart_km", 0.0)),
                rscale_km=float(base_vel_sw.get("rscale_km", 1.0)),
            )

            azshear_overlay = _overlay_from_polar_array(azshear_polar, base_vel_sw)

            # Push to per-station history ring buffer for composite requests
            _azshear_meta = {
                "rstart_km": float(base_vel_sw.get("rstart_km", 0.0)),
                "rscale_km": float(base_vel_sw.get("rscale_km", 1.0)),
                "nrays":     int(base_vel_sw.get("nrays", azshear_polar.shape[0])),
                "nbins":     int(base_vel_sw.get("nbins", azshear_polar.shape[1])),
                "max_range_km": float(base_vel_sw.get("max_range_km", 250.0)),
                "elevation": float(base_vel_sw.get("elevation", 0.5)),
            }
            with _AZSHEAR_HISTORY_LOCK:
                _stn_key = station_info.get("_name", "") or station_info.get("code", "unknown")
                hist = _AZSHEAR_HISTORY.setdefault(_stn_key, [])
                hist.insert(0, {"polar": azshear_polar, "meta": _azshear_meta})
                if len(hist) > _AZSHEAR_HISTORY_MAXLEN:
                    hist[_AZSHEAR_HISTORY_MAXLEN:] = []

            # -Mesocyclone probability
            # azshear_polar is now available as a tight-couplet bonus signal.
            meso_polar = _compute_meso_probability(
                nrot_sub,
                shear_sub,
                refl_sub,
                continuity=continuity,
                azshear_polar=azshear_polar,
            )
            meso_overlay = _overlay_from_polar_array(meso_polar, base_vel_sw)

            # -TVS / mesocyclone markers 
            # azshear_polar is passed for level-upgrade and output annotation.
            tvs_markers = _compute_tvs_markers(
                nrot_sub,
                base_vel_sw,
                float(meta["lat"]),
                float(meta["lon"]),
                refl_sub,
                azshear_polar=azshear_polar,
            )

    products["echo_tops"] = _derived_polar(echo18_overlay)
    products["echo_tops_0"] = _derived_polar(echo0_overlay)
    products["nrot"] = _derived_polar(nrot_overlay)
    products["srv"] = _derived_polar(srv_overlay)
    products["meso"] = _derived_polar(meso_overlay)
    products["azimuthal_shear"] = _derived_polar(azshear_overlay)
    products["mesh"] = _derived_polar(mesh_overlay)

    # TVS markers are a separate overlay (not a selectable product)
    _tvs_payload = {"tvsMarkers": tvs_markers} if tvs_markers else {"tvsMarkers": []}

    debug_info = None
    if provider == "knmi":
        stats = {}
        for q, sweeps in sweeps_by_qty.items():
            if not sweeps:
                continue
            arr = sweeps[0]["_polar"]
            finite = np.isfinite(arr)
            stats[q] = {
                "finite": int(np.count_nonzero(finite)),
                "total": int(arr.size),
                "min": float(np.nanmin(arr)) if np.any(finite) else None,
                "max": float(np.nanmax(arr)) if np.any(finite) else None,
            }
        debug_info = {"quantities": sorted(sweeps_by_qty.keys()), "stats": stats}

    return {
        "datetime": datetime_value,
        "bounds": _bounds_from_range(
            float(meta["lat"]),
            float(meta["lon"]),
            float(meta["max_range_km"]),
        ),
        "products": products,
        "tvsData": _tvs_payload,
        "stationMeta": {
            "lat": float(meta["lat"]),
            "lon": float(meta["lon"]),
            "range_km": float(meta["max_range_km"]),
        },
        "debug": debug_info,
    }

# DWD synoptic surface observations  (server-side; bz2-compressed GeoJSON)
_DWD_METOBS_CACHE: Optional[List[dict]] = None
_DWD_METOBS_CACHE_TIME: float = 0.0
_DWD_METOBS_TTL = 15 * 60  # seconds

def _fetch_dwd_metobs() -> List[dict]:
    """Return a list of station dicts [{sid, name, lon, lat, temp, dewpt, wspd, wdir}].

    Downloads the latest full-hour SYNOP GeoJSON (bz2) from DWD open data,
    decompresses it server-side, and extracts the four fields we need.
    The entire function is exception-safe and returns [] on any failure.

    DWD BUFR-to-GeoJSON property names (checked against actual files):
      stationOrSiteName | airTemperature (K) | dewpointTemperature (K)
      windSpeed (m/s)   | windDirectionFromWhichBlowing (°)
    """
    global _DWD_METOBS_CACHE, _DWD_METOBS_CACHE_TIME
    now = time.time()
    if _DWD_METOBS_CACHE is not None and (now - _DWD_METOBS_CACHE_TIME) < _DWD_METOBS_TTL:
        return _DWD_METOBS_CACHE

    try:
        # -Parse directory listing 
        dir_html = urlopen(DWD_METOBS_BASE, timeout=15).read().decode("utf-8", errors="replace")

        # Apache autoindex: filenames may have commas URL-encoded as %2C in hrefs.
        candidates = []
        href_re = re.compile(
            r'href="([^"]*\.geojson\.bz2)"',
            re.IGNORECASE,
        )
        size_re = re.compile(r'(\d+)\s*$')
        sizek_re = re.compile(r'([\d.]+)K\s*$')

        for m in href_re.finditer(dir_html):
            raw_name = m.group(1)
            ts_m = re.search(r'(\d{14})', raw_name)
            if not ts_m:
                continue
            ts14 = ts_m.group(1)
            line_rest = dir_html[m.end():m.end() + 200]
            size = 0
            sm = sizek_re.search(line_rest)
            if sm:
                size = int(float(sm.group(1)) * 1024)
            else:
                sm2 = size_re.search(line_rest.split('\n')[0])
                if sm2:
                    size = int(sm2.group(1))
            fname = raw_name.replace('%2C', ',').replace('%2c', ',')
            candidates.append((ts14, fname, size))

        # Prefer full-hour synoptic dumps (minute field ≤ 02) over partials.
        full_hour = [(ts, fn, sz) for ts, fn, sz in candidates
                     if int(ts[10:12]) <= 2 and sz >= 200_000]
        if not full_hour:
            full_hour = [(ts, fn, sz) for ts, fn, sz in candidates if sz >= 200_000]
        if not full_hour and candidates:
            full_hour = candidates

        if not full_hour:
            return []

        full_hour.sort(key=lambda x: (x[0], x[2]), reverse=True)
        latest_fname = full_hour[0][1]
        url = DWD_METOBS_BASE + latest_fname

        # -Download + decompress 
        raw       = urlopen(url, timeout=30).read()
        data_bytes = bz2.decompress(raw)
        fc         = json.loads(data_bytes.decode("utf-8"))

    except Exception as exc:
        raise RuntimeError(f"[dwd-metobs] fetch/decompress failed: {exc}") from exc

    # -Parse GeoJSON features 
    stations_out: List[dict] = []
    _feats = fc.get("features", [])
    try:
        for feat in fc.get("features", []):
            props  = feat.get("properties") or {}
            geom   = feat.get("geometry")   or {}
            coords = geom.get("coordinates")
            if not coords or len(coords) < 2:
                continue
            try:
                lon, lat = float(coords[0]), float(coords[1])
            except (TypeError, ValueError):
                continue

            # -Temperature (Kelvin → °C) 
            def _k2c(key: str) -> Optional[float]:
                v = props.get(key)
                if v is None:
                    return None
                try:
                    f = float(v)
                    # Values > 100 are almost certainly Kelvin
                    return round(f - 273.15, 1) if f > 100 else round(f, 1)
                except (TypeError, ValueError):
                    return None

            def _fval(*keys: str) -> Optional[float]:
                for k in keys:
                    v = props.get(k)
                    if v is not None:
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            pass
                return None

            # DWD BUFR GeoJSON uses camelCase; try several known aliases
            temp  = (_k2c("airTemperature") or
                     _k2c("air_temperature") or
                     _k2c("temperature"))
            dewpt = (_k2c("dewpointTemperature") or
                     _k2c("dewpoint_temperature") or
                     _k2c("dewPointTemperature") or
                     _k2c("dew_point_temperature"))
            wspd  = _fval("windSpeed", "wind_speed", "FF")
            wdir_f = _fval("windDirectionFromWhichBlowing", "windDirection",
                           "wind_direction", "wind_from_direction", "DD")
            wdir  = round(wdir_f) if wdir_f is not None else None
            name  = str(props.get("stationOrSiteName") or
                        props.get("station_or_site_name") or
                        props.get("name") or "").strip()

            # Derive dew point from RH when direct value absent
            if dewpt is None and temp is not None:
                rh_raw = _fval("relativeHumidity")
                if rh_raw is not None and 0 < rh_raw <= 100:
                    a_, b_ = 17.27, 237.3
                    g = (a_ * temp) / (b_ + temp) + (rh_raw / 100.0)
                    dewpt = round((b_ * g) / (a_ - g), 1)

            if temp is None and dewpt is None and wspd is None:
                continue  # no useful obs at this station

            stations_out.append({
                "sid":   str(props.get("stationNumber") or props.get("stationIdentification") or ""),
                "name":  name,
                "lon":   round(lon, 4),
                "lat":   round(lat, 4),
                "temp":  temp,
                "dewpt": dewpt,
                "wspd":  wspd,
                "wdir":  wdir,
            })
    except Exception:
        return []

    _DWD_METOBS_CACHE      = stations_out
    _DWD_METOBS_CACHE_TIME = now
    return stations_out

def _safe_fetch_dwd_metobs() -> List[dict]:
    """Exception-safe wrapper — returns [] on any unexpected error."""
    try:
        return _fetch_dwd_metobs()
    except Exception:
        return []

# Norway Frost API  (server-side)
_FROST_METOBS_CACHE: Optional[List[dict]] = None
_FROST_METOBS_CACHE_TIME: float = 0.0
_FROST_METOBS_TTL = 10 * 60  # seconds

def _fetch_frost_metobs() -> List[dict]:
    """Fetch latest surface observations for Norwegian stations via the Frost API.

    Returns list of {sid, name, lon, lat, temp, dewpt, wspd, wdir}.
    Requires FROST_CLIENT_ID to be set; returns [] silently if empty.
    """
    global _FROST_METOBS_CACHE, _FROST_METOBS_CACHE_TIME
    if not FROST_CLIENT_ID:
        return []
    now = time.time()
    if _FROST_METOBS_CACHE is not None and (now - _FROST_METOBS_CACHE_TIME) < _FROST_METOBS_TTL:
        return _FROST_METOBS_CACHE

    import base64 as _b64
    import urllib.request as _req
    from urllib.error import URLError as _URLError

    auth = _b64.b64encode(f"{FROST_CLIENT_ID}:".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Accept": "application/json"}

    def _frost_get(url: str) -> dict:
        request = _req.Request(url, headers=headers)
        with _req.urlopen(request, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # 1. Sources: all Norwegian SensorSystem stations with geometry
    try:
        src_url = (
            "https://frost.met.no/sources/v0.jsonld"
            "?types=SensorSystem&country=NO&fields=id,name,geometry"
        )
        src_data = _frost_get(src_url)
    except Exception:
        return []

    sources: Dict[str, dict] = {}
    for s in src_data.get("data", []):
        coords = (s.get("geometry") or {}).get("coordinates")
        if coords and len(coords) >= 2:
            # id might be "SN18700:0" — strip sensor suffix
            sid = str(s.get("id", "")).split(":")[0]
            sources[sid] = {
                "lon": float(coords[0]), "lat": float(coords[1]),
                "name": str(s.get("name") or sid),
            }

    # 2. Observations: latest values for T, Td, wind.
    # Frost v0 does not reliably accept sources=SN* wildcard or referencetime=latest
    # when combined with an elements filter — returns HTTP 400.
    # Workaround: query a 3-hour window ending now (ISO 8601 interval).
    # The API returns all observations in that window; we keep the newest per station.
    from datetime import datetime, timezone, timedelta
    _now = datetime.now(timezone.utc)
    _from = (_now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M")
    _to   = _now.strftime("%Y-%m-%dT%H:%M")
    try:
        obs_url = (
            "https://frost.met.no/observations/v0.jsonld"
            f"?referencetime={_from}/{_to}"
            "&elements=air_temperature,dew_point_temperature,wind_speed,wind_from_direction"
            "&timeresolutions=PT1H,PT10M"
        )
        obs_data = _frost_get(obs_url)
    except Exception:
        return []

    # obs_map: sid - {element - (referenceTime, value)}
    # Keep only the most-recent observation per station per element.
    obs_map: Dict[str, dict] = {}
    for series in obs_data.get("data", []):
        sid = str(series.get("sourceId", "")).split(":")[0]
        ref_t = series.get("referenceTime", "")
        if sid not in obs_map:
            obs_map[sid] = {}
        for ob in (series.get("observations") or []):
            eid = ob.get("elementId", "")
            if eid not in ("air_temperature", "dew_point_temperature",
                           "wind_speed", "wind_from_direction"):
                continue
            try:
                v = float(ob["value"])
            except (KeyError, TypeError, ValueError):
                continue
            # Keep the newer timestamp
            prev_t, _ = obs_map[sid].get(eid, ("", None))
            if ref_t >= prev_t:
                obs_map[sid][eid] = (ref_t, v)

    stations_out: List[dict] = []
    for sid, elems in obs_map.items():
        src = sources.get(sid)
        if not src:
            continue
        def _v(eid): return round(elems[eid][1], 1) if eid in elems else None
        temp  = _v("air_temperature")
        dewpt = _v("dew_point_temperature")
        wspd  = _v("wind_speed")
        wdir  = round(elems["wind_from_direction"][1]) if "wind_from_direction" in elems else None
        if not any(x is not None for x in (temp, dewpt, wspd)):
            continue
        stations_out.append({
            "sid":  sid,
            "name": src["name"],
            "lon":  round(src["lon"], 4),
            "lat":  round(src["lat"], 4),
            "temp":  temp,
            "dewpt": dewpt,
            "wspd":  wspd,
            "wdir":  wdir,
        })

    _FROST_METOBS_CACHE = stations_out
    _FROST_METOBS_CACHE_TIME = now
    return stations_out

def _safe_fetch_frost_metobs() -> List[dict]:
    try:
        return _fetch_frost_metobs()
    except Exception:
        return []

# KNMI in-situ meteorological observations  (server-side)
_KNMI_METOBS_CACHE: Optional[List[dict]] = None
_KNMI_METOBS_CACHE_TIME: float = 0.0
_KNMI_METOBS_TTL = 12 * 60  # seconds

_GEOSPHERE_METOBS_CACHE: Optional[List[dict]] = None
_GEOSPHERE_METOBS_CACHE_TIME: float = 0.0
_GEOSPHERE_METOBS_TTL = 10 * 60  # seconds – matches 10-min update cadence

def _fetch_knmi_metobs() -> List[dict]:
    """Fetch latest 10-min in-situ observations from KNMI Open Data.

    Downloads the most recent NetCDF4 file from the
    '10-minute-in-situ-meteorological-observations' dataset and extracts
    temperature, dew-point, wind speed and direction for each AWS station.
    Returns list of {sid, name, lon, lat, temp, dewpt, wspd, wdir}.
    """
    global _KNMI_METOBS_CACHE, _KNMI_METOBS_CACHE_TIME
    now = time.time()
    if _KNMI_METOBS_CACHE is not None and (now - _KNMI_METOBS_CACHE_TIME) < _KNMI_METOBS_TTL:
        return _KNMI_METOBS_CACHE

    DATASET  = "10-minute-in-situ-meteorological-observations"
    VERSION  = "1.0"

    try:
        listing = _knmi_json_get(
            f"/datasets/{DATASET}/versions/{VERSION}/files",
            {"maxKeys": 1, "orderBy": "created", "sorting": "desc"},
        )
        files = listing.get("files") or []
        if not files:
            return []
        fname = str(files[0].get("filename") or "").strip()
        if not fname:
            return []

        url_info = _knmi_json_get(
            f"/datasets/{DATASET}/versions/{VERSION}/files/{fname}/url",
            {},
        )
        download_url = str(url_info.get("temporaryDownloadUrl") or "").strip()
        if not download_url:
            return []

        file_bytes = _bytes_get(download_url, {})
        file_bytes = _decompress_if_needed(file_bytes)
    except Exception:
        return []

    # Parse the NetCDF4 file with h5py
    stations_out: List[dict] = []
    try:
        with h5py.File(io.BytesIO(file_bytes), "r") as nc:
            # Helper to load a dataset or return None
            def _nc_arr(key: str) -> Optional["np.ndarray"]:
                try:
                    obj = nc.get(key)
                    if obj is None:
                        return None
                    return np.asarray(obj[:])
                except Exception:
                    return None

            def _nc_str(key: str) -> Optional[List[str]]:
                try:
                    obj = nc.get(key)
                    if obj is None:
                        return None
                    raw = np.asarray(obj[:])
                    if raw.ndim == 0:
                        return [str(raw)]
                    result = []
                    for row in raw:
                        if isinstance(row, (bytes, np.bytes_)):
                            result.append(row.decode("utf-8", errors="replace").strip())
                        elif hasattr(row, "tobytes"):
                            result.append(bytes(row).decode("utf-8", errors="replace").strip().rstrip("\x00"))
                        else:
                            result.append(str(row).strip())
                    return result
                except Exception:
                    return None

            lats      = _nc_arr("lat")   or _nc_arr("latitude")
            lons      = _nc_arr("lon")   or _nc_arr("longitude")
            names     = _nc_str("station_name") or _nc_str("stationname") or _nc_str("name")
            sids_raw  = _nc_arr("station") or _nc_arr("stid") or _nc_arr("station_id")

            # Temperature – try multiple CF/KNMI naming conventions
            temp_arr  = (_nc_arr("ta")  or _nc_arr("T")   or _nc_arr("air_temperature")
                         or _nc_arr("T2m"))
            # Dew-point
            dewp_arr  = (_nc_arr("td")  or _nc_arr("Td")  or _nc_arr("dew_point_temperature")
                         or _nc_arr("D"))
            # Wind speed
            ff_arr    = (_nc_arr("ff")  or _nc_arr("F")   or _nc_arr("wind_speed")
                         or _nc_arr("wsp"))
            # Wind direction
            dd_arr    = (_nc_arr("dd")  or _nc_arr("D")   or _nc_arr("wind_from_direction")
                         or _nc_arr("wdir"))

            if lats is None or lons is None:
                return []

            n_stations = lats.size

            def _last_obs(arr: Optional["np.ndarray"]) -> Optional["np.ndarray"]:
                """Return the most recent time slice from a (time, station) or (station,) array."""
                if arr is None:
                    return None
                if arr.ndim == 2:
                    # Assume (time, station); take last time step
                    return arr[-1, :]
                if arr.ndim == 1 and arr.size == n_stations:
                    return arr
                return None

            temp_vals = _last_obs(temp_arr)
            dewp_vals = _last_obs(dewp_arr)
            ff_vals   = _last_obs(ff_arr)
            dd_vals   = _last_obs(dd_arr)

            def _fv(arr: Optional["np.ndarray"], i: int) -> Optional[float]:
                if arr is None or i >= arr.size:
                    return None
                v = float(arr.flat[i])
                if not np.isfinite(v) or abs(v) > 9998:
                    return None
                # Convert Kelvin → Celsius if necessary
                if abs(v) > 100:
                    v -= 273.15
                return round(v, 1)

            for i in range(n_stations):
                lat = float(lats.flat[i]) if i < lats.size else None
                lon = float(lons.flat[i]) if i < lons.size else None
                if lat is None or lon is None or not np.isfinite(lat) or not np.isfinite(lon):
                    continue
                name = names[i] if (names and i < len(names)) else f"NL-{i}"
                sid  = str(int(sids_raw.flat[i])) if sids_raw is not None and i < sids_raw.size else str(i)

                temp  = _fv(temp_vals, i)
                dewpt = _fv(dewp_vals, i)
                wspd  = _fv(ff_vals,   i)
                wdir_v = _fv(dd_vals,  i)
                wdir  = round(wdir_v) if wdir_v is not None else None

                if temp is None and dewpt is None and wspd is None:
                    continue

                stations_out.append({
                    "sid":   sid,
                    "name":  name,
                    "lon":   round(lon, 4),
                    "lat":   round(lat, 4),
                    "temp":  temp,
                    "dewpt": dewpt,
                    "wspd":  wspd,
                    "wdir":  wdir,
                })
    except Exception:
        return []

    _KNMI_METOBS_CACHE      = stations_out
    _KNMI_METOBS_CACHE_TIME = now
    return stations_out


def _safe_fetch_knmi_metobs() -> List[dict]:
    """Exception-safe wrapper — returns [] on any unexpected error."""
    try:
        return _fetch_knmi_metobs()
    except Exception:
        return []


# GeoSphere Austria – TAWES klima-v2-10min current observations
def _fetch_geosphere_metobs() -> List[dict]:
    """Fetch latest 10-min surface observations from GeoSphere Austria.

    Calls the public, no-auth Dataset API endpoint:
      https://dataset.api.hub.geosphere.at/v1/station/current/klima-v2-10min
    Parameters fetched: TL (air temp °C), TD (dewpoint °C),
                        FF (wind speed m/s), DD (wind direction °).

    Returns list of {sid, name, lon, lat, temp, dewpt, wspd, wdir}.
    """
    global _GEOSPHERE_METOBS_CACHE, _GEOSPHERE_METOBS_CACHE_TIME
    now = time.time()
    if _GEOSPHERE_METOBS_CACHE is not None and (now - _GEOSPHERE_METOBS_CACHE_TIME) < _GEOSPHERE_METOBS_TTL:
        return _GEOSPHERE_METOBS_CACHE

    BASE = "https://dataset.api.hub.geosphere.at/v1/station/current/klima-v2-10min"
    params = urlencode({"parameters": "TL,TD,FF,DD", "output_format": "geojson"})
    url = f"{BASE}?{params}"

    try:
        req = Request(url, headers={"User-Agent": "eme_tt/1.0 (radar dashboard)"})
        ctx = ssl.create_default_context()
        with urlopen(req, context=ctx, timeout=20) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"[geosphere-metobs] fetch failed: {exc}") from exc

    features = data.get("features") or []
    stations_out: List[dict] = []

    for feat in features:
        try:
            geom  = feat.get("geometry") or {}
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                continue

            props = feat.get("properties") or {}
            # Station ID and name – GeoSphere puts them directly in properties
            sid  = str(props.get("station_id") or props.get("id") or "")
            name = str(props.get("name") or props.get("station_name") or sid or "AT")

            parameters = props.get("parameters") or {}

            def _pval(key: str) -> Optional[float]:
                """Extract the most recent scalar value for a parameter."""
                p = parameters.get(key)
                if p is None:
                    return None
                d = p.get("data")
                if d is None:
                    return None
                # data is a list; take the last non-null element
                if isinstance(d, list):
                    for v in reversed(d):
                        if v is not None:
                            try:
                                fv = float(v)
                                if abs(fv) < 9998:
                                    return round(fv, 1)
                            except (TypeError, ValueError):
                                pass
                    return None
                try:
                    fv = float(d)
                    return round(fv, 1) if abs(fv) < 9998 else None
                except (TypeError, ValueError):
                    return None

            temp  = _pval("TL")
            dewpt = _pval("TD")
            wspd  = _pval("FF")
            wdir_v = _pval("DD")
            wdir  = int(round(wdir_v)) if wdir_v is not None else None

            if temp is None and dewpt is None and wspd is None:
                continue

            stations_out.append({
                "sid":   sid,
                "name":  name,
                "lon":   round(lon, 4),
                "lat":   round(lat, 4),
                "temp":  temp,
                "dewpt": dewpt,
                "wspd":  wspd,
                "wdir":  wdir,
            })
        except Exception:
            continue

    _GEOSPHERE_METOBS_CACHE      = stations_out
    _GEOSPHERE_METOBS_CACHE_TIME = now
    return stations_out


def _safe_fetch_geosphere_metobs() -> List[dict]:
    """Exception-safe wrapper — returns [] on any unexpected error."""
    try:
        return _fetch_geosphere_metobs()
    except Exception:
        return []

# ESWD (European Severe Weather Database) reports

def _fetch_eswd_reports(
    sd_utc: Optional[datetime] = None,
    ed_utc: Optional[datetime] = None,
) -> dict:
    """Fetch ESWD report list for Denmark and return a GeoJSON FeatureCollection.

    Parameters
    ----------
    sd_utc : start of window (defaults to start-of-today UTC)
    ed_utc : end of window   (defaults to now UTC)

    Authentication
    --------------
    Set ``ESWD_API_TOKEN`` (or legacy ``ESWD_API_KEY``) to your ESWD API **Bearer**
    token from account settings — see https://eswd.eu/en/api/docs . Without it,
    no upstream request is made and an empty collection is returned.
    """
    now_utc = datetime.now(timezone.utc)
    if ed_utc is None:
        ed_utc = now_utc
    if sd_utc is None:
        sd_utc = ed_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    sd_str = sd_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    ed_str = ed_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    token = (os.environ.get("ESWD_API_TOKEN") or os.environ.get("ESWD_API_KEY") or "").strip()
    if not token:
        return {
            "type": "FeatureCollection",
            "features": [],
            "meta": {
                "sd": sd_str,
                "ed": ed_str,
                "configured": False,
                "hint": (
                    "ESWD overlay needs ESWD_API_TOKEN (Bearer token from your ESWD account). "
                    "See https://eswd.eu/en/api/docs — restart the server after setting it."
                ),
            },
        }

    params = {
        "sd":       sd_str,
        "ed":       ed_str,
        "x0":       str(ESWD_DK_X0),
        "x1":       str(ESWD_DK_X1),
        "y0":       str(ESWD_DK_Y0),
        "y1":       str(ESWD_DK_Y1),
        "countries": "DK",
        "types":    ESWD_TYPES,
        "levels":   ESWD_LEVELS,
    }
    url = ESWD_API_BASE + "?" + urlencode(params)

    def _make_req() -> Request:
        r = Request(url)
        r.add_header("Authorization", f"Bearer {token}")
        r.add_header("Accept", "application/json")
        r.add_header("User-Agent", "DNSROP-Radar/1.0")
        return r

    def _do_fetch(ctx_inner: ssl.SSLContext) -> Tuple[bytes, str]:
        """Return (body, transport_error). HTTP 4xx/5xx still yields a body."""
        try:
            with urlopen(_make_req(), timeout=25, context=ctx_inner) as resp:
                return resp.read(), ""
        except HTTPError as he:
            try:
                return he.read(), f"HTTP {he.code}"
            except Exception:
                return b"", f"HTTP {he.code}"
        except Exception as exc:
            return b"", str(exc)

    ctx = ssl.create_default_context()
    raw, fetch_err = _do_fetch(ctx)
    if fetch_err and not raw:
        ctx2 = ssl.create_default_context()
        ctx2.check_hostname = False
        ctx2.verify_mode = ssl.CERT_NONE
        raw, fetch_err = _do_fetch(ctx2)

    if fetch_err and not raw:
        print(f"[ESWD] fetch failed ({fetch_err}); returning empty FeatureCollection")
        return {
            "type": "FeatureCollection",
            "features": [],
            "meta": {"sd": sd_str, "ed": ed_str, "error": fetch_err},
        }

    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as exc:
        print(f"[ESWD] JSON parse failed: {exc}")
        return {
            "type": "FeatureCollection",
            "features": [],
            "meta": {"sd": sd_str, "ed": ed_str, "error": f"JSON parse error: {exc}"},
        }

    # Normalize report list; guard against null `reports` / error-only JSON (avoids server 500).
    reports: List[Any]
    if isinstance(data, list):
        reports = data
    elif isinstance(data, dict):
        api_msg = data.get("message") or data.get("error") or data.get("detail")
        rr = data.get("reports")
        if rr is None:
            rr = data.get("data")
        if not isinstance(rr, list):
            rr = []
        if isinstance(api_msg, str) and api_msg.strip() and not rr:
            print(f"[ESWD] API: {api_msg.strip()}")
            meta: Dict[str, Any] = {
                "sd": sd_str,
                "ed": ed_str,
                "api_message": api_msg.strip(),
            }
            if fetch_err:
                meta["http_status"] = fetch_err
            return {"type": "FeatureCollection", "features": [], "meta": meta}
        reports = rr
    else:
        reports = []

    # Colour / icon mapping by event type
    TYPE_COLOURS: Dict[str, str] = {
        "TORNADO": "#cc00ff",
        "WIND":    "#ff6600",
        "HAIL":    "#00ccff",
        "PRECIP":  "#0099ff",
    }
    TYPE_ICONS: Dict[str, str] = {
        "TORNADO": "🌪",
        "WIND":    "💨",
        "HAIL":    "🧊",
        "PRECIP":  "🌧",
    }

    features = []
    for r in reports:
        if not isinstance(r, dict):
            continue
        lon = r.get("lon") or r.get("longitude")
        lat = r.get("lat") or r.get("latitude")
        if lon is None or lat is None:
            continue
        try:
            lon, lat = float(lon), float(lat)
        except (TypeError, ValueError):
            continue
        rtype = str(r.get("type") or r.get("event_type") or "WIND").upper()
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "type":        rtype,
                "label":       TYPE_ICONS.get(rtype, "⚡"),
                "colour":      TYPE_COLOURS.get(rtype, "#ffff00"),
                "time":        str(r.get("datetime") or r.get("time") or ""),
                "description": str(r.get("description") or r.get("comment") or ""),
                "location":    str(r.get("location") or r.get("place") or ""),
                "qc":          str(r.get("qc") or r.get("quality") or ""),
                "id":          str(r.get("id") or ""),
            },
        })

    return {
        "type":     "FeatureCollection",
        "features": features,
        "meta": {
            "sd":    sd_str,
            "ed":    ed_str,
            "count": len(features),
        },
    }


# Dashboard builder
def build_payload(
    *,
    dmi_target_utc: Optional[datetime] = None,
    providers: Optional[set] = None,
    station_filter: Optional[set] = None,
    dealias_unravel: bool = False,
) -> dict:
    rendered: Dict[str, dict] = {}
    errors: Dict[str, str] = {}

    stations = _all_stations()

    # Collect stations to process after applying provider / name filters.
    work_items = [
        (station_name, station_info)
        for station_name, station_info in stations.items()
        if (providers is None or station_info.get("provider") in providers)
        and (station_filter is None or station_name in station_filter)
    ]

    def _process_one(station_name: str, station_info: dict) -> Tuple[str, dict, Optional[str]]:
        """Fetch + render a single station; returns (name, result_dict, error_str|None)."""
        try:
            if not HAS_RADAR_DEPS:
                raise RuntimeError(f"Missing dependencies: {RADAR_DEPS_ERROR}")
            rendered_station = _render_station_products(
                {**station_info, "_name": station_name},
                dmi_target_utc=dmi_target_utc,
                dealias_unravel=dealias_unravel,
            )
            # -Non-reporting check
            _dt_str = rendered_station.get("datetime", "")
            _reporting = True
            if str(station_info.get("provider", "")).lower() == "smhi":
                _reporting = True
            elif _dt_str and _dt_str != "n/a":
                try:
                    _obs_dt = _parse_rfc3339(_dt_str)
                    _age_min = (datetime.now(timezone.utc) - _obs_dt).total_seconds() / 60.0
                    if _age_min > 60.0:
                        _reporting = False
                except Exception:
                    pass
            rendered_station["reporting"] = _reporting
            return station_name, rendered_station, None
        except Exception as error:
            err_result = {
                "datetime": "n/a",
                "bounds": None,
                "products": {pk: None for pk in PRODUCTS},
                "tvsData": {"tvsMarkers": []},
                "reporting": False,
            }
            return station_name, err_result, str(error)

    # Fan-out network I/O across stations in parallel.  CPU-bound post-
    # processing (NumPy) is moderate per station so threads are fine; the
    # dominant cost is HTTP latency, which parallelises perfectly.
    max_workers = min(32, max(1, len(work_items)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_one, name, info): name
            for name, info in work_items
        }
        for future in concurrent.futures.as_completed(futures):
            station_name, result, err = future.result()
            rendered[station_name] = result
            if err is not None:
                errors[station_name] = err
            else:
                # Back-propagate resolved lat/lon/range into the stations dict
                # (mirrors the original behaviour for downstream payload use).
                sm = result.get("stationMeta") or {}
                if sm:
                    orig = stations[station_name]
                    stations[station_name] = {
                        **orig,
                        "lat": float(sm.get("lat", orig.get("lat", 51.0))),
                        "lon": float(sm.get("lon", orig.get("lon", 10.0))),
                        "range_km": float(sm.get("range_km", orig.get("range_km", 180.0))),
                    }

    return {
        "mapboxStyles": MAPBOX_STYLES,
        "defaultMapboxStyleKey": DEFAULT_MAPBOX_STYLE_KEY,
        "mapboxToken": MAPBOX_ACCESS_TOKEN,
        "buildInfo": {
            "builtAtUtc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "hasRadarDeps": bool(HAS_RADAR_DEPS),
            "radarDepsError": RADAR_DEPS_ERROR,
            "hasUnravel": bool(HAS_UNRAVEL),
            "unravelImportError": UNRAVEL_IMPORT_ERROR,
        },
        "stations": {
            k: {ek: ev for ek, ev in v.items() if ek != "file_bytes"}
            for k, v in stations.items()
        },
        "products": {k: v["label"] for k, v in PRODUCTS.items()},
        "productsMeta": {k: v.get("base", False) for k, v in PRODUCTS.items()},
        "rendered": rendered,
        "errors": errors,
        "colormaps": {
            key: [{"value": value, "color": list(color)} for value, color in cmap]
            for key, cmap in COLORMAPS.items()
        },
        "legendSteps": LEGEND_STEPS,
        "tvsIcons": _load_tvs_icons(),
        "dmiApiKey": DMI_API_KEY,
        "frostClientId": FROST_CLIENT_ID,
        "dwdMetObs": _safe_fetch_dwd_metobs(),
        "frostMetObs": _safe_fetch_frost_metobs(),
        "knmiMetObs": _safe_fetch_knmi_metobs(),
        "geosphereMetObs": _safe_fetch_geosphere_metobs(),
    }

def build_dashboard(
    output_path: Path = Path("dmrat.html"),
    *,
    dmi_target_utc: Optional[datetime] = None,
    providers: Optional[set] = None,
    station_filter: Optional[set] = None,
    dealias_unravel: bool = False,
    _prebuilt_payload: Optional[dict] = None,
) -> Path:
    # Accept a pre-built payload so callers that already ran build_payload()
    # (e.g. _bg_prefetch) can pass it in and avoid a redundant second build.
    payload = _prebuilt_payload if _prebuilt_payload is not None else build_payload(
        dmi_target_utc=dmi_target_utc,
        providers=providers,
        station_filter=station_filter,
        dealias_unravel=dealias_unravel,
    )

    html = r"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Danish Radar Workstation</title>
<link href='https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.css' rel='stylesheet'>
<style>
*{box-sizing:border-box;}
body{margin:0;background:#c8c8c8;font-family:Tahoma,Segoe UI,sans-serif;}
#topbar{height:24px;display:flex;align-items:center;gap:4px;padding:0 5px;border-bottom:1px solid #707070;background:linear-gradient(#efefef,#c9c9c9);}
select,button{height:20px;border:1px solid #666;background:#f6f6f6;font-size:11px;box-shadow:inset 0 1px #fff;border-radius:0;outline:none;}
select{padding:0 2px;}
button{padding:0 7px;cursor:pointer;}
optgroup{font-size:11px;font-style:normal;color:#444;background:#f6f6f6;}
optgroup option{font-style:normal;color:#000;background:#f6f6f6;}
#wrapper{display:flex;flex-direction:column;height:calc(100vh - 24px);}
#grid{display:grid;grid-template-columns:1fr 1fr;gap:0;padding:0;flex:1;min-height:0;}
.panel{border:1px solid #666;background:#1d1d1d;display:flex;flex-direction:column;}
.panel.time-focus{box-shadow:inset 0 0 0 2px #0b5cad;}
.head{height:22px;display:flex;justify-content:space-between;align-items:center;padding:0 5px;border-bottom:1px solid #858585;background:#dcdcdc;font-size:11px;gap:6px;}
.subhead{height:16px;display:flex;align-items:center;justify-content:space-between;padding:0 5px;border-bottom:1px solid #3a3a3a;background:#2a2a2a;color:#cfcfcf;font-size:10px;font-family:Consolas,ui-monospace,monospace;white-space:nowrap;overflow:hidden;}
.subhead-left{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex-shrink:1;min-width:0;}
.subhead-right{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex-shrink:0;max-width:55%;margin-left:10px;color:#8ab4e8;}
.panel-content{display:flex;flex:1;min-height:0;position:relative;}
.legend{position:absolute;left:0;top:0;bottom:0;z-index:3;width:44px;background:transparent;border-right:none;color:#f0f0f0;padding:2px 1px;display:flex;flex-direction:column;align-items:center;min-height:0;pointer-events:none;}
.legend-title{display:none;}
.legend-scale{position:relative;flex:1;width:38px;min-height:0;align-self:stretch;}
.legend-bar{position:absolute;left:18px;width:12px;top:0;bottom:0;border:1px solid #999;}
.legend-tick{position:absolute;left:0;right:0;height:1px;}
.legend-tick-line{position:absolute;left:14px;width:6px;height:1px;background:#fff;top:0;}
.legend-tick-label{position:absolute;left:0;top:-6px;width:12px;font-size:9px;text-align:right;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,0.8);}
.map{flex:1;min-width:0;width:100%;}
.mapboxgl-ctrl-logo,.mapboxgl-ctrl-attrib{display:none !important;}
.mapboxgl-ctrl-bottom-left,.mapboxgl-ctrl-bottom-right{display:none !important;}
.hidden{display:none !important;}
.head-left{display:flex;align-items:center;gap:8px;}
#csPanel{border-top:2px solid #555;background:#181818;display:none;flex-direction:column;flex-shrink:0;min-height:160px;max-height:30vh;}
#csPanel.cs-visible{display:flex;}
#csHead{height:26px;display:flex;align-items:center;gap:6px;padding:0 6px;background:#2c2c2c;border-bottom:1px solid #555;color:#ddd;font-size:11px;flex-shrink:0;}
#csHead select,#csHead button{height:22px;font-size:11px;border-radius:0;}
#csHead label{color:#bbb;}
#csCanvasWrap{flex:1;overflow:hidden;position:relative;}
#csCanvas{display:block;width:100%;height:100%;}
#csOverlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#666;font-size:13px;pointer-events:none;}
#csInstruction{color:#e07000;font-size:11px;font-weight:bold;}
button.cs-active{background:#e07000;color:#fff;border-color:#b05000;}
#tvsBtn.tvs-active{background:#e07000;color:#fff;border-color:#b05000;}
#eswdBtn.eswd-active{background:#8800cc;color:#fff;border-color:#660099;}
#dealiasBtn.dealias-active{background:#e07000;color:#fff;border-color:#b05000;}
#iconD2Btn.icon-active{background:#0b5cad;color:#fff;border-color:#073b71;}
#timeControls{display:flex;align-items:center;gap:4px;font-size:11px;}
#timeControls input{height:20px;font-size:11px;padding:0 4px;border:1px solid #666;background:#f6f6f6;}
#timeControls span{min-width:72px;text-align:right;color:#333;}
#radarLive{background:#1a5c1a;color:#fff;border:1px solid #0e3d0e;font-weight:bold;white-space:nowrap;}
#radarLive:hover{background:#1e6e1e;}
#radarCountdown.archive-paused{color:#888;font-style:italic;}
.modal{position:fixed;inset:0;background:rgba(0,0,0,0.5);display:none;align-items:center;justify-content:center;z-index:10010;}
.modal.open{display:flex;}
.modal-card{background:#f3f3f3;border:1px solid #5a5a5a;width:min(600px,95vw);padding:14px;box-shadow:0 10px 24px rgba(0,0,0,0.35);}
.modal-card h3{margin:0 0 10px 0;font-size:16px;}.modal-card ul{margin:0;padding-left:20px;font-size:13px;}.modal-card li{margin:6px 0;}
.modal-close-row{margin-top:12px;text-align:right;}
.settings-row{display:flex;align-items:center;gap:8px;margin:8px 0;}
.settings-grid{display:grid;grid-template-columns:1fr auto auto;gap:8px;align-items:center;margin-top:12px;}
.status-ok{color:#106910;font-size:12px;}.status-empty{color:#555;font-size:12px;}
.settings-errors{margin-top:8px;padding:8px;border:1px solid #a6a6a6;background:#fff;color:#7d0000;font-size:12px;white-space:pre-wrap;max-height:180px;overflow:auto;}
.settings-errors.status-empty{color:#555;}
#forceRefreshBtn{background:#c0392b;color:#fff;border:1px solid #922b21;padding:5px 14px;cursor:pointer;font-size:13px;border-radius:2px;}
#forceRefreshBtn:hover:not(:disabled){background:#a93226;}
#forceRefreshBtn:disabled{background:#888;border-color:#666;cursor:not-allowed;opacity:0.7;}
#forceRefreshStatus{font-size:11px;color:#555;margin-left:8px;}
.settings-tabs{display:flex;gap:0;border-bottom:2px solid #bbb;margin-bottom:12px;}
.settings-tab{background:none;border:none;border-bottom:2px solid transparent;margin-bottom:-2px;padding:6px 16px;font-size:13px;cursor:pointer;color:#555;font-weight:500;transition:color 0.15s,border-color 0.15s;}
.settings-tab:hover{color:#222;}
.settings-tab.active{color:#1a5fa8;border-bottom:2px solid #1a5fa8;font-weight:600;}
.settings-tab-panel{display:none;}
.settings-tab-panel.active{display:block;}
.controls-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:4px;}
.controls-table th{text-align:left;padding:5px 8px;background:#e0e0e0;color:#333;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;border-bottom:1px solid #bbb;}
.controls-table td{padding:4px 8px;border-bottom:1px solid #eee;color:#333;vertical-align:middle;}
.controls-table tr:last-child td{border-bottom:none;}
.controls-table tr:hover td{background:#f0f4fa;}
.controls-section-header{font-size:11px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.06em;padding:8px 0 2px 0;border-top:1px solid #e0e0e0;margin-top:4px;}
.controls-table tr.section-row td{background:#f5f5f5;color:#666;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;padding:5px 8px 3px 8px;}
kbd{display:inline-block;padding:1px 5px;font-size:11px;font-family:monospace;background:#f3f3f3;border:1px solid #ccc;border-radius:3px;box-shadow:0 1px 1px rgba(0,0,0,0.12);color:#222;white-space:nowrap;}
.kbd-plus{margin:0 2px;color:#888;font-size:10px;}
#addRadarBtn{white-space:nowrap;}
#customRadarModal .modal-card{width:min(500px,95vw);}
#customRadarModal label{display:block;font-size:12px;color:#333;margin-bottom:2px;margin-top:8px;}
#customRadarModal input[type=text],#customRadarModal input[type=number],#customRadarModal input[type=file]{width:100%;padding:4px 6px;font-size:12px;border:1px solid #999;height:40px;box-sizing:border-box;background:#fff;}
#customRadarModal input[type=file]{height:auto;padding:3px;}
#customRadarModal .coord-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;}
#customRadarModal .coord-row label{margin-top:0;}
#customRadarStatus{margin-top:10px;font-size:12px;padding:6px 8px;border-radius:2px;display:none;}
#customRadarStatus.ok{background:#d4edda;border:1px solid #c3e6cb;color:#155724;display:block;}
#customRadarStatus.err{background:#f8d7da;border:1px solid #f5c6cb;color:#721c24;display:block;}
#customRadarStatus.loading{background:#fff3cd;border:1px solid #ffeeba;color:#856404;display:block;}
#customRadarUploadBtn{margin-top:12px;background:#2a6a2a;color:#fff;border:1px solid #1a4a1a;padding:5px 14px;cursor:pointer;font-size:13px;}
#customRadarUploadBtn:disabled{background:#888;border-color:#666;cursor:not-allowed;}
</style></head>
<body>
<div id='topbar'>
  <label>Country</label><select id='country'></select>
  <label>Station</label><select id='station'></select>
  <label>Windows</label><select id='count'><option value='1'>1</option><option value='2' selected>2</option><option value='4'>4</option></select>
  <button id='csBtn' title='Draw a line on the map to generate a vertical cross-section'>✛ Cross Section</button>
  <button id='tvsBtn' title='Toggle TVS marker overlay' style='display:flex;align-items:center;padding:0 5px;height:24px;'><img src='tvs.png' style='height:18px;width:18px;object-fit:contain;image-rendering:crisp-edges;vertical-align:middle;' alt='TVS'></button>
  <button id='eswdBtn' title='Toggle ESWD severe-weather reports overlay (Denmark)' style='display:flex;align-items:center;padding:0 5px;height:24px;'><img src='exclamation-circle.svg' style='height:18px;width:18px;object-fit:contain;image-rendering:crisp-edges;vertical-align:middle;' alt='ESWD'></button>
  <button id='dealiasBtn' title='Toggle velocity dealiasing (NLradar dual-PRF / UNRAVEL) for all radar panels' style='display:flex;align-items:center;padding:0 5px;height:24px;'><img src='radar.svg' style='height:18px;width:18px;object-fit:contain;image-rendering:crisp-edges;vertical-align:middle;' alt='Dealias'></button>
  <button id='iconD2Btn' title='Toggle ICON-D2 NWP overlay in the right panel' style='display:flex;align-items:center;padding:0 5px;height:24px;'><img src='cloud-bolt.svg' style='height:18px;width:18px;object-fit:contain;image-rendering:crisp-edges;vertical-align:middle;' alt='ICON-D2'></button>
  <select id='iconD2ProdSel' title='ICON-D2 product' style='display:none;font-size:11px;height:24px;padding:0 4px;border-radius:4px;border:1px solid #bbb;background:#fff;cursor:pointer;'>
    <option value='dbz_cmax'>Simulated Refl.</option>
    <option value='cape_ml'>CAPE (ML)</option>
    <option value='echotop'>Echo Top</option>
  </select>
  <button id='addRadarBtn' title='Upload a local radar volume file (ODIM/KNMI HDF5, NetCDF/CF-Radial, IRIS/Sigmet RAW*, Furuno, BUFR)' style='display:flex;align-items:center;padding:0 5px;height:24px;'><img src='file-upload.svg' style='height:18px;width:18px;object-fit:contain;image-rendering:crisp-edges;vertical-align:middle;' alt='Add Radar'></button>
  <span id='csInstruction' style='display:none'></span>

  <div id='cursorInfo' style='display:flex;align-items:center;gap:10px;font-family:Consolas,ui-monospace,monospace;font-size:11px;flex:1;padding:0 6px;min-width:0;overflow:hidden;justify-content:flex-end;'>
    <span id='cursorCoord' style='color:#111;letter-spacing:0.3px;white-space:nowrap'>—</span>
    <span id='cursorVal'   style='color:#555;white-space:nowrap'>—</span>
    <span id='cursorDist'  style='color:#888;white-space:nowrap'>—</span>
    <span id='removeRadarBtn' style='display:none;color:#7d0000;cursor:pointer;white-space:nowrap;text-decoration:underline;font-family:inherit;font-size:11px;' title='Remove this custom radar'>Remove radar</span>
  </div>

  <button id='settingsBtn' style='display:flex;align-items:center;padding:0 5px;height:24px;'><img src='settings.svg' style='height:18px;width:18px;object-fit:contain;image-rendering:crisp-edges;vertical-align:middle;' alt='Settings'></button>
  <div id='timeControls'>
    <input id='radarDate' type='date' title='UTC date for archive fetch'>
    <input id='radarTime' placeholder='HH:MM' size='5'>
    <button id='radarGo'>Go</button>
    <button id='radarLive' style='display:none' title='Return to live data'>↩ Live</button>
    <span id='radarCountdown'></span>
  </div>
  <span id='warn' style='color:#7d0000;font-size:11px'></span>
</div>

<div id='wrapper'>
  <div id='grid'>
    <div class='panel' id='panel1'>
      <div class='head'><div class='head-left'><span id='h1'>Panel 1</span><label>Product</label><select id='product1'></select><span id='azshear-ctrl1' style='display:none;align-items:center;gap:3px'><label style='white-space:nowrap'>Scans</label><select id='azshear-n1' title='Composite: merge this many past scans (max intensity)'><option value='1'>1 (live)</option><option value='2'>2</option><option value='3'>3</option><option value='5'>5</option><option value='10'>10</option></select></span></div><span id='t1'></span></div>
      <div class='subhead' id='meta1'><span class='subhead-left' id='meta1-left'></span><span class='subhead-right' id='meta1-right'></span></div>
      <div class='panel-content'><div id='legend1' class='legend'></div><div id='map1' class='map'></div></div>
    </div>
    <div class='panel' id='panel2'>
      <div class='head'><div class='head-left'><span id='h2'>Panel 2</span><label>Product</label><select id='product2'></select><span id='azshear-ctrl2' style='display:none;align-items:center;gap:3px'><label style='white-space:nowrap'>Scans</label><select id='azshear-n2' title='Composite: merge this many past scans (max intensity)'><option value='1'>1 (live)</option><option value='2'>2</option><option value='3'>3</option><option value='5'>5</option><option value='10'>10</option></select></span></div><span id='t2'></span></div>
      <div class='subhead' id='meta2'><span class='subhead-left' id='meta2-left'></span><span class='subhead-right' id='meta2-right'></span></div>
      <div class='panel-content'><div id='legend2' class='legend'></div><div id='map2' class='map'></div></div>
    </div>
    <div class='panel hidden' id='panel3'>
      <div class='head'><div class='head-left'><span id='h3'>Panel 3</span><label>Product</label><select id='product3'></select><span id='azshear-ctrl3' style='display:none;align-items:center;gap:3px'><label style='white-space:nowrap'>Scans</label><select id='azshear-n3' title='Composite: merge this many past scans (max intensity)'><option value='1'>1 (live)</option><option value='2'>2</option><option value='3'>3</option><option value='5'>5</option><option value='10'>10</option></select></span></div><span id='t3'></span></div>
      <div class='subhead' id='meta3'><span class='subhead-left' id='meta3-left'></span><span class='subhead-right' id='meta3-right'></span></div>
      <div class='panel-content'><div id='legend3' class='legend'></div><div id='map3' class='map'></div></div>
    </div>
    <div class='panel hidden' id='panel4'>
      <div class='head'><div class='head-left'><span id='h4'>Panel 4</span><label>Product</label><select id='product4'></select><span id='azshear-ctrl4' style='display:none;align-items:center;gap:3px'><label style='white-space:nowrap'>Scans</label><select id='azshear-n4' title='Composite: merge this many past scans (max intensity)'><option value='1'>1 (live)</option><option value='2'>2</option><option value='3'>3</option><option value='5'>5</option><option value='10'>10</option></select></span></div><span id='t4'></span></div>
      <div class='subhead' id='meta4'><span class='subhead-left' id='meta4-left'></span><span class='subhead-right' id='meta4-right'></span></div>
      <div class='panel-content'><div id='legend4' class='legend'></div><div id='map4' class='map'></div></div>
    </div>
  </div>

  <!-- Cross-section panel -->
  <div id='csPanel'>
    <div id='csHead'>
      <strong style='color:#00ccff'>▼ Vertical Cross-Section</strong>
      <label>Product</label><select id='csProdSel'></select>
      <label style='margin-left:8px'>Max height</label>
      <select id='csMaxAlt'><option value='5'>5 km</option><option value='10' selected>10 km</option><option value='15'>15 km</option><option value='20'>20 km</option></select>
      <button id='csRedraw' title='Redraw with current settings'>↺ Redraw</button>
      <button id='csClose' style='margin-left:auto'>✕ Close</button>
      <span id='csInfo' style='color:#888;font-size:10px;margin-left:8px'></span>
    </div>
    <div id='csCanvasWrap'>
      <canvas id='csCanvas'></canvas>
      <div id='csOverlay'>Draw a line on the map to generate a cross-section</div>
    </div>
  </div>
</div>

<div id='customRadarModal' class='modal' aria-hidden='true'>
  <div class='modal-card' id='customRadarCard'>
    <h3 style='margin:0 0 8px'>Add Custom Radar</h3>
    <div style='display:flex;gap:0;border-bottom:2px solid #bbb;margin-bottom:12px;'>
      <button id='customRadarTabFile' style='background:none;border:none;border-bottom:2px solid #2a6a2a;margin-bottom:-2px;padding:5px 14px;font-size:12px;font-weight:600;cursor:pointer;color:#2a6a2a;'>Upload File</button>
      <button id='customRadarTabApi' style='background:none;border:none;border-bottom:2px solid transparent;margin-bottom:-2px;padding:5px 14px;font-size:12px;font-weight:600;cursor:pointer;color:#555;'>API Endpoint</button>
    </div>
    <div id='customRadarPanelFile'>
    <p style='font-size:12px;color:#555;margin:0 0 10px'>Upload a local radar volume file (ODIM/KNMI HDF5, NetCDF/CF-Radial, IRIS/Sigmet RAW*, Furuno <code>.scn/.scnx</code>, or BUFR via converter hook). Processed on the server and rendered immediately.</p>
    <label style='display:block;font-size:12px;color:#333;margin-bottom:2px' for='customRadarName'>Radar name <span style='color:#c00'>*</span></label>
    <input type='text' id='customRadarName' placeholder='e.g. Brdy (CHMI)' autocomplete='off' style='width:100%;padding:4px 6px;font-size:12px;border:1px solid #999;height:40px;box-sizing:border-box'>
    <label style='display:block;font-size:12px;color:#333;margin:8px 0 2px' for='customRadarFile'>Radar file - dBZ / primary volume (ODIM HDF5, NetCDF, IRIS RAW*, Furuno, BUFR) <span style='color:#c00'>*</span></label>
    <input type='file' id='customRadarFile' style='width:100%;font-size:12px;box-sizing:border-box'>
    <p style='font-size:11px;color:#777;margin:8px 0 4px'>If your radar exports separate files per product, upload them below (all optional):</p>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;margin-bottom:4px'>
      <div><label style='display:block;font-size:11px;color:#555;margin-bottom:2px'>V — Velocity</label><input type='file' id='customRadarFile_V' style='width:100%;font-size:11px;box-sizing:border-box'></div>
      <div><label style='display:block;font-size:11px;color:#555;margin-bottom:2px'>PhiDP — Differential Phase</label><input type='file' id='customRadarFile_PhiDP' style='width:100%;font-size:11px;box-sizing:border-box'></div>
      <div><label style='display:block;font-size:11px;color:#555;margin-bottom:2px'>RhoHV — Correlation</label><input type='file' id='customRadarFile_RhoHV' style='width:100%;font-size:11px;box-sizing:border-box'></div>
      <div><label style='display:block;font-size:11px;color:#555;margin-bottom:2px'>W — Spectrum Width</label><input type='file' id='customRadarFile_W' style='width:100%;font-size:11px;box-sizing:border-box'></div>
      <div><label style='display:block;font-size:11px;color:#555;margin-bottom:2px'>ZDR — Differential Reflectivity</label><input type='file' id='customRadarFile_ZDR' style='width:100%;font-size:11px;box-sizing:border-box'></div>
    </div>
    <p style='font-size:11px;color:#777;margin:8px 0 4px'>Coordinates are read automatically from file metadata. Override only if needed:</p>
    <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px'>
      <div>
        <label style='display:block;font-size:11px;color:#555;margin-bottom:2px' for='customRadarLat'>Latitude</label>
        <input type='number' id='customRadarLat' placeholder='auto' step='0.0001' style='width:100%;padding:3px 5px;font-size:12px;border:1px solid #999;height:24px;box-sizing:border-box'>
      </div>
      <div>
        <label style='display:block;font-size:11px;color:#555;margin-bottom:2px' for='customRadarLon'>Longitude</label>
        <input type='number' id='customRadarLon' placeholder='auto' step='0.0001' style='width:100%;padding:3px 5px;font-size:12px;border:1px solid #999;height:24px;box-sizing:border-box'>
      </div>
      <div>
        <label style='display:block;font-size:11px;color:#555;margin-bottom:2px' for='customRadarRange'>Range (km)</label>
        <input type='number' id='customRadarRange' placeholder='auto' step='1' min='1' style='width:100%;padding:3px 5px;font-size:12px;border:1px solid #999;height:24px;box-sizing:border-box'>
      </div>
    </div>
    </div><!-- end customRadarPanelFile -->
    <div id='customRadarPanelApi' style='display:none'>
    <p style='font-size:12px;color:#555;margin:0 0 10px'>Enter a public radar data API endpoint. The server will probe it, find the most recent volume file, download and render it automatically.</p>
    <p style='font-size:11px;color:#777;margin:0 0 6px'>Examples of supported endpoint styles:</p>
    <ul style='font-size:11px;color:#555;margin:0 0 10px;padding-left:18px;line-height:1.7'>
      <li>OGC API items: <code style='font-size:10px'>https://opendataapi.dmi.dk/v1/radardata/collections/volume/items</code></li>
      <li>HTTP directory: <code style='font-size:10px'>https://opendata.dwd.de/weather/radar/sites/sweep_vol_z/asb/hdf5/filter_polarimetric/</code></li>
      <li>Apache listing: <code style='font-size:10px'>https://opendata.meteoromania.ro/radar/BUC/</code></li>
    </ul>
    <label style='display:block;font-size:12px;color:#333;margin-bottom:2px' for='customRadarApiName'>Radar name <span style='color:#c00'>*</span></label>
    <input type='text' id='customRadarApiName' placeholder='e.g. My Radar (DWD Boostedt)' autocomplete='off' style='width:100%;padding:4px 6px;font-size:12px;border:1px solid #999;height:40px;box-sizing:border-box;margin-bottom:8px'>
    <label style='display:block;font-size:12px;color:#333;margin-bottom:2px' for='customRadarApiUrl'>Endpoint URL <span style='color:#c00'>*</span></label>
    <input type='text' id='customRadarApiUrl' placeholder='https://…' autocomplete='off' style='width:100%;padding:4px 6px;font-size:12px;border:1px solid #999;height:40px;box-sizing:border-box;margin-bottom:8px'>
    <p style='font-size:11px;color:#777;margin:4px 0'>Coordinates are read automatically from file metadata. Override only if needed:</p>
    <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:4px'>
      <div>
        <label style='display:block;font-size:11px;color:#555;margin-bottom:2px' for='customRadarApiLat'>Latitude</label>
        <input type='number' id='customRadarApiLat' placeholder='auto' step='0.0001' style='width:100%;padding:3px 5px;font-size:12px;border:1px solid #999;height:24px;box-sizing:border-box'>
      </div>
      <div>
        <label style='display:block;font-size:11px;color:#555;margin-bottom:2px' for='customRadarApiLon'>Longitude</label>
        <input type='number' id='customRadarApiLon' placeholder='auto' step='0.0001' style='width:100%;padding:3px 5px;font-size:12px;border:1px solid #999;height:24px;box-sizing:border-box'>
      </div>
      <div>
        <label style='display:block;font-size:11px;color:#555;margin-bottom:2px' for='customRadarApiRange'>Range (km)</label>
        <input type='number' id='customRadarApiRange' placeholder='auto' step='1' min='1' style='width:100%;padding:3px 5px;font-size:12px;border:1px solid #999;height:24px;box-sizing:border-box'>
      </div>
    </div>
    </div><!-- end customRadarPanelApi -->
    <div id='customRadarStatus' style='margin-top:10px;font-size:12px;padding:6px 8px;border-radius:2px;display:none'></div>
    <div class='modal-close-row' style='display:flex;gap:8px;align-items:center;margin-top:12px'>
      <button id='customRadarUploadBtn' style='background:#2a6a2a;color:#fff;border:1px solid #1a4a1a;padding:5px 14px;cursor:pointer;font-size:13px'>Upload &amp; Render</button>
      <button id='customRadarFetchBtn' style='background:#2a6a2a;color:#fff;border:1px solid #1a4a1a;padding:5px 14px;cursor:pointer;font-size:13px;display:none'>Fetch &amp; Render</button>
      <button id='customRadarClose' style='margin-left:auto'>Cancel</button>
    </div>
  </div>
</div>
<div id='settingsModal' class='modal' aria-hidden='true'><div class='modal-card' style='max-height:85vh;display:flex;flex-direction:column;overflow:hidden;'><h3 style='margin:0 0 8px 0;font-size:16px;'>Settings</h3><div class='settings-tabs'><button class='settings-tab active' data-tab='general'>General</button><button class='settings-tab' data-tab='colors'>Colors</button><button class='settings-tab' data-tab='controls'>Controls</button><button class='settings-tab' data-tab='credits'>Credits</button></div><div style='overflow-y:auto;flex:1;'><div id='settingsTabGeneral' class='settings-tab-panel active'><div class='settings-row'><label for='mapStyleSel'>Map style</label><select id='mapStyleSel'></select></div><h4>Layers</h4><div id='layerRows'></div><h4>Data warnings</h4><div id='settingsErrors' class='settings-errors status-empty'>Data warnings: none</div><h4>Data</h4><div class='settings-row'><button id='forceRefreshBtn'>&#8635; Force Data Refresh</button><span id='forceRefreshStatus'></span></div></div><div id='settingsTabColors' class='settings-tab-panel'><h4 style='margin-top:4px;'>Color table files (.csv / .pal)</h4><p style='font-size:11px;color:#666;margin:0 0 10px 0;'>Load a custom .csv or .pal color table for each product. Hit Reset to revert to the built-in preset.</p><div id='colortableRows' class='settings-grid'></div></div><div id='settingsTabControls' class='settings-tab-panel'><table class='controls-table'><thead><tr><th>Key</th><th>Action</th></tr></thead><tbody><tr class='section-row'><td colspan='2'>Products</td></tr><tr><td><kbd>Z</kbd></td><td>Reflectivity</td></tr><tr><td><kbd>V</kbd></td><td>Velocity</td></tr><tr><td><kbd>C</kbd></td><td>Correlation Coefficient (RhoHV)</td></tr><tr><td><kbd>S</kbd></td><td>Spectrum Width</td></tr><tr><td><kbd>D</kbd></td><td>Differential Reflectivity (ZDR)</td></tr><tr><td><kbd>P</kbd></td><td>Specific Differential Phase (KDP)</td></tr><tr><td><kbd>E</kbd></td><td>Echo Tops</td></tr><tr><td><kbd>0</kbd></td><td>Echo Tops (0 °C isotherm)</td></tr><tr><td><kbd>N</kbd></td><td>Normalized Rotation (NROT)</td></tr><tr><td><kbd>R</kbd></td><td>Storm-Relative Velocity (SRV)</td></tr><tr><td><kbd>M</kbd></td><td>Mesocyclone (MESO)</td></tr><tr><td><kbd>U</kbd></td><td>Azimuthal Shear</td></tr><tr><td><kbd>H</kbd></td><td>Max Expected Hail Size (MESH)</td></tr><tr class='section-row'><td colspan='2'>Tilt / Elevation</td></tr><tr><td><kbd>↑</kbd></td><td>Step up one tilt angle</td></tr><tr><td><kbd>↓</kbd></td><td>Step down one tilt angle</td></tr><tr class='section-row'><td colspan='2'>Time Navigation</td></tr><tr><td><kbd>←</kbd></td><td>Step backward in scan time (archive)</td></tr><tr><td><kbd>→</kbd></td><td>Step forward in scan time / return to live</td></tr><tr class='section-row'><td colspan='2'>Dual-Panel / Model Mode</td></tr><tr><td><kbd>K</kbd></td><td>Focus time navigation on radar</td></tr><tr><td><kbd>L</kbd></td><td>Focus time navigation on model</td></tr><tr class='section-row'><td colspan='2'>Cross-Section</td></tr><tr><td><kbd>Esc</kbd></td><td>Cancel / end cross-section drawing</td></tr></tbody></table></div><div id='settingsTabCredits' class='settings-tab-panel'><ul style='padding-left:18px;margin:4px 0;font-size:12px;line-height:1.8;color:#333;'><li><strong>Mapbox</strong> for the map rendering engine.</li><li><strong>DMI</strong> (Danish Meteorological Institute) for Danish radar data.</li><li><strong>SMHI</strong> (Swedish Meteorological and Hydrological Institute) for Swedish radar data.</li><li><strong>FMI</strong> (Finnish Meteorological Institute) for Finnish radar data via public S3.</li><li><strong>DWD</strong> (Deutscher Wetterdienst) for German radar data via open data portal.</li><li><strong>KNMI</strong> (Koninklijk Nederlands Meteorologisch Instituut) for Netherlands radar and in-situ observation data via KNMI Open Data API.</li><li><strong>CHMI</strong> (Czech Hydrometeorological Institute) for Czech Republic radar data via open data portal.</li><li><strong>GeoSphere Austria</strong> for Austrian radar data (Hochficht) and TAWES surface observations via the GeoSphere Dataset API.</li><li><strong>Meteo-France / AERIS</strong> for French radial PVOL products via the SEDOO API.</li><li><strong>MeteoRomania</strong> for Romanian radar data via open data portal.</li><li><strong>Keskkonnaportaal KAIA</strong> for Estonian radar file downloads.</li><li><strong>Meteogate</strong> for Norwegian radar API test endpoint.</li><li><strong>ARPA Piemonte</strong> for Italian radar data (Bric della Croce &amp; Settepani) via open Apache directory.</li><li><strong>NASA</strong> for the Blue Marble basemap.</li><li><strong>Noa Baltzer</strong> as the author.</li><li><strong>Wx Tools</strong> as the color table contributor.</li></ul></div></div><div class='modal-close-row'><button id='settingsClose'>Close</button></div></div></div>
  <script src='https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.js'></script>
  <script src='https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js'></script>
<script>
const APP = __PAYLOAD__;

//  refs 
const countrySel    = document.getElementById('country');
const stationSel    = document.getElementById('station');
const countSel      = document.getElementById('count');
  const product1Sel   = document.getElementById('product1');
  const product2Sel   = document.getElementById('product2');
  const product3Sel   = document.getElementById('product3');
  const product4Sel   = document.getElementById('product4');
  const warn          = document.getElementById('warn');
  const settingsErrors = document.getElementById('settingsErrors');
  const mapStyleSel   = document.getElementById('mapStyleSel');
  const csProdSel     = document.getElementById('csProdSel');
  const metaLeftEls   = [document.getElementById('meta1-left'), document.getElementById('meta2-left'), document.getElementById('meta3-left'), document.getElementById('meta4-left')];
  const metaRightEls  = [document.getElementById('meta1-right'), document.getElementById('meta2-right'), document.getElementById('meta3-right'), document.getElementById('meta4-right')];

  const buildInfo = APP.buildInfo || {};
  if (warn && !warn.textContent) {
    if (buildInfo.hasRadarDeps === false) {
      warn.textContent = `Radar deps missing: ${buildInfo.radarDepsError || 'unknown import error'}`;
    } else if (typeof DecompressionStream === 'undefined'
               && !(typeof pako !== 'undefined' && typeof pako.ungzip === 'function')) {
      warn.textContent = 'Browser lacks gzip decoder for radar overlays; try Chrome/Edge or ensure pako loads.';
    }
  }
  // Dealiasing uses NumPy on the server (NLradar; optional UNRAVEL package)
  try {
    const _dealiasBtnEl = document.getElementById('dealiasBtn');
    if (_dealiasBtnEl && buildInfo.hasRadarDeps === false) {
      _dealiasBtnEl.disabled = true;
      _dealiasBtnEl.title = `Dealias unavailable (${buildInfo.radarDepsError || 'radar dependencies missing'})`;
    } else if (_dealiasBtnEl && buildInfo.hasUnravel === false) {
      _dealiasBtnEl.title = `Velocity dealias (NLradar). Install Python package unravel for UNRAVEL fallback: ${buildInfo.unravelImportError || 'not installed'}`;
    }
  } catch(e) {}
  try {
    if (iconD2Btn && location.protocol === 'file:') {
      iconD2Btn.disabled = true;
      iconD2Btn.title = 'ICON-D2 requires running the dashboard with --serve';
      if (iconD2ProdSel) iconD2ProdSel.disabled = true;
    }
  } catch(e) {}
const csMaxAlt      = document.getElementById('csMaxAlt');
const csPanel       = document.getElementById('csPanel');
const csCanvas      = document.getElementById('csCanvas');
const csOverlay     = document.getElementById('csOverlay');
const csInfo        = document.getElementById('csInfo');
const csBtn         = document.getElementById('csBtn');
const csInstruction = document.getElementById('csInstruction');
const dealiasBtn    = document.getElementById('dealiasBtn');
const iconD2Btn     = document.getElementById('iconD2Btn');
const iconD2ProdSel = document.getElementById('iconD2ProdSel');
const radarDateInp  = document.getElementById('radarDate');
const radarTimeInp  = document.getElementById('radarTime');
const radarGoBtn    = document.getElementById('radarGo');
const radarLiveBtn  = document.getElementById('radarLive');
const radarCountdown= document.getElementById('radarCountdown');
const panelEls      = [document.getElementById('panel1'), document.getElementById('panel2'), document.getElementById('panel3'), document.getElementById('panel4')];

// archiveMode: true while viewing a historical scan — pauses auto-refresh
let archiveMode = false;
let iconD2Enabled = false;
let iconD2Inventory = null;
let iconD2CurrentFrame = null;
let iconD2CurrentIndex = -1;
let iconD2Loading = false;
let iconD2PrevProduct2 = 'velocity';
let iconD2PrevCount = '2';
let timeNavFocus = 'radar';
let iconD2SyncedToRadar = true;
// Active ICON-D2 product key (must match a key in the Python ICON_D2_PRODUCTS dict)
let iconD2Product = 'dbz_cmax';
const ICON_D2_PRODUCT_OPTIONS = [
  { key: 'dbz_cmax', label: 'Simulated Refl.', colormap: 'reflectivity' },
  { key: 'cape_ml',  label: 'CAPE (ML)',       colormap: 'cape_ml'      },
  { key: 'echotop',  label: 'Echo Top',        colormap: 'echo_tops'    },
];

function _enterArchiveMode(iso) {
  archiveMode = true;
  // Stop the countdown ticker
  if(autoUpdateCfg.timerId){ clearInterval(autoUpdateCfg.timerId); autoUpdateCfg.timerId=null; }
  if(radarCountdown){ radarCountdown.textContent='⏸ paused'; radarCountdown.classList.add('archive-paused'); }
  if(radarLiveBtn) radarLiveBtn.style.display='';
  if(warn) warn.textContent = iso
    ? `Archive: ${iso.replace('T',' ').replace('Z',' UTC')}`
    : '';
}

function _exitArchiveMode() {
  archiveMode = false;
  timeOffsetMin = 0;
  if(radarLiveBtn) radarLiveBtn.style.display='none';
  if(radarCountdown) radarCountdown.classList.remove('archive-paused');
  if(warn) warn.textContent = '';
}

// Provider → country mapping 
const PROVIDER_COUNTRY = {
  dmi: 'Denmark',
  smhi: 'Sweden',
  fmi: 'Finland',
  dwd: 'Germany',
  knmi: 'Netherlands',
  knmi_cabauw: 'Netherlands',
  chmi: 'Czech Republic',
  shmu: 'Slovakia',
  geosphere: 'Austria',
  // fr_mf: 'France',  // DISABLED — SEDOO
  romania: 'Romania',
  estonia: 'Estonia',
  meteogate: 'Norway',
  meteogate_test: 'Norway',
  arpa_piemonte: 'Italy',
  arpa_lombardia: 'Italy',
  custom: 'Custom',
};
const COUNTRY_ORDER    = ['Denmark', 'Sweden', 'Finland', 'Germany', 'Netherlands', 'Czech Republic', 'Slovakia', 'Austria', 'Romania', 'Estonia', 'Norway', 'Italy', 'Custom'];  // 'France' removed — SEDOO disabled

function stationCountry(sName) {
  const info = APP.stations[sName] || {};
  return PROVIDER_COUNTRY[(info.provider || 'dmi').toLowerCase()] || 'Other';
}

function populateCountry() {
  // Build set of countries that actually have stations
  const present = new Set(Object.keys(APP.stations).map(stationCountry));
  countrySel.innerHTML = '';
  COUNTRY_ORDER.forEach(c => { if (present.has(c)) countrySel.add(new Option(c, c)); });
  // Also add any unexpected countries
  present.forEach(c => { if (!COUNTRY_ORDER.includes(c)) countrySel.add(new Option(c, c)); });
}

function populateStationsForCountry(country) {
  stationSel.innerHTML = '';
  Object.keys(APP.stations)
    .filter(n => stationCountry(n) === country)
    .forEach(n => stationSel.add(new Option(n, n)));
}

populateCountry();
populateStationsForCountry(countrySel.value);

// Populate product dropdowns with Base / Derived optgroups 
function populateProductSelect(sel) {
  sel.innerHTML = '';
  const baseGrp    = document.createElement('optgroup'); baseGrp.label    = 'Base Products';
  const derivedGrp = document.createElement('optgroup'); derivedGrp.label = 'Derived Products';
  // Use PRODUCTS metadata shipped via APP.productsMeta; fall back to APP.products labels
  const meta = APP.productsMeta || {};
  Object.keys(APP.products).forEach(k => {
    const isBase = meta[k] != null ? meta[k] : false;
    const opt = new Option(APP.products[k], k);
    (isBase ? baseGrp : derivedGrp).appendChild(opt);
  });
  if (baseGrp.children.length)    sel.appendChild(baseGrp);
  if (derivedGrp.children.length) sel.appendChild(derivedGrp);
}

[product1Sel, product2Sel, product3Sel, product4Sel, csProdSel].forEach(populateProductSelect);
Object.entries(APP.mapboxStyles).forEach(([sk, sc]) => mapStyleSel.add(new Option(sc.label, sk)));

stationSel.value  = APP.stations['Sindal'] ? 'Sindal' : stationSel.options[0]?.value || '';
product1Sel.value = 'reflectivity';
product2Sel.value = 'velocity';
product3Sel.value = 'correlation';
product4Sel.value = 'echo_tops';
csProdSel.value   = 'reflectivity';
mapStyleSel.value = APP.defaultMapboxStyleKey;
// Hide composite controls initially (will show if azimuthal_shear is selected)
// (azshear-ctrl1/2/3/4 already have display:none in HTML; _azshearCtrl declared further below)

// Per-panel tilt navigation state 
// tiltIdx[i] = index into pd.allTiltOverlays for panel i (0 = lowest/default)
const tiltIdx = [0, 0, 0, 0];

//  Scan time navigation state 
// timeOffsetMin: minutes offset from current scan (0 = latest, negative = older)
let timeOffsetMin = 0;
const TIME_STEP_MIN = 5;

  const productSels = [product1Sel, product2Sel, product3Sel, product4Sel];
  const panelProducts = productSels.map(sel => () => sel.value);
  function selectedPanelCount() {
    if (iconD2Enabled) return 2;
    const n = parseInt(countSel.value, 10);
    return n === 4 ? 4 : (n === 1 ? 1 : 2);
  }
  function visiblePanelIndices() {
    return Array.from({length: selectedPanelCount()}, (_v, i) => i);
  }
  const userColormaps = {}, userLegendSteps = {};
  const cartesianImageCache = {};
  let tvsOverlayEnabled = false;
  let eswdOverlayEnabled = false;
  let _eswdGeoJSON = null;   // cached GeoJSON from last ESWD fetch
  let dealiasEnabled = false;

  function _formatIsoUtc(iso) {
    if (!iso || iso === 'n/a') return 'n/a';
    try { return iso.replace('T', ' ').replace('Z', ' UTC'); } catch (_e) { return iso; }
  }

  function _formatForecastMinutes(totalMinutes) {
    const mins = Math.max(0, Number(totalMinutes) || 0);
    const hh = String(Math.floor(mins / 60)).padStart(2, '0');
    const mm = String(mins % 60).padStart(2, '0');
    return `F+${hh}:${mm}`;
  }

  function _currentRadarIso() {
    const sn = stationSel.value;
    const sd = APP.rendered[sn] || {};
    return sd.datetime || null;
  }

  function _updateTimeNavFocusUi() {
    panelEls.forEach((el, idx) => {
      if (!el) return;
      const active = iconD2Enabled && ((timeNavFocus === 'radar' && idx === 0) || (timeNavFocus === 'model' && idx === 1));
      el.classList.toggle('time-focus', active);
    });
  }

  async function _buildEncodedCartesianUrl(enc, pk, sig) {
    if (!enc || !enc.url) return null;
    if (cartesianDecodeCache[sig]) return cartesianDecodeCache[sig];

    return new Promise(resolve => {
      const img = new Image();
      img.onload = () => {
        const W = img.naturalWidth, H = img.naturalHeight;
        const c = document.createElement('canvas'); c.width = W; c.height = H;
        const ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const raw = ctx.getImageData(0, 0, W, H).data;
        const mn = Number(enc.min), mx = Number(enc.max);
        const denom = mx - mn || 1;
        const stops = getColormap(pk).slice().sort((a,b)=>a.value-b.value);
        const out = new Uint8ClampedArray(W * H * 4);
        for (let i = 0; i < W * H; i++) {
          const j = i * 4;
          if (raw[j+3] < 128) { out[j+3] = 0; continue; }
          const ev = (raw[j] << 8) | raw[j+1];
          const val = mn + (ev / 65534.0) * denom;
          const rgb = interpolateRgb(stops, val);
          out[j] = rgb[0]; out[j+1] = rgb[1]; out[j+2] = rgb[2]; out[j+3] = 255;
        }
        const oc = document.createElement('canvas'); oc.width = W; oc.height = H;
        oc.getContext('2d').putImageData(new ImageData(out, W, H), 0, 0);
        cartesianDecodeCache[sig] = oc.toDataURL('image/png');
        resolve(cartesianDecodeCache[sig]);
      };
      img.onerror = () => resolve(null);
      img.src = enc.url;
    });
  }

  function renderSettingsErrors() {
    if (!settingsErrors) return;
    const entries = Object.entries(APP.errors || {});
    if (!entries.length) {
      settingsErrors.classList.add('status-empty');
      settingsErrors.textContent = 'Data warnings: none';
      return;
    }
    settingsErrors.classList.remove('status-empty');
    settingsErrors.textContent = entries.map(([k, v]) => `${k}: ${v}`).join('\n');
  }

  function updatePanelMeta(idx) {
    if (iconD2Enabled && idx === 1) {
      if (metaLeftEls[idx]) {
        metaLeftEls[idx].textContent = iconD2CurrentFrame
          ? `ICON-D2 [${_iconD2ProductLabel()}] - Run ${_formatIsoUtc(iconD2CurrentFrame.runIso)} - ${_formatForecastMinutes(iconD2CurrentFrame.forecastMinutes)}`
          : `ICON-D2 [${_iconD2ProductLabel()}] - loading`;
      }
      if (metaRightEls[idx]) {
        metaRightEls[idx].textContent = iconD2CurrentFrame
          ? `Valid ${_formatIsoUtc(iconD2CurrentFrame.validIso)} - ${iconD2CurrentIndex + 1}/${(iconD2Inventory && iconD2Inventory.frames ? iconD2Inventory.frames.length : 0) || 1}`
          : 'No model frame';
      }
      return;
    }
    const sn = stationSel.value;
    const sd = APP.rendered[sn] || {};
    const pk = panelProducts[idx]();
    const label = APP.products[pk] || pk;
    if (metaLeftEls[idx]) {
      metaLeftEls[idx].textContent = `${sn} - ${label} - ${_formatIsoUtc(sd.datetime || 'n/a')}`;
    }
    let right = '';
    const pd = (sd.products || {})[pk];
    if (!pd) {
      right = 'No data';
    } else if (pd.type === 'polar') {
      const ov = pd.overlayData;
      if (!ov || !ov.b64) {
        right = 'No data';
      } else {
        const tilt = (typeof ov.elevationDeg === 'number') ? `Tilt ${ov.elevationDeg.toFixed(1)}°` : '';
        const src  = pd.sourceFile ? `src ${pd.sourceFile}` : '';
        right = [tilt, src].filter(Boolean).join(' · ');
      }
    } else if (pd.type === 'cartesian') {
      right = (pd.valueEncoded && pd.valueEncoded.url) ? 'Grid' : 'No data';
    }
    if (metaRightEls[idx]) metaRightEls[idx].textContent = right;
  }

  function updateAllPanelMeta() {
    panelEls.forEach((_el, idx) => updatePanelMeta(idx));
  }
const layerSettings = {
  rangeRings: true,
  coordGrid: false,
  lightning: true,
  legends: true,
  geocolor: false,
  hrv: false,
  airmass: false,
  metObs: false,
  bluemarble: false,
  stationSize: 6,
  ringWidth: 1.5,
};

// Persist UI state across reloads
const UI_STATE_KEY = 'radar-ui-state-v1';
function saveUiState(){
  try{
    const state={
      country: countrySel.value,
      station: stationSel.value,
      count: countSel.value,
      product1: product1Sel.value,
      product2: product2Sel.value,
      product3: product3Sel.value,
      product4: product4Sel.value,
      csProd: csProdSel.value,
      csMaxAlt: csMaxAlt.value,
      mapStyle: mapStyleSel.value,
      layerSettings: {...layerSettings},
      tvsOverlayEnabled,
      eswdOverlayEnabled,
      dealiasEnabled,
      tiltIdx: [...tiltIdx],
    };
    localStorage.setItem(UI_STATE_KEY, JSON.stringify(state));
  }catch(e){}
}
function restoreUiState(){
  try{
    const raw=localStorage.getItem(UI_STATE_KEY);
    if(!raw) return;
    const s=JSON.parse(raw);
    if(s.country){
      // Set country first, repopulate stations, then set station
      const cOpt=[...countrySel.options].find(o=>o.value===s.country);
      if(cOpt){ countrySel.value=s.country; populateStationsForCountry(s.country); }
    }
    if(s.station && APP.stations[s.station]) stationSel.value=s.station;
    if(s.count) countSel.value=String(s.count);
    if(s.product1 && APP.products[s.product1]) product1Sel.value=s.product1;
    if(s.product2 && APP.products[s.product2]) product2Sel.value=s.product2;
    if(s.product3 && APP.products[s.product3]) product3Sel.value=s.product3;
    if(s.product4 && APP.products[s.product4]) product4Sel.value=s.product4;
    if(s.csProd && APP.products[s.csProd]) csProdSel.value=s.csProd;
    if(s.csMaxAlt) csMaxAlt.value=String(s.csMaxAlt);
    if(s.mapStyle && APP.mapboxStyles[s.mapStyle]) mapStyleSel.value=s.mapStyle;
    if(s.layerSettings){
      Object.keys(layerSettings).forEach(k=>{
        if(typeof s.layerSettings[k] !== 'undefined') layerSettings[k]=s.layerSettings[k];
      });
    }
    if(typeof s.tvsOverlayEnabled === 'boolean') tvsOverlayEnabled = s.tvsOverlayEnabled;
    if(typeof s.eswdOverlayEnabled === 'boolean') eswdOverlayEnabled = s.eswdOverlayEnabled;
    if(typeof s.dealiasEnabled === 'boolean') dealiasEnabled = s.dealiasEnabled;
    if(Array.isArray(s.tiltIdx)){ tiltIdx.forEach((_v,i)=>{ tiltIdx[i]=s.tiltIdx[i]||0; }); }
  }catch(e){}
}
restoreUiState();
// Sync TVS button visual state after restore
(function syncTvsBtn(){
  const btn=document.getElementById('tvsBtn');
  if(btn && tvsOverlayEnabled) btn.classList.add('tvs-active');
})();
// Sync ESWD button visual state after restore
(function syncEswdBtn(){
  const btn=document.getElementById('eswdBtn');
  if(btn && eswdOverlayEnabled) btn.classList.add('eswd-active');
})();
// Sync Dealias button visual state after restore
(function syncDealiasBtn(){
  const btn=document.getElementById('dealiasBtn');
  if(btn && dealiasEnabled) btn.classList.add('dealias-active');
})();
window.addEventListener('beforeunload', saveUiState);

// Mapbox maps
mapboxgl.accessToken = APP.mapboxToken;
function currentMapStyleUri() { return APP.mapboxStyles[mapStyleSel.value].style; }
const maps = [
  new mapboxgl.Map({ container:'map1', style:currentMapStyleUri(), center:[10.9,56.1], zoom:5.8, attributionControl:false }),
  new mapboxgl.Map({ container:'map2', style:currentMapStyleUri(), center:[10.9,56.1], zoom:5.8, attributionControl:false }),
  new mapboxgl.Map({ container:'map3', style:currentMapStyleUri(), center:[10.9,56.1], zoom:5.8, attributionControl:false }),
  new mapboxgl.Map({ container:'map4', style:currentMapStyleUri(), center:[10.9,56.1], zoom:5.8, attributionControl:false }),
];
const mapReady = [false, false, false, false];
let syncingMove = false;
function syncFrom(src) {
  if (syncingMove) return;
  syncingMove = true;
  const c = src.getCenter(), z = src.getZoom(), b = src.getBearing(), p = src.getPitch();
  maps.forEach(m => { if (m !== src) m.jumpTo({center:c,zoom:z,bearing:b,pitch:p}); });
  syncingMove = false;
}

// Colormap helpers 
function getColormap(pk) { return userColormaps[pk] || APP.colormaps[pk] || []; }
function getLegendStep(pk) { return userLegendSteps[pk] || APP.legendSteps[pk] || 1; }
function interpolateRgb(stops, value) {
  if (!stops.length) return [0,0,0];
  if (value <= stops[0].value) return stops[0].color;
  if (value >= stops[stops.length-1].value) return stops[stops.length-1].color;
  for (let i = 0; i < stops.length-1; i++) {
    const a = stops[i], b = stops[i+1];
    if (value < a.value || value > b.value) continue;
    if (a.value === b.value) return b.color;
    const t = (value - a.value) / (b.value - a.value);
    return [0,1,2].map(idx => Math.round(a.color[idx] + (b.color[idx] - a.color[idx]) * t));
  }
  return stops[stops.length-1].color;
}
function colorAtValue(pk, value) {
  const stops = getColormap(pk).slice().sort((a,b)=>a.value-b.value);
  if (!stops.length) return 'rgb(0,0,0)';
  return `rgb(${interpolateRgb(stops,value).join(',')})`;
}

// Gzip decode helper 
  async function decodeGzipUint16(b64, n) {
    const binary = atob(b64);
    const bytes  = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    let buf = null;
    if (typeof DecompressionStream !== 'undefined') {
      const ds     = new DecompressionStream('gzip');
      const writer = ds.writable.getWriter();
      writer.write(bytes);
      writer.close();
      const chunks = [], reader = ds.readable.getReader();
      while (true) { const {done,value} = await reader.read(); if (done) break; chunks.push(value); }
      let total = 0; for (const c of chunks) total += c.length;
      buf = new Uint8Array(total);
      let off = 0; for (const c of chunks) { buf.set(c, off); off += c.length; }
    } else if (typeof pako !== 'undefined' && typeof pako.ungzip === 'function') {
      buf = pako.ungzip(bytes);
    } else {
      throw new Error('No gzip decoder available (DecompressionStream/pako)');
    }
    if (!buf || buf.length < n * 2) {
      throw new Error(`Gzip decode short (${buf ? buf.length : 0} bytes for ${n} values)`);
    }
    const data = new Uint16Array(n);
    for (let i = 0; i < n; i++) data[i] = (buf[i*2] << 8) | buf[i*2+1];
    return data;
  }


// WebGL GEOMETRY-BASED RADAR LAYER
//
// Each radar gate is rendered as an exact trapezoid (2 triangles) in
// geographic space.  No polar texture or Mercator-quad sampling — the vertex
// shader computes the four geographic corners of every gate directly from
// (azimuth, range) using the forward geodesic formula, so quality is
// pixel-perfect at any zoom level and for any gate width.
// 

// Vertex shader
// Per-vertex:   a_corner (2 floats) — unit-quad corner selector (0 or 1)
// Per-instance: a_azStart, a_azEnd (degrees), a_rStart, a_rEnd (km),
//               a_val (encoded uint16 as float, 1..65535; 0 = nodata)
// The shader selects az = mix(azStart, azEnd, corner.x) and
//                   r  = mix(rStart,  rEnd,  corner.y),
// converts (az, r) → (lat, lon) via the geodesic formula, then to
// Web Mercator tile coordinates and on into clip space via u_matrix.
const GATE_VS = `
precision highp float;

attribute vec2  a_corner;
attribute float a_azStart;
attribute float a_azEnd;
attribute float a_rStart;
attribute float a_rEnd;
attribute float a_val;

uniform mat4  u_matrix;
uniform float u_stationLat;
uniform float u_stationLon;
uniform float u_minVal;
uniform float u_valRange;
uniform float u_cmapMin;
uniform float u_cmapRange;

varying float v_cmapT;

const float PI  = 3.141592653589793;
const float DEG = PI / 180.0;
const float RE  = 6371.0;

// Forward geodesic: given radar origin (lat1,lon1), bearing (az degrees from
// North), and slant range (km), return destination (lat, lon) in degrees.
vec2 destPoint(float lat1d, float lon1d, float azDeg, float rangeKm) {
  float la1 = lat1d * DEG;
  float lo1 = lon1d * DEG;
  float d   = rangeKm / RE;          // angular distance (radians)
  float b   = azDeg   * DEG;         // bearing (radians)
  float la2 = asin(clamp(
    sin(la1)*cos(d) + cos(la1)*sin(d)*cos(b), -1.0, 1.0));
  float lo2 = lo1 + atan(
    sin(b)*sin(d)*cos(la1),
    cos(d) - sin(la1)*sin(la2));
  return vec2(la2 / DEG, lo2 / DEG);  // (lat, lon) in degrees
}

// Spherical Web Mercator projection → tile coordinate in [0,1]²
vec2 mercator(float lon, float lat) {
  float x    = (lon + 180.0) / 360.0;
  float sinL = sin(lat * DEG);
  float y    = 0.5 - log((1.0 + sinL) / (1.0 - sinL)) / (4.0 * PI);
  return vec2(x, y);
}

void main() {
  // Select this corner's (az, range) from the gate's start/end bounds
  float az    = mix(a_azStart, a_azEnd, a_corner.x);
  float range = mix(a_rStart,  a_rEnd,  a_corner.y);

  vec2 ll     = destPoint(u_stationLat, u_stationLon, az, range);
  vec2 merc   = mercator(ll.y, ll.x);
  gl_Position = u_matrix * vec4(merc, 0.0, 1.0);

  // Decode uint16 value → physical value → colormap [0,1]
  float val = u_minVal + ((a_val - 1.0) / 65534.0) * u_valRange;
  v_cmapT   = clamp((val - u_cmapMin) / u_cmapRange, 0.0, 1.0);
}`;

// Fragment shader 
// Trivial: the vertex shader already computed the colormap t, so the fragment
// just looks it up in the 1-D colormap texture (sampled NEAREST for hard stops).
const GATE_FS = `
precision highp float;
varying   float       v_cmapT;
uniform   sampler2D   u_cmap;
void main() {
  gl_FragColor = texture2D(u_cmap, vec2(v_cmapT, 0.5));
}`;

// Grid vertex shader (ICON-D2 regular lat/lon cells)
// Each instance carries (lonStart, latStart, lonEnd, latEnd, encodedVal).
// The corner attribute interpolates lon/lat across the cell quad, then the
// result is projected through the same Mercator function as GATE_VS.
// The fragment shader is the shared GATE_FS above (colormap LUT lookup).
const GRID_VS = `
precision highp float;

attribute vec2  a_corner;
attribute float a_lonStart;
attribute float a_latStart;
attribute float a_lonEnd;
attribute float a_latEnd;
attribute float a_val;

uniform mat4  u_matrix;
uniform float u_minVal;
uniform float u_valRange;
uniform float u_cmapMin;
uniform float u_cmapRange;

varying float v_cmapT;

const float PI  = 3.141592653589793;
const float DEG = PI / 180.0;

vec2 mercator(float lon, float lat) {
  float x    = (lon + 180.0) / 360.0;
  float sinL = sin(lat * DEG);
  float y    = 0.5 - log((1.0 + sinL) / (1.0 - sinL)) / (4.0 * PI);
  return vec2(x, y);
}

void main() {
  float lon   = mix(a_lonStart, a_lonEnd, a_corner.x);
  float lat   = mix(a_latStart, a_latEnd, a_corner.y);
  vec2  merc  = mercator(lon, lat);
  gl_Position = u_matrix * vec4(merc, 0.0, 1.0);

  // ev is in [0, 65534]; reconstruct physical value then map to colormap [0,1]
  float val = u_minVal + (a_val / 65534.0) * u_valRange;
  v_cmapT   = clamp((val - u_cmapMin) / u_cmapRange, 0.0, 1.0);
}`;

// Shared WebGL helpers
function compileShader(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error('Shader: ' + gl.getShaderInfoLog(s));
  return s;
}
function linkProgram(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, vs); gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error('Link: ' + gl.getProgramInfoLog(p));
  return p;
}

// GateRadarLayer 
// Mapbox GL JS CustomLayerInterface implementation.
// Uses hardware-instanced rendering: one draw call renders all valid gates.
//
// Instance buffer layout (5 × float32 per gate, stride = 20 bytes):
//   offset  0: azStart  (degrees)
//   offset  4: azEnd    (degrees)
//   offset  8: rStart   (km from radar)
//   offset 12: rEnd     (km from radar)
//   offset 16: encodedVal (uint16 as float32, 1–65535)
//
// WebGL2 (gl.drawArraysInstanced) is used when available; otherwise the
// universally-supported ANGLE_instanced_arrays extension is used so the
// renderer works in both WebGL1 and WebGL2 contexts.
class GateRadarLayer {
  constructor(id, stationInfo, overlayData, cmapStops) {
    this.id            = id;
    this.type          = 'custom';
    this.renderingMode = '2d';
    this.stationInfo   = stationInfo;
    this.overlayData   = overlayData;
    this.cmapStops     = cmapStops;
    this._ready        = false;
    this._cmapTex      = null;
    this._cornerBuf    = null;  // shared unit-quad (6 verts × 2 floats)
    this._instBuf      = null;  // per-gate instance buffer
    this._program      = null;
    this._instCount    = 0;
    this._cmapMin      = 0;
    this._cmapRange    = 1;
    this._map          = null;
    this._inst         = null;  // { div, draw } instancing dispatch
  }

  // Return instancing function wrappers for the current GL context.
  // Tries WebGL2 native first, then ANGLE extension (WebGL1 / older drivers).
  _getInstFns(gl) {
    if (typeof WebGL2RenderingContext !== 'undefined' &&
        gl instanceof WebGL2RenderingContext) {
      return {
        div:  (loc, d) => gl.vertexAttribDivisor(loc, d),
        draw: (mode, first, count, n) => gl.drawArraysInstanced(mode, first, count, n),
      };
    }
    const ext = gl.getExtension('ANGLE_instanced_arrays');
    if (!ext) return null;
    return {
      div:  (loc, d) => ext.vertexAttribDivisorANGLE(loc, d),
      draw: (mode, first, count, n) => ext.drawArraysInstancedANGLE(mode, first, count, n),
    };
  }

  onAdd(map, gl) {
    this._map  = map;
    this._inst = this._getInstFns(gl);
    if (!this._inst) {
      console.error('GateRadarLayer: instanced arrays not supported by this GL context');
      return;
    }
    try {
      const vs = compileShader(gl, gl.VERTEX_SHADER,   GATE_VS);
      const fs = compileShader(gl, gl.FRAGMENT_SHADER, GATE_FS);
      this._program = linkProgram(gl, vs, fs);
    } catch(e) {
      console.error('GateRadarLayer shader error:', e);
      return;
    }

    // Unit quad: 6 vertices, two triangles.
    //   corner (0,0) → az=azStart, r=rStart  (near-left)
    //   corner (1,0) → az=azEnd,   r=rStart  (near-right)
    //   corner (0,1) → az=azStart, r=rEnd    (far-left)
    //   corner (1,1) → az=azEnd,   r=rEnd    (far-right)
    const corners = new Float32Array([
      0,0,  1,0,  0,1,   // triangle 1
      1,0,  1,1,  0,1,   // triangle 2
    ]);
    this._cornerBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._cornerBuf);
    gl.bufferData(gl.ARRAY_BUFFER, corners, gl.STATIC_DRAW);

    this._buildCmapTexture(gl);
    this._decodeAndBuildInstances(gl);
  }

  // Upload a 1-D RGBA8 colormap texture (1024 texels, NEAREST sampling).
  // NEAREST is intentional: hard colour-stop transitions (e.g. the green-to-
  // yellow boundary in reflectivity) should be crisp, not GPU-interpolated.
  _buildCmapTexture(gl) {
    const N     = 1024;
    const stops = this.cmapStops.slice().sort((a,b) => a.value - b.value);
    this._cmapMin   = stops[0].value;
    this._cmapRange = (stops[stops.length-1].value - stops[0].value) || 1;
    const rgba = new Uint8Array(N * 4);
    for (let i = 0; i < N; i++) {
      const t   = i / (N - 1);
      const val = this._cmapMin + t * this._cmapRange;
      const rgb = interpolateRgb(stops, val);
      rgba[i*4]   = rgb[0]; rgba[i*4+1] = rgb[1];
      rgba[i*4+2] = rgb[2]; rgba[i*4+3] = 255;
    }
    if (this._cmapTex) gl.deleteTexture(this._cmapTex);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, N, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this._cmapTex = tex;
  }

  // Decode the gzip+base64 uint16 polar payload and build the GPU instance
  // buffer.  Only valid (non-zero) gates are included, which typically
  // reduces the instance count by 60–80% vs the full (nrays×nbins) grid
  // and speeds up the draw call noticeably for sparse reflectivity fields.
  async _decodeAndBuildInstances(gl) {
    const info = this.overlayData;
    try {
      const { nrays, nbins, rscaleKm, minVal, maxVal } = info;
      const rstartKm = info.rstartKm || 0;
      const uint16   = await decodeGzipUint16(info.b64, nrays * nbins);
      const azStep   = 360.0 / nrays;

      // Pre-allocate for the worst case (all gates valid) then slice.
      const tmp  = new Float32Array(nrays * nbins * 5);
      let count  = 0;
      for (let ray = 0; ray < nrays; ray++) {
        const azS = ray       * azStep;
        const azE = (ray + 1) * azStep;
        for (let bin = 0; bin < nbins; bin++) {
          const v = uint16[ray * nbins + bin];
          if (v === 0) continue;   // nodata — skip entirely (no transparent quad)
          const off    = count * 5;
          tmp[off]     = azS;
          tmp[off + 1] = azE;
          tmp[off + 2] = rstartKm +  bin      * rscaleKm;
          tmp[off + 3] = rstartKm + (bin + 1) * rscaleKm;
          tmp[off + 4] = v;        // uint16 value passed as float to vertex shader
          count++;
        }
      }

      const instData = count < nrays * nbins
        ? tmp.subarray(0, count * 5)
        : tmp;

      if (this._instBuf) gl.deleteBuffer(this._instBuf);
      this._instBuf   = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this._instBuf);
      gl.bufferData(gl.ARRAY_BUFFER, instData, gl.STATIC_DRAW);
      this._instCount = count;
      this._ready     = true;
      if (this._map) this._map.triggerRepaint();
      // Re-raise metObs symbol layers so barbs/labels stay above radar.
      try {
        if (typeof maps !== 'undefined')
          maps.forEach(m => { if (m) raiseMetObsLayers(m); });
      } catch(_e) {}
    } catch(e) {
      console.error('GateRadarLayer decode error:', e);
      if (typeof warn !== 'undefined' && warn && !warn.textContent)
        warn.textContent = `Radar decode failed: ${e && e.message ? e.message : e}`;
    }
  }

  // Called by onColormapChanged — rebuilds the colormap texture in-place.
  updateColormap(gl, cmapStops) {
    this.cmapStops = cmapStops;
    this._buildCmapTexture(gl);
    if (this._map) this._map.triggerRepaint();
  }

  render(gl, matrix) {
    if (!this._ready || !this._instBuf || !this._cmapTex ||
        !this._program || !this._inst || this._instCount === 0) return;

    const prog = this._program;
    const inst = this._inst;

    gl.useProgram(prog);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // Uniforms
    const u = n => gl.getUniformLocation(prog, n);
    gl.uniformMatrix4fv(u('u_matrix'),      false, matrix);
    gl.uniform1f(u('u_stationLat'), this.stationInfo.lat);
    gl.uniform1f(u('u_stationLon'), this.stationInfo.lon);
    gl.uniform1f(u('u_minVal'),     this.overlayData.minVal);
    gl.uniform1f(u('u_valRange'),   this.overlayData.maxVal - this.overlayData.minVal);
    gl.uniform1f(u('u_cmapMin'),    this._cmapMin);
    gl.uniform1f(u('u_cmapRange'),  this._cmapRange);

    // Colormap texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._cmapTex);
    gl.uniform1i(u('u_cmap'), 0);

    // Corner buffer (non-instanced, one set of 6 verts shared by all gates)
    gl.bindBuffer(gl.ARRAY_BUFFER, this._cornerBuf);
    const aCorner = gl.getAttribLocation(prog, 'a_corner');
    gl.enableVertexAttribArray(aCorner);
    gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);
    inst.div(aCorner, 0);  // divisor 0 → advances per vertex (shared)

    // Instance buffer (5 floats per gate, stride = 20 bytes)
    gl.bindBuffer(gl.ARRAY_BUFFER, this._instBuf);
    const STRIDE = 20;  // 5 floats × 4 bytes
    const instAttrs = [
      ['a_azStart', 0],
      ['a_azEnd',   4],
      ['a_rStart',  8],
      ['a_rEnd',   12],
      ['a_val',    16],
    ];
    for (const [name, byteOffset] of instAttrs) {
      const loc = gl.getAttribLocation(prog, name);
      if (loc < 0) continue;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 1, gl.FLOAT, false, STRIDE, byteOffset);
      inst.div(loc, 1);  // divisor 1 → advances per instance (per gate)
    }

    inst.draw(gl.TRIANGLES, 0, 6, this._instCount);
  }

  onRemove(_map, gl) {
    if (this._program)   gl.deleteProgram(this._program);
    if (this._cmapTex)   gl.deleteTexture(this._cmapTex);
    if (this._cornerBuf) gl.deleteBuffer(this._cornerBuf);
    if (this._instBuf)   gl.deleteBuffer(this._instBuf);
    this._ready = false;
  }
}

// GridRadarLayer 
// Mapbox GL JS CustomLayerInterface for regular lat/lon grids (e.g. ICON-D2).
// Uses hardware-instanced rendering: one draw call, one quad per grid cell,
// colormap applied in the GRID_VS/GATE_FS shader pair.  Identical fidelity
// to GateRadarLayer — no Mapbox raster resampling artifacts.
//
// Instance buffer layout (5 × float32 per cell, stride = 20 bytes):
//   offset  0: lonStart  (degrees, west edge)
//   offset  4: latStart  (degrees, south edge)
//   offset  8: lonEnd    (degrees, east edge)
//   offset 12: latEnd    (degrees, north edge)
//   offset 16: encodedVal (uint16 as float32, 0–65534)
class GridRadarLayer {
  constructor(id, frameData, cmapStops) {
    this.id            = id;
    this.type          = 'custom';
    this.renderingMode = '2d';
    this.frameData     = frameData;
    this.cmapStops     = cmapStops;
    this._ready        = false;
    this._cmapTex      = null;
    this._cornerBuf    = null;
    this._instBuf      = null;
    this._program      = null;
    this._instCount    = 0;
    this._cmapMin      = 0;
    this._cmapRange    = 1;
    this._minVal       = 0;
    this._valRange     = 1;
    this._map          = null;
    this._inst         = null;
  }

  _getInstFns(gl) {
    if (typeof WebGL2RenderingContext !== 'undefined' &&
        gl instanceof WebGL2RenderingContext) {
      return {
        div:  (loc, d) => gl.vertexAttribDivisor(loc, d),
        draw: (mode, first, count, n) => gl.drawArraysInstanced(mode, first, count, n),
      };
    }
    const ext = gl.getExtension('ANGLE_instanced_arrays');
    if (!ext) return null;
    return {
      div:  (loc, d) => ext.vertexAttribDivisorANGLE(loc, d),
      draw: (mode, first, count, n) => ext.drawArraysInstancedANGLE(mode, first, count, n),
    };
  }

  onAdd(map, gl) {
    this._map  = map;
    this._inst = this._getInstFns(gl);
    if (!this._inst) {
      console.error('GridRadarLayer: instanced arrays not supported by this GL context');
      return;
    }
    try {
      const vs = compileShader(gl, gl.VERTEX_SHADER,   GRID_VS);
      const fs = compileShader(gl, gl.FRAGMENT_SHADER, GATE_FS);  // shared fragment shader
      this._program = linkProgram(gl, vs, fs);
    } catch(e) {
      console.error('GridRadarLayer shader error:', e);
      return;
    }

    // Unit quad: 6 vertices, two triangles.
    //   corner.x ∈ {0,1} → interpolates lon  (0=west edge, 1=east edge)
    //   corner.y ∈ {0,1} → interpolates lat  (0=south edge, 1=north edge)
    const corners = new Float32Array([
      0,0,  1,0,  0,1,
      1,0,  1,1,  0,1,
    ]);
    this._cornerBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._cornerBuf);
    gl.bufferData(gl.ARRAY_BUFFER, corners, gl.STATIC_DRAW);

    this._buildCmapTexture(gl);
    this._decodeAndBuild(gl);
  }

  // 1-D RGBA8 colormap texture, NEAREST sampling for crisp color stops.
  _buildCmapTexture(gl) {
    const N     = 1024;
    const stops = this.cmapStops.slice().sort((a,b) => a.value - b.value);
    this._cmapMin   = stops[0].value;
    this._cmapRange = (stops[stops.length-1].value - stops[0].value) || 1;
    const rgba = new Uint8Array(N * 4);
    for (let i = 0; i < N; i++) {
      const t   = i / (N - 1);
      const val = this._cmapMin + t * this._cmapRange;
      const rgb = interpolateRgb(stops, val);
      rgba[i*4]   = rgb[0]; rgba[i*4+1] = rgb[1];
      rgba[i*4+2] = rgb[2]; rgba[i*4+3] = 255;
    }
    if (this._cmapTex) gl.deleteTexture(this._cmapTex);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, N, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this._cmapTex = tex;
  }

  // Decode the valueEncoded PNG (R/G = uint16 big-endian, alpha = validity)
  // and build the GPU instance buffer.  Nodata cells are skipped entirely.
  async _decodeAndBuild(gl) {
    const { valueEncoded, bounds, gridShape } = this.frameData;
    // bounds: [[west,north],[east,north],[east,south],[west,south]]
    const west  = bounds[0][0], north = bounds[0][1];
    const east  = bounds[1][0], south = bounds[2][1];
    const ny    = gridShape[0], nx = gridShape[1];
    const dlon  = (east - west) / nx;
    const dlat  = (north - south) / ny;
    const mn    = Number(valueEncoded.min);
    const mx    = Number(valueEncoded.max);

    try {
      const raw = await new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          const c = document.createElement('canvas');
          c.width = nx; c.height = ny;
          c.getContext('2d').drawImage(img, 0, 0);
          resolve(c.getContext('2d').getImageData(0, 0, nx, ny).data);
        };
        img.onerror = reject;
        img.src = valueEncoded.url;
      });

      // Pre-allocate worst-case (all cells valid), then slice to actual count.
      const tmp = new Float32Array(nx * ny * 5);
      let count = 0;
      for (let row = 0; row < ny; row++) {
        const latEnd   = north - row * dlat;        // north edge of this cell row
        const latStart = north - (row + 1) * dlat;  // south edge
        for (let col = 0; col < nx; col++) {
          const j = (row * nx + col) * 4;
          if (raw[j+3] < 128) continue;  // nodata — skip, no transparent quad
          const ev     = (raw[j] << 8) | raw[j+1];  // 0–65534
          const off    = count * 5;
          tmp[off]     = west + col * dlon;          // lonStart
          tmp[off + 1] = latStart;
          tmp[off + 2] = west + (col + 1) * dlon;   // lonEnd
          tmp[off + 3] = latEnd;
          tmp[off + 4] = ev;
          count++;
        }
      }

      const instData = count < nx * ny ? tmp.subarray(0, count * 5) : tmp;
      if (this._instBuf) gl.deleteBuffer(this._instBuf);
      this._instBuf   = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this._instBuf);
      gl.bufferData(gl.ARRAY_BUFFER, instData, gl.STATIC_DRAW);
      this._instCount = count;
      this._minVal    = mn;
      this._valRange  = mx - mn || 1;
      this._ready     = true;
      if (this._map) this._map.triggerRepaint();
      try {
        if (typeof maps !== 'undefined')
          maps.forEach(m => { if (m) raiseMetObsLayers(m); });
      } catch(_e) {}
    } catch(e) {
      console.error('GridRadarLayer decode error:', e);
    }
  }

  updateColormap(gl, cmapStops) {
    this.cmapStops = cmapStops;
    this._buildCmapTexture(gl);
    if (this._map) this._map.triggerRepaint();
  }

  render(gl, matrix) {
    if (!this._ready || !this._instBuf || !this._cmapTex ||
        !this._program || !this._inst || this._instCount === 0) return;

    const prog = this._program;
    const inst = this._inst;

    gl.useProgram(prog);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const u = n => gl.getUniformLocation(prog, n);
    gl.uniformMatrix4fv(u('u_matrix'),    false, matrix);
    gl.uniform1f(u('u_minVal'),    this._minVal);
    gl.uniform1f(u('u_valRange'),  this._valRange);
    gl.uniform1f(u('u_cmapMin'),   this._cmapMin);
    gl.uniform1f(u('u_cmapRange'), this._cmapRange);

    // Colormap texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._cmapTex);
    gl.uniform1i(u('u_cmap'), 0);

    // Corner buffer (non-instanced — shared across all cell quads)
    gl.bindBuffer(gl.ARRAY_BUFFER, this._cornerBuf);
    const aCorner = gl.getAttribLocation(prog, 'a_corner');
    gl.enableVertexAttribArray(aCorner);
    gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);
    inst.div(aCorner, 0);

    // Instance buffer (5 floats per cell, stride = 20 bytes)
    gl.bindBuffer(gl.ARRAY_BUFFER, this._instBuf);
    const STRIDE = 20;
    for (const [name, byteOffset] of [
      ['a_lonStart',  0],
      ['a_latStart',  4],
      ['a_lonEnd',    8],
      ['a_latEnd',   12],
      ['a_val',      16],
    ]) {
      const loc = gl.getAttribLocation(prog, name);
      if (loc < 0) continue;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 1, gl.FLOAT, false, STRIDE, byteOffset);
      inst.div(loc, 1);
    }

    inst.draw(gl.TRIANGLES, 0, 6, this._instCount);
  }

  onRemove(_map, gl) {
    if (this._program)   gl.deleteProgram(this._program);
    if (this._cmapTex)   gl.deleteTexture(this._cmapTex);
    if (this._cornerBuf) gl.deleteBuffer(this._cornerBuf);
    if (this._instBuf)   gl.deleteBuffer(this._instBuf);
    this._ready = false;
  }
}

// Active radar layers (one per panel) 
const radarLayers = [null, null, null, null];

// Cartesian PNG overlay for derived products
const cartesianDecodeCache = {};

async function buildCartesianUrl(sn, pk) {
  const pd = ((APP.rendered[sn] || {}).products || {})[pk];
  if (!pd || pd.type !== 'cartesian' || !pd.valueEncoded || !pd.valueEncoded.url) return null;
  const sig = `${sn}|${pk}|${JSON.stringify(getColormap(pk))}`;
  return _buildEncodedCartesianUrl(pd.valueEncoded, pk, sig);
}

async function fetchIconD2Inventory(force=false) {
  if (location.protocol === 'file:') throw new Error('ICON-D2 requires --serve mode');
  const suffix = force ? '&force=1' : '';
  const resp = await fetch(`icon_d2_inventory?product=${encodeURIComponent(iconD2Product)}&_ts=${Date.now()}${suffix}`, {cache:'no-store'});
  if (!resp.ok) throw new Error(`ICON-D2 inventory HTTP ${resp.status}`);
  iconD2Inventory = await resp.json();
  return iconD2Inventory;
}

function findNearestIconD2FrameIndex(targetIso) {
  const frames = (iconD2Inventory && iconD2Inventory.frames) || [];
  if (!frames.length) return -1;
  const targetMs = targetIso ? new Date(targetIso).getTime() : Date.now();
  if (!Number.isFinite(targetMs)) return 0;
  let bestIdx = 0;
  let bestDiff = Math.abs(new Date(frames[0].validIso).getTime() - targetMs);
  for (let i = 1; i < frames.length; i++) {
    const diff = Math.abs(new Date(frames[i].validIso).getTime() - targetMs);
    if (diff < bestDiff) { bestDiff = diff; bestIdx = i; }
  }
  return bestIdx;
}

function updateIconD2Ui() {
  if (iconD2Btn) iconD2Btn.classList.toggle('icon-active', iconD2Enabled);
  if (iconD2ProdSel) iconD2ProdSel.style.display = iconD2Enabled ? '' : 'none';
  if (product2Sel) product2Sel.disabled = !!iconD2Enabled;
  if (countSel) countSel.disabled = !!iconD2Enabled;
  _updateTimeNavFocusUi();
}

async function loadIconD2FrameByIndex(index) {
  const frames = (iconD2Inventory && iconD2Inventory.frames) || [];
  if (!frames.length) {
    iconD2CurrentIndex = -1;
    iconD2CurrentFrame = null;
    updatePanelMeta(1);
    return;
  }
  const nextIndex = Math.max(0, Math.min(index, frames.length - 1));
  const frameMeta = frames[nextIndex];
  iconD2Loading = true;
  updatePanelMeta(1);
  try {
    const resp = await fetch(`icon_d2_frame?key=${encodeURIComponent(frameMeta.key)}&product=${encodeURIComponent(iconD2Product)}&_ts=${Date.now()}`, {cache:'no-store'});
    if (!resp.ok) throw new Error(`ICON-D2 frame HTTP ${resp.status}`);
    iconD2CurrentFrame = await resp.json();
    iconD2CurrentIndex = nextIndex;
    warn.textContent = '';
    void applyOverlayToMap(maps[1], 1);
    updatePanelMeta(1);
    renderLegend(1);
  } catch (err) {
    warn.textContent = `ICON-D2 fetch failed: ${err}`;
  } finally {
    iconD2Loading = false;
    updatePanelMeta(1);
  }
}

async function syncIconD2ToRadar(forceInventory=false) {
  try {
    const inv = iconD2Inventory && !forceInventory ? iconD2Inventory : await fetchIconD2Inventory(forceInventory);
    if (!inv || !inv.frames || !inv.frames.length) throw new Error('No completed ICON-D2 runs available');
    const idx = findNearestIconD2FrameIndex(_currentRadarIso());
    if (idx < 0) throw new Error('No ICON-D2 frames available');
    iconD2SyncedToRadar = true;
    await loadIconD2FrameByIndex(idx);
  } catch (err) {
    warn.textContent = `ICON-D2 unavailable: ${err}`;
  }
}

async function stepIconD2Frame(delta) {
  if (!iconD2Enabled) return;
  if (!iconD2Inventory || !iconD2Inventory.frames || !iconD2Inventory.frames.length) {
    await syncIconD2ToRadar(false);
    return;
  }
  iconD2SyncedToRadar = false;
  await loadIconD2FrameByIndex(iconD2CurrentIndex + delta);
}

async function setIconD2Enabled(enabled) {
  if (!!enabled === !!iconD2Enabled) return;
  iconD2Enabled = !!enabled;
  if (iconD2Enabled) {
    timeNavFocus = 'radar';
    iconD2PrevProduct2 = product2Sel.value || iconD2PrevProduct2;
    iconD2PrevCount = countSel.value || iconD2PrevCount;
    product2Sel.value = 'reflectivity';
    tiltIdx[1] = 0;
    _onProductChanged(1);
    countSel.value = '2';
    countSel.onchange();
    updateIconD2Ui();
    document.getElementById('h2').textContent = 'ICON-D2';
    document.getElementById('t2').textContent = _iconD2ProductLabel() + ' • loading…';
    iconD2CurrentFrame = null;
    void applyOverlayToMap(maps[1], 1);
    await syncIconD2ToRadar(false);
  } else {
    timeNavFocus = 'radar';
    if (APP.products[iconD2PrevProduct2]) product2Sel.value = iconD2PrevProduct2;
    _onProductChanged(1);
    if (iconD2PrevCount === '1' || iconD2PrevCount === '2' || iconD2PrevCount === '4') countSel.value = iconD2PrevCount;
    countSel.onchange();
    updateIconD2Ui();
    refreshAllMaps(false);
    renderAllLegends();
  }
}

function _iconD2ProductLabel() {
  const opt = ICON_D2_PRODUCT_OPTIONS.find(o => o.key === iconD2Product);
  return opt ? opt.label : iconD2Product;
}

function _iconD2Colormap() {
  // Prefer the colormap reported by the live inventory (server is authoritative);
  // fall back to the client-side mapping table.
  if (iconD2Inventory && iconD2Inventory.colormap) return iconD2Inventory.colormap;
  const opt = ICON_D2_PRODUCT_OPTIONS.find(o => o.key === iconD2Product);
  return opt ? opt.colormap : 'reflectivity';
}

// Wire up the ICON-D2 product selector dropdown
if (iconD2ProdSel) {
  iconD2ProdSel.value = iconD2Product;
  iconD2ProdSel.addEventListener('change', async () => {
    const newProduct = iconD2ProdSel.value;
    if (newProduct === iconD2Product) return;
    iconD2Product = newProduct;
    iconD2Inventory = null;
    iconD2CurrentFrame = null;
    iconD2CurrentIndex = -1;
    document.getElementById('t2').textContent = _iconD2ProductLabel() + ' • loading…';
    void applyOverlayToMap(maps[1], 1);
    renderLegend(1);
    await syncIconD2ToRadar(false);
  });
}

// Map layer helpers
function mapPrefix(i) { return `panel-${i}`; }
function removeLayerIfExists(m, id)  { if (m.getLayer  && m.getLayer(id))  m.removeLayer(id);  }
function removeSourceIfExists(m, id) { if (m.getSource && m.getSource(id)) m.removeSource(id); }
function getOverlayBeforeLayerId(m, cachedLayerIds) {
  try {
    const style = m.getStyle && m.getStyle();
    const layers = (style && style.layers) || [];
    // First pass: explicit admin/country boundary IDs that radar must render below.
    // We intentionally avoid settlement labels here so boundary layers (admin-1/admin-2)
    // stay above radar in custom styles.
    // Also anchor before 'water' so the water fill renders above the radar overlay.
    for (const l of layers) {
      const id = l.id || '';
      const lid = id.toLowerCase();
      if (
        id === 'water (1)' ||
        id === 'country-boundaries' ||
        id === 'admin-0-boundary' ||
        id === 'admin-0-boundary-bg' ||
        id === 'admin-1-boundary' ||
        id === 'admin-2-boundary' ||
        /(?:^|[-_])(admin[-_][012]|country)[-_]?boundar(?:y|ies)(?:$|[-_])/.test(lid)
      ) return id;
    }
    for (const l of layers) {
      const id = l.id || '';
      // Skip our own overlay layers
      if (id.startsWith('metobs-') || id.startsWith('panel-') ||
          id.startsWith('lightning-') || id.startsWith('sat-') ||
          id.startsWith('cs-')) continue;
      // Country/admin boundaries (classic Mapbox style names)
      if (/boundary|admin|border|country/.test(id)) return id;
      // Standard style label layers
      if (id === 'road-label' || id === 'transit-label' || id === 'poi-label') return id;
      // Any road / label / place layers (broader match for custom styles)
      if (/road|street|highway|place|label|city|town/.test(id)) return id;
      // Fallback: first symbol layer with text
      if (l.type === 'symbol' && l.layout && l.layout['text-field']) return id;
    }
    // Last resort: first line layer (roads drawn as lines in simple styles)
    for (const l of layers) {
      const id = l.id || '';
      if (id.startsWith('metobs-') || id.startsWith('panel-') ||
          id.startsWith('lightning-') || id.startsWith('sat-') ||
          id.startsWith('cs-')) continue;
      if (l.type === 'line') return id;
    }
  } catch(e) { /* style not ready */ }
  return null;
}
function buildRangeRingGeoJSON(station, radiusKm, pts = 180) {
  const R=6371, lat1=station.lat*Math.PI/180, lon1=station.lon*Math.PI/180, d=radiusKm/R, coords=[];
  for (let i=0; i<=pts; i++){
    const b = (i/pts)*2*Math.PI;
    const lat2 = Math.asin(Math.sin(lat1)*Math.cos(d)+Math.cos(lat1)*Math.sin(d)*Math.cos(b));
    const lon2 = lon1+Math.atan2(Math.sin(b)*Math.sin(d)*Math.cos(lat1),Math.cos(d)-Math.sin(lat1)*Math.sin(lat2));
    coords.push([lon2*180/Math.PI, lat2*180/Math.PI]);
  }
  return {type:'FeatureCollection',features:[{type:'Feature',geometry:{type:'LineString',coordinates:coords},properties:{}}]};
}


function removeTvsFromMap(map, prefix) {
  removeLayerIfExists(map, `${prefix}-tvs-lyr`);
  removeSourceIfExists(map, `${prefix}-tvs-src`);
}

async function ensureTvsImages(map) {
  const icons = APP.tvsIcons || {};
  const jobs = Object.entries(icons).map(([level, dataUrl]) => {
    const imageId = `tvs-icon-${level}`;
    if (map.hasImage && map.hasImage(imageId)) return Promise.resolve();
    return new Promise(resolve => {
      const img = new Image();
      img.onload = () => {
        try {
          if (!map.hasImage(imageId)) map.addImage(imageId, img);
        } catch (e) {
          console.warn('TVS addImage failed', e);
        }
        resolve();
      };
      img.onerror = () => resolve();
      img.src = dataUrl;
    });
  });
  await Promise.all(jobs);
}

async function applyTvsToMap(map, prefix, markers, before) {
  removeTvsFromMap(map, prefix);
  if (!markers || !markers.length) return;

  await ensureTvsImages(map);

  const fc = {
    type: 'FeatureCollection',
    features: markers
      .filter(m => Number.isFinite(m.lon) && Number.isFinite(m.lat) && Number.isFinite(m.level))
      .map(m => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [m.lon, m.lat] },
        properties: {
          level: m.level,
          iconId: `tvs-icon-${Math.max(1, Math.min(5, Math.round(m.level)))}`,
        },
      })),
  };

  map.addSource(`${prefix}-tvs-src`, { type: 'geojson', data: fc });
  _addLayerOnTop(map, {
    id: `${prefix}-tvs-lyr`,
    type: 'symbol',
    source: `${prefix}-tvs-src`,
    layout: {
      'icon-image': ['get', 'iconId'],
      'icon-size': 0.025,   // icons are 3000×3000px; 0.025 renders at ~75px
      'icon-rotate': 180,   // source icons are upside-down; rotate to correct orientation
      'icon-allow-overlap': true,
      'icon-ignore-placement': true,
    },
  });
}


// ESWD severe-weather overlay
function removeEswdFromMap(map, prefix) {
  removeLayerIfExists(map, `${prefix}-eswd-labels`);
  removeLayerIfExists(map, `${prefix}-eswd-circles`);
  removeSourceIfExists(map, `${prefix}-eswd-src`);
}

async function applyEswdToMap(map, prefix, geojson) {
  removeEswdFromMap(map, prefix);
  if (!geojson || !geojson.features || !geojson.features.length) return;

  map.addSource(`${prefix}-eswd-src`, { type: 'geojson', data: geojson });

  // Circle glow behind each marker
  _addLayerOnTop(map, {
    id:     `${prefix}-eswd-circles`,
    type:   'circle',
    source: `${prefix}-eswd-src`,
    paint: {
      'circle-radius':       10,
      'circle-color':        ['get', 'colour'],
      'circle-opacity':      0.75,
      'circle-stroke-width': 1.5,
      'circle-stroke-color': '#ffffff',
    },
  });

  // Emoji label
  _addLayerOnTop(map, {
    id:     `${prefix}-eswd-labels`,
    type:   'symbol',
    source: `${prefix}-eswd-src`,
    layout: {
      'text-field':                ['get', 'label'],
      'text-size':                 14,
      'text-allow-overlap':        true,
      'text-ignore-placement':     true,
      'text-anchor':               'center',
    },
    paint: {
      'text-color': '#ffffff',
      'text-halo-color': '#000000',
      'text-halo-width': 1,
    },
  });

  // Popup on click
  map.on('click', `${prefix}-eswd-circles`, (e) => {
    const f = e.features && e.features[0];
    if (!f) return;
    const p = f.properties;
    const lines = [
      `<strong>ESWD – ${p.type}</strong>`,
      p.time       ? `Time: ${p.time.replace('T',' ').replace('Z',' UTC')}`  : '',
      p.location   ? `Location: ${p.location}`   : '',
      p.description? `${p.description}`           : '',
      p.qc         ? `QC: ${p.qc}`               : '',
    ].filter(Boolean).join('<br>');
    new mapboxgl.Popup({ closeButton: true, maxWidth: '280px' })
      .setLngLat(e.lngLat)
      .setHTML(`<div style="font-size:12px;line-height:1.5">${lines}</div>`)
      .addTo(map);
  });
  map.on('mouseenter', `${prefix}-eswd-circles`, () => { map.getCanvas().style.cursor = 'pointer'; });
  map.on('mouseleave', `${prefix}-eswd-circles`, () => { map.getCanvas().style.cursor = ''; });
}

async function _fetchAndApplyEswd(archiveDt) {
  // archiveDt: ISO string (archive mode) or null (live = today)
  const url = archiveDt
    ? `eswd_reports?dt=${encodeURIComponent(archiveDt)}`
    : 'eswd_reports';
  try {
    const resp = await fetch(url, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`ESWD ${resp.status}`);
    const gj = await resp.json();
    const m = gj && gj.meta;
    if (m && (m.hint || m.api_message || m.error)) {
      const msg = m.hint || m.api_message || m.error;
      if (!window._eswdMetaWarned || window._eswdMetaWarned !== msg) {
        window._eswdMetaWarned = msg;
        console.warn('[ESWD]', msg);
      }
    }
    _eswdGeoJSON = gj;
  } catch (err) {
    console.warn('[ESWD] fetch failed:', err);
    _eswdGeoJSON = null;
  }
  // Apply (or clear) on all ready maps
  maps.forEach((map, i) => {
    if (!mapReady[i]) return;
    const prefix = mapPrefix(i);
    if (eswdOverlayEnabled && _eswdGeoJSON) {
      applyEswdToMap(map, prefix, _eswdGeoJSON);
    } else {
      removeEswdFromMap(map, prefix);
    }
  });
}

// Main overlay application
async function applyOverlayToMap(map, mapIndex) {
  if (!mapReady[mapIndex]) return;

  const sn     = stationSel.value;
  const pk     = panelProducts[mapIndex]();
  const prefix = mapPrefix(mapIndex);
  // Layer ordering: use cached values computed at style-load time.
  const _slotStyle    = _mapIsSlot[mapIndex];
  // Always anchor radar/rings to the cached map boundary layer (e.g. country-boundaries).
  // Do NOT use _firstMetObsLayerId here: metobs layers live above country-boundaries
  // (added via _addLayerOnTop), so inserting radar "before" a metobs layer would
  // place it above country-boundaries and settlement-labels.
  const _classicBefore = _slotStyle ? undefined
                       : (_mapBeforeId[mapIndex] || undefined);

  // Remove existing polar radar layer
  if (radarLayers[mapIndex]) {
    try { map.removeLayer(radarLayers[mapIndex].id); } catch(e){}
    radarLayers[mapIndex] = null;
  }
  removeLayerIfExists(map,  `${prefix}-cart-lyr`);
  removeSourceIfExists(map, `${prefix}-cart-src`);
  removeLayerIfExists(map,  `${prefix}-ring-lyr`);
  removeSourceIfExists(map, `${prefix}-ring-src`);
  removeTvsFromMap(map, prefix);

  const sd = APP.rendered[sn] || {};
  const pd = (sd.products || {})[pk];
  const iconMode = iconD2Enabled && mapIndex === 1;

  // Resolve the overlay for the current tilt
  let activeOverlay = pd && pd.overlayData;
  if (!iconMode && pd && pd.allTiltOverlays && pd.allTiltOverlays.length) {
    const clampedIdx = Math.max(0, Math.min(tiltIdx[mapIndex], pd.allTiltOverlays.length - 1));
    tiltIdx[mapIndex] = clampedIdx;
    activeOverlay = pd.allTiltOverlays[clampedIdx];
  }

  // Layer ordering: satellite must be added BEFORE radar so it renders below it.
  // Both satellite and radar use the same anchor (slot:'bottom' or beforeId),
  // so whichever is added first is rendered lower.
  if (anySatelliteEnabled()) applySatelliteToMap(map, mapIndex);
  else removeSatelliteFromMap(map, mapIndex);

  if (iconMode) {
    // Render ICON-D2 using the same instanced WebGL pipeline as polar radar
    // (GridRadarLayer) — one quad per grid cell, colormap in GPU shader.
    // No Mapbox raster source, no bilinear resampling artifacts.
    const frame = iconD2CurrentFrame;
    if (frame && frame.valueEncoded && frame.valueEncoded.url &&
        frame.bounds && frame.gridShape) {
      const layer = new GridRadarLayer(
        `${prefix}-icon-grid`,
        frame,
        getColormap(_iconD2Colormap()),
      );
      radarLayers[mapIndex] = layer;
      try {
        if (_slotStyle) { layer.slot = 'bottom'; map.addLayer(layer); }
        else { map.addLayer(layer, _classicBefore); }
      } catch(e) { console.error('addLayer icon-grid', e); }
    }
  } else if (pd && pd.type === 'polar' && activeOverlay && activeOverlay.b64) {
    const layer = new GateRadarLayer(
      `${prefix}-polar`,
      APP.stations[sn],
      activeOverlay,
      getColormap(pk),
    );
    radarLayers[mapIndex] = layer;
    try {
      if (_slotStyle) { layer.slot = 'bottom'; map.addLayer(layer); }
      else { map.addLayer(layer, _classicBefore); }
    } catch(e) { console.error('addLayer polar', e); }
  } else if (pd && pd.type === 'cartesian') {
    const imgUrl = await buildCartesianUrl(sn, pk);
    if (imgUrl && sd.bounds) {
      map.addSource(`${prefix}-cart-src`, {type:'image', url:imgUrl, coordinates:sd.bounds});
      const spec = {id:`${prefix}-cart-lyr`, type:'raster', source:`${prefix}-cart-src`, paint:{'raster-opacity':1}};
      if (_slotStyle) map.addLayer({...spec, slot:'bottom'});
      else map.addLayer(spec, _classicBefore);
    }
  }

  // TVS marker overlay
  const tvsData = (APP.rendered[sn] || {}).tvsData;
  if (!iconMode && tvsOverlayEnabled && tvsData && tvsData.tvsMarkers && tvsData.tvsMarkers.length) {
    await applyTvsToMap(map, prefix, tvsData.tvsMarkers, _classicBefore);
  }

  // ESWD overlay (uses separately-fetched GeoJSON, not per-station payload)
  if (eswdOverlayEnabled && _eswdGeoJSON) {
    await applyEswdToMap(map, prefix, _eswdGeoJSON);
  } else {
    removeEswdFromMap(map, prefix);
  }

  // Range rings
  const si = APP.stations[sn];
  if (!iconMode && si && layerSettings.rangeRings) {
    const rr = (si.provider === 'dwd') ? 180
             : (si.provider === 'fmi') ? 250
             : (typeof si.range_km === 'number' ? si.range_km : 120);
    map.addSource(`${prefix}-ring-src`, {type:'geojson', data:buildRangeRingGeoJSON(si, rr)});
    const ringSpec = {id:`${prefix}-ring-lyr`, type:'line', source:`${prefix}-ring-src`,
      paint:{'line-color':'#FFFFFF','line-width':layerSettings.ringWidth,'line-opacity':0.95}};
    if (_slotStyle) map.addLayer({...ringSpec, slot:'bottom'});
    else map.addLayer(ringSpec, _classicBefore);
  }

  if (iconMode) {
    document.getElementById(`h${mapIndex+1}`).textContent = 'ICON-D2';
    document.getElementById(`t${mapIndex+1}`).textContent =
      iconD2CurrentFrame
        ? `${_iconD2ProductLabel()} • ${iconD2CurrentFrame.validIso || 'n/a'}`
        : (iconD2Loading ? `${_iconD2ProductLabel()} • loading…` : `${_iconD2ProductLabel()} • n/a`);
    renderLegend(mapIndex);
    raiseMetObsLayers(map);
    return;
  }

  document.getElementById(`h${mapIndex+1}`).textContent = sn;
  document.getElementById(`t${mapIndex+1}`).textContent =
    `${APP.products[pk]} • ${sd.datetime || 'n/a'}`;
  const owner = ((APP.stations[sn]||{}).provider || 'dmi').toUpperCase();
  const tiltOvs = (pd && pd.allTiltOverlays) ? pd.allTiltOverlays : (pd && pd.overlayData ? [pd.overlayData] : []);
  const tiltCount = tiltOvs.length;
  const curTilt   = tiltCount ? Math.max(0, Math.min(tiltIdx[mapIndex], tiltCount - 1)) : 0;
  const tilt = (activeOverlay && typeof activeOverlay.elevationDeg === 'number')
    ? `${activeOverlay.elevationDeg.toFixed(1)}°`
    : (pd && pd.overlayData && typeof pd.overlayData.elevationDeg === 'number')
    ? `${pd.overlayData.elevationDeg.toFixed(1)}°`
    : 'n/a';
  const tiltLabel = `Tilt ${tilt}`;
  const sourceFile = (pd && pd.sourceFile) ? pd.sourceFile : '';
  const metaLeftEl  = document.getElementById(`meta${mapIndex+1}-left`);
  const metaRightEl = document.getElementById(`meta${mapIndex+1}-right`);
  if (metaLeftEl)  metaLeftEl.textContent  = `${owner}  •  ${tiltLabel}  •  ${sd.datetime || 'n/a'}`;
  if (metaRightEl) {
    let rightText = sourceFile;
    const provider = ((APP.stations[sn] || {}).provider || '').toLowerCase();
    if (sd && sd.debug) {
      const matched = pd && pd.matchedQty;
      if (matched && sd.debug.stats && sd.debug.stats[matched]) {
        const st = sd.debug.stats[matched];
        const fin = (st && typeof st.finite === 'number') ? st.finite : 0;
        const tot = (st && typeof st.total === 'number') ? st.total : 0;
        const statText = `fin ${fin}/${tot}`;
        rightText = rightText ? `${rightText}  •  ${statText}` : statText;
      }
      if ((tilt === 'n/a' || provider === 'knmi')
          && Array.isArray(sd.debug.quantities) && sd.debug.quantities.length) {
        const qtys = sd.debug.quantities;
        const sample = qtys.slice(0, 6).join(',');
        const more = qtys.length > 6 ? '…' : '';
        const qtyText = `qtys: ${sample}${more}`;
        rightText = rightText ? `${rightText}  •  ${qtyText}` : qtyText;
      }
      if (provider === 'knmi' && matched) {
        const mText = `match: ${matched}`;
        rightText = rightText ? `${rightText}  •  ${mText}` : mText;
      }
    }
    metaRightEl.textContent = rightText;
  }
  renderLegend(mapIndex);

  // Always raise metObs layers above the newly-added radar layer
  raiseMetObsLayers(map);
}

  function refreshAllMaps(recenter = true) {
    maps.forEach((m, i) => void applyOverlayToMap(m, i));
    if (recenter) {
      const s = APP.stations[stationSel.value];
      if (s) maps.forEach(m => m.easeTo({center:[s.lon,s.lat],zoom:7.1,duration:450}));
    }
    updateAllPanelMeta();
  }

// ── Colormap change: rebuild WebGL colormap texture ───────────────────────
function onColormapChanged(pk) {
  maps.forEach((map, idx) => {
    if (panelProducts[idx]() !== pk) return;
    const layer = radarLayers[idx];
    if (!layer) return;
    const gl = map.painter && map.painter.context && map.painter.context.gl;
    if (gl) layer.updateColormap(gl, getColormap(pk));
  });
  Object.keys(cartesianDecodeCache).forEach(k => { if (k.includes(`|${pk}|`)) delete cartesianDecodeCache[k]; });
  renderAllLegends();
  refreshAllMaps(false);
}

// Legend
function uniqueSortedStops(pk){const s=getColormap(pk).map(i=>i.value);return[...new Set(s)].sort((a,b)=>a-b);}
function buildLegendTicks(mn,mx,step){const first=Math.ceil(mn/step)*step,ticks=[];for(let v=first;v<=mx+1e-9;v+=step)ticks.push(Number(v.toFixed(6)));if(!ticks.includes(mn))ticks.unshift(mn);if(!ticks.includes(mx))ticks.push(mx);return[...new Set(ticks)].sort((a,b)=>b-a);}
function renderLegend(idx){
  // When ICON-D2 is active on panel 2, override the product key with the
  // ICON-D2 product's colormap so the legend reflects what's being rendered.
  const rawPk = panelProducts[idx]();
  const pk = (iconD2Enabled && idx === 1) ? _iconD2Colormap() : rawPk;
  const stops=uniqueSortedStops(pk),el=document.getElementById(`legend${idx+1}`);
  if(!el||!stops.length) return;
  if(!layerSettings.legends){ el.innerHTML=''; return; }
  const mn=stops[0],mx=stops[stops.length-1],step=getLegendStep(pk)||Math.max(1,Math.round((mx-mn)/10));
  const ticks=buildLegendTicks(mn,mx,step);
  const unitByProduct={reflectivity:'dBZ',velocity:'m/s',correlation:'%',spectrum_width:'m/s',zdr:'dBZ',kdp:'°/km',echo_tops:'km',echo_tops_0:'km',nrot:'m/s',srv:'m/s',meso:'%',azimuthal_shear:'×10⁻³/s',mesh:'cm',cape_ml:'J/kg'};
  const unit=unitByProduct[pk]||'';
  const range=mx-mn||1;
  const gStops=[];
  // Build the CSS gradient by mapping every raw colormap stop directly to its
  // position percentage.  Two consecutive stops sharing the same value produce
  // the same % in the gradient string, which CSS treats as an instant hard stop —
  // this correctly handles both encoding conventions used in the colortables:
  //   Exact duplicate values  (e.g. reflectivity: [40,amber],[40,orange])
  //   N / N+0.999 step pairs  (e.g. echo_tops:    [5,orange],[5.999,orange],[6,red])
  const cmapStops=getColormap(pk).slice().sort((a,b)=>a.value-b.value);
  cmapStops.forEach(s=>{
    const pct=((s.value-mn)/range*100).toFixed(2);
    const col=`rgb(${s.color[0]},${s.color[1]},${s.color[2]})`;
    gStops.push(`${col} ${pct}%`);
  });
  const tickHtml=ticks.map(v=>{const y=((mx-v)/(range||1))*100,lab=Number.isInteger(v)?`${v}`:v.toFixed(1);return`<div class='legend-tick' style='top:${y}%;'><span class='legend-tick-label'>${lab}</span><span class='legend-tick-line'></span></div>`;}).join('');
  el.innerHTML=`<div class='legend-scale'><div class='legend-bar' style='background:linear-gradient(to top,${gStops.join(',')});'></div>${tickHtml}</div>`;
}
function renderAllLegends(){panelEls.forEach((_el, idx) => renderLegend(idx));}

//Station markers 
// 3-4 letter radar station acronyms
const STATION_ACRONYMS = {
  // Denmark
  'Rømø':          'RØM',
  'Sindal':        'SIN',
  'Stevns':        'STV',
  'Bornholm':      'BOR',
  'Samsø':         'SAM',
  // Sweden
  'Ängelholm':     'ÄNG',
  'Åtvidaberg':    'ÅTV',
  'Bålsta':        'BÅL',
  'Hemse':         'HEM',
  'Hudiksvall':    'HUD',
  'Karlskrona':    'KRN',
  'Kiruna':        'KIR',
  'Leksand':       'LEK',
  'Luleå':         'LUL',
  'Örnsköldsvik':  'ÖRN',
  'Östersund':     'ÖSD',
  'Vara':          'VAR',
  // Finland
  'Korppoo':       'KOR',
  'Vihti':         'VIH',
  'Anjalankoski':  'ANJ',
  'Kankaanpää':    'KAN',
  'Kesälahti':     'KES',
  'Petäjävesi':    'PET',
  'Kuopio':        'KUO',
  'Vimpeli':       'VIM',
  'Nurmes':        'NUR',
  'Utajärvi':      'UTA',
  'Luosto':        'LUO',
  'Kaunispää':     'KAU',
  // Netherlands
  'Herwijnen':     'HWN',
  'Den Helder':    'DNH',
  // Czech Republic
  'Brdy':          'BRD',
  'Skalky':        'SKA',
  // Slovakia
  'Javorník':          'JAV',
  'Kojšovská hoľa':    'KOJ',
  'Kubínska hoľa':     'KUB',
  'Lazany':            'LAZ',
  // Austria
  'Hochficht':     'HOC',
  // Cabauw
  'Cabauw':        'CAB',
};

function stationAcronym(name) {
  if (STATION_ACRONYMS[name]) return STATION_ACRONYMS[name];
  // DWD stations: name is "DWD XXX" – use the site code portion uppercased
  const dwdMatch = name.match(/^DWD\s+([A-Za-z]{2,4})$/i);
  if (dwdMatch) return dwdMatch[1].toUpperCase();
  // fallback: first 3 chars uppercased
  return name.replace(/[^A-Za-zÀ-ÖØ-öø-ÿ]/g, '').slice(0, 3).toUpperCase();
}

function stationMarkerEl(sel, name) {
  const size = layerSettings.stationSize || 6;
  const sz   = sel ? size : Math.max(2, size - 2);
  const acr  = stationAcronym(name);
  const notReporting = APP.rendered[name]?.reporting === false;

  const wrap = document.createElement('div');
  wrap.style.cssText = 'display:flex;flex-direction:column;align-items:center;cursor:pointer;';

  const dot = document.createElement('div');
  dot.style.cssText = `width:${sz}px;height:${sz}px;border-radius:50%;`
    + (sel
      ? (notReporting
          ? 'background:#ff4444;box-shadow:0 0 0 2px #aa0000,0 0 5px #ff2222;'
          : 'background:#00ffff;box-shadow:0 0 0 2px #007aaa,0 0 5px #00ccff;')
      : (notReporting
          ? 'background:#cc2222;box-shadow:0 0 0 1px #800000;'
          : 'background:#00e5e5;box-shadow:0 0 0 1px #005f80;'));

  const lbl = document.createElement('div');
  lbl.textContent = acr;
  lbl.style.cssText = 'font:bold 8px Consolas,monospace;color:#00ffff;'
    + 'text-shadow:0 0 3px #000,-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,1px 1px 0 #000;'
    + 'margin-top:1px;line-height:1;pointer-events:none;white-space:nowrap;'
    + (notReporting ? 'color:#ff8888;' : (sel ? 'color:#ffffff;' : ''));

  wrap.appendChild(dot);
  wrap.appendChild(lbl);
  return wrap;
}

const stationMarkers=Array.from({length: panelEls.length}, () => []);
function rebuildMarkers(map,idx){
  stationMarkers[idx].forEach(m=>m.remove()); stationMarkers[idx]=[];
  Object.entries(APP.stations).forEach(([name,s])=>{
    const mk=new mapboxgl.Marker(stationMarkerEl(name===stationSel.value, name)).setLngLat([s.lon,s.lat]).setPopup(new mapboxgl.Popup().setHTML(`<b>${name}</b><br>Click to select`)).addTo(map);
    // Track pointer-down position so we can reject drag-initiated "clicks"
    let _pdX=0,_pdY=0;
    mk.getElement().addEventListener('pointerdown',e=>{_pdX=e.clientX;_pdY=e.clientY;});
    mk.getElement().addEventListener('click',e=>{
      // If the pointer moved more than 5 px between down and up, it was a map drag — ignore
      if(Math.hypot(e.clientX-_pdX,e.clientY-_pdY)>5) return;
      tiltIdx.fill(0); timeOffsetMin=0;
      iconD2SyncedToRadar = true;
      stationSel.value=name;
      // Sync country dropdown to match selected station
      const newCountry=stationCountry(name);
      if(countrySel.value!==newCountry){
        countrySel.value=newCountry;
        populateStationsForCountry(newCountry);
        stationSel.value=name;
      }
      refreshAllMaps(true);
      maps.forEach((m,i)=>rebuildMarkers(m,i));
      try{saveUiState();}catch(ex){}
    });
    stationMarkers[idx].push(mk);
  });
}

// Cross-section state object
const CS = { drawing:false, ptA:null, ptB:null, markerA:null, markerB:null };

// Satellite WMS overlay
const SAT_WMS_BASE_URL =
  'https://view.eumetsat.int/geoserver/ows?service=WMS&version=1.3.0&request=GetMap'
  + '&styles=&format=image/png&transparent=true&CRS=EPSG:3857'
  + '&WIDTH=512&HEIGHT=512&BBOX={bbox-epsg-3857}';

const SAT_LAYERS = {
  geocolor: { label: 'GeoColor RGB MTG', layerName: 'mtg_fd:rgb_geocolour' },
  hrv:      { label: 'European HRV RGB', layerName: 'msg_fes:rgb_eview' },
  airmass:  { label: 'Airmass RGB (0°)', layerName: 'msg_0deg:rgb_airmass' },
};

const LIGHTNING_API_BASE='https://opendataapi.dmi.dk/v2/lightningdata/collections/observation/items';

// ── Per-map layer-ordering cache ─────────────────────────────────────────
// Computed once at map load time so every subsequent render never calls
// getStyle() (which throws "Style is not done loading" during async ops).
const _mapIsSlot   = Array.from({length: panelEls.length}, () => false);   // true = Standard/slot style
const _mapBeforeId = Array.from({length: panelEls.length}, () => null);    // first admin/label layer id for classic styles

function _cacheMapStyle(map, idx) {
  let isSlot = false;
  let beforeId = null;
  try {
    const style = map.getStyle && map.getStyle();
    if (style) {
      if (style.schema != null) {
        isSlot = true;
      } else {
        isSlot = (style.layers || []).some(l => l.slot != null);
      }
      if (!isSlot) {
        // Dump ALL layer IDs so we can see what the custom style actually exposes.
        // Look at the browser console after startup to find the right insertion layer.
        const allIds = (style.layers || []).map(l => l.id);
        console.log(`[map${idx}] ALL layer IDs (${allIds.length}):`, allIds.join(', '));
        beforeId = getOverlayBeforeLayerId(map, allIds);
      }
    }
  } catch(e) { console.warn(`[map${idx}] _cacheMapStyle error:`, e); }
  _mapIsSlot[idx]   = isSlot;
  _mapBeforeId[idx] = beforeId;
  console.log(`[map${idx}] slot=${isSlot}  beforeId=${beforeId}`);
}

function bindStyleLoad(map,idx){
  map.on('style.load',()=>{
    _cacheMapStyle(map, idx);
    mapReady[idx]=true;
    rebuildMarkers(map,idx);
    applyOverlayToMap(map,idx);
    reapplyCsLine(map,idx);
    if(idx===0) { refreshLightning(); }
    if(layerSettings.metObs && _lastMetObsFC) applyMetObsToMaps(_lastMetObsFC);
    if(anySatelliteEnabled()) applySatelliteToMap(map,idx);
    if(layerSettings.bluemarble) applyBluemarbleToMap(map,idx);
    updateCoordGridForMap(map,idx);
  });
}
maps.forEach((m,i)=>{
  m.keyboard.disable();
  m.on('move',()=>syncFrom(m));
  m.on('load',()=>{
    _cacheMapStyle(m, i);
    mapReady[i]=true;
    rebuildMarkers(m,i);
    applyOverlayToMap(m,i);
    if(i===0){refreshLightning();refreshMetObs();}
    if(anySatelliteEnabled())applySatelliteToMap(m,i);
    if(layerSettings.bluemarble) applyBluemarbleToMap(m,i);
    updateCoordGridForMap(m,i);
  });
  bindStyleLoad(m,i);
});

const satUpdateCfg={intervalMs:30*60*1000,retryMs:10*60*1000,cycle:0,timerId:null};

async function loadSelectedStationData() {
  if (location.protocol === 'file:') return;
  const sn = stationSel.value || '';
  if (!sn) return;
  const _dealiasQs = dealiasEnabled ? '&dealias=1' : '';
  try {
    const resp = await fetch(`payload.json?station=${encodeURIComponent(sn)}${_dealiasQs}&_ts=${Date.now()}`, {cache:'no-store'});
    if (!resp.ok) return;
    const data = await resp.json();
    if (data && data.rendered) {
      APP.rendered = data.rendered;
      APP.errors = data.errors || {};
      renderSettingsErrors();
      warn.textContent = '';
      refreshAllMaps(true);
      maps.forEach((m,i)=>rebuildMarkers(m,i));
      if (iconD2Enabled) void syncIconD2ToRadar(false);
      try{initTimeInputsFromCurrent();}catch(e){}
      if (iconD2Enabled && iconD2SyncedToRadar) void syncIconD2ToRadar(false);
    }
  } catch(e) {}
}

// Azimuthal shear composite logic
const _azshearCtrl = [
  document.getElementById('azshear-ctrl1'),
  document.getElementById('azshear-ctrl2'),
  document.getElementById('azshear-ctrl3'),
  document.getElementById('azshear-ctrl4'),
];
const _azshearNSel = [
  document.getElementById('azshear-n1'),
  document.getElementById('azshear-n2'),
  document.getElementById('azshear-n3'),
  document.getElementById('azshear-n4'),
];

// Active composite overlays per panel (null = use live payload)
const _compositeOverride = Array.from({length: panelEls.length}, () => null);

function _onProductChanged(panelIdx) {
  const isAzShear = panelProducts[panelIdx]() === 'azimuthal_shear';
  const ctrl = _azshearCtrl[panelIdx];
  if (ctrl) ctrl.style.display = isAzShear ? 'flex' : 'none';
  // When switching away from azimuthal_shear, clear any composite override
  if (!isAzShear) {
    _compositeOverride[panelIdx] = null;
  } else {
    // Trigger composite fetch for whatever N is currently selected
    _fetchComposite(panelIdx);
  }
}

function _fetchComposite(panelIdx) {
  const nSel = _azshearNSel[panelIdx];
  const n = nSel ? parseInt(nSel.value, 10) : 1;
  const station = stationSel.value;

  if (n <= 1) {
    // n=1 means just use the live single-scan data; clear override
    _compositeOverride[panelIdx] = null;
    void applyOverlayToMap(maps[panelIdx], panelIdx);
    return;
  }

  if (location.protocol === 'file:') {
    warn.textContent = 'Composite requires: python eme_t.py --serve';
    setTimeout(() => { warn.textContent = ''; }, 4000);
    return;
  }

  const url = `azshear_composite.json?station=${encodeURIComponent(station)}&n=${n}&_ts=${Date.now()}`;
  fetch(url, { cache: 'no-store' })
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(data => {
      if (data && data.overlayData) {
        _compositeOverride[panelIdx] = data;
        void applyOverlayToMap(maps[panelIdx], panelIdx);
        renderLegend(panelIdx);
      }
    })
    .catch(err => {
      warn.textContent = `Composite fetch failed (${err}). Using single scan.`;
      _compositeOverride[panelIdx] = null;
      setTimeout(() => { warn.textContent = ''; }, 3500);
    });
}

// Wire composite N-selectors
_azshearNSel.forEach((sel, i) => {
  if (sel) sel.onchange = () => _fetchComposite(i);
});

// Patch applyOverlayToMap to inject composite override when active:
// We wrap it so that when azimuthal_shear is selected and a composite
// override exists, the composite overlay data is used instead of the
// payload's single-scan overlay.
const _origApplyOverlayToMap = applyOverlayToMap;
applyOverlayToMap = async function(map, mapIndex) {
  const pk = panelProducts[mapIndex]();
  const override = _compositeOverride[mapIndex];
  if (pk === 'azimuthal_shear' && override && override.overlayData) {
    // Temporarily swap the rendered product's overlayData with the composite
    const sn = stationSel.value;
    const sd = APP.rendered[sn];
    if (sd && sd.products && sd.products['azimuthal_shear']) {
      const orig = sd.products['azimuthal_shear'];
      const origOd = orig.overlayData;
      orig.overlayData = override.overlayData;
      await _origApplyOverlayToMap.call(this, map, mapIndex);
      orig.overlayData = origOd;
      return;
    }
  }
  return _origApplyOverlayToMap.call(this, map, mapIndex);
};

stationSel.onchange  = () => {
  tiltIdx.fill(0); timeOffsetMin=0;
  iconD2SyncedToRadar = true;
  refreshAllMaps(true);
  maps.forEach((m,i)=>rebuildMarkers(m,i));
  try{initTimeInputsFromCurrent();}catch(e){}
  saveUiState();
  // Custom radars are fully rendered in memory at upload time — skipping the
  // server refetch prevents the layer being torn down and rebuilt a second
  // time, which was causing the visible disappear-then-reappear flicker.
  const _selProvider = ((APP.stations[stationSel.value]||{}).provider||'').toLowerCase();
  if (_selProvider !== 'custom') void loadSelectedStationData();
  else if (iconD2Enabled) void syncIconD2ToRadar(false);
  // Show/hide remove link for custom radars
  try{ if(typeof syncRemoveBtn==='function') syncRemoveBtn(); }catch(_e){}
};
countrySel.onchange  = () => {
  populateStationsForCountry(countrySel.value);
  tiltIdx.fill(0); timeOffsetMin=0;
  iconD2SyncedToRadar = true;
  refreshAllMaps(true);
  maps.forEach((m,i)=>rebuildMarkers(m,i));
  try{initTimeInputsFromCurrent();}catch(e){}
  saveUiState();
  void loadSelectedStationData();
};
productSels.forEach((sel, idx) => {
  sel.onchange = () => { tiltIdx[idx]=0; _onProductChanged(idx); refreshAllMaps(false); saveUiState(); };
});
countSel.onchange    = () => {
  if (iconD2Enabled) countSel.value = '2';
  const visibleCount = selectedPanelCount();
  panelEls.forEach((el, idx) => {
    if (el) el.classList.toggle('hidden', idx >= visibleCount);
  });
  setTimeout(()=>maps.forEach(m=>m.resize()), 80);
  saveUiState();
};
mapStyleSel.onchange = () => { const ns=currentMapStyleUri(); maps.forEach(m=>m.setStyle(ns)); saveUiState(); };
if (iconD2Btn) {
  iconD2Btn.onclick = () => { void setIconD2Enabled(!iconD2Enabled); };
}
updateIconD2Ui();

// ── Arrow-key navigation: Up/Down = tilts, Left/Right = scan time ─────────────
document.addEventListener('keydown', e => {
  // Don't fire when typing in inputs or when cross-section drawing is active
  const tag = (e.target||document.body).tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (CS.drawing) {
    if (e.key === 'Escape') endCsDrawing();
    return;
  }

  if (e.key === 'k' || e.key === 'K') {
    if (iconD2Enabled) {
      e.preventDefault();
      timeNavFocus = 'radar';
      _updateTimeNavFocusUi();
    }
    return;
  }

  if (e.key === 'l' || e.key === 'L') {
    if (iconD2Enabled) {
      e.preventDefault();
      timeNavFocus = 'model';
      _updateTimeNavFocusUi();
    }
    return;
  }

  if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
    e.preventDefault();
    const delta = e.key === 'ArrowUp' ? 1 : -1;
    let changed = false;
    (iconD2Enabled ? [0] : visiblePanelIndices()).forEach(idx => {
      const pk = panelProducts[idx]();
      const sn = stationSel.value;
      const pd = ((APP.rendered[sn] || {}).products || {})[pk];
      if (!pd || !pd.allTiltOverlays) return;
      const maxIdx = pd.allTiltOverlays.length - 1;
      const next = Math.max(0, Math.min(tiltIdx[idx] + delta, maxIdx));
      if (next !== tiltIdx[idx]) { tiltIdx[idx] = next; changed = true; }
    });
    if (changed) { maps.forEach((m,i) => void applyOverlayToMap(m, i)); saveUiState(); }
    return;
  }

  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    e.preventDefault();
    if (iconD2Enabled && timeNavFocus === 'model') {
      void stepIconD2Frame(e.key === 'ArrowLeft' ? -1 : 1);
      return;
    }
    const delta = e.key === 'ArrowLeft' ? -TIME_STEP_MIN : TIME_STEP_MIN;
    const sn = stationSel.value;
    const provider = ((APP.stations[sn] || {}).provider || 'dmi').toLowerCase();

    if (location.protocol === 'file:') {
      warn.textContent = 'Scan time navigation requires: python emraw.py --serve';
      setTimeout(() => { warn.textContent = ''; }, 4000);
      return;
    }

    // For scan time navigation we use the server's archive endpoint.
    // Only DMI currently supports archive; show a notice for others.
    if (provider !== 'dmi') {
      warn.textContent = `Scan time navigation is only supported for DMI stations.`;
      setTimeout(() => { warn.textContent = ''; }, 3000);
      return;
    }

    // Compute new target time
    const sd = APP.rendered[sn] || {};
    const baseDt = sd.datetime ? new Date(sd.datetime) : new Date();
    if (timeOffsetMin === 0 && delta > 0) return;  // already at latest

    timeOffsetMin = Math.min(0, timeOffsetMin + delta);
    const targetMs  = baseDt.getTime() + timeOffsetMin * 60 * 1000;
    const targetDt  = new Date(targetMs);
    const iso = targetDt.toISOString().replace('.000Z', 'Z');

    // Returning to offset 0 → exit archive mode and restore live
    if (timeOffsetMin === 0) {
      _exitArchiveMode();
      startAutoUpdateCountdown();
    }

    //Check client-side scan cache first
    const cached = _scanCache.get(iso);
    if (cached) {
      APP.rendered = cached.rendered;
      APP.errors   = cached.errors || {};
      renderSettingsErrors();
      if(timeOffsetMin < 0) _enterArchiveMode(iso);
      refreshAllMaps(false);
      renderAllLegends();
      // Prefetch the step before this one in the background
      _prefetchScan(new Date(targetMs - 15*60*1000).toISOString().replace('.000Z','Z'));
      // Refresh ESWD for this archive moment if overlay is on
      if(eswdOverlayEnabled) _fetchAndApplyEswd(iso);
      return;
    }

    warn.textContent = `Loading scan at ${iso.replace('T',' ').replace('Z',' UTC')} …`;
    const _dealiasQs = dealiasEnabled ? '&dealias=1' : '';
    const _stationQs = stationSel.value ? `&station=${encodeURIComponent(stationSel.value)}` : '';
    fetch(`payload.json?dmi_dt=${encodeURIComponent(iso)}${_stationQs}${_dealiasQs}&_ts=${Date.now()}`, {cache:'no-store'})
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => {
        if (data && data.rendered) {
          _scanCache.set(iso, {rendered: data.rendered, errors: data.errors || {}});
          APP.rendered = data.rendered;
          APP.errors   = data.errors || {};
          renderSettingsErrors();
          if(timeOffsetMin < 0) _enterArchiveMode(iso);
          refreshAllMaps(false);
          renderAllLegends();
          // Prefetch adjacent scans in the background
          _prefetchScan(new Date(targetMs - 15*60*1000).toISOString().replace('.000Z','Z'));
          _prefetchScan(new Date(targetMs - 30*60*1000).toISOString().replace('.000Z','Z'));
          // Refresh ESWD for this archive moment if overlay is on
          if(eswdOverlayEnabled) _fetchAndApplyEswd(iso);
        }
      })
      .catch(err => {
        warn.textContent = `Archive fetch failed (${err}). Try pressing → to return to latest.`;
        timeOffsetMin -= delta; // revert
      });
    return;
  }


  // Product keybindings
  const PRODUCT_KEYS = {
    'z': 'reflectivity',
    'v': 'velocity',
    'c': 'correlation',
    's': 'spectrum_width',
    'd': 'zdr',
    'p': 'kdp',
    '0': 'echo_tops_0',
    'e': 'echo_tops',
    'n': 'nrot',
    'r': 'srv',
    'm': 'meso',
    'u': 'azimuthal_shear',
    'h': 'mesh',
  };
  const pk = PRODUCT_KEYS[e.key.toLowerCase()];
  if (pk && APP.products[pk]) {
    e.preventDefault();
    (iconD2Enabled ? [0] : visiblePanelIndices()).forEach(idx => {
      productSels[idx].value = pk;
      tiltIdx[idx] = 0;
      _onProductChanged(idx);
    });
    refreshAllMaps(false);
    saveUiState();
  }
});

setTimeout(()=>{ try{ countSel.onchange(); }catch(e){} }, 0);
renderSettingsErrors();
void loadSelectedStationData();

// Lightning overlay 
async function fetchLightningGeoJSON(){
  const resp=await fetch(`${LIGHTNING_API_BASE}?period=latest-10-minutes&bbox=7,54,16,58&limit=5000`);
  if(!resp.ok) throw new Error(`Lightning API ${resp.status}`);
  return resp.json();
}
function removeLightningFromMap(map,idx){const s=`lightning-src-${idx}`,l=`lightning-lyr-${idx}`;if(map.getLayer&&map.getLayer(l))map.removeLayer(l);if(map.getSource&&map.getSource(s))map.removeSource(s);}
async function applyLightningToMap(map,idx,data){
  if(!mapReady[idx]) return;
  const srcId=`lightning-src-${idx}`,lyrId=`lightning-lyr-${idx}`;
  const before = _mapIsSlot[idx] ? undefined : (_mapBeforeId[idx] || undefined);
  if(map.getSource(srcId)){map.getSource(srcId).setData(data);}
  else{
    map.addSource(srcId,{type:'geojson',data});
    map.addLayer({id:lyrId,type:'circle',source:srcId,paint:{'circle-radius':3,'circle-color':['case',['>'  ,['get','amp'],0],'#ffff66','#ff00ff'],'circle-opacity':0.9}},before);
  }
}
async function refreshLightning(){
  if(!layerSettings.lightning){maps.forEach((m,i)=>removeLightningFromMap(m,i));return;}
  try{const data=await fetchLightningGeoJSON();maps.forEach((m,i)=>applyLightningToMap(m,i,data));}
  catch(e){console.warn('Lightning fetch failed',e);}
}

// Meteorological observations overlay (DMI + SMHI + NO + DE) 
// DMI metObs API v2
const MET_OBS_API     = 'https://opendataapi.dmi.dk/v2/metObs/collections/observation/items';
const MET_OBS_DMI_KEY = APP.dmiApiKey || '8546dc5e-fdc8-436b-9f07-0e2e313d0af1';
const DMI_OBS_PARAMS  = ['temp_dry', 'wind_speed_past1h', 'wind_dir_past1h', 'humidity_past1h'];
// SMHI metObs: no auth, station-set/all, latest-hour
// param 1=temp, 39=dewpoint(direct), 4=windspeed, 3=winddir
const SMHI_METOBS_BASE = 'https://opendata-download-metobs.smhi.se/api/version/latest/parameter';
const SMHI_OBS_PARAMS  = { temp: 1, dewpoint: 39, windspeed: 4, winddir: 3 };
// Norway Frost & Germany DWD: fetched server-side (auth / bz2 compression),
// embedded in the payload as APP.frostMetObs and APP.dwdMetObs respectively.

let _metObsUpdateTimer = null;
const MET_OBS_REFRESH_MS = 10 * 60 * 1000;

// Client-side scan cache (keyed by ISO datetime string) 
// Avoids round-trips for already-visited archive scans; enables near-instant ← → navigation
const _scanCache = new Map();   // iso -> {rendered, errors}

// Returns the ISO datetime string of the currently-displayed archive scan, or
// null if we are showing the live (latest) scan.
function _currentArchiveIso() {
  if (timeOffsetMin === 0) return null;
  const sn = stationSel.value;
  const sd = APP.rendered[sn] || {};
  const baseDt = sd.datetime ? new Date(sd.datetime) : null;
  if (!baseDt || Number.isNaN(baseDt.getTime())) return null;
  const targetMs = baseDt.getTime() + timeOffsetMin * 60 * 1000;
  return new Date(targetMs).toISOString().replace('.000Z','Z');
}
async function _prefetchScan(iso) {
  if (_scanCache.has(iso) || location.protocol === 'file:') return;
  try {
    const _dealiasQs = dealiasEnabled ? '&dealias=1' : '';
    const _stationQs = stationSel.value ? `&station=${encodeURIComponent(stationSel.value)}` : '';
    const resp = await fetch(`payload.json?dmi_dt=${encodeURIComponent(iso)}${_stationQs}${_dealiasQs}&_ts=${Date.now()}`, {cache:'no-store'});
    if (!resp.ok) return;
    const data = await resp.json();
    if (data && data.rendered) _scanCache.set(iso, {rendered: data.rendered, errors: data.errors || {}});
  } catch(e) { /* silent background prefetch */ }
}

// Magnus dew-point formula 
function calcDewPoint(tempC, rh) {
  if (!Number.isFinite(tempC) || !Number.isFinite(rh) || rh <= 0) return null;
  const a = 17.27, b = 237.3;
  const g = (a * tempC) / (b + tempC) + Math.log(Math.max(0.01, rh) / 100);
  return (b * g) / (a - g);
}

// Wind barb SVG 
// Convention: shaft points UP (from station toward wind origin = wind-from-N).
// Rotate by windDir° clockwise in Mapbox to orient geographically.
// Canvas: 44×44, station dot at centre (22, 22), tip at (22, 3).
function windBarbSVG(speedMs) {
  const kts = speedMs * 1.944;
  const W = 44, H = 44, cx = 22, dotY = 22, tipY = 3;
  const WHT = '#ffffff', BLK = '#000000';

  function ln(x1,y1,x2,y2,sw,col,cap) {
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${sw}" stroke-linecap="${cap||'round'}"/>`;
  }
  function poly(pts, fill) {
    return `<polygon points="${pts}" fill="${fill}" stroke="none"/>`;
  }
  // Outline + fill shorthand
  function lineWO(x1,y1,x2,y2) { return ln(x1,y1,x2,y2,3,BLK)+ln(x1,y1,x2,y2,1.4,WHT); }
  function polyWO(pts) { return poly(pts,BLK)+poly(pts,WHT); } // draw black then white for outline effect

  // Calm: two concentric rings, no shaft
  if (kts < 2) {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
      <circle cx="${cx}" cy="${dotY}" r="8" fill="none" stroke="${BLK}" stroke-width="3"/>
      <circle cx="${cx}" cy="${dotY}" r="8" fill="none" stroke="${WHT}" stroke-width="1.4"/>
      <circle cx="${cx}" cy="${dotY}" r="3" fill="none" stroke="${BLK}" stroke-width="3"/>
      <circle cx="${cx}" cy="${dotY}" r="3" fill="none" stroke="${WHT}" stroke-width="1.4"/>
    </svg>`;
  }

  let pennants = Math.floor(kts / 50);
  let rem      = kts % 50;
  let longs    = Math.floor(rem / 10);
  let shorts   = Math.floor((rem % 10) / 5);

  const parts = [];
  // Shaft (drawn first so barbs overlay it cleanly)
  parts.push(lineWO(cx, dotY, cx, tipY));
  // Station dot on top of shaft
  parts.push(`<circle cx="${cx}" cy="${dotY}" r="3.5" fill="${BLK}"/>`)
  parts.push(`<circle cx="${cx}" cy="${dotY}" r="2.5" fill="${WHT}"/>`);

  // Barbs start at the tip and step downward toward the dot
  let by = tipY;
  const BL = 12, BS = 6, BD = 4, SP = 5; // long-len, short-len, slope-down, spacing

  for (let i = 0; i < pennants; i++) {
    // Filled triangle: left edge on shaft, right apex at barb length
    const pts = `${cx},${by} ${cx+BL},${by+BD+1} ${cx},${by+SP*2}`;
    // Draw black outline polygon then white fill on top
    parts.push(poly(pts, BLK));
    // Slightly inset white fill
    const inPts = `${cx+0.5},${by+0.5} ${cx+BL-1},${by+BD+1} ${cx+0.5},${by+SP*2-0.5}`;
    parts.push(poly(inPts, WHT));
    by += SP * 2 + 1;
  }
  for (let i = 0; i < longs; i++) {
    parts.push(lineWO(cx, by, cx + BL, by + BD));
    by += SP;
  }
  for (let i = 0; i < shorts; i++) {
    parts.push(lineWO(cx, by, cx + BS, by + 3));
    by += SP;
  }

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">${parts.join('')}</svg>`;
}

function _svgToImg(svgStr, size) {
  return new Promise((resolve, reject) => {
    const img = new Image(size, size);
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgStr);
  });
}

async function _ensureBarbImages(map, bucketsKts) {
  for (const kts of bucketsKts) {
    const id = `metbarb-${kts}`;
    if (!map.hasImage(id)) {
      try {
        const img = await _svgToImg(windBarbSVG(kts / 1.944), 44);
        if (!map.hasImage(id)) map.addImage(id, img);
      } catch(e) { console.warn('barb image failed', kts, e); }
    }
  }
}

// DMI fetch helper 
async function _fetchOneDmiParam(paramId, from, to) {
  const url = `${MET_OBS_API}?api-key=${MET_OBS_DMI_KEY}&limit=10000`
            + `&datetime=${encodeURIComponent(from + '/' + to)}`
            + `&parameterId=${encodeURIComponent(paramId)}`;
  const resp = await fetch(url);
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error(`DMI metObs ${resp.status} (${paramId}): ${body.slice(0, 200)}`);
  }
  return resp.json();
}

// SMHI fetch helper 
async function _fetchSmhiParam(fieldName, paramId) {
  const url = `${SMHI_METOBS_BASE}/${paramId}/station-set/all/period/latest-hour/data.json`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`SMHI metObs ${resp.status} (param ${paramId})`);
  const data = await resp.json();
  const out = {};
  if (!data || !Array.isArray(data.station)) return out;
  for (const st of data.station) {
    if (!st.value || !st.value.length) continue;
    const entry = st.value[st.value.length - 1];
    if (entry.value == null) continue;
    const val = Number(entry.value);
    if (!Number.isFinite(val)) continue;
    out['smhi_' + st.key] = {
      lon: st.longitude, lat: st.latitude,
      stationName: st.name,
      fieldName, val,
    };
  }
  return out;
}

// Norway Frost payload extractor 
// APP.frostMetObs is pre-fetched server-side (Frost requires Basic auth).
// Schema: [{sid, name, lon, lat, temp, dewpt, wspd, wdir}]
function _extractFrostStationData() {
  const out = {};
  for (const s of (APP.frostMetObs || [])) {
    if (!Number.isFinite(s.lon) || !Number.isFinite(s.lat)) continue;
    out['no_' + s.sid] = {
      lon: s.lon, lat: s.lat, stationName: s.name || String(s.sid),
      provider: 'frost',
      params: { temp: s.temp, dewpt: s.dewpt, windspeed: s.wspd, winddir: s.wdir },
      timestamps: {},
    };
  }
  return out;
}

// DWD (Germany) payload extractor 
// APP.dwdMetObs is pre-processed on the Python backend; convert to internal format.
function _extractDwdStationData() {
  const list = (APP.dwdMetObs || []);
  const out = {};
  for (const s of list) {
    if (!Number.isFinite(s.lon) || !Number.isFinite(s.lat)) continue;
    out['dwd_' + s.sid] = {
      lon: s.lon, lat: s.lat, stationName: s.name || String(s.sid),
      provider: 'dwd_de',
      params: { temp: s.temp, dewpt: s.dewpt, windspeed: s.wspd, winddir: s.wdir },
    };
  }
  return out;
}

// KNMI (Netherlands) payload extractor
// APP.knmiMetObs is pre-fetched server-side (same schema as frostMetObs/dwdMetObs).
// Schema: [{sid, name, lon, lat, temp, dewpt, wspd, wdir}]
function _extractKnmiStationData() {
  const list = (APP.knmiMetObs || []);
  const out = {};
  for (const s of list) {
    if (!Number.isFinite(s.lon) || !Number.isFinite(s.lat)) continue;
    out['nl_' + s.sid] = {
      lon: s.lon, lat: s.lat, stationName: s.name || String(s.sid),
      provider: 'knmi_nl',
      params: { temp: s.temp, dewpt: s.dewpt, windspeed: s.wspd, winddir: s.wdir },
      timestamps: {},
    };
  }
  return out;
}

// GeoSphere Austria (TAWES) payload extractor 
// APP.geosphereMetObs is pre-fetched server-side via klima-v2-10min current API.
// Schema: [{sid, name, lon, lat, temp, dewpt, wspd, wdir}]
function _extractGeosphereStationData() {
  const list = (APP.geosphereMetObs || []);
  const out = {};
  for (const s of list) {
    if (!Number.isFinite(s.lon) || !Number.isFinite(s.lat)) continue;
    out['at_' + s.sid] = {
      lon: s.lon, lat: s.lat, stationName: s.name || String(s.sid),
      provider: 'geosphere_at',
      params: { temp: s.temp, dewpt: s.dewpt, windspeed: s.wspd, winddir: s.wdir },
      timestamps: {},
    };
  }
  return out;
}

// Combined fetch 
async function fetchMetObsGeoJSON() {
  const now  = new Date();
  const from = new Date(now.getTime() - 90 * 60 * 1000).toISOString().replace('.000Z','Z');
  const to   = now.toISOString().replace('.000Z','Z');

  // Fire all requests in parallel (DMI + SMHI only; NO and DE come from payload)
  const [dmiResults, smhiResults] = await Promise.all([
    Promise.allSettled(DMI_OBS_PARAMS.map(p => _fetchOneDmiParam(p, from, to))),
    Promise.allSettled(
      Object.entries(SMHI_OBS_PARAMS).map(([field, id]) => _fetchSmhiParam(field, id))
    ),
  ]);

  // Aggregate DMI stations
  const stationData = {};

  for (const res of dmiResults) {
    if (res.status !== 'fulfilled') { console.warn('DMI metObs param failed:', res.reason); continue; }
    const fc = res.value;
    if (!fc || !fc.features) continue;
    for (const feat of fc.features) {
      const p      = feat.properties || {};
      const sid    = 'dmi_' + String(p.stationId || p.station_id || '');
      if (sid === 'dmi_') continue;
      const paramId = String(p.parameterId || p.parameter_id || '');
      const val = p.value;
      if (val == null || !Number.isFinite(Number(val))) continue;
      const observed = p.observed || p.time || '';
      const coords = feat.geometry && feat.geometry.coordinates;
      if (!coords || coords.length < 2) continue;

      if (!stationData[sid]) {
        stationData[sid] = {
          lon: coords[0], lat: coords[1],
          stationName: p.stationName || p.station_name || sid,
          provider: 'dmi',
          params: {}, timestamps: {},
        };
      }
      const prev = stationData[sid].timestamps[paramId];
      if (!prev || observed > prev) {
        stationData[sid].params[paramId] = Number(val);
        stationData[sid].timestamps[paramId] = observed;
      }
    }
  }

  // Aggregate SMHI stations
  for (const res of smhiResults) {
    if (res.status !== 'fulfilled') { console.warn('SMHI metObs param failed:', res.reason); continue; }
    const byStation = res.value;
    for (const [sid, info] of Object.entries(byStation)) {
      if (!stationData[sid]) {
        stationData[sid] = {
          lon: info.lon, lat: info.lat,
          stationName: info.stationName,
          provider: 'smhi',
          params: {}, timestamps: {},
        };
      }
      stationData[sid].params[info.fieldName] = info.val;
    }
  }

  // Merge Norway Frost (from payload)
  const frostData = _extractFrostStationData();
  for (const [sid, s] of Object.entries(frostData)) {
    stationData[sid] = { ...s, timestamps: {} };
  }

  // Merge DWD Germany (from payload) 
  const dwdData = _extractDwdStationData();
  for (const [sid, s] of Object.entries(dwdData)) {
    stationData[sid] = { ...s, timestamps: {} };
  }

  // Merge KNMI Netherlands (from payload)
  const knmiData = _extractKnmiStationData();
  for (const [sid, s] of Object.entries(knmiData)) {
    stationData[sid] = { ...s };
  }

  // Merge GeoSphere Austria (from payload) 
  const geosphereData = _extractGeosphereStationData();
  for (const [sid, s] of Object.entries(geosphereData)) {
    stationData[sid] = { ...s };
  }

  // Build unified GeoJSON features 
  const features = Object.values(stationData)
    .filter(s => Number.isFinite(s.lon) && Number.isFinite(s.lat))
    .map(s => {
      // Normalise field names across providers
      let temp, rh, wSpd, wDir, dewpt;
      switch (s.provider) {
        case 'dmi':
          temp  = s.params['temp_dry'];
          rh    = s.params['humidity_past1h'];
          wSpd  = s.params['wind_speed_past1h'];
          wDir  = s.params['wind_dir_past1h'];
          dewpt = calcDewPoint(temp, rh);
          break;
        case 'smhi':
          temp  = s.params['temp'];
          dewpt = s.params['dewpoint'];
          wSpd  = s.params['windspeed'];
          wDir  = s.params['winddir'];
          break;
        case 'frost':
        case 'dwd_de':
        case 'knmi_nl':
        case 'geosphere_at':
          temp  = s.params['temp'];
          dewpt = s.params['dewpt'];
          wSpd  = s.params['windspeed'];
          wDir  = s.params['winddir'];
          break;
        default:
          temp = dewpt = wSpd = wDir = null;
      }

      const kts      = Number.isFinite(wSpd) ? wSpd * 1.944 : null;
      const buckKts  = kts != null ? Math.round(kts / 5) * 5 : null;
      const barbId   = buckKts != null ? `metbarb-${buckKts}` : '';

      const fmt = (v, dec, unit) => v != null && Number.isFinite(v) ? `${v.toFixed(dec)}\u202f${unit}` : 'n/a';
      const flag = s.provider === 'dmi' ? '🇩🇰'
                 : s.provider === 'smhi' ? '🇸🇪'
                 : s.provider === 'frost' ? '🇳🇴'
                 : s.provider === 'knmi_nl' ? '🇳🇱'
                 : s.provider === 'geosphere_at' ? '🇦🇹'
                 : '🇩🇪';
      const popupHtml = `<div style="font:11px Consolas,monospace;background:#1e1e1e;padding:7px 10px;border-radius:5px;line-height:1.7">
        <b style="color:#9cf;font-size:12px">${flag} ${s.stationName}</b><br>
        <span style="color:#ff6666">T &nbsp;&nbsp; ${fmt(temp,1,'°C')}</span><br>
        <span style="color:#66dd66">Td &nbsp; ${fmt(dewpt,1,'°C')}</span><br>
        ${rh != null ? `<span style="color:#aaa">RH &nbsp; ${fmt(rh,0,'%')}</span><br>` : ''}
        <span style="color:#ddd">Wsp ${fmt(wSpd,1,'m/s')} &nbsp; Wdir ${wDir!=null?Math.round(wDir)+'°':'n/a'}</span>
      </div>`;

      return {
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [s.lon, s.lat] },
        properties: {
          stationName: s.stationName,
          tempLabel:   temp  != null && Number.isFinite(temp)  ? String(Math.round(temp))  : '',
          dewptLabel:  dewpt != null && Number.isFinite(dewpt) ? String(Math.round(dewpt)) : '',
          barbId,
          windDir: wDir != null ? wDir : 0,
          popupHtml,
        },
      };
    });

  return { type: 'FeatureCollection', features };
}

//Map layer management 
const _metObsPopups = [];
const MET_OBS_LAYERS = ['metobs-barb','metobs-dot','metobs-temp','metobs-dewpt'];

function removeMetObsFromMaps() {
  _metObsPopups.forEach(p => p.remove());
  _metObsPopups.length = 0;
  maps.forEach(map => {
    MET_OBS_LAYERS.forEach(id => {
      if (map.getLayer  && map.getLayer(id))   map.removeLayer(id);
    });
    if (map.getSource && map.getSource('metobs-src')) map.removeSource('metobs-src');
  });
}

// metObs dot icon (symbol, always above basemap line layers) 
async function _ensureMetObsDotImage(map) {
  const id = 'metobs-dot-icon';
  if (map.hasImage && map.hasImage(id)) return;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">
    <circle cx="8" cy="8" r="5" fill="white" stroke="black" stroke-width="1.5"/>
  </svg>`;
  return new Promise(resolve => {
    const img = new Image(16, 16);
    img.onload = () => {
      try { if (!map.hasImage(id)) map.addImage(id, img); } catch(e) {}
      resolve();
    };
    img.onerror = () => resolve();
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  });
}

async function applyMetObsToMaps(fc) {
  if (!fc || !fc.features || !fc.features.length) { removeMetObsFromMaps(); return; }

  // Pre-load all required barb images
  const buckets = new Set();
  fc.features.forEach(f => {
    const bid = f.properties.barbId;
    if (bid) buckets.add(parseInt(bid.replace('metbarb-',''), 10));
  });

  for (let idx = 0; idx < maps.length; idx++) {
    const map = maps[idx];
    if (!mapReady[idx]) continue;

    await _ensureBarbImages(map, [...buckets]);
    await _ensureMetObsDotImage(map);

    // Update existing source in-place (no layer churn)
    if (map.getSource && map.getSource('metobs-src')) {
      map.getSource('metobs-src').setData(fc);
      // For classic styles: re-raise layers above any newly-added radar layer.
      // For slot-based styles: the slot:'top' property is permanent — no action needed.
      raiseMetObsLayers(map);
      continue;
    }

    // Remove any stale metObs from this specific map
    for (const id of MET_OBS_LAYERS) {
      try { if (map.getLayer && map.getLayer(id)) map.removeLayer(id); } catch(e) {}
    }
    try { if (map.getSource && map.getSource('metobs-src')) map.removeSource('metobs-src'); } catch(e) {}

    // Source
    map.addSource('metobs-src', { type: 'geojson', data: fc });

    // 1. Wind barbs — symbol layer, always above non-symbol layers
    _addLayerOnTop(map, {
      id: 'metobs-barb', type: 'symbol', source: 'metobs-src',
      filter: ['!=', ['get','barbId'], ''],
      layout: {
        'icon-image': ['get','barbId'],
        'icon-rotate': ['get','windDir'],
        'icon-rotation-alignment': 'map',
        'icon-allow-overlap': true,
        'icon-ignore-placement': true,
        'icon-anchor': 'center',
      },
    });

    // 2. Station dot — symbol so it participates in the symbol rendering pass
    _addLayerOnTop(map, {
      id: 'metobs-dot', type: 'symbol', source: 'metobs-src',
      layout: {
        'icon-image': 'metobs-dot-icon',
        'icon-allow-overlap': true,
        'icon-ignore-placement': true,
        'icon-anchor': 'center',
        'icon-size': 1,
      },
    });

    // 3. Temperature label – red, upper-left of station
    _addLayerOnTop(map, {
      id: 'metobs-temp', type: 'symbol', source: 'metobs-src',
      filter: ['!=', ['get','tempLabel'], ''],
      layout: {
        'text-field': ['get','tempLabel'],
        'text-size': 11,
        'text-offset': [-1.4, -0.75],
        'text-anchor': 'center',
        'text-allow-overlap': true,
        'text-ignore-placement': true,
        'text-font': ['DIN Offc Pro Medium','Arial Unicode MS Regular'],
      },
      paint: {
        'text-color': '#ff4444',
        'text-halo-color': '#000000',
        'text-halo-width': 1.5,
      },
    });

    // 4. Dew point label – green, lower-left of station
    _addLayerOnTop(map, {
      id: 'metobs-dewpt', type: 'symbol', source: 'metobs-src',
      filter: ['!=', ['get','dewptLabel'], ''],
      layout: {
        'text-field': ['get','dewptLabel'],
        'text-size': 11,
        'text-offset': [-1.4, 0.75],
        'text-anchor': 'center',
        'text-allow-overlap': true,
        'text-ignore-placement': true,
        'text-font': ['DIN Offc Pro Medium','Arial Unicode MS Regular'],
      },
      paint: {
        'text-color': '#44dd44',
        'text-halo-color': '#000000',
        'text-halo-width': 1.5,
      },
    });

    // Ensure all 4 layers are at the very top of the layer stack, above radar
    raiseMetObsLayers(map);

    // Click popup on dot
    map.on('click', 'metobs-dot', e => {
      const f = e.features && e.features[0];
      if (!f) return;
      const popup = new mapboxgl.Popup({ closeButton: true, closeOnClick: true, maxWidth: '240px' })
        .setLngLat(e.lngLat)
        .setHTML(f.properties.popupHtml)
        .addTo(map);
      _metObsPopups.push(popup);
    });
    map.on('mouseenter','metobs-dot',() => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave','metobs-dot',() => { map.getCanvas().style.cursor = ''; });
  }
}

// Slot-aware layer helpers
// Mapbox Standard v3+ uses named rendering slots: 'bottom' / 'middle' / 'top'.
// Classic styles (Streets v11, Outdoors…) have no slots; the slot property is
// silently ignored but that means ordering is purely by layer-array position.
//
// Strategy (unified for both Standard and Classic styles):
//   Radar/rings/sat: inserted BEFORE the first admin/label layer via beforeId,
//   so they appear below country borders. getOverlayBeforeLayerId() handles both
//   classic (looks for 'boundary'/'admin') and Standard (looks for 'road-label' etc.).
//   MetObs symbols: added via _addLayerOnTop() which uses slot:'top' in Standard
//   style and moveLayer(id) in classic — ensuring metObs is always above borders.
//
// Detection: if any style layer already has a slot property, we're in Standard.
// Standard style detection.
// In Mapbox GL JS v3, Standard style includes a 'schema' property in the serialized
// style object. Classic styles (Streets, Outdoors, Satellite) never have this.
// Fallback: if any loaded layer already carries a slot= prop, it must be Standard.
function isSlotStyle(map) {
  try {
    const style = map.getStyle && map.getStyle();
    if (!style) return false;
    if (style.schema != null) return true;          // definitive: Standard/v3
    return (style.layers || []).some(l => l.slot != null);
  } catch(e) { return false; }
}

// Add a symbol/icon layer that must always render above country labels & radar.
function _addLayerOnTop(map, spec) {
  const _idx = maps.indexOf(map);
  const _slot = _idx >= 0 ? _mapIsSlot[_idx] : isSlotStyle(map);
  if (_slot) {
    try { map.addLayer({ ...spec, slot: 'top' }); } catch(e) {
      try { map.addLayer(spec); } catch(e2) {}
    }
    try { if (map.getLayer && map.getLayer(spec.id)) map.moveLayer(spec.id); } catch(e) {}
  } else {
    try { map.addLayer(spec); } catch(e) {}
    try { if (map.getLayer && map.getLayer(spec.id)) map.moveLayer(spec.id); } catch(e) {}
  }
}

// Raise metObs layers to the absolute top of the layer stack in all style types.
// Called after initial add, after every radar repaint, and after setData.
// moveLayer(id) with no beforeId: end-of-slot in slot styles, end-of-stack in classic.
function raiseMetObsLayers(map) {
  for (const id of ['metobs-barb','metobs-dot','metobs-temp','metobs-dewpt']) {
    try { if (map.getLayer && map.getLayer(id)) map.moveLayer(id); } catch(e) {}
  }
}

// Return the first existing metObs layer id on this map, or null.
// Used so radar/rings are inserted BEFORE (below) any metObs layers.
function _firstMetObsLayerId(map) {
  for (const id of ['metobs-barb','metobs-dot','metobs-temp','metobs-dewpt']) {
    try { if (map.getLayer && map.getLayer(id)) return id; } catch(e) {}
  }
  return null;
}

let _lastMetObsFC = null;
async function refreshMetObs() {
  if (!layerSettings.metObs) { removeMetObsFromMaps(); return; }
  try {
    const fc = await fetchMetObsGeoJSON();
    _lastMetObsFC = fc;
    await applyMetObsToMaps(fc);
  } catch(e) { console.warn('MetObs fetch failed', e); }
}

function scheduleMetObsUpdates() {
  if (_metObsUpdateTimer) clearInterval(_metObsUpdateTimer);
  _metObsUpdateTimer = setInterval(() => {
    if (layerSettings.metObs) refreshMetObs();
  }, MET_OBS_REFRESH_MS);
}
scheduleMetObsUpdates();

function satTilesUrl(layerName){
  return `${SAT_WMS_BASE_URL}&layers=${encodeURIComponent(layerName)}&_ts=${Date.now()}`;
}

function applySatelliteToMap(map,idx){
  removeSatelliteFromMap(map,idx);
  const _slotStyle = _mapIsSlot[idx];
  const before = _slotStyle ? undefined
               : (_mapBeforeId[idx] || undefined);
  Object.keys(SAT_LAYERS).forEach(key => {
    if(!layerSettings[key]) return;
    const info = SAT_LAYERS[key];
    const srcId = `sat-${key}-src-${idx}`;
    const lyrId = `sat-${key}-lyr-${idx}`;
    map.addSource(srcId,{type:'raster',tiles:[satTilesUrl(info.layerName)],tileSize:512});
    const spec = {id:lyrId, type:'raster', source:srcId, paint:{'raster-opacity':1}};
    if (_slotStyle) map.addLayer({...spec, slot:'bottom'});
    else map.addLayer(spec, before);
  });
}

function removeSatelliteFromMap(map,idx){
  Object.keys(SAT_LAYERS).forEach(key=>{
    removeLayerIfExists(map,`sat-${key}-lyr-${idx}`);
    removeSourceIfExists(map,`sat-${key}-src-${idx}`);
  });
}

// Blue Marble GeoTIFF basemap (served locally as XYZ tiles)
function applyBluemarbleToMap(map, idx) {
  removeBluemarbleFromMap(map, idx);
  if (!layerSettings.bluemarble) return;
  const srcId = `bluemarble-src-${idx}`;
  const lyrId = `bluemarble-lyr-${idx}`;
  const tileUrl = `${window.location.origin}/bluemarble/{z}/{x}/{y}.png`;
  try {
    map.addSource(srcId, { type: 'raster', tiles: [tileUrl], tileSize: 256, minzoom: 0, maxzoom: 8 });
    const spec = { id: lyrId, type: 'raster', source: srcId, paint: { 'raster-opacity': 1 } };
    if (_mapIsSlot[idx]) map.addLayer({ ...spec, slot: 'bottom' });
    else map.addLayer(spec, _mapBeforeId[idx] || undefined);
  } catch(e) { console.warn('[bluemarble] addLayer error', e); }
}

function removeBluemarbleFromMap(map, idx) {
  removeLayerIfExists(map, `bluemarble-lyr-${idx}`);
  removeSourceIfExists(map, `bluemarble-src-${idx}`);
}

function applyBluemarbleToAllMaps() {
  maps.forEach((m, i) => applyBluemarbleToMap(m, i));
}

function anySatelliteEnabled(){
  return Object.keys(SAT_LAYERS).some(k=>layerSettings[k]);
}

async function refreshSatelliteOnce(){
  if(!anySatelliteEnabled()) return true;
  let ok=true;
  maps.forEach((map,idx)=>{
    try{
      Object.keys(SAT_LAYERS).forEach(key=>{
        if(!layerSettings[key]) return;
        const info = SAT_LAYERS[key];
        const src = map.getSource && map.getSource(`sat-${key}-src-${idx}`);
        if(src && src.setTiles) src.setTiles([satTilesUrl(info.layerName)]);
      });
    }catch(e){
      console.warn('Satellite refresh failed',e);
      ok=false;
    }
  });
  return ok;
}

function scheduleSatelliteAutoUpdate(){
  if(satUpdateCfg.timerId) clearTimeout(satUpdateCfg.timerId);
  const delay=satUpdateCfg.cycle===0?satUpdateCfg.intervalMs:satUpdateCfg.retryMs;
  satUpdateCfg.timerId=setTimeout(async()=>{
    const ok=await refreshSatelliteOnce();
    satUpdateCfg.cycle = ok ? 0 : satUpdateCfg.cycle+1;
    scheduleSatelliteAutoUpdate();
  },delay);
}

scheduleSatelliteAutoUpdate();

// 
// CROSS-SECTION FEATURE
function startCsDrawing(){
  CS.drawing=true; CS.ptA=null; CS.ptB=null;
  if(CS.markerA){CS.markerA.remove();CS.markerA=null;}
  if(CS.markerB){CS.markerB.remove();CS.markerB=null;}
  removeCsLineFromMaps();
  csBtn.classList.add('cs-active');
  csInstruction.style.display='inline';
  csInstruction.textContent='Click point A on the map';
  maps.forEach(m=>m.getCanvas().style.cursor='crosshair');
}
function endCsDrawing(){
  CS.drawing=false;
  csBtn.classList.remove('cs-active');
  csInstruction.style.display='none';
  maps.forEach(m=>m.getCanvas().style.cursor='');
}
csBtn.onclick=()=>{ if(CS.drawing) endCsDrawing(); else startCsDrawing(); };
document.getElementById('tvsBtn').onclick=()=>{
  tvsOverlayEnabled=!tvsOverlayEnabled;
  const btn=document.getElementById('tvsBtn');
  btn.classList.toggle('tvs-active', tvsOverlayEnabled);
  refreshAllMaps(false);
  try{saveUiState();}catch(e){}
};

//ESWD overlay toggle
document.getElementById('eswdBtn').onclick=()=>{
  eswdOverlayEnabled=!eswdOverlayEnabled;
  const btn=document.getElementById('eswdBtn');
  btn.classList.toggle('eswd-active', eswdOverlayEnabled);
  try{saveUiState();}catch(e){}
  if(eswdOverlayEnabled){
    // Determine if we are currently viewing an archive scan
    const archiveIso = _currentArchiveIso();
    _fetchAndApplyEswd(archiveIso);
  } else {
    // Remove overlay from all maps immediately
    maps.forEach((map, i) => {
      if(!mapReady[i]) return;
      removeEswdFromMap(map, mapPrefix(i));
    });
    _eswdGeoJSON = null;
  }
};

// Dealias toggle (server-side NLradar / optional UNRAVEL)
document.getElementById('dealiasBtn').onclick=()=>{
  if(location.protocol==='file:'){
    warn.textContent = "Dealiasing requires: python eme.py --serve";
    setTimeout(()=>{ warn.textContent=''; }, 3500);
    return;
  }
  if(buildInfo && buildInfo.hasRadarDeps === false){
    warn.textContent = "Dealiasing unavailable (radar dependencies missing on server)";
    setTimeout(()=>{ warn.textContent=''; }, 3500);
    return;
  }
  dealiasEnabled = !dealiasEnabled;
  const btn=document.getElementById('dealiasBtn');
  btn.classList.toggle('dealias-active', dealiasEnabled);
  try{ _scanCache && _scanCache.clear && _scanCache.clear(); }catch(e){}
  timeOffsetMin = 0;
  refreshPayloadInPlace();
  try{saveUiState();}catch(e){}
};


maps.forEach((map,mapIdx)=>{
  map.on('click',async e=>{
    if(!CS.drawing) return;
    const ll=[e.lngLat.lng,e.lngLat.lat];
    if(!CS.ptA){
      CS.ptA=ll; CS.markerA=makeCsPointMarker('A',ll).addTo(map);
      csInstruction.textContent='📍 Click point B on the map';
    } else if(!CS.ptB){
      CS.ptB=ll; CS.markerB=makeCsPointMarker('B',ll).addTo(map);
      endCsDrawing(); drawCsLineOnMaps(); await triggerCrossSection();
    }
  });
});

function makeCsPointMarker(label,lnglat){
  const el=document.createElement('div');
  el.style.cssText='width:20px;height:20px;border-radius:50%;background:#ff6600;border:2px solid #fff;display:flex;align-items:center;justify-content:center;color:#fff;font-size:10px;font-weight:bold;box-shadow:0 0 0 3px rgba(255,102,0,0.4)';
  el.textContent=label;
  return new mapboxgl.Marker(el).setLngLat(lnglat);
}

const CS_LINE_SOURCE='cs-line-src', CS_LINE_LAYER='cs-line-lyr';
function drawCsLineOnMaps(){
  if(!CS.ptA||!CS.ptB) return;
  const geojson={type:'Feature',geometry:{type:'LineString',coordinates:[CS.ptA,CS.ptB]},properties:{}};
  maps.forEach(map=>{
    if(!mapReady[maps.indexOf(map)]) return;
    if(map.getSource(CS_LINE_SOURCE)){
      map.getSource(CS_LINE_SOURCE).setData(geojson);
    } else {
      map.addSource(CS_LINE_SOURCE,{type:'geojson',data:geojson});
      map.addLayer({id:CS_LINE_LAYER,type:'line',source:CS_LINE_SOURCE,paint:{'line-color':'#ff6600','line-width':2.5,'line-dasharray':[4,2]}});
      map.addSource('cs-labels-src',{type:'geojson',data:{type:'FeatureCollection',features:[
        {type:'Feature',geometry:{type:'Point',coordinates:CS.ptA},properties:{label:'A'}},
        {type:'Feature',geometry:{type:'Point',coordinates:CS.ptB},properties:{label:'B'}},
      ]}});
      map.addLayer({id:'cs-labels-lyr',type:'symbol',source:'cs-labels-src',
        layout:{'text-field':['get','label'],'text-size':14,'text-offset':[0,-1.5]},
        paint:{'text-color':'#ff6600','text-halo-color':'#000','text-halo-width':1.5}});
    }
  });
}
function reapplyCsLine(map,_idx){ if(CS.ptA&&CS.ptB) drawCsLineOnMaps(); }
function removeCsLineFromMaps(){
  maps.forEach(map=>{
    ['cs-labels-lyr','cs-labels-src',CS_LINE_LAYER,CS_LINE_SOURCE].forEach(id=>{
      if(map.getLayer&&map.getLayer(id)) map.removeLayer(id);
      if(map.getSource&&map.getSource(id)) map.removeSource(id);
    });
  });
}

document.getElementById('csClose').onclick=()=>{
  csPanel.classList.remove('cs-visible');
  removeCsLineFromMaps();
  if(CS.markerA){CS.markerA.remove();CS.markerA=null;}
  if(CS.markerB){CS.markerB.remove();CS.markerB=null;}
  CS.ptA=CS.ptB=null;
  setTimeout(()=>maps.forEach(m=>m.resize()),80);
};
document.getElementById('csRedraw').onclick=()=>triggerCrossSection();
csProdSel.onchange=()=>{ if(CS.ptA&&CS.ptB) triggerCrossSection(); };
csMaxAlt.onchange =()=>{ if(CS.ptA&&CS.ptB) triggerCrossSection(); };

// sweep decode cache 
const sweepDecodeCache = {};
async function decodeSweep(si) {
  if (sweepDecodeCache[si.b64]) return sweepDecodeCache[si.b64];
  const data = await decodeGzipUint16(si.b64, si.nrays * si.nbins);
  sweepDecodeCache[si.b64] = data;
  return data;
}

function lookupPolar(data, si, azimuthDeg, rangeKm) {
  const {nrays, nbins, rscaleKm, rstartKm, minVal, maxVal} = si;
  const binIdx = Math.round((rangeKm - rstartKm) / rscaleKm);
  if (binIdx < 0 || binIdx >= nbins) return NaN;
  const rayIdx = Math.round((azimuthDeg / 360) * nrays) % nrays;
  const enc = data[rayIdx * nbins + binIdx];
  if (enc === 0) return NaN;
  return minVal + ((enc - 1) / 65534.0) * (maxVal - minVal);
}

function haversineKm(lat1,lon1,lat2,lon2){
  const R=6371,dLat=(lat2-lat1)*Math.PI/180,dLon=(lon2-lon1)*Math.PI/180;
  const a=Math.sin(dLat/2)**2+Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)**2;
  return 2*R*Math.asin(Math.min(1,Math.sqrt(a)));
}
function lerpLatLon(lat1,lon1,lat2,lon2,t){ return [lat1+t*(lat2-lat1),lon1+t*(lon2-lon1)]; }
function toRadarPolar(rLat,rLon,lat,lon){
  const R=6371,lat1=rLat*Math.PI/180,lat2=lat*Math.PI/180;
  const dLat=(lat-rLat)*Math.PI/180,dLon=(lon-rLon)*Math.PI/180;
  const a=Math.sin(dLat/2)**2+Math.cos(lat1)*Math.cos(lat2)*Math.sin(dLon/2)**2;
  const range=2*R*Math.asin(Math.min(1,Math.sqrt(a)));
  const y=Math.sin(dLon)*Math.cos(lat2);
  const x=Math.cos(lat1)*Math.sin(lat2)-Math.sin(lat1)*Math.cos(lat2)*Math.cos(dLon);
  return {range, azimuth:((Math.atan2(y,x)*180/Math.PI)+360)%360};
}
function beamHeightKm(rangeKm, elevDeg) {
  const elRad = elevDeg * Math.PI / 180;
  const ke = 4/3, Re = 6371, keRe = ke * Re;
  return Math.sqrt(rangeKm**2 + keRe**2 + 2*rangeKm*keRe*Math.sin(elRad)) - keRe;
}

// Coordinate grid 
function gridStepForZoom(z) {
  if (z < 4)  return 10;
  if (z < 5)  return 5;
  if (z < 6)  return 2;
  if (z < 7)  return 1;
  if (z < 8)  return 0.5;
  if (z < 9)  return 0.25;
  if (z < 11) return 0.1;
  return 0.05;
}
function buildCoordGridGeoJSON(bounds, zoom) {
  const step = gridStepForZoom(zoom);
  const pad  = step * 2;
  const features = [];
  const lonMin = Math.floor((bounds.getWest()  - pad) / step) * step;
  const lonMax = Math.ceil( (bounds.getEast()  + pad) / step) * step;
  const latMin = Math.max(-85, Math.floor((bounds.getSouth() - pad) / step) * step);
  const latMax = Math.min( 85, Math.ceil( (bounds.getNorth() + pad) / step) * step);
  // Meridians (lon lines)
  const latPts = Math.max(32, Math.ceil((latMax - latMin) / step * 8));
  for (let lon = lonMin; lon <= lonMax + 1e-9; lon = +(lon + step).toFixed(10)) {
    const cs = [];
    for (let i = 0; i <= latPts; i++) cs.push([lon, latMin + (latMax - latMin) * i / latPts]);
    if (cs.length >= 2) features.push({type:'Feature',geometry:{type:'LineString',coordinates:cs},properties:{}});
  }
  // Parallels (lat lines)
  const lonPts = Math.max(32, Math.ceil((lonMax - lonMin) / step * 8));
  for (let lat = latMin; lat <= latMax + 1e-9; lat = +(lat + step).toFixed(10)) {
    const cs = [];
    for (let i = 0; i <= lonPts; i++) cs.push([lonMin + (lonMax - lonMin) * i / lonPts, lat]);
    if (cs.length >= 2) features.push({type:'Feature',geometry:{type:'LineString',coordinates:cs},properties:{}});
  }
  return {type:'FeatureCollection', features};
}
function updateCoordGridForMap(map, idx) {
  if (!mapReady[idx]) return;
  const SRC = 'coord-grid-src', LYR = 'coord-grid-lyr';
  if (!layerSettings.coordGrid) {
    removeLayerIfExists(map, LYR);
    removeSourceIfExists(map, SRC);
    return;
  }
  const fc = buildCoordGridGeoJSON(map.getBounds(), map.getZoom());
  if (map.getSource(SRC)) { map.getSource(SRC).setData(fc); return; }
  map.addSource(SRC, {type:'geojson', data:fc});
  const spec = {
    id: LYR, type: 'line', source: SRC,
    paint: {
      'line-color': 'rgba(255,255,255,0.38)',
      'line-width': 0.75,
      'line-dasharray': [3, 7],
    },
  };
  if (_mapIsSlot[idx]) map.addLayer({...spec, slot:'bottom'});
  else map.addLayer(spec, _mapBeforeId[idx] || undefined);
}

// Cursor coordinate / product-value / distance readout
const _cursorCoordEl = document.getElementById('cursorCoord');
const _cursorValEl   = document.getElementById('cursorVal');
const _cursorDistEl  = document.getElementById('cursorDist');
function _clearCursorInfo() {
  if (_cursorCoordEl) _cursorCoordEl.textContent = '—';
  if (_cursorValEl)   _cursorValEl.textContent   = '—';
  if (_cursorDistEl)  _cursorDistEl.textContent  = '—';
}
maps.forEach((map, mapIndex) => {
  map.on('mousemove', e => {
    const {lng, lat} = e.lngLat;
    // Coordinates
    if (_cursorCoordEl)
      _cursorCoordEl.textContent = `${lat.toFixed(7)}  ${lng.toFixed(7)}`;
    // Station info
    const sn = stationSel.value;
    const si = APP.stations[sn];
    // Distance from radar center
    if (si && _cursorDistEl) {
      const d = haversineKm(lat, lng, si.lat, si.lon);
      _cursorDistEl.textContent = `${d.toFixed(1)} km`;
    } else if (_cursorDistEl) {
      _cursorDistEl.textContent = '—';
    }
    // Product value lookup (polar only, uses the sweep decode cache)
    if (_cursorValEl) {
      try {
        const pk = panelProducts[mapIndex]();
        const sd = APP.rendered[sn] || {};
        const pd = (sd.products || {})[pk];
        if (pd && pd.type === 'polar' && si) {
          let ov = pd.overlayData;
          if (pd.allTiltOverlays && pd.allTiltOverlays.length) {
            const ci = Math.max(0, Math.min(tiltIdx[mapIndex], pd.allTiltOverlays.length - 1));
            ov = pd.allTiltOverlays[ci];
          }
          if (ov && ov.b64) {
            const polar = toRadarPolar(si.lat, si.lon, lat, lng);
            const productLabel = APP.products[pk] || pk;
            const cached = sweepDecodeCache[ov.b64];
            if (cached) {
              const val = lookupPolar(cached, ov, polar.azimuth, polar.range);
              _cursorValEl.textContent = Number.isFinite(val)
                ? `${val.toFixed(2)}  [${productLabel}]`
                : 'no data';
            } else {
              // Trigger decode so next move will hit cache; show pending
              _cursorValEl.textContent = '…';
              decodeSweep(ov).then(data => {
                const v2 = lookupPolar(data, ov, polar.azimuth, polar.range);
                if (_cursorValEl) _cursorValEl.textContent = Number.isFinite(v2)
                  ? `${v2.toFixed(2)}  [${productLabel}]`
                  : 'no data';
              }).catch(() => { if (_cursorValEl) _cursorValEl.textContent = '—'; });
            }
          } else {
            _cursorValEl.textContent = '—';
          }
        } else {
          _cursorValEl.textContent = '—';
        }
      } catch(_e) { _cursorValEl.textContent = '—'; }
    }
  });
  map.on('mouseleave', () => _clearCursorInfo());
  map.on('mouseout',   () => _clearCursorInfo());
  // Update grid on pan/zoom
  map.on('moveend', () => { if (layerSettings.coordGrid) updateCoordGridForMap(map, mapIndex); });
});

async function triggerCrossSection(){
  if(!CS.ptA||!CS.ptB) return;
  csPanel.classList.add('cs-visible');
  csOverlay.textContent='⏳ Decoding sweep data…';
  csOverlay.style.display='flex';
  setTimeout(()=>maps.forEach(m=>m.resize()),80);

  const station=stationSel.value, product=csProdSel.value;
  const stInfo=APP.stations[station];
  const sweepInfos=((APP.rendered[station]||{}).products||{})[product];

  if(!sweepInfos||!sweepInfos.sweeps||!sweepInfos.sweeps.length){
    csOverlay.textContent=`⚠ No sweep data for ${station} / ${APP.products[product]}`;
    return;
  }
  try{
    const decoded=await Promise.all(sweepInfos.sweeps.map(si=>decodeSweep(si)));
    csOverlay.style.display='none';
    paintCrossSection(decoded, sweepInfos.sweeps, stInfo, product);
  }catch(err){
    csOverlay.textContent=`⚠ Error decoding sweeps: ${err.message}`;
  }
}

function paintCrossSection(decodedArrays, sweepInfos, stInfo, productKey) {
  const [lngA, latA] = CS.ptA, [lngB, latB] = CS.ptB;
  const totalDistKm  = haversineKm(latA, lngA, latB, lngB);

  const wrap = document.getElementById('csCanvasWrap');
  const W = wrap.clientWidth || 900, H = wrap.clientHeight || 300;
  csCanvas.width = W; csCanvas.height = H;
  csCanvas.style.width = W + 'px'; csCanvas.style.height = H + 'px';

  const ML = 58, MR = 82, MT = 40, MB = 46;
  const DW = W - ML - MR, DH = H - MT - MB;
  const ctx = csCanvas.getContext('2d');

  const maxHeightKm = parseFloat(csMaxAlt.value) || 10;
  const hy = h => MT + (1 - h / maxHeightKm) * DH;
  const dx = d => ML + (d / totalDistKm) * DW;

  ctx.fillStyle = '#090c10';
  ctx.fillRect(0, 0, W, H);

  const atmGrad = ctx.createLinearGradient(0, MT, 0, MT + DH);
  atmGrad.addColorStop(0.00, '#070b14');
  atmGrad.addColorStop(0.35, '#080d18');
  atmGrad.addColorStop(0.70, '#090f1a');
  atmGrad.addColorStop(1.00, '#0a1018');
  ctx.fillStyle = atmGrad;
  ctx.fillRect(ML, MT, DW, DH);

  if (maxHeightKm >= 10) {
    const trpY = hy(11);
    if (trpY > MT && trpY < MT + DH) {
      ctx.fillStyle = 'rgba(80,140,200,0.06)';
      ctx.fillRect(ML, trpY - 1, DW, 3);
    }
  }

  const N = DW;
  const samples = new Array(N);
  let stationXfrac = null;
  {
    const dLatLine = latB - latA, dLonLine = lngB - lngA;
    const lenSq = dLatLine**2 + dLonLine**2;
    if (lenSq > 1e-12) {
      const t0 = ((stInfo.lat - latA)*dLatLine + (stInfo.lon - lngA)*dLonLine) / lenSq;
      if (t0 >= 0 && t0 <= 1) {
        const projLat = latA + t0*dLatLine, projLon = lngA + t0*dLonLine;
        const distFromLine = haversineKm(stInfo.lat, stInfo.lon, projLat, projLon);
        if (distFromLine < 5) stationXfrac = t0;
      }
    }
  }

  for (let xi = 0; xi < N; xi++) {
    const t = xi / Math.max(N - 1, 1);
    const [sLat, sLon] = lerpLatLon(latA, lngA, latB, lngB, t);
    const { range, azimuth } = toRadarPolar(stInfo.lat, stInfo.lon, sLat, sLon);
    const bh = sweepInfos.map(si => beamHeightKm(range, si.elevation));
    samples[xi] = { range, azimuth, beamH: bh, distKm: t * totalDistKm };
  }

  const BLIND_RANGE_KM = sweepInfos[0] ? (sweepInfos[0].rstartKm || 0.5) : 0.5;

  const imgData = ctx.createImageData(DW, DH);
  const px      = imgData.data;
  const stops   = getColormap(productKey).slice().sort((a, b) => a.value - b.value);

  for (let xi = 0; xi < N; xi++) {
    const { range, azimuth, beamH } = samples[xi];
    const inBlind = range < BLIND_RANGE_KM;

    const vals = sweepInfos.map((si, idx) => {
      if (inBlind) return NaN;
      if (range < si.rstartKm - 0.5 || range > si.maxRangeKm + 0.5) return NaN;
      return lookupPolar(decodedArrays[idx], si, azimuth, range);
    });

    const lowestBeamH = beamH[0];

    for (let yi = 0; yi < DH; yi++) {
      const h = (1 - yi / Math.max(DH - 1, 1)) * maxHeightKm;

      if (inBlind) {
        if ((xi + yi) % 6 < 1) {
          const pIdx = (yi * DW + xi) * 4;
          px[pIdx] = 35; px[pIdx+1] = 35; px[pIdx+2] = 40; px[pIdx+3] = 180;
        }
        continue;
      }

      if (h < lowestBeamH) {
        const val = vals[0];
        if (isNaN(val)) continue;
        const depthFrac = Math.min(1, (lowestBeamH - h) / Math.max(lowestBeamH, 0.5));
        const alpha = Math.round(255 * (1.0 - depthFrac * 0.45));
        const rgb = interpolateRgb(stops, val);
        const pIdx = (yi * DW + xi) * 4;
        px[pIdx] = rgb[0]; px[pIdx+1] = rgb[1]; px[pIdx+2] = rgb[2]; px[pIdx+3] = alpha;
        continue;
      }

      let lo = -1, hi = -1;
      for (let pi = 0; pi < sweepInfos.length - 1; pi++) {
        if (beamH[pi] <= h && beamH[pi + 1] >= h) { lo = pi; hi = pi + 1; break; }
      }

      let val, alpha = 255;
      if (lo >= 0) {
        const hspan = beamH[hi] - beamH[lo];
        const t2    = hspan > 1e-4 ? (h - beamH[lo]) / hspan : 0;
        const vlo = vals[lo], vhi = vals[hi];
        if (isNaN(vlo) && isNaN(vhi)) continue;
        val = isNaN(vlo) ? vhi : isNaN(vhi) ? vlo : vlo + t2 * (vhi - vlo);
        if (hspan > 3.0) alpha = Math.round(255 * Math.max(0.4, 1 - (hspan - 3.0) / 12.0));
      } else {
        const distAbove = h - beamH[sweepInfos.length - 1];
        if (distAbove > 2.0) continue;
        val = vals[sweepInfos.length - 1];
        alpha = Math.round(255 * Math.max(0.1, 1 - distAbove / 2.0));
      }
      if (isNaN(val)) continue;

      const rgb = interpolateRgb(stops, val);
      const pIdx = (yi * DW + xi) * 4;
      px[pIdx] = rgb[0]; px[pIdx+1] = rgb[1]; px[pIdx+2] = rgb[2]; px[pIdx+3] = alpha;
    }
  }
  ctx.putImageData(imgData, ML, MT);

  ctx.save();
  ctx.beginPath(); ctx.rect(ML, MT, DW, DH); ctx.clip();

  const hMajor = maxHeightKm <= 5 ? 1 : maxHeightKm <= 10 ? 2 : maxHeightKm <= 15 ? 2 : 5;
  const hMinor = hMajor / 2;
  for (let h = 0; h <= maxHeightKm + 1e-9; h += hMinor) {
    const y    = hy(h);
    const isMaj = Math.abs(h % hMajor) < 1e-9;
    ctx.strokeStyle = isMaj ? 'rgba(100,140,180,0.20)' : 'rgba(70,100,130,0.10)';
    ctx.lineWidth   = isMaj ? 1 : 0.5;
    ctx.setLineDash(isMaj ? [6, 8] : [2, 8]);
    ctx.beginPath(); ctx.moveTo(ML, y); ctx.lineTo(ML + DW, y); ctx.stroke();
  }
  ctx.setLineDash([]);

  if (maxHeightKm >= 10) {
    const trpY = hy(11);
    if (trpY > MT + 6 && trpY < MT + DH - 6) {
      ctx.setLineDash([2, 5]);
      ctx.strokeStyle = 'rgba(80,140,200,0.30)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ML, trpY); ctx.lineTo(ML + DW, trpY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(80,140,200,0.55)';
      ctx.font = '8px "Courier New",monospace'; ctx.textAlign = 'left';
      ctx.fillText('TROP ~11 km', ML + 4, trpY - 3);
    }
  }

  const rawXStep = totalDistKm / 7;
  const xStep = [1,2,5,10,20,25,50,100,150,200,250,500].find(s => s >= rawXStep) || 100;
  ctx.strokeStyle = 'rgba(70,100,130,0.14)'; ctx.lineWidth = 0.5; ctx.setLineDash([3, 8]);
  for (let d = xStep; d < totalDistKm - xStep * 0.1; d += xStep) {
    const x = dx(d);
    ctx.beginPath(); ctx.moveTo(x, MT); ctx.lineTo(x, MT + DH); ctx.stroke();
  }
  ctx.setLineDash([]);

  for (let si = 0; si < sweepInfos.length; si++) {
    const elev = sweepInfos[si].elevation;
    const maxRange = sweepInfos[si].maxRangeKm;
    ctx.beginPath();
    let first = true;
    for (let xi = 0; xi < N; xi += 2) {
      const s = samples[xi];
      if (s.range > maxRange + 1 || s.range < BLIND_RANGE_KM) { first = true; continue; }
      const h = s.beamH[si];
      if (h > maxHeightKm * 1.02) { first = true; break; }
      const x = ML + (xi / Math.max(N - 1, 1)) * DW;
      const y = hy(h);
      if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(220,190,60,0.35)';
    ctx.lineWidth = 0.8; ctx.setLineDash([4, 8]);
    ctx.stroke(); ctx.setLineDash([]);

    let labelX = null, labelY = null;
    for (let xi = Math.floor(N * 0.05); xi < N; xi += 4) {
      const s = samples[xi];
      if (s.range < BLIND_RANGE_KM || s.range > maxRange) continue;
      const h = s.beamH[si];
      if (h >= 0 && h <= maxHeightKm) {
        labelX = ML + (xi / Math.max(N - 1, 1)) * DW;
        labelY = hy(h);
        break;
      }
    }
    if (labelX !== null) {
      ctx.fillStyle = 'rgba(220,190,60,0.70)';
      ctx.font = '7.5px "Courier New",monospace'; ctx.textAlign = 'left';
      ctx.fillText(`${elev.toFixed(1)}°`, labelX + 2, labelY - 2);
    }
  }

  const groundY = MT + DH;
  const earthGrd = ctx.createLinearGradient(0, groundY - 4, 0, groundY + 6);
  earthGrd.addColorStop(0, 'rgba(110,82,30,0.85)');
  earthGrd.addColorStop(1, 'rgba(40,28,10,0.95)');
  ctx.fillStyle = earthGrd;
  ctx.fillRect(ML, groundY - 3, DW, 10);
  ctx.strokeStyle = 'rgba(190,150,60,0.80)'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(ML, groundY - 2); ctx.lineTo(ML + DW, groundY - 2); ctx.stroke();

  {
    let blindEndXi = 0;
    for (let xi = 0; xi < N; xi++) {
      if (samples[xi].range >= BLIND_RANGE_KM) { blindEndXi = xi; break; }
    }
    if (blindEndXi > 0) {
      const bw = (blindEndXi / Math.max(N - 1, 1)) * DW;
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(ML, MT, bw, DH);
      ctx.strokeStyle = 'rgba(80,80,80,0.5)'; ctx.lineWidth = 0.8; ctx.setLineDash([3,6]);
      for (let i = -DH; i < bw + DH; i += 10) {
        ctx.beginPath();
        ctx.moveTo(ML + Math.max(0, i), MT);
        ctx.lineTo(ML + Math.min(bw, i + DH), MT + Math.min(DH, bw - i));
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
  }

  if (stationXfrac !== null) {
    const sx = ML + stationXfrac * DW;
    ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 1.5; ctx.setLineDash([3,5]);
    ctx.beginPath(); ctx.moveTo(sx, MT); ctx.lineTo(sx, MT + DH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255,255,255,0.80)';
    ctx.beginPath(); ctx.moveTo(sx, MT + DH - 4); ctx.lineTo(sx - 5, MT + DH + 4); ctx.lineTo(sx + 5, MT + DH + 4); ctx.closePath(); ctx.fill();
    ctx.fillStyle = 'rgba(200,220,240,0.75)'; ctx.font = 'bold 8px Tahoma,sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(stationSel.value.toUpperCase(), sx, MT + DH + 14);
  }

  ctx.restore();

  ctx.strokeStyle = '#1e3048'; ctx.lineWidth = 1.5;
  ctx.strokeRect(ML, MT, DW, DH);
  ctx.strokeStyle = 'rgba(60,100,140,0.3)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(ML, MT + DH); ctx.lineTo(ML, MT); ctx.lineTo(ML + DW, MT); ctx.stroke();

  ctx.fillStyle = '#7a9db8'; ctx.font = '10px "Courier New",monospace'; ctx.textAlign = 'right';
  for (let h = 0; h <= maxHeightKm + 1e-9; h += hMajor) {
    const y = hy(h);
    ctx.strokeStyle = '#1e3048'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ML - 4, y); ctx.lineTo(ML, y); ctx.stroke();
    ctx.fillStyle = '#7a9db8';
    ctx.fillText(h % 1 === 0 ? `${h}` : h.toFixed(1), ML - 6, y + 3.5);
  }
  ctx.save();
  ctx.translate(12, MT + DH / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#4a6880'; ctx.font = 'bold 9px Tahoma,sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('HEIGHT  (km AGL)', 0, 0);
  ctx.restore();

  ctx.fillStyle = '#7a9db8'; ctx.font = '10px "Courier New",monospace'; ctx.textAlign = 'center';
  for (let d = 0; d <= totalDistKm + xStep * 0.01; d += xStep) {
    const x = dx(d);
    ctx.strokeStyle = '#1e3048'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, MT + DH); ctx.lineTo(x, MT + DH + 5); ctx.stroke();
    ctx.fillStyle = '#7a9db8';
    ctx.fillText(`${Math.round(d)}`, x, MT + DH + 16);
  }
  ctx.fillStyle = '#4a6880'; ctx.font = 'bold 9px Tahoma,sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('RANGE ALONG SECTION  (km)', ML + DW / 2, MT + DH + 34);

  ctx.font = 'bold 12px Tahoma,sans-serif';
  ctx.fillStyle = '#ff7733';
  ctx.textAlign = 'center';
  ctx.fillText('A', ML, MT + DH + 14);
  ctx.fillText('B', ML + DW, MT + DH + 14);
  function bearing(lat1, lon1, lat2, lon2) {
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const lat1r = lat1 * Math.PI / 180, lat2r = lat2 * Math.PI / 180;
    const y = Math.sin(dLon) * Math.cos(lat2r);
    const x = Math.cos(lat1r) * Math.sin(lat2r) - Math.sin(lat1r) * Math.cos(lat2r) * Math.cos(dLon);
    return ((Math.atan2(y, x) * 180 / Math.PI) + 360) % 360;
  }
  const brg = bearing(latA, lngA, latB, lngB);
  ctx.fillStyle = '#4a6880'; ctx.font = '8px "Courier New",monospace'; ctx.textAlign = 'left';
  ctx.fillText(`BRG ${brg.toFixed(0)}°`, ML + 2, MT + DH + 34);

  const cbX = ML + DW + 14, cbW = 14, cbH = DH;
  const cbMin = stops[0].value, cbMax = stops[stops.length - 1].value;
  const cbGrad = ctx.createLinearGradient(0, MT + cbH, 0, MT);
  for (let i = 0; i < stops.length; i++) {
    const t = (stops[i].value - cbMin) / Math.max(cbMax - cbMin, 1);
    const c = stops[i].color;
    cbGrad.addColorStop(Math.max(0, Math.min(1, t)), `rgb(${c[0]},${c[1]},${c[2]})`);
  }
  ctx.fillStyle = cbGrad;
  ctx.fillRect(cbX, MT, cbW, cbH);
  ctx.strokeStyle = '#1e3048'; ctx.lineWidth = 1;
  ctx.strokeRect(cbX, MT, cbW, cbH);
  ctx.strokeStyle = 'rgba(100,160,200,0.15)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cbX, MT + cbH); ctx.lineTo(cbX, MT); ctx.lineTo(cbX + cbW, MT); ctx.stroke();

  const cbStep = getLegendStep(productKey) || Math.max(1, Math.round((cbMax - cbMin) / 8));
  ctx.fillStyle = '#7a9db8'; ctx.font = '8.5px "Courier New",monospace'; ctx.textAlign = 'left';
  for (let v = Math.ceil(cbMin / cbStep) * cbStep; v <= cbMax + 1e-9; v += cbStep) {
    const ty = MT + (1 - (v - cbMin) / (cbMax - cbMin)) * cbH;
    ctx.strokeStyle = '#7a9db8'; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(cbX + cbW, ty); ctx.lineTo(cbX + cbW + 4, ty); ctx.stroke();
    ctx.fillText(Number.isInteger(v) ? `${v}` : v.toFixed(1), cbX + cbW + 6, ty + 3.5);
  }
  const units = { reflectivity:'dBZ', velocity:'m/s', correlation:'ρhv', spectrum_width:'m/s',
                  echo_tops:'km', nrot:'m/s', srv:'m/s', meso:'%', azimuthal_shear:'×10⁻³/s', mesh:'cm' };
  const unit = units[productKey] || '';
  if (unit) {
    ctx.save();
    ctx.translate(W - 7, MT + cbH / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#4a6880'; ctx.font = 'bold 9px Tahoma,sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(unit, 0, 0);
    ctx.restore();
  }

  const dtStr = (APP.rendered[stationSel.value] || {}).datetime || '';
  ctx.fillStyle = '#d0e8f8'; ctx.font = 'bold 12px Tahoma,sans-serif'; ctx.textAlign = 'left';
  ctx.fillText(`${APP.products[productKey].toUpperCase()}`, ML, MT - 22);
  ctx.fillStyle = '#5080a0'; ctx.font = '10px Tahoma,sans-serif';
  ctx.fillText(`  ${stationSel.value}  ·  ${Math.round(totalDistKm)} km section`, ML + ctx.measureText(APP.products[productKey].toUpperCase()).width, MT - 22);
  const elevStr = sweepInfos.map(s => `${s.elevation.toFixed(1)}°`).join('  ');
  ctx.fillStyle = '#3d6080'; ctx.font = '8px "Courier New",monospace'; ctx.textAlign = 'left';
  ctx.fillText(`EL  ${elevStr}`, ML, MT - 9);
  if (dtStr) {
    ctx.fillStyle = '#3d6080'; ctx.font = '8px "Courier New",monospace'; ctx.textAlign = 'right';
    ctx.fillText(dtStr.replace('T', ' ').replace('Z', 'Z'), ML + DW, MT - 9);
  }

  csInfo.textContent = `${sweepInfos.length} sweeps · ${Math.round(totalDistKm)} km · 0–${maxHeightKm} km AGL`;
}

// Settings / credits modals
function setupModal(tId,mId,cId){
  document.getElementById(tId).onclick=()=>{const m=document.getElementById(mId);m.classList.add('open');m.setAttribute('aria-hidden','false');};
  document.getElementById(cId).onclick=()=>{const m=document.getElementById(mId);m.classList.remove('open');m.setAttribute('aria-hidden','true');};
  document.getElementById(mId).onclick=e=>{if(e.target.id===mId){e.currentTarget.classList.remove('open');e.currentTarget.setAttribute('aria-hidden','true');}};
}
setupModal('settingsBtn','settingsModal','settingsClose');

// Settings tab switching
document.querySelectorAll('#settingsModal .settings-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('#settingsModal .settings-tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.querySelectorAll('#settingsModal .settings-tab-panel').forEach(p => p.classList.toggle('active', p.id === 'settingsTab' + tab.charAt(0).toUpperCase() + tab.slice(1)));
  });
});

// Custom Radar Upload 
(function(){
  const addBtn      = document.getElementById('addRadarBtn');
  const removeBtn   = document.getElementById('removeRadarBtn');
  const modal       = document.getElementById('customRadarModal');
  const closeBtn    = document.getElementById('customRadarClose');
  const uploadBtn   = document.getElementById('customRadarUploadBtn');
  const nameInp     = document.getElementById('customRadarName');
  const fileInp     = document.getElementById('customRadarFile');
  const fileInpV    = document.getElementById('customRadarFile_V');
  const fileInpPhiDP= document.getElementById('customRadarFile_PhiDP');
  const fileInpRhoHV= document.getElementById('customRadarFile_RhoHV');
  const fileInpW    = document.getElementById('customRadarFile_W');
  const fileInpZDR  = document.getElementById('customRadarFile_ZDR');
  const latInp      = document.getElementById('customRadarLat');
  const lonInp      = document.getElementById('customRadarLon');
  const rangeInp    = document.getElementById('customRadarRange');
  const statusDiv   = document.getElementById('customRadarStatus');
  if (!addBtn || !modal) return;

  function openModal(){
    statusDiv.style.display='none'; statusDiv.textContent='';
    uploadBtn.disabled=false;
    modal.classList.add('open'); modal.setAttribute('aria-hidden','false');
  }
  function closeModal(){
    modal.classList.remove('open'); modal.setAttribute('aria-hidden','true');
  }
  addBtn.onclick = openModal;
  closeBtn.onclick = closeModal;
  modal.onclick = e=>{ if(e.target===modal) closeModal(); };

  function setStatus(msg, type){
    // type: 'ok' | 'err' | 'loading'
    statusDiv.textContent=msg;
    statusDiv.style.display='block';
    const colors = {
      ok:      {bg:'#d4edda',border:'#c3e6cb',color:'#155724'},
      err:     {bg:'#f8d7da',border:'#f5c6cb',color:'#721c24'},
      loading: {bg:'#fff3cd',border:'#ffeeba',color:'#856404'},
    };
    const c = colors[type] || colors.loading;
    statusDiv.style.background=c.bg;
    statusDiv.style.border=`1px solid ${c.border}`;
    statusDiv.style.color=c.color;
  }

  // Show/hide remove button based on selected station's provider
  function syncRemoveBtn(){
    const sn = stationSel.value;
    const info = APP.stations[sn] || {};
    if(removeBtn) removeBtn.style.display = (info.provider==='custom') ? '' : 'none';
  }
  stationSel.addEventListener('change', syncRemoveBtn);

  if(removeBtn){
    removeBtn.onclick = async ()=>{
      const sn = stationSel.value;
      const info = APP.stations[sn] || {};
      if(info.provider !== 'custom') return;
      if(!confirm(`Remove custom radar "${sn}"?`)) return;
      try{
        await fetch('/remove_custom_radar?name='+encodeURIComponent(sn), {method:'POST'});
      }catch(e){}
      // Remove from APP
      delete APP.stations[sn];
      delete APP.rendered[sn];
      populateCountry();
      // Select first available station
      const firstCountry = countrySel.options[0]?.value;
      if(firstCountry){
        countrySel.value=firstCountry;
        populateStationsForCountry(firstCountry);
      }
      syncRemoveBtn();
      stationSel.dispatchEvent(new Event('change'));
    };
  }

  // Tab switching for file vs API endpoint
  const tabFileBtn  = document.getElementById('customRadarTabFile');
  const tabApiBtn   = document.getElementById('customRadarTabApi');
  const panelFile   = document.getElementById('customRadarPanelFile');
  const panelApi    = document.getElementById('customRadarPanelApi');
  const fetchBtn    = document.getElementById('customRadarFetchBtn');
  const apiNameInp  = document.getElementById('customRadarApiName');
  const apiUrlInp   = document.getElementById('customRadarApiUrl');
  const apiLatInp   = document.getElementById('customRadarApiLat');
  const apiLonInp   = document.getElementById('customRadarApiLon');
  const apiRangeInp = document.getElementById('customRadarApiRange');
  let activeTab = 'file';

  function switchCustomTab(tab) {
    activeTab = tab;
    const onFile = tab === 'file';
    tabFileBtn.style.borderBottomColor = onFile ? '#2a6a2a' : 'transparent';
    tabFileBtn.style.color = onFile ? '#2a6a2a' : '#555';
    tabFileBtn.style.fontWeight = onFile ? '600' : '400';
    tabApiBtn.style.borderBottomColor = onFile ? 'transparent' : '#2a6a2a';
    tabApiBtn.style.color = onFile ? '#555' : '#2a6a2a';
    tabApiBtn.style.fontWeight = onFile ? '400' : '600';
    panelFile.style.display = onFile ? '' : 'none';
    panelApi.style.display  = onFile ? 'none' : '';
    uploadBtn.style.display = onFile ? '' : 'none';
    if(fetchBtn) fetchBtn.style.display = onFile ? 'none' : '';
    statusDiv.style.display = 'none';
  }
  if(tabFileBtn) tabFileBtn.onclick = () => switchCustomTab('file');
  if(tabApiBtn)  tabApiBtn.onclick  = () => switchCustomTab('api');

  //Fetch & Render (API endpoint)
  if(fetchBtn){
    fetchBtn.onclick = async () => {
      const name = (apiNameInp && apiNameInp.value.trim()) || '';
      const url  = (apiUrlInp  && apiUrlInp.value.trim())  || '';
      if(!name){ setStatus('Please enter a radar name.','err'); return; }
      if(!url) { setStatus('Please enter an API endpoint URL.','err'); return; }
      fetchBtn.disabled = true;
      setStatus(`Fetching latest file from endpoint and rendering "${name}" — this may take 20–90 s…`,'loading');
      try {
        const body = JSON.stringify({
          name,
          url,
          lat:      apiLatInp   ? (apiLatInp.value.trim()   || null) : null,
          lon:      apiLonInp   ? (apiLonInp.value.trim()   || null) : null,
          range_km: apiRangeInp ? (apiRangeInp.value.trim() || null) : null,
        });
        const resp = await fetch('/add_radar_from_url', {method:'POST', headers:{'Content-Type':'application/json'}, body});
        const data = await resp.json();
        if(!resp.ok){ setStatus('Server error: '+(data.error||resp.statusText),'err'); fetchBtn.disabled=false; return; }
        APP.stations[data.name] = data.stationMeta;
        APP.rendered[data.name] = data.rendered;
        populateCountry();
        const targetCountry = 'Custom';
        if([...countrySel.options].some(o=>o.value===targetCountry)){
          countrySel.value = targetCountry;
          populateStationsForCountry(targetCountry);
        }
        if(APP.stations[data.name]) stationSel.value = data.name;
        syncRemoveBtn();
        stationSel.dispatchEvent(new Event('change'));
        setStatus(`✓ "${data.name}" added successfully.`,'ok');
        fetchBtn.disabled = false;
        setTimeout(closeModal, 1500);
        if(apiNameInp)  apiNameInp.value  = '';
        if(apiUrlInp)   apiUrlInp.value   = '';
        if(apiLatInp)   apiLatInp.value   = '';
        if(apiLonInp)   apiLonInp.value   = '';
        if(apiRangeInp) apiRangeInp.value = '';
      } catch(err) {
        setStatus('Fetch failed: '+err,'err');
        fetchBtn.disabled = false;
      }
    };
  }

  uploadBtn.onclick = async ()=>{
    const name = nameInp.value.trim();
    if(!name){ setStatus('Please enter a radar name.','err'); return; }
    const file = fileInp.files && fileInp.files[0];
    if(!file){ setStatus('Please select a radar file.','err'); return; }

    uploadBtn.disabled=true;
    setStatus(`Uploading and rendering "${name}" — this may take 10–60 s depending on file size…`,'loading');

    try{
      const fd = new FormData();
      fd.append('name', name);
      fd.append('file', file);
      const lat   = latInp.value.trim();
      const lon   = lonInp.value.trim();
      const range = rangeInp.value.trim();
      if(lat)   fd.append('lat',      lat);
      if(lon)   fd.append('lon',      lon);
      if(range) fd.append('range_km', range);
      // Optional per-product files
      const _extraFiles = [
        ['file_V',     fileInpV],
        ['file_PhiDP', fileInpPhiDP],
        ['file_RhoHV', fileInpRhoHV],
        ['file_W',     fileInpW],
        ['file_ZDR',   fileInpZDR],
      ];
      for (const [fieldName, inp] of _extraFiles) {
        if (inp && inp.files && inp.files[0]) fd.append(fieldName, inp.files[0]);
      }

      const resp = await fetch('/upload_radar', {method:'POST', body:fd});
      const data = await resp.json();
      if(!resp.ok){ setStatus('Server error: '+(data.error||resp.statusText),'err'); uploadBtn.disabled=false; return; }

      // Inject into APP
      APP.stations[data.name] = data.stationMeta;
      APP.rendered[data.name] = data.rendered;

      // Repopulate dropdowns and select new station
      populateCountry();
      const targetCountry = 'Custom';
      if([...countrySel.options].some(o=>o.value===targetCountry)){
        countrySel.value = targetCountry;
        populateStationsForCountry(targetCountry);
      }
      if(APP.stations[data.name]) stationSel.value = data.name;
      syncRemoveBtn();
      stationSel.dispatchEvent(new Event('change'));

      setStatus(`✓ "${data.name}" added successfully.`,'ok');
      uploadBtn.disabled=false;
      // Auto-close after 1.5 s
      setTimeout(closeModal, 1500);

      // Reset form
      nameInp.value=''; fileInp.value=''; latInp.value=''; lonInp.value=''; rangeInp.value='';
      if(fileInpV)     fileInpV.value='';
      if(fileInpPhiDP) fileInpPhiDP.value='';
      if(fileInpRhoHV) fileInpRhoHV.value='';
      if(fileInpW)     fileInpW.value='';
      if(fileInpZDR)   fileInpZDR.value='';
    }catch(err){
      setStatus('Upload failed: '+err,'err');
      uploadBtn.disabled=false;
    }
  };
})();

// Force Data Refresh 
(function(){
  const btn    = document.getElementById('forceRefreshBtn');
  const status = document.getElementById('forceRefreshStatus');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = '⏳ Fetching…';
    status.style.color = '#555';
    status.textContent = 'Building fresh data — this may take 30–60 s…';
    try {
      const url = new URL('force_refresh', window.location.href);
      url.searchParams.set('_ts', String(Date.now()));
      const resp = await fetch(url.toString(), { cache: 'no-store' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      if (data && data.rendered) {
        APP.rendered = data.rendered;
        APP.errors   = data.errors || {};
        renderSettingsErrors();
        warn.textContent = '';
        refreshAllMaps(false);
        renderAllLegends();
        try { refreshLightning(); }  catch(e) {}
        try { refreshMetObs(); }     catch(e) {}
        // Re-anchor the countdown to the freshly-fetched scan time
        autoUpdateCfg.initialMs = computeInitialCountdownFromScanTime();
        startAutoUpdateCountdown();
        const ts = new Date().toLocaleTimeString();
        status.style.color = '#106910';
        status.textContent = `✓ Refreshed at ${ts}`;
      } else {
        throw new Error('Empty payload returned');
      }
    } catch (e) {
      console.warn('[forceRefresh] failed', e);
      status.style.color = '#7d0000';
      status.textContent = `✗ Failed: ${e.message}`;
    } finally {
      btn.disabled = false;
      btn.textContent = '↺ Force Data Refresh';
    }
  });
})();

// Color table upload
function parseColorCsv(content){const lines=content.split('\n').map(l=>l.trim()).filter(Boolean);const stops=[];let step=null;for(const line of lines){const lower=line.toLowerCase();if(lower.startsWith('step:')){const v=Number(line.split(':').slice(1).join(':').trim());if(Number.isFinite(v))step=v;continue;}const parts=line.split(',').map(s=>s.trim());if(parts.length!==4)continue;const nums=parts.map(Number);if(!nums.every(Number.isFinite))continue;stops.push({value:nums[0],color:[nums[1],nums[2],nums[3]]});}if(!stops.length)throw new Error('No valid color stops in CSV');stops.sort((a,b)=>a.value-b.value);return{stops,step};}
function parseColorPal(content){const lines=content.split('\n').map(l=>l.trim()).filter(Boolean);const stops=[];let step=null;for(const line of lines){const lower=line.toLowerCase();if(lower.startsWith('step:')){const v=Number(line.split(':').slice(1).join(':').trim());if(Number.isFinite(v))step=v;continue;}if(!lower.startsWith('color:'))continue;const body=line.split(':').slice(1).join(':').trim();const parts=body.split(' ').map(s=>s.trim()).filter(Boolean);if(parts.length<4)continue;const nums=parts.slice(0,4).map(Number);if(!nums.every(Number.isFinite))continue;stops.push({value:nums[0],color:[nums[1],nums[2],nums[3]]});}if(!stops.length)throw new Error('No valid Color entries in PAL');stops.sort((a,b)=>a.value-b.value);return{stops,step};}
function parseColorTableFile(name,content){const n=name.toLowerCase();if(n.endsWith('.csv'))return parseColorCsv(content);if(n.endsWith('.pal'))return parseColorPal(content);throw new Error('Unsupported file type');}

function buildColorTableSettingsRows(){
  const root=document.getElementById('colortableRows'); root.innerHTML='';
  Object.entries(APP.products).forEach(([pk,pl])=>{
    const label=document.createElement('div'); label.textContent=pl;
    const btnRow=document.createElement('div'); btnRow.style.cssText='display:flex;gap:4px;align-items:center;';
    const btn=document.createElement('button'); btn.type='button'; btn.textContent='Select file';
    const resetBtn=document.createElement('button'); resetBtn.type='button'; resetBtn.textContent='Reset';
    resetBtn.title='Revert to preset color table';
    resetBtn.style.cssText='font-size:11px;padding:2px 8px;opacity:0.5;cursor:not-allowed;';
    resetBtn.disabled=true;
    const status=document.createElement('div'); status.id=`ct-status-${pk}`; status.textContent='Default'; status.className='status-empty';
    const input=document.createElement('input'); input.type='file'; input.accept='.csv,.pal'; input.style.display='none';
    function setCustomActive(active){
      resetBtn.disabled=!active;
      resetBtn.style.opacity=active?'1':'0.5';
      resetBtn.style.cursor=active?'pointer':'not-allowed';
    }
    // Reflect any already-loaded custom colormap on rebuild
    if(userColormaps[pk]) setCustomActive(true);
    btn.addEventListener('click',()=>input.click());
    resetBtn.addEventListener('click',()=>{
      delete userColormaps[pk];
      delete userLegendSteps[pk];
      status.textContent='Default'; status.className='status-empty';
      setCustomActive(false);
      onColormapChanged(pk);
      try{saveUiState();}catch(e){}
    });
    input.addEventListener('change',async()=>{
      const file=input.files&&input.files[0]; if(!file) return;
      try{
        const parsed=parseColorTableFile(file.name,await file.text());
        userColormaps[pk]=parsed.stops;
        if(parsed.step&&parsed.step>0) userLegendSteps[pk]=parsed.step;
        status.textContent=`Loaded: ${file.name}`; status.className='status-ok';
        setCustomActive(true);
        onColormapChanged(pk);
        try{saveUiState();}catch(e){}
      }catch(err){status.textContent=`Error: ${err&&err.message?err.message:'parse error'}`; status.className='status-empty';}
      finally{input.value='';}
    });
    btnRow.appendChild(btn); btnRow.appendChild(resetBtn);
    root.appendChild(label); root.appendChild(btnRow); root.appendChild(status); root.appendChild(input);
  });
}
buildColorTableSettingsRows();

function buildLayerSettingsRows(){
  const root=document.getElementById('layerRows'); if(!root) return; root.innerHTML='';
  function makeCheckboxRow(id,label,checked,onChange){
    const row=document.createElement('div'); row.className='settings-row';
    const inp=document.createElement('input'); inp.type='checkbox'; inp.id=id; inp.checked=checked;
    const lab=document.createElement('label'); lab.htmlFor=id; lab.textContent=label;
    inp.addEventListener('change',()=>{ onChange(inp.checked); try{saveUiState();}catch(e){} });
    row.appendChild(inp); row.appendChild(lab); root.appendChild(row);
  }
  makeCheckboxRow('layerRangeRings','Range rings',layerSettings.rangeRings,v=>{layerSettings.rangeRings=v;refreshAllMaps(false);});
  makeCheckboxRow('layerCoordGrid','Coordinate grid (lat/lon)',layerSettings.coordGrid,v=>{layerSettings.coordGrid=v;maps.forEach((m,i)=>updateCoordGridForMap(m,i));});
  makeCheckboxRow('layerLightning','Lightning overlay',layerSettings.lightning,v=>{layerSettings.lightning=v;if(v)refreshLightning();else maps.forEach((m,i)=>removeLightningFromMap(m,i));});
  makeCheckboxRow('layerMetObs','Met. observations (DK/SE/NO/DE/NL/AT)',layerSettings.metObs,v=>{
    layerSettings.metObs=v;
    if(v) refreshMetObs();
    else removeMetObsFromMaps();
  });
  makeCheckboxRow('layerGeoColor','GeoColor RGB MTG',layerSettings.geocolor,v=>{
    layerSettings.geocolor=v;
    maps.forEach((m,i)=>applySatelliteToMap(m,i));
  });
  makeCheckboxRow('layerEurHRV','European HRV RGB',layerSettings.hrv,v=>{
    layerSettings.hrv=v;
    maps.forEach((m,i)=>applySatelliteToMap(m,i));
  });
  makeCheckboxRow('layerAirmass','Airmass RGB (0°)',layerSettings.airmass,v=>{
    layerSettings.airmass=v;
    maps.forEach((m,i)=>applySatelliteToMap(m,i));
  });
  makeCheckboxRow('layerBluemarble','Blue Marble basemap (local GeoTIFF)',layerSettings.bluemarble,v=>{
    layerSettings.bluemarble=v;
    applyBluemarbleToAllMaps();
  });
  makeCheckboxRow('layerLegends','Color legend overlay',layerSettings.legends,v=>{layerSettings.legends=v;renderAllLegends();});
  const sizeRow=document.createElement('div'); sizeRow.className='settings-row';
  const sizeLabel=document.createElement('label'); sizeLabel.textContent='Station marker size';
  const sizeInput=document.createElement('input'); sizeInput.type='range'; sizeInput.min='2'; sizeInput.max='18'; sizeInput.value=String(layerSettings.stationSize);
  sizeInput.addEventListener('input',()=>{layerSettings.stationSize=Number(sizeInput.value)||6;maps.forEach((m,i)=>rebuildMarkers(m,i)); try{saveUiState();}catch(e){} });
  sizeRow.appendChild(sizeLabel); sizeRow.appendChild(sizeInput); root.appendChild(sizeRow);
  const ringRow=document.createElement('div'); ringRow.className='settings-row';
  const ringLabel=document.createElement('label'); ringLabel.textContent='Range ring width';
  const ringInput=document.createElement('input'); ringInput.type='range'; ringInput.min='0.5'; ringInput.max='5'; ringInput.step='0.5'; ringInput.value=String(layerSettings.ringWidth);
  ringInput.addEventListener('input',()=>{layerSettings.ringWidth=Number(ringInput.value)||1.5;refreshAllMaps(false); try{saveUiState();}catch(e){} });
  ringRow.appendChild(ringLabel); ringRow.appendChild(ringInput); root.appendChild(ringRow);
}
buildLayerSettingsRows();

renderAllLegends();

// Time controls / auto-update countdown 
const RADAR_UPDATE_INTERVAL_MS = 15 * 60 * 1000;  // 15 min nominal radar update cycle
const autoUpdateCfg={initialMs:RADAR_UPDATE_INTERVAL_MS,retryMs:2.5*60*1000,cycle:0,remainingMs:0,timerId:null,updating:false};
function formatCountdown(ms){const totalSec=Math.max(0,Math.floor(ms/1000));const m=Math.floor(totalSec/60),s=totalSec%60;return`${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;}
function scheduleNextCountdown(){autoUpdateCfg.remainingMs=autoUpdateCfg.cycle===0?autoUpdateCfg.initialMs:autoUpdateCfg.retryMs;}

// Compute initial countdown from the actual scan timestamp so the timer
// fires when the next scan is due, not blindly after 15 min.
function computeInitialCountdownFromScanTime() {
  try {
    const sn = stationSel.value;
    const sd = APP.rendered[sn] || {};
    const dtStr = sd.datetime;
    if (!dtStr || dtStr === 'n/a') return RADAR_UPDATE_INTERVAL_MS;
    const scanMs = new Date(dtStr).getTime();
    if (!Number.isFinite(scanMs)) return RADAR_UPDATE_INTERVAL_MS;
    const elapsedMs = Date.now() - scanMs;
    const remaining = RADAR_UPDATE_INTERVAL_MS - elapsedMs;
    // If remaining is negative, the scan is overdue – trigger soon (30s)
    return remaining > 0 ? remaining : 30 * 1000;
  } catch(e) { return RADAR_UPDATE_INTERVAL_MS; }
}

async function refreshPayloadInPlace(){
  // Do not clobber archive view with live data
  if(archiveMode) return;
  autoUpdateCfg.updating=true;
  try{
    try{ if(typeof saveUiState==='function') saveUiState(); }catch(e){}
    const url=new URL('payload.json', window.location.href);
    const cur=new URL(window.location.href);
    const iso=cur.searchParams.get('dmi_dt');
    if(iso) url.searchParams.set('dmi_dt', iso);
    const liveStation = stationSel && stationSel.value ? stationSel.value : '';
    if(liveStation) url.searchParams.set('station', liveStation);
    if(dealiasEnabled) url.searchParams.set('dealias','1');
    url.searchParams.set('_ts', String(Date.now()));
    const resp=await fetch(url.toString(),{cache:'no-store'});
    if(!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data=await resp.json();
      if(data&&data.rendered){
        APP.rendered=data.rendered;
        APP.errors=data.errors||{};
        renderSettingsErrors();
        warn.textContent='';
        refreshAllMaps(false);
        renderAllLegends();
        refreshLightning();
        refreshMetObs();
        // Refresh ESWD live data (only when in live mode, not archive)
        if(eswdOverlayEnabled && timeOffsetMin === 0) _fetchAndApplyEswd(null);
      }
    autoUpdateCfg.cycle=0;
    // Re-compute countdown based on the freshly-loaded scan timestamp
    autoUpdateCfg.initialMs=computeInitialCountdownFromScanTime();
  }catch(e){
    console.warn('Auto-update failed',e);
    warn.textContent='Auto-update failed; keeping last data.';
    autoUpdateCfg.cycle++;
  }finally{
    scheduleNextCountdown();
    radarCountdown.textContent=formatCountdown(autoUpdateCfg.remainingMs);
    autoUpdateCfg.updating=false;
  }
}

function startAutoUpdateCountdown(){
  if(!radarCountdown) return;
  autoUpdateCfg.cycle=0;
  autoUpdateCfg.initialMs=computeInitialCountdownFromScanTime();
  scheduleNextCountdown();
  radarCountdown.textContent=formatCountdown(autoUpdateCfg.remainingMs);
  if(autoUpdateCfg.timerId) clearInterval(autoUpdateCfg.timerId);
  autoUpdateCfg.timerId=setInterval(()=>{
    if(autoUpdateCfg.updating || archiveMode) return;
    autoUpdateCfg.remainingMs-=1000;
    if(autoUpdateCfg.remainingMs<=0){
      radarCountdown.textContent='00:00';
      refreshPayloadInPlace();
      return;
    }
    radarCountdown.textContent=formatCountdown(autoUpdateCfg.remainingMs);
  },1000);
}
function initTimeInputsFromCurrent(){
  if(!radarDateInp||!radarTimeInp) return;
  // Always populate with a sensible default, even when sd.datetime is unavailable
  const sn=stationSel.value,sd=APP.rendered[sn],iso=sd&&sd.datetime;
  let d;
  if(iso){ d=new Date(iso); if(Number.isNaN(d.getTime())) d=null; }
  if(!d) d=new Date();  // fall back to now (UTC)
  // type="date" value must be YYYY-MM-DD
  const yyyy=d.getUTCFullYear();
  const mm=String(d.getUTCMonth()+1).padStart(2,'0');
  const dd=String(d.getUTCDate()).padStart(2,'0');
  radarDateInp.value=`${yyyy}-${mm}-${dd}`;
  radarTimeInp.value=`${String(d.getUTCHours()).padStart(2,'0')}:${String(d.getUTCMinutes()).padStart(2,'0')}`;
}
function setupTimeControls(){
  if(!radarGoBtn) return;
  try{
    const u=new URL(window.location.href);
    const iso=u.searchParams.get('dmi_dt');
    if(iso){
      const d=new Date(iso);
      if(!Number.isNaN(d.getTime())){
        // type="date" needs YYYY-MM-DD
        const yyyy=d.getUTCFullYear();
        const mm=String(d.getUTCMonth()+1).padStart(2,'0');
        const dd=String(d.getUTCDate()).padStart(2,'0');
        radarDateInp.value=`${yyyy}-${mm}-${dd}`;
        radarTimeInp.value=`${String(d.getUTCHours()).padStart(2,'0')}:${String(d.getUTCMinutes()).padStart(2,'0')}`;
      } else initTimeInputsFromCurrent();
    } else initTimeInputsFromCurrent();
  }catch(e){ initTimeInputsFromCurrent(); }

  // "Go back to live" button 
  if(radarLiveBtn){
    radarLiveBtn.onclick = async () => {
      _exitArchiveMode();
      radarLiveBtn.disabled = true;
      radarLiveBtn.textContent = '↩ …';
      try {
        // Reload the live payload for the selected station
        const liveStation = stationSel.value || '';
        const _dealiasQs = dealiasEnabled ? '&dealias=1' : '';
        const resp = await fetch(
          `payload.json?station=${encodeURIComponent(liveStation)}${_dealiasQs}&_ts=${Date.now()}`,
          {cache:'no-store'}
        );
        if(!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if(data && data.rendered){
          APP.rendered = data.rendered;
          APP.errors   = data.errors || {};
          renderSettingsErrors();
          refreshAllMaps(false);
          renderAllLegends();
          // Sync date/time inputs to the freshly-loaded scan time
          initTimeInputsFromCurrent();
          // Refresh ESWD to today's live window
          if(eswdOverlayEnabled) _fetchAndApplyEswd(null);
        }
      } catch(err){
        if(warn) warn.textContent = `Live reload failed: ${err}`;
      } finally {
        radarLiveBtn.disabled = false;
        radarLiveBtn.textContent = '↩ Live';
        // Restart the auto-update countdown from the freshly-loaded scan time
        startAutoUpdateCountdown();
      }
    };
  }

  // "Go" button 
  radarGoBtn.onclick=()=>{
    // radarDateInp.value is YYYY-MM-DD (native date input)
    // Accept both YYYY-MM-DD (type=date) and any reasonable text fallback
    const rawDate = (radarDateInp.value||'').trim();
    const rawTime = (radarTimeInp.value||'').trim();
    // Primary: YYYY-MM-DD from type="date"
    let isoDate = null;
    const mYMD = /^(\d{4})-(\d{2})-(\d{2})$/.exec(rawDate);
    if(mYMD) isoDate = `${mYMD[1]}-${mYMD[2]}-${mYMD[3]}`;
    // Fallback: DD-MM-YYYY typed manually
    if(!isoDate){
      const mDMY = /^(\d{2})[.\-\/](\d{2})[.\-\/](\d{4})$/.exec(rawDate);
      if(mDMY) isoDate = `${mDMY[3]}-${mDMY[2]}-${mDMY[1]}`;
    }
    const mTime = /^(\d{1,2}):(\d{2})$/.exec(rawTime);
    if(!isoDate || !mTime){
      warn.textContent='Invalid date/time – use the date picker and HH:MM (UTC)';
      return;
    }
    const hh = String(mTime[1]).padStart(2,'0');
    const mm = String(mTime[2]).padStart(2,'0');
    const iso=`${isoDate}T${hh}:${mm}:00Z`;
    const st=APP.stations[stationSel.value]||{};
    if(((st.provider||'').toLowerCase())!=='dmi'){
      warn.textContent='Archive fetch only supported for DMI (Denmark) stations';
      return;
    }
    const now=Date.now(), req=new Date(iso).getTime();
    const maxAge=180*24*60*60*1000;
    if(!Number.isFinite(req)||(now-req)>maxAge){
      warn.textContent='Archive datetime must be within last 180 days (UTC)';
      return;
    }
    if(req>now){warn.textContent='Archive datetime cannot be in the future';return;}
    if(location.protocol==='file:'){
      warn.textContent='Archive fetch requires running: python custom.py --serve';
      return;
    }
    radarGoBtn.disabled=true;
    radarGoBtn.textContent='…';
    warn.textContent=`Fetching archive scan for ${iso.replace('T',' ').replace('Z',' UTC')} …`;
    // Check client cache first — instant if already visited
    const _cachedArchive = _scanCache.get(iso);
    if (_cachedArchive) {
      APP.rendered=_cachedArchive.rendered;
      APP.errors=_cachedArchive.errors||{};
      renderSettingsErrors();
      timeOffsetMin=0;
      refreshAllMaps(false);
      renderAllLegends();
      radarGoBtn.disabled=false; radarGoBtn.textContent='Go';
      _enterArchiveMode(iso);
      // Refresh ESWD for this archive moment if overlay is on
      if(eswdOverlayEnabled) _fetchAndApplyEswd(iso);
      return;
    }
    const archiveStation = stationSel.value || '';
    const _dealiasQs = dealiasEnabled ? '&dealias=1' : '';
    fetch(`payload.json?dmi_dt=${encodeURIComponent(iso)}&station=${encodeURIComponent(archiveStation)}${_dealiasQs}&_ts=${Date.now()}`,{cache:'no-store'})
      .then(r=>r.ok?r.json():r.text().then(t=>{throw new Error(`HTTP ${r.status}: ${t.slice(0,200)}`)}))
      .then(data=>{
        if(data&&data.rendered){
          _scanCache.set(iso, {rendered: data.rendered, errors: data.errors||{}});
          APP.rendered=data.rendered;
          APP.errors=data.errors||{};
          renderSettingsErrors();
          timeOffsetMin=0;
          refreshAllMaps(false);
          renderAllLegends();
          _enterArchiveMode(iso);
          // Pre-warm the cache for surrounding scans
          const reqMs = new Date(iso).getTime();
          _prefetchScan(new Date(reqMs - 15*60*1000).toISOString().replace('.000Z','Z'));
          _prefetchScan(new Date(reqMs - 30*60*1000).toISOString().replace('.000Z','Z'));
          // Refresh ESWD for this archive moment if overlay is on
          if(eswdOverlayEnabled) _fetchAndApplyEswd(iso);
        }else{
          warn.textContent='Archive response missing rendered data';
        }
      })
      .catch(err=>{warn.textContent=`Archive fetch failed: ${err}`;})
      .finally(()=>{radarGoBtn.disabled=false;radarGoBtn.textContent='Go';});
  };
  startAutoUpdateCountdown();
}
setupTimeControls();
</script></body></html>"""

    output_path.write_text(
        html.replace("__PAYLOAD__", json.dumps(payload)),
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMI/SMHI radar dashboard generator")
    parser.add_argument("--output", default="dmrat.html", help="Output HTML path")
    parser.add_argument(
        "--dmi-dt",
        default=None,
        help="UTC datetime for DMI archive (RFC3339, e.g. 2026-03-02T17:35:00Z)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously rebuild output on an interval (for file:// auto-refresh)",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=15,
        help="Rebuild interval in minutes (watch mode)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the dashboard on localhost with live rebuilds (enables archive Go button).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    output_path = Path(args.output)

    dmi_target: Optional[datetime] = None
    if args.dmi_dt:
        try:
            dmi_target = _parse_rfc3339(args.dmi_dt)
        except Exception as exc:
            raise SystemExit(f"Invalid --dmi-dt: {exc}") from exc

    def _archive_ok(dt: datetime) -> bool:
        now = datetime.now(timezone.utc)
        if dt > now:
            return False
        return (now - dt) <= timedelta(days=180)

    if dmi_target is not None and not _archive_ok(dmi_target):
        raise SystemExit("--dmi-dt must be within the last 180 days (UTC)")

    if args.serve:
        tmp_path = output_path.with_suffix(".served.html")

        # Background payload prefetch thread (also handles initial build)
        def _bg_prefetch() -> None:
            global _PAYLOAD_CACHE, _PAYLOAD_CACHE_TIME
            first = True
            while True:
                # If a force-refresh ran recently, skip this cycle so we don't
                # immediately clobber the freshly-built cache with a redundant build.
                with _NEXT_PREFETCH_LOCK:
                    skip_until = _NEXT_PREFETCH_TIME
                if not first and time.time() < skip_until:
                    print(f"[prefetch] skipping cycle — force-refresh was recent, "
                          f"next build in {max(0, skip_until - time.time()):.0f}s")
                    time.sleep(_PAYLOAD_CACHE_INTERVAL)
                    continue
                try:
                    with _BUILD_LOCK:
                        payload = build_payload(
                            dmi_target_utc=None,
                            providers=SERVE_PREFETCH_PROVIDERS,
                        )
                    body = json.dumps(payload).encode("utf-8")
                    with _PAYLOAD_CACHE_LOCK:
                        _PAYLOAD_CACHE = body
                        _PAYLOAD_CACHE_TIME = time.time()
                    if first:
                        with _BUILD_LOCK:
                            build_dashboard(
                                tmp_path,
                                _prebuilt_payload=payload,
                                dmi_target_utc=None,
                                providers=SERVE_PREFETCH_PROVIDERS,
                            )
                        first = False
                except Exception as exc:
                    print(f"[prefetch] failed: {exc}")
                time.sleep(_PAYLOAD_CACHE_INTERVAL)

        _prefetch_thread = threading.Thread(target=_bg_prefetch, daemon=True)
        _prefetch_thread.start()

        class _Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, ctype: str, body: bytes) -> None:
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
                    pass

            def do_GET(self) -> None:  # noqa: N802
                global _PAYLOAD_CACHE, _PAYLOAD_CACHE_TIME
                parsed = urlparse(self.path)

                qs = parse_qs(parsed.query or "")
                dt_str = (qs.get("dmi_dt") or [None])[0]
                station_param = (qs.get("station") or [None])[0]
                dealias_q = str((qs.get("dealias") or [None])[0] or "").strip().lower()
                dealias_on = dealias_q in ("1", "true", "yes", "on", "unravel")
                dt_val: Optional[datetime] = None
                if dt_str:
                    try:
                        dt_val = _parse_rfc3339(str(dt_str))
                    except Exception:
                        dt_val = None
                    if dt_val is not None and not _archive_ok(dt_val):
                        dt_val = None

                if parsed.path == "/payload.json":
                    try:
                        if dt_val is not None:
                            # Archive request - only rebuild the requested station(s) and
                            # merge back into the live cache so non-DMI panels stay current.
                            # Reading optional ?station=Name param lets us skip the other 4
                            # DMI stations → cuts build time from ~150 s to ~30 s.
                            s_filter: Optional[set] = {station_param} if station_param else None

                            with _PAYLOAD_CACHE_LOCK:
                                cached_body = _PAYLOAD_CACHE
                            if cached_body:
                                base = json.loads(cached_body)
                                with _BUILD_LOCK:
                                    dmi_only = build_payload(
                                        dmi_target_utc=dt_val,
                                        providers={"dmi"},
                                        station_filter=s_filter,
                                        dealias_unravel=dealias_on,
                                    )
                                base["rendered"].update(dmi_only["rendered"])
                                base["errors"].update(dmi_only["errors"])
                                payload = base
                            else:
                                # No cache yet - build DMI-only fallback for archive requests.
                                with _BUILD_LOCK:
                                    payload = build_payload(
                                        dmi_target_utc=dt_val,
                                        providers={"dmi"},
                                        station_filter=s_filter,
                                        dealias_unravel=dealias_on,
                                    )
                            body = json.dumps(payload).encode("utf-8")
                        else:
                            # Live request - serve from cache instantly (unless dealias requested)
                            with _PAYLOAD_CACHE_LOCK:
                                cached_body = _PAYLOAD_CACHE
                            if cached_body is None:
                                # Cache not ready yet - build synchronously once
                                try:
                                    with _BUILD_LOCK:
                                        payload = build_payload(
                                            dmi_target_utc=None,
                                            providers=SERVE_PREFETCH_PROVIDERS,
                                        )
                                    cached_body = json.dumps(payload).encode("utf-8")
                                    with _PAYLOAD_CACHE_LOCK:
                                        _PAYLOAD_CACHE = cached_body
                                        _PAYLOAD_CACHE_TIME = time.time()
                                except Exception as exc:
                                    print(f"[serve] synchronous build failed: {exc}")
                                    cached_body = json.dumps({"rendered": {}, "errors": {}, "stations": {}}).encode()

                            if dealias_on or station_param:
                                # Build only the requested station and/or affected providers
                                # and merge into the cached payload.
                                base = json.loads(cached_body)
                                try:
                                    with _BUILD_LOCK:
                                        if station_param:
                                            d = build_payload(
                                                dmi_target_utc=None,
                                                station_filter={station_param},
                                                dealias_unravel=dealias_on,
                                            )
                                        else:
                                            d = build_payload(
                                                dmi_target_utc=None,
                                                dealias_unravel=dealias_on,
                                            )
                                    base["rendered"].update(d.get("rendered") or {})
                                    if station_param and station_param not in (d.get("errors") or {}):
                                        try:
                                            base.get("errors", {}).pop(station_param, None)
                                        except Exception:
                                            pass
                                    base["errors"].update(d.get("errors") or {})
                                except Exception as _merge_exc:
                                    print(f"[serve] payload merge failed: {_merge_exc}")
                                    import traceback as _tb
                                    _tb.print_exc()
                                body = json.dumps(base).encode("utf-8")
                            else:
                                body = cached_body
                        # Gzip compress if the client supports it - halves transfer time
                        accept_enc = self.headers.get("Accept-Encoding", "")
                        if "gzip" in accept_enc:
                            body = gzip.compress(body, compresslevel=3)
                            try:
                                self.send_response(200)
                                self.send_header("Content-Type", "application/json; charset=utf-8")
                                self.send_header("Content-Encoding", "gzip")
                                self.send_header("Cache-Control", "no-store")
                                self.send_header("Access-Control-Allow-Origin", "*")
                                self.end_headers()
                                self.wfile.write(body)
                            except (BrokenPipeError, ConnectionAbortedError,
                                    ConnectionResetError, OSError): pass
                        else:
                            self._send(200, "application/json; charset=utf-8", body)
                    except Exception as exc:
                        import traceback as _tb; _tb.print_exc()
                        self._send(500, "text/plain; charset=utf-8",
                                   f"Build error: {exc}".encode("utf-8", errors="replace"))
                    return

                if parsed.path == "/force_refresh":
                    # Full cache-busting rebuild - bypasses _PAYLOAD_CACHE entirely,
                    # writes a fresh build back into the cache, and pushes the
                    # background prefetch thread's next cycle out by a full interval
                    # so it doesn't immediately clobber the result.
                    try:
                        print("[force_refresh] starting full rebuild…")
                        with _BUILD_LOCK:
                            payload = build_payload(
                                dmi_target_utc=None,
                                providers=SERVE_PREFETCH_PROVIDERS,
                            )
                        body = json.dumps(payload).encode("utf-8")
                        now = time.time()
                        with _PAYLOAD_CACHE_LOCK:
                            _PAYLOAD_CACHE = body
                            _PAYLOAD_CACHE_TIME = now
                        # Prevent bg thread from immediately rebuilding over our work
                        with _NEXT_PREFETCH_LOCK:
                            _NEXT_PREFETCH_TIME = now + _PAYLOAD_CACHE_INTERVAL
                        print("[force_refresh] done — cache updated, bg thread deferred")
                        accept_enc = self.headers.get("Accept-Encoding", "")
                        if "gzip" in accept_enc:
                            compressed = gzip.compress(body, compresslevel=3)
                            try:
                                self.send_response(200)
                                self.send_header("Content-Type", "application/json; charset=utf-8")
                                self.send_header("Content-Encoding", "gzip")
                                self.send_header("Cache-Control", "no-store")
                                self.send_header("Access-Control-Allow-Origin", "*")
                                self.end_headers()
                                self.wfile.write(compressed)
                            except (BrokenPipeError, ConnectionAbortedError,
                                    ConnectionResetError, OSError):
                                pass
                        else:
                            self._send(200, "application/json; charset=utf-8", body)
                    except Exception as exc:
                        import traceback as _tb; _tb.print_exc()
                        self._send(500, "text/plain; charset=utf-8",
                                   f"Force refresh error: {exc}".encode("utf-8", errors="replace"))
                    return

                if parsed.path == "/azshear_composite.json":
                    try:
                        n_str = (qs.get("n") or [None])[0]
                        n_scans = max(1, min(int(n_str or "1"), _AZSHEAR_HISTORY_MAXLEN))
                        stn = station_param or ""
                        with _AZSHEAR_HISTORY_LOCK:
                            hist = list((_AZSHEAR_HISTORY.get(stn) or [])[:n_scans])
                        if not hist:
                            body = json.dumps({"error": f"No azimuthal shear history for station {stn!r}"}).encode()
                            self._send(404, "application/json", body)
                            return
                        polars = [h["polar"] for h in hist]
                        meta   = hist[0]["meta"]
                        comp   = _composite_azshear_polar(polars)
                        template_sw = {
                            "rscale_km":    meta["rscale_km"],
                            "rstart_km":    meta["rstart_km"],
                            "max_range_km": meta["max_range_km"],
                            "nrays":        comp.shape[0],
                            "nbins":        comp.shape[1],
                            "elevation":    meta["elevation"],
                            "_polar":       comp,
                        }
                        overlay_data = _encode_sweep_for_overlay(template_sw)
                        if overlay_data is None:
                            body = json.dumps({"error": "Composite produced no valid data"}).encode()
                            self._send(204, "application/json", body)
                            return
                        result = {
                            "type": "polar",
                            "overlayData": overlay_data,
                            "sweeps": [],
                            "scansComposited": len(polars),
                        }
                        body = json.dumps(result).encode("utf-8")
                        accept_enc = self.headers.get("Accept-Encoding", "")
                        if "gzip" in accept_enc:
                            body = gzip.compress(body, compresslevel=3)
                            try:
                                self.send_response(200)
                                self.send_header("Content-Type", "application/json; charset=utf-8")
                                self.send_header("Content-Encoding", "gzip")
                                self.send_header("Cache-Control", "no-store")
                                self.send_header("Access-Control-Allow-Origin", "*")
                                self.end_headers()
                                self.wfile.write(body)
                            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError): pass
                        else:
                            self._send(200, "application/json; charset=utf-8", body)
                    except Exception as exc:
                        self._send(500, "text/plain", f"Composite error: {exc}".encode())
                    return

                # eswd_reports
                if parsed.path == "/icon_d2_inventory":
                    try:
                        force_flag = str((qs.get("force") or [""])[0] or "").strip().lower()
                        product_key = str((qs.get("product") or [ICON_D2_DEFAULT_PRODUCT])[0] or ICON_D2_DEFAULT_PRODUCT).strip()
                        if product_key not in ICON_D2_PRODUCTS:
                            product_key = ICON_D2_DEFAULT_PRODUCT
                        inventory = _icon_d2_get_inventory(
                            product_key=product_key,
                            force=force_flag in ("1", "true", "yes", "on")
                        )
                        body = json.dumps(_icon_d2_inventory_payload(inventory, product_key)).encode("utf-8")
                        accept_enc = self.headers.get("Accept-Encoding", "")
                        if "gzip" in accept_enc:
                            body = gzip.compress(body, compresslevel=3)
                            try:
                                self.send_response(200)
                                self.send_header("Content-Type", "application/json; charset=utf-8")
                                self.send_header("Content-Encoding", "gzip")
                                self.send_header("Cache-Control", "no-store")
                                self.send_header("Access-Control-Allow-Origin", "*")
                                self.end_headers()
                                self.wfile.write(body)
                            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
                                pass
                        else:
                            self._send(200, "application/json; charset=utf-8", body)
                    except Exception as exc:
                        self._send(
                            500,
                            "application/json; charset=utf-8",
                            json.dumps({"error": str(exc)}).encode("utf-8", errors="replace"),
                        )
                    return

                if parsed.path == "/icon_d2_frame":
                    try:
                        frame_key = str((qs.get("key") or [""])[0] or "").strip()
                        product_key = str((qs.get("product") or [ICON_D2_DEFAULT_PRODUCT])[0] or ICON_D2_DEFAULT_PRODUCT).strip()
                        if product_key not in ICON_D2_PRODUCTS:
                            product_key = ICON_D2_DEFAULT_PRODUCT
                        if not frame_key:
                            self._send(
                                400,
                                "application/json; charset=utf-8",
                                json.dumps({"error": "Missing 'key'"}).encode("utf-8"),
                            )
                            return
                        try:
                            payload = _icon_d2_get_frame_payload(frame_key, product_key=product_key)
                        except KeyError:
                            self._send(
                                404,
                                "application/json; charset=utf-8",
                                json.dumps({"error": f"Unknown frame key: {frame_key}"}).encode("utf-8"),
                            )
                            return
                        body = json.dumps(payload).encode("utf-8")
                        accept_enc = self.headers.get("Accept-Encoding", "")
                        if "gzip" in accept_enc:
                            body = gzip.compress(body, compresslevel=3)
                            try:
                                self.send_response(200)
                                self.send_header("Content-Type", "application/json; charset=utf-8")
                                self.send_header("Content-Encoding", "gzip")
                                self.send_header("Cache-Control", "no-store")
                                self.send_header("Access-Control-Allow-Origin", "*")
                                self.end_headers()
                                self.wfile.write(body)
                            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
                                pass
                        else:
                            self._send(200, "application/json; charset=utf-8", body)
                    except Exception as exc:
                        self._send(
                            500,
                            "application/json; charset=utf-8",
                            json.dumps({"error": str(exc)}).encode("utf-8", errors="replace"),
                        )
                    return

                if parsed.path == "/eswd_reports":
                    try:
                        eswd_dt_str = (qs.get("dt") or [None])[0]
                        eswd_ed: Optional[datetime] = None
                        eswd_sd: Optional[datetime] = None
                        if eswd_dt_str:
                            try:
                                eswd_ed = _parse_rfc3339(str(eswd_dt_str))
                                # Start of that same UTC day
                                eswd_sd = eswd_ed.replace(
                                    hour=0, minute=0, second=0, microsecond=0
                                )
                            except Exception:
                                eswd_ed = None
                                eswd_sd = None
                        geojson = _fetch_eswd_reports(sd_utc=eswd_sd, ed_utc=eswd_ed)
                        body = json.dumps(geojson).encode("utf-8")
                        accept_enc = self.headers.get("Accept-Encoding", "")
                        if "gzip" in accept_enc:
                            body = gzip.compress(body, compresslevel=3)
                            try:
                                self.send_response(200)
                                self.send_header("Content-Type", "application/json; charset=utf-8")
                                self.send_header("Content-Encoding", "gzip")
                                self.send_header("Cache-Control", "no-store")
                                self.send_header("Access-Control-Allow-Origin", "*")
                                self.end_headers()
                                self.wfile.write(body)
                            except (BrokenPipeError, ConnectionAbortedError,
                                    ConnectionResetError, OSError):
                                pass
                        else:
                            self._send(200, "application/json; charset=utf-8", body)
                    except Exception as exc:
                        import traceback as _tb; _tb.print_exc()
                        self._send(500, "application/json",
                                   json.dumps({"error": str(exc)}).encode())
                    return

                # Blue Marble XYZ tile endpoint: /bluemarble/{z}/{x}/{y}.png
                _bm_m = re.match(r"^/bluemarble/(\d+)/(\d+)/(\d+)\.png$", parsed.path)
                if _bm_m:
                    try:
                        _bm_z, _bm_x, _bm_y = int(_bm_m.group(1)), int(_bm_m.group(2)), int(_bm_m.group(3))
                        png_bytes = _serve_bluemarble_tile(_bm_z, _bm_x, _bm_y)
                        if png_bytes:
                            self._send(200, "image/png", png_bytes)
                        else:
                            self._send(503, "text/plain", b"Blue Marble GeoTIFF unavailable (rasterio missing or file not found)")
                    except Exception as _bm_exc:
                        self._send(500, "text/plain", f"Tile error: {_bm_exc}".encode())
                    return

                if parsed.path in ("/", "/index.html"):
                    try:
                        if dt_val is not None:
                            with _BUILD_LOCK:
                                build_dashboard(tmp_path, dmi_target_utc=dt_val)
                        body = tmp_path.read_bytes() if tmp_path.exists() else b"Building..."
                        self._send(200, "text/html; charset=utf-8", body)
                    except Exception as exc:
                        self._send(500, "text/plain; charset=utf-8",
                                   f"Build error: {exc}".encode("utf-8", errors="replace"))
                    return

                # Static image files – serve .png / .jpg from the script directory
                _img_m = re.match(r"^/([A-Za-z0-9_\-]+\.(png|jpg|jpeg|gif|svg))$", parsed.path)
                if _img_m:
                    _img_path = Path(__file__).resolve().parent / _img_m.group(1)
                    if _img_path.exists():
                        _ext = _img_m.group(2).lower()
                        _ctype = {
                            "png":  "image/png",
                            "jpg":  "image/jpeg",
                            "jpeg": "image/jpeg",
                            "gif":  "image/gif",
                            "svg":  "image/svg+xml",
                        }.get(_ext, "application/octet-stream")
                        self._send(200, _ctype, _img_path.read_bytes())
                        return

                self._send(404, "text/plain", b"Not found")

            def log_message(self, _format: str, *_args) -> None:
                return

            def do_OPTIONS(self) -> None:  # noqa: N802
                """Handle CORS pre-flight for POST requests."""
                try:
                    self.send_response(204)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.send_header("Access-Control-Allow-Headers", "Content-Type")
                    self.end_headers()
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
                    pass

            def do_POST(self) -> None:  # noqa: N802
                global _CUSTOM_STATIONS, _CUSTOM_RENDERED
                parsed = urlparse(self.path)

                # upload_radar 
                if parsed.path == "/upload_radar":
                    try:
                        ctype = self.headers.get("Content-Type", "")
                        if "multipart/form-data" not in ctype:
                            self._send(400, "application/json",
                                       json.dumps({"error": "Expected multipart/form-data"}).encode())
                            return

                        # Parse multipart/form-data without the removed `cgi` module 
                        # Build a minimal RFC 2822 message so email.parser can split it.
                        content_length = int(self.headers.get("Content-Length", 0))
                        raw_body = self.rfile.read(content_length)

                        # email.parser needs the Content-Type header attached to the body
                        from email import policy as _ep
                        from email.parser import BytesParser as _BP
                        fake_msg = (
                            b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw_body
                        )
                        parsed_msg = _BP(policy=_ep.compat32).parsebytes(fake_msg)

                        # Helper: get a named part's payload (returns bytes or None)
                        def _get_field(name: str):
                            for part in parsed_msg.get_payload():
                                cd = part.get("Content-Disposition", "")
                                if f'name="{name}"' in cd or f"name={name}" in cd:
                                    return part
                            return None

                        def _field_text(name: str) -> str:
                            p = _get_field(name)
                            if p is None:
                                return ""
                            raw = p.get_payload(decode=True)
                            return (raw or b"").decode("utf-8", errors="replace").strip()

                        def _field_bytes(name: str):
                            p = _get_field(name)
                            if p is None:
                                return None, None
                            cd = p.get("Content-Disposition", "")
                            fname = ""
                            for token in cd.split(";"):
                                token = token.strip()
                                if token.startswith("filename="):
                                    fname = token[9:].strip().strip('"')
                            return p.get_payload(decode=True), fname

                        name_field = _field_text("name")
                        if not name_field:
                            self._send(400, "application/json",
                                       json.dumps({"error": "Missing 'name' field"}).encode())
                            return

                        file_bytes, filename = _field_bytes("file")
                        if file_bytes is None:
                            self._send(400, "application/json",
                                       json.dumps({"error": "Missing 'file' field"}).encode())
                            return
                        filename = str(filename or "custom.h5")

                        def _fval(key: str) -> Optional[float]:
                            v = _field_text(key)
                            try: return float(v) if v else None
                            except ValueError: return None

                        user_lat      = _fval("lat")
                        user_lon      = _fval("lon")
                        user_range_km = _fval("range_km")

                        # Collect optional per-product extra files
                        # (file_V, file_PhiDP, file_RhoHV, file_W, file_ZDR)
                        _extra_product_files: List[Tuple[bytes, str]] = []
                        for _pfield in ("file_V", "file_PhiDP", "file_RhoHV",
                                        "file_W", "file_ZDR"):
                            _pb, _pname = _field_bytes(_pfield)
                            if _pb:
                                _extra_product_files.append((_pb, str(_pname or _pfield + ".h5")))

                        station_info = {
                            "provider":            "custom",
                            "file_bytes":          file_bytes,
                            "filename":            filename,
                            "lat":                 user_lat,
                            "lon":                 user_lon,
                            "range_km":            user_range_km or 200.0,
                            "_name":               name_field,
                            "extra_product_files": _extra_product_files,
                        }

                        with _BUILD_LOCK:
                            rendered = _render_station_products(station_info)

                        # Fill in lat/lon/range_km from rendered metadata if available
                        sm = rendered.get("stationMeta") or {}
                        lat      = sm.get("lat")      or user_lat      or station_info.get("lat")
                        lon      = sm.get("lon")      or user_lon      or station_info.get("lon")
                        range_km = sm.get("range_km") or user_range_km or station_info.get("range_km") or 200.0

                        full_entry = {
                            "provider":   "custom",
                            "file_bytes": file_bytes,
                            "filename":   filename,
                            "lat":        lat,
                            "lon":        lon,
                            "range_km":   range_km,
                        }
                        with _CUSTOM_LOCK:
                            _CUSTOM_STATIONS[name_field] = full_entry
                            _CUSTOM_RENDERED[name_field] = rendered

                        # Build the JSON-safe station entry (no file_bytes)
                        station_meta = {
                            "provider": "custom",
                            "lat":      lat,
                            "lon":      lon,
                            "range_km": range_km,
                        }
                        response = {
                            "name":        name_field,
                            "stationMeta": station_meta,
                            "rendered":    rendered,
                        }
                        body = json.dumps(response).encode("utf-8")
                        accept_enc = self.headers.get("Accept-Encoding", "")
                        if "gzip" in accept_enc:
                            body = gzip.compress(body, compresslevel=3)
                            try:
                                self.send_response(200)
                                self.send_header("Content-Type", "application/json; charset=utf-8")
                                self.send_header("Content-Encoding", "gzip")
                                self.send_header("Cache-Control", "no-store")
                                self.send_header("Access-Control-Allow-Origin", "*")
                                self.end_headers()
                                self.wfile.write(body)
                            except (BrokenPipeError, ConnectionAbortedError,
                                    ConnectionResetError, OSError):
                                pass
                        else:
                            self._send(200, "application/json; charset=utf-8", body)

                    except Exception as exc:
                        import traceback as _tb; _tb.print_exc()
                        self._send(500, "application/json",
                                   json.dumps({"error": str(exc)}).encode("utf-8", errors="replace"))
                    return

                # add_radar_from_url 
                # Accepts JSON: {name, url, lat?, lon?, range_km?}
                # Probes the URL, discovers the latest radar file, downloads
                # and renders it exactly like /upload_radar.
                if parsed.path == "/add_radar_from_url":
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body_bytes = self.rfile.read(length) if length else b""
                        try:
                            req_data = json.loads(body_bytes)
                        except Exception:
                            self._send(400, "application/json",
                                       json.dumps({"error": "Expected JSON body"}).encode())
                            return

                        name_field    = (req_data.get("name") or "").strip()
                        endpoint_url  = (req_data.get("url")  or "").strip()
                        if not name_field:
                            self._send(400, "application/json",
                                       json.dumps({"error": "Missing 'name'"}).encode())
                            return
                        if not endpoint_url:
                            self._send(400, "application/json",
                                       json.dumps({"error": "Missing 'url'"}).encode())
                            return

                        def _fval_opt(key: str) -> Optional[float]:
                            v = req_data.get(key)
                            try: return float(v) if v is not None else None
                            except (TypeError, ValueError): return None

                        user_lat      = _fval_opt("lat")
                        user_lon      = _fval_opt("lon")
                        user_range_km = _fval_opt("range_km")

                        #  Helpers
                        _TS_RE_URL = re.compile(r"(20\d{6,14})")

                        def _ts_from_name(nm: str) -> str:
                            m = _TS_RE_URL.search(nm)
                            return m.group(1) if m else "0"

                        def _looks_like_hdf5(nm: str) -> bool:
                            nl = nm.lower().split("?")[0]
                            if nl.endswith(("/", ".html", ".htm", ".txt", ".xml",
                                            ".json", ".css", ".js")):
                                return False
                            if nl.endswith((".hd5", ".h5", ".hdf5", "-hd5", "-h5",
                                            ".hd5.bz2", ".h5.bz2", ".hdf5.bz2",
                                            "-hd5.bz2", "-h5.bz2",
                                            ".hd5.gz",  ".h5.gz",  ".hdf5.gz",
                                            ".nc", ".nc4", ".pvol", ".raw",
                                            ".scn", ".scnx")):
                                return True
                            if re.search(r"hdf5?|sweep|20\d{10}", nl):
                                return True
                            return False

                        def _url_dir_list(url: str) -> List[str]:
                            """Like _http_dir_list but also handles OGC JSON items."""
                            if not url.endswith("/"):
                                url += "/"
                            # Try JSON first (OGC API)
                            try:
                                req = Request(url, headers={
                                    "Accept": "application/json, application/geo+json, */*",
                                    "User-Agent": "radr/1.0",
                                })
                                with urlopen(req, timeout=30) as r:
                                    raw = r.read()
                                pj = json.loads(raw)
                                items = (pj.get("features") or pj.get("items")
                                         or (pj if isinstance(pj, list) else []))
                                if items:
                                    # Return asset hrefs as the "listing"
                                    hrefs = []
                                    def _item_ts2(it: dict) -> str:
                                        props = it.get("properties") or it
                                        for k in ("datetime", "timestamp", "time",
                                                  "observed", "phenomenonTime",
                                                  "resultTime"):
                                            v = props.get(k) or ""
                                            if v: return str(v)
                                        return ""
                                    items_s = sorted(items, key=_item_ts2, reverse=True)
                                    for item in items_s[:20]:
                                        links = item.get("links") or item.get("assets") or {}
                                        if isinstance(links, dict):
                                            links = list(links.values())
                                        for lnk in links:
                                            href = (lnk.get("href") or lnk.get("url")
                                                    if isinstance(lnk, dict) else str(lnk))
                                            href = (href or "").strip()
                                            if href:
                                                hrefs.append(href)
                                    return hrefs
                            except Exception:
                                pass
                            # HTML directory listing
                            req = Request(url, headers={"User-Agent": "radr/1.0"})
                            with urlopen(req, timeout=30) as r:
                                html = r.read().decode("utf-8", errors="ignore")
                            return [
                                h.strip()
                                for h in DWD_HREF_RE.findall(html)
                                if h and h not in ("../", "./") and not h.startswith("?")
                            ]

                        def _find_latest_in_tree(base_url: str,
                                                  entries: List[str],
                                                  max_depth: int = 5
                                                  ) -> Optional[str]:
                            """
                            Recursively walk directory tree (newest dir first at each
                            level) looking for HDF5/NetCDF files.  Returns absolute URL
                            of the most recent file, or None.
                            """
                            cur_root, cur_entries = base_url, entries
                            for _ in range(max_depth):
                                # Files at this level
                                files = [e for e in cur_entries
                                         if not e.endswith("/") and _looks_like_hdf5(e)]
                                if files:
                                    # Pick newest by embedded timestamp
                                    files.sort(key=lambda e: _ts_from_name(
                                        e.split("/")[-1]), reverse=True)
                                    return urljoin(cur_root, files[0])

                                # Subdirectories: descend into the newest one
                                dirs = sorted(
                                    [e for e in cur_entries
                                     if e.endswith("/") and e not in ("../", "./")],
                                    reverse=True,  # newest timestamp dir first
                                )
                                if not dirs:
                                    break
                                cur_root = urljoin(cur_root, dirs[0])
                                try:
                                    cur_entries = _url_dir_list(cur_root)
                                except Exception:
                                    break
                            return None

                        # Probe the endpoint
                        file_url: Optional[str] = None
                        filename: str = "custom.h5"
                        probe_error: str = ""

                        try:
                            entries = _url_dir_list(endpoint_url)
                            if not endpoint_url.endswith("/"):
                                endpoint_url += "/"
                            file_url = _find_latest_in_tree(endpoint_url, entries)
                            if file_url:
                                filename = file_url.split("/")[-1].split("?")[0] or "custom.h5"
                        except Exception as exc:
                            probe_error = str(exc)

                        if not file_url:
                            msg = (f"No downloadable HDF5/NetCDF radar file found at "
                                   f"{endpoint_url!r}.")
                            if probe_error:
                                msg += f" Probe error: {probe_error}"
                            self._send(404, "application/json",
                                       json.dumps({"error": msg}).encode())
                            return

                        # Download
                        try:
                            req = Request(file_url, headers={"User-Agent": "radr/1.0"})
                            with urlopen(req, timeout=120) as r:
                                file_bytes = r.read()
                        except Exception as exc:
                            self._send(502, "application/json",
                                       json.dumps({"error": f"Download failed: {exc}"
                                                   }).encode())
                            return

                        # Render
                        station_info = {
                            "provider":            "custom",
                            "file_bytes":          file_bytes,
                            "filename":            filename,
                            "lat":                 user_lat,
                            "lon":                 user_lon,
                            "range_km":            user_range_km or 200.0,
                            "_name":               name_field,
                            "extra_product_files": [],
                        }
                        with _BUILD_LOCK:
                            rendered = _render_station_products(station_info)

                        sm       = rendered.get("stationMeta") or {}
                        lat      = sm.get("lat")      or user_lat      or station_info.get("lat")
                        lon      = sm.get("lon")      or user_lon      or station_info.get("lon")
                        range_km = (sm.get("range_km") or user_range_km
                                    or station_info.get("range_km") or 200.0)

                        full_entry = {
                            "provider":   "custom",
                            "file_bytes": file_bytes,
                            "filename":   filename,
                            "lat":        lat,
                            "lon":        lon,
                            "range_km":   range_km,
                        }
                        with _CUSTOM_LOCK:
                            _CUSTOM_STATIONS[name_field] = full_entry
                            _CUSTOM_RENDERED[name_field] = rendered

                        station_meta = {
                            "provider": "custom",
                            "lat":      lat,
                            "lon":      lon,
                            "range_km": range_km,
                        }
                        response = {
                            "name":        name_field,
                            "stationMeta": station_meta,
                            "rendered":    rendered,
                        }
                        resp_body = json.dumps(response).encode("utf-8")
                        accept_enc = self.headers.get("Accept-Encoding", "")
                        if "gzip" in accept_enc:
                            resp_body = gzip.compress(resp_body, compresslevel=3)
                            try:
                                self.send_response(200)
                                self.send_header("Content-Type",
                                                 "application/json; charset=utf-8")
                                self.send_header("Content-Encoding", "gzip")
                                self.send_header("Cache-Control", "no-store")
                                self.send_header("Access-Control-Allow-Origin", "*")
                                self.end_headers()
                                self.wfile.write(resp_body)
                            except (BrokenPipeError, ConnectionAbortedError,
                                    ConnectionResetError, OSError):
                                pass
                        else:
                            self._send(200, "application/json; charset=utf-8", resp_body)

                    except Exception as exc:
                        import traceback as _tb; _tb.print_exc()
                        self._send(500, "application/json",
                                   json.dumps({"error": str(exc)}).encode(
                                       "utf-8", errors="replace"))
                    return

                # remove_custom_radar
                if parsed.path == "/remove_custom_radar":
                    try:
                        qs = parse_qs(urlparse(self.path).query)
                        name = (qs.get("name") or [None])[0]
                        # Also try reading from POST body
                        if not name:
                            length = int(self.headers.get("Content-Length", 0))
                            body_bytes = self.rfile.read(length) if length else b""
                            try:
                                body_data = json.loads(body_bytes)
                                name = body_data.get("name")
                            except Exception:
                                body_qs = parse_qs(body_bytes.decode("utf-8", errors="ignore"))
                                name = (body_qs.get("name") or [None])[0]
                        if not name:
                            self._send(400, "application/json",
                                       json.dumps({"error": "Missing 'name'"}).encode())
                            return
                        removed = False
                        with _CUSTOM_LOCK:
                            if name in _CUSTOM_STATIONS:
                                del _CUSTOM_STATIONS[name]
                                removed = True
                            if name in _CUSTOM_RENDERED:
                                del _CUSTOM_RENDERED[name]
                        self._send(200, "application/json",
                                   json.dumps({"removed": removed, "name": name}).encode())
                    except Exception as exc:
                        self._send(500, "application/json",
                                   json.dumps({"error": str(exc)}).encode())
                    return

                self._send(404, "text/plain", b"Not found")

        server = ThreadingHTTPServer((args.host, args.port), _Handler)
        print(f"Serving radar dashboard at http://{args.host}:{args.port}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif args.watch:
        interval_s = max(5, int(args.interval_minutes) * 60)
        while True:
            with _BUILD_LOCK:
                build_dashboard(output_path, dmi_target_utc=dmi_target)
            time.sleep(interval_s)
    else:
        with _BUILD_LOCK:
            build_dashboard(output_path, dmi_target_utc=dmi_target)