"""
MSEF Pipeline — Stage 2: Source Detection & Aperture Photometry
================================================================
Takes a folder of calibrated/denoised images (output of Stage 1) and:

  1. Builds a median reference frame from a sample of frames
  2. Detects stars on the reference using DAOStarFinder
  3. Re-centres each star per-frame via centroid fitting
  4. Measures aperture flux + annular sky background for every star on
     every frame
  5. Computes per-star metrics: median flux, std, MAD, SNR, instrumental
     magnitude, sharpness, saturation flag
  6. Optionally runs Box Least Squares (BLS) transit search per star
  7. Writes per-star light-curve CSVs and a master_catalog.csv
  8. Saves diagnostic plots (source map, RMS scatter map, top-variable
     light curves)

"""

from __future__ import annotations

import glob
import math
import os
import shutil
import threading
import time
import traceback
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageOps, ImageTk

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time as AstropyTime
from photutils.detection import DAOStarFinder
from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    aperture_photometry,
)
from photutils.centroids import centroid_com
try:
    from astropy.timeseries import BoxLeastSquares
    HAS_BLS = True
except Exception:
    HAS_BLS = False
    warnings.warn("astropy.timeseries.BoxLeastSquares not available — BLS disabled.")

# Constants
APERTURE_RADIUS   = 5.0      # px  — star aperture radius
SKY_INNER         = 9.0      # px  — sky annulus inner edge
SKY_OUTER         = 14.0     # px  — sky annulus outer edge
MIN_SNR           = 5.0      # sigma threshold for source detection
MAX_SOURCES       = 800      # cap on number of detected stars
FWHM_GUESS        = 3.5      # px  — initial FWHM guess for DAOStarFinder
RE_CENTER         = True     # refine centroids per frame?
RECENTER_BOX      = 7        # px  — half-width of centroid box
SATURATION_LEVEL  = 0.97     # fraction of max dynamic range → flagged saturated

RUN_BLS           = True
BLS_MIN_PERIOD    = 0.1      # days (or frame-index units if no time header)
BLS_MAX_PERIOD    = 15.0
BLS_N_PERIODS     = 800
BLS_DURATIONS     = [0.03, 0.05, 0.08, 0.12]  # trial transit durations

# Priority-ordered FITS time keywords (spec §Robust Time Extraction)
TIME_HEADER_KEYS  = [
    "DATE-OBS",   # ISO date/time string (primary)
    "MJD-OBS",    # Modified Julian Date float (days)
    "JD",         # Julian Date float → converted to MJD
    "JD-OBS",     # Alternate JD keyword
    "BJD-OBS",    # Barycentric JD
    "HJD-OBS",    # Heliocentric JD
    "TIME-OBS",   # Time string — combined with DATE-OBS when needed
    "UTSTART",    # Alternate separate time keyword
    "UTC-OBS",    # UTC time string
]
# All formats accepted for preview; FITS_ONLY used during photometry
IMAGE_EXTENSIONS     = ("*.fits", "*.fit", "*.fts",
                         "*.png",  "*.jpg", "*.jpeg",
                         "*.tif",  "*.tiff")
FITS_ONLY_EXTENSIONS = ("*.fits", "*.fit", "*.fts")

PREVIEW_SIZE = 480   # canvas pixels (square)

def list_image_files(folder: str) -> list[str]:
    """Return a sorted, deduplicated list of supported image files."""
    seen: set[str] = set()
    files: list[str] = []
    for ext in IMAGE_EXTENSIONS:
        for f in glob.glob(os.path.join(folder, ext)):
            real = os.path.realpath(f)
            if real not in seen:
                seen.add(real)
                files.append(f)
        for f in glob.glob(os.path.join(folder, ext.upper())):
            real = os.path.realpath(f)
            if real not in seen:
                seen.add(real)
                files.append(f)
    return sorted(files)


def load_image_and_time(path: str,
                        fits_only: bool = False
                        ) -> tuple[np.ndarray, float | None, str, dict]:
    """
    Load a single image and try to extract an observation timestamp.

    Parameters
    ----------
    path      : file path
    fits_only : if True, raise ValueError for non-FITS files so the
                pipeline can skip them gracefully during photometry

    Returns
    -------
    data        : float64 2-D array, normalised to [0, 1]
    t           : float MJD timestamp in days, or None
    time_source : provenance label e.g. "DATE-OBS", "MJD-OBS", ""
    header      : dict-like (FITS header or empty dict)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".fits", ".fit", ".fts"):
        # ignore_missing_simple + output_verify="fix" tolerate non-standard
        # header cards written by telescope software (e.g. values formatted as
        # '+8.000e+001' with a leading '+' on the exponent, which is invalid FITS).
        # hdul.verify("silentfix") repairs remaining card issues in-place.
        # header.copy() is required because lazy FITS header properties become
        # inaccessible after the 'with' block closes the file handle.
        with fits.open(path, memmap=False,
                       ignore_missing_simple=True,
                       output_verify="fix") as hdul:
            hdul.verify("silentfix")
            raw    = hdul[0].data
            header = hdul[0].header.copy()   # copy before hdul closes
        if raw is None:
            raise ValueError(f"No image data in primary HDU of {path}")
        arr = np.nan_to_num(np.asarray(raw, dtype=float))
        while arr.ndim > 2:
            arr = arr[0]
        t, time_source = _extract_time(header)

    else:
        if fits_only:
            raise ValueError(
                f"Non-FITS file skipped during photometry: {os.path.basename(path)}")
        pil = ImageOps.exif_transpose(Image.open(path)).convert("L")
        arr = np.asarray(pil, dtype=float)
        header = {}
        t, time_source = None, ""

    # normalise to [0, 1]
    lo, hi = arr.min(), arr.max()
    if hi - lo > 0:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    return arr, t, time_source, header


# Provenance labels stored per-frame in CSV
TIME_SOURCE_FALLBACK = "frame_index"  # no header time found


def _extract_time(header) -> tuple[float | None, str]:
    """
    Extract observation timestamp from a FITS header using astropy Time.

    Returns
    -------
    (mjd, source_label)  — MJD float and the keyword that provided it,
    or (None, "")        — if no time information is found.

    Priority:
      1. DATE-OBS [+ TIME-OBS / UTSTART if date-only]
      2. MJD-OBS
      3. JD / JD-OBS  → converted via astropy
      4. BJD-OBS / HJD-OBS
      5. DATE + UTC-OBS / UTSTART  (separate date + time keywords)
    """
    # ── 1. DATE-OBS (± TIME-OBS / UTSTART) ───────────────────────────────────
    date_obs = header.get("DATE-OBS")
    if date_obs:
        date_str = str(date_obs).strip()
        # If no time component present, try to append TIME-OBS / UTSTART
        if "T" not in date_str and ":" not in date_str:
            for tkey in ("TIME-OBS", "UTSTART", "UTC-OBS"):
                t_part = header.get(tkey)
                if t_part:
                    date_str = f"{date_str}T{str(t_part).strip()}"
                    break
        date_str = date_str.replace(" ", "T").rstrip("Z")
        try:
            return float(AstropyTime(date_str, format="isot", scale="utc").mjd), "DATE-OBS"
        except Exception:
            try:
                return float(AstropyTime(date_str, scale="utc").mjd), "DATE-OBS"
            except Exception:
                pass

    # ── 2. MJD-OBS ───────────────────────────────────────────────────────────
    if "MJD-OBS" in header:
        try:
            return float(AstropyTime(float(header["MJD-OBS"]),
                                     format="mjd", scale="utc").mjd), "MJD-OBS"
        except Exception:
            pass

    # ── 3. JD / JD-OBS ───────────────────────────────────────────────────────
    for jd_key in ("JD", "JD-OBS"):
        if jd_key in header:
            try:
                return float(AstropyTime(float(header[jd_key]),
                                         format="jd", scale="utc").mjd), jd_key
            except Exception:
                pass

    # ── 4. BJD-OBS / HJD-OBS ─────────────────────────────────────────────────
    for bjd_key in ("BJD-OBS", "HJD-OBS"):
        if bjd_key in header:
            try:
                return float(AstropyTime(float(header[bjd_key]),
                                         format="jd", scale="utc").mjd), bjd_key
            except Exception:
                pass

    # ── 5. Separate DATE + UTC-OBS / UTSTART ─────────────────────────────────
    t_bare   = header.get("UTC-OBS") or header.get("UTSTART")
    date_bare = header.get("DATE")
    if t_bare and date_bare:
        try:
            combined = f"{str(date_bare).strip()}T{str(t_bare).strip()}".replace(" ", "T")
            return float(AstropyTime(combined, format="isot", scale="utc").mjd), "DATE+UTC-OBS"
        except Exception:
            pass

    return None, ""

def detect_sources(image: np.ndarray,
                   fwhm: float = FWHM_GUESS,
                   threshold_sigma: float = MIN_SNR,
                   max_sources: int | None = MAX_SOURCES
                   ) -> tuple[object, np.ndarray]:
    """Run DAOStarFinder on *image*, return (source table, Nx2 xy array)."""
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    if std <= 0:
        return None, np.zeros((0, 2))
    finder  = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std,
                            brightest=max_sources)
    sources = finder(image - median)
    if sources is None or len(sources) == 0:
        return None, np.zeros((0, 2))
    sources.sort("flux")
    sources.reverse()
    if max_sources:
        sources = sources[:max_sources]
    xy = np.column_stack([sources["xcentroid"], sources["ycentroid"]])
    return sources, xy


def recenter_positions(image: np.ndarray,
                       positions: np.ndarray,
                       box: int = RECENTER_BOX) -> np.ndarray:
    """Refine each (x, y) position using centroid_com on a small cutout."""
    H, W   = image.shape
    newpos = []
    for (x0, y0) in positions:
        xi, yi = int(round(x0)), int(round(y0))
        x1, x2 = max(0, xi - box), min(W, xi + box + 1)
        y1, y2 = max(0, yi - box), min(H, yi + box + 1)
        cut    = image[y1:y2, x1:x2]
        if cut.size == 0 or cut.sum() == 0:
            newpos.append((x0, y0))
            continue
        try:
            cy, cx = centroid_com(cut)
            newpos.append((x1 + cx, y1 + cy))
        except Exception:
            newpos.append((x0, y0))
    return np.asarray(newpos)


def measure_apertures(image: np.ndarray,
                      positions: np.ndarray,
                      ap_r: float  = APERTURE_RADIUS,
                      sky_in: float = SKY_INNER,
                      sky_out: float = SKY_OUTER
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Aperture photometry with proper annular sky subtraction.

    Returns
    -------
    net_flux : background-subtracted flux per source
    sky_mean : mean sky counts per pixel per source
    """
    if len(positions) == 0:
        return np.array([]), np.array([])

    star_ap  = CircularAperture(positions, r=ap_r)
    sky_ann  = CircularAnnulus(positions, r_in=sky_in, r_out=sky_out)

    star_tbl = aperture_photometry(image, star_ap)
    sky_tbl  = aperture_photometry(image, sky_ann)

    raw_flux  = np.asarray(star_tbl["aperture_sum"], dtype=float)
    sky_sum   = np.asarray(sky_tbl["aperture_sum"],  dtype=float)
    sky_area  = sky_ann.area                        # scalar (same for all)
    sky_mean  = sky_sum / sky_area                  # counts / pixel
    ap_area   = star_ap.area
    net_flux  = raw_flux - sky_mean * ap_area

    return net_flux, sky_mean


def compute_snr(flux: np.ndarray, sky_mean: np.ndarray,
                ap_r: float = APERTURE_RADIUS,
                gain: float = 1.0) -> np.ndarray:
    """Poisson-noise SNR estimate: signal / sqrt(signal + n_pix * sky)."""
    n_pix   = math.pi * ap_r ** 2
    signal  = np.maximum(flux, 0.0) * gain
    noise   = np.sqrt(signal + n_pix * np.maximum(sky_mean, 0.0) * gain)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(noise > 0, signal / noise, 0.0)
    return snr


def instrumental_magnitude(flux: np.ndarray,
                            zero_point: float = 25.0) -> np.ndarray:
    """Convert flux to instrumental magnitude: mag = ZP - 2.5 * log10(flux)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(flux > 0, zero_point - 2.5 * np.log10(flux), np.nan)
    return mag


def ensure_numeric(values) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)
    flat = pd.to_numeric(pd.Series(arr.ravel()), errors="coerce").to_numpy(float)
    return flat.reshape(arr.shape)


# Minimum frames needed for BLS — intentionally low so small datasets work.
# BLS quality is governed by time SPAN, not raw count.
BLS_MIN_POINTS = 8

def run_bls(times: np.ndarray,
            norm_flux: np.ndarray) -> tuple[float, float, float, float, str]:
    """
    Run BLS transit search over multiple trial durations.

    Returns
    -------
    (best_period, best_power, best_depth, best_snr, reason)

    reason is "" on success or a short human-readable string explaining
    why BLS returned NaN (logged by the caller so failures are visible).

    Stage 4 requires bls_depth and bls_snr to gate transit candidates —
    returning all four values here avoids a silent catalog gap.
    """
    nan4 = (np.nan, np.nan, np.nan, np.nan)

    if not HAS_BLS:
        return (*nan4, "BLS not available (astropy.timeseries missing)")

    mask = np.isfinite(times) & np.isfinite(norm_flux)
    n_ok = int(mask.sum())
    if n_ok < BLS_MIN_POINTS:
        return (*nan4, f"too few valid points ({n_ok} < {BLS_MIN_POINTS})")

    try:
        t = times[mask]
        y = norm_flux[mask] - 1.0          # BLS expects flux centred on 0

        tspan = float(t.max() - t.min())
        if tspan <= 0:
            return (*nan4, "zero time span — all frames have identical timestamp")

        min_p = max(BLS_MIN_PERIOD, tspan / 5_000.0)
        max_p = min(BLS_MAX_PERIOD, tspan * 0.9)
        if min_p >= max_p:
            return (*nan4,
                    f"period range degenerate: min={min_p:.4f} >= max={max_p:.4f} "
                    f"(tspan={tspan:.4f} d)")

        periods   = np.linspace(min_p, max_p, BLS_N_PERIODS)
        bls_model = BoxLeastSquares(t, y)

        best_period = np.nan
        best_power  = -np.inf
        best_depth  = np.nan
        best_snr    = np.nan

        for dur_frac in BLS_DURATIONS:
            # BLS duration is a fraction of the shortest trial period
            dur = dur_frac * min_p
            res = bls_model.power(periods, dur)
            idx = int(np.nanargmax(res.power))
            if res.power[idx] > best_power:
                best_power  = float(res.power[idx])
                best_period = float(res.period[idx])
                best_depth  = float(res.depth[idx])
                best_snr    = float(res.depth_snr[idx])

        return best_period, best_power, best_depth, best_snr, ""

    except Exception as exc:
        return (*nan4, f"exception: {exc}")

def stretch(arr: np.ndarray, mode: str = "sqrt") -> np.ndarray:
    """Stretch a [0,1] float image for display."""
    arr = np.clip(arr, 0.0, 1.0)
    if mode == "sqrt":
        return np.sqrt(arr)
    if mode == "log":
        return np.log1p(arr * 999.0) / np.log1p(999.0)
    if mode == "asinh":
        a = 0.05
        return np.arcsinh(arr / a) / np.arcsinh(1.0 / a)
    return arr   # linear


def array_to_tkimage(arr: np.ndarray, size: int = PREVIEW_SIZE,
                     stretch_mode: str = "sqrt") -> ImageTk.PhotoImage:
    """Convert 2-D float array → resized Tk PhotoImage."""
    s   = stretch(arr, stretch_mode)
    pil = Image.fromarray((s * 255).astype(np.uint8))
    pil.thumbnail((size, size), Image.LANCZOS)
    return ImageTk.PhotoImage(pil)

class Stage2GUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("MSEF — Stage 2  ·  Source Detection & Photometry")
        root.minsize(1100, 720)

        self.selected_folder: str | None = None
        self.ref_xy:   np.ndarray | None = None
        self.ref_image: np.ndarray | None = None
        self._tk_preview                  = None   # prevent GC
        self._worker_thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._output_folder: str | None   = None   # set at run time

        self._build_ui()
        self.log("Stage 2 ready.  Select an image folder, then click  Run Stage 2.")

    def _build_ui(self):
        root = self.root

        tk.Label(root,
                 text="MSEF Pipeline  ·  Stage 2: Source Detection & Photometry",
                 font=("Arial", 14, "bold")).pack(pady=(10, 4))

        btn_row = tk.Frame(root)
        btn_row.pack(fill="x", padx=10, pady=4)

        btn_cfg = dict(width=20, pady=4, relief="groove")
        tk.Button(btn_row, text="📁  Select Image Folder",
                  command=self.select_folder, **btn_cfg).pack(side="left", padx=3)
        tk.Button(btn_row, text="🔍  Preview Single Image",
                  command=self.preview_single, **btn_cfg).pack(side="left", padx=3)
        self.run_btn = tk.Button(btn_row, text="▶  Run Stage 2",
                                 command=self.run_pipeline_async,
                                 width=16, pady=4, relief="groove",
                                 bg="#2a7a2a", fg="white",
                                 activebackground="#1f5c1f")
        self.run_btn.pack(side="left", padx=3)
        self.cancel_btn = tk.Button(btn_row, text="⛔  Cancel",
                                    command=self.request_cancel,
                                    width=10, pady=4, relief="groove",
                                    state="disabled")
        self.cancel_btn.pack(side="left", padx=3)
        tk.Button(btn_row, text="💾  Export Catalog",
                  command=self.export_catalog, **btn_cfg).pack(side="left", padx=3)
        tk.Button(btn_row, text="📊  View Plots",
                  command=self.view_plots, **btn_cfg).pack(side="left", padx=3)
        tk.Button(btn_row, text="✖  Quit",
                  command=root.quit, width=8, pady=4,
                  relief="groove").pack(side="right", padx=3)

        # ── options strip ──
        opts = tk.Frame(root, relief="groove", bd=1)
        opts.pack(fill="x", padx=10, pady=(0, 4))

        def _lbl(txt, col):
            tk.Label(opts, text=txt, font=("Arial", 8)).grid(
                row=0, column=col, padx=(8, 2), pady=3, sticky="e")

        def _entry(default, col, width=6):
            e = ttk.Entry(opts, width=width)
            e.insert(0, str(default))
            e.grid(row=0, column=col, padx=(0, 4))
            return e

        _lbl("Min SNR:",      0);  self.e_snr     = _entry(MIN_SNR,        1)
        _lbl("Max sources:",  2);  self.e_maxsrc  = _entry(MAX_SOURCES,    3)
        _lbl("FWHM (px):",    4);  self.e_fwhm    = _entry(FWHM_GUESS,     5)
        _lbl("Aper r (px):",  6);  self.e_apr     = _entry(APERTURE_RADIUS,7)
        _lbl("Sky in/out:",   8)
        self.e_skyin  = _entry(SKY_INNER,  9,  5)
        tk.Label(opts, text="/").grid(row=0, column=10)
        self.e_skyout = _entry(SKY_OUTER, 11,  5)

        self.v_recenter = tk.BooleanVar(value=RE_CENTER)
        ttk.Checkbutton(opts, text="Re-centre", variable=self.v_recenter).grid(
            row=0, column=12, padx=6)
        self.v_bls = tk.BooleanVar(value=RUN_BLS and HAS_BLS)
        ttk.Checkbutton(opts, text="Run BLS", variable=self.v_bls).grid(
            row=0, column=13, padx=6)

        _lbl("Stretch:", 14)
        self.v_stretch = tk.StringVar(value="sqrt")
        ttk.Combobox(opts, textvariable=self.v_stretch,
                     values=["linear", "sqrt", "log", "asinh"],
                     width=7, state="readonly").grid(row=0, column=15, padx=4)

        # ── progress ──
        prog_frame = tk.Frame(root)
        prog_frame.pack(fill="x", padx=10, pady=(2, 0))
        self.progress = ttk.Progressbar(prog_frame, orient="horizontal",
                                        mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True)
        self.pct_lbl = tk.Label(prog_frame, text="", width=10, font=("Arial", 8))
        self.pct_lbl.pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Idle")
        tk.Label(root, textvariable=self.status_var,
                 font=("Arial", 9), fg="#444").pack(anchor="w", padx=12)

        mid = tk.Frame(root)
        mid.pack(fill="both", expand=True, padx=10, pady=6)

        # preview
        pf = ttk.LabelFrame(mid, text="Reference Frame  (detected sources in red)")
        pf.pack(side="left", fill="both", expand=False, padx=(0, 6))
        self.canvas = tk.Canvas(pf, width=PREVIEW_SIZE, height=PREVIEW_SIZE,
                                bg="#0a0a0a", cursor="crosshair")
        self.canvas.pack(padx=4, pady=4)
        # star-count label below canvas
        self.src_count_var = tk.StringVar(value="No sources yet")
        tk.Label(pf, textvariable=self.src_count_var,
                 font=("Arial", 8), fg="#555").pack(pady=(0, 4))

        # log
        lf = ttk.LabelFrame(mid, text="Log")
        lf.pack(side="left", fill="both", expand=True)
        self.log_box = tk.Text(lf, wrap="word", font=("Courier", 8),
                               bg="#1a1a2e", fg="#e0e0e0",
                               insertbackground="white")
        sb = ttk.Scrollbar(lf, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

        root.protocol("WM_DELETE_WINDOW", root.quit)

    def log(self, msg: str):
        """Append a timestamped line to the log box (thread-safe)."""
        ts   = time.strftime("%H:%M:%S")
        line = f"[{ts}]  {msg}\n"
        self.root.after(0, lambda: (
            self.log_box.insert("end", line),
            self.log_box.see("end"),
            self.status_var.set(msg[:120]),
        ))

    def _set_progress(self, pct: float, label: str = ""):
        self.root.after(0, lambda: (
            self.progress.config(value=pct),
            self.pct_lbl.config(text=label),
        ))

    def _set_buttons(self, running: bool):
        state_run    = "disabled" if running else "normal"
        state_cancel = "normal"   if running else "disabled"
        self.root.after(0, lambda: (
            self.run_btn.config(state=state_run),
            self.cancel_btn.config(state=state_cancel),
        ))

    def update_canvas(self, image: np.ndarray,
                      overlay_xy: np.ndarray | None = None):
        """
        Render *image* on the preview canvas (fully thread-safe).

        All PIL work (stretch, resize, circle drawing) is done here — in
        whatever thread called us — so _draw() only ever does ONE Tk call.
        That prevents both the per-oval Tk stall and the v_stretch cross-
        thread read.
        """
        from PIL import ImageDraw

        # Read stretch mode safely: try the Tk variable (works on main thread),
        # fall back to "sqrt" if called from a worker thread.
        try:
            smode = self.v_stretch.get()
        except Exception:
            smode = "sqrt"

        H, W  = image.shape
        scale = min(PREVIEW_SIZE / W, PREVIEW_SIZE / H)
        disp_w = max(1, int(W * scale))
        disp_h = max(1, int(H * scale))

        # Stretch + resize entirely in PIL/NumPy (no Tk)
        s = stretch(image, smode)
        pil = Image.fromarray((np.clip(s, 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((disp_w, disp_h), Image.LANCZOS)

        # Draw all source markers into the PIL image (fast, no Tk involvement)
        n = 0
        if overlay_xy is not None and len(overlay_xy):
            n    = len(overlay_xy)
            rgba = pil.convert("RGBA")
            draw = ImageDraw.Draw(rgba)
            r    = 4
            col  = (255, 68, 68, 220)
            for (x, y) in overlay_xy:
                cx, cy = int(x * scale), int(y * scale)
                if 0 <= cx < disp_w and 0 <= cy < disp_h:
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                                 outline=col, width=1)
            del draw
            final_pil = rgba.convert("RGB")
        else:
            final_pil = pil.convert("RGB")

        tkimg = ImageTk.PhotoImage(final_pil)
        n_captured = n  # capture for closure

        # Single Tk call — safe from any thread via .after()
        def _draw():
            self._tk_preview = tkimg   # keep reference alive (prevents GC blank)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=tkimg)
            self.src_count_var.set(f"{n_captured} sources detected")

        self.root.after(0, _draw)

    def _opt(self, entry: ttk.Entry, default: float) -> float:
        try:
            return float(entry.get())
        except ValueError:
            return default

    def _build_ref_preview(self, folder: str, snr: float, maxsrc: int, fwhm: float):
        """Load a few frames, build median reference, detect sources.
        Parameters are pre-read on the main thread to avoid cross-thread Tk access."""
        files = list_image_files(folder)
        if not files:
            self.log("No image files found in folder.")
            return
        n = len(files)
        idxs = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
        imgs = []
        for i in idxs:
            try:
                im, _, _, _ = load_image_and_time(files[i])
                imgs.append(im)
            except Exception as e:
                self.log(f"  Skipped frame {i}: {e}")
        if not imgs:
            self.log("Could not load any frames for preview.")
            return
        ref = np.median(np.stack(imgs), axis=0)
        _, xy = detect_sources(ref, fwhm=fwhm,
                               threshold_sigma=snr, max_sources=maxsrc)
        self.ref_image = ref
        self.ref_xy    = xy
        self.update_canvas(ref, xy)
        self.log(f"Reference built from {len(imgs)} frames — "
                 f"{len(xy)} sources detected  ({n} frames total).")

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder with images")
        if not folder:
            return
        self.selected_folder = folder
        self.log(f"Folder selected: {folder}")
        # Snapshot widget values HERE on the main thread before spawning worker
        snr    = self._opt(self.e_snr,    MIN_SNR)
        maxsrc = int(self._opt(self.e_maxsrc, MAX_SOURCES))
        fwhm   = self._opt(self.e_fwhm,   FWHM_GUESS)
        threading.Thread(target=self._build_ref_preview,
                         args=(folder, snr, maxsrc, fwhm), daemon=True).start()

    def preview_single(self):
        path = filedialog.askopenfilename(
            title="Select single image",
            filetypes=[("Images",
                        "*.fits *.fit *.fts *.png *.jpg *.jpeg *.tif *.tiff"),
                       ("All", "*.*")])
        if not path:
            return
        # Snapshot widget values on the main thread before the worker starts
        snr    = self._opt(self.e_snr,    MIN_SNR)
        maxsrc = int(self._opt(self.e_maxsrc, MAX_SOURCES))
        fwhm   = self._opt(self.e_fwhm,   FWHM_GUESS)
        def _load():
            try:
                img, _, _, _ = load_image_and_time(path)
                _, xy  = detect_sources(img, fwhm=fwhm,
                                        threshold_sigma=snr, max_sources=maxsrc)
                self.ref_xy    = xy
                self.ref_image = img
                self.update_canvas(img, xy)
                self.log(f"Preview: {os.path.basename(path)} — {len(xy)} sources.")
            except Exception as e:
                self.log(f"Preview error: {e}")
                self.root.after(0, lambda: messagebox.showerror("Preview error", str(e)))
        threading.Thread(target=_load, daemon=True).start()

    def request_cancel(self):
        self._cancel.set()
        self.log("Cancel requested — will stop after current frame.")

    def export_catalog(self):
        if not self._output_folder:
            messagebox.showinfo("No output yet",
                                "Run the pipeline first to generate a catalog.")
            return
        cat_path = os.path.join(self._output_folder, "master_catalog.csv")
        if not os.path.exists(cat_path):
            messagebox.showinfo("No catalog",
                                f"master_catalog.csv not found in:\n{self._output_folder}")
            return
        dest = filedialog.asksaveasfilename(
            title="Save master catalog",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if not dest:
            return
        shutil.copyfile(cat_path, dest)
        messagebox.showinfo("Saved", f"Master catalog saved to:\n{dest}")

    def view_plots(self):
        if not self._output_folder:
            messagebox.showinfo("No output yet", "Run the pipeline first.")
            return
        plots_dir = os.path.join(self._output_folder, "plots")
        if not os.path.isdir(plots_dir):
            messagebox.showinfo("No plots", f"Plots folder not found:\n{plots_dir}")
            return
        pngs = glob.glob(os.path.join(plots_dir, "*.png"))
        if not pngs:
            messagebox.showinfo("No plots", "No PNG plots found yet.")
            return
        # Open plots with the OS default viewer (avoids calling plt.show()
        # from a non-main thread, which crashes on macOS/Windows)
        import subprocess, platform
        for p in sorted(pngs):
            try:
                if platform.system() == "Darwin":
                    subprocess.Popen(["open", p])
                elif platform.system() == "Windows":
                    os.startfile(p)
                else:
                    subprocess.Popen(["xdg-open", p])
            except Exception as e:
                self.log(f"Could not open {os.path.basename(p)}: {e}")


    def run_pipeline_async(self):
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Already running",
                                "The pipeline is already running. "
                                "Click Cancel to stop it first.")
            return
        folder = self.selected_folder
        if not folder:
            folder = filedialog.askdirectory(title="Select folder with images")
            if not folder:
                return
            self.selected_folder = folder

        # ── Snapshot ALL widget values on the main thread ─────────────────
        # Worker threads must never read Tk widgets — doing so races with the
        # event loop and causes deadlocks / stalls.
        try:
            params = {
                "snr":      float(self.e_snr.get()),
                "maxsrc":   int(float(self.e_maxsrc.get())),
                "fwhm":     float(self.e_fwhm.get()),
                "ap_r":     float(self.e_apr.get()),
                "sky_in":   float(self.e_skyin.get()),
                "sky_out":  float(self.e_skyout.get()),
                "recenter": bool(self.v_recenter.get()),
                "do_bls":   bool(self.v_bls.get()) and HAS_BLS,
            }
        except ValueError as e:
            messagebox.showerror("Bad parameter", f"Check your settings:\n{e}")
            return

        self._cancel.clear()
        self._set_buttons(running=True)
        self._set_progress(0, "")
        self._worker_thread = threading.Thread(
            target=self._pipeline_worker, args=(folder, params), daemon=True)
        self._worker_thread.start()

    def _pipeline_worker(self, folder: str, params: dict):
        try:
            self._run_pipeline(folder, params)
        except Exception:
            tb = traceback.format_exc()
            self.log(f"UNHANDLED ERROR:\n{tb}")
            self.root.after(0, lambda: messagebox.showerror(
                "Pipeline error", tb[:600]))
        finally:
            self._set_buttons(running=False)
            self._cancel.clear()

    def _run_pipeline(self, folder: str, params: dict):
        out_root  = os.path.join(folder, "stage2_output")
        lc_dir    = os.path.join(out_root, "lightcurves")
        plots_dir = os.path.join(out_root, "plots")
        for d in (out_root, lc_dir, plots_dir):
            os.makedirs(d, exist_ok=True)
        self._output_folder = out_root

        # Collect files — FITS only for photometry (non-FITS have no time headers)
        all_files = list_image_files(folder)
        all_files = [f for f in all_files
                     if not os.path.realpath(f).startswith(
                         os.path.realpath(out_root))]
        fits_exts = {".fits", ".fit", ".fts"}
        files     = [f for f in all_files
                     if os.path.splitext(f)[1].lower() in fits_exts]
        n_skipped = len(all_files) - len(files)
        if n_skipped:
            self.log(f"  ⚠  Skipped {n_skipped} non-FITS file(s) "
                     f"(PNG/JPG/TIF carry no time headers).")
        if not files:
            files = all_files   # graceful fallback if folder has no FITS at all
            self.log("  ⚠  No FITS files found — falling back to all image types. "
                     "Times will use frame-index fallback.")
        n_frames = len(files)
        if n_frames == 0:
            self.log("ERROR: No image files found in the selected folder.")
            return
        self.log(f"Found {n_frames} frames for photometry in  {folder}")

        # Unpack pre-validated params — never touch Tk widgets from here on
        snr      = params["snr"]
        maxsrc   = params["maxsrc"]
        fwhm     = params["fwhm"]
        ap_r     = params["ap_r"]
        sky_in   = params["sky_in"]
        sky_out  = params["sky_out"]
        recenter = params["recenter"]
        do_bls   = params["do_bls"]  # defined HERE — safe to use below

        # BLS status diagnostic — now safe because do_bls is defined
        if do_bls:
            self.log(f"BLS enabled — will run on stars with ≥ {BLS_MIN_POINTS} "
                     f"valid frames  (period range {BLS_MIN_PERIOD}–{BLS_MAX_PERIOD} d)")
        else:
            self.log("BLS disabled (unchecked in GUI or astropy.timeseries unavailable)")

        if sky_in >= sky_out:
            self.log("ERROR: Sky inner radius must be less than outer radius.")
            return
        if ap_r >= sky_in:
            self.log("WARNING: Aperture radius overlaps sky annulus — "
                     "sky subtraction may be biased.")

        self.log("Building median reference frame …")
        sample_n   = min(7, n_frames)
        sample_idx = np.linspace(0, n_frames - 1, sample_n, dtype=int)
        ref_imgs   = []
        for i in sample_idx:
            im, _, _, _ = load_image_and_time(files[i])
            ref_imgs.append(im)
        ref = np.median(np.stack(ref_imgs), axis=0)

        self.log(f"Detecting sources  (SNR≥{snr}, FWHM={fwhm}px, max={maxsrc}) …")
        sources, ref_xy = detect_sources(ref, fwhm=fwhm,
                                         threshold_sigma=snr,
                                         max_sources=maxsrc)
        if len(ref_xy) == 0:
            self.log("ERROR: No sources detected. Try lowering Min SNR.")
            return
        n_src = len(ref_xy)
        self.log(f"Detected {n_src} sources on reference.")
        self.ref_xy    = ref_xy
        self.ref_image = ref
        self.update_canvas(ref, ref_xy)

        # compute sharpness on reference (peak / total flux)
        ref_sharpness = np.full(n_src, np.nan)
        H_ref, W_ref  = ref.shape
        box = RECENTER_BOX
        for sid, (x0, y0) in enumerate(ref_xy):
            xi, yi = int(round(x0)), int(round(y0))
            x1, x2 = max(0, xi - box), min(W_ref, xi + box + 1)
            y1, y2 = max(0, yi - box), min(H_ref, yi + box + 1)
            cut = ref[y1:y2, x1:x2]
            if cut.size:
                ref_sharpness[sid] = cut.max() / (cut.sum() or 1.0)

        flux_mat     = np.full((n_src, n_frames), np.nan)
        bg_mat       = np.full((n_src, n_frames), np.nan)
        snr_mat      = np.full((n_src, n_frames), np.nan)
        sat_mat      = np.zeros((n_src, n_frames), dtype=bool)
        times        = np.full(n_frames, np.nan)
        time_sources = [""] * n_frames   # per-frame provenance

        self.log("Running per-frame aperture photometry …")
        for fi, path in enumerate(files):
            if self._cancel.is_set():
                self.log(f"Cancelled at frame {fi+1}/{n_frames}.")
                return

            try:
                im, t, t_src, hdr = load_image_and_time(path, fits_only=False)
            except Exception as e:
                self.log(f"  Frame {fi+1}: LOAD ERROR — {e}")
                continue

            if t is not None and np.isfinite(t):
                times[fi]        = t
                time_sources[fi] = t_src
            else:
                times[fi]        = float(fi)
                time_sources[fi] = TIME_SOURCE_FALLBACK
                self.log(f"  Frame {fi+1}: ⚠ no timestamp "
                         f"({os.path.basename(path)}) — using frame index {fi}")

            pos = ref_xy.copy()
            if recenter:
                pos = recenter_positions(im, pos)

            try:
                flux, sky = measure_apertures(im, pos, ap_r, sky_in, sky_out)
            except Exception as e:
                self.log(f"  Frame {fi+1}: PHOTOMETRY ERROR — {e}")
                continue

            frame_snr   = compute_snr(flux, sky, ap_r)
            saturated   = (im.max() >= SATURATION_LEVEL)

            flux_mat[:, fi] = flux
            bg_mat[:,   fi] = sky
            snr_mat[:,  fi] = frame_snr
            if saturated:
                # flag all sources in this frame as potentially saturated
                sat_mat[:, fi] = True

            pct = 100.0 * (fi + 1) / n_frames
            self._set_progress(pct, f"{fi+1}/{n_frames}")
            if fi % max(1, n_frames // 40) == 0:
                self.log(f"  Frame {fi+1}/{n_frames}  ({pct:.0f}%)  "
                         f"— {path.split(os.sep)[-1]}")

        times = ensure_numeric(times)
        t_valid = times[np.isfinite(times)]
        t0      = t_valid[0] if t_valid.size else 0.0
        times   = times - t0

        # ── time diagnostics ──────────────────────────────────────────────────
        n_fb  = time_sources.count(TIME_SOURCE_FALLBACK)
        n_hdr = sum(1 for s in time_sources if s and s != TIME_SOURCE_FALLBACK)
        if n_fb > 0 and n_hdr > 0:
            self.log(f"  ⚠  Mixed time sources: {n_hdr} header + {n_fb} frame-index. "
                     "Light curves may have uneven time spacing.")
        elif n_fb == n_frames:
            self.log("  ⚠  All times are frame-index fallbacks — "
                     "no FITS timestamps found in any frame.")
        else:
            srcs = sorted({s for s in time_sources if s and s != TIME_SOURCE_FALLBACK})
            self.log(f"  ✓  Time source(s): {', '.join(srcs)}")

        # ── monotonicity check + sort ─────────────────────────────────────────
        if t_valid.size > 1 and not np.all(np.diff(t_valid) >= 0):
            self.log("  ⚠  Times not monotonically increasing — sorting by time.")
            order        = np.argsort(times)
            times        = times[order]
            time_sources = [time_sources[i] for i in order]
            flux_mat     = flux_mat[:, order]
            bg_mat       = bg_mat[:,   order]
            snr_mat      = snr_mat[:,  order]
            sat_mat      = sat_mat[:,  order]

        # ── time span sanity ──────────────────────────────────────────────────
        fin = times[np.isfinite(times)]
        tspan = float(fin.max() - fin.min()) if fin.size > 1 else 0.0
        if tspan == 0.0:
            self.log("  ⚠  Time span is zero — all frames share the same timestamp.")

        # Dominant time system (for catalog metadata)
        dominant_src = (max(set(time_sources), key=time_sources.count)
                        if any(time_sources) else TIME_SOURCE_FALLBACK)
        time_unit  = "days (MJD offset)" if dominant_src != TIME_SOURCE_FALLBACK                      else "frame index"
        time_scale = "UTC" if dominant_src != TIME_SOURCE_FALLBACK else "N/A"

        self.log("Computing per-star statistics …")
        catalog_rows = []

        for sid in range(n_src):
            flux     = ensure_numeric(flux_mat[sid])
            sky      = ensure_numeric(bg_mat[sid])
            snr_vals = ensure_numeric(snr_mat[sid])
            sat_vals = sat_mat[sid]

            good = np.isfinite(flux)
            n_good = int(good.sum())

            if n_good >= 5:
                med_flux  = float(np.nanmedian(flux[good]))
                std_flux  = float(np.nanstd(flux[good]))
                mad_flux  = float(np.nanmedian(np.abs(flux[good] - med_flux)))
                mean_snr  = float(np.nanmean(snr_vals[good]))
                norm_flux = flux / med_flux if med_flux != 0 else flux * np.nan
                med_mag   = float(np.nanmedian(instrumental_magnitude(flux[good])))
            else:
                med_flux  = std_flux = mad_flux = mean_snr = med_mag = np.nan
                norm_flux = np.full_like(flux, np.nan)

            sat_frac = float(sat_vals.mean())

            # save per-star CSV (time_source column records provenance per frame)
            lc_df = pd.DataFrame({
                "time":        times,
                "time_source": time_sources,
                "flux":        flux,
                "bg":          sky,
                "norm_flux":   norm_flux,
                "flux_use":    norm_flux,
                "snr":         snr_vals,
                "saturated":   sat_vals.astype(int),
            })
            lc_path = os.path.join(lc_dir, f"star_{sid:04d}.csv")
            lc_df.to_csv(lc_path, index=False, float_format="%.8f")

            # ── BLS transit search ───────────────────────────────────────────
            # Runs whenever do_bls=True AND the star has enough valid frames.
            # All four output values always defined regardless of path taken.
            bls_period = bls_power = bls_depth = bls_snr = np.nan
            bls_reason = "BLS disabled"
            if do_bls and n_good >= BLS_MIN_POINTS:
                (bls_period, bls_power,
                 bls_depth, bls_snr, bls_reason) = run_bls(times, norm_flux)
                if bls_reason:
                    # Only log failures — not every star (avoids 800-line flood)
                    self.log(f"  Star {sid:04d}: BLS skipped — {bls_reason}")
            elif do_bls:
                bls_reason = f"too few good frames ({n_good} < {BLS_MIN_POINTS})"

            # is_candidate: True when BLS found a credible transit signal.
            # Stage 4 reads this flag directly from the catalog.
            # Threshold: SNR ≥ 3 (conservative — Stage 4 applies its own cut).
            is_cand = (
                np.isfinite(bls_snr) and float(bls_snr) >= 3.0
                and np.isfinite(bls_period) and float(bls_period) > 0
            )

            x, y = ref_xy[sid]
            catalog_rows.append({
                "star_id":      sid,
                "x":            round(float(x), 3),
                "y":            round(float(y), 3),
                "n_points":     n_good,
                "median_flux":  med_flux,
                "std_flux":     std_flux,
                "mad_flux":     mad_flux,
                "mean_snr":     mean_snr,
                "instr_mag":    med_mag,
                "sharpness":    float(ref_sharpness[sid]),
                "sat_fraction": sat_frac,
                # BLS results — all four values needed by Stage 3 and Stage 4
                "bls_period":   bls_period,
                "bls_power":    bls_power,
                "bls_depth":    bls_depth,
                "bls_snr":      bls_snr,
                # is_candidate flag — Stage 4 reads this to gate candidates
                "is_candidate": int(is_cand),
                # time provenance metadata
                "time_unit":    time_unit,
                "time_scale":   time_scale,
                "time_source":  dominant_src,
                "t_ref_mjd":    round(float(t0), 6),
                "lc_file":      os.path.relpath(lc_path, out_root),
            })

        # ── BLS summary log (aggregate, not per-star) ────────────────────────
        if do_bls:
            n_cands = sum(1 for r in catalog_rows if r.get("is_candidate", 0))
            n_ran   = sum(1 for r in catalog_rows
                          if np.isfinite(r.get("bls_period", np.nan)))
            self.log(f"BLS complete — ran on {n_ran}/{n_src} stars, "
                     f"{n_cands} transit candidate(s) (SNR ≥ 3.0)")
            if n_cands:
                for r in catalog_rows:
                    if r.get("is_candidate", 0):
                        self.log(f"  ★ Star {int(r['star_id']):04d}  "
                                 f"P={r['bls_period']:.4f} d  "
                                 f"depth={r['bls_depth']:.5f}  "
                                 f"SNR={r['bls_snr']:.2f}")

        #write master catalog
        cat_df   = pd.DataFrame(catalog_rows)
        cat_path = os.path.join(out_root, "master_catalog.csv")
        cat_df.to_csv(cat_path, index=False)
        self.log(f"Master catalog → {cat_path}  ({n_src} stars)")

        # diagnostic plots
        self.log("Saving diagnostic plots …")
        self._plot_source_map(cat_df, plots_dir)
        self._plot_top_variable(cat_df, lc_dir, plots_dir, times)
        if do_bls:
            self._plot_bls_candidates(cat_df, lc_dir, plots_dir, times)

        self._set_progress(100, "Done")
        self.log("─" * 60)
        self.log(f"Stage 2 complete.  Outputs in:  {out_root}")

    def _plot_source_map(self, cat: pd.DataFrame, plots_dir: str):
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # std_flux scatter map
            sc = axes[0].scatter(cat["x"], cat["y"],
                                 c=cat["std_flux"].fillna(0),
                                 s=12, cmap="plasma")
            axes[0].invert_yaxis()
            axes[0].set_title("Sources — colour = flux std")
            axes[0].set_xlabel("X (px)");  axes[0].set_ylabel("Y (px)")
            plt.colorbar(sc, ax=axes[0], label="std_flux")

            # magnitude histogram
            mags = cat["instr_mag"].dropna()
            axes[1].hist(mags, bins=30, color="steelblue", edgecolor="white", lw=0.4)
            axes[1].set_xlabel("Instrumental magnitude")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Magnitude distribution")

            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, "source_map.png"), dpi=130)
            plt.close(fig)
            self.log("  Saved  source_map.png")
        except Exception as e:
            self.log(f"  source_map plot failed: {e}")

    def _plot_top_variable(self, cat: pd.DataFrame, lc_dir: str,
                           plots_dir: str, times: np.ndarray, top_n: int = 5):
        """Plot the light curves of the N most variable stars."""
        try:
            valid = cat.dropna(subset=["std_flux", "median_flux"]).copy()
            valid = valid[valid["median_flux"] > 0]
            # fractional variability
            valid["frac_std"] = valid["std_flux"] / valid["median_flux"]
            top = valid.nlargest(top_n, "frac_std")

            fig, axes = plt.subplots(top_n, 1,
                                     figsize=(10, 2.5 * top_n), sharex=True)
            if top_n == 1:
                axes = [axes]

            for ax, (_, row) in zip(axes, top.iterrows()):
                lc_path = os.path.join(lc_dir,
                                       f"star_{int(row['star_id']):04d}.csv")
                if not os.path.exists(lc_path):
                    continue
                lc = pd.read_csv(lc_path)
                t  = np.asarray(lc["time"],     dtype=float)
                y  = np.asarray(lc["norm_flux"], dtype=float)
                m  = np.isfinite(t) & np.isfinite(y)
                ax.plot(t[m], y[m], ".", ms=4, color="steelblue")
                ax.axhline(1.0, color="gray", lw=0.6, ls="--")
                ax.set_ylabel("Norm. flux")
                label = (f"Star {int(row['star_id']):04d}  "
                         f"x={row['x']:.1f} y={row['y']:.1f}  "
                         f"σ/μ={row['frac_std']:.4f}")
                if not np.isnan(row.get("bls_period", np.nan)):
                    label += f"  BLS P={row['bls_period']:.3f}"
                ax.set_title(label, fontsize=8)

            axes[-1].set_xlabel("Time (days)")
            plt.suptitle("Top Variable Stars", fontsize=11)
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, "top_variable_lc.png"), dpi=130)
            plt.close(fig)
            self.log("  Saved  top_variable_lc.png")
        except Exception as e:
            self.log(f"  top_variable plot failed: {e}")

    def _plot_bls_candidates(self, cat: pd.DataFrame, lc_dir: str,
                              plots_dir: str, times: np.ndarray, top_n: int = 3):
        """Phase-fold and plot the top BLS candidates, ranked by SNR."""
        try:
            bls_ok = cat.dropna(subset=["bls_period", "bls_power"])
            bls_ok = bls_ok[bls_ok["bls_period"] > 0]
            if bls_ok.empty:
                self.log("  No BLS candidates to plot — all bls_period values are NaN.")
                return
            # Rank by SNR if available, else fall back to power
            rank_col = "bls_snr" if ("bls_snr" in bls_ok.columns
                                      and bls_ok["bls_snr"].notna().any()) else "bls_power"
            top = bls_ok.nlargest(top_n, rank_col)

            fig, axes = plt.subplots(top_n, 1,
                                     figsize=(9, 3 * top_n), sharex=False)
            if top_n == 1:
                axes = [axes]

            for ax, (_, row) in zip(axes, top.iterrows()):
                lc_path = os.path.join(lc_dir,
                                       f"star_{int(row['star_id']):04d}.csv")
                if not os.path.exists(lc_path):
                    continue
                lc = pd.read_csv(lc_path)
                t  = np.asarray(lc["time"],     dtype=float)
                y  = np.asarray(lc["norm_flux"], dtype=float)
                m  = np.isfinite(t) & np.isfinite(y)
                t, y = t[m], y[m]
                P = float(row["bls_period"])
                phase = (t % P) / P
                ax.plot(phase, y, ".", ms=3, color="coral", alpha=0.7)
                ax.axhline(1.0, color="gray", lw=0.6, ls="--")
                ax.set_ylabel("Norm. flux")
                ax.set_xlabel("Phase")
                depth_str = (f"  depth={row['bls_depth']:.5f}" 
                             if "bls_depth" in row and np.isfinite(row.get("bls_depth", np.nan))
                             else "")
                snr_str   = (f"  SNR={row['bls_snr']:.2f}"
                             if "bls_snr" in row and np.isfinite(row.get("bls_snr", np.nan))
                             else "")
                ax.set_title(
                    f"Star {int(row['star_id']):04d}  "
                    f"P={P:.4f} d  power={row['bls_power']:.3f}"
                    f"{depth_str}{snr_str}",
                    fontsize=8)

            plt.suptitle("Top BLS Transit Candidates (Phase-folded)", fontsize=11)
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, "bls_candidates.png"), dpi=130)
            plt.close(fig)
            self.log("  Saved  bls_candidates.png")
        except Exception as e:
            self.log(f"  BLS candidates plot failed: {e}")

def main():
    root = tk.Tk()
    app  = Stage2GUI(root)
    root.geometry("1200x780")
    root.mainloop()


if __name__ == "__main__":
    main()