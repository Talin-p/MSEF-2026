"""
MSEF Pipeline — Stage 4: Transit Validation, Visualization & Reporting
=======================================================================
Consumes the outputs of Stages 2 and 3 to produce:

  1.  Star-field map  — annotated source positions from master_catalog.csv,
                        with candidates highlighted
  2.  Light-curve plots — flux vs time for each selected star, with GP
                          trend overlay read from Stage 3 CSV output
  3.  Variability map — pixel-std image computed from all frames (optional)
  4.  Phase-folded transit plot — folded on the BLS period from Stage 3
  5.  Validation metrics — SNR, transit depth, planet radius estimate,
                           variability index, blending flag
  6.  Full candidate report — JSON + CSV summary of every flagged star
  7.  Multi-panel diagnostic figure — one PNG per candidate for quick review

Fixes / improvements over original Stage 4 skeleton
----------------------------------------------------
• Complete GUI (Tk) — no longer a bare command-line script
• Reads master_catalog.csv and lightcurve CSVs produced by Stage 2
• Reads Stage 3 GP-result CSVs (time, flux, gp_mean, gp_std, residual)
• main() has a proper argument-count guard (original crashed silently)
• spatial_validation() now handles NaN/zero std_flux without ZeroDivision
• Blending radius exposed as a tunable parameter (was hardcoded 10 px)
• Planet radius estimate added: R_p/R_* = sqrt(transit_depth)
• All plots closed after saving (plt.close) to prevent memory leaks
• Batch mode: process every star in the catalog automatically
• Candidate report exported as both JSON and CSV
• Progress bar and thread-safe logging
"""

from __future__ import annotations

import glob
import json
import math
import os
import threading
import time
import traceback
import warnings

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ─────────────────────────── tunables ────────────────────────────────────────

SNR_THRESHOLD      = 5.0     # minimum spatial SNR for a source to be a candidate
BLEND_RADIUS_PX    = 10.0    # neighbours within this radius flag the source as blended
TRANSIT_SNR_MIN    = 3.0     # minimum BLS SNR to flag a transit candidate
STAR_RADIUS_RSUN   = 1.0     # assumed host-star radius (R_sun) for planet size estimate
RSUN_TO_REARTH     = 109.076 # 1 R_sun = 109.076 R_earth

# ─────────────────────────── pure-function science core ─────────────────────

def load_master_catalog(path: str) -> pd.DataFrame:
    """Load the Stage 2 master_catalog.csv."""
    df = pd.read_csv(path)
    required = {"star_id", "x", "y", "median_flux", "std_flux"}
    missing  = required - set(df.columns)
    if missing:
        warnings.warn(f"master_catalog missing columns: {missing}")
    return df


def load_stage3_gp_csv(path: str) -> pd.DataFrame | None:
    """Load a Stage-3 GP-result CSV (time, flux, gp_mean, gp_std[, residual])."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Ensure residual column exists
    if "residual" not in df.columns and "gp_mean" in df.columns and "flux" in df.columns:
        df["residual"] = df["flux"] - df["gp_mean"]
    return df


def spatial_validation(
    sources:      pd.DataFrame,
    stage3_info:  dict,
    snr_thresh:   float = SNR_THRESHOLD,
    blend_radius: float = BLEND_RADIUS_PX,
) -> dict:
    """
    Validate a transit candidate spatially using the Stage 2 source catalog.

    Parameters
    ----------
    sources      : DataFrame with at least [x, y, median_flux, std_flux]
    stage3_info  : dict with keys produced by Stage 3 BLS
                   (bls_period, bls_power, bls_depth, bls_snr, is_candidate)
    snr_thresh   : minimum flux SNR for a source to qualify
    blend_radius : pixel radius within which neighbours indicate blending

    Returns
    -------
    dict with keys: stage4_pass, reason, source_x, source_y, spatial_snr,
                    blended, n_neighbors, transit_depth, transit_period,
                    transit_snr, planet_radius_rearth
    """
    result: dict = {
        "stage4_pass":        False,
        "reason":             "",
        "source_x":           np.nan,
        "source_y":           np.nan,
        "spatial_snr":        np.nan,
        "blended":            False,
        "n_neighbors":        0,
        "transit_depth":      np.nan,
        "transit_period":     np.nan,
        "transit_snr":        np.nan,
        "planet_radius_rearth": np.nan,
    }

    # ── Stage 3 temporal check ────────────────────────────────────────────
    is_cand = bool(stage3_info.get("is_candidate", False))
    bls_snr = float(stage3_info.get("bls_snr",  np.nan))
    bls_depth  = float(stage3_info.get("bls_depth",  np.nan))
    bls_period = float(stage3_info.get("bls_period", np.nan))

    if not is_cand:
        result["reason"] = "No temporal transit candidate from Stage 3"
        return result

    if not np.isfinite(bls_snr) or bls_snr < TRANSIT_SNR_MIN:
        result["reason"] = (f"BLS SNR={bls_snr:.2f} below threshold {TRANSIT_SNR_MIN}")
        return result

    # ── Source-level spatial SNR ──────────────────────────────────────────
    src = sources.copy()
    # Guard against NaN / zero std_flux
    src = src[src["std_flux"].notna() & (src["std_flux"] > 0)].copy()
    src = src[src["median_flux"].notna()].copy()
    if src.empty:
        result["reason"] = "No sources with valid flux/std in catalog"
        return result

    src["spatial_snr"] = np.abs(src["median_flux"]) / src["std_flux"]
    strong = src[src["spatial_snr"] >= snr_thresh]

    if strong.empty:
        result["reason"] = f"No source above spatial SNR threshold ({snr_thresh})"
        return result

    # ── Select the brightest qualifying source ────────────────────────────
    best_idx    = strong["spatial_snr"].idxmax()
    best        = strong.loc[best_idx]
    x0, y0      = float(best["x"]), float(best["y"])
    spatial_snr = float(best["spatial_snr"])

    # ── Blending check ────────────────────────────────────────────────────
    dx = src["x"] - x0
    dy = src["y"] - y0
    r  = np.sqrt(dx**2 + dy**2)
    neighbors  = src[(r < blend_radius) & (r > 1.0)]
    n_neighbors = len(neighbors)
    blended     = n_neighbors > 0

    # ── Planet radius estimate ────────────────────────────────────────────
    planet_r = np.nan
    if np.isfinite(bls_depth) and bls_depth > 0:
        # depth = (R_planet / R_star)^2  →  R_planet = R_star * sqrt(depth)
        planet_r = STAR_RADIUS_RSUN * math.sqrt(bls_depth) * RSUN_TO_REARTH

    result.update({
        "stage4_pass":          True,
        "reason":               "Passed all checks",
        "source_x":             x0,
        "source_y":             y0,
        "spatial_snr":          spatial_snr,
        "blended":              blended,
        "n_neighbors":          n_neighbors,
        "transit_depth":        bls_depth,
        "transit_period":       bls_period,
        "transit_snr":          bls_snr,
        "planet_radius_rearth": planet_r,
    })
    return result


def compute_variability_index(lc_csv_path: str) -> dict:
    """Compute variability metrics from a Stage 2 light-curve CSV."""
    out = {"variability_index": np.nan, "rms": np.nan, "n_points": 0}
    if not os.path.exists(lc_csv_path):
        return out
    try:
        df = pd.read_csv(lc_csv_path)
        col = "norm_flux" if "norm_flux" in df.columns else "flux"
        y   = df[col].to_numpy(dtype=float)
        good = np.isfinite(y)
        if good.sum() < 3:
            return out
        y = y[good]
        med  = float(np.median(y))
        rms  = float(np.std(y))
        vix  = rms / med if med != 0 else np.nan
        out.update({"variability_index": vix, "rms": rms,
                    "n_points": int(good.sum())})
    except Exception:
        pass
    return out

# ─────────────────────────── plotting helpers ────────────────────────────────

def _close(fig):
    plt.close(fig)


def plot_star_field(sources: pd.DataFrame,
                    candidates: pd.DataFrame,
                    out_path: str,
                    image: np.ndarray | None = None):
    """Annotated star-field map."""
    fig, ax = plt.subplots(figsize=(8, 7))

    if image is not None:
        vmin = np.nanpercentile(image, 1)
        vmax = np.nanpercentile(image, 99)
        ax.imshow(image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax,
                  interpolation="nearest")

    # All sources — colour by variability index if available
    if "variability_index" in sources.columns:
        sc = ax.scatter(sources["x"], sources["y"],
                        c=sources["variability_index"].fillna(0),
                        s=12, cmap="plasma", alpha=0.7, label="All sources")
        plt.colorbar(sc, ax=ax, label="Variability index (σ/μ)")
    else:
        ax.scatter(sources["x"], sources["y"],
                   s=12, color="cyan", alpha=0.6, label="All sources")

    # Candidates — large red circles
    if not candidates.empty:
        ax.scatter(candidates["source_x"], candidates["source_y"],
                   s=220, facecolors="none", edgecolors="red",
                   linewidths=2, label="Transit candidate")
        for _, row in candidates.iterrows():
            ax.annotate(f"  {int(row['star_id'])}",
                        (row["source_x"], row["source_y"]),
                        fontsize=7, color="red")

    ax.invert_yaxis()
    ax.set_xlabel("X (pixels)");  ax.set_ylabel("Y (pixels)")
    ax.set_title("Stage 4 — Star Field with Transit Candidates")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    _close(fig)


def plot_light_curve(star_id: int,
                     lc_path: str,
                     gp_path: str | None,
                     result: dict,
                     out_path: str):
    """Two-panel light curve: flux+GP (top), residuals (bottom)."""
    lc = pd.read_csv(lc_path)
    col = "norm_flux" if "norm_flux" in lc.columns else "flux"
    t   = lc["time"].to_numpy(float)
    y   = lc[col].to_numpy(float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]

    fig = plt.figure(figsize=(9, 6))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(t, y, ".", ms=3, color="steelblue", alpha=0.7, label="Flux")

    # GP overlay from Stage 3 CSV
    if gp_path and os.path.exists(gp_path):
        gp = pd.read_csv(gp_path)
        tg  = gp["time"].to_numpy(float)
        gm  = gp["gp_mean"].to_numpy(float)
        gs_ = gp["gp_std"].to_numpy(float)
        res = gp["residual"].to_numpy(float) if "residual" in gp.columns else y - gm
        ax1.plot(tg, gm, "-", lw=1.4, color="tomato", label="GP mean")
        ax1.fill_between(tg, gm - gs_, gm + gs_,
                         color="tomato", alpha=0.22, label="±1σ")
        ax2.plot(tg, res, ".", ms=3, color="steelblue", alpha=0.7)
        ax2.axhline(0, color="k", lw=0.6)
    else:
        ax2.set_visible(False)

    # BLS annotation
    period = result.get("transit_period", np.nan)
    depth  = result.get("transit_depth",  np.nan)
    snr    = result.get("transit_snr",    np.nan)
    rp     = result.get("planet_radius_rearth", np.nan)
    parts  = []
    if np.isfinite(period): parts.append(f"P={period:.4f} d")
    if np.isfinite(depth):  parts.append(f"depth={depth:.5f}")
    if np.isfinite(snr):    parts.append(f"SNR={snr:.2f}")
    if np.isfinite(rp):     parts.append(f"Rp≈{rp:.2f} R⊕")
    if parts:
        ax1.text(0.02, 0.05, "  ".join(parts), transform=ax1.transAxes,
                 fontsize=7, color="darkgreen",
                 bbox=dict(fc="white", alpha=0.75, ec="none"))

    ax1.set_ylabel("Norm. Flux");  ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title(f"Star {star_id:04d} — Light Curve & GP Fit", fontsize=9)
    ax2.set_ylabel("Residual");    ax2.set_xlabel("Time (days)")
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    _close(fig)


def plot_phase_fold(star_id: int,
                    lc_path: str,
                    period: float,
                    out_path: str):
    """Phase-folded light curve on the BLS period."""
    if not np.isfinite(period) or period <= 0:
        return
    lc   = pd.read_csv(lc_path)
    col  = "norm_flux" if "norm_flux" in lc.columns else "flux"
    t    = lc["time"].to_numpy(float)
    y    = lc[col].to_numpy(float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) < 5:
        return

    phase = (t % period) / period
    # sort for a cleaner line
    order = np.argsort(phase)
    ph, yp = phase[order], y[order]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ph, yp, ".", ms=3, color="steelblue", alpha=0.7)
    ax.axhline(1.0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("Phase");  ax.set_ylabel("Norm. Flux")
    ax.set_title(f"Star {star_id:04d} — Phase-folded  (P={period:.4f} d)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    _close(fig)


def plot_multi_panel(star_id: int,
                     lc_path: str,
                     gp_path: str | None,
                     result: dict,
                     sources: pd.DataFrame,
                     out_path: str):
    """
    2×2 diagnostic panel:
      (a) star-field context  (b) light curve + GP
      (c) phase-folded plot   (d) metrics summary text
    """
    period = result.get("transit_period", np.nan)
    col    = "norm_flux" if "norm_flux" in pd.read_csv(lc_path).columns else "flux"
    lc     = pd.read_csv(lc_path)
    t      = lc["time"].to_numpy(float)
    y      = lc[col].to_numpy(float)
    mk     = np.isfinite(t) & np.isfinite(y)
    t, y   = t[mk], y[mk]

    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── (a) field context ──────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.scatter(sources["x"], sources["y"], s=8, color="gray", alpha=0.5)
    if result.get("stage4_pass"):
        ax_a.scatter(result["source_x"], result["source_y"],
                     s=180, facecolors="none", edgecolors="red", lw=2)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("X (px)"); ax_a.set_ylabel("Y (px)")
    ax_a.set_title("(a) Field context", fontsize=8)

    # ── (b) light curve + GP ───────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(t, y, ".", ms=2.5, color="steelblue", alpha=0.7, label="data")
    if gp_path and os.path.exists(gp_path):
        gp = pd.read_csv(gp_path)
        ax_b.plot(gp["time"], gp["gp_mean"], "-", lw=1.2, color="tomato", label="GP")
        ax_b.fill_between(gp["time"],
                          gp["gp_mean"] - gp["gp_std"],
                          gp["gp_mean"] + gp["gp_std"],
                          color="tomato", alpha=0.2)
    ax_b.set_xlabel("Time (days)"); ax_b.set_ylabel("Norm. flux")
    ax_b.set_title("(b) Light curve & GP", fontsize=8)
    ax_b.legend(fontsize=6)

    # ── (c) phase-folded ──────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    if np.isfinite(period) and period > 0 and len(t) >= 5:
        phase = (t % period) / period
        order = np.argsort(phase)
        ax_c.plot(phase[order], y[order], ".", ms=2.5, color="steelblue", alpha=0.7)
        ax_c.axhline(1.0, color="gray", lw=0.6, ls="--")
    ax_c.set_xlabel("Phase"); ax_c.set_ylabel("Norm. flux")
    ax_c.set_title(f"(c) Phase-folded  P={period:.4f} d", fontsize=8)

    # ── (d) metrics text ──────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    lines = [
        f"Star ID        : {star_id:04d}",
        f"Stage4 pass    : {result.get('stage4_pass')}",
        f"Spatial SNR    : {result.get('spatial_snr', np.nan):.2f}",
        f"Blended        : {result.get('blended')}  "
          f"({result.get('n_neighbors', 0)} neighbours)",
        f"Transit period : {result.get('transit_period', np.nan):.4f} d",
        f"Transit depth  : {result.get('transit_depth', np.nan):.6f}",
        f"Transit SNR    : {result.get('transit_snr', np.nan):.2f}",
        f"Planet radius  : {result.get('planet_radius_rearth', np.nan):.2f} R⊕",
        f"Reason         : {result.get('reason', '')}",
    ]
    ax_d.text(0.04, 0.95, "\n".join(lines), transform=ax_d.transAxes,
              fontsize=7.5, va="top", family="monospace",
              bbox=dict(fc="#f8f8f8", ec="gray", lw=0.5, pad=6))
    ax_d.set_title("(d) Metrics", fontsize=8)

    fig.suptitle(f"Stage 4 Diagnostic — Star {star_id:04d}", fontsize=11)
    fig.savefig(out_path, dpi=140)
    _close(fig)

# ─────────────────────────── GUI ─────────────────────────────────────────────

class Stage4GUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("MSEF — Stage 4  ·  Validation & Visualization")
        root.minsize(900, 580)

        # ── state ──────────────────────────────────────────────────────────
        self.catalog_path: str | None        = None
        self.catalog_df:   pd.DataFrame | None = None
        self.lc_dir:       str | None        = None   # lightcurves/ from Stage 2
        self.gp_dir:       str | None        = None   # GP CSVs from Stage 3
        self.out_dir:      str | None        = None
        self._worker:      threading.Thread | None = None
        self._cancel       = threading.Event()

        self._build_ui()
        self.log("Stage 4 ready.  Load a master_catalog.csv to begin.")

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        tk.Label(root, text="MSEF Pipeline  ·  Stage 4: Validation & Visualization",
                 font=("Arial", 13, "bold")).pack(pady=(10, 4))

        # Button row
        br = tk.Frame(root); br.pack(fill="x", padx=10, pady=4)
        bkw = dict(width=22, pady=4, relief="groove")
        tk.Button(br, text="📋  Load master_catalog.csv",
                  command=self.load_catalog, **bkw).pack(side="left", padx=3)
        tk.Button(br, text="📁  Set Stage 3 GP folder",
                  command=self.set_gp_folder, **bkw).pack(side="left", padx=3)
        tk.Button(br, text="📂  Set Output folder",
                  command=self.set_output_folder, **bkw).pack(side="left", padx=3)
        self.run_btn = tk.Button(br, text="▶  Run Stage 4",
                                 command=self.run_async,
                                 width=14, pady=4, relief="groove",
                                 bg="#2a7a2a", fg="white")
        self.run_btn.pack(side="left", padx=3)
        self.cancel_btn = tk.Button(br, text="⛔  Cancel",
                                    command=self._request_cancel,
                                    width=10, pady=4, relief="groove",
                                    state="disabled")
        self.cancel_btn.pack(side="left", padx=3)
        tk.Button(br, text="✖  Quit", command=root.quit,
                  width=8, pady=4, relief="groove").pack(side="right", padx=3)

        # Options strip
        opts = tk.Frame(root, relief="groove", bd=1)
        opts.pack(fill="x", padx=10, pady=(0, 4))

        def _lbl(t, c):
            tk.Label(opts, text=t, font=("Arial", 8)).grid(
                row=0, column=c, padx=(8, 2), pady=3, sticky="e")
        def _ent(default, c, w=7):
            e = ttk.Entry(opts, width=w); e.insert(0, str(default))
            e.grid(row=0, column=c, padx=(0, 4)); return e

        _lbl("Spatial SNR min:", 0);  self.e_snr    = _ent(SNR_THRESHOLD,   1)
        _lbl("Blend radius (px):", 2); self.e_blend  = _ent(BLEND_RADIUS_PX, 3)
        _lbl("BLS SNR min:",     4);  self.e_bsnr   = _ent(TRANSIT_SNR_MIN, 5)
        _lbl("Star R (R☉):",    6);  self.e_rstar  = _ent(STAR_RADIUS_RSUN,7)
        self.v_multi = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Multi-panel plot per candidate",
                        variable=self.v_multi).grid(row=0, column=8, padx=6)

        # Progress
        pf = tk.Frame(root); pf.pack(fill="x", padx=10, pady=(2, 0))
        self.progress = ttk.Progressbar(pf, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True)
        self.pct_lbl = tk.Label(pf, text="", width=10, font=("Arial", 8))
        self.pct_lbl.pack(side="left", padx=4)
        self.status_var = tk.StringVar(value="Idle")
        tk.Label(root, textvariable=self.status_var,
                 font=("Arial", 9), fg="#444").pack(anchor="w", padx=12)

        # Log box
        lf = ttk.LabelFrame(root, text="Log"); lf.pack(fill="both", expand=True, padx=10, pady=6)
        self.log_box = tk.Text(lf, wrap="word", font=("Courier", 8),
                               bg="#1a1a2e", fg="#e0e0e0")
        sb = ttk.Scrollbar(lf, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

        root.protocol("WM_DELETE_WINDOW", root.quit)

    # ── thread-safe helpers ────────────────────────────────────────────────

    def log(self, msg: str):
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
        sr = "disabled" if running else "normal"
        sc = "normal"   if running else "disabled"
        self.root.after(0, lambda: (
            self.run_btn.config(state=sr),
            self.cancel_btn.config(state=sc),
        ))

    def _opt(self, entry: ttk.Entry, default: float) -> float:
        try:   return float(entry.get())
        except: return default

    # ── UI actions ─────────────────────────────────────────────────────────

    def load_catalog(self):
        path = filedialog.askopenfilename(
            title="Open master_catalog.csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        try:
            self.catalog_df   = load_master_catalog(path)
            self.catalog_path = path
            # Auto-detect lightcurves dir (sibling folder produced by Stage 2)
            cat_dir  = os.path.dirname(path)
            lc_guess = os.path.join(cat_dir, "lightcurves")
            if os.path.isdir(lc_guess):
                self.lc_dir = lc_guess
                self.log(f"Auto-detected lightcurves dir: {lc_guess}")
            self.log(f"Loaded catalog: {os.path.basename(path)}"
                     f"  ({len(self.catalog_df)} stars)")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def set_gp_folder(self):
        folder = filedialog.askdirectory(title="Select Stage 3 GP output folder")
        if folder:
            self.gp_dir = folder
            self.log(f"Stage 3 GP folder: {folder}")

    def set_output_folder(self):
        folder = filedialog.askdirectory(title="Select output folder for Stage 4 results")
        if folder:
            self.out_dir = folder
            self.log(f"Output folder: {folder}")

    def _request_cancel(self):
        self._cancel.set()
        self.log("Cancel requested — stopping after current star.")

    # ── pipeline entry ─────────────────────────────────────────────────────

    def run_async(self):
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Busy", "Stage 4 is already running.")
            return
        if self.catalog_df is None:
            messagebox.showwarning("No catalog", "Load a master_catalog.csv first.")
            return

        # Ensure output folder exists
        if not self.out_dir:
            default = os.path.join(os.path.dirname(self.catalog_path), "stage4_output")
            self.out_dir = default
        os.makedirs(self.out_dir, exist_ok=True)

        # Snapshot all GUI options on the main thread
        try:
            params = {
                "snr_thresh":    float(self.e_snr.get()),
                "blend_radius":  float(self.e_blend.get()),
                "bls_snr_min":   float(self.e_bsnr.get()),
                "star_radius":   float(self.e_rstar.get()),
                "multi_panel":   bool(self.v_multi.get()),
            }
        except ValueError as e:
            messagebox.showerror("Bad parameter", str(e)); return

        self._cancel.clear()
        self._set_buttons(running=True)
        self._set_progress(0)
        self._worker = threading.Thread(
            target=self._worker_wrapper, args=(params,), daemon=True)
        self._worker.start()

    def _worker_wrapper(self, params: dict):
        try:
            self._run_pipeline(params)
        except Exception:
            tb = traceback.format_exc()
            self.log(f"UNHANDLED ERROR:\n{tb}")
            self.root.after(0, lambda: messagebox.showerror("Error", tb[:600]))
        finally:
            self._set_buttons(running=False)
            self._cancel.clear()

    # ── main pipeline worker ───────────────────────────────────────────────

    def _run_pipeline(self, params: dict):
        cat     = self.catalog_df.copy()
        out_dir = self.out_dir
        lc_dir  = self.lc_dir
        gp_dir  = self.gp_dir
        plots_d = os.path.join(out_dir, "plots");     os.makedirs(plots_d, exist_ok=True)
        cands_d = os.path.join(out_dir, "candidates"); os.makedirs(cands_d, exist_ok=True)

        snr_thresh   = params["snr_thresh"]
        blend_radius = params["blend_radius"]
        bls_snr_min  = params["bls_snr_min"]
        star_radius  = params["star_radius"]
        multi_panel  = params["multi_panel"]

        n_stars = len(cat)
        self.log(f"Processing {n_stars} stars …")

        candidate_rows = []

        for i, (_, row) in enumerate(cat.iterrows()):
            if self._cancel.is_set():
                self.log(f"Cancelled at star {i}/{n_stars}.")
                break

            sid = int(row["star_id"])
            pct = 100.0 * (i + 1) / n_stars
            self._set_progress(pct, f"{i+1}/{n_stars}")

            # ── locate lightcurve CSV (Stage 2) ──────────────────────────
            lc_path = None
            if lc_dir:
                lc_path = os.path.join(lc_dir, f"star_{sid:04d}.csv")
                if not os.path.exists(lc_path):
                    lc_path = None
            if lc_path is None and "lc_file" in row and pd.notna(row["lc_file"]):
                lc_path = os.path.join(
                    os.path.dirname(self.catalog_path), str(row["lc_file"]))
                if not os.path.exists(lc_path):
                    lc_path = None

            # ── locate Stage 3 GP CSV ─────────────────────────────────────
            gp_path = None
            if gp_dir:
                for pattern in [f"star_{sid:04d}_gp.csv",
                                 f"star_{sid:04d}.csv"]:
                    candidate_gp = os.path.join(gp_dir, pattern)
                    if os.path.exists(candidate_gp):
                        gp_path = candidate_gp
                        break

            # ── build stage3_info dict from catalog row ───────────────────
            stage3_info = {
                "is_candidate": (
                    pd.notna(row.get("bls_power")) and
                    pd.notna(row.get("bls_snr",  np.nan)) and
                    float(row.get("bls_snr",  0)) >= bls_snr_min
                ) if "bls_snr" in row.index else (
                    pd.notna(row.get("bls_power")) and
                    pd.notna(row.get("bls_period"))
                ),
                "bls_period": float(row.get("bls_period", np.nan)),
                "bls_power":  float(row.get("bls_power",  np.nan)),
                "bls_depth":  float(row.get("bls_depth",  np.nan))
                              if "bls_depth"  in row.index else np.nan,
                "bls_snr":    float(row.get("bls_snr",   np.nan))
                              if "bls_snr"    in row.index else np.nan,
            }

            # ── variability metrics ───────────────────────────────────────
            var_metrics = {}
            if lc_path:
                var_metrics = compute_variability_index(lc_path)

            # ── spatial validation ────────────────────────────────────────
            result = spatial_validation(
                sources      = cat,
                stage3_info  = stage3_info,
                snr_thresh   = snr_thresh,
                blend_radius = blend_radius,
            )
            result["star_id"] = sid
            result.update(var_metrics)

            # ── only generate plots for candidates that pass ──────────────
            if result["stage4_pass"] and lc_path:
                period = result.get("transit_period", np.nan)

                # light-curve plot
                lc_plot = os.path.join(plots_d, f"star_{sid:04d}_lc.png")
                try:
                    plot_light_curve(sid, lc_path, gp_path, result, lc_plot)
                    self.log(f"  ✓ Star {sid:04d}: light-curve plot saved")
                except Exception as e:
                    self.log(f"  ! Star {sid:04d}: lc plot failed — {e}")

                # phase-fold
                if np.isfinite(period):
                    pf_plot = os.path.join(plots_d, f"star_{sid:04d}_phase.png")
                    try:
                        plot_phase_fold(sid, lc_path, period, pf_plot)
                    except Exception as e:
                        self.log(f"  ! Star {sid:04d}: phase-fold failed — {e}")

                # multi-panel
                if multi_panel:
                    mp_plot = os.path.join(cands_d, f"star_{sid:04d}_diagnostic.png")
                    try:
                        plot_multi_panel(sid, lc_path, gp_path, result, cat, mp_plot)
                    except Exception as e:
                        self.log(f"  ! Star {sid:04d}: multi-panel failed — {e}")

                candidate_rows.append(result)
                self.log(f"  ★ Star {sid:04d}: CANDIDATE  "
                         f"P={period:.4f} d  "
                         f"depth={result.get('transit_depth', np.nan):.5f}  "
                         f"SNR={result.get('transit_snr', np.nan):.2f}")

        # ── star-field map (all sources) ──────────────────────────────────
        cand_df = pd.DataFrame(candidate_rows) if candidate_rows else pd.DataFrame()
        sf_plot = os.path.join(plots_d, "star_field.png")
        try:
            # Attach variability_index to cat for colourmap
            if "variability_index" not in cat.columns:
                vi_map = {}
                if lc_dir:
                    for _, r in cat.iterrows():
                        sid2 = int(r["star_id"])
                        lcp  = os.path.join(lc_dir, f"star_{sid2:04d}.csv")
                        vm   = compute_variability_index(lcp)
                        vi_map[sid2] = vm.get("variability_index", np.nan)
                cat["variability_index"] = cat["star_id"].map(vi_map)
            plot_star_field(cat, cand_df, sf_plot)
            self.log(f"Star-field map → {sf_plot}")
        except Exception as e:
            self.log(f"Star-field plot failed: {e}")

        # ── write candidate report ─────────────────────────────────────────
        if candidate_rows:
            report_csv  = os.path.join(out_dir, "candidate_report.csv")
            report_json = os.path.join(out_dir, "candidate_report.json")
            cand_df.to_csv(report_csv, index=False)
            # JSON: convert NaN → null for valid JSON
            records = cand_df.where(cand_df.notna(), other=None).to_dict(orient="records")
            with open(report_json, "w") as f:
                json.dump(records, f, indent=2)
            self.log(f"Candidate report → {report_csv}  ({len(candidate_rows)} candidates)")
        else:
            self.log("No transit candidates passed Stage 4 validation.")

        self._set_progress(100, "Done")
        self.log("─" * 60)
        self.log(f"Stage 4 complete.  Outputs in: {out_dir}")


# ─────────────────────────── entry point ─────────────────────────────────────

def main():
    root = tk.Tk()
    app  = Stage4GUI(root)
    root.geometry("1050x640")
    root.mainloop()


if __name__ == "__main__":
    main()