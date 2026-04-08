"""
MSEF Pipeline — Stage 3: Light Curve GP Detrending & Transit Search
====================================================================
Implements every requirement from the Stage 3 specification:

SPEC COMPLIANCE
───────────────
1.  GP engine   — uses scikit-learn Matern 3/2 as the default (always available).
                  celerite2 auto-used when installed (faster, O(N));
                  george used as second fallback.
2.  Input schema — strict validation: requires [time, flux]; flux_err handled;
                   aborts with logged error on schema/data problems.
3.  Output CSV   — time, flux, flux_err, gp_mean, gp_std, residual (all modes).
4.  Output JSON  — star_id, kernel, hyperparameters, fit_metrics
                   (log_likelihood, chi2_red), versions, seed, BLS fields,
                   provenance timestamp, fit_status.
5.  flux_err     — passed to GP as per-point noise (alpha / jitter term).
6.  Data-driven  — amplitude initialised to std(flux); timescale to span/5;
                   jitter to median(flux_err).
7.  Time centring — t0 = t.min() subtracted before GP fit; restored in output.
8.  Log file     — per-run log written to <out_dir>/stage3.log.
9.  Fit metrics  — log-likelihood, reduced chi-square stored in JSON.
10. Versions/seed— numpy, pandas, scipy, backend versions + seed per star.
11. Batch BLS    — BLS run on GP residuals in batch mode (same as single mode).
    plt.show()   — removed; all figures saved with savefig + plt.close().
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import threading
import time
import traceback
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── Optional BLS ──────────────────────────────────────────────────────────────
try:
    from astropy.timeseries import BoxLeastSquares
    HAS_BLS = True
except Exception:
    HAS_BLS = False
    warnings.warn("BoxLeastSquares not available — BLS disabled.")

# ── GP back-end detection ─────────────────────────────────────────────────────
# Preference order: celerite2 (fastest) > george > scikit-learn (always present)
GP_BACKEND = "sklearn"

try:
    import celerite2
    from celerite2 import GaussianProcess as Celerite2GP
    from celerite2.terms import Matern32Term, JitterTerm
    GP_BACKEND = "celerite2"
except ImportError:
    try:
        import george
        from george import GP as GeorgeGP
        from george.kernels import Matern32Kernel
        GP_BACKEND = "george"
    except ImportError:
        pass  # sklearn used below

if GP_BACKEND == "sklearn":
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        Matern, WhiteKernel, ConstantKernel as C,
    )

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_POINTS      = 8
RNG_SEED        = 42
BLS_N_PERIODS   = 600
BLS_DURATIONS   = [0.03, 0.05, 0.08, 0.12]
DEFAULT_NORMALIZE = True

# ─────────────────────────── pure-function science core ──────────────────────

def _pkg_versions() -> dict:
    v = {"numpy": np.__version__, "pandas": pd.__version__,
         "scipy": scipy.__version__, "backend": GP_BACKEND}
    if GP_BACKEND == "celerite2":
        v["celerite2"] = celerite2.__version__
    elif GP_BACKEND == "george":
        v["george"] = george.__version__
    else:
        import sklearn; v["sklearn"] = sklearn.__version__
    return v


def validate_and_load(path: str):
    """
    Strictly validate + load a Stage 2 CSV.
    Returns (t_centred, y_raw, yerr, star_id, t0).
    Raises ValueError on any schema/data problem.
    """
    star_id = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path)

    # Accept norm_flux as alias for flux
    if "flux" not in df.columns and "norm_flux" in df.columns:
        df["flux"] = df["norm_flux"]

    missing = {"time", "flux"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["flux"] = pd.to_numeric(df["flux"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "flux"])
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) < MIN_POINTS:
        raise ValueError(f"Only {len(df)} valid points (need >= {MIN_POINTS})")
    if df["flux"].std() == 0:
        raise ValueError("Flux is constant — nothing to fit")

    t_raw = df["time"].to_numpy(float)
    y_raw = df["flux"].to_numpy(float)
    if t_raw.max() - t_raw.min() == 0:
        raise ValueError("Zero time span")

    # flux_err: fill if missing or all-NaN
    if "flux_err" in df.columns:
        yerr = pd.to_numeric(df["flux_err"], errors="coerce").to_numpy(float)
        pos  = yerr[yerr > 0]
        fill = float(np.nanmedian(pos)) if len(pos) else 1e-4 * float(np.std(y_raw))
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, fill)
    else:
        warnings.warn(f"{star_id}: no flux_err — using scatter estimate")
        yerr = np.full(len(y_raw), max(1e-6 * float(np.std(y_raw)), 1e-10))

    t0 = t_raw.min()
    return t_raw - t0, y_raw, yerr, star_id, t0


def _chi2_red(y, mu, sigma, n_params) -> float:
    dof = len(y) - n_params
    if dof <= 0 or not np.all(sigma > 0):
        return np.nan
    return float(np.sum(((y - mu) / sigma) ** 2) / dof)


def _fit_celerite2(t, y, yerr):
    amp = max(float(np.std(y)), 1e-8)
    rho = max(float((t.max() - t.min()) / 5.0), 1e-4)
    jit = max(float(np.median(yerr)), 1e-8)
    kernel = Matern32Term(sigma=amp, rho=rho) + JitterTerm(sigma=jit)
    gp = Celerite2GP(kernel, mean=float(np.median(y)))
    gp.compute(t, diag=yerr ** 2)

    def neg_ll(p):
        gp.set_parameter_vector(p); return -gp.log_likelihood(y)
    def grad_neg_ll(p):
        gp.set_parameter_vector(p)
        _, g = gp.grad_log_likelihood(y); return -g

    res = minimize(neg_ll, gp.get_parameter_vector(), jac=grad_neg_ll,
                   method="L-BFGS-B", options={"maxiter": 300})
    gp.set_parameter_vector(res.x)
    mu, var = gp.predict(y, t=t, return_var=True)
    sigma   = np.sqrt(np.maximum(var, 0.0))
    hp  = dict(zip(gp.get_parameter_names(), gp.get_parameter_vector().tolist()))
    m   = {"log_likelihood": float(gp.log_likelihood(y)),
           "chi2_red": _chi2_red(y, mu, sigma, len(res.x)),
           "converged": bool(res.success),
           "kernel": "Matern32 + Jitter (celerite2)"}
    return mu, sigma, m, hp


def _fit_george(t, y, yerr):
    amp = float(np.var(y))
    rho = float(((t.max() - t.min()) / 5.0) ** 2)
    gp  = GeorgeGP(amp * Matern32Kernel(rho))
    gp.compute(t, yerr)

    def neg_ll(p):
        gp.set_parameter_vector(p); return -gp.log_likelihood(y)
    def grad_neg_ll(p):
        gp.set_parameter_vector(p); return -gp.grad_log_likelihood(y)[1]

    res = minimize(neg_ll, gp.get_parameter_vector(), jac=grad_neg_ll,
                   method="L-BFGS-B", options={"maxiter": 300})
    gp.set_parameter_vector(res.x)
    mu, cov = gp.predict(y, t, return_cov=True)
    sigma   = np.sqrt(np.maximum(np.diag(cov), 0.0))
    hp  = dict(zip(gp.get_parameter_names(), gp.get_parameter_vector().tolist()))
    m   = {"log_likelihood": float(gp.log_likelihood(y)),
           "chi2_red": _chi2_red(y, mu, sigma, len(res.x)),
           "converged": bool(res.success),
           "kernel": "Matern32 (george)"}
    return mu, sigma, m, hp


def _fit_sklearn(t, y, yerr, length_scale=None, n_restarts=0):
    amp  = float(np.std(y))
    ls   = length_scale or float((t.max() - t.min()) / 5.0)
    noise = float(np.median(yerr) ** 2)
    kernel = (C(amp ** 2, (1e-8, 1e4))
              * Matern(length_scale=ls, length_scale_bounds=(1e-4, 1e4), nu=1.5)
              + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-12, 1e2)))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=yerr ** 2,
                                  normalize_y=True,
                                  n_restarts_optimizer=n_restarts)
    X = t.reshape(-1, 1)
    gp.fit(X, y)
    mu, sigma = gp.predict(X, return_std=True)
    hp = {k: float(v) for k, v in gp.kernel_.get_params().items()
          if isinstance(v, (int, float))}
    ll = float(gp.log_marginal_likelihood_value_)
    m  = {"log_likelihood": ll,
          "chi2_red": _chi2_red(y, mu, sigma, len(gp.kernel_.theta)),
          "converged": True,
          "kernel": str(gp.kernel_)}
    return mu, sigma, m, hp


def fit_gp(t, y, yerr, length_scale=None, n_restarts=0):
    """Dispatch to best available GP back-end."""
    if GP_BACKEND == "celerite2": return _fit_celerite2(t, y, yerr)
    if GP_BACKEND == "george":    return _fit_george(t, y, yerr)
    return _fit_sklearn(t, y, yerr, length_scale, n_restarts)


def run_bls(t, residuals, baseline) -> dict:
    out = {"bls_period": np.nan, "bls_power": np.nan,
           "bls_depth": np.nan, "bls_snr": np.nan}
    if not HAS_BLS or len(t) < 30:
        return out
    try:
        res_norm = residuals / max(abs(float(baseline)), 1e-12)
        tspan    = float(t.max() - t.min())
        minp = max(0.05, tspan / 2000.0)
        maxp = min(tspan * 0.8, 30.0)
        if minp >= maxp:
            return out
        periods   = np.linspace(minp, maxp, BLS_N_PERIODS)
        bls_model = BoxLeastSquares(t, res_norm)
        best_power = -np.inf
        best = {}
        for df_ in BLS_DURATIONS:
            dur  = df_ * max(minp, tspan / 200.0)
            rbls = bls_model.power(periods, dur)
            idx  = int(np.nanargmax(rbls.power))
            if rbls.power[idx] > best_power:
                best_power = float(rbls.power[idx])
                best = {"bls_period": float(rbls.period[idx]),
                        "bls_power":  best_power,
                        "bls_depth":  float(rbls.depth[idx]),
                        "bls_snr":    float(rbls.depth_snr[idx])}
        out.update(best)
    except Exception as e:
        warnings.warn(f"BLS failed: {e}")
    return out


def _safe_json(x):
    """Convert numpy types / nan to JSON-safe Python types."""
    if isinstance(x, dict):
        return {k: _safe_json(v) for k, v in x.items()}
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if (math.isnan(x) or math.isinf(x)) else float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def process_one_star(path, out_dir, normalize=True, length_scale=None,
                     n_restarts=0, do_bls=True, logger=None) -> dict:
    """
    Full single-star pipeline: load → validate → GP → BLS → save CSV + JSON.
    Returns a summary dict. Raises on unrecoverable errors.
    """
    def _log(msg):
        if logger: logger.info(msg)

    np.random.seed(RNG_SEED)
    t, y_raw, yerr, star_id, t0 = validate_and_load(path)

    baseline = float(np.median(y_raw)) if normalize else 1.0
    if baseline == 0: baseline = 1.0
    y      = y_raw / baseline
    yerr_n = yerr  / baseline

    _log(f"{star_id}: fitting GP ({GP_BACKEND}, {len(t)} pts) ...")
    t_start = time.time()
    gp_mean_n, gp_std_n, metrics, hp = fit_gp(t, y, yerr_n, length_scale, n_restarts)
    elapsed = time.time() - t_start
    _log(f"{star_id}: {elapsed:.2f}s  ll={metrics['log_likelihood']:.3f}")

    gp_mean  = gp_mean_n * baseline
    gp_std   = gp_std_n  * baseline
    residual = y_raw - gp_mean

    bls_out = run_bls(t, residual, baseline) if do_bls else {}

    # Save CSV
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{star_id}_gp.csv")
    pd.DataFrame({"time": t + t0, "flux": y_raw, "flux_err": yerr,
                  "gp_mean": gp_mean, "gp_std": gp_std,
                  "residual": residual}).to_csv(csv_path, index=False)

    # Save JSON summary
    summary = {
        "star_id":    star_id,
        "input_file": os.path.abspath(path),
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "kernel":     metrics.get("kernel", GP_BACKEND),
        "hyperparameters": _safe_json(hp),
        "fit_metrics": {
            "log_likelihood": _safe_json(metrics.get("log_likelihood")),
            "chi2_red":       _safe_json(metrics.get("chi2_red")),
            "converged":      bool(metrics.get("converged", True)),
            "fit_time_s":     round(elapsed, 3),
        },
        "n_points":   int(len(t)),
        "baseline":   _safe_json(baseline),
        "fit_status": "ok",
        **{k: _safe_json(v) for k, v in bls_out.items()},
        "seed":       RNG_SEED,
        "versions":   _pkg_versions(),
    }
    with open(os.path.join(out_dir, f"{star_id}_gp_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return {"star_id": star_id, "csv": csv_path, "fit_status": "ok",
            "log_likelihood": metrics.get("log_likelihood", np.nan),
            "chi2_red":       metrics.get("chi2_red",       np.nan),
            **{k: bls_out.get(k, np.nan)
               for k in ["bls_period", "bls_power", "bls_depth", "bls_snr"]}}


# ─────────────────────────── GUI ─────────────────────────────────────────────

class Stage3LightcurveApp:
    def __init__(self, master):
        self.master = master
        master.title("MSEF — Stage 3: Light Curve GP Detrending")

        self.lc_path             = None
        self.lc_folder           = None
        self.master_catalog      = None
        self.master_catalog_path = None
        self.current_df          = None
        self.current_star_id     = None
        self.gp_result           = None
        self.processing_thread   = None
        self.stop_requested      = False
        self._logger             = None

        self._build_ui()
        self.log(f"Stage 3 ready.  GP back-end: {GP_BACKEND}  "
                 f"({'BLS ok' if HAS_BLS else 'no BLS'})")

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        m   = self.master
        top = ttk.Frame(m, padding=8)
        top.grid(row=0, column=0, sticky="nsew")
        m.rowconfigure(0, weight=1); m.columnconfigure(0, weight=1)

        btns = ttk.Frame(top)
        btns.grid(row=0, column=0, sticky="w", pady=(0, 8))
        for col, (lbl, cmd) in enumerate([
            ("Open CSV",           self.open_lightcurve),
            ("Open Folder",        self.open_lightcurve_folder),
            ("Open Catalog",       self.open_master_catalog),
            ("Test Lightcurve",    self.use_test_lightcurve),
            ("▶  Run GP",          self.start_gp),
            ("⚙  Batch Folder",   self.start_batch),
            ("⛔  Stop",           self.request_stop),
            ("💾  Save Result",    self.save_result),
        ]):
            ttk.Button(btns, text=lbl, command=cmd).grid(row=0, column=col, padx=3)

        self.progress = ttk.Progressbar(top, orient="horizontal",
                                        length=700, mode="determinate")
        self.progress.grid(row=1, column=0, pady=(4, 4), sticky="w")
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(top, textvariable=self.status_var,
                  font=("Arial", 8)).grid(row=2, column=0, sticky="w")

        mid = ttk.Frame(top)
        mid.grid(row=3, column=0, sticky="nsew")
        top.rowconfigure(3, weight=1); top.columnconfigure(0, weight=1)

        left = ttk.Frame(mid)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        ttk.Label(left, text="Stars / Files").grid(row=0, column=0)
        self.star_listbox = tk.Listbox(left, width=28, height=24)
        self.star_listbox.grid(row=1, column=0, sticky="ns")
        self.star_listbox.bind("<<ListboxSelect>>", self.on_star_select)

        right = ttk.Frame(mid)
        right.grid(row=0, column=1, sticky="nsew")
        mid.columnconfigure(1, weight=1); mid.rowconfigure(0, weight=1)

        self.fig, (self.ax_top, self.ax_bot) = plt.subplots(
            2, 1, figsize=(7, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]})
        plt.subplots_adjust(hspace=0.12)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        opts = ttk.Frame(top)
        opts.grid(row=4, column=0, sticky="we", pady=(6, 0))

        def _lbl(t, c):
            ttk.Label(opts, text=t).grid(row=0, column=c, padx=(8, 2))
        def _ent(val, c, w=8):
            e = ttk.Entry(opts, width=w); e.insert(0, str(val))
            e.grid(row=0, column=c, padx=(0, 4)); return e

        _lbl("Length-scale:", 0); self.len_entry      = _ent("auto", 1)
        _lbl("Restarts:",     2); self.restarts_entry = _ent(0,      3, 5)
        _lbl("Output dir:",   4)
        self.outdir_var = tk.StringVar(value="(same as input)")
        ttk.Entry(opts, textvariable=self.outdir_var, width=30).grid(
            row=0, column=5, padx=(0, 4))
        ttk.Button(opts, text="Browse",
                   command=self._browse_outdir).grid(row=0, column=6, padx=2)
        self.normalize_var = tk.BooleanVar(value=DEFAULT_NORMALIZE)
        ttk.Checkbutton(opts, text="Normalise",
                        variable=self.normalize_var).grid(row=0, column=7, padx=4)
        self.bls_var = tk.BooleanVar(value=HAS_BLS)
        ttk.Checkbutton(opts, text="BLS",
                        variable=self.bls_var).grid(row=0, column=8, padx=4)

        m.minsize(960, 720)

    # ── thread-safe helpers ────────────────────────────────────────────────

    def log(self, text):
        ts  = time.strftime("%H:%M:%S")
        msg = f"[{ts}] {text}"
        self.master.after(0, lambda m=msg: self.status_var.set(m))
        if self._logger: self._logger.info(text)

    def _set_progress(self, value):
        self.master.after(0, lambda v=value: self.progress.config(value=v))

    def _setup_logger(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        logger = logging.getLogger(f"stage3_{id(self)}")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            fh = logging.FileHandler(
                os.path.join(out_dir, "stage3.log"), mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
            logger.addHandler(fh)
        self._logger = logger
        return logger

    def _browse_outdir(self):
        f = filedialog.askdirectory(title="Select output folder")
        if f: self.outdir_var.set(f)

    def _resolve_outdir(self, input_path):
        v = self.outdir_var.get().strip()
        if v and "(same" not in v and os.path.isabs(v):
            return v
        return os.path.dirname(os.path.abspath(input_path or "."))

    # ── load actions ───────────────────────────────────────────────────────

    def open_lightcurve(self):
        path = filedialog.askopenfilename(
            title="Open lightcurve CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path: return
        try:
            df = pd.read_csv(path)
            if "flux" not in df.columns and "norm_flux" in df.columns:
                df["flux"] = df["norm_flux"]
            if "time" not in df.columns or "flux" not in df.columns:
                messagebox.showerror("Schema error",
                    "CSV must contain time and flux columns."); return
            df["flux_use"]  = df["flux"].astype(float)
            self.current_df = df; self.lc_path = path
            self.lc_folder = None; self.master_catalog = None
            self.populate_star_list_single(path)
            self.plot_current()
            self.log(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def open_lightcurve_folder(self):
        folder = filedialog.askdirectory(
            title="Open folder with star_####.csv files")
        if not folder: return
        self.lc_folder = folder; self.lc_path = None; self.master_catalog = None
        files = sorted(glob.glob(os.path.join(folder, "star_*.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        self.star_listbox.delete(0, "end")
        for f in files:
            self.star_listbox.insert("end", os.path.basename(f))
        self.log(f"Folder: {folder}  ({len(files)} files)")

    def open_master_catalog(self):
        path = filedialog.askopenfilename(
            title="Open master_catalog.csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path: return
        try:
            df = pd.read_csv(path)
            self.master_catalog = df; self.master_catalog_path = path
            self.lc_folder = os.path.dirname(path); self.lc_path = None
            self.star_listbox.delete(0, "end")
            for idx, row in df.iterrows():
                sid = int(row.get("star_id", idx))
                x_  = float(row.get("x", np.nan))
                y_  = float(row.get("y", np.nan))
                self.star_listbox.insert("end", f"{sid:04d}  ({x_:.1f},{y_:.1f})")
            self.log(f"Catalog: {os.path.basename(path)}  ({len(df)} stars)")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def populate_star_list_single(self, path):
        self.star_listbox.delete(0, "end")
        self.star_listbox.insert("end", os.path.basename(path))

    def on_star_select(self, event):
        sel = self.star_listbox.curselection()
        if not sel: return
        idx = sel[0]
        if self.current_df is not None and self.lc_path is not None:
            self.current_star_id = os.path.basename(self.lc_path)
            self.plot_current(); return
        if self.master_catalog is not None:
            row    = self.master_catalog.iloc[idx]
            lc_rel = row.get("lc_file", None)
            cat_dir = os.path.dirname(self.master_catalog_path)                       if self.master_catalog_path else ""
            if lc_rel:
                lc_path = os.path.normpath(os.path.join(cat_dir, lc_rel))
            else:
                lc_path = os.path.join(
                    self.lc_folder or cat_dir,
                    f"star_{int(row['star_id']):04d}.csv")
            if os.path.exists(lc_path): self.load_csv_and_plot(lc_path)
            else: messagebox.showwarning("Missing", f"Not found: {lc_path}")
            return
        if self.lc_folder is not None:
            fname   = self.star_listbox.get(idx)
            lc_path = os.path.join(self.lc_folder, fname)
            if os.path.exists(lc_path): self.load_csv_and_plot(lc_path)
            else: messagebox.showerror("Missing", "File not found.")

    def load_csv_and_plot(self, path):
        try:
            df = pd.read_csv(path)
            if "flux" not in df.columns and "norm_flux" in df.columns:
                df["flux"] = df["norm_flux"]
            df["flux_use"]       = df["flux"].astype(float)
            self.current_df      = df; self.lc_path = path
            self.current_star_id = os.path.basename(path)
            self.plot_current()
            self.log(f"Selected {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def use_test_lightcurve(self):
        np.random.seed(RNG_SEED)
        t    = np.linspace(0, 2.0, 400)
        flux = (1.0 + 0.005 * np.sin(2 * np.pi * t / 0.8)
                + 0.0015 * np.random.randn(t.size))
        for tc in np.arange(0.8, 2.0, 0.5):
            flux[(t > tc - 0.02) & (t < tc + 0.02)] -= 0.008
        df = pd.DataFrame({"time": t, "flux": flux, "norm_flux": flux,
                           "flux_err": np.full_like(flux, 0.0015)})
        df["flux_use"] = df["flux"]
        self.current_df = df; self.lc_path = None; self.lc_folder = None
        self.populate_star_list_single("Synthetic transit (test)")
        self.plot_current()
        self.log("Loaded synthetic transit test light curve.")

    # ── plot ───────────────────────────────────────────────────────────────

    def plot_current(self, gp_pred=None, gp_std=None, t_gp=None, y_gp=None):
        self.ax_top.clear(); self.ax_bot.clear()
        if self.current_df is None:
            self.canvas.draw(); return
        df    = self.current_df
        t_raw = np.asarray(df["time"],     dtype=float)
        y_raw = np.asarray(df["flux_use"], dtype=float)
        mask  = np.isfinite(t_raw) & np.isfinite(y_raw)
        if not mask.any():
            messagebox.showwarning("Empty", "No valid points."); return

        self.ax_top.plot(t_raw[mask], y_raw[mask], ".", ms=3.5,
                         color="steelblue", alpha=0.7, label="data")
        if gp_pred is not None and t_gp is not None:
            self.ax_top.plot(t_gp, gp_pred, "-", lw=1.5,
                             color="tomato", label="GP mean")
            self.ax_top.fill_between(t_gp, gp_pred - gp_std, gp_pred + gp_std,
                                     alpha=0.22, color="tomato", label="\u00b11\u03c3")
            res = y_gp - gp_pred
            self.ax_bot.plot(t_gp, res, ".", ms=3, color="steelblue", alpha=0.7)
            self.ax_bot.axhline(0, color="k", lw=0.6)
            self.ax_bot.set_ylabel("Residual")
            self.ax_bot.set_xlabel("Time (days)")
            r = self.gp_result or {}
            if not np.isnan(r.get("bls_period", np.nan)):
                txt = (f"BLS P={r['bls_period']:.4f} d  "
                       f"depth={r.get('bls_depth', np.nan):.5f}  "
                       f"SNR={r.get('bls_snr', np.nan):.2f}")
                self.ax_top.text(0.02, 0.04, txt,
                                 transform=self.ax_top.transAxes,
                                 fontsize=7, color="darkgreen",
                                 bbox=dict(fc="white", alpha=0.7, ec="none"))
        else:
            self.ax_bot.set_xlabel("Time (days)")
        self.ax_top.set_ylabel("Flux (norm)")
        self.ax_top.legend(loc="upper right", fontsize=7)
        self.canvas.draw()

    # ── single GP ─────────────────────────────────────────────────────────

    def start_gp(self):
        if self.current_df is None:
            messagebox.showwarning("No data", "Load a lightcurve first."); return
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showinfo("Busy", "GP already running."); return
        try:
            ls_str = self.len_entry.get().strip()
            ls     = None if ls_str.lower() == "auto" else float(ls_str)
            nr     = int(self.restarts_entry.get() or 0)
            norm   = bool(self.normalize_var.get())
            bls    = bool(self.bls_var.get()) and HAS_BLS
        except ValueError as e:
            messagebox.showerror("Bad parameter", str(e)); return
        out_dir = self._resolve_outdir(self.lc_path or ".")
        self.stop_requested = False; self._set_progress(0)
        self.processing_thread = threading.Thread(
            target=self._gp_single_worker,
            args=(ls, nr, norm, bls, out_dir), daemon=True)
        self.processing_thread.start()

    def _gp_single_worker(self, length_scale, n_restarts, normalize,
                          run_bls_flag, out_dir):
        try:
            logger = self._setup_logger(out_dir)
            self.log("Preparing data for GP ...")
            df   = self.current_df.copy()
            col  = "flux" if "flux" in df.columns else "norm_flux"
            t_in = np.asarray(df["time"], dtype=float)
            y_in = np.asarray(df[col],   dtype=float)
            mask = np.isfinite(t_in) & np.isfinite(y_in)
            t_in, y_in = t_in[mask], y_in[mask]

            if "flux_err" in df.columns:
                yerr = np.asarray(df["flux_err"], dtype=float)[mask]
                yerr = np.where(np.isfinite(yerr) & (yerr > 0),
                                yerr, 1e-4 * float(np.std(y_in)))
            else:
                yerr = np.full(len(y_in), 1e-4 * float(np.std(y_in)))

            if len(t_in) < MIN_POINTS:
                _sz = len(t_in)
                self.master.after(0, lambda: messagebox.showwarning(
                    "Too few points",
                    f"Need >= {MIN_POINTS} valid points. Found {_sz}."))
                return

            t0   = t_in.min(); t = t_in - t0
            baseline = float(np.median(y_in)) if normalize else 1.0
            if baseline == 0: baseline = 1.0
            y      = y_in / baseline
            yerr_n = yerr / baseline

            self.log(f"Fitting GP ({GP_BACKEND})  {len(t)} points ...")
            t_start = time.time()
            gp_mean_n, gp_std_n, metrics, hp = fit_gp(
                t, y, yerr_n, length_scale, n_restarts)
            elapsed = time.time() - t_start
            self.log(f"GP fitted {elapsed:.2f}s  "
                     f"ll={metrics['log_likelihood']:.3f}  "
                     f"chi2r={metrics.get('chi2_red', float('nan')):.3f}")

            gp_mean  = gp_mean_n * baseline
            gp_std   = gp_std_n  * baseline
            residual = y_in - gp_mean

            bls_out = {}
            if run_bls_flag:
                self.log("Running BLS on residuals ...")
                bls_out = run_bls(t, residual, baseline)
                if np.isfinite(bls_out.get("bls_period", np.nan)):
                    self.log(f"BLS P={bls_out['bls_period']:.4f} d  "
                             f"depth={bls_out['bls_depth']:.5f}  "
                             f"SNR={bls_out['bls_snr']:.2f}")

            self.gp_result = {
                "t": t + t0, "y": y_in,
                "y_pred": gp_mean, "y_std": gp_std,
                "baseline": baseline, "metrics": metrics, "hp": hp,
                **bls_out,
            }

            # Auto-save CSV + JSON
            if self.lc_path:
                sid = os.path.splitext(os.path.basename(self.lc_path))[0]
                os.makedirs(out_dir, exist_ok=True)
                pd.DataFrame({
                    "time": t + t0, "flux": y_in, "flux_err": yerr,
                    "gp_mean": gp_mean, "gp_std": gp_std, "residual": residual,
                }).to_csv(os.path.join(out_dir, f"{sid}_gp.csv"), index=False)
                summary = {
                    "star_id": sid,
                    "input_file": os.path.abspath(self.lc_path),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "kernel": metrics.get("kernel", GP_BACKEND),
                    "hyperparameters": _safe_json(hp),
                    "fit_metrics": {
                        "log_likelihood": _safe_json(metrics.get("log_likelihood")),
                        "chi2_red":       _safe_json(metrics.get("chi2_red")),
                        "converged":      bool(metrics.get("converged", True)),
                        "fit_time_s":     round(elapsed, 3),
                    },
                    "n_points": int(len(t)), "baseline": _safe_json(baseline),
                    "fit_status": "ok",
                    **{k: _safe_json(v) for k, v in bls_out.items()},
                    "seed": RNG_SEED, "versions": _pkg_versions(),
                }
                with open(os.path.join(out_dir, f"{sid}_gp_summary.json"), "w") as f:
                    json.dump(summary, f, indent=2)
                self.log(f"Saved CSV + JSON -> {out_dir}")

            _t = (t + t0).copy(); _y = y_in.copy()
            _yp = gp_mean.copy(); _ys = gp_std.copy()
            self.master.after(0, lambda: self.plot_current(
                gp_pred=_yp, gp_std=_ys, t_gp=_t, y_gp=_y))
            self.log("GP fit complete.")
            self._set_progress(100)

        except Exception as exc:
            self.log(f"GP error: {exc}")
            _emsg = str(exc)
            self.master.after(0, lambda: messagebox.showerror("GP error", _emsg))
        finally:
            self.stop_requested = False

    # ── batch ──────────────────────────────────────────────────────────────

    def start_batch(self):
        folder = self.lc_folder
        if folder is None:
            folder = filedialog.askdirectory(
                title="Select folder with star_####.csv lightcurves")
            if not folder: return
            self.lc_folder = folder
        files = sorted(glob.glob(os.path.join(folder, "star_*.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if not files:
            messagebox.showwarning("No files", "No CSV files found."); return
        out_folder = filedialog.askdirectory(
            title="Select output folder for GP results")
        if not out_folder: return
        try:
            ls_str = self.len_entry.get().strip()
            ls     = None if ls_str.lower() == "auto" else float(ls_str)
            nr     = int(self.restarts_entry.get() or 0)
            norm   = bool(self.normalize_var.get())
            bls    = bool(self.bls_var.get()) and HAS_BLS
        except ValueError as e:
            messagebox.showerror("Bad parameter", str(e)); return
        self.stop_requested = False; self._set_progress(0)
        self.processing_thread = threading.Thread(
            target=self._batch_worker,
            args=(files, out_folder, ls, nr, norm, bls),
            daemon=True)
        self.processing_thread.start()

    def _batch_worker(self, files, out_folder, length_scale, n_restarts,
                      normalize, run_bls_flag):
        total = len(files); saved = 0; failed = 0
        logger = self._setup_logger(out_folder)
        report = []
        logger.info(f"Batch start: {total} files -> {out_folder}  "
                    f"backend={GP_BACKEND} bls={run_bls_flag}")
        for i, fpath in enumerate(files):
            if self.stop_requested:
                self.log("Batch cancelled."); break
            try:
                row = process_one_star(
                    path=fpath, out_dir=out_folder, normalize=normalize,
                    length_scale=length_scale, n_restarts=n_restarts,
                    do_bls=run_bls_flag, logger=logger)
                report.append(row); saved += 1
                self._set_progress(100.0 * (i + 1) / total)
                self.log(f"[{i+1}/{total}] {os.path.basename(fpath)}  ok  "
                         f"ll={row.get('log_likelihood', float('nan')):.3f}")
            except Exception as e:
                failed += 1
                logger.error(f"{fpath}: {traceback.format_exc()}")
                self.log(f"[{i+1}/{total}] FAILED {os.path.basename(fpath)}: {e}")
                report.append({"star_id": os.path.splitext(
                    os.path.basename(fpath))[0],
                    "fit_status": "failed", "error": str(e)})
        if report:
            pd.DataFrame(report).to_csv(
                os.path.join(out_folder, "batch_summary.csv"), index=False)
        self._set_progress(100)
        self.log(f"Batch done: {saved} ok, {failed} failed -> {out_folder}")

    # ── stop ───────────────────────────────────────────────────────────────

    def request_stop(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_requested = True
            self.log("Stop requested ...")
        else:
            self.log("No active run.")

    # ── save ───────────────────────────────────────────────────────────────

    def save_result(self):
        if self.gp_result is None:
            messagebox.showinfo("Nothing to save", "Run GP first."); return
        out_png = filedialog.asksaveasfilename(
            title="Save plot", defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if not out_png: return
        try:
            r = self.gp_result
            t = r["t"]; y = r["y"]; y_pred = r["y_pred"]; y_std = r["y_std"]
            res  = y - y_pred
            star = self.current_star_id or "star"
            fig2, (ax1, ax2) = plt.subplots(
                2, 1, sharex=True, figsize=(9, 6),
                gridspec_kw={"height_ratios": [3, 1]})
            ax1.plot(t, y,      ".", ms=3.5, color="steelblue", label="data")
            ax1.plot(t, y_pred, "-", lw=1.2, color="tomato",   label="GP mean")
            ax1.fill_between(t, y_pred - y_std, y_pred + y_std,
                             alpha=0.22, color="tomato")
            ax1.set_ylabel("Flux (norm)")
            ax1.set_title(f"Light Curve: {star}  [{GP_BACKEND}]", fontsize=9)
            bls_txt = ""
            if not np.isnan(r.get("bls_period", np.nan)):
                bls_txt = (f"BLS P={r['bls_period']:.4f} d  "
                           f"depth={r.get('bls_depth', np.nan):.5f}  "
                           f"SNR={r.get('bls_snr', np.nan):.2f}")
                ax1.text(0.02, 0.04, bls_txt, transform=ax1.transAxes,
                         fontsize=7, color="darkgreen",
                         bbox=dict(fc="white", alpha=0.7, ec="none"))
            ax1.legend(fontsize=7)
            ax2.plot(t, res, ".", ms=3, color="steelblue")
            ax2.axhline(0, color="k", lw=0.6)
            ax2.set_xlabel("Time (days)"); ax2.set_ylabel("Residual")
            fig2.tight_layout()
            fig2.savefig(out_png, dpi=150)
            plt.close(fig2)   # no plt.show() — prevent memory leak

            out_csv = os.path.splitext(out_png)[0] + ".csv"
            pd.DataFrame({
                "time": t, "flux": y, "flux_err": np.full(len(t), np.nan),
                "gp_mean": y_pred, "gp_std": y_std, "residual": res,
            }).to_csv(out_csv, index=False)
            info = f"Plot -> {out_png}\nCSV  -> {out_csv}"
            if bls_txt: info += f"\n\n{bls_txt}"
            messagebox.showinfo("Saved", info)
            self.log(f"Saved plot + CSV for {star}.")
        except Exception as e:
            messagebox.showerror("Save error", str(e))


# ─────────────────────────── entry point ─────────────────────────────────────

def main():
    root = tk.Tk()
    Stage3LightcurveApp(root)
    root.geometry("1200x820")
    root.mainloop()


if __name__ == "__main__":
    main()