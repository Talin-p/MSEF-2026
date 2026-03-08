import os
import threading
import time
import glob
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Optional BLS
try:
    from astropy.timeseries import BoxLeastSquares
    HAS_BLS = True
except Exception:
    HAS_BLS = False
    warnings.warn("astropy.timeseries.BoxLeastSquares not available. BLS disabled.")

# Defaults / Tunables
DEFAULT_KERNEL = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
DEFAULT_N_RESTARTS = 0          # keep small for speed; set >0 to optimize hyperparams more
DEFAULT_NORMALIZE = True        # normalize flux before GP fit (divide by median)
MIN_POINTS_FOR_GP = 8


class Stage3LightcurveApp:
    def __init__(self, master):
        self.master = master
        master.title("MSEF — Stage 3 (Lightcurve GPR)")

        # state
        self.lc_path = None
        self.lc_folder = None
        self.master_catalog = None
        self.current_df = None
        self.current_star_id = None
        self.gp_result = None
        self.processing_thread = None
        self.stop_requested = False

        # UI layout
        top = ttk.Frame(master, padding=8)
        top.grid(row=0, column=0, sticky="nsew")
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)

        # Buttons
        btns = ttk.Frame(top)
        btns.grid(row=0, column=0, sticky="w", pady=(0,8))

        ttk.Button(btns, text="Open Lightcurve CSV", command=self.open_lightcurve).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Open Lightcurve Folder", command=self.open_lightcurve_folder).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="Open master_catalog.csv", command=self.open_master_catalog).grid(row=0, column=2, padx=4)
        ttk.Button(btns, text="Use Test Lightcurve", command=self.use_test_lightcurve).grid(row=0, column=3, padx=4)

        self.run_btn = ttk.Button(btns, text="Run GP Fit", command=self.start_gp)
        self.run_btn.grid(row=0, column=4, padx=4)
        self.batch_btn = ttk.Button(btns, text="Run GP on Folder (Batch)", command=self.start_batch)
        self.batch_btn.grid(row=0, column=5, padx=4)
        ttk.Button(btns, text="Stop", command=self.request_stop).grid(row=0, column=6, padx=4)
        ttk.Button(btns, text="Save Plot / Result", command=self.save_result).grid(row=0, column=7, padx=4)

        # Progress & status
        self.progress = ttk.Progressbar(top, orient="horizontal", length=600, mode="determinate")
        self.progress.grid(row=1, column=0, pady=(4,8), sticky="w")
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(top, textvariable=self.status_var).grid(row=2, column=0, sticky="w")

        # Middle: star list (if folder loaded) and plot area
        mid = ttk.Frame(top)
        mid.grid(row=3, column=0, sticky="nsew")
        top.rowconfigure(3, weight=1)
        top.columnconfigure(0, weight=1)

        left = ttk.Frame(mid)
        left.grid(row=0, column=0, sticky="ns", padx=(0,8))
        ttk.Label(left, text="Stars / Files").grid(row=0, column=0)
        self.star_listbox = tk.Listbox(left, width=30, height=25)
        self.star_listbox.grid(row=1, column=0, sticky="ns")
        self.star_listbox.bind("<<ListboxSelect>>", self.on_star_select)

        right = ttk.Frame(mid)
        right.grid(row=0, column=1, sticky="nsew")
        mid.columnconfigure(1, weight=1)
        mid.rowconfigure(0, weight=1)

        # Matplotlib figure: top = data+GP, bottom = residuals
        self.fig, (self.ax_top, self.ax_bot) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                                           gridspec_kw={"height_ratios": [3, 1]})
        plt.subplots_adjust(hspace=0.15)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # GP options frame
        opts = ttk.Frame(top)
        opts.grid(row=4, column=0, sticky="we", pady=(6,0))
        ttk.Label(opts, text="Kernel length-scale (initial):").grid(row=0, column=0, padx=(0,4))
        self.len_entry = ttk.Entry(opts, width=8)
        self.len_entry.insert(0, "1.0")
        self.len_entry.grid(row=0, column=1, padx=(0,8))

        ttk.Label(opts, text="n_restarts_optimizer:").grid(row=0, column=2, padx=(0,4))
        self.restarts_entry = ttk.Entry(opts, width=6)
        self.restarts_entry.insert(0, str(DEFAULT_N_RESTARTS))
        self.restarts_entry.grid(row=0, column=3, padx=(0,8))

        self.normalize_var = tk.BooleanVar(value=DEFAULT_NORMALIZE)
        ttk.Checkbutton(opts, text="Normalize flux (median)", variable=self.normalize_var).grid(row=0, column=4, padx=(0,8))

        self.bls_var = tk.BooleanVar(value=HAS_BLS)
        ttk.Checkbutton(opts, text="Run BLS on residuals (if available)", variable=self.bls_var).grid(row=0, column=5, padx=(0,8))

        # set minimum window size
        master.minsize(900, 700)

        self.log("Stage 3 ready.")

    # Helpers
    def log(self, text):
        t = time.strftime("%H:%M:%S")
        self.status_var.set(f"[{t}] {text}")
        self.master.update_idletasks()

    def open_lightcurve(self):
        path = filedialog.askopenfilename(
            title="Open lightcurve CSV (time, flux[, ...])",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            if "time" not in df.columns or ("flux" not in df.columns and "norm_flux" not in df.columns):
                messagebox.showerror("Bad file", "CSV must contain 'time' and either 'flux' or 'norm_flux' columns.")
                return
            # prefer 'norm_flux' if present
            if "norm_flux" in df.columns:
                df["flux_use"] = df["norm_flux"].astype(float)
            else:
                df["flux_use"] = df["flux"].astype(float)
            self.current_df = df
            self.lc_path = path
            self.lc_folder = None
            self.master_catalog = None
            self.populate_star_list_single(path)
            self.plot_current()
            self.log(f"Loaded lightcurve: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def open_lightcurve_folder(self):
        folder = filedialog.askdirectory(title="Open folder containing star_####.csv files")
        if not folder:
            return
        self.lc_folder = folder
        self.lc_path = None
        self.master_catalog = None
        files = sorted(glob.glob(os.path.join(folder, "star_*.csv")))
        if not files:
            # accept any csv as fallback
            files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        self.star_listbox.delete(0, "end")
        for f in files:
            self.star_listbox.insert("end", os.path.basename(f))
        self.log(f"Loaded folder: {folder} ({len(files)} files)")

    def open_master_catalog(self):
        path = filedialog.askopenfilename(title="Open master_catalog.csv", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
            # expect columns star_id and lc_file (or x,y)
            self.master_catalog = df
            self.master_catalog_path = path
            self.lc_folder = os.path.dirname(path)
            self.lc_path = None
            # populate listbox with entries for user selection
            self.star_listbox.delete(0, "end")
            for idx, row in df.iterrows():
                label = f"{int(row.get('star_id', idx)):04d}  ({row.get('x',np.nan):.1f},{row.get('y',np.nan):.1f})"
                self.star_listbox.insert("end", label)
            self.log(f"Loaded master catalog: {os.path.basename(path)} ({len(df)} entries)")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def populate_star_list_single(self, path):
        self.star_listbox.delete(0, "end")
        self.star_listbox.insert("end", os.path.basename(path))

    def on_star_select(self, event):
        sel = self.star_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        # determine source path
        if self.current_df is not None and self.lc_path is not None:
            # single file selected
            self.current_star_id = os.path.basename(self.lc_path)
            self.plot_current()
            return
        if self.master_catalog is not None:
            # catalog selection
            row = self.master_catalog.iloc[idx]
            lc_rel = row.get("lc_file", None)
            if lc_rel:
                lc_path = os.path.join(os.path.dirname(self.master_catalog_path if hasattr(self, "master_catalog_path") else ""), lc_rel)
                if not os.path.exists(lc_path):
                    # try relative to the catalog folder
                    lc_path = os.path.join(os.path.dirname(self.master_catalog_path) if hasattr(self, "master_catalog_path") else "", lc_rel)
                if os.path.exists(lc_path):
                    self.load_csv_and_plot(lc_path)
                else:
                    messagebox.showwarning("Missing file", f"Lightcurve file not found: {lc_rel}")
            else:
                # if master catalog doesn't include lc_file, just try to match star_* files in folder
                folder = self.lc_folder or os.getcwd()
                candidate = os.path.join(folder, f"star_{int(row['star_id']):04d}.csv")
                if os.path.exists(candidate):
                    self.load_csv_and_plot(candidate)
                else:
                    messagebox.showwarning("Missing file", f"Can't locate lightcurve for star {int(row.get('star_id', idx))}")
            return
        if self.lc_folder is not None:
            # folder list selection
            filename = self.star_listbox.get(idx)
            lc_path = os.path.join(self.lc_folder, filename)
            if os.path.exists(lc_path):
                self.load_csv_and_plot(lc_path)
            else:
                messagebox.showerror("Missing file", "Selected file not found.")
            return

    def load_csv_and_plot(self, path):
        try:
            df = pd.read_csv(path)
            if "norm_flux" in df.columns:
                df["flux_use"] = df["norm_flux"].astype(float)
            elif "flux" in df.columns:
                df["flux_use"] = df["flux"].astype(float)
            else:
                messagebox.showerror("Bad file", "CSV must contain 'flux' or 'norm_flux'")
                return
            self.current_df = df
            self.lc_path = path
            self.current_star_id = os.path.basename(path)
            self.plot_current()
            self.log(f"Selected {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def use_test_lightcurve(self):
        # create a synthetic transit + noise
        t = np.linspace(0, 1.0, 200)
        flux = np.ones_like(t) + 0.002 * np.random.randn(t.size)
        # add a synthetic box transit at t~0.4
        intransit = (t > 0.39) & (t < 0.42)
        flux[intransit] -= 0.01
        df = pd.DataFrame({"time": t, "flux": flux, "norm_flux": flux})
        df["flux_use"] = df["norm_flux"]
        self.current_df = df
        self.lc_path = None
        self.lc_folder = None
        self.populate_star_list_single("Test lightcurve")
        self.plot_current()
        self.log("Loaded synthetic test lightcurve.")

    def plot_current(self, gp_pred=None, gp_std=None):
        self.ax_top.clear()
        self.ax_bot.clear()
        if self.current_df is None:
            self.canvas.draw()
            return
        df = self.current_df
        # ensure arrays
        t = np.asarray(df["time"], dtype=float)
        y = np.asarray(df["flux_use"], dtype=float)
        mask = np.isfinite(t) & np.isfinite(y)
        if np.sum(mask) == 0:
            messagebox.showwarning("Empty data", "No valid time/flux points to plot.")
            return
        t = t[mask]
        y = y[mask]
        self.ax_top.plot(t, y, ".", ms=4, label="data")
        if gp_pred is not None:
            self.ax_top.plot(t, gp_pred, "-", lw=1.5, label="GP mean")
            # shaded 1-sigma
            lower = gp_pred - gp_std
            upper = gp_pred + gp_std
            self.ax_top.fill_between(t, lower, upper, alpha=0.25, label="±1σ")
        self.ax_top.set_ylabel("Flux (norm)")
        self.ax_top.legend(loc="upper right")
        # residuals
        if gp_pred is not None:
            res = y - gp_pred
            self.ax_bot.plot(t, res, ".", ms=3)
            self.ax_bot.axhline(0, color="k", lw=0.6)
            self.ax_bot.set_ylabel("Residual")
            self.ax_bot.set_xlabel("Time")
        else:
            self.ax_bot.set_xlabel("Time")
        self.canvas.draw()

    # GP worker
    def start_gp(self):
        if self.current_df is None:
            messagebox.showwarning("No data", "Load a lightcurve first.")
            return
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showinfo("Busy", "A GP run is already in progress.")
            return
        # read options
        try:
            length_scale = float(self.len_entry.get())
        except Exception:
            length_scale = 1.0
        try:
            n_restarts = int(self.restarts_entry.get())
        except Exception:
            n_restarts = DEFAULT_N_RESTARTS
        normalize = bool(self.normalize_var.get())
        run_bls = bool(self.bls_var.get()) and HAS_BLS

        self.stop_requested = False
        self.progress["value"] = 0
        self.processing_thread = threading.Thread(target=self._gp_worker,
                                                  args=(length_scale, n_restarts, normalize, run_bls),
                                                  daemon=True)
        self.processing_thread.start()

    def _gp_worker(self, length_scale, n_restarts, normalize, run_bls):
        try:
            self.log("Preparing data for GP...")
            df = self.current_df.copy()
            t = np.asarray(df["time"], dtype=float)
            y = np.asarray(df["flux_use"], dtype=float)
            mask = np.isfinite(t) & np.isfinite(y)
            t = t[mask]
            y = y[mask]
            if t.size < MIN_POINTS_FOR_GP:
                messagebox.showwarning("Too few points", f"Need at least {MIN_POINTS_FOR_GP} valid points for GP. Found {t.size}.")
                return

            # optionally normalize
            baseline = np.median(y) if normalize else 1.0
            y_norm = y / baseline

            # Prepare X for sklearn (n_samples, n_features)
            X = t.reshape(-1, 1)

            # Kernel: Constant * RBF + WhiteKernel
            kernel = C(1.0, (1e-6, 1e6)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-4, 1e4)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-12, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=n_restarts)

            self.log(f"Training GP on {X.shape[0]} points (length_scale={length_scale}, restarts={n_restarts}) ...")
            t0 = time.time()
            gp.fit(X, y_norm)
            t1 = time.time()
            self.log(f"GP trained in {t1 - t0:.2f} s. Predicting...")
            # prediction (we predict at the observed times for plotting)
            y_pred, y_std = gp.predict(X, return_std=True)
            # convert back to original scale
            y_pred_scaled = y_pred * baseline
            y_std_scaled = y_std * baseline

            # save results to state
            self.gp_result = {"gp": gp, "t": t, "y": y, "y_pred": y_pred_scaled, "y_std": y_std_scaled, "baseline": baseline}

            # update plot (on main thread)
            self.master.after(0, lambda: self.plot_current(gp_pred=y_pred_scaled, gp_std=y_std_scaled))

            # optionally run BLS on residuals
            if run_bls:
                try:
                    self.log("Running BLS on residuals...")
                    res = y - y_pred_scaled
                    # BLS expects time in days and flux normalized around zero baseline:
                    bls = BoxLeastSquares(t, res)
                    # heuristics for periods: from 0.02*span to 0.5*span (user can tune later)
                    tspan = t.max() - t.min()
                    minp = max(0.02, 0.5 * (tspan / 1000.0))
                    maxp = max(0.1, 0.5 * tspan)
                    periods = np.linspace(minp, maxp, 500)
                    duration = 0.05 * max(1.0, tspan)  # rough guess
                    power = bls.power(periods, duration).power
                    best_idx = np.nanargmax(power)
                    best_period = periods[best_idx]
                    best_power = power[best_idx]
                    self.gp_result["bls_period"] = float(best_period)
                    self.gp_result["bls_power"] = float(best_power)
                    self.log(f"BLS best period ~ {best_period:.5g}, power={best_power:.5g}")
                except Exception as e:
                    self.log(f"BLS failed: {e}")

            self.log("GP fit complete.")
            self.progress["value"] = 100

        except Exception as exc:
            self.log(f"GP error: {exc}")
            messagebox.showerror("GP error", str(exc))
        finally:
            self.stop_requested = False

    # Batch processing: run GP on all lightcurves in folder
    def start_batch(self):
        # choose folder if not loaded
        folder = self.lc_folder
        if folder is None:
            folder = filedialog.askdirectory(title="Select folder with star_####.csv lightcurves")
            if not folder:
                return
        files = sorted(glob.glob(os.path.join(folder, "star_*.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if not files:
            messagebox.showwarning("No files", "No CSV files found in folder.")
            return
        # ask where to save results
        out_folder = filedialog.askdirectory(title="Select output folder for GP results (will create inside selected folder)")
        if not out_folder:
            return
        self.stop_requested = False
        self.progress["value"] = 0
        self.processing_thread = threading.Thread(target=self._batch_worker, args=(files, out_folder), daemon=True)
        self.processing_thread.start()

    def _batch_worker(self, files, out_folder):
        total = len(files)
        saved = 0
        for i, fpath in enumerate(files):
            if self.stop_requested:
                self.log("Batch cancelled by user.")
                break
            try:
                df = pd.read_csv(fpath)
                if "norm_flux" in df.columns:
                    df["flux_use"] = df["norm_flux"].astype(float)
                elif "flux" in df.columns:
                    df["flux_use"] = df["flux"].astype(float)
                else:
                    continue
                # quick GP with defaults
                t = np.asarray(df["time"], dtype=float)
                y = np.asarray(df["flux_use"], dtype=float)
                mask = np.isfinite(t) & np.isfinite(y)
                if np.sum(mask) < MIN_POINTS_FOR_GP:
                    continue
                tdata = t[mask]
                ydata = y[mask]
                baseline = np.median(ydata)
                y_norm = ydata / baseline
                X = tdata.reshape(-1, 1)
                kernel = C(1.0) * RBF(length_scale=float(self.len_entry.get() or 1.0)) + WhiteKernel(noise_level=1e-6)
                gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=int(self.restarts_entry.get() or 0))
                gp.fit(X, y_norm)
                y_pred, y_std = gp.predict(X, return_std=True)
                y_pred_scaled = y_pred * baseline
                y_std_scaled = y_std * baseline
                # save a CSV with GP prediction appended
                out_df = pd.DataFrame({"time": tdata, "flux": ydata, "gp_mean": y_pred_scaled, "gp_std": y_std_scaled})
                save_path = os.path.join(out_folder, os.path.basename(fpath).replace(".csv", "_gp.csv"))
                out_df.to_csv(save_path, index=False)
                saved += 1
                self.progress["value"] = 100.0 * (i + 1) / total
                self.log(f"Processed {i+1}/{total}: saved {save_path}")
            except Exception as e:
                self.log(f"Batch error on {fpath}: {e}")
        self.log(f"Batch finished. Saved {saved} GP files to {out_folder}")

    # Stop request
    def request_stop(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_requested = True
            self.log("Stop requested; will stop as soon as possible.")
        else:
            self.log("No active processing to stop.")

    # Save plots/results for current star
    def save_result(self):
        if self.gp_result is None:
            messagebox.showinfo("No result", "No GP result to save. Run GP first.")
            return
        out_png = filedialog.asksaveasfilename(title="Save plot as PNG", defaultextension=".png", filetypes=[("PNG","*.png")])
        if not out_png:
            return
        try:
            # redraw nicely before saving
            t = self.gp_result["t"]
            y = self.gp_result["y"]
            y_pred = self.gp_result["y_pred"]
            y_std = self.gp_result["y_std"]
            fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,6), gridspec_kw={"height_ratios":[3,1]})
            ax1.plot(t, y, ".", ms=4, label="data")
            ax1.plot(t, y_pred, "-", lw=1.2, label="GP mean")
            ax1.fill_between(t, y_pred - y_std, y_pred + y_std, alpha=0.25)
            ax1.legend()
            ax2.plot(t, y - y_pred, ".", ms=3)
            ax2.axhline(0, color="k", lw=0.6)
            ax2.set_xlabel("Time")
            ax1.set_ylabel("Flux")
            ax2.set_ylabel("Residual")
            fig2.savefig(out_png, dpi=150)
            plt.close(fig2)
            # also save numeric results next to it
            out_csv = os.path.splitext(out_png)[0] + ".csv"
            dfout = pd.DataFrame({"time": self.gp_result["t"], "flux": self.gp_result["y"], "gp_mean": self.gp_result["y_pred"], "gp_std": self.gp_result["y_std"]})
            dfout.to_csv(out_csv, index=False)
            messagebox.showinfo("Saved", f"Plot saved to {out_png}\nResults saved to {out_csv}")
            self.log(f"Saved plot and CSV for current star.")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

def main():
    root = tk.Tk()
    app = Stage3LightcurveApp(root)
    root.geometry("1200x800")
    root.mainloop()

if __name__ == "__main__":
    main()
