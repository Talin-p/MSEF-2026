import os
import glob
import warnings
import threading
import time
from functools import partial
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_com
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageOps, ImageTk

# Optional: BoxLeastSquares for transit search
try:
    from astropy.timeseries import BoxLeastSquares
    HAS_BLS = True
except Exception:
    HAS_BLS = False
    warnings.warn("astropy.timeseries.BoxLeastSquares not available. BLS disabled.")

# Default configuration (tweak as needed in the app or in code)
APERTURE_RADIUS = 4.0
SKY_INNER = 8.0
SKY_OUTER = 12.0
MIN_SNR_FOR_DETECTION = 5.0
MAX_SOURCES = 800
RE_CENTER = True
RECENTER_BOX = 7

RUN_BLS = True
BLS_MIN_PERIOD = 0.2
BLS_MAX_PERIOD = 20.0
BLS_N_PERIODS = 1000

TIME_HEADER_KEYS = ["DATE-OBS", "MJD-OBS", "JD", "TIME-OBS"]

# Output folders (created automatically)
OUTPUT_FOLDER = "stage2_output"
LIGHTCURVES_DIR = os.path.join(OUTPUT_FOLDER, "lightcurves")
PLOTS_DIR = os.path.join(OUTPUT_FOLDER, "output_plots")
MASTER_CATALOG = os.path.join(OUTPUT_FOLDER, "master_catalog.csv")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LIGHTCURVES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Utility functions (core pipeline)
def list_image_files(folder):
    exts = ("*.fits", "*.fit", "*.fts", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files += sorted(glob.glob(os.path.join(folder, e)))
    return files

def load_image_and_time(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fits", ".fit", ".fts"):
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data.astype(float)
            header = hdul[0].header
            t = None
            for key in TIME_HEADER_KEYS:
                if key in header:
                    val = header[key]
                    try:
                        if key in ("MJD-OBS", "MJD"):
                            t = float(val)
                        elif key == "JD":
                            t = float(val)
                        else:
                            from datetime import datetime
                            s = str(val)
                            try:
                                dt = datetime.fromisoformat(s.replace("Z", ""))
                                t = dt.timestamp() / 86400.0
                            except Exception:
                                pass
                    except Exception:
                        pass
            return data, t, header
    else:
        pil = ImageOps.exif_transpose(Image.open(path))
        pil = pil.convert("L")
        arr = np.asarray(pil).astype(float)
        maxv = arr.max() if arr.max() != 0 else 1.0
        arr /= maxv
        return arr, None, {}

def detect_sources_on_ref(image, fwhm=3.0, threshold_sigma=5.0, max_sources=None):
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return [], np.zeros((0,2))
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = daofind(image - median)
    if sources is None:
        return [], np.zeros((0,2))
    sources.sort("flux")
    sources = sources[::-1]
    if max_sources is not None:
        sources = sources[:max_sources]
    xy = np.vstack((sources["xcentroid"], sources["ycentroid"])).T
    return sources, xy

def measure_apertures(image, positions, aperture_radius=APERTURE_RADIUS,
                      sky_inner=SKY_INNER, sky_outer=SKY_OUTER):
    apertures = CircularAperture(positions, r=aperture_radius)
    phot = aperture_photometry(image, apertures)
    aper_sum = phot["aperture_sum"].data.astype(float)

    big_ap = CircularAperture(positions, r=sky_outer)
    inner_ap = CircularAperture(positions, r=sky_inner)
    big_sum = aperture_photometry(image, big_ap)["aperture_sum"].data.astype(float)
    inner_sum = aperture_photometry(image, inner_ap)["aperture_sum"].data.astype(float)
    annulus_area = np.pi * (sky_outer**2 - sky_inner**2)
    sky_total = big_sum - inner_sum
    sky_mean = sky_total / annulus_area
    return aper_sum, sky_mean

# GUI Application
class Stage2GUI:
    def __init__(self, root):
        self.root = root
        root.title("MSEF — Stage 2 Photometry (GUI)")

        # state
        self.selected_folder = None
        self.selected_image_path = None
        self.ref_xy = None
        self.ref_sources = None
        self.processing_thread = None
        self.stop_requested = False

        # top frame: controls
        frm = ttk.Frame(root, padding=8)
        frm.grid(row=0, column=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=0, column=0, sticky="w")

        ttk.Button(btn_frame, text="Select Image Folder", command=self.select_folder).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frame, text="Select Single Image (preview)", command=self.select_single_image).grid(row=0, column=1, padx=4)
        self.run_btn = ttk.Button(btn_frame, text="Run Stage 2", command=self.run_pipeline_async)
        self.run_btn.grid(row=0, column=2, padx=4)
        ttk.Button(btn_frame, text="Save Master Catalog", command=self.save_master_catalog).grid(row=0, column=3, padx=4)
        ttk.Button(btn_frame, text="Quit", command=root.quit).grid(row=0, column=4, padx=4)

        # progress and status
        self.progress = ttk.Progressbar(frm, orient="horizontal", length=600, mode="determinate")
        self.progress.grid(row=1, column=0, pady=(6,4), sticky="w")
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(frm, textvariable=self.status_var).grid(row=2, column=0, sticky="w")

        # middle: image preview and logs
        mid = ttk.Frame(frm)
        mid.grid(row=3, column=0, sticky="nsew", pady=(8,0))
        frm.rowconfigure(3, weight=1)
        frm.columnconfigure(0, weight=1)

        # preview canvas
        preview_frame = ttk.LabelFrame(mid, text="Image Preview (reference / selected image)")
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(preview_frame, width=500, height=500, bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas_image = None

        # logs
        log_frame = ttk.LabelFrame(mid, text="Log")
        log_frame.grid(row=0, column=1, sticky="nsew")
        mid.columnconfigure(1, weight=1)
        self.log_text = tk.Text(log_frame, width=60, height=25, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # bottom: cancel and options
        bottom = ttk.Frame(frm)
        bottom.grid(row=4, column=0, sticky="we", pady=(8,0))
        ttk.Button(bottom, text="Cancel Run", command=self.request_stop).grid(row=0, column=0, padx=4)
        # options small UI
        ttk.Label(bottom, text="Min SNR:").grid(row=0, column=1, padx=(8,2))
        self.snr_entry = ttk.Entry(bottom, width=6)
        self.snr_entry.insert(0, str(MIN_SNR_FOR_DETECTION))
        self.snr_entry.grid(row=0, column=2, padx=2)
        ttk.Label(bottom, text="Max sources:").grid(row=0, column=3, padx=(8,2))
        self.maxsrc_entry = ttk.Entry(bottom, width=6)
        self.maxsrc_entry.insert(0, str(MAX_SOURCES))
        self.maxsrc_entry.grid(row=0, column=4, padx=2)

        root.protocol("WM_DELETE_WINDOW", root.quit)
        self.log("GUI ready.")

    # helpers
    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {msg}\n")
        self.log_text.see("end")
        self.status_var.set(msg)
        self.root.update_idletasks()

    def update_canvas(self, image_arr, overlay_xy=None):
        # image_arr: 2D grayscale float
        arr = np.clip(image_arr, 0.0, 1.0)
        im = Image.fromarray(np.uint8(arr * 255), mode="L")
        im.thumbnail((500, 500), Image.LANCZOS)
        draw = ImageOps.colorize(im.convert("L"), black="black", white="white")
        tkimg = ImageTk.PhotoImage(draw)
        self.canvas_image = tkimg
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=tkimg)
        # overlay stars if provided (positions are in original image coords -> need scaling)
        if overlay_xy is not None and overlay_xy.size > 0:
            # compute scale factors
            ow, oh = im.size
            # original image size
            H, W = image_arr.shape
            sx = ow / W
            sy = oh / H
            for (x, y) in overlay_xy:
                cx = int(x * sx)
                cy = int(y * sy)
                # small cross
                r = 3
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="red", width=1)

        self.root.update_idletasks()

    # UI actions
    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder with images")
        if not folder:
            return
        self.selected_folder = folder
        self.log(f"Selected folder: {folder}")
        # show a preview using a reference median of first/mid/last if possible
        files = list_image_files(folder)
        if not files:
            messagebox.showwarning("No files", "No image files found in selected folder.")
            return
        # create ref image
        try:
            idxs = [0, len(files)//2, max(0, len(files)-1)]
            imgs = []
            for i in idxs:
                im, t, hdr = load_image_and_time(files[i])
                imgs.append(im)
            ref = np.median(np.stack(imgs), axis=0)
            srcs, xy = detect_sources_on_ref(ref, fwhm=3.0, threshold_sigma=float(self.snr_entry.get()), max_sources=int(self.maxsrc_entry.get()))
            self.ref_xy = xy
            self.ref_sources = srcs
            self.update_canvas(ref, overlay_xy=xy)
            self.log(f"Reference built and {len(xy)} sources detected (preview).")
        except Exception as e:
            self.log(f"Preview failed: {e}")

    def select_single_image(self):
        path = filedialog.askopenfilename(title="Select single image", filetypes=[("Images","*.fits *.fit *.fts *.png *.jpg *.jpeg *.tif *.tiff"),("All","*.*")])
        if not path:
            return
        try:
            img, _, _ = load_image_and_time(path)
            srcs, xy = detect_sources_on_ref(img, fwhm=3.0, threshold_sigma=float(self.snr_entry.get()), max_sources=int(self.maxsrc_entry.get()))
            self.ref_xy = xy
            self.ref_sources = srcs
            self.selected_image_path = path
            self.update_canvas(img, overlay_xy=xy)
            self.log(f"Loaded image: {os.path.basename(path)} (detected {len(xy)} sources).")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def request_stop(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_requested = True
            self.log("Stop requested; finishing current step...")
        else:
            self.log("No active processing to stop.")

    def save_master_catalog(self):
        if not os.path.exists(MASTER_CATALOG):
            messagebox.showinfo("No catalog", "No master catalog yet. Run pipeline first.")
            return
        save_to = filedialog.asksaveasfilename(title="Save master catalog as...", defaultextension=".csv", filetypes=[("CSV",".csv")])
        if not save_to:
            return
        try:
            import shutil
            shutil.copyfile(MASTER_CATALOG, save_to)
            messagebox.showinfo("Saved", f"Master catalog saved to {save_to}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    # Pipeline runner (background)
    def run_pipeline_async(self):
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showinfo("Running", "Pipeline is already running.")
            return
        folder = self.selected_folder
        if not folder:
            folder = filedialog.askdirectory(title="Select folder with images to process")
            if not folder:
                return
            self.selected_folder = folder
        self.stop_requested = False
        self.progress["value"] = 0
        self.processing_thread = threading.Thread(target=self._run_pipeline_worker, args=(self.selected_folder,), daemon=True)
        self.processing_thread.start()

    def _run_pipeline_worker(self, folder):
        try:
            self.log(f"Starting pipeline on folder: {folder}")
            files = list_image_files(folder)
            if not files:
                raise RuntimeError("No image files found in folder.")

            n_frames = len(files)
            self.log(f"Found {n_frames} frames.")
            # build reference
            sample_idxs = [0, n_frames//2, max(0, n_frames-1)]
            sample_imgs = []
            for i in sample_idxs:
                im, t, hdr = load_image_and_time(files[i])
                sample_imgs.append(im)
            ref = np.median(np.stack(sample_imgs), axis=0)
            # detect sources
            thr = float(self.snr_entry.get())
            maxsrc = int(self.maxsrc_entry.get())
            sources, ref_xy = detect_sources_on_ref(ref, threshold_sigma=thr, max_sources=maxsrc)
            if ref_xy is None or len(ref_xy) == 0:
                raise RuntimeError("No sources detected on reference frame. Try lowering SNR.")
            self.ref_xy = ref_xy
            self.ref_sources = sources
            self.update_canvas(ref, overlay_xy=ref_xy)
            n_sources = len(ref_xy)
            self.log(f"Detected {n_sources} sources on reference.")

            # prepare containers
            flux_matrix = np.full((n_sources, n_frames), np.nan, dtype=float)
            bg_matrix = np.full((n_sources, n_frames), np.nan, dtype=float)
            times = np.full(n_frames, np.nan, dtype=float)

            for fi, path in enumerate(files):
                if self.stop_requested:
                    self.log("Run cancelled by user.")
                    self.progress["value"] = 0
                    return
                im, t, hdr = load_image_and_time(path)
                if t is None or not np.isfinite(t):
                    t = float(fi)  # fallback to frame index as time
                times[fi] = t

                positions = ref_xy.copy()
                if RE_CENTER:
                    H, W = im.shape
                    newpos = []
                    half = RECENTER_BOX
                    for (x0, y0) in positions:
                        xi, yi = int(round(x0)), int(round(y0))
                        x1, x2 = max(0, xi-half), min(W, xi+half+1)
                        y1, y2 = max(0, yi-half), min(H, yi+half+1)
                        cut = im[y1:y2, x1:x2]
                        if cut.size == 0:
                            newpos.append((x0, y0))
                        else:
                            try:
                                cy, cx = centroid_com(cut)
                                nx, ny = x1 + cx, y1 + cy
                                newpos.append((nx, ny))
                            except Exception:
                                newpos.append((x0, y0))
                    positions = np.array(newpos)

                aper_sum, sky_mean = measure_apertures(im, positions)
                flux_matrix[:, fi] = aper_sum - sky_mean * (np.pi * APERTURE_RADIUS**2)
                bg_matrix[:, fi] = sky_mean

                prog = 100.0 * (fi+1) / n_frames
                self.progress["value"] = prog
                self.log(f"Processed frame {fi+1}/{n_frames} ({prog:.1f}%)")
                # small UI breathe
                time.sleep(0.01)

            # times normalize
            valid_times = times[np.isfinite(times)]
            if valid_times.size == 0:
                t0 = 0.0
            else:
                t0 = valid_times[0]
            times = times - t0

            # write per-star lightcurves and compute metrics + optional BLS
            master_rows = []
            for sid in range(n_sources):
                flux = flux_matrix[sid, :]
                bg = bg_matrix[sid, :]
                good = np.isfinite(flux)
                n_points = int(np.sum(good))
                if n_points < 5:
                    meanf = np.nan
                    stdf = np.nan
                    mad = np.nan
                    norm_flux = np.full_like(flux, np.nan)
                else:
                    meanf = np.nanmedian(flux[good])
                    stdf = np.nanstd(flux[good])
                    mad = np.nanmedian(np.abs(flux[good] - meanf))
                    norm_flux = flux / meanf

                # save lc
                df = pd.DataFrame({"time": times, "flux": flux, "bg": bg, "norm_flux": norm_flux})
                df["flux_use"] = df["norm_flux"]
                lc_file = os.path.join(LIGHTCURVES_DIR, f"star_{sid:04d}.csv")
                df.to_csv(lc_file, index=False)

                bls_period = np.nan
                bls_power = np.nan
                if RUN_BLS and HAS_BLS and n_points >= 30:
                    try:
                        mask = np.isfinite(times) & np.isfinite(norm_flux)
                        if np.sum(mask) >= 30:
                            t_bls = times[mask]
                            y_bls = norm_flux[mask] - 1.0
                            model = BoxLeastSquares(t_bls, y_bls)
                            tspan = t_bls.max() - t_bls.min()
                            min_period = max(BLS_MIN_PERIOD, 0.5 * (tspan / 1000.0))
                            max_period = min(BLS_MAX_PERIOD, tspan)
                            if np.isfinite(min_period) and np.isfinite(max_period) and max_period > min_period:
                                periods = np.linspace(min_period, max_period, BLS_N_PERIODS)
                                duration = 0.05
                                res = model.power(periods, duration)
                                idx = np.nanargmax(res.power)
                                bls_period = float(res.period[idx])
                                bls_power = float(res.power[idx])
                    except Exception:
                        pass

                x, y = ref_xy[sid]
                master_rows.append({
                    "star_id": sid,
                    "x": float(x),
                    "y": float(y),
                    "n_points": int(n_points),
                    "median_flux": float(meanf) if not np.isnan(meanf) else np.nan,
                    "std_flux": float(stdf) if not np.isnan(stdf) else np.nan,
                    "mad_flux": float(mad) if not np.isnan(mad) else np.nan,
                    "bls_period": bls_period,
                    "bls_power": bls_power,
                    "lc_file": os.path.relpath(lc_file, OUTPUT_FOLDER)
                })

            master_df = pd.DataFrame(master_rows)
            master_df.to_csv(MASTER_CATALOG, index=False)
            self.log(f"Pipeline complete. Master catalog saved to {MASTER_CATALOG}")
            self.progress["value"] = 100
            # create quick diagnostic plot
            try:
                plt.figure(figsize=(8,6))
                plt.scatter(master_df["x"], master_df["y"], s=10, c=master_df["std_flux"].fillna(0), cmap="viridis")
                plt.gca().invert_yaxis()
                plt.colorbar(label="std_flux")
                plt.title("Detected Sources (color=std_flux)")
                plt.savefig(os.path.join(PLOTS_DIR, "sources_std.png"))
                plt.close()
                self.log(f"Diagnostic plot saved to {os.path.join(PLOTS_DIR, 'sources_std.png')}")
            except Exception as e:
                self.log(f"Failed to save diagnostic plot: {e}")

        except Exception as exc:
            self.log(f"ERROR: {exc}")
            messagebox.showerror("Pipeline error", str(exc))
        finally:
            self.stop_requested = False
            self.log("Run finished or stopped.")

# Run application
def main():
    root = tk.Tk()
    app = Stage2GUI(root)
    root.geometry("1100x700")
    root.mainloop()

if __name__ == "__main__":
    main()