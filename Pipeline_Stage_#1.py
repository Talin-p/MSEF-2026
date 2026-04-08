"""
MSEF Pipeline — Stage 1: Image Preparation & Wavelet Denoising
---------------------------------------------------------------
Supports:
  • Single-image mode  — load one file, preview, save individually
  • Batch-folder mode  — load a whole folder (FITS / PNG / JPG / TIF),
                         denoise every image, and write results to
                         <source_folder>/denoised_images/  (auto-created)
"""

import os
import glob
import threading
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

from skimage import restoration, img_as_ubyte
from skimage.util import img_as_float

try:
    from astropy.io import fits
    from astropy.time import Time as AstropyTime
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    AstropyTime = None
    warnings.warn("astropy not found — FITS support disabled. Install with: pip install astropy")

# Pipeline version stamped into every output FITS header
PIPELINE_VERSION = "MSEF-1.0"

IMAGE_EXTENSIONS = ("*.fits", "*.fit", "*.fts", "*.png", "*.jpg", "*.jpeg",
                    "*.tif", "*.tiff")

PREVIEW_SIZE = (340, 340)


def collect_images(folder: str) -> list[str]:
    """Return sorted list of image files found in *folder*."""
    files = []
    for ext in IMAGE_EXTENSIONS:
        files += glob.glob(os.path.join(folder, ext))
        files += glob.glob(os.path.join(folder, ext.upper()))
    return sorted(set(files))


def load_image(filepath: str) -> tuple[np.ndarray, object]:
    """
    Load an image from *filepath*.

    Returns
    -------
    image : float64 ndarray in [0, 1], shape (H, W)
    header : FITS header or None
    """
    ext = os.path.splitext(filepath)[1].lower()
    header = None

    if ext in (".fits", ".fit", ".fts"):
        if not ASTROPY_AVAILABLE:
            raise ImportError("astropy is required to open FITS files.")
        # ignore_missing_simple + output_verify="fix" tolerate non-standard
        # header cards (e.g. values like '+8.000e+001' with a leading '+' on
        # the exponent, which many camera/telescope programs write). Without
        # this, astropy raises VerifyError and refuses to open the file.
        with fits.open(filepath, memmap=False,
                       ignore_missing_simple=True,
                       output_verify="fix") as hdul:
            hdul.verify("silentfix")        # repair any remaining card issues
            raw    = hdul[0].data
            header = hdul[0].header.copy()  # copy before hdul closes
        if raw is None:
            raise ValueError(f"No image data in primary HDU of {filepath}")
        image = np.nan_to_num(np.asarray(raw, dtype=float))
        # collapse colour / cube dimensions if present
        while image.ndim > 2:
            image = image[0]
    else:
        from skimage import io as skio
        raw = skio.imread(filepath)
        if raw.ndim == 3:
            # RGB → luminance
            image = np.mean(raw[:, :, :3], axis=2).astype(float)
        elif raw.ndim == 2:
            image = raw.astype(float)
        else:
            raise ValueError(f"Unexpected image shape {raw.shape} in {filepath}")

    # Normalize to [0, 1]
    lo, hi = image.min(), image.max()
    if hi - lo <= 0:
        return np.zeros_like(image, dtype=float), header
    return (image - lo) / (hi - lo), header


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Apply wavelet denoising and return result in [0, 1]."""
    denoised = restoration.denoise_wavelet(
        image,
        channel_axis=None,
        rescale_sigma=True,
        method="BayesShrink",
        mode="soft",
    )
    return np.clip(denoised, 0.0, 1.0)


def noise_stats(original: np.ndarray, denoised: np.ndarray) -> dict:
    diff = original - denoised
    return {
        "sigma_original": float(np.std(original)),
        "sigma_denoised": float(np.std(denoised)),
        "sigma_removed":  float(np.std(diff)),
        "noise_reduction_pct": float(
            100.0 * (1.0 - np.std(denoised) / max(np.std(original), 1e-12))
        ),
    }



def _prepare_output_header(header, src_path: str):
    """
    Return a modified *copy* of *header* with pipeline provenance keywords
    added/updated, as specified in the Stage 1 FITS-preservation spec.

    Changes made
    ───────────
    • ORIGIN   — set to PIPELINE_VERSION
    • HISTORY  — line noting wavelet denoising step + source filename
    • DATE-PROC — UTC timestamp of this processing run (new keyword)
    • MJD-OBS  — computed from DATE-OBS via astropy if not already present
    • DATE-OBS / other original time keywords are NOT touched

    The input header is never modified in place; a deep copy is returned.
    """
    if header is None:
        return None

    # Deep-copy so we never mutate the header that came from fits.open()
    hdr = header.copy()

    # ── provenance keywords ───────────────────────────────────────────────────
    hdr["ORIGIN"]    = (PIPELINE_VERSION, "Processing pipeline name/version")
    hdr["DATE-PROC"] = (
        __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "UTC timestamp of Stage 1 denoising"
    )
    hdr.add_history(f"Wavelet denoised by {PIPELINE_VERSION} (BayesShrink, soft)")
    hdr.add_history(f"Source file: {os.path.basename(src_path)}")

    # ── compute MJD-OBS from DATE-OBS if not already present ─────────────────
    if ASTROPY_AVAILABLE and AstropyTime is not None:
        if "MJD-OBS" not in hdr and "DATE-OBS" in hdr:
            try:
                date_str = str(hdr["DATE-OBS"]).strip().replace(" ", "T").rstrip("Z")
                # Combine with TIME-OBS if DATE-OBS is date-only
                if "T" not in date_str and ":" not in date_str:
                    for tkey in ("TIME-OBS", "UTSTART", "UTC-OBS"):
                        if tkey in hdr:
                            date_str = f"{date_str}T{str(hdr[tkey]).strip()}"
                            break
                t_obj = AstropyTime(date_str, format="isot", scale="utc")
                hdr["MJD-OBS"] = (round(float(t_obj.mjd), 8),
                                  "MJD of observation (computed from DATE-OBS)")
            except Exception as e:
                hdr.add_comment(f"MJD-OBS not computed: {e}")

        # If MJD-OBS exists but JD is absent, add JD for completeness
        if "MJD-OBS" in hdr and "JD" not in hdr:
            try:
                hdr["JD"] = (round(float(hdr["MJD-OBS"]) + 2_400_000.5, 8),
                             "Julian Date (derived from MJD-OBS)")
            except Exception:
                pass

    return hdr

def save_denoised(image: np.ndarray, src_path: str, out_folder: str,
                  header=None) -> str:
    """
    Save *image* to *out_folder*.

    • FITS source → writes a denoised .fits (header updated per spec) +
                    a .png preview alongside it
    • Other source → writes .png only

    Returns the path of the primary saved file (FITS or PNG).
    """
    os.makedirs(out_folder, exist_ok=True)
    basename = os.path.splitext(os.path.basename(src_path))[0]
    ext      = os.path.splitext(src_path)[1].lower()

    if ext in (".fits", ".fit", ".fts") and ASTROPY_AVAILABLE:
        # Build an updated copy of the header with provenance keywords
        out_hdr = _prepare_output_header(header, src_path)
        fits_out = os.path.join(out_folder, basename + "_denoised.fits")
        hdu = fits.PrimaryHDU(data=image.astype(np.float32), header=out_hdr)
        # output_verify="silentfix" repairs any non-standard header cards
        # (e.g. exponent '+' signs) rather than raising VerifyError on write
        hdu.writeto(fits_out, overwrite=True, output_verify="silentfix")
        # PNG preview alongside (Stage 2 will use the FITS; preview is for humans)
        png_out = os.path.join(out_folder, basename + "_denoised_preview.png")
        Image.fromarray(img_as_ubyte(image)).save(png_out)
        return fits_out
    else:
        png_out = os.path.join(out_folder, basename + "_denoised.png")
        Image.fromarray(img_as_ubyte(image)).save(png_out)
        return png_out


def array_to_tk(arr: np.ndarray, size=PREVIEW_SIZE) -> ImageTk.PhotoImage:
    """Convert a float [0,1] 2-D array to a resized Tk-compatible PhotoImage."""
    pil = Image.fromarray(img_as_ubyte(arr))
    pil.thumbnail(size, Image.LANCZOS)
    return ImageTk.PhotoImage(pil)

class Stage1App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("MSEF — Stage 1: Wavelet Denoising")
        root.geometry("1050x720")
        root.minsize(900, 620)

        self.current_original: np.ndarray | None = None
        self.current_denoised: np.ndarray | None = None
        self.current_header = None
        self.current_src_path: str | None = None
        self._tk_orig = None        # hold refs so GC doesn't kill PhotoImages
        self._tk_den  = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

        self._build_ui()
        self.log("Stage 1 ready.  Load a single image or choose a folder for batch processing.")

    def _build_ui(self):
        root = self.root

        # title bar
        tk.Label(root, text="MSEF Pipeline  ·  Stage 1: Wavelet Denoising",
                 font=("Arial", 15, "bold")).pack(pady=(10, 4))

        # button row
        btn_row = tk.Frame(root)
        btn_row.pack(fill="x", padx=12, pady=4)

        btn_cfg = dict(width=18, relief="groove", pady=4)
        tk.Button(btn_row, text="📂  Single Image",
                  command=self.choose_single_file, **btn_cfg).pack(side="left", padx=4)
        tk.Button(btn_row, text="📁  Process Folder",
                  command=self.choose_folder, **btn_cfg).pack(side="left", padx=4)
        tk.Button(btn_row, text="🔭  Sample Image",
                  command=self.load_sample, **btn_cfg).pack(side="left", padx=4)
        tk.Button(btn_row, text="📊  Show Analysis",
                  command=self.show_graphs, **btn_cfg).pack(side="left", padx=4)
        tk.Button(btn_row, text="💾  Save Current",
                  command=self.save_current, **btn_cfg).pack(side="left", padx=4)
        self.cancel_btn = tk.Button(btn_row, text="⛔  Cancel Batch",
                                    command=self.cancel_batch,
                                    width=14, relief="groove", pady=4,
                                    state="disabled")
        self.cancel_btn.pack(side="left", padx=4)

        # progress & status
        prog_frame = tk.Frame(root)
        prog_frame.pack(fill="x", padx=12, pady=(2, 0))
        self.progress = ttk.Progressbar(prog_frame, orient="horizontal",
                                        length=700, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True)
        self.batch_label = tk.Label(prog_frame, text="", width=14,
                                    font=("Arial", 9))
        self.batch_label.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="Idle")
        tk.Label(root, textvariable=self.status_var,
                 font=("Arial", 9), fg="#444").pack(anchor="w", padx=14)

        # preview panels
        preview_row = tk.Frame(root)
        preview_row.pack(fill="both", expand=True, padx=12, pady=8)

        lf_orig = ttk.LabelFrame(preview_row, text="Original")
        lf_orig.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.canvas_orig = tk.Canvas(lf_orig, bg="#111",
                                     width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        self.canvas_orig.pack(fill="both", expand=True, padx=4, pady=4)

        lf_den = ttk.LabelFrame(preview_row, text="Denoised")
        lf_den.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.canvas_den = tk.Canvas(lf_den, bg="#111",
                                    width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        self.canvas_den.pack(fill="both", expand=True, padx=4, pady=4)

        # log box
        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.log_box = tk.Text(log_frame, height=8, wrap="word",
                               font=("Courier", 8), bg="#1e1e1e", fg="#d4d4d4",
                               insertbackground="white")
        sb = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        full = f"[{ts}]  {msg}\n"
        self.root.after(0, lambda: (
            self.log_box.insert("end", full),
            self.log_box.see("end"),
            self.status_var.set(msg),
        ))

    def set_progress(self, value: float, batch_text: str = ""):
        self.root.after(0, lambda: (
            self.progress.config(value=value),
            self.batch_label.config(text=batch_text),
        ))

    def _draw_on_canvas(self, canvas: tk.Canvas, arr: np.ndarray,
                        attr: str):
        """Draw *arr* on *canvas*, keeping a Tk reference in *self.attr*."""
        tkimg = array_to_tk(arr, PREVIEW_SIZE)
        setattr(self, attr, tkimg)           # prevent garbage collection
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tkimg)

    def update_previews(self, original: np.ndarray, denoised: np.ndarray):
        self.root.after(0, lambda: (
            self._draw_on_canvas(self.canvas_orig, original, "_tk_orig"),
            self._draw_on_canvas(self.canvas_den,  denoised,  "_tk_den"),
        ))

    def _lock_ui(self, locked: bool):
        state = "disabled" if locked else "normal"
        cancel = "normal" if locked else "disabled"
        self.root.after(0, lambda: self.cancel_btn.config(state=cancel))

    def choose_single_file(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Astronomical / image files",
                 "*.fits *.fit *.fts *.png *.jpg *.jpeg *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._run_single(path)

    def load_sample(self):
        """Use scikit-image's built-in Hubble deep-field sample."""
        try:
            from skimage import data as skdata
            raw = skdata.hubble_deep_field()[0:500, 0:500]
            if raw.ndim == 3:
                raw = np.mean(raw, axis=2)
            sample = img_as_float(raw)
            sample = (sample - sample.min()) / max(sample.max() - sample.min(), 1e-12)
            self._run_array(sample, label="Sample (Hubble Deep Field)")
        except Exception as e:
            messagebox.showerror("Sample load error", str(e))

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing images")
        if folder:
            self._run_batch(folder)

    def cancel_batch(self):
        self._cancel.set()
        self.log("Cancel requested — stopping after current image…")

    def save_current(self):
        if self.current_denoised is None:
            messagebox.showinfo("Nothing to save", "Process an image first.")
            return

        # If the source was a FITS file, offer FITS as the default format
        src_is_fits = (
            self.current_src_path is not None and
            os.path.splitext(self.current_src_path)[1].lower() in (".fits", ".fit", ".fts")
            and ASTROPY_AVAILABLE
        )

        if src_is_fits:
            filetypes = [
                ("FITS image", "*.fits"),
                ("PNG",        "*.png"),
                ("TIFF",       "*.tiff"),
                ("All",        "*.*"),
            ]
            default_ext = ".fits"
        else:
            filetypes = [
                ("PNG",   "*.png"),
                ("TIFF",  "*.tiff"),
                ("JPEG",  "*.jpg"),
                ("All",   "*.*"),
            ]
            default_ext = ".png"

        path = filedialog.asksaveasfilename(
            title="Save denoised image",
            defaultextension=default_ext,
            filetypes=filetypes,
        )
        if not path:
            return

        try:
            out_ext = os.path.splitext(path)[1].lower()
            if out_ext in (".fits", ".fit", ".fts") and ASTROPY_AVAILABLE:
                # Write a proper FITS with updated header
                out_hdr = _prepare_output_header(
                    self.current_header, self.current_src_path or path)
                hdu = fits.PrimaryHDU(
                    data=self.current_denoised.astype(np.float32),
                    header=out_hdr)
                hdu.writeto(path, overwrite=True, output_verify="silentfix")
            else:
                Image.fromarray(img_as_ubyte(self.current_denoised)).save(path)
            self.log(f"Saved → {path}")
            messagebox.showinfo("Saved", f"Denoised image saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def show_graphs(self):
        if self.current_original is None or self.current_denoised is None:
            messagebox.showinfo("No data", "Process an image first.")
            return
        # run in thread so it doesn't block
        threading.Thread(target=self._plot_graphs, daemon=True).start()

    def _plot_graphs(self):
        import subprocess, platform, tempfile
        orig  = self.current_original
        den   = self.current_denoised
        diff  = orig - den
        stats = noise_stats(orig, den)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].imshow(orig, cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(den, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title("Denoised")
        axes[0, 1].axis("off")

        axes[1, 0].hist(orig.ravel(), bins=60, alpha=0.55, label="Original", color="steelblue")
        axes[1, 0].hist(den.ravel(),  bins=60, alpha=0.55, label="Denoised", color="coral")
        axes[1, 0].set_title("Pixel Intensity Distribution")
        axes[1, 0].legend()

        im = axes[1, 1].imshow(diff, cmap="seismic", vmin=-0.1, vmax=0.1)
        axes[1, 1].set_title("Removed Noise  (Original − Denoised)")
        plt.colorbar(im, ax=axes[1, 1])

        fig.suptitle(
            f"σ original={stats['sigma_original']:.4f}   "
            f"σ denoised={stats['sigma_denoised']:.4f}   "
            f"σ removed={stats['sigma_removed']:.4f}   "
            f"noise ↓ {stats['noise_reduction_pct']:.1f}%"
        )
        plt.tight_layout()

        # Save to a temp file and open with the OS viewer — calling plt.show()
        # from a daemon thread crashes on macOS (NSInternalInconsistencyException)
        # and can freeze on Windows.
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            fig.savefig(tmp.name, dpi=130)
            plt.close(fig)
            if platform.system() == "Darwin":
                subprocess.Popen(["open", tmp.name])
            elif platform.system() == "Windows":
                os.startfile(tmp.name)
            else:
                subprocess.Popen(["xdg-open", tmp.name])
        except Exception as e:
            self.log(f"Could not open analysis plot: {e}")

    # single-image processing

    def _run_single(self, path: str):
        if self._thread and self._thread.is_alive():
            messagebox.showinfo("Busy", "Processing is already running.")
            return
        self._thread = threading.Thread(target=self._single_worker,
                                        args=(path,), daemon=True)
        self._thread.start()

    def _run_array(self, arr: np.ndarray, label: str):
        if self._thread and self._thread.is_alive():
            messagebox.showinfo("Busy", "Processing is already running.")
            return
        self._thread = threading.Thread(target=self._array_worker,
                                        args=(arr, label), daemon=True)
        self._thread.start()

    def _single_worker(self, path: str):
        self._lock_ui(True)
        try:
            self.log(f"Loading  {os.path.basename(path)} …")
            self.set_progress(10)
            image, header = load_image(path)
            self.set_progress(30, "")
            self.log("Denoising…")
            denoised = denoise_image(image)
            self.set_progress(90)
            self.current_original = image
            self.current_denoised = denoised
            self.current_header   = header
            self.current_src_path = path
            self.update_previews(image, denoised)
            st = noise_stats(image, denoised)
            self.log(
                f"Done.  σ_orig={st['sigma_original']:.4f}  "
                f"σ_den={st['sigma_denoised']:.4f}  "
                f"noise ↓ {st['noise_reduction_pct']:.1f}%"
            )
            self.set_progress(100)
        except Exception as e:
            self.log(f"ERROR: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self._lock_ui(False)

    def _array_worker(self, arr: np.ndarray, label: str):
        self._lock_ui(True)
        try:
            self.log(f"Processing {label}…")
            self.set_progress(20)
            denoised = denoise_image(arr)
            self.current_original = arr
            self.current_denoised = denoised
            self.current_header   = None
            self.current_src_path = None
            self.update_previews(arr, denoised)
            st = noise_stats(arr, denoised)
            self.log(
                f"Done ({label}).  σ_orig={st['sigma_original']:.4f}  "
                f"σ_den={st['sigma_denoised']:.4f}  "
                f"noise ↓ {st['noise_reduction_pct']:.1f}%"
            )
            self.set_progress(100)
        except Exception as e:
            self.log(f"ERROR: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self._lock_ui(False)

    # batch-folder processing

    def _run_batch(self, folder: str):
        if self._thread and self._thread.is_alive():
            messagebox.showinfo("Busy", "Processing is already running.")
            return
        out_folder = os.path.join(folder, "denoised_images")
        all_files  = collect_images(folder)
        # Skip files that live inside our own output folder (prevents self-feed
        # on second run when the denoised_images subfolder already exists)
        out_real = os.path.realpath(out_folder)
        files = [f for f in all_files
                 if not os.path.realpath(f).startswith(out_real + os.sep)
                 and os.path.realpath(f) != out_real]
        if not files:
            messagebox.showwarning(
                "No images found",
                f"No supported image files found in:\n{folder}\n\n"
                f"Supported: {', '.join(IMAGE_EXTENSIONS)}"
            )
            return
        self._cancel.clear()
        self._thread = threading.Thread(target=self._batch_worker,
                                        args=(files, out_folder), daemon=True)
        self._thread.start()

    def _batch_worker(self, files: list[str], out_folder: str):
        self._lock_ui(True)
        os.makedirs(out_folder, exist_ok=True)
        self.log(f"Batch start — {len(files)} images → {out_folder}")

        n = len(files)
        ok = 0
        failed = 0

        for i, path in enumerate(files):
            if self._cancel.is_set():
                self.log(f"Cancelled after {i} of {n} images.")
                break

            fname = os.path.basename(path)
            self.log(f"[{i+1}/{n}]  {fname}")
            self.set_progress(100.0 * i / n, f"{i+1} / {n}")

            try:
                image, header = load_image(path)
                denoised = denoise_image(image)

                # update the live preview with every processed image
                self.current_original = image
                self.current_denoised = denoised
                self.update_previews(image, denoised)

                saved_path = save_denoised(image, path, out_folder, header)
                st = noise_stats(image, denoised)
                self.log(
                    f"  ✓  saved → {os.path.basename(saved_path)}   "
                    f"noise ↓ {st['noise_reduction_pct']:.1f}%"
                )
                ok += 1

            except Exception as e:
                self.log(f"  ✗  FAILED ({fname}): {e}")
                failed += 1

        self.set_progress(100, f"{ok}/{n} done")
        self.log(
            f"Batch complete — {ok} succeeded, {failed} failed.  "
            f"Output: {out_folder}"
        )
        self.root.after(
            0,
            lambda: messagebox.showinfo(
                "Batch complete",
                f"{ok} of {n} images denoised successfully.\n"
                + (f"{failed} failed (see log).\n" if failed else "")
                + f"\nSaved to:\n{out_folder}",
            ),
        )
        self._lock_ui(False)

def main():
    root = tk.Tk()
    app = Stage1App(root)
    root.mainloop()


if __name__ == "__main__":
    main()