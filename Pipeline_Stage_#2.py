import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from skimage import io, color, data, img_as_float
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry

# Try importing astropy FITS (for real telescope data)
try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


class EnsemblePhotometryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MSEF Stage 2 – Ensemble Photometry (Brightness Calibration)")
        self.root.geometry("900x700")

        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        ttk.Button(frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=10)
        ttk.Button(frame, text="Use Test Image", command=self.load_test_image).grid(row=0, column=1, padx=10)
        ttk.Button(frame, text="Run Ensemble Photometry", command=self.run_photometry).grid(row=0, column=2, padx=10)

        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=10)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.image = None
        self.phot_table = None

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an astronomical image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.fits")]
        )

        if not file_path:
            return

        if file_path.lower().endswith(".fits") and ASTROPY_AVAILABLE:
            with fits.open(file_path) as hdul:
                self.image = hdul[0].data.astype(float)
        else:
            img = io.imread(file_path)
            if img.ndim == 3:
                img = color.rgb2gray(img)
            self.image = img_as_float(img)

        self.display_image(self.image, title="Loaded Image")

    def load_test_image(self):
        img = data.hubble_deep_field()[0:500, 0:500]  # smaller crop for speed
        img = color.rgb2gray(img)
        self.image = img_as_float(img)
        self.display_image(self.image, title="Test Image Loaded (Hubble Deep Field)")

    def run_photometry(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded.")
            return

        thread = threading.Thread(target=self.photometry_process)
        thread.start()

    def photometry_process(self):
        try:
            image = self.image

            # Step 1: Compute statistics
            mean, median, std = sigma_clipped_stats(image, sigma=3.0)
            daofind = DAOStarFinder(fwhm=3.0, threshold=5. * std)
            sources = daofind(image - median)

            if sources is None or len(sources) == 0:
                self.root.after(0, lambda: messagebox.showwarning(
                    "No Stars Found", "No stars detected in image."))
                self.root.after(0, self.progress.stop)
                return

            # Step 2: Aperture photometry
            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
            apertures = CircularAperture(positions, r=5.)
            phot_table = aperture_photometry(image, apertures)

            # Step 3: Compute brightness ratio
            target_flux = phot_table['aperture_sum'][0]
            control_fluxes = phot_table['aperture_sum'][1:6]
            control_mean = np.mean(control_fluxes)
            corrected_brightness = target_flux / control_mean

            print(f"Target: {target_flux:.3f} | Control avg: {control_mean:.3f} | Normalized: {corrected_brightness:.3f}")

            # Schedule GUI update (safe)
            self.root.after(0, lambda: self.display_results(image, positions, corrected_brightness))
            self.root.after(0, self.progress.stop)
        
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error during photometry", str(e)))
            self.root.after(0, self.progress.stop)

    def display_image(self, img, title="Image"):
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis("off")
        plt.show()

    def display_results(self, image, positions, corrected_brightness):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(image, cmap='gray', origin='lower')
        ax.set_title(f"Detected Stars (Relative Brightness: {corrected_brightness:.3f})")

        apertures = CircularAperture(positions, r=5.)
        apertures.plot(color='lime', lw=1.2, ax=ax)

        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = EnsemblePhotometryApp(root)
    root.mainloop()