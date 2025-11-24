import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from skimage import io, restoration, img_as_ubyte, data
import threading
import matplotlib.pyplot as plt

# load FITS if astronomy images
try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


# IMAGE PROCESSING

def load_image(filepath):
    """Load image from file, supports FITS, PNG, JPG. Converts to grayscale if needed."""
    if filepath.lower().endswith('.fits'):
        if not ASTROPY_AVAILABLE:
            raise ImportError("Install astropy to open FITS files: pip install astropy")
        data = fits.getdata(filepath)
        image = np.nan_to_num(data)
    else:
        image = io.imread(filepath)
        if image.ndim == 3:
            image = np.mean(image, axis=2)

    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def load_sample_image():
    sample = data.hubble_deep_field()[0:500, 0:500]
    if sample.ndim == 3:
        sample = np.mean(sample, axis=2)
    sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
    run_in_thread(sample, "Sample Image")


def denoise_image(image, progress_callback):
    progress_callback(30, "Applying wavelet denoising...")
    denoised = restoration.denoise_wavelet(
        image, channel_axis=None, rescale_sigma=True
    )
    progress_callback(90, "Finalizing...")
    return denoised

def save_image():
    """Save the denoised image as PNG/JPG/TIFF."""
    global current_denoised
    if current_denoised is None:
        messagebox.showerror("Error", "No denoised image available to save.")
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[
            ("PNG Image", "*.png"),
            ("JPEG Image", "*.jpg"),
            ("TIFF Image", "*.tiff"),
            ("All Files", "*.*")
        ],
        title="Save Denoised Image"
    )

    if filepath:
        try:
            img = Image.fromarray(img_as_ubyte(current_denoised))
            img.save(filepath)
            messagebox.showinfo("Saved", f"Denoised image saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error Saving Image", str(e))

def show_analysis_graphs(original, denoised):
    diff = original - denoised
    orig_std = np.std(original)
    den_std = np.std(denoised)
    diff_std = np.std(diff)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(denoised, cmap='gray')
    axes[0, 1].set_title("Denoised Image")
    axes[0, 1].axis('off')

    axes[1, 0].hist(original.ravel(), bins=50, alpha=0.5, label='Original')
    axes[1, 0].hist(denoised.ravel(), bins=50, alpha=0.5, label='Denoised')
    axes[1, 0].set_title("Pixel Intensity Distribution")
    axes[1, 0].legend()

    im = axes[1, 1].imshow(diff, cmap='seismic', vmin=-0.1, vmax=0.1)
    axes[1, 1].set_title("Removed Noise (Original - Denoised)")
    plt.colorbar(im, ax=axes[1, 1])

    fig.suptitle(
        f"Noise Reduction: σ(original)={orig_std:.4f}, "
        f"σ(denoised)={den_std:.4f}, σ(diff)={diff_std:.4f}"
    )
    plt.tight_layout()
    plt.show()


# GUI DISPLAY

def display_images(original, denoised):
    global img_panel_original, img_panel_denoised, img_original_data, img_denoised_data

    orig_disp = Image.fromarray(img_as_ubyte(original))
    den_disp = Image.fromarray(img_as_ubyte(denoised))

    orig_disp = orig_disp.resize((300, 300))
    den_disp = den_disp.resize((300, 300))

    img_original_data = ImageTk.PhotoImage(orig_disp)
    img_denoised_data = ImageTk.PhotoImage(den_disp)

    img_panel_original.config(image=img_original_data)
    img_panel_denoised.config(image=img_denoised_data)


def process_image(image, label):
    global current_original, current_denoised
    try:
        progress_bar["value"] = 10
        status_label.config(text=f"Loading {label}...")
        img = image

        progress_bar["value"] = 20
        status_label.config(text="Image loaded!")

        denoised = denoise_image(img, lambda val, msg: (
            progress_bar.config(value=val), status_label.config(text=msg))
        )

        progress_bar["value"] = 100
        status_label.config(text="Denoising complete!")

        current_original, current_denoised = img, denoised
        display_images(img, denoised)
        show_graphs_button.config(state=tk.NORMAL)
        save_button.config(state=tk.NORMAL)     # <--- enable saving
        status_label.config(text="Ready — click 'Show Graphs' for analysis")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_label.config(text="Error occurred.")


def run_in_thread(image_or_path, label="Image"):
    show_graphs_button.config(state=tk.DISABLED)
    save_button.config(state=tk.DISABLED)
    threading.Thread(
        target=process_image,
        args=(image_or_path if isinstance(image_or_path, np.ndarray) else load_image(image_or_path), label),
        daemon=True
    ).start()


def choose_file():
    filepath = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.fits"), ("All files", "*.*")]
    )
    if filepath:
        run_in_thread(filepath, "File Image")


# BUILD GUI
root = tk.Tk()
root.title("Wavelet Denoising Tool")
root.geometry("900x650")

label = tk.Label(root, text="MSEF Pipeline Stage #1: Wavelet Denoising", font=("Arial", 16))
label.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack()

choose_button = tk.Button(button_frame, text="Choose Image", command=choose_file)
choose_button.grid(row=0, column=0, padx=10)

sample_button = tk.Button(button_frame, text="Load Sample Image", command=load_sample_image)
sample_button.grid(row=0, column=1, padx=10)

show_graphs_button = tk.Button(button_frame, text="Show Graphs", state=tk.DISABLED,
                               command=lambda: show_analysis_graphs(current_original, current_denoised))
show_graphs_button.grid(row=0, column=2, padx=10)

# Save button 
save_button = tk.Button(button_frame, text="Save Denoised Image", state=tk.DISABLED, command=save_image)
save_button.grid(row=0, column=3, padx=10)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=500, mode="determinate")
progress_bar.pack(pady=10)

status_label = tk.Label(root, text="Idle", font=("Arial", 10))
status_label.pack(pady=10)

image_frame = tk.Frame(root)
image_frame.pack(pady=10)

img_panel_original = tk.Label(image_frame)
img_panel_original.grid(row=0, column=0, padx=15)

img_panel_denoised = tk.Label(image_frame)
img_panel_denoised.grid(row=0, column=1, padx=15)

root.mainloop()
