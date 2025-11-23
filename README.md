MSEF 2025 – Image Processing Pipeline

This project contains a multi-stage image-processing workflow built for the 2025 MSEF astronomy project. The goal is to take raw telescope images, clean them, enhance them, and export presentation-ready results. Each script handles a specific part of the process.

📌 Project Structure

The pipeline is divided into four stages. You can run them in order or use each file on its own.

Pipeline_Stage_#1.py   → Load + prepare data  
Pipeline_Stage_#2.py   → Noise reduction + calibration  
Pipeline_Stage_#3.py   → Contrast + enhancement + formatting  
Pipeline_Stage_#4.py   → Visualization + final outputs

🔭 Stage-by-Stage Overview
Stage 1 — Data Loading & Preparation

File: Pipeline_Stage_#1.py
What it does:

Loads raw input images (FITS or high-bit depth formats).

Converts them into NumPy arrays.

Applies initial cropping, scaling, or orientation fixes.

This stage ensures the data is in a clean, consistent format before any heavy processing.

Stage 2 — Noise Reduction & Calibration

File: Pipeline_Stage_#2.py
What it does:

Removes sensor noise and hot pixels.

Applies dark frame, flat frame, or bias corrections if supplied.

Smooths the image while keeping important details intact.

After this stage, the image is noticeably cleaner and ready for enhancement.

Stage 3 — Image Enhancement

File: Pipeline_Stage_#3.py
What it does:

Adjusts contrast and brightness.

Clips or normalizes the array values.

Converts processed arrays into images using Pillow (Image.fromarray).

⚠️ Note: You may see a Pillow DeprecationWarning about the mode parameter. It won’t affect current output, but Pillow plans to remove it in 2026.

This is usually the slowest stage because it handles the bulk of processing on large image arrays.

Stage 4 — Final Output & Visualization

File: Pipeline_Stage_#4.py
What it does:

Creates the final PNG/JPEG files.

Applies optional false-color or composite techniques.

Outputs files ready for reports or presentations.

This transforms the processed data into the final images you’ll use.

▶️ How to Run the Pipeline

Place your raw images in the project’s data folder (or whichever directory the scripts point to).

Run each stage in order:

python Pipeline_Stage_#1.py
python Pipeline_Stage_#2.py
python Pipeline_Stage_#3.py
python Pipeline_Stage_#4.py


Processed images will appear in the output folder.

You can also run any stage individually if you only need that part.

🎯 Project Goals

Create clean and scientifically useful astronomical images.

Build a pipeline that’s modular, readable, and easy for anyone to follow.

Produce consistent outputs for analysis and presentation.
