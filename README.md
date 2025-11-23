**MSEF 2025 – Image Processing Pipeline**
This project contains a multi-stage image-processing workflow built for the 2025 MSEF astronomy project. The goal is to take raw telescope images, clean them, enhance them, and export presentation-ready results. Each script handles a specific part of the process.

📌 **Project Structure**
The pipeline is divided into four stages. You can run them in order or use each file on its own.

Pipeline_Stage_#1.py   → Load + prepare data  
Pipeline_Stage_#2.py   → Noise reduction + calibration  
Pipeline_Stage_#3.py   → Contrast + enhancement + formatting  
Pipeline_Stage_#4.py   → Visualization + final outputs

▶️ **How to Run the Pipeline**
Place your raw images in the project’s data folder (or whichever directory the scripts point to).

Run each stage in order:
python Pipeline_Stage_#1.py
python Pipeline_Stage_#2.py
python Pipeline_Stage_#3.py
python Pipeline_Stage_#4.py

Processed images will appear in the output folder.
You can also run any stage individually if you only need that part.

🎯 **Project Goals**

1. Create clean and scientifically useful astronomical images.
2. Build a pipeline that’s modular, readable, and easy for anyone to follow.
3. Produce consistent outputs for analysis and presentation.

⚠️ Note: You may see a Pillow DeprecationWarning about the mode parameter. It won’t affect current output, but Pillow plans to remove it in 2026.
