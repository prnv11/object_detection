# ECE304 - AI & Machine Learning: Assignment 5 (YOLO11n Object Detection)

**Student Name:** Pranav Talwar  
**Roll No:** 2310110555  

---

## 📌 Project Overview
This repository contains **Assignment 5** for the ECE304 Artificial Intelligence and Machine Learning course. The project explores Object Detection using the state-of-the-art **YOLO11n** model provided by Ultralytics.

The assignment is divided into two primary tasks:
1. **Task 1 – Multi-Object Detection using Pre-trained YOLO11n:**  
   Evaluates out-of-the-box multi-object detection capability using YOLO11n pre-trained on the MS COCO dataset on custom test images.
2. **Task 2 – Custom Fine-Tuning for Signature Detection:**  
   Fine-tunes the YOLO11n architecture on a custom handwritten signature dataset, evaluating detection performance across intermediate and best-performing trained epochs.

---

## 📁 Repository Structure
```text
.
├── 2310110555_Assignment5.ipynb   # Main Jupyter Notebook with code & results
├── test_images/                   # Sample images used for Task 1 inference
│   ├── table_chairs.png           # Dining set (furniture detection)
│   ├── man_dog.png                # Person walking a dog
│   ├── signatures.png             # Handwritten signatures grid
│   ├── coffee_table.png           # Living room setup
│   └── man_dogs_snow.png          # Person with two dogs in snow
├── signature_dataset/             # Custom signature dataset folder
└── README.md                      # Project documentation
```

---

## 🚀 Setup & Execution

### Prerequisites
Make sure Python 3.8+ and `pip` are installed. Install required packages:
```bash
pip install torch timm ultralytics albumentations matplotlib opencv-python
```

### Running the Notebook
You can run the notebook locally using Jupyter Notebook / JupyterLab or open it in Google Colab:
```bash
jupyter notebook 2310110555_Assignment5.ipynb
```

---

## 🛠️ Workflow Highlights

### Task 1: Pre-trained Inference
```python
from ultralytics import YOLO

# Load pre-trained YOLO11n
model = YOLO("yolo11n.pt")

# Run inference on test images
results = model("test_images/man_dog.png")
results[0].show()
```

### Task 2: Fine-Tuning YOLO11n
```python
# Fine-tune model on custom signature dataset
model.train(data="signature_dataset/data.yaml", epochs=50, imgsz=640)
```

---

## 📄 License
This repository is submitted as part of academic coursework for ECE304 at Shiv Nadar Institution of Eminence (SNIoE).
