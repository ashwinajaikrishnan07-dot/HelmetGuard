# 🚦 HelmetGuard AI

**AI-Powered Traffic Helmet Violation Detection**

HelmetGuard AI is a Streamlit web app that uses a trained YOLOv8s object detection model to find and localize riders who are not wearing helmets in traffic images. Upload an image and the app counts the detected violations, draws a bounding box around each one, and lets you export the results.

---

## How It Works

```
Upload Image
    ↓
YOLOv8s (best.pt) · imgsz=960 · conf=0.10 · iou=0.45 · augment=True
    ↓
Raw YOLO predictions
    ↓
Class-specific filtering  →  no_helmet ≥ 0.45 · helmet ≥ 0.30
    ↓
Filtered detections
    ↓
Bounding boxes · Counts · Metrics · CSV · Annotated image
```

YOLO runs with a low inference threshold, so weak predictions are still examined. The app then applies stricter, per-class display thresholds, and only the filtered detections are counted, drawn, or exported.

## Features

- **Violation detection**: counts confident `no_helmet` detections as *Detected Helmet Violations*
- **Localization**: red boxes for **NO HELMET** and green boxes for **HELMET**, each labelled with its confidence
- **Side-by-side comparison**: original image next to the AI detection result
- **Dashboard metrics**: violations, average confidence, locations found, and total raw detections
- **Violation details**: confidence and pixel coordinates for each detection
- **Exports**: download the annotated PNG and a CSV violation report
- **Debug tools**: a collapsible raw YOLO predictions table and model info in the sidebar

## Model

| Property        | Value                        |
|-----------------|------------------------------|
| Architecture    | YOLOv8s                      |
| Task            | Object Detection             |
| Classes         | `0: helmet`, `1: no_helmet`  |
| Training size   | 640                          |
| Inference size  | 960                          |
| Weights         | `model/best.pt`              |

The app checks the model's classes on startup and stops with an error if they don't match.

## Project Structure

```
HelmetGuard/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
└── model/
    └── best.pt         # Trained YOLOv8s weights
```

## Getting Started

**Requirements:** Python 3.10 or newer (tested on 3.12)

```bash
git clone https://github.com/ashwinajaikrishnan07-dot/HelmetGuard.git
cd HelmetGuard

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## Usage

1. Upload a traffic image (JPG, JPEG or PNG).
2. Click **Analyze Image**.
3. Review the result card, metrics and annotated image.
4. Download the annotated image or the CSV report.

### Result States

| State | Meaning |
|-------|---------|
| 🚨 **Helmet Violation Detected** | At least one `no_helmet` detection has confidence ≥ 0.45 |
| 🔎 **No No-Helmet Detection Found** | Objects were detected, but no `no_helmet` detection reached the threshold |
| ⚠️ **No Objects Detected** | YOLO returned no predictions at the selected threshold |

### Settings

| Setting | Default | Notes |
|---------|---------|-------|
| Confidence Threshold | 0.10 | YOLO inference threshold (0.10 to 0.90) |
| Show helmet detections | On | Draws green helmet boxes |
| Show confidence scores | On | Adds percentages to box labels |

The class-specific display thresholds (0.45 for no-helmet, 0.30 for helmet) are fixed in `app.py`, so the slider cannot lower them.

## CSV Report Format

| Violation ID | Class | Confidence | X1 | Y1 | X2 | Y2 |
|---|---|---|---|---|---|---|
| 1 | no_helmet | 0.7741 | 889 | 37 | 1073 | 293 |

Coordinates are in pixels relative to the original uploaded image.

## Limitations

- Results are *potential* violations based on model predictions, not confirmed legal violations.
- *Total Detected Objects* counts all raw predictions across classes. It is **not** a rider count.
- Accuracy depends on image quality, rider size, occlusion and lighting.

## Tech Stack

Python · Streamlit · Ultralytics YOLO · PyTorch · OpenCV · Pillow · NumPy · Pandas

---

<p align="center"><b>HelmetGuard AI</b> • Deep Learning-Based Traffic Safety System<br>Powered by YOLO Object Detection</p>
