# Video Anomaly Detection System

AI-powered system for detecting unusual events in surveillance videos. Upload a video and the system automatically identifies frames where something abnormal is happening.

---

## What Does This Do?

This system analyzes surveillance videos frame-by-frame to detect anomalies — unusual events that differ from normal patterns. Examples include:

- **Unusual movement patterns** (running, erratic behavior)
- **Unexpected objects** (vehicles where they shouldn't be, abandoned items)
- **Abnormal crowd density** (sudden gatherings or empty spaces)
- **Irregular activities** (people in restricted areas, unusual gestures)

**How it works:** The AI model learns what "normal" looks like from training data. When it sees something different, it flags it as an anomaly based on how poorly it can reconstruct the unusual frame.

---

## System Architecture

The system uses a **convolutional autoencoder** — a neural network trained to reconstruct normal surveillance footage.

```
Video Upload
    ↓
Frame Extraction (OpenCV)
    ↓
Preprocessing (Grayscale, 64×64 resize)
    ↓
AI Model (Autoencoder)
    ↓
Reconstruction Error Calculation
    ↓
Threshold Comparison
    ↓
Anomaly Flags + Scores
```

**Key Concept:** The model learns to recreate "normal" frames accurately. When it encounters something unusual, the reconstruction quality drops — this error spike indicates an anomaly.

**Technical Details:**
- **Input:** 64×64 grayscale frames
- **Architecture:** Encoder (compress) → Latent space (256-dim) → Decoder (reconstruct)
- **Output:** Per-frame reconstruction error (0.0–1.0 scale)
- **Threshold:** Statistical cutoff (95th percentile of validation set errors)

---

## Project Structure

```
├── app.py                    # FastAPI web service (backend API)
├── dashboard.py              # Streamlit interactive UI (frontend)
├── main.py                   # Training and evaluation pipeline
├── config.py                 # Configuration management
├── settings.py               # Application settings
├── requirements.txt          # Python dependencies
├── models/
│   ├── autoencoder.py        # Neural network architecture
│   └── detector.py           # Training and inference logic
├── data/
│   ├── preprocessing.py      # Video frame extraction & processing
│   ├── dataset.py            # Data loading utilities
│   └── synthetic_data.py     # Synthetic data generation
├── evaluation/
│   ├── metrics.py            # Performance evaluation (Precision, Recall, AUC)
│   └── visualizer.py         # Result visualization
├── utils/
│   ├── logging_utils.py      # Structured logging
│   ├── gpu_utils.py          # GPU memory monitoring
│   └── file_utils.py         # File handling utilities
└── outputs/
    └── trained_model.pth     # Pre-trained model weights
```

---

## How to Run

### Requirements
- Python 3.10+
- 2GB disk space
- Optional: NVIDIA GPU for faster processing

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start API backend (Terminal 1)
python app.py
# API available at http://localhost:8000

# Launch dashboard (Terminal 2)
streamlit run dashboard.py
# Dashboard opens at http://localhost:8501
```

---

## Features

### Interactive Dashboard
- Drag-and-drop video upload
- Interactive timeline showing reconstruction errors
- Real-time threshold adjustment — change sensitivity without reprocessing
- Frame viewer — inspect specific anomalies
- Export results to JSON or CSV

### REST API
- Simple POST request for video analysis
- JSON response with per-frame anomaly scores
- Adjustable thresholds via API endpoints
- Swagger documentation at `/docs`

### Threshold Presets

| Preset | Anomaly Rate | Best For |
|--------|--------------|----------|
| Conservative | 5% | Minimizing false alarms |
| Balanced | 10% | General surveillance (default) |
| Moderate | 25% | High-sensitivity monitoring |
| Sensitive | 40% | Maximum detection |

---

## API Usage

### Analyze Video

```http
POST /analyze-video
Content-Type: multipart/form-data
```

**Response:**
```json
{
  "frame_count": 60,
  "anomaly_count": 8,
  "anomaly_rate": 0.13,
  "anomaly_scores": [0.002, 0.008, 0.012],
  "processing_time": 0.85,
  "model_info": {
    "device": "cuda",
    "threshold": 0.005069
  }
}
```

### Set Threshold Preset

```http
POST /set-threshold-preset
Content-Type: application/json

{
  "preset": "balanced"
}
```

---

## Training the Model

### Using UCSD Ped2 Dataset
```bash
python main.py --mode ucsd --dataset_name ped2 \
    --data_path data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/ \
    --epochs 50
```

### Using Custom Footage
```bash
python main.py --mode custom \
    --data_path data/my_cameras/normal_behavior/ \
    --epochs 50 \
    --batch_size 64
```

---

## Model Performance

- **Precision:** 92.47%
- **Recall:** 83.78%
- **F1 Score:** 87.91%
- **AUC:** 0.7438

---

## Configuration

Default settings work for most cases. Customize via environment variables:

```bash
APP_MAX_FILE_SIZE_MB=100
APP_MAX_VIDEO_DURATION_SEC=300
APP_BATCH_SIZE=64
APP_DEVICE=cuda
APP_THRESHOLD=0.005069
```

---

## Technologies Used

- **PyTorch** — Deep learning framework
- **FastAPI** — REST API framework
- **Streamlit** — Dashboard framework
- **OpenCV** — Video processing
- **Plotly** — Interactive visualizations
