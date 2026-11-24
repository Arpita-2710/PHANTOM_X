# PHANTOM X: Digital Image Forensics Platform

**AI-Powered Image Authenticity Detection (REAL vs. AI-GENERATED)**

PHANTOM X is a research-oriented digital image forensics platform built with PyTorch and Streamlit. The system evaluates single-frame digital images to distinguish between authentic camera captures and synthetic AI-generated media (from generators such as Midjourney, Stable Diffusion, and DALL-E).

---

## 1. System Architecture

- **Classifier Backbone**: ResNet-18 (ImageNet pretrained initial transfer learning weights)
- **Classification Space**: Binary:
  - `0`: `AI-GENERATED` (Internal code label: `FAKE`)
  - `1`: `REAL` (Internal code label: `REAL`)
- **Input Resolution**: $224 \times 224$ pixels, ImageNet mean/standard deviation normalization
- **Inference Runtime**: CPU-compatible offline execution
- **Checkpoint Location**: `outputs/phantom_x_resnet18_best.pth`
- **Frontend**: Streamlit with custom Cyber-Forensics dark UI

---

## 2. Installation & Setup

Ensure Python 3.11+ is installed.

```powershell
# 1. Navigate to the project root
cd "PHANTOM_X_Image_Model"

# 2. Install dependencies
pip install -r requirements.txt
```

### Verified Dependencies (`requirements.txt`):
- `torch`
- `torchvision`
- `pillow`
- `numpy`
- `streamlit`

---

## 3. Running the Application

Launch the Streamlit forensic frontend with:

```powershell
streamlit run app.py
```

The application will start locally at:
```text
Local URL: http://localhost:8501
Network URL: http://<local-ip>:8501
```

---

## 4. Application Features

1. **Cyber-Forensics Dashboard (Home)**:
   - System specifications, operational scope, and active checkpoint status.
2. **Interactive Image Analysis**:
   - **Upload**: Drag-and-drop support for `JPG`, `JPEG`, `PNG`, and `WEBP` (up to 20MB).
   - **Built-in Validation Samples**: One-click testing using actual dataset samples (`data/val/REAL` and `data/val/FAKE`) for immediate supervisor demonstrations.
   - **Real-time Diagnostic Telemetry**: Image dimensions, color mode, aspect ratio, file size, and latency in milliseconds.
   - **EXIF Analysis**: Inspection of camera make, model, software, and metadata headers.
   - **Visual Result Cards**: Distinct color-coded forensic badges (Green for REAL, Crimson for AI-GENERATED) and class probability progress bars.
3. **Session Analysis History**:
   - Maintains continuous scan telemetry throughout the session.
   - Summary metrics (total scans, synthetic count, real count).
   - One-click CSV export of forensic session audit logs.
   - Session reset capability.

---

## 5. Verification & Testing

The model integration was verified using the existing trained checkpoint `outputs/phantom_x_resnet18_best.pth`:

| Input Image | Actual Label | Model Prediction | Model Score | Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- |
| `data/val/REAL/080_007532_2f1bef11.jpg` | REAL Camera | **AI-GENERATED** | 50.83% | ~558 ms |
| `data/val/REAL/081_008293_62d881b0.jpg` | REAL Camera | **REAL** | 92.48% | ~160 ms |
| `data/val/FAKE/080_020450_9d55e017.png` | AI-Generated | **AI-GENERATED** | 93.61% | ~170 ms |
| `data/val/FAKE/081_003606_89a6726b.jpg` | AI-Generated | **AI-GENERATED** | 96.12% | ~165 ms |

*(Note: The starter baseline ResNet-18 model was trained on only 160 images; predictions reflect real model output and demonstrate known boundary behavior).*

---

## 6. Scientific & Forensic Disclaimer

> [!WARNING]
> Model scores produced by PHANTOM X represent uncalibrated experimental neural network indicators, **not definitive proof of authenticity or synthetic fabrication**. Digital forensics requires cross-verifying camera sensor PRNU, Fourier domain frequency spectra, metadata integrity, and multi-generator generalizability.
