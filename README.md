# PHANTOM X
### AI-Powered Multimodal Digital Forensics and Incident Reconstruction System

PHANTOM X is a research-oriented project aimed at developing an intelligent digital forensics platform that analyses digital evidence and assists in identifying manipulated or AI-generated content.

## Current Development: Image Authenticity Detection

The first development module focuses on distinguishing between real-world photographs and AI-generated images using deep learning.

### Technology Stack
- Python
- PyTorch
- Torchvision
- ResNet18
- Pillow
- NumPy

### Current Implementation
- Image dataset preparation and organisation
- Image preprocessing and normalisation
- ResNet18-based binary classification
- ImageNet transfer learning
- Model training and checkpoint saving
- Individual image prediction
- Model evaluation using a confusion matrix

### Initial Dataset

| Category | Training | Validation |
|---|---:|---:|
| REAL | 80 | 20 |
| FAKE | 80 | 20 |
| Total | 160 | 40 |

Dataset source: Hugging Face – LuckyCow/Ai_vs_real_web_images

### Initial Experimental Results

- Model: ResNet18
- Training epochs: 5
- Training accuracy: 96.88%
- Validation accuracy: 60%
- Evaluation images: 40

These are preliminary results from a small dataset. Further training and evaluation are required to improve generalisation and reduce overfitting.

### Project Structure

```text
PHANTOM_X/
├── phantom_x/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── check_environment.py
├── data/                 # Local dataset; not tracked
├── outputs/              # Local model checkpoints
├── requirements.txt
├── README.md
└── .gitignore
```

### Future Development
1. Expand and diversify the image dataset.
2. Improve model generalisation and reduce overfitting.
3. Compare different deep-learning architectures.
4. Develop image manipulation detection and localisation.
5. Build an image-upload and analysis interface.
6. Integrate the image module into the larger PHANTOM X platform.

### Dataset and Checkpoint Information

Datasets and trained model checkpoints are maintained separately from this GitHub repository. Refer to the project documentation for dataset access and setup instructions.

**Project Status:** Initial image classification prototype implemented; further research and development ongoing.
