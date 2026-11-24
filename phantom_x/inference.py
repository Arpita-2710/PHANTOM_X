import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
from torchvision import transforms
from PIL import Image, ExifTags

from phantom_x.model import build_model

DEFAULT_CHECKPOINT_PATH = Path("outputs/phantom_x_resnet18_best.pth")

# Single standard class definitions
CLASS_MAPPING = {
    "FAKE": {"index": 0, "display_name": "AI-GENERATED", "color": "#ff3860"},
    "REAL": {"index": 1, "display_name": "REAL", "color": "#00d1b2"}
}

def get_transforms() -> transforms.Compose:
    """Exact preprocessing transform pipeline used during training and evaluation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_forensic_model(checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Loads the ResNet-18 forensic model and checkpoint metadata on CPU."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Please ensure 'outputs/phantom_x_resnet18_best.pth' is present."
        )

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = build_model(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    metadata = {
        "architecture": checkpoint.get("architecture", "ResNet18"),
        "input_size": checkpoint.get("input_size", 224),
        "class_to_idx": checkpoint.get("class_to_idx", {"FAKE": 0, "REAL": 1}),
        "checkpoint_path": str(checkpoint_path)
    }

    return model, metadata

def extract_image_metadata(image: Image.Image, filename: str, file_size_bytes: int) -> Dict[str, Any]:
    """Extracts forensic metadata including dimensions, format, color mode, and EXIF attributes."""
    width, height = image.size
    aspect_ratio = round(width / height, 3) if height > 0 else 1.0
    megapixels = round((width * height) / 1_000_000, 2)
    file_size_kb = round(file_size_bytes / 1024, 2)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 3)

    exif_data: Dict[str, str] = {}
    has_exif = False
    try:
        raw_exif = image.getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                # Only keep human-readable string/number tags of reasonable length
                val_str = str(value)
                if len(val_str) < 100:
                    exif_data[tag_name] = val_str
            if exif_data:
                has_exif = True
    except Exception:
        pass

    return {
        "filename": filename,
        "format": image.format or "UNKNOWN",
        "mode": image.mode,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "megapixels": megapixels,
        "file_size_bytes": file_size_bytes,
        "file_size_kb": file_size_kb,
        "file_size_mb": file_size_mb,
        "has_exif": has_exif,
        "exif_data": exif_data
    }

def run_inference(
    model: torch.nn.Module,
    image: Image.Image,
    model_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes forensic inference on the input image using CPU PyTorch.
    
    Returns:
        Dict containing predicted class, model scores, latency, and detailed output.
    """
    start_time = time.perf_counter()
    device = torch.device("cpu")

    # Safe RGB conversion handling transparent alpha channels
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        rgba_image = image.convert("RGBA")
        rgb_image.paste(rgba_image, mask=rgba_image.split()[3])
    else:
        rgb_image = image.convert("RGB")

    transform = get_transforms()
    tensor_input = transform(rgb_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1)[0]

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    class_to_idx = model_metadata.get("class_to_idx", {"FAKE": 0, "REAL": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    fake_idx = class_to_idx.get("FAKE", 0)
    real_idx = class_to_idx.get("REAL", 1)

    fake_prob = float(probs[fake_idx].item())
    real_prob = float(probs[real_idx].item())

    pred_idx = int(probs.argmax().item())
    pred_raw_class = idx_to_class.get(pred_idx, "UNKNOWN")

    if pred_raw_class == "FAKE":
        display_label = "AI-GENERATED"
        confidence_score = fake_prob
        indicator = "AI-GENERATED"
        theme_color = "#ff3860"
    else:
        display_label = "REAL"
        confidence_score = real_prob
        indicator = "REAL"
        theme_color = "#00d1b2"

    return {
        "predicted_label": display_label,
        "raw_class": pred_raw_class,
        "predicted_index": pred_idx,
        "model_score_pct": round(confidence_score * 100, 2),
        "scores": {
            "AI-GENERATED": round(fake_prob * 100, 2),
            "REAL": round(real_prob * 100, 2)
        },
        "raw_probabilities": {
            "FAKE": fake_prob,
            "REAL": real_prob
        },
        "theme_color": theme_color,
        "latency_ms": latency_ms,
        "device": "CPU"
    }
