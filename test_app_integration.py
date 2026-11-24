from pathlib import Path
from PIL import Image, ImageDraw
import json
from phantom_x.inference import load_forensic_model, run_inference, extract_image_metadata

def run_tests():
    print("=" * 60)
    print("PHANTOM X INTEGRATION TEST SUITE")
    print("=" * 60)

    # 1. Checkpoint verification
    model, metadata = load_forensic_model()
    print("Checkpoint loaded successfully!")
    print(f"Architecture: {metadata['architecture']}")
    print(f"Input size: {metadata['input_size']}")
    print(f"Class mapping: {metadata['class_to_idx']}")
    print("-" * 60)

    test_images = [
        ("VAL REAL 1", "data/val/REAL/080_007532_2f1bef11.jpg"),
        ("VAL REAL 2", "data/val/REAL/081_008293_62d881b0.jpg"),
        ("VAL FAKE 1", "data/val/FAKE/080_020450_9d55e017.png"),
        ("VAL FAKE 2", "data/val/FAKE/081_003606_89a6726b.jpg"),
    ]

    # Create an external sample image to test external image handling
    external_path = Path("external_sample_test.jpg")
    img_ext = Image.new("RGB", (640, 480), color=(120, 160, 200))
    d = ImageDraw.Draw(img_ext)
    d.rectangle([(50, 50), (590, 430)], outline=(255, 255, 255), width=3)
    d.text((80, 80), "External Photography Test Sample", fill=(255, 255, 255))
    img_ext.save(external_path, "JPEG", quality=90)
    test_images.append(("EXTERNAL IMAGE", str(external_path)))

    results = []
    for tag, img_path in test_images:
        path = Path(img_path)
        if not path.exists():
            print(f"Error: {path} not found!")
            continue

        file_bytes = path.stat().st_size
        img = Image.open(path)
        meta = extract_image_metadata(img, path.name, file_bytes)
        inf_result = run_inference(model, img, metadata)

        entry = {
            "test_case": tag,
            "filename": path.name,
            "resolution": f"{meta['width']}x{meta['height']}",
            "format": meta["format"],
            "file_size_kb": meta["file_size_kb"],
            "prediction": inf_result["predicted_label"],
            "score_pct": inf_result["model_score_pct"],
            "ai_score_pct": inf_result["scores"]["AI-GENERATED"],
            "real_score_pct": inf_result["scores"]["REAL"],
            "latency_ms": inf_result["latency_ms"]
        }
        results.append(entry)

        print(f"[{tag}] File: {path.name}")
        print(f"  Dimensions: {entry['resolution']} | Format: {entry['format']} | Size: {entry['file_size_kb']} KB")
        print(f"  Predicted Label: {entry['prediction']}")
        print(f"  Model Score: {entry['score_pct']}% (AI: {entry['ai_score_pct']}%, Real: {entry['real_score_pct']}%)")
        print(f"  Inference Latency: {entry['latency_ms']} ms (CPU)")
        print("-" * 60)

    print("\nSummary Table:")
    print(f"{'Test Case':<16} | {'Filename':<26} | {'Prediction':<14} | {'Score':<8} | {'Latency':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['test_case']:<16} | {r['filename']:<26} | {r['prediction']:<14} | {r['score_pct']:<7}% | {r['latency_ms']:<6} ms")

if __name__ == "__main__":
    run_tests()
