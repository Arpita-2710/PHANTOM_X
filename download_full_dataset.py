
import time
from pathlib import Path

from datasets import load_dataset

# Configuration
DATASET_NAME = "LuckyCow/Ai_vs_real_web_images"
OUTPUT_DIR = Path("raw_sample_full")
FAILED_LOG = Path("failed_images.txt")
MAX_RETRIES = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load dataset with retry handling."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print("Connecting to Hugging Face...")
            dataset = load_dataset(
                DATASET_NAME,
                split="train"
            )
            print("Dataset loaded successfully!")
            return dataset

        except Exception as error:
            print(f"Loading failed ({attempt}/{MAX_RETRIES}): {error}")
            time.sleep(5 * attempt)

    raise RuntimeError("Could not load dataset after retries.")


def get_category(label):
    """Map dataset labels into real and ai folders."""
    label = str(label).lower()

    if "real" in label:
        return "real"

    if any(word in label for word in ["ai", "fake", "synthetic", "generated"]):
        return "ai"

    raise ValueError(f"Unknown dataset label: {label}")


def main():

    dataset = load_data()

    print("Total images:", len(dataset))
    print("Dataset features:", dataset.features)

    label_names = dataset.features["label"].names

    print("Available labels:", label_names)

    failed_indices = []

    for i in range(len(dataset)):

        # Retry only images that have not already been saved
        try:
            item = dataset[i]
            category = get_category(label_names[item["label"]])
        except Exception as error:
            print(f"Could not read image {i}: {error}")
            failed_indices.append(i)
            continue

        folder = OUTPUT_DIR / category
        folder.mkdir(parents=True, exist_ok=True)

        destination = folder / f"{i:06d}.jpg"

        # Resume: skip existing images
        if destination.exists():
            continue

        saved = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                image = item["image"].convert("RGB")
                image.save(destination, format="JPEG", quality=95)

                saved = True
                break

            except Exception as error:
                print(
                    f"Image {i} failed "
                    f"({attempt}/{MAX_RETRIES}): {error}"
                )
                time.sleep(3 * attempt)

                # Reload the row in case the connection failed
                if attempt < MAX_RETRIES:
                    try:
                        item = dataset[i]
                    except Exception:
                        pass

        if not saved:
            failed_indices.append(i)
            print(f"Could not save image {i}. Will retry later.")

        if (i + 1) % 500 == 0:
            real_count = len(list((OUTPUT_DIR / "real").glob("*.jpg")))
            ai_count = len(list((OUTPUT_DIR / "ai").glob("*.jpg")))

            print(
                f"\nProgress: {i + 1}/{len(dataset)}"
                f"\nREAL saved: {real_count}"
                f"\nAI saved: {ai_count}\n"
            )

    # Save failed image indices for another retry
    with FAILED_LOG.open("w", encoding="utf-8") as file:
        for index in failed_indices:
            file.write(f"{index}\n")

    real_count = len(list((OUTPUT_DIR / "real").glob("*.jpg")))
    ai_count = len(list((OUTPUT_DIR / "ai").glob("*.jpg")))

    print("\n========== EXTRACTION SUMMARY ==========")
    print("Total dataset records:", len(dataset))
    print("REAL images saved:", real_count)
    print("AI images saved:", ai_count)
    print("Total images saved:", real_count + ai_count)
    print("Failed records:", len(failed_indices))
    print("Output folder:", OUTPUT_DIR.resolve())
    print("Failed image log:", FAILED_LOG.resolve())


if __name__ == "__main__":
    main()
