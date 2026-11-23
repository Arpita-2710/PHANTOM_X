
from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
import random
import shutil

REPO = "LuckyCow/Ai_vs_real_web_images"
SAMPLE_SIZE = 100

print("Inspecting dataset file list...")
api = HfApi()
files = api.list_repo_files(REPO, repo_type="dataset")

extensions = (".jpg", ".jpeg", ".png", ".webp")

real_files = [
    f for f in files
    if f.lower().startswith("real/") and f.lower().endswith(extensions)
]

ai_files = [
    f for f in files
    if f.lower().startswith("ai/") and f.lower().endswith(extensions)
]

print("Available real images:", len(real_files))
print("Available AI images:", len(ai_files))

if len(real_files) < SAMPLE_SIZE or len(ai_files) < SAMPLE_SIZE:
    raise RuntimeError("Not enough images in one of the classes.")

random.seed(42)
selected_real = random.sample(real_files, SAMPLE_SIZE)
selected_ai = random.sample(ai_files, SAMPLE_SIZE)
selected = selected_real + selected_ai

print("Downloading only 200 selected images...")

local_dir = snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    allow_patterns=selected,
    local_dir="raw_sample",
    max_workers=8
)

for category, selected_files in [
    ("REAL", selected_real),
    ("FAKE", selected_ai)
]:
    random.shuffle(selected_files)

    for i, relative_path in enumerate(selected_files):
        split = "train" if i < 80 else "val"

        source = Path(local_dir) / relative_path
        destination = (
            Path("data") / split / category /
            f"{i:03d}_{source.name}"
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

print("\nSample preparation completed!")
print("REAL: 80 train + 20 validation")
print("FAKE: 80 train + 20 validation")
print("Dataset location: data/")
