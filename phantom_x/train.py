import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not fetch ImageNet pretrained weights.")
    args = parser.parse_args()

    data = Path(args.data)
    required = [data/"train", data/"val"]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Create data/train/REAL, data/train/FAKE, "
                "data/val/REAL and data/val/FAKE first."
            )

    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(data/"train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data/"val", transform=eval_tfms)
    print("Class mapping:", train_ds.class_to_idx)
    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError("Train and validation class folders must match.")
    if len(train_ds.classes) != 2:
        raise ValueError("Expected exactly two class folders, e.g. FAKE and REAL.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cpu":
        print("CPU training selected. Start with --epochs 1 for a quick smoke test.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0
    Path("outputs").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = correct / max(total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={running_loss/max(total,1):.4f} "
              f"train_acc={train_acc:.4f} "
              f"val_loss={val_loss/max(val_total,1):.4f} "
              f"val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "architecture": "ResNet18",
                "input_size": 224
            }, "outputs/phantom_x_resnet18_best.pth")
            print("Saved best checkpoint.")

    print("Training complete. Best validation accuracy:", round(best_acc, 4))
    print("Checkpoint: outputs/phantom_x_resnet18_best.pth")

if __name__ == "__main__":
    main()
