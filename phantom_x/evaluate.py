
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import build_model

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder("data/val", transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

checkpoint = torch.load(
    "outputs/phantom_x_resnet18_best.pth",
    map_location=device
)

model = build_model(pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

correct = 0
total = 0
tp = tn = fp = fn = 0

# Class mapping: FAKE=0, REAL=1
with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        for actual, predicted in zip(labels, predictions):
            actual = actual.item()
            predicted = predicted.item()

            total += 1
            correct += int(actual == predicted)

            if actual == 0 and predicted == 0:
                tp += 1
            elif actual == 1 and predicted == 1:
                tn += 1
            elif actual == 1 and predicted == 0:
                fp += 1
            else:
                fn += 1

print("\nPHANTOM X EVALUATION")
print("--------------------")
print("Total images:", total)
print("Correct:", correct)
print("Accuracy:", round(100 * correct / total, 2), "%")
print("\nConfusion Matrix (actual rows, predicted columns)")
print("                 Pred FAKE   Pred REAL")
print("Actual FAKE:       ", tp, "        ", fn)
print("Actual REAL:       ", fp, "        ", tn)
