import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--checkpoint", default="outputs/phantom_x_resnet18_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    image = Image.open(args.image).convert("RGB")
    x = tfm(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pred = int(probs.argmax())
    print("Prediction:", idx_to_class[pred])
    print("Model score:", f"{probs[pred].item()*100:.2f}%")
    print("NOTE: This is a model score, not proof of authenticity.")

if __name__ == "__main__":
    main()
