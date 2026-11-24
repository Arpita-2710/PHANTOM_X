PHANTOM X - IMAGE AUTHENTICITY CLASSIFIER (STARTER IMPLEMENTATION)

Model:
- ResNet18 CNN
- Transfer learning (ImageNet pretrained weights by default)
- Two classes: REAL and FAKE (AI-GENERATED)
- PyTorch framework
- Input size: 224 x 224

IMPORTANT:
This package contains the model architecture and training/inference code.
It is NOT a trained detector yet. A trained checkpoint is created only after
running train.py with labelled images. Do not report accuracy until evaluated.

1) Install:
   py -m pip install -r requirements.txt

2) Expected data folder structure:
   data/
     train/
       FAKE/   (AI-generated training images)
       REAL/   (camera/real training images)
     val/
       FAKE/
       REAL/

   Optional test set:
     data/
       test/
         FAKE/
         REAL/

   Keep test images separate; do not train on them.

3) Check hardware:
   py check_environment.py

4) Train:
   py train.py --data data --epochs 5 --batch-size 16

   If pretrained-weight download is unavailable, use:
   py train.py --data data --epochs 5 --batch-size 16 --no-pretrained

5) Prediction after training:
   py predict.py "path\\to\\image.jpg"

Output checkpoint:
   outputs/phantom_x_resnet18_best.pth

For a quick code/data pipeline check, use a small labelled subset and:
   py train.py --data data --epochs 1 --batch-size 8

Research cautions:
- Dataset-source bias can make a model distinguish dataset origins rather than
  AI-generation traces. Use diverse sources and an independent test set.
- Split by source/original image where possible; remove duplicates.
- Report precision, recall, F1, confusion matrix and cross-generator results,
  not accuracy alone.
- A model prediction is not forensic proof. Include an inconclusive/review
  path in the final application.
