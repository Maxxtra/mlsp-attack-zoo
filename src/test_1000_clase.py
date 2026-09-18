"""Does evaluating on 10 classes hide the success of a one-step attack?

The grid slices the 1000 ImageNet logits down to the 10 Imagenette classes, so a
prediction that moved to some other ImageNet class still counts as correct. This
runs the same attack and scores the same images both ways.

    python src/test_1000_clase.py
    python src/test_1000_clase.py --model convnext --eps 16/255

Works one batch at a time: 200 images at 224x224 with gradients do not fit on a T4.
"""
import argparse
import torch
import torchattacks
from data import get_loader, IMAGENETTE_TO_IMAGENET
from models import load
from run_attack import parse_eps

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="resnet50")
ap.add_argument("--eps", default="32/255")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
eps = parse_eps(args.eps)

model = load(args.model, "imagenette", args.device)   # sliced to 10 classes, as in the grid
attack = torchattacks.FGSM(model, eps=eps)            # the same attack as in the grid
table = torch.tensor(IMAGENETTE_TO_IMAGENET, device=args.device)

correct = {"clean10": 0, "clean1000": 0, "adv10": 0, "adv1000": 0}
total = 0

for x, y in get_loader("imagenette", n=args.n, batch=args.batch):
    x, y = x.to(args.device), y.to(args.device)
    y_1000 = table[y]
    x_adv = attack(x, y)

    with torch.no_grad():
        correct["clean10"] += (model(x).argmax(1) == y).sum().item()
        correct["adv10"] += (model(x_adv).argmax(1) == y).sum().item()

        # the full head, all 1000 classes, with the label mapped through the same table
        correct["clean1000"] += (model.backbone((x - model.mean) / model.std).argmax(1) == y_1000).sum().item()
        correct["adv1000"] += (model.backbone((x_adv - model.mean) / model.std).argmax(1) == y_1000).sum().item()

    total += len(y)

print(f"{args.model}, fgsm eps={args.eps}, {total} images\n")
print(f"  clean, 10 classes:      {correct['clean10'] / total:.3f}")
print(f"  clean, 1000 classes:    {correct['clean1000'] / total:.3f}")
print(f"  attacked, 10 classes:   {correct['adv10'] / total:.3f}")
print(f"  attacked, 1000 classes: {correct['adv1000'] / total:.3f}")