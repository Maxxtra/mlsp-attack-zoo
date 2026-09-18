"""Does evaluating on 10 classes hide the success of a one-step attack?

The grid slices the 1000 ImageNet logits down to the 10 Imagenette classes, so a
prediction that moved to some other ImageNet class still counts as correct. This
runs the same attack and scores the same images both ways.

    python src/test_1000_clase.py
    python src/test_1000_clase.py --model convnext --eps 16/255
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
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
eps = parse_eps(args.eps)

model = load(args.model, "imagenette", args.device)   # sliced to 10 classes, as in the grid

xs, ys = [], []
for x, y in get_loader("imagenette", n=args.n):
    xs.append(x); ys.append(y)
x, y = torch.cat(xs).to(args.device), torch.cat(ys).to(args.device)

x_adv = torchattacks.FGSM(model, eps=eps)(x, y)       # the same attack as in the grid

with torch.no_grad():
    acc10 = (model(x_adv).argmax(1) == y).float().mean().item()

    # the full head, all 1000 classes, with the label mapped through the same table
    logits = model.backbone((x_adv - model.mean) / model.std)
    y_1000 = torch.tensor(IMAGENETTE_TO_IMAGENET, device=args.device)[y]
    acc1000 = (logits.argmax(1) == y_1000).float().mean().item()

    clean10 = (model(x).argmax(1) == y).float().mean().item()
    clean1000 = (model.backbone((x - model.mean) / model.std).argmax(1) == y_1000).float().mean().item()

print(f"{args.model}, fgsm eps={args.eps}, {len(y)} images\n")
print(f"  clean, 10 classes:     {clean10:.3f}")
print(f"  clean, 1000 classes:   {clean1000:.3f}")
print(f"  attacked, 10 classes:  {acc10:.3f}")
print(f"  attacked, 1000 classes:{acc1000:.3f}")