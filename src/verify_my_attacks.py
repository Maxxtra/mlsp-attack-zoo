"""Checks that my FGSM and PGD give the same images as torchattacks.

    python src/verify_my_attacks.py
    python src/verify_my_attacks.py --model vit

Runs on the CPU by default: on the GPU the same attack can differ slightly
between runs, which makes an exact comparison meaningless.
"""
import argparse
import torch
import torchattacks
from data import get_loader
from models import load
from my_attacks import fgsm, pgd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="resnet50")
ap.add_argument("--dataset", default="imagenette")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

model = load(args.model, args.dataset, args.device)
images, labels = next(iter(get_loader(args.dataset, n=args.n, batch=args.n)))
images, labels = images.to(args.device), labels.to(args.device)
print(f"{args.model}, {args.n} images\n")

everything_matches = True

for eps in [1 / 255, 2 / 255, 4 / 255]:
    alpha = eps / 4

    mine = {
        "FGSM": fgsm(model, images, labels, eps),
        "PGD": pgd(model, images, labels, eps, alpha, args.steps, random_start=False),
    }
    theirs = {
        "FGSM": torchattacks.FGSM(model, eps=eps)(images, labels),
        "PGD": torchattacks.PGD(model, eps=eps, alpha=alpha, steps=args.steps,
                                random_start=False)(images, labels),
    }

    print(f"eps = {round(eps * 255)}/255")
    for name in mine:
        difference = (mine[name] - theirs[name]).abs().max().item()
        perturbation = (mine[name] - images).abs().max().item()
        matches = difference < 1e-6 and perturbation <= eps + 1e-6
        everything_matches = everything_matches and matches
        print(f"  {name:<5} difference {difference:.1e}   "
              f"largest pixel change {perturbation:.5f}   "
              f"{'same' if matches else 'DIFFERENT'}")

print("\nall match" if everything_matches else "\nmismatch, look for a bug")