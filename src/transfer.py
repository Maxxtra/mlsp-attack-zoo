"""Matricea de transfer: exemple adversariale generate pe modelul sursa, evaluate pe toate modelele.

    python src/transfer.py --dataset imagenette --eps 4/255
Scrie results/transfer.csv cu coloanele: source, target, robust_acc
"""
import argparse, os, csv, torch, torchattacks
from data import get_loader
from models import load, NAMES
from run_attack import parse_eps

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "transfer.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imagenette"); ap.add_argument("--eps", default="4/255")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); eps = parse_eps(a.eps)
    models = {k: load(k, a.dataset, a.device) for k in NAMES}
    xs, ys = [], []
    for x, y in get_loader(a.dataset, n=a.n): xs.append(x); ys.append(y)
    x, y = torch.cat(xs).to(a.device), torch.cat(ys).to(a.device)

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source","target","robust_acc"])
        for src, ms in models.items():
            x_adv = torchattacks.PGD(ms, eps=eps, alpha=eps/4, steps=10, random_start=True)(x, y)
            for tgt, mt in models.items():
                with torch.no_grad(): acc = (mt(x_adv).argmax(1) == y).float().mean().item()
                print(f"{src:>12} -> {tgt:<12} robust={acc:.3f}"); w.writerow([src, tgt, f"{acc:.4f}"])

if __name__ == "__main__":
    main()
