import argparse
import torch
import torchattacks
from data import get_loader
from models import load
from my_attacks import fgsm, pgd
from run_attack import parse_eps

TOL = 1e-6

def compare(name, mine, theirs, model, x, y, eps):
    max_diff = (mine - theirs).abs().max().item()
    budget = (mine - x).abs().max().item()
    with torch.no_grad():
        acc_mine = (model(mine).argmax(1) == y).float().mean().item()
        acc_theirs = (model(theirs).argmax(1) == y).float().mean().item()
    ok = max_diff < TOL and budget <= eps + TOL
    print(f"  {name:>4}  diff {max_diff:.2e}   robust mine {acc_mine:.3f} / lib {acc_theirs:.3f}   "
          f"max perturbation {budget:.5f}   {'OK' if ok else 'DIFFERENT'}")
    return ok
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imagenette")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--eps", nargs="+", default=["1/255", "2/255", "4/255"])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
 
    model = load(a.model, a.dataset, a.device)
    x, y = next(iter(get_loader(a.dataset, n=a.n, batch=a.n)))
    x, y = x.to(a.device), y.to(a.device)
 
    with torch.no_grad():
        clean = (model(x).argmax(1) == y).float().mean().item()
    print(f"{a.dataset} {a.model}, {len(y)} images, clean acc {clean:.3f}\n")
 
    all_ok = True
    for eps_str in a.eps:
        eps = parse_eps(eps_str)
        alpha = eps / 4
        print(f"eps = {eps_str}")
        all_ok &= compare("FGSM", fgsm(model, x, y, eps),
                          torchattacks.FGSM(model, eps=eps)(x, y),
                          model, x, y, eps)
        all_ok &= compare("PGD", pgd(model, x, y, eps, alpha, a.steps, random_start=False),
                          torchattacks.PGD(model, eps=eps, alpha=alpha, steps=a.steps,
                                           random_start=False)(x, y),
                          model, x, y, eps)
 
    print("\nall match" if all_ok else "\nmismatch, look for a bug")
 
 
if __name__ == "__main__":
    main()
 