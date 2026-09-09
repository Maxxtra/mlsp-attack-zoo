"""Ruleaza un atac la un eps si scrie o linie in results/results.csv.

    python src/run_attack.py --dataset imagenette --model resnet50 --attack pgd --eps 4/255
Atacuri: fgsm, bim, pgd, deepfool, cw   (norma: linf sau l2)
"""
import argparse, os, csv, time, torch, torchattacks
from data import get_loader
from models import load

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "results.csv")

def parse_eps(s):  # accepta "4/255" sau "0.0157"
    return eval(s) if "/" in s else float(s)

def make_attack(name, model, eps, norm, steps):
    if name == "fgsm": return torchattacks.FGSM(model, eps=eps)
    if name == "bim":  return torchattacks.BIM(model, eps=eps, alpha=eps/4, steps=steps)
    if name == "pgd":
        return torchattacks.PGD(model, eps=eps, alpha=eps/4, steps=steps, random_start=True) if norm=="linf" \
          else torchattacks.PGDL2(model, eps=eps, alpha=eps/4, steps=steps, random_start=True)
    if name == "deepfool": return torchattacks.DeepFool(model, steps=50)
    if name == "cw": return torchattacks.CW(model, c=1, kappa=0, steps=steps, lr=0.01)
    raise ValueError(name)

@torch.no_grad()
def accuracy(model, x, y): return (model(x).argmax(1) == y).float().sum().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imagenette"); ap.add_argument("--model", default="resnet50")
    ap.add_argument("--attack", default="pgd"); ap.add_argument("--eps", default="4/255")
    ap.add_argument("--norm", default="linf", choices=["linf","l2"]); ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); eps = parse_eps(a.eps)

    model = load(a.model, a.dataset, a.device)
    atk = make_attack(a.attack, model, eps, a.norm, a.steps)
    loader = get_loader(a.dataset, n=a.n)
    clean = rob = tot = 0; t0 = time.time()
    for x, y in loader:
        x, y = x.to(a.device), y.to(a.device)
        clean += accuracy(model, x, y)
        x_adv = atk(x, y)
        rob += accuracy(model, x_adv, y); tot += len(y)
    clean_acc, rob_acc, sec = clean/tot, rob/tot, time.time()-t0
    print(f"{a.dataset} {a.model} {a.attack} {a.norm} eps={a.eps}: clean={clean_acc:.3f} robust={rob_acc:.3f} ({sec:.0f}s)")

    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["dataset","model","attack","norm","eps","n","clean_acc","robust_acc","seconds"])
        w.writerow([a.dataset, a.model, a.attack, a.norm, a.eps, tot, f"{clean_acc:.4f}", f"{rob_acc:.4f}", f"{sec:.0f}"])

if __name__ == "__main__":
    main()
