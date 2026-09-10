"""AutoAttack, referinta standard. Necesita: pip install git+https://github.com/fra31/auto-attack

    python src/run_autoattack.py --dataset imagenette --model resnet50 --eps 4/255
"""
import argparse, os, csv, time, torch
from autoattack import AutoAttack
from data import get_loader
from models import load
from run_attack import RESULTS, parse_eps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imagenette"); ap.add_argument("--model", default="resnet50")
    ap.add_argument("--eps", default="4/255"); ap.add_argument("--norm", default="linf", choices=["linf","l2"])
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fast", action="store_true", help="doar APGD-CE, pentru o prima cifra pe CPU")
    a = ap.parse_args(); eps = parse_eps(a.eps)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    model = load(a.model, a.dataset, a.device)
    xs, ys = [], []
    for x, y in get_loader(a.dataset, n=a.n): xs.append(x); ys.append(y)
    x, y = torch.cat(xs).to(a.device), torch.cat(ys).to(a.device)
    with torch.no_grad(): clean = (model(x).argmax(1) == y).float().mean().item()

    t0 = time.time()
    if a.fast:
        adv = AutoAttack(model, norm={"linf":"Linf","l2":"L2"}[a.norm], eps=eps, version="custom", attacks_to_run=["apgd-ce"], device=a.device)
        adv.apgd.n_restarts = 1
    else:
        adv = AutoAttack(model, norm={"linf":"Linf","l2":"L2"}[a.norm], eps=eps, version="standard", device=a.device)
    x_adv = adv.run_standard_eval(x, y, bs=64 if a.device == "cuda" else 16)
    with torch.no_grad(): rob = (model(x_adv).argmax(1) == y).float().mean().item()
    sec = time.time()-t0
    print(f"{a.dataset} {a.model} autoattack {a.norm} eps={a.eps}: clean={clean:.3f} robust={rob:.3f} ({sec:.0f}s)")
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["dataset","model","attack","norm","eps","n","clean_acc","robust_acc","seconds"])
        w.writerow([a.dataset, a.model, "autoattack-fast" if a.fast else "autoattack", a.norm, a.eps, len(y), f"{clean:.4f}", f"{rob:.4f}", f"{sec:.0f}"])

if __name__ == "__main__":
    main()
