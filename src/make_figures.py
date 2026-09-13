import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
 
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results", "results.csv")
FIGURES = os.path.join(ROOT, "figures")
 
MODEL_ORDER = ["resnet50", "efficientnet", "vit", "convnext"]
ATTACK_ORDER = ["fgsm", "bim", "pgd", "deepfool", "cw", "autoattack"]
 
 
def eps_to_float(s):
    s = str(s).strip()
    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den)
    return float(s)
 
 
def load_results(path, dataset, norm):
    df = pd.read_csv(path)
    df = df[(df["dataset"] == dataset) & (df["norm"] == norm)].copy()
    if df.empty:
        raise SystemExit(f"no rows for dataset={dataset} norm={norm} in {path}")
    df["eps_value"] = df["eps"].map(eps_to_float)
    # keep the most recent run of each combination
    df = df.drop_duplicates(subset=["model", "attack", "eps"], keep="last")
    return df
 
 
def order_by(values, preferred):
    known = [v for v in preferred if v in values]
    return known + sorted(v for v in values if v not in preferred)
 
 
def figure_robust_vs_eps(df, dataset, norm):
    attacks = order_by(df["attack"].unique(), ATTACK_ORDER)
    models = order_by(df["model"].unique(), MODEL_ORDER)
    fig, axes = plt.subplots(1, len(attacks), figsize=(4 * len(attacks), 3.6),
                             sharey=True, squeeze=False)
    for ax, attack in zip(axes[0], attacks):
        sub = df[df["attack"] == attack]
        for model in models:
            curve = sub[sub["model"] == model].sort_values("eps_value")
            if curve.empty:
                continue
            ax.plot(curve["eps_value"] * 255, curve["robust_acc"], marker="o", label=model)
        ax.set_title(attack)
        ax.set_xlabel("eps (steps out of 255)")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("robust accuracy")
    axes[0][0].set_ylim(0, 1)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"Robust accuracy vs perturbation budget, {dataset}, {norm}")
    fig.tight_layout()
    out = os.path.join(FIGURES, f"robust_acc_vs_eps_{dataset}_{norm}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
 
 
def figure_clean_vs_worst(df, dataset, norm):
    models = order_by(df["model"].unique(), MODEL_ORDER)
    clean = df.groupby("model")["clean_acc"].max()
    worst = df.groupby("model")["robust_acc"].min()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    positions = range(len(models))
    ax.bar([p - 0.2 for p in positions], [clean.get(m, 0) for m in models],
           width=0.4, label="clean")
    ax.bar([p + 0.2 for p in positions], [worst.get(m, 0) for m in models],
           width=0.4, label="worst case over all attacks and budgets")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Clean vs worst-case accuracy, {dataset}, {norm}")
    fig.tight_layout()
    out = os.path.join(FIGURES, f"clean_vs_robust_{dataset}_{norm}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--dataset", default="imagenette")
    ap.add_argument("--norm", default="linf", choices=["linf", "l2"])
    a = ap.parse_args()
 
    os.makedirs(FIGURES, exist_ok=True)
    df = load_results(a.results, a.dataset, a.norm)
    for path in (figure_robust_vs_eps(df, a.dataset, a.norm),
                 figure_clean_vs_worst(df, a.dataset, a.norm)):
        print("written:", os.path.relpath(path, ROOT))
 
 
if __name__ == "__main__":
    main()
 