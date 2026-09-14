"""Draws the main figure from results/results.csv. Never edit a figure by hand.

    python src/make_figures.py
    python src/make_figures.py --norm l2

One panel per attack, one curve per model: accuracy under attack against the
perturbation budget.
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")   # draw into a file, not on screen
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="imagenette")
ap.add_argument("--norm", default="linf")
args = ap.parse_args()

table = pd.read_csv(os.path.join(ROOT, "results", "results.csv"))
table = table[(table["dataset"] == args.dataset) & (table["norm"] == args.norm)]
if len(table) == 0:
    raise SystemExit(f"no rows for {args.dataset} {args.norm} in results.csv")

# "4/255" is text; turn it into a number so it can go on an axis
table["eps_number"] = table["eps"].map(lambda s: eval(s))

# if a run was repeated, keep the last one
table = table.drop_duplicates(subset=["model", "attack", "eps"], keep="last")

attacks = sorted(table["attack"].unique())
models = sorted(table["model"].unique())

figure, panels = plt.subplots(1, len(attacks), figsize=(4 * len(attacks), 4), sharey=True)
if len(attacks) == 1:
    panels = [panels]

for panel, attack in zip(panels, attacks):
    for model in models:
        curve = table[(table["attack"] == attack) & (table["model"] == model)]
        curve = curve.sort_values("eps_number")
        if len(curve) > 0:
            panel.plot(curve["eps_number"] * 255, curve["robust_acc"], marker="o", label=model)
    panel.set_title(attack)
    panel.set_xlabel("eps (steps out of 255)")
    panel.set_xscale("log", base=2)
    panel.grid(alpha=0.3)

panels[0].set_ylabel("accuracy under attack")
panels[0].set_ylim(0, 1)
panels[-1].legend(fontsize=8)
figure.suptitle(f"{args.dataset}, {args.norm}")
figure.tight_layout()

os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
output = os.path.join(ROOT, "figures", f"robust_acc_{args.dataset}_{args.norm}.png")
figure.savefig(output, dpi=150)
print("written:", output)