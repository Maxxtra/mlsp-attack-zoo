#!/usr/bin/env bash
# Runs every attack at every budget, for one model, in one norm.
#
#   bash scripts/grid.sh resnet50            # linf, n=200
#   bash scripts/grid.sh resnet50 200 l2     # l2
#
# The budgets differ per norm: Linf limits how far one pixel moves, L2 limits
# the length of the whole perturbation vector over all 150528 pixels.
#
# DeepFool searches for the smallest perturbation that changes the class, so it
# ignores --eps and returns the same number at every budget. It runs once.

MODEL=${1:-resnet50}
N=${2:-200}
NORM=${3:-linf}

if [ "$NORM" = "l2" ]; then
  EPS_LIST="0.25 0.5 1.0 2.0 4.0 8.0"
  ATTACKS="pgd"          # only pgd has an L2 version in run_attack.py
else
  EPS_LIST="1/255 2/255 4/255 8/255 16/255 32/255"
  ATTACKS="fgsm bim pgd"
fi

for attack in $ATTACKS; do
  for eps in $EPS_LIST; do
    echo "--- $MODEL $attack $NORM eps=$eps"
    python src/run_attack.py --dataset imagenette --model "$MODEL" \
                             --attack "$attack" --eps "$eps" --norm "$NORM" --n "$N"
  done
done

if [ "$NORM" = "linf" ]; then
  echo "--- $MODEL deepfool (budget not used)"
  python src/run_attack.py --dataset imagenette --model "$MODEL" \
                           --attack deepfool --eps 8/255 --n "$N"
fi