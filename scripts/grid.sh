#!/usr/bin/env bash
# Runs every attack at every budget, for one model.
#
#   bash scripts/grid.sh resnet50
#   bash scripts/grid.sh vit 500
#
# DeepFool searches for the smallest perturbation that changes the class, so it
# ignores --eps and returns the same number at every budget. It runs once.

MODEL=${1:-resnet50}
N=${2:-200}

for attack in fgsm bim pgd; do
  for eps in 1/255 2/255 4/255 8/255 16/255 32/255; do
    echo "--- $MODEL $attack eps=$eps"
    python src/run_attack.py --dataset imagenette --model "$MODEL" \
                             --attack "$attack" --eps "$eps" --n "$N"
  done
done

echo "--- $MODEL deepfool (budget not used)"
python src/run_attack.py --dataset imagenette --model "$MODEL" \
                         --attack deepfool --eps 8/255 --n "$N"