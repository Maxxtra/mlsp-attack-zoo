#!/usr/bin/env bash
# Grila principala: 4 atacuri x 4 modele x 6 eps, pe un dataset. Ruleaza din radacina repo-ului.
set -u
 
DS=${1:-imagenette}
MODELS=${2:-"resnet50 efficientnet vit convnext"}
N=${N:-500}
ATTACKS=${ATTACKS:-"fgsm bim pgd deepfool"}
EPS_LIST=${EPS_LIST:-"1/255 2/255 4/255 8/255 16/255 32/255"}
NORM=${NORM:-linf}
 
for model in $MODELS; do
  for attack in $ATTACKS; do
    for eps in $EPS_LIST; do
      echo "--- $DS $model $attack $NORM eps=$eps n=$N"
      python src/run_attack.py --dataset "$DS" --model "$model" --attack "$attack" \
                               --eps "$eps" --norm "$NORM" --n "$N"
    done
  done
done
 