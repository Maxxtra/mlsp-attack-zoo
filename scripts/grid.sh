#!/usr/bin/env bash
# Grila principala: 4 atacuri x 4 modele x 6 eps, pe un dataset. Ruleaza din radacina repo-ului.
DS=${1:-imagenette}
for model in resnet50 efficientnet vit convnext; do
  for attack in fgsm bim pgd deepfool; do
    for eps in 1/255 2/255 4/255 8/255 16/255 32/255; do
      python src/run_attack.py --dataset $DS --model $model --attack $attack --eps $eps
    done
  done
done
