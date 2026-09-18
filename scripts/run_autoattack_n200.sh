#!/bin/bash
# autoattack standard pe 4 modele, imagenette, linf, eps 1/2/4 pe 255, n=200 ca grila andreei
# are nevoie de: python3 src/data.py --download imagenette
#                pip install git+https://github.com/fra31/auto-attack
cd "$(dirname "$0")/.."

for model in resnet50 efficientnet vit convnext; do
  for eps in 1/255 2/255 4/255; do
    echo "--- $model autoattack linf eps=$eps"
    python3 src/run_autoattack.py --dataset imagenette --model $model --eps $eps --n 200
  done
done
