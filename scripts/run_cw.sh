#!/bin/bash
# carlini-wagner pe 4 modele, imagenette, norma l2, 100 de pasi, n=200
# cw cauta cea mai mica perturbatie care schimba clasa, deci ignora eps:
# valoarea 8/255 din csv e doar eticheta, nu un buget
# are nevoie de imagenette descarcat: python3 src/data.py --download imagenette
cd "$(dirname "$0")/.."

for model in resnet50 efficientnet vit convnext; do
  echo "--- $model cw l2"
  python3 src/run_attack.py --dataset imagenette --model $model --attack cw --norm l2 --eps 8/255 --n 200 --steps 100
done
