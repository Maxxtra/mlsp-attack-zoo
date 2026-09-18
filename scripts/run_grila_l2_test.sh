#!/bin/bash
# varianta mica a grilei l2, ca sa se vada ca merge cap-coada: un model, 2 bugete, n=20
# are nevoie de imagenette descarcat: python3 src/data.py --download imagenette
cd "$(dirname "$0")/.."

for attack in fgsm bim pgd; do
  for eps in 0.5 4.0; do
    echo "--- resnet50 $attack l2 eps=$eps"
    python3 src/run_attack.py --dataset imagenette --model resnet50 --attack $attack --eps $eps --norm l2 --n 20
  done
done