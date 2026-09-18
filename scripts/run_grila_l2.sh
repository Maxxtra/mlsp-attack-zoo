#!/bin/bash
# grila l2 completa: fgsm, bim si pgd pe 4 modele x 6 bugete, imagenette, n=200
# fgsm si bim in l2 sunt scrise de mana in src/my_attacks.py, torchattacks le are doar in linf
# bugetele l2 nu sunt aceleasi ca in linf: acolo eps limiteaza un pixel, aici lungimea
# intregii perturbatii peste toti cei 150528 de pixeli (4/255 in linf ~ 6 in l2)
# are nevoie de imagenette descarcat: python3 src/data.py --download imagenette
cd "$(dirname "$0")/.."

for model in resnet50 efficientnet convnext vit; do
  for attack in fgsm bim pgd; do
    for eps in 0.25 0.5 1.0 2.0 4.0 8.0; do
      echo "--- $model $attack l2 eps=$eps"
      python3 src/run_attack.py --dataset imagenette --model $model --attack $attack --eps $eps --norm l2 --n 200
    done
  done
done