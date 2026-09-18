#!/bin/bash
# varianta mica, ca sa se vada ca merge cap-coada: un model, un buget, n=20
# are nevoie de imagenette descarcat: python3 src/data.py --download imagenette
cd "$(dirname "$0")/.."

echo "--- efficientnet autoattack linf eps=4/255 n=20"
python3 src/run_autoattack.py --dataset imagenette --model efficientnet --eps 4/255 --n 20
