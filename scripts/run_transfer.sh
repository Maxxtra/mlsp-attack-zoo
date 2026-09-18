#!/bin/bash
# matricea de transfer 4x4: pgd generat pe modelul sursa, evaluat pe toate 4 modelele
# imagenette, eps 4/255, n=200. scrie results/transfer.csv
# are nevoie de imagenette descarcat: python3 src/data.py --download imagenette
cd "$(dirname "$0")/.."

python3 src/transfer.py --dataset imagenette --eps 4/255 --n 200
